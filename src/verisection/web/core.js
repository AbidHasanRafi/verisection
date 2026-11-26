/* ===== core: state, helpers, colormaps, payload, layers panel, inspector ===== */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const k in (attrs || {})) {
    const v = attrs[k];
    if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== false && v !== null && v !== undefined) e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat(Infinity)) { if (c == null || c === false) continue; e.append(c.nodeType ? c : document.createTextNode(String(c))); }
  return e;
}
const fmt = (x, n = 3) => (x === null || x === undefined || (typeof x === 'number' && !isFinite(x))) ? '—'
  : (typeof x === 'number' ? (Math.abs(x) >= 1000 ? x.toFixed(0) : String(+x.toFixed(n))) : String(x));
const pct = (x, n = 1) => (x === null || x === undefined) ? '—' : (100 * x).toFixed(n) + '%';
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
function hex2rgb(s) { const n = parseInt(s.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function lut(stops) {
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255; let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k], [t1, c1] = stops[k + 1];
    const u = clamp((t - t0) / (t1 - t0), 0, 1);
    for (let c = 0; c < 3; c++) out[i * 3 + c] = c0[c] + (c1[c] - c0[c]) * u;
  }
  return out;
}
const CM = {
  he: lut([[0, [250, 243, 247]], [0.35, [236, 170, 205]], [0.65, [142, 74, 184]], [1, [22, 28, 96]]]),
  magma: lut([[0, [0, 0, 4]], [.25, [80, 18, 123]], [.5, [182, 54, 121]], [.75, [251, 135, 97]], [1, [252, 253, 191]]]),
  viridis: lut([[0, [68, 1, 84]], [.25, [59, 82, 139]], [.5, [33, 145, 140]], [.75, [94, 201, 98]], [1, [253, 231, 37]]]),
  inferno: lut([[0, [0, 0, 4]], [.25, [87, 16, 110]], [.5, [188, 55, 84]], [.75, [249, 142, 9]], [1, [252, 255, 164]]]),
  turbo: lut([[0, [48, 18, 59]], [.25, [70, 134, 251]], [.5, [164, 252, 60]], [.75, [250, 130, 22]], [1, [122, 4, 3]]]),
  gray: lut([[0, [0, 0, 0]], [1, [255, 255, 255]]]),
};
const PROV_RGB = { 0: [60, 60, 60], 1: [43, 123, 186], 2: [240, 160, 48], 3: [176, 74, 176] };
const PROV_NAME = { 0: 'unsampled', 1: 'observed', 2: 'interpolated', 3: 'synthetic' };

const S = {
  P: null, dims: [1, 1, 1], cur: [0, 0, 0], vol: null, palette: [], classOn: [], rw: [],
  layout: 'quad', active: 'axial', theme: 'dark', tool: 'cursor',
  L: {
    base: { on: true, op: 1 }, labels: { on: true, op: 0.5 }, unc: { on: false, op: 0.75, cmap: 'magma' },
    prov: { on: true, op: 0.75 }, truth: { on: false, op: 1 }, err: { on: false, op: 0.5 }, varReg: { on: false, op: 0.5 },
  },
  insp: {},
  view: { cross: true, scale: true, smooth: false, ruler: true, labels: true },
  R: { mode: 0, cls: 1, steps: 192, density: 1.6, light: 0.8, cut: false, clip: [0, 0, 0, 1, 1, 1], auto: false, slabs: true, frames: true, iso: 0.5, thr: 0.35, cmap: 'magma' },
  zoom: { axial: 1, coronal: 1, sagittal: 1 }, pan: { axial: [0, 0], coronal: [0, 0], sagittal: [0, 0] },
  measures: [], roi: null, tab: 'Summary', canv: {}, off: {}, tables: {},
};
const IDX = (z, y, x) => (z * S.dims[1] + y) * S.dims[2] + x;
function b64ToU8(s) { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
function toast(msg, ms = 2600) {
  let t = $('#toast');
  if (!t) { t = h('div', { id: 'toast', role: 'status' }); document.body.append(t); }
  t.textContent = msg; t.style.display = 'block'; clearTimeout(t._t); t._t = setTimeout(() => { t.style.display = 'none'; }, ms);
}

function loadPayload(p) {
  S.P = p; S.dims = p.dims;
  const v = p.volumes;
  S.vol = {
    labels: b64ToU8(v.labels), prov: b64ToU8(v.prov), unc: b64ToU8(v.unc), intensity: b64ToU8(v.intensity), varReg: b64ToU8(v.varReg),
    probs: v.probs.map(b64ToU8), truth: v.truth ? b64ToU8(v.truth) : null, err: v.err ? b64ToU8(v.err) : null,
  };
  const [Z, Y, X] = p.dims;
  S.cur = [Math.floor(Z / 2), Math.floor(Y / 2), Math.floor(X / 2)];
  S.palette = p.classColors.map(hex2rgb);
  S.classOn = p.classNames.map(() => true);
  S.rw = p.classNames.map(n => /lumen/i.test(n) ? 0.0 : /stroma/i.test(n) ? 0.03 : /immune/i.test(n) ? 3.0 : /epith/i.test(n) ? 1.9 : 1.0);
  S.R.cls = Math.max(0, p.classNames.findIndex(n => /epith/i.test(n)));
  S.measures = []; S.roi = null;
  S.L.truth.on = false; S.L.err.on = false;
  S.zoom = { axial: 1, coronal: 1, sagittal: 1 }; S.pan = { axial: [0, 0], coronal: [0, 0], sagittal: [0, 0] };
  document.title = 'VeriSection Workstation — ' + p.title;
  buildLayersPanel(); buildTabs(); buildInspector();
  GL3D.upload();
  refreshAll();
  setStatus('Loaded ' + p.title + '  (' + Z + '×' + Y + '×' + X + ' voxels)');
}
function refreshAll() { renderAllSlices(); GL3D.dirty = true; updateInspector(); updateSteppers(); }

/* ---------- shared control patterns (no sliders: presets, steppers and selects only) ---------- */
const OPACITY_PRESETS = [0.25, 0.5, 0.75, 1];
function opacitySeg(key) {
  const L = S.L[key];
  return h('div', { class: 'seg fill', role: 'group', 'aria-label': 'Opacity' }, OPACITY_PRESETS.map(o =>
    h('button', { class: Math.abs(L.op - o) < 0.01 ? 'on' : '', title: 'Opacity ' + Math.round(o * 100) + '%', onclick: e => {
      L.op = o; $$('button', e.currentTarget.parentElement).forEach(b => b.classList.toggle('on', b === e.currentTarget)); refreshAll();
    } }, Math.round(o * 100) + '%')));
}
function emptyState(text, label, fn) { return h('div', { class: 'empty' }, h('span', null, text), label ? h('button', { class: 'btn sm', onclick: fn }, label) : null); }

/* ---------- layers panel ---------- */
function buildLayersPanel() {
  const root = $('#layers'); root.innerHTML = '';
  if (!S.P) { root.append(h('div', { class: 'note' }, 'No volume loaded. Use File ▸ New simulation… (server mode) or open an exported volume.')); return; }
  // one layer-card pattern: [checkbox + name] / opacity presets / optional detail rows / optional hint
  const lcard = (title, key, { body = null, hint = null, noOp = false } = {}) => {
    const L = S.L[key];
    return h('section', { class: 'lcard' },
      h('label', { class: 'lhead' }, h('input', { type: 'checkbox', checked: L.on, onchange: e => { L.on = e.target.checked; refreshAll(); syncToolbar(); } }), h('span', { class: 'lname', title }, title)),
      h('div', { class: 'lbody' }, noOp ? null : h('div', { class: 'lrow2' }, h('span', { class: 'cap' }, 'Opacity'), opacitySeg(key)), body),
      hint ? h('div', { class: 'hint' }, hint) : null);
  };
  const classRows = h('div', { class: 'lrow2' }, h('span', { class: 'cap' }, 'Classes'), ...S.P.classNames.map((n, c) => h('label', { class: 'crow' },
    h('input', { type: 'checkbox', checked: S.classOn[c], onchange: e => { S.classOn[c] = e.target.checked; refreshAll(); } }),
    h('span', { class: 'sw', style: 'background:' + S.P.classColors[c] }), h('span', null, n),
    h('span', { class: 'tlabel', id: 'cfrac' + c }, ''))));
  const cmapRow = h('div', { class: 'lrow2' }, h('span', { class: 'cap' }, 'Colormap'),
    h('select', { onchange: e => { S.L.unc.cmap = e.target.value; refreshAll(); } },
      ['magma', 'inferno', 'viridis', 'turbo', 'gray'].map(c => h('option', { value: c, selected: c === S.L.unc.cmap }, c))));
  root.append(
    lcard('Image (registered sections)', 'base', { hint: 'Observed planes show the registered section; gap planes show a model-derived fill.' }),
    lcard('Segmentation (MAP label)', 'labels', { body: classRows }),
    lcard('Uncertainty (entropy)', 'unc', { body: cmapRow }),
    lcard('Provenance', 'prov', { hint: 'Hatched orange = interpolated (model inference). Dark = unsampled.' }),
    S.P.hasTruth ? lcard('Truth contour (simulation)', 'truth', { noOp: true }) : null,
    S.P.hasTruth ? lcard('Error map (simulation)', 'err') : null,
    lcard('Registration variance', 'varReg'));
  const fr = S.P.analysis && S.P.analysis.summary && S.P.analysis.summary.predicted_class_fraction;
  if (fr) fr.forEach((f, c) => { const e = $('#cfrac' + c); if (e) e.textContent = pct(f, 0); });
}

/* ---------- inspector ---------- */
function buildInspector() {
  const r = $('#inspector'); r.innerHTML = '';
  if (!S.P) { r.append(emptyState('Click in any slice view to probe a voxel once a volume is loaded.')); return; }
  // one inspector-card pattern: collapsible, state remembered per section
  const card = (id, title, kids) => h('details', { class: 'icard', open: S.insp[id] !== false, ontoggle: e => { S.insp[id] = e.target.open; } },
    h('summary', null, title), h('div', { class: 'ibody' }, kids));
  r.append(card('pos', 'Position', h('div', { id: 'insp-pos' })), card('vox', 'Voxel', h('div', { id: 'insp-vox', class: 'kv' })),
    card('probs', 'Class posterior', h('div', { id: 'insp-probs' })), card('roi', 'Region of interest', h('div', { id: 'insp-roi' })),
    card('meas', 'Measurements', h('div', { id: 'insp-meas' })));
}
function updateInspector() {
  if (!S.P || !$('#insp-vox')) return;
  const [z, y, x] = S.cur, V = S.vol, k = IDX(z, y, x), P = S.P, px = P.pixelUm;
  const pv = V.prov[k], lb = V.labels[k], C = P.classNames.length;
  $('#insp-pos').innerHTML = '';
  $('#insp-pos').append(h('div', { class: 'kv' }, h('span', null, 'index (z,y,x)'), h('span', null, `${z}, ${y}, ${x}`),
    h('span', null, 'position µm'), h('span', null, `z ${fmt(P.zUm[z], 1)}  y ${fmt(y * px, 1)}  x ${fmt(x * px, 1)}`)));
  const kv = $('#insp-vox'); kv.innerHTML = '';
  const rows = [
    ['provenance', h('span', { class: 'badge', style: 'background:rgb(' + PROV_RGB[pv].join(',') + ')' }, PROV_NAME[pv])],
    ['MAP class', lb > 0 ? P.classNames[lb - 1] : '—'],
    ['entropy', fmt(V.unc[k] / 255 * P.uncScale, 3) + ' bits'],
    ['confidence', fmt(1 - V.unc[k] / 255, 2)],
    ['image (OD-like)', fmt(V.intensity[k] / 255, 2)],
    ['registration var', fmt(V.varReg[k] / 255, 2) + ' (rel.)'],
  ];
  const gap = P.analysis && P.analysis.provenance && P.analysis.provenance.planes && P.analysis.provenance.planes[z];
  if (gap) rows.push(['distance to nearest slab', fmt(gap.gap_um, 1) + ' µm']);
  if (V.truth) { const t = V.truth[k]; rows.push(['true class', t > 0 ? P.classNames[t - 1] : '—'], ['prediction', t === lb ? h('span', { class: 'chip good' }, 'correct') : h('span', { class: 'chip bad' }, 'wrong')]); }
  rows.forEach(([a, b]) => kv.append(h('span', null, a), h('span', null, b)));
  const pr = $('#insp-probs'); pr.innerHTML = '';
  for (let c = 0; c < C; c++) {
    const p = V.probs[c][k] / 255;
    pr.append(h('div', { class: 'irow' }, h('span', { class: 'sw', style: 'background:' + P.classColors[c] }), h('span', null, P.classNames[c]),
      h('div', { class: 'bar' }, h('i', { style: `width:${(p * 100).toFixed(0)}%;background:${P.classColors[c]}` })), h('span', { style: 'text-align:right', class: 'tlabel' }, pct(p, 0))));
  }
  const rm = $('#insp-roi');
  if (S.roi) rm.replaceChildren(h('div', { class: 'kv' }, ...roiRows()), h('button', { class: 'btn sm', style: 'margin-top:8px', onclick: () => { S.roi = null; refreshAll(); } }, 'Clear ROI'));
  else rm.replaceChildren(emptyState('Drag a rectangle in a slice view to get class fractions, provenance and entropy.', 'Draw ROI (R)', () => setTool('roi')));
  const mm = $('#insp-meas');
  if (S.measures.length) mm.replaceChildren(h('div', { class: 'kv' }, ...S.measures.flatMap((m, i) => [h('span', null, '#' + (i + 1)), h('span', null, `${fmt(m.d, 1)} µm  (Δz ${fmt(m.dz, 1)}, Δy ${fmt(m.dy, 1)}, Δx ${fmt(m.dx, 1)})`)])),
    h('button', { class: 'btn sm', style: 'margin-top:8px', onclick: () => { S.measures = []; S.pendingMeasure = []; refreshAll(); } }, 'Clear measurements'));
  else mm.replaceChildren(emptyState('Pick two points in a slice view to measure a 3D distance in µm.', 'Measure distance (D)', () => setTool('measure')));
}
function roiRows() {
  const r = S.roi, P = S.P; if (!r) return [];
  const out = [h('span', null, 'plane'), h('span', null, r.plane), h('span', null, 'size'), h('span', null, `${r.n} voxels · ${fmt(r.areaUm2, 0)} µm²`),
    h('span', null, 'observed'), h('span', null, pct(r.observed, 0)), h('span', null, 'interpolated'), h('span', null, pct(r.interp, 0)),
    h('span', null, 'mean entropy'), h('span', null, fmt(r.entropy, 3) + ' bits')];
  r.classFrac.forEach((f, c) => out.push(h('span', null, P.classNames[c]), h('span', null, pct(f, 0))));
  if (r.accuracy !== null) out.push(h('span', null, 'vs truth'), h('span', null, pct(r.accuracy, 0) + ' agreement'));
  return out;
}
function setStatus(msg) { const e = $('#st-msg'); if (e) e.textContent = msg; }
function buildStatus() {
  const sb = $('#statusbar'); sb.innerHTML = '';
  sb.append(h('span', { class: 'grow', id: 'st-msg' }, 'Ready'), h('span', { id: 'st-cur' }, ''), h('span', { id: 'st-prov' }, ''), h('span', null, 'research use only'));
}
function updateStatus() {
  if (!S.P) return;
  const [z, y, x] = S.cur, k = IDX(z, y, x);
  $('#st-cur').textContent = `z ${S.cur[0]} (${fmt(S.P.zUm[z], 0)} µm)  y ${y}  x ${x}  ·  ${PROV_NAME[S.vol.prov[k]]}`;
  const f = S.P.analysis && S.P.analysis.summary && S.P.analysis.summary.provenance_fraction;
  if (f) $('#st-prov').textContent = `observed ${pct(f.observed, 0)} · interpolated ${pct(f.interpolated, 0)}`;
}
