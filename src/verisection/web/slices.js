/* ===== orthogonal slice views (axial / coronal / sagittal) ===== */
const PLANES = ['axial', 'coronal', 'sagittal'];
const PLANE_TITLE = { axial: 'Axial  (x–y)', coronal: 'Coronal  (x–z)', sagittal: 'Sagittal  (y–z)' };
function dimsOf(pl) {
  const [Z, Y, X] = S.dims, px = S.P.pixelUm, dz = S.P.dzUm;
  return pl === 'axial' ? { W: X, H: Y, sw: px, sh: px, n: Z } : pl === 'coronal' ? { W: X, H: Z, sw: px, sh: dz, n: Y } : { W: Y, H: Z, sw: px, sh: dz, n: X };
}
const sliceIndex = pl => pl === 'axial' ? S.cur[0] : pl === 'coronal' ? S.cur[1] : S.cur[2];
function setSliceIndex(pl, v) {
  const d = dimsOf(pl); v = clamp(Math.round(v), 0, d.n - 1);
  if (pl === 'axial') S.cur[0] = v; else if (pl === 'coronal') S.cur[1] = v; else S.cur[2] = v;
}
function voxIndex(pl, i, j) {
  const [, Y, X] = S.dims, [cz, cy, cx] = S.cur;
  if (pl === 'axial') return (cz * Y + j) * X + i;
  if (pl === 'coronal') return (j * Y + cy) * X + i;
  return (j * Y + i) * X + cx;
}
function cursorInPlane(pl) { return pl === 'axial' ? [S.cur[2], S.cur[1]] : pl === 'coronal' ? [S.cur[2], S.cur[0]] : [S.cur[1], S.cur[0]]; }
function setCursorFromPlane(pl, i, j) {
  const d = dimsOf(pl); i = clamp(Math.round(i), 0, d.W - 1); j = clamp(Math.round(j), 0, d.H - 1);
  if (pl === 'axial') { S.cur[2] = i; S.cur[1] = j; } else if (pl === 'coronal') { S.cur[2] = i; S.cur[0] = j; } else { S.cur[1] = i; S.cur[0] = j; }
}
function viewXform(pl, size) {   // size = {cw, ch} in CSS px; defaults to the live canvas (exports pass their own)
  const c = S.canv[pl]; const d = dimsOf(pl);
  const cw = size ? size.cw : c.width / devicePixelRatio, ch = size ? size.ch : c.height / devicePixelRatio, small = ch < 170 || cw < 230;
  // margins: room for the z ruler (mx), axis labels (mt) and the footer legend (mb); compact panels drop the annotations instead of overlapping them
  const mx = small ? 14 : 32, mt = small ? 6 : 22, mb = small ? 6 : 26;
  const scale = Math.min((cw - mx * 2) / (d.W * d.sw), (ch - mt - mb) / (d.H * d.sh)) * S.zoom[pl];
  const dw = d.W * d.sw * scale, dh = d.H * d.sh * scale;
  const ox = (cw - dw) / 2 + S.pan[pl][0], oy = mt + (ch - mt - mb - dh) / 2 + S.pan[pl][1];
  return { scale, dw, dh, ox, oy, d, cw, ch, small };
}
/* one viewport pattern: header (title · stepper · position · tools) over a canvas body with a footer legend */
function stepperFor(pl) {
  const go = v => { setSliceIndex(pl, v); S.active = pl; refreshAll(); };
  const inp = h('input', { type: 'number', id: 'sl-' + pl, min: 0, max: 0, step: 1, value: 0, 'aria-label': PLANE_TITLE[pl] + ' slice index',
    onchange: e => go(+e.target.value), onfocus: e => e.target.select() });
  return h('div', { class: 'stepper', role: 'group' },
    h('button', { title: 'Previous slice (↑)', 'aria-label': 'Previous slice', onclick: () => go(sliceIndex(pl) - 1) }, '‹'), inp,
    h('span', { class: 'of', id: 'of-' + pl }, ''),
    h('button', { title: 'Next slice (↓)', 'aria-label': 'Next slice', onclick: () => go(sliceIndex(pl) + 1) }, '›'));
}
function buildViewports() {
  const root = $('#viewports'); root.innerHTML = ''; S.canv = {};
  const tools = (resetFn, id) => h('span', { class: 'vtools' },
    h('button', { class: 'ibtn', title: 'Reset view (double-click)', 'aria-label': 'Reset view', onclick: resetFn }, '⟲'),
    h('button', { class: 'ibtn', title: 'Maximise / restore this view', 'aria-label': 'Maximise view', onclick: () => { S.active = id; setLayout(S.layout === 'single' ? 'quad' : 'single'); } }, '⤢'));
  PLANES.forEach(pl => {
    const cv = h('canvas', {}); S.canv[pl] = cv;
    const vp = h('div', { class: 'vp', id: 'vp-' + pl },
      h('div', { class: 'vhead' }, h('b', { class: 'vtitle' }, PLANE_TITLE[pl].split(/\s+\(/)[0], h('span', { class: 'vplane' }, ' (' + PLANE_TITLE[pl].split(/\s+\(/)[1])), stepperFor(pl), h('span', { class: 'vpos', id: 'sn-' + pl }, ''),
        tools(() => { S.zoom[pl] = 1; S.pan[pl] = [0, 0]; renderSlice(pl); }, pl)),
      h('div', { class: 'vbody' }, cv, h('div', { class: 'vfoot', id: 'vf-' + pl }, '')));
    root.append(vp);
    attachSliceEvents(pl, cv, vp);
  });
  const vp3 = h('div', { class: 'vp', id: 'vp-3d' },
    h('div', { class: 'vhead' }, h('b', { class: 'vtitle' }, '3D volume'), h('span', { class: 'vpos', id: 'mode3d' }, ''), tools(() => GL3D.resetCamera(), '3d')),
    h('div', { class: 'vbody' }, h('canvas', { id: 'cv3d' }), h('canvas', { id: 'ov3d', style: 'pointer-events:none' }), h('div', { class: 'vfoot', id: 'vf-3d' }, '')));
  root.append(vp3);
  vp3.addEventListener('mousedown', () => { S.active = '3d'; markActive(); });
  new ResizeObserver(() => { fitCanvases(); }).observe(root);
}
function markActive() { $$('.vp').forEach(v => v.classList.toggle('active', v.id === 'vp-' + S.active)); }
function fitCanvases() {
  const dpr = devicePixelRatio || 1;
  $$('#viewports canvas').forEach(c => {
    const r = c.parentElement.getBoundingClientRect();
    const w = Math.max(2, Math.round(r.width * dpr)), hh = Math.max(2, Math.round(r.height * dpr));
    if (c.width !== w || c.height !== hh) { c.width = w; c.height = hh; }
  });
  if (S.P) { renderAllSlices(); GL3D.dirty = true; try { GL3D.draw(); GL3D.dirty = false; } catch (e) { console.error(e); } }  // redraw at once: resizing clears the canvas
}
function updateSteppers() {
  if (!S.P) return;
  PLANES.forEach(pl => {
    const d = dimsOf(pl), s = $('#sl-' + pl); if (!s) return;
    s.max = d.n - 1; if (document.activeElement !== s) s.value = sliceIndex(pl);
    $('#of-' + pl).textContent = '/ ' + (d.n - 1);
    const ax = pl === 'axial' ? 'z' : pl === 'coronal' ? 'y' : 'x', um = pl === 'axial' ? S.P.zUm[sliceIndex(pl)] : sliceIndex(pl) * S.P.pixelUm;
    $('#sn-' + pl).textContent = `${ax} = ${fmt(um, 0)} µm`;
  });
  updateStatus(); markActive();
}
const offCanvas = (pl, W, H) => {
  let o = S.off[pl]; if (!o || o.w !== W || o.h !== H) { const c = document.createElement('canvas'); c.width = W; c.height = H; o = S.off[pl] = { c, w: W, h: H, ctx: c.getContext('2d') }; }
  return o;
};
function renderAllSlices() { if (!S.P) return; PLANES.forEach(pl => renderSlice(pl)); }

/* renderSlice draws one plane. With no target it paints the live canvas; exports pass
   tgt = { cv, k, cw, ch }: a canvas of (cw*k) x (ch*k) pixels for a cw x ch CSS-pixel panel. */
function renderSlice(pl, tgt) {
  if (!S.P || !S.canv[pl]) return;
  const live = !tgt, dpr = devicePixelRatio || 1;
  const cv = live ? S.canv[pl] : tgt.cv, k = live ? dpr : tgt.k, cw = live ? cv.width / dpr : tgt.cw, ch = live ? cv.height / dpr : tgt.ch;
  const ctx = cv.getContext('2d'), V = S.vol, P = S.P;
  ctx.setTransform(k, 0, 0, k, 0, 0);
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--vp') || '#000';
  ctx.fillRect(0, 0, cw, ch);
  const T = viewXform(pl, { cw, ch }), d = T.d;
  const off = offCanvas(pl, d.W, d.H), img = off.ctx.createImageData(d.W, d.H), D = img.data;
  const Lb = S.L.base, Ll = S.L.labels, Lu = S.L.unc, Lp = S.L.prov, Lt = S.L.truth, Le = S.L.err, Lv = S.L.varReg;
  const cmU = CM[Lu.cmap] || CM.magma, pal = S.palette;
  for (let j = 0; j < d.H; j++) {
    for (let i = 0; i < d.W; i++) {
      const k = voxIndex(pl, i, j);
      let r, g, b;
      if (Lb.on) { const v = V.intensity[k] * 3; r = CM.he[v] * Lb.op + 18 * (1 - Lb.op); g = CM.he[v + 1] * Lb.op + 18 * (1 - Lb.op); b = CM.he[v + 2] * Lb.op + 18 * (1 - Lb.op); } else { r = g = b = 18; }
      const lb = V.labels[k], pv = V.prov[k];
      let a;
      if (Ll.on && lb > 0 && S.classOn[lb - 1]) { const c = pal[lb - 1]; a = Ll.op; r += (c[0] - r) * a; g += (c[1] - g) * a; b += (c[2] - b) * a; }
      if (Lu.on && pv > 0) { const u = V.unc[k]; a = Lu.op * (0.2 + 0.8 * u / 255); r += (cmU[u * 3] - r) * a; g += (cmU[u * 3 + 1] - g) * a; b += (cmU[u * 3 + 2] - b) * a; }
      if (Lv.on && pv > 0) { const u = V.varReg[k]; a = Lv.op * u / 255; r += (CM.turbo[u * 3] - r) * a; g += (CM.turbo[u * 3 + 1] - g) * a; b += (CM.turbo[u * 3 + 2] - b) * a; }
      if (Le.on && V.err && V.err[k]) { a = Le.op; r += (255 - r) * a; g += (40 - g) * a; b += (40 - b) * a; }
      if (Lp.on) {
        if (pv === 2) { const c = PROV_RGB[2]; a = ((i + j) & 3) < 1 ? Lp.op * 0.95 : Lp.op * 0.13; r += (c[0] - r) * a; g += (c[1] - g) * a; b += (c[2] - b) * a; }
        else if (pv === 0) { r *= 0.35; g *= 0.35; b *= 0.35; }
      }
      if (Lt.on && V.truth) {
        const t = V.truth[k], kr = i + 1 < d.W ? voxIndex(pl, i + 1, j) : k, kd = j + 1 < d.H ? voxIndex(pl, i, j + 1) : k;
        if (V.truth[kr] !== t || V.truth[kd] !== t) { r = 255; g = 255; b = 255; }
      }
      const o = (j * d.W + i) * 4; D[o] = r; D[o + 1] = g; D[o + 2] = b; D[o + 3] = 255;
    }
  }
  off.ctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = !!S.view.smooth;
  ctx.drawImage(off.c, T.ox, T.oy, T.dw, T.dh);
  ctx.strokeStyle = '#ffffff30'; ctx.lineWidth = 1; ctx.strokeRect(T.ox - .5, T.oy - .5, T.dw + 1, T.dh + 1);
  const sx = d.sw * T.scale, sy = d.sh * T.scale;
  // provenance ruler along the z axis for the two vertical views
  if (S.view.ruler && pl !== 'axial') {
    for (let j = 0; j < d.H; j++) { const pv = planeProv(j); const c = PROV_RGB[pv]; ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.fillRect(T.ox - 9, T.oy + j * sy, 5, Math.max(1, sy)); }
    ctx.fillStyle = '#9fb0c6'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.save(); ctx.translate(T.ox - 20, T.oy + T.dh / 2); ctx.rotate(-Math.PI / 2);
    if (!T.small) ctx.fillText(T.dh > 150 ? 'z · section planes' : 'z', 0, 0); ctx.restore(); ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';   // the color key lives in the footer legend
  }
  // measurements and ROI overlays
  ctx.lineWidth = 1.5;
  S.measures.forEach(m => {
    const p = m.p.map(q => planeCoord(pl, q)), c0 = [T.ox + (p[0][0] + .5) * sx, T.oy + (p[0][1] + .5) * sy], c1 = [T.ox + (p[1][0] + .5) * sx, T.oy + (p[1][1] + .5) * sy];
    ctx.strokeStyle = '#ffd24d'; ctx.setLineDash([5, 3]); ctx.beginPath(); ctx.moveTo(...c0); ctx.lineTo(...c1); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#ffd24d'; ctx.font = '11px sans-serif'; ctx.fillText(fmt(m.d, 1) + ' µm', (c0[0] + c1[0]) / 2 + 4, (c0[1] + c1[1]) / 2 - 4);
  });
  if (S.pendingMeasure && S.pendingMeasure.length) {
    const q = planeCoord(pl, S.pendingMeasure[0]); ctx.fillStyle = '#ffd24d'; ctx.beginPath(); ctx.arc(T.ox + (q[0] + .5) * sx, T.oy + (q[1] + .5) * sy, 4, 0, 7); ctx.fill();
  }
  if (S.roi && S.roi.plane === pl) { ctx.strokeStyle = '#5dd39e'; ctx.setLineDash([4, 2]); ctx.strokeRect(T.ox + S.roi.i0 * sx, T.oy + S.roi.j0 * sy, (S.roi.i1 - S.roi.i0 + 1) * sx, (S.roi.j1 - S.roi.j0 + 1) * sy); ctx.setLineDash([]); }
  if (S.dragRoi && S.dragRoi.plane === pl) { const r = S.dragRoi; ctx.strokeStyle = '#5dd39e'; ctx.setLineDash([2, 2]); ctx.strokeRect(T.ox + Math.min(r.i0, r.i1) * sx, T.oy + Math.min(r.j0, r.j1) * sy, (Math.abs(r.i1 - r.i0) + 1) * sx, (Math.abs(r.j1 - r.j0) + 1) * sy); ctx.setLineDash([]); }
  // crosshair
  if (S.view.cross) {
    const [ci, cj] = cursorInPlane(pl), px_ = T.ox + (ci + .5) * sx, py_ = T.oy + (cj + .5) * sy;
    ctx.strokeStyle = '#4da3ffcc'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px_, T.oy); ctx.lineTo(px_, T.oy + T.dh); ctx.moveTo(T.ox, py_); ctx.lineTo(T.ox + T.dw, py_); ctx.stroke();
  }
  // scale bar + axis labels
  if (S.view.scale) {
    const targets = [10, 20, 50, 100, 200, 500], want = T.dw / 5 / (T.scale);
    let nice = targets[0]; for (const t of targets) if (t <= want) nice = t;
    const len = nice * T.scale; ctx.fillStyle = '#fff'; ctx.fillRect(T.ox + T.dw - len - 8, T.oy + T.dh - 12, len, 3);
    ctx.font = '11px sans-serif'; ctx.textAlign = 'right'; ctx.fillStyle = '#e8eef8'; ctx.fillText(nice + ' µm', T.ox + T.dw - 8, T.oy + T.dh - 17); ctx.textAlign = 'left';
  }
  if (S.view.labels && !T.small) { ctx.fillStyle = '#9fb0c6'; ctx.font = '11px sans-serif'; ctx.fillText(pl === 'sagittal' ? 'y →' : 'x →', T.ox + T.dw - 30, T.oy - 4); ctx.fillText(pl === 'axial' ? 'y ↓' : 'z ↓', T.ox + 2, T.oy - 4); }
  if (!live && pl !== 'axial' && S.view.ruler && !T.small) {   // exports: the footer legend is drawn on the canvas
    ctx.font = '11px "Segoe UI",sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; let lx = 10;
    [[PROV_RGB[1], 'observed plane'], [PROV_RGB[2], 'interpolated']].forEach(([c, t]) => { ctx.fillStyle = `rgb(${c.join(',')})`; ctx.fillRect(lx, ch - 15, 9, 9); ctx.fillStyle = '#cfd8e6'; ctx.fillText(t, lx + 14, ch - 10); lx += 14 + ctx.measureText(t).width + 14; });
    ctx.textBaseline = 'alphabetic';
  }
  const f = live ? $('#vf-' + pl) : null;
  if (f) {
    const k = voxIndex(pl, ...cursorInPlane(pl)), sw = (c, t) => h('span', { class: 'lg' }, h('i', { style: 'background:rgb(' + c.join(',') + ')' }), t);
    const kids = [h('span', null, `${PROV_NAME[S.vol.prov[k]]} · ${S.zoom[pl].toFixed(1)}×`)];
    if (pl !== 'axial' && S.view.ruler && !T.small) kids.push(h('span', { class: 'hintx', style: 'display:inline-flex;gap:10px' }, sw(PROV_RGB[1], 'observed plane'), sw(PROV_RGB[2], 'interpolated')));
    f.replaceChildren(...kids); f.classList.toggle('hidden', T.small);   // compact panels: the cursor info is in the header and status bar
  }
}
function planeProv(j) {   // provenance of z-plane j (majority of sampled voxels), cached
  const z = S._pp || (S._pp = (() => {
    const [Z, Y, X] = S.dims, out = new Uint8Array(Z), n = Y * X;
    for (let k = 0; k < Z; k++) { const c = [0, 0, 0, 0]; for (let q = 0; q < n; q += 7) c[S.vol.prov[k * n + q]]++; out[k] = c[1] >= c[2] && c[1] > c[0] ? 1 : c[2] > c[0] ? 2 : 0; }
    return out;
  })());
  return z[j];
}
function planeCoord(pl, p) { return pl === 'axial' ? [p.x, p.y] : pl === 'coronal' ? [p.x, p.z] : [p.y, p.z]; }

function attachSliceEvents(pl, cv, vp) {
  const toVox = e => {
    const r = cv.getBoundingClientRect(), T = viewXform(pl), d = T.d;
    const x = e.clientX - r.left, y = e.clientY - r.top;
    return { i: (x - T.ox) / (d.sw * T.scale) - 0.5, j: (y - T.oy) / (d.sh * T.scale) - 0.5, T };
  };
  let drag = null;
  cv.addEventListener('mousedown', e => {
    if (!S.P) return; S.active = pl; markActive();
    const { i, j } = toVox(e);
    if (e.button === 2 || e.shiftKey || e.button === 1) { drag = { type: 'pan', x: e.clientX, y: e.clientY, p0: S.pan[pl].slice() }; e.preventDefault(); return; }
    if (S.tool === 'cursor') { drag = { type: 'cur' }; setCursorFromPlane(pl, i, j); refreshAll(); }
    else if (S.tool === 'measure') {
      const d = dimsOf(pl); const cx = clamp(Math.round(i), 0, d.W - 1), cy = clamp(Math.round(j), 0, d.H - 1);
      const p = { z: S.cur[0], y: S.cur[1], x: S.cur[2] };
      if (pl === 'axial') { p.x = cx; p.y = cy; } else if (pl === 'coronal') { p.x = cx; p.z = cy; } else { p.y = cx; p.z = cy; }
      S.pendingMeasure = S.pendingMeasure || []; S.pendingMeasure.push(p);
      if (S.pendingMeasure.length === 2) {
        const [a, b] = S.pendingMeasure, px = S.P.pixelUm, dxu = (b.x - a.x) * px, dyu = (b.y - a.y) * px, dzu = S.P.zUm[b.z] - S.P.zUm[a.z];
        S.measures.push({ p: [a, b], dx: dxu, dy: dyu, dz: dzu, d: Math.hypot(dxu, dyu, dzu) }); S.pendingMeasure = [];
        setStatus('Distance ' + fmt(S.measures[S.measures.length - 1].d, 1) + ' µm');
      }
      refreshAll();
    } else if (S.tool === 'roi') { const d = dimsOf(pl); drag = { type: 'roi' }; S.dragRoi = { plane: pl, i0: clamp(Math.round(i), 0, d.W - 1), j0: clamp(Math.round(j), 0, d.H - 1) }; S.dragRoi.i1 = S.dragRoi.i0; S.dragRoi.j1 = S.dragRoi.j0; }
  });
  window.addEventListener('mousemove', e => {
    if (!drag || !S.P) return;
    if (drag.type === 'pan') { S.pan[pl] = [drag.p0[0] + e.clientX - drag.x, drag.p0[1] + e.clientY - drag.y]; renderSlice(pl); }
    else if (drag.type === 'cur') { const { i, j } = toVox(e); setCursorFromPlane(pl, i, j); refreshAll(); }
    else if (drag.type === 'roi') { const { i, j } = toVox(e), d = dimsOf(pl); S.dragRoi.i1 = clamp(Math.round(i), 0, d.W - 1); S.dragRoi.j1 = clamp(Math.round(j), 0, d.H - 1); renderSlice(pl); }
  });
  window.addEventListener('mouseup', () => {
    if (drag && drag.type === 'roi' && S.dragRoi) { finishRoi(); }
    drag = null;
  });
  cv.addEventListener('contextmenu', e => e.preventDefault());
  cv.addEventListener('wheel', e => {
    if (!S.P) return; e.preventDefault();
    if (e.ctrlKey || e.metaKey) { S.zoom[pl] = clamp(S.zoom[pl] * (e.deltaY < 0 ? 1.12 : 1 / 1.12), 0.4, 12); renderSlice(pl); }
    else { setSliceIndex(pl, sliceIndex(pl) + (e.deltaY > 0 ? 1 : -1)); S.active = pl; refreshAll(); }
  }, { passive: false });
  cv.addEventListener('dblclick', () => { S.zoom[pl] = 1; S.pan[pl] = [0, 0]; renderSlice(pl); });
}
function finishRoi() {
  const r = S.dragRoi; S.dragRoi = null; if (!r) return;
  const pl = r.plane, i0 = Math.min(r.i0, r.i1), i1 = Math.max(r.i0, r.i1), j0 = Math.min(r.j0, r.j1), j1 = Math.max(r.j0, r.j1);
  const C = S.P.classNames.length, cnt = new Array(C).fill(0); let n = 0, obs = 0, itp = 0, ent = 0, agree = 0, tn = 0;
  for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
    const k = voxIndex(pl, i, j), pv = S.vol.prov[k]; if (pv === 0) continue;
    n++; cnt[S.vol.labels[k] - 1]++; if (pv === 1) obs++; if (pv === 2) itp++; ent += S.vol.unc[k] / 255 * S.P.uncScale;
    if (S.vol.truth) { tn++; if (S.vol.truth[k] === S.vol.labels[k]) agree++; }
  }
  const d = dimsOf(pl);
  S.roi = { plane: pl, i0, i1, j0, j1, n, areaUm2: (i1 - i0 + 1) * (j1 - j0 + 1) * d.sw * d.sh, observed: n ? obs / n : 0, interp: n ? itp / n : 0, entropy: n ? ent / n : 0, classFrac: cnt.map(c => n ? c / n : 0), accuracy: tn ? agree / tn : null };
  refreshAll();
}
function setLayout(name) {
  S.layout = name; const root = $('#viewports'); if (!root) return;
  const v = id => $('#vp-' + id);
  ['axial', 'coronal', 'sagittal', '3d'].forEach(id => { const e = v(id); if (e) { e.style.gridArea = ''; e.classList.remove('hidden'); } });
  root.style.gridTemplateColumns = ''; root.style.gridTemplateRows = ''; root.style.gridTemplateAreas = '';
  const area = (id, a) => { const e = v(id); if (e) e.style.gridArea = a; };
  if (name === 'quad') { root.style.gridTemplateAreas = '"a c" "s t"'; root.style.gridTemplateColumns = '1fr 1fr'; root.style.gridTemplateRows = '1fr 1fr'; area('axial', 'a'); area('coronal', 'c'); area('sagittal', 's'); area('3d', 't'); }
  else if (name === 'focus3d') { root.style.gridTemplateAreas = '"t a" "t c" "t s"'; root.style.gridTemplateColumns = '2fr 1fr'; root.style.gridTemplateRows = '1fr 1fr 1fr'; area('axial', 'a'); area('coronal', 'c'); area('sagittal', 's'); area('3d', 't'); }
  else if (name === 'mprrow') { root.style.gridTemplateAreas = '"a c s" "t t t"'; root.style.gridTemplateColumns = '1fr 1fr 1fr'; root.style.gridTemplateRows = '1fr 1.3fr'; area('axial', 'a'); area('coronal', 'c'); area('sagittal', 's'); area('3d', 't'); }
  else if (name === 'mpr') { root.style.gridTemplateAreas = '"a c s"'; root.style.gridTemplateColumns = '1fr 1fr 1fr'; root.style.gridTemplateRows = '1fr'; area('axial', 'a'); area('coronal', 'c'); area('sagittal', 's'); v('3d').classList.add('hidden'); }
  else if (name === 'single') { root.style.gridTemplateAreas = '"x"'; root.style.gridTemplateColumns = '1fr'; root.style.gridTemplateRows = '1fr'; ['axial', 'coronal', 'sagittal', '3d'].forEach(id => { if (id === S.active) area(id, 'x'); else v(id).classList.add('hidden'); }); }
  setTimeout(() => { fitCanvases(); }, 30);
}
