/* ===== Image audit workspace: real 2D patches -> stain, quality, texture, descriptor separability, stability, interpolation =====
   A second workspace beside the reconstruction one. It reuses the shell (menubar, docks, analysis dock, export) and swaps the
   content: gallery instead of slice views, dataset filter instead of layers, image inspector instead of voxel inspector. */
const AUD_SRC_COL = { LC25000: '#4da3ff', BreakHis: '#f0a030', Oral: '#5dd39e', Ovarian: '#c77dff' };
const AUD_VERDICT_COL = { accept: '#5dd39e', flag: '#e0b04a', exclude: '#e5645f' };
const AUD_TABS = ['Overview', 'Stain', 'Quality', 'Texture', 'Separability', 'Explanation', 'Stability', 'Interpolation', 'Method & data'];
const AUD_FEATS = [['glcm_contrast_2px', 'GLCM contrast · 2 px'], ['glcm_homogeneity_2px', 'GLCM homogeneity · 2 px'], ['glcm_entropy_2px', 'GLCM entropy · 2 px'],
  ['glcm_correlation_2px', 'GLCM correlation · 2 px'], ['glcm_contrast_8px', 'GLCM contrast · 8 px']];
const AUD_REFS = [
  'Borkowski, A.A., et al. (2019). Lung and colon cancer histopathological image dataset (LC25000). arXiv:1912.12142.',
  'Spanhol, F.A., et al. (2016). A dataset for breast cancer histopathological image classification. IEEE Trans. Biomed. Eng., 63(7), 1455–1462.',
  'Rahman, T.Y., et al. (2020). Histopathological imaging database for oral cancer analysis. Data in Brief, 29, 105114.',
  'Kasture, K.R., et al. (2021). A new deep learning method for automatic ovarian cancer prediction & subtype classification. Turkish J. Comput. Math. Educ., 12(12), 1233–1242.'];
Object.assign(HELP, {
  'Stain concentration (H vs E)': 'One point per patch: mean hematoxylin (x) and eosin (y) concentration over stained pixels, from Macenko-estimated vectors. Separate clouds per collection are stain / domain shift, not biology.',
  'Stain-model residual by class': 'RMS optical-density error of the two-stain model. High values mean the H&E model does not describe the patch (pen marks, blur, unusual stain).',
  'Estimated H vector vs Ruifrok default (angle)': 'Angle between the hematoxylin vector estimated from the patch and the textbook Ruifrok vector. Large angles mean a different stain recipe or scanner.',
  'Stained-pixel fraction by class': 'Share of pixels that carry stain (summed OD above 0.45). Low values mean pale or sparse tissue, which makes stain estimates less stable.',
  'Focus vs noise': 'Variance of the Laplacian (sharpness, x) against Haar noise sigma (y), log scales. Color = QC verdict from robust z-scores.',
  'QC verdicts by collection': 'Share of patches accepted, flagged or excluded. Pooled: judged against all patches together (a collection that looks different is flagged). Within-source: each collection is its own reference.',
  'Texture descriptor space (PCA)': 'First two principal components of the 24 GLCM descriptors (4 offsets x 6 features). Distance means similar texture. Overlap between classes means these descriptors alone do not separate them.',
  'Descriptor by class': 'Distribution of one GLCM descriptor per class (box = quartiles, line = median).',
  'Descriptor separability (5-fold)': 'Cross-validated accuracy of a random forest on the descriptors. Dashed line = majority-class baseline. This measures whether descriptors carry signal, not diagnostic performance.',
  'Domain shift: in-source vs leave-one-source-out': 'AUROC for tumor vs non-tumor. Blue: trained and tested inside one collection. Orange: trained on the other three collections, tested on this one. The gap is domain shift.',
  'What drives the tumor score (TreeSHAP share)': 'Share of the mean absolute SHAP value carried by each descriptor group in a random forest trained on the accepted patches. A large stain share means the model leans on how the images were made, not on tissue structure.',
  'Stability (ICC lower bound), all perturbations': 'Lower 95% bound of ICC(2,1) across noise, blur and optical-density perturbations. A descriptor is certified only if it reaches the threshold (dashed).',
  'Stability by perturbation family': 'Same measure, one family of perturbations at a time. Shows which acquisition change breaks which descriptor.',
  'Stability vs GLCM offset': 'Lower ICC bound for all perturbations as the co-occurrence offset grows (pixels). Larger scales are usually more stable against noise.',
  'Texture bias under naive interpolation': 'Relative bias of GLCM features in an intensity-interpolated plane (alpha = distance fraction between two sections; 0 = observed). Real texture, simulated sectioning.',
});

const AS = S.aud = { on: {}, colorBy: 'class', scen: 'independent', sel: null, selfeat: 'glcm_contrast_2px', sepKind: 'combined', qcScope: 'pooled', verdictFilter: 'all' };
const audRowsAll = () => (S.A ? S.A.rows : []);
const audRows = () => audRowsAll().filter(r => AS.on[r.class] !== false);
const audOk = () => audRows().filter(r => r.verdict === 'accept');
const audGateNote = () => { const n = audRows().length - audOk().length; return n ? audNote(n + ' patch' + (n > 1 ? 'es' : '') + ' failed the quality gate (see Quality) and ' + (n > 1 ? 'are' : 'is') + ' left out of this tab, as in every texture experiment.') : null; };
const audById = id => audRowsAll().find(r => r.id === id);
function audShade(hex, t) { const c = hex2rgb(hex).map(v => Math.round(t >= 0 ? v + (255 - v) * t : v * (1 + t))); return '#' + c.map(v => v.toString(16).padStart(2, '0')).join(''); }
function audClassColor(key) {
  const cl = S.A.classes, c = cl.find(k => k.key === key), same = cl.filter(k => k.source === c.source), i = same.indexOf(c);
  return audShade(AUD_SRC_COL[c.source] || '#8d96a6', same.length > 1 ? -0.35 + 0.7 * i / (same.length - 1) : 0);
}
const audColor = r => AS.colorBy === 'source' ? AUD_SRC_COL[r.source] : audClassColor(r.class);
function audGroups(rows) {   // [{name, color, rows}] in registry order
  if (AS.colorBy === 'source') return Object.keys(AUD_SRC_COL).map(s => ({ name: s, color: AUD_SRC_COL[s], rows: rows.filter(r => r.source === s) })).filter(g => g.rows.length);
  return S.A.classes.map(c => ({ name: c.name, color: audClassColor(c.key), rows: rows.filter(r => r.class === c.key) })).filter(g => g.rows.length);
}
const audMedian = a => { const b = a.filter(Number.isFinite).sort((x, y) => x - y); return b.length ? b[b.length >> 1] : NaN; };
const audMean = a => { const b = a.filter(Number.isFinite); return b.length ? b.reduce((s, v) => s + v, 0) / b.length : NaN; };
function audPlot(div, traces, layout, pick) {
  plot(div, traces, layout);
  if (pick) requestAnimationFrame(() => { if (div.on) { try { div.removeAllListeners('plotly_click'); } catch (e) { /* none */ } div.on('plotly_click', ev => { const p = ev.points && ev.points[0]; if (p && p.customdata !== undefined) audSelect(p.customdata); }); } });
}

/* ---------- workspace switching ---------- */
function setWorkspace(ws) {
  if (ws === 'audit') {
    if (!S.A) { if (window.VS_SERVER) auditDialog(); else toast('No image audit loaded. Run:  verisection audit DATASET_DIR --out audit.html'); syncToolbar(); return; }
    S.ws = 'audit'; document.body.classList.add('ws-audit'); hideWelcome();
    $('#dockL-title').textContent = 'Collections'; $('#dockR-title').textContent = 'Image inspector';
    buildAuditDocks(); buildTabs(); audGallery(); audInspect(); setStatus('Image audit: ' + S.A.title); document.title = 'VeriSection Workstation — ' + S.A.title;
  } else {
    S.ws = 'volume'; document.body.classList.remove('ws-audit');
    $('#dockL-title').textContent = 'Layers'; $('#dockR-title').textContent = 'Inspector';
    buildTabs(); if (!S.P) showWelcome(); else hideWelcome(); setTimeout(() => { fitCanvases(); GL3D.dirty = true; }, 30);
    document.title = 'VeriSection Workstation' + (S.P ? ' — ' + S.P.title : '');
  }
  syncToolbar(); closeMenus();
}
function loadAudit(p) {
  S.A = p; AS.on = {}; p.classes.forEach(c => { AS.on[c.key] = true; });
  AS.sel = (p.rows.find(r => p.thumbs[r.id] !== undefined) || p.rows[0] || {}).id;
  setWorkspace('audit');
}

/* ---------- left dock: collections filter ---------- */
function buildAuditDocks() {
  const root = $('#audfilter'); root.innerHTML = '';
  const A = S.A, bySrc = {};
  A.classes.forEach(c => (bySrc[c.source] = bySrc[c.source] || []).push(c));
  const count = k => audRowsAll().filter(r => r.class === k).length;
  const setSrc = (src, on) => { bySrc[src].forEach(c => { AS.on[c.key] = on; }); buildAuditDocks(); audRefresh(); };
  root.append(h('div', { class: 'lrow2' }, h('span', { class: 'cap' }, 'Color by'), h('div', { class: 'seg fill' }, ['class', 'source'].map(k =>
    h('button', { class: AS.colorBy === k ? 'on' : '', onclick: () => { AS.colorBy = k; buildAuditDocks(); audRefresh(); } }, k)))));
  Object.keys(bySrc).forEach(src => {
    const all = bySrc[src].every(c => AS.on[c.key]);
    root.append(h('section', { class: 'lcard' },
      h('label', { class: 'lhead', title: A.sources[src] || src }, h('input', { type: 'checkbox', checked: all, onchange: e => setSrc(src, e.target.checked) }), h('span', { class: 'sw', style: 'width:10px;height:10px;border-radius:2px;background:' + AUD_SRC_COL[src] }), h('span', { class: 'lname' }, src),
        h('span', { class: 'tlabel' }, bySrc[src].reduce((s, c) => s + count(c.key), 0))),
      h('div', { class: 'lbody' }, bySrc[src].map(c => h('label', { class: 'crow', title: c.name + ' (' + c.group + ')' },
        h('input', { type: 'checkbox', checked: AS.on[c.key], onchange: e => { AS.on[c.key] = e.target.checked; buildAuditDocks(); audRefresh(); } }),
        h('span', { class: 'sw', style: 'background:' + audClassColor(c.key) }), h('span', null, c.name), h('span', { class: 'tlabel' }, count(c.key)))))));
  });
  root.append(h('div', { class: 'hint' }, 'Untick a class or collection to remove it from every chart, table and the gallery. Colors are shades of the collection hue.'));
}
function audRefresh() { audGallery(); if (S.ws === 'audit') openTab(S.tab); audInspect(); }

/* ---------- center: gallery ---------- */
function audGallery() {
  const root = $('#audview'); root.innerHTML = '';
  const rows = audRows().filter(r => S.A.thumbs[r.id] !== undefined && (AS.verdictFilter === 'all' || r.verdict === AS.verdictFilter));
  const head = h('div', { class: 'vhead' }, h('span', { class: 'vtitle' }, 'Gallery'), h('span', { class: 'vpos' }, rows.length + ' patches with thumbnails · click one to inspect it'),
    h('span', { class: 'vtools', style: 'gap:6px;align-items:center' }, h('span', { class: 'tlabel' }, 'show'), h('div', { class: 'seg' }, ['all', 'accept', 'flag', 'exclude'].map(v =>
      h('button', { class: AS.verdictFilter === v ? 'on' : '', onclick: () => { AS.verdictFilter = v; audGallery(); } }, v)))));
  const grid = h('div', { class: 'gal' });
  if (!rows.length) grid.append(emptyState('No patch matches the current filter.'));
  rows.forEach(r => {
    const fr = audColor(r), mark = r.verdict !== 'accept' ? h('i', { class: 'vmark', style: 'background:' + AUD_VERDICT_COL[r.verdict], title: r.verdict + (r.reasons ? ': ' + r.reasons : '') }) : null;
    grid.append(h('div', { class: 'tile' + (AS.sel === r.id ? ' sel' : ''), style: '--fr:' + fr, title: r.class_name + ' · ' + r.file + ' · ' + r.verdict, onclick: () => audSelect(r.id) },
      h('img', { src: 'data:image/jpeg;base64,' + S.A.thumbs[r.id], alt: r.class_name, loading: 'lazy' }), mark, h('span', { class: 'tcl' }, r.class_name)));
  });
  root.append(head, grid);
}
function audSelect(id) {
  AS.sel = id; $$('#audview .tile').forEach(t => t.classList.remove('sel'));
  const r = audById(id); if (!r) return;
  audGallery(); audInspect(); setStatus(r.class_name + ' · ' + r.file + ' · ' + r.verdict);
}

/* ---------- right dock: image inspector ---------- */
const RUI = (() => { const n = v => { const m = Math.hypot(...v); return v.map(x => x / m); }; return [n([0.644, 0.717, 0.267]), n([0.093, 0.954, 0.283])]; })();
function audStainPreview(img, canvases) {   // Ruifrok deconvolution of the thumbnail in the browser: H and E concentration maps
  const w = img.naturalWidth, hh = img.naturalHeight, c = document.createElement('canvas'); c.width = w; c.height = hh;
  const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(img, 0, 0); const d = g.getImageData(0, 0, w, hh).data;
  const [H, E] = RUI, a = H[0] * H[0] + H[1] * H[1] + H[2] * H[2], b = H[0] * E[0] + H[1] * E[1] + H[2] * E[2], e2 = E[0] * E[0] + E[1] * E[1] + E[2] * E[2], det = a * e2 - b * b;
  const ch = [new Float32Array(w * hh), new Float32Array(w * hh)]; let mx = [1e-6, 1e-6];
  for (let i = 0; i < w * hh; i++) {
    const od = [0, 1, 2].map(k => -Math.log10(Math.max(d[i * 4 + k], 1) / 255)), p = od[0] * H[0] + od[1] * H[1] + od[2] * H[2], q = od[0] * E[0] + od[1] * E[1] + od[2] * E[2];
    const ch0 = Math.max(0, (e2 * p - b * q) / det), ch1 = Math.max(0, (a * q - b * p) / det); ch[0][i] = ch0; ch[1][i] = ch1; mx[0] = Math.max(mx[0], ch0); mx[1] = Math.max(mx[1], ch1);
  }
  const tint = [[0.15, 0.2, 0.0], [0.0, 0.55, 0.2]];   // how strongly each stain darkens R, G, B (hematoxylin: blue-purple; eosin: pink)
  canvases.forEach((cv, s) => {
    cv.width = w; cv.height = hh; const o = cv.getContext('2d'), im = o.createImageData(w, hh);
    for (let i = 0; i < w * hh; i++) { const t = Math.min(1, ch[s][i] / (0.9 * mx[s])); im.data[i * 4] = 255 * (1 - t * (s ? 0.12 : 0.7)); im.data[i * 4 + 1] = 255 * (1 - t * (s ? 0.62 : 0.72)); im.data[i * 4 + 2] = 255 * (1 - t * (s ? 0.38 : 0.12)); im.data[i * 4 + 3] = 255; }
    o.putImageData(im, 0, 0);
  });
}
function audExplainCard(r) {   // forest score and the three descriptors that moved it most (TreeSHAP); describes the model, not the tissue
  const mx = Math.max(...r.xtop.map(t => Math.abs(t[1])), 1e-6), nice = n => n.replace(/^glcm_/, 'GLCM ').replace(/_(\d+)px$/, ' · $1 px').replace(/_/g, ' ');
  return h('div', null, h('div', { class: 'kv' }, h('span', null, 'forest score'), h('span', null, fmt(r.xp, 2) + ' (in-sample, tumor)')),
    ...r.xtop.map(t => h('div', { style: 'margin-top:5px' }, h('div', { style: 'display:flex;justify-content:space-between;font-size:11px' }, h('span', null, nice(t[0])), h('span', null, (t[1] > 0 ? '+' : '') + fmt(t[1], 3))),
      h('div', { style: 'height:6px;border-radius:3px;background:' + (t[1] > 0 ? '#e5645f' : '#4da3ff') + ';width:' + Math.max(4, 100 * Math.abs(t[1]) / mx) + '%' }))),
    h('div', { class: 'hint' }, 'Red pushes the forest score toward tumor, blue toward non-tumor. This describes the model, not the tissue, and it is not a diagnosis.'));
}
function audInspect() {
  const root = $('#audinsp'); if (!root) return; root.innerHTML = '';
  if (!S.A) return;
  const r = audById(AS.sel), card2 = (id, title, kids) => h('details', { class: 'icard', open: S.insp[id] !== false, ontoggle: e => { S.insp[id] = e.target.open; } }, h('summary', null, title), h('div', { class: 'ibody' }, kids));
  if (!r) { root.append(emptyState('Select a thumbnail in the gallery, or a point in a scatter plot, to inspect one patch.')); return; }
  const th = S.A.thumbs[r.id], img = h('img', { alt: r.class_name, style: 'width:100%;image-rendering:auto;border-radius:4px;display:block' });
  const ch = [h('canvas', { style: 'width:100%;border-radius:4px;display:block' }), h('canvas', { style: 'width:100%;border-radius:4px;display:block' })];
  if (th) { img.onload = () => audStainPreview(img, ch); img.src = 'data:image/jpeg;base64,' + th; }
  const kv = rows => h('div', { class: 'kv' }, rows.flatMap(([k, v]) => [h('span', null, k), h('span', null, v)]));
  const vchip = h('span', { class: 'badge', style: 'background:' + AUD_VERDICT_COL[r.verdict] + ';color:#10151c' }, r.verdict);
  root.append(
    card2('aimg', 'Patch', h('div', null, th ? img : emptyState('No thumbnail kept for this patch (the table rows are complete).'),
      th ? h('div', { style: 'display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:6px' }, h('div', null, ch[0], h('div', { class: 'cap', style: 'text-align:center;margin-top:2px' }, 'hematoxylin')), h('div', null, ch[1], h('div', { class: 'cap', style: 'text-align:center;margin-top:2px' }, 'eosin'))) : null,
      th ? h('div', { class: 'hint' }, 'Right: the two stain channels recovered by color deconvolution (Ruifrok vectors, computed in the browser from the thumbnail).') : null)),
    card2('aid', 'Identity', kv([['class', r.class_name], ['collection', r.source], ['group', r.group], ['file', r.file], ['QC verdict', vchip], ['worst z', fmt(r.z_max, 2)], ['reason', r.reasons || '—']])),
    card2('astain', 'Stain', kv([['stained pixels', pct(r.tissue_fraction, 0)], ['mean H', fmt(r.mean_h, 3)], ['mean E', fmt(r.mean_e, 3)], ['model residual', fmt(r.residual_rms, 4)], ['H angle vs Ruifrok', fmt(r.angle_h_deg, 1) + '°'], ['OD mean ± sd', fmt(r.od_mean, 3) + ' ± ' + fmt(r.od_sd, 3)]])),
    card2('aqual', 'Quality', kv([['black pixels', pct(r.black_fraction, 1)], ['focus (var. Laplacian)', fmt(r.focus_lapvar, 5)], ['noise σ (Haar)', fmt(r.noise_sigma, 5)]])),
    r.xtop ? card2('aexp', 'Explanation (tumor score)', audExplainCard(r)) : null,
    card2('atex', 'Texture (GLCM, pixels)', kv([['contrast · 2 px', fmt(r.glcm_contrast_2px, 3)], ['homogeneity · 2 px', fmt(r.glcm_homogeneity_2px, 3)], ['entropy · 2 px', fmt(r.glcm_entropy_2px, 3)], ['correlation · 2 px', fmt(r.glcm_correlation_2px, 3)], ['contrast · 8 px', fmt(r.glcm_contrast_8px, 3)]])));
}

/* ---------- analysis tabs ---------- */
function buildAuditTabs() {
  const bar = $('#tabs'); bar.innerHTML = '';
  if (!AUD_TABS.includes(S.tab)) S.tab = 'Overview';
  AUD_TABS.forEach(n => bar.append(h('div', { class: 'tab' + (n === S.tab ? ' on' : ''), onclick: () => openTab(n) }, n)));
  openAuditTab(S.tab);
}
function openAuditTab(name) {
  S.tab = name; S.tables[name] = [];
  $$('#tabs .tab').forEach(t => t.classList.toggle('on', t.textContent === name));
  const pane = $('#tabpane'); pane.innerHTML = ''; later.length = 0;
  try { pane.append(AUD_TAB[name](S.A.analysis || {})); } catch (e) { console.error(e); pane.append(h('div', { class: 'note warn' }, 'This tab could not be drawn: ' + e.message)); }
  requestAnimationFrame(() => later.forEach(f => { try { f(); } catch (e) { console.error(e); } }));
}
const audSeg = (opts, cur, set) => seg(opts, cur, v => { set(v); openTab(S.tab); });
const audBox = (div, groups, key, ytitle, log) => audPlot(div, groups.map(g => ({ type: 'box', name: g.name, y: g.rows.map(r => r[key]), marker: { color: g.color }, line: { width: 1.2 }, boxpoints: false })),
  { yaxis: { title: ytitle, type: log ? 'log' : 'linear' }, xaxis: { tickangle: -35, automargin: true }, showlegend: false, margin: { b: 96, l: 58 } });
const audDark = () => document.body.classList.contains('dark');
function audNote(text, warn) { return h('div', { class: 'note' + (warn ? ' warn' : '') }, text); }
function audGrid(items) { const grid = h('div', { class: 'grid2' }); items.forEach(([title, fn]) => { const { d } = plotDiv(); grid.append(h('div', null, head(title), d)); later.push(() => fn(d)); }); return grid; }

const AUD_TAB = {
  Overview() {
    const A = S.A, rows = audRows(), root = h('div', null);
    const fl = rows.filter(r => r.verdict !== 'accept').length;
    root.append(h('div', { class: 'card-row' }, card('patches', rows.length, A.classes.filter(c => AS.on[c.key]).length + ' of ' + A.classes.length + ' classes'), card('collections', new Set(rows.map(r => r.source)).size, 'public datasets'),
      card('stained pixels', pct(audMedian(rows.map(r => r.tissue_fraction)), 0), 'median per patch'), card('flagged or excluded', pct(fl / Math.max(rows.length, 1), 0), 'quality gate: ' + fl + ' of ' + rows.length), card('residual (median)', fmt(audMedian(rows.map(r => r.residual_rms)), 4), 'OD, two-stain model')));
    root.append(audNote('Independent 2D patches, not serial sections. They exercise the layers of VeriSection that need only one image: stain / optical-density diagnostics, image quality, physical-scale texture and the stability certification of texture descriptors. No 3D reconstruction is evaluated here, and nothing is a diagnostic claim.'));
    root.append(h('h4', null, 'Samples by class'), h('p', { class: 'help' }, 'Counts are all sampled patches; the means use accepted patches only. The same table is available as CSV from File ▸ Export ▸ Current tab tables.'),
      tbl(['class', 'collection', 'group', 'patches', 'stained px', 'mean H', 'mean E', 'residual', 'focus', 'noise σ', 'accept', 'flag', 'exclude'],
        A.classes.filter(c => AS.on[c.key]).map(c => { const all = audRowsAll().filter(r => r.class === c.key), g = all.filter(r => r.verdict === 'accept'), n = v => all.filter(r => r.verdict === v).length;
          return [c.name, c.source, c.group, all.length, +audMean(g.map(r => r.tissue_fraction)).toFixed(3), +audMean(g.map(r => r.mean_h)).toFixed(3), +audMean(g.map(r => r.mean_e)).toFixed(3), +audMean(g.map(r => r.residual_rms)).toFixed(4), +audMean(g.map(r => r.focus_lapvar)).toFixed(5), +audMean(g.map(r => r.noise_sigma)).toFixed(5), n('accept'), n('flag'), n('exclude')]; }), 'samples by class'));
    root.append(audNote('Next: Stain and Quality describe how the collections differ before any model sees them; Texture and Separability ask what the descriptors carry; Stability asks which of them can be trusted; Interpolation shows what gap-filling does to them.'));
    return root;
  },
  Stain() {
    const rows = audOk(), root = h('div', null), groups = audGroups(rows), srcGroups = Object.keys(AUD_SRC_COL).map(s => ({ name: s, color: AUD_SRC_COL[s], rows: rows.filter(r => r.source === s) })).filter(g => g.rows.length);
    root.append(h('div', { class: 'lrow' }, h('span', { class: 'tlabel' }, 'Color by'), audSeg(['class', 'source'], AS.colorBy, v => { AS.colorBy = v; buildAuditDocks(); audGallery(); })));
    root.append(audGateNote());
    root.append(audGrid([
      ['Stain concentration (H vs E)', d => audPlot(d, groups.map(g => ({ type: 'scattergl', mode: 'markers', name: g.name, x: g.rows.map(r => r.mean_h), y: g.rows.map(r => r.mean_e), customdata: g.rows.map(r => r.id), marker: { color: g.color, size: 6, opacity: 0.8 }, hovertemplate: g.name + '<br>H %{x:.3f}  E %{y:.3f}<extra></extra>' })),
        { xaxis: { title: 'mean hematoxylin (OD units)' }, yaxis: { title: 'mean eosin (OD units)' }, legend: { orientation: 'v', x: 1.02, y: 1, font: { size: 10 } }, margin: { r: 150, t: 8, b: 44, l: 54 } }, true)],
      ['Stain-model residual by class', d => audBox(d, groups, 'residual_rms', 'RMS OD residual', true)],
      ['Estimated H vector vs Ruifrok default (angle)', d => audBox(d, srcGroups, 'angle_h_deg', 'angle (degrees)', false)],
      ['Stained-pixel fraction by class', d => audBox(d, groups, 'tissue_fraction', 'fraction of pixels', false)]]));
    root.append(audNote('Reading: a collection with its own H–E cloud and a different H-vector angle was stained or scanned differently. A model or a texture threshold fitted on one collection will not transfer for that reason alone, before any biology.'));
    const sd = (a, m) => Math.sqrt(audMean(a.map(v => (v - m) ** 2)));
    root.append(h('h4', null, 'By collection'), tbl(['collection', 'patches', 'mean H', 'mean E', 'H angle (°)', 'residual (median)', 'stained px'], srcGroups.map(g => [g.name, g.rows.length, +audMean(g.rows.map(r => r.mean_h)).toFixed(3), +audMean(g.rows.map(r => r.mean_e)).toFixed(3),
      +audMean(g.rows.map(r => r.angle_h_deg)).toFixed(1), +audMedian(g.rows.map(r => r.residual_rms)).toFixed(4), +audMean(g.rows.map(r => r.tissue_fraction)).toFixed(3)]), 'stain by collection'));
    return root;
  },
  Quality() {
    const rows = audRows(), root = h('div', null), A = S.A.analysis, vs = ['accept', 'flag', 'exclude'];
    root.append(audGrid([
      ['Focus vs noise', d => audPlot(d, vs.map(v => ({ type: 'scattergl', mode: 'markers', name: v, x: rows.filter(r => r.verdict === v).map(r => r.focus_lapvar), y: rows.filter(r => r.verdict === v).map(r => Math.max(r.noise_sigma, 1e-4)), customdata: rows.filter(r => r.verdict === v).map(r => r.id),
        marker: { color: AUD_VERDICT_COL[v], size: 6, opacity: 0.8 } })), { xaxis: { title: 'focus: variance of Laplacian', type: 'log' }, yaxis: { title: 'noise σ (Haar; floored at 1e-4)', type: 'log' } }, true)],
      ['QC verdicts by collection', d => { const src = Object.keys(AUD_SRC_COL).filter(s => rows.some(r => r.source === s)); const q = (A.qc_rates || []).filter(r => r.scope === AS.qcScope);
        audPlot(d, vs.map(v => ({ type: 'bar', name: v, x: src, y: src.map(s => { const m = q.find(r => r.source === s && r.verdict === v); return m ? m.fraction : 0; }), marker: { color: AUD_VERDICT_COL[v] } })), { barmode: 'stack', yaxis: { title: 'fraction of patches', range: [0, 1] } }); }]]));
    root.append(h('div', { class: 'lrow' }, h('span', { class: 'tlabel' }, 'Verdict scope (right chart)'), audSeg(['pooled', 'within_source'], AS.qcScope, v => { AS.qcScope = v; })));
    root.append(audNote('Hard rule: more than 2% black pixels (letterbox padding, pen blobs) excludes a patch outright, because its stain and noise numbers are then not about tissue. Soft rule (v0): robust z of log focus (low is bad), log noise (high is bad) and log stain residual (high is bad); flag above 3.5, exclude above 6. Pooled verdicts show how far a collection sits from the rest; within-source verdicts show outliers inside each collection. Neither is validated against downstream failure yet.'));
    const bad = rows.filter(r => r.verdict !== 'accept').sort((a, b) => b.z_max - a.z_max).slice(0, 40);
    root.append(h('h4', null, 'Flagged and excluded patches (worst 40)'), bad.length ? tbl(['id', 'class', 'file', 'verdict', 'black px', 'worst z', 'reason'], bad.map(r => [r.id, r.class_name, r.file, r.verdict, pct(r.black_fraction, 0), +r.z_max.toFixed(2), r.reasons]), 'flagged patches') : audNote('No patch is flagged with the current filter.'));
    return root;
  },
  Texture() {
    const rows = audOk(), root = h('div', null), groups = audGroups(rows), A = S.A.analysis, pv = A.pca_var || [0, 0], byId = new Map(rows.map(r => [r.id, r]));
    const pts = (A.pca || []).filter(p => byId.has(p.id));
    root.append(h('div', { class: 'lrow' }, h('span', { class: 'tlabel' }, 'Color by'), audSeg(['class', 'source'], AS.colorBy, v => { AS.colorBy = v; buildAuditDocks(); audGallery(); }), h('span', { class: 'tlabel', style: 'margin-left:14px' }, 'Descriptor'),
      h('select', { onchange: e => { AS.selfeat = e.target.value; openTab(S.tab); } }, AUD_FEATS.map(f => h('option', { value: f[0], selected: f[0] === AS.selfeat }, f[1])))));
    root.append(audGrid([
      ['Texture descriptor space (PCA)', d => audPlot(d, groups.map(g => { const ids = new Set(g.rows.map(r => r.id)), q = pts.filter(p => ids.has(p.id)); return { type: 'scattergl', mode: 'markers', name: g.name, x: q.map(p => p.PC1), y: q.map(p => p.PC2), customdata: q.map(p => p.id), marker: { color: g.color, size: 6, opacity: 0.8 } }; }),
        { xaxis: { title: `PC1 (${(100 * pv[0]).toFixed(0)}% of variance)` }, yaxis: { title: `PC2 (${(100 * pv[1]).toFixed(0)}%)` }, legend: { orientation: 'v', x: 1.02, y: 1, font: { size: 10 } }, margin: { r: 150, t: 8, b: 44, l: 54 } }, true)],
      ['Descriptor by class', d => audBox(d, groups, AS.selfeat, (AUD_FEATS.find(f => f[0] === AS.selfeat) || [0, ''])[1], false)]]));
    root.append(audNote('Offsets are in pixels (these collections do not record pixel size), gray levels are fixed 0.05-OD bins of the luminance with a fixed white point, so values are comparable across patches. Overlapping clouds mean the descriptors alone do not separate the classes; separation by collection means the descriptor also encodes how the images were made.'));
    root.append(audGateNote());
    const cols = AUD_FEATS.map(f => f[0]);
    root.append(h('h4', null, 'Class means'), tbl(['class'].concat(AUD_FEATS.map(f => f[1])), S.A.classes.filter(c => AS.on[c.key]).map(c => { const g = audRowsAll().filter(r => r.class === c.key && r.verdict === 'accept'); return [c.name].concat(cols.map(k => +audMean(g.map(r => r[k])).toFixed(4))); }), 'texture class means'));
    return root;
  },
  Separability() {
    const A = S.A.analysis, root = h('div', null), sep = A.separability || [], ds = A.domain_shift || [], kinds = ['stain', 'texture', 'combined'], tasks = [...new Set(sep.map(r => r.task))];
    const kc = { stain: '#4da3ff', texture: '#f0a030', combined: '#5dd39e' };
    root.append(audNote('Descriptors are scored with a random forest. This asks whether stain and texture carry tissue-type signal and how much of it is specific to a collection. It is a property of the descriptors, not diagnostic performance: not a medical device, not for diagnosis.', true));
    root.append(h('div', { class: 'lrow' }, h('span', { class: 'tlabel' }, 'Descriptors (right chart)'), audSeg(kinds, AS.sepKind, v => { AS.sepKind = v; })));
    root.append(audGrid([
      ['Descriptor separability (5-fold)', d => { audPlot(d, kinds.map(k => { const q = tasks.map(t => sep.find(r => r.task === t && r.descriptors === k)); return { type: 'bar', name: k, x: tasks, y: q.map(r => r ? r.accuracy : null),
        error_y: { type: 'data', symmetric: false, array: q.map(r => r ? r.accuracy_hi95 - r.accuracy : 0), arrayminus: q.map(r => r ? r.accuracy - r.accuracy_lo95 : 0), thickness: 1.2, width: 3 }, marker: { color: kc[k] } }; }),
        { barmode: 'group', yaxis: { title: 'accuracy', range: [0, 1.02] }, shapes: [] }); }],
      ['Domain shift: in-source vs leave-one-source-out', d => { const srcs = [...new Set(ds.map(r => r.source))], st = ['in-source (5-fold)', 'leave-one-source-out'], col = ['#4da3ff', '#f0a030'];
        audPlot(d, st.map((s, i) => { const q = srcs.map(x => ds.find(r => r.source === x && r.setting === s && r.descriptors === AS.sepKind)); return { type: 'bar', name: s, x: srcs, y: q.map(r => r ? r.auroc : null),
          error_y: { type: 'data', symmetric: false, array: q.map(r => r ? r.auroc_hi95 - r.auroc : 0), arrayminus: q.map(r => r ? r.auroc - r.auroc_lo95 : 0), thickness: 1.2, width: 3 }, marker: { color: col[i] } }; }),
          { barmode: 'group', yaxis: { title: 'AUROC, tumor vs non-tumor', range: [0, 1.02] }, shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: '#8d96a6', dash: 'dash', width: 1 } }] }); }]]));
    root.append(audNote('Reading the right chart: blue is trained and tested inside one collection; orange is trained on the other three. A large drop means the descriptors learned collection-specific appearance. Augmented copies were excluded from sampling, but LC25000 patches are themselves augmentations of fewer originals, so its blue bar can still be optimistic.'));
    root.append(h('h4', null, 'Cross-validated accuracy'), tbl(['task', 'descriptors', 'images', 'accuracy', '95% CI low', '95% CI high', 'balanced acc.', 'majority baseline'], sep.map(r => [r.task, r.descriptors, r.n_images, +r.accuracy.toFixed(3), +r.accuracy_lo95.toFixed(3), +r.accuracy_hi95.toFixed(3), +r.balanced_accuracy.toFixed(3), +r.majority_baseline.toFixed(3)]), 'separability'));
    root.append(h('h4', null, 'Domain shift'), tbl(['descriptors', 'collection', 'setting', 'test images', 'AUROC', '95% CI low', '95% CI high', 'accuracy'], ds.map(r => [r.descriptors, r.source, r.setting, r.n_test, +r.auroc.toFixed(3), +r.auroc_lo95.toFixed(3), +r.auroc_hi95.toFixed(3), +r.accuracy.toFixed(3)]), 'domain shift'));
    return root;
  },
  Explanation() {
    const A = S.A.analysis, root = h('div', null), ex = A.explain;
    if (!ex) return audNote('No explanation in this audit: the optional shap package was missing or there were too few accepted patches. Install it with  pip install shap  and run the audit again.');
    const g = ex.groups.slice().sort((a, b) => a.share - b.share), stainShare = ex.groups.filter(x => !x.group.startsWith('GLCM')).reduce((s, x) => s + x.share, 0);
    root.append(audNote('TreeSHAP of a random forest that scores tumor against non-tumor patches from stain and texture descriptors. It explains what the model uses, which is how reliable or how shortcut-prone it is, and it is not a statement about the tissue: not a medical device, not for diagnosis.', true));
    root.append(h('div', { class: 'card-row' }, card('stain share', pct(stainShare, 0), 'of mean |SHAP| in stain descriptors'), card('texture share', pct(1 - stainShare, 0), 'in GLCM descriptors'), card('patches explained', S.A.rows.filter(r => r.xtop).length, 'accepted patches')));
    root.append(audGrid([['What drives the tumor score (TreeSHAP share)', d => audPlot(d, [{ type: 'bar', orientation: 'h', y: g.map(x => x.group), x: g.map(x => 100 * x.share), marker: { color: g.map(x => x.group.startsWith('GLCM') ? '#f0a030' : '#4da3ff') }, text: g.map(x => (100 * x.share).toFixed(0) + '%'), textposition: 'outside' }],
      { xaxis: { title: 'share of mean |SHAP| (%)' }, margin: { l: 170, t: 8, b: 44, r: 30 }, showlegend: false })]]));
    root.append(audNote('Blue bars are stain descriptors, orange bars texture. Select a patch in the gallery to see the three descriptors that moved its score most. A model that leans mostly on stain (color, concentration, residual) is probably recognizing the collection; check the Separability tab for how it behaves when a collection is left out. ' + ex.method + '.'));
    root.append(h('h4', null, 'Descriptor groups'), tbl(['group', 'share of mean |SHAP|'], ex.groups.map(x => [x.group, +(100 * x.share).toFixed(1) + ' %']), 'explanation groups'));
    return root;
  },
  Stability() {
    const A = S.A.analysis, root = h('div', null), st = A.stability || [], feats = [...new Set(st.map(r => r.feature))], fams = [...new Set(st.filter(r => r.offset_px === 2).map(r => r.perturbation))].filter(f => f !== 'all');
    const thr = st.length ? st[0].threshold : 0.75, line = { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: thr, y1: thr, line: { color: '#e5645f', dash: 'dash', width: 1.2 } }, fc = ['#4da3ff', '#f0a030', '#5dd39e', '#c77dff'];
    root.append(h('div', { class: 'card-row' }, card('descriptors certified', st.filter(r => r.offset_px === 2 && r.perturbation === 'all' && r.certified).length + ' / ' + feats.length, 'lower ICC bound ≥ ' + thr + ' · 2 px'), card('windows', st.length ? st[0].n_windows : '—', '128 × 128 px each')));
    root.append(audGrid([
      ['Stability (ICC lower bound), all perturbations', d => { const q = feats.map(f => st.find(r => r.feature === f && r.offset_px === 2 && r.perturbation === 'all')); audPlot(d, [{ type: 'bar', x: feats, y: q.map(r => r ? r.icc_lower95 : null), marker: { color: q.map(r => r && r.certified ? '#5dd39e' : '#e5645f') }, text: q.map(r => r ? r.icc_lower95.toFixed(2) : ''), textposition: 'outside' }], { yaxis: { title: 'ICC(2,1) lower 95% bound', range: [Math.min(0, ...q.map(r => r ? r.icc_lower95 : 0)) - 0.05, 1.08] }, shapes: [line] }); }],
      ['Stability by perturbation family', d => audPlot(d, fams.map((f, i) => { const q = feats.map(x => st.find(r => r.feature === x && r.offset_px === 2 && r.perturbation === f)); return { type: 'bar', name: f, x: feats, y: q.map(r => r ? r.icc_lower95 : null), marker: { color: fc[i % 4] } }; }), { barmode: 'group', yaxis: { title: 'ICC(2,1) lower 95% bound', range: [-0.05, 1.05] }, shapes: [line] })],
      ['Stability vs GLCM offset', d => audPlot(d, feats.map((f, i) => { const q = st.filter(r => r.feature === f && r.perturbation === 'all').sort((a, b) => a.offset_px - b.offset_px); return { type: 'scatter', mode: 'lines+markers', name: f, x: q.map(r => r.offset_px), y: q.map(r => r.icc_lower95), line: { width: 2 } }; }),
        { xaxis: { title: 'GLCM offset (pixels)', type: 'log', tickvals: [1, 2, 4, 8] }, yaxis: { title: 'ICC(2,1) lower 95% bound' }, shapes: [line] })]]));
    root.append(audNote('Method: the descriptor is computed on 128-pixel optical-density windows unperturbed and under noise (5%, 10% of sd), blur (σ 0.7 and 1.5 px), optical-density scaling (±10%) and offset (±0.05 sd). ICC(2,1) measures absolute agreement across those conditions; bootstrap over windows gives the lower bound. A descriptor that is not certified should not be used for a quantitative claim without recalibration.'));
    root.append(h('h4', null, 'ICC table'), tbl(['descriptor', 'offset (px)', 'perturbation', 'ICC(2,1)', 'lower 95%', 'certified'], st.map(r => [r.feature, r.offset_px, r.perturbation, +r.icc.toFixed(3), +r.icc_lower95.toFixed(3), r.certified ? 'yes' : 'no']), 'stability'));
    return root;
  },
  Interpolation() {
    const A = S.A.analysis, root = h('div', null), ib = A.interpolation || [], scens = [...new Set(ib.map(r => r.scenario))], feats = [...new Set(ib.map(r => r.feature))];
    if (!ib.length) return audNote('No interpolation experiment in this audit.');
    if (!scens.includes(AS.scen)) AS.scen = scens[0];
    root.append(h('div', { class: 'lrow' }, h('span', { class: 'tlabel' }, 'Neighboring sections'), audSeg(scens, AS.scen, v => { AS.scen = v; })));
    root.append(audGrid([['Texture bias under naive interpolation', d => audPlot(d, feats.flatMap((f, i) => { const q = ib.filter(r => r.scenario === AS.scen && r.feature === f).sort((a, b) => a.alpha - b.alpha), x = q.map(r => r.alpha), c = ['#4da3ff', '#f0a030', '#5dd39e', '#c77dff', '#e5645f', '#8d96a6'][i % 6];
      return [{ type: 'scatter', mode: 'lines', x: x.concat(x.slice().reverse()), y: q.map(r => 100 * r.relative_bias_hi95).concat(q.map(r => 100 * r.relative_bias_lo95).reverse()), fill: 'toself', fillcolor: c + '33', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
        { type: 'scatter', mode: 'lines+markers', name: f, x, y: q.map(r => 100 * r.relative_bias_mean), line: { color: c, width: 2 } }]; }),
      { xaxis: { title: 'distance fraction between the two sections (0 = observed plane)' }, yaxis: { title: 'relative bias of the feature (%)', zeroline: true, zerolinecolor: '#8d96a6' }, shapes: [] })]]));
    root.append(audNote('Two sections bracket a gap. The interpolated plane is (1−α)·A + α·B. Independent: B is a different field of view (spacing much larger than the correlation length). Shift k: B is A displaced by k pixels (registration error). The feature is compared with the mean of the two observed planes. This is why VeriSection masks interpolated voxels out of texture features (observed-support rule) instead of computing them on the filled volume.'));
    root.append(h('h4', null, 'Bias table'), tbl(['scenario', 'α', 'feature', 'pairs', 'bias mean (%)', 'low 95%', 'high 95%'], ib.filter(r => r.alpha > 0).map(r => [r.scenario, r.alpha, r.feature, r.n_pairs, +(100 * r.relative_bias_mean).toFixed(2), +(100 * r.relative_bias_lo95).toFixed(2), +(100 * r.relative_bias_hi95).toFixed(2)]), 'interpolation bias'));
    return root;
  },
  'Method & data'() {
    const root = h('div', null), A = S.A;
    root.append(h('h4', null, 'What was run'), h('ol', { class: 'note' }, [
      'Sampling: random patches per class, never augmented copies (files named aug_*). Fixed seed.',
      'Quality gate: patches with more than 2% black pixels, or a robust-z outlier in focus, noise or stain residual, are flagged or excluded and kept out of every statistic below the Quality tab.',
      'Stain: optical density with a fixed white point; two-stain Macenko-style vectors estimated per patch; concentrations by least squares; residual = RMS OD error. Tissue = summed OD above 0.45.',
      'Quality: focus = variance of the Laplacian; noise = Haar detail MAD; verdicts by robust z (flag > 3.5, exclude > 6).',
      'Texture: GLCM on luminance optical density, fixed 0.05-OD bins (32 levels), offsets 1, 2, 4, 8 px, four directions averaged; six Haralick-style features.',
      'Separability: random forest, 5-fold stratified; domain shift by leave-one-collection-out.',
      'Stability: ICC(2,1) with bootstrap lower bound across noise, blur and OD perturbations (package default threshold 0.75).',
      'Interpolation: linear intensity blending of real patches under independent and shifted neighbors.'].map(t => h('li', null, t))));
    root.append(h('h4', null, 'Limits'), h('ul', { class: 'note' }, (A.notes || []).concat(['Descriptor results characterize features; they are not diagnostic accuracy.', 'One sample per class, one seed: the intervals are bootstrap or t-intervals over patches, not over datasets.']).map(t => h('li', null, t))));
    root.append(h('h4', null, 'Source datasets'), h('ol', { class: 'note' }, AUD_REFS.map(t => h('li', null, t))));
    return root;
  },
};

/* ---------- running an audit from the browser (server mode) ---------- */
function auditDialog() {
  if (!window.VS_SERVER) { toast('Running an audit needs the server: run  verisection ui'); return; }
  const f = {}, inp = (k, el) => { f[k] = el; return el; };
  const form = h('div', { class: 'form' }, h('div', { class: 'fh' }, 'Dataset'),
    h('label', null, 'Folder of class sub-folders'), inp('root', h('input', { type: 'text', value: 'datasets', style: 'width:100%' })),
    h('label', null, 'Patches per class'), inp('per_class', h('input', { type: 'number', value: 16, min: 4, max: 120, step: 4, style: 'width:90px' })),
    h('label', null, 'Random seed'), inp('seed', h('input', { type: 'number', value: 0, min: 0, max: 100000, step: 1, style: 'width:90px' })),
    h('div', { class: 'fh' }, 'Analysis'),
    h('label', null, 'Include stability and interpolation'), inp('slow', h('input', { type: 'checkbox', checked: true })), h('span'), h('span', { class: 'note' }, 'adds the ICC certification and the interpolation experiment (about a minute)'));
  modal('New image audit', h('div', null, h('div', { class: 'note' }, 'Reads independent 2D patches (never files named aug_*), measures stain, quality and texture, and fills the Image audit workspace. About 1 s per patch.'), form),
    [{ label: 'Cancel' }, { label: 'Run audit', primary: true, act: () => { const p = {}; for (const k in f) p[k] = f[k].type === 'checkbox' ? f[k].checked : f[k].value; startAudit(p); } }], { width: '640px' });
}
async function startAudit(params) {
  const r = await fetch('/api/audit', { method: 'POST', body: JSON.stringify(params) });
  if (!r.ok) { toast('Could not start: ' + (await r.json().catch(() => ({}))).error); return; }
  const dlg = progressDialog();
  const timer = setInterval(async () => {
    try {
      const st = await (await fetch('/api/status')).json(); dlg.set(st.progress, st.message, st.elapsed);
      if (st.state === 'done') { clearInterval(timer); const pj = await (await fetch('/api/result')).json(); dlg.close(); loadAudit(pj); }
      else if (st.state === 'error') { clearInterval(timer); dlg.close(); modal('Audit failed', h('pre', { class: 'm' }, st.error)); }
    } catch (e) { clearInterval(timer); dlg.close(); toast('Lost connection to the server'); }
  }, 700);
}
