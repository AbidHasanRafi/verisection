/* ===== app: menus, toolbar, dialogs, server interaction, shortcuts, init ===== */
const MODES = ['Class composite', 'Uncertainty MIP', 'Provenance (gaps highlighted)', 'Image volume (DVR)', 'Isosurface of class probability', 'Error map (simulation)', 'Ground-truth classes (simulation)'];
const TFS = {
  'Epithelium emphasis (glands, lumen as cavity)': names => ({ on: names.map(() => true), w: names.map(n => /lumen/i.test(n) ? 0 : /stroma/i.test(n) ? 0.03 : /immune/i.test(n) ? 3.0 : 1.9) }),
  'All classes': names => ({ on: names.map(() => true), w: names.map(() => 1.0) }),
  'Epithelium only': names => ({ on: names.map(n => /epith/i.test(n)), w: names.map(() => 1.6) }),
  'Immune cells only': names => ({ on: names.map(n => /immune/i.test(n)), w: names.map(() => 3.0) }),
  'Stroma only': names => ({ on: names.map(n => /stroma/i.test(n)), w: names.map(() => 0.5) }),
  'Lumen (cavities) only': names => ({ on: names.map(n => /lumen/i.test(n)), w: names.map(() => 2.0) }),
};
function applyTF(name) { const t = TFS[name](S.P.classNames); S.classOn = t.on; S.rw = t.w; buildLayersPanel(); refreshAll(); }
function setR(k, v) { if (k === 'mode' && v >= 5 && S.P && !S.P.hasTruth) { toast('Truth-based modes need a simulation with ground truth'); return; } S.R[k] = v; if (k === 'mode' || k === 'cmap') GL3D.setLUT(); if (k === 'cls') GL3D.setProbClass(); GL3D.dirty = true; renderMenusOpenState(); syncToolbar(); }
function toggleView(k) { S.view[k] = !S.view[k]; refreshAll(); }
function setTool(t) { S.tool = t; S.pendingMeasure = []; syncToolbar(); setStatus({ cursor: 'Cursor: click or drag in a slice view', measure: 'Distance: click two points', roi: 'ROI: drag a rectangle in a slice view' }[t]); renderAllSlices(); }
function jumpPlane(dir, kind) {
  const [Z] = S.dims; let z = S.cur[0];
  for (let i = 0; i < Z; i++) { z += dir; if (z < 0 || z >= Z) return; if (planeProv(z) === kind) { S.cur[0] = z; refreshAll(); return; } }
}
function jumpUncertain() {
  const [Z, Y, X] = S.dims, n = Y * X; let best = 0, bz = 0;
  for (let z = 0; z < Z; z++) { let s = 0; for (let q = 0; q < n; q += 5) s += S.vol.unc[z * n + q]; if (s > best) { best = s; bz = z; } }
  S.cur[0] = bz; refreshAll();
}
function fullscreen() { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen(); }
function setTheme(t) { S.theme = t; document.body.classList.toggle('dark', t === 'dark'); document.body.classList.toggle('light', t === 'light'); if (S.P) { refreshAll(); openTab(S.tab); } }
function dockHeight(px) { $('#dockB').style.height = px + 'px'; setTimeout(() => { fitCanvases(); $$('#tabpane .plot').forEach(d => { try { Plotly.Plots.resize(d); } catch (e) { /* ignore */ } }); }, 30); }
function resetLayout() { setLayout('quad'); $('#dockB').style.height = ''; setTimeout(fitCanvases, 30); $('#dockL').classList.remove('hidden'); $('#dockR').classList.remove('hidden'); $('#dockB').classList.remove('hidden'); }

/* ---------- downloads and data export (image export lives in export.js) ---------- */
function download(name, blob) {
  if (window.VS_TEST_NODL) { document.body.dataset.downloaded = name + ' ' + blob.size; return; }   // tests: record instead of clicking (a headless browser stalls on downloads)
  const a = h('a', { href: URL.createObjectURL(blob), download: name }); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 60000); }
function exportJSON() { const a = S.ws === 'audit' && S.A ? S.A.analysis : S.P.analysis; download(S.ws === 'audit' ? 'verisection_audit.json' : 'verisection_analysis.json', new Blob([JSON.stringify(a, null, 2)], { type: 'application/json' })); }
function exportCSV() {
  const txt = tablesCsv(S.tab); if (!txt) { toast('This tab has no tables'); return; }
  download('verisection_' + S.tab.replace(/\W+/g, '_') + '.csv', new Blob([txt], { type: 'text/csv' }));
}
function exportTXT() {
  if (S.ws === 'audit') { const rows = audRowsAll(), L = ['VeriSection image audit (research use only; independent 2D patches, not serial sections)', '', 'patches ' + rows.length, 'collections ' + new Set(rows.map(r => r.source)).size, 'flagged or excluded ' + rows.filter(r => r.verdict !== 'accept').length]; download('verisection_audit_summary.txt', new Blob([L.join('\n')], { type: 'text/plain' })); return; }
  const A = S.P.analysis, s = A.summary || {}, L = ['VeriSection summary (research use only; synthetic phantom if simulation)', '', 'dims ' + (s.dims || []).join('x')];
  if (A.segmentation) L.push('accuracy ' + A.segmentation.all.accuracy, 'balanced accuracy ' + A.segmentation.all.balanced_accuracy);
  if (A.calibration) L.push('ECE ' + A.calibration.all.ece, 'Brier ' + A.calibration.all.brier);
  if (A.registration) L.push('mean TRE um ' + A.registration.mean_tre_um);
  if (A.measurements) A.measurements.rows.forEach(r => L.push(`${r.class}: ${r.estimate} [${r.lo90}, ${r.hi90}] truth ${r.truth} verdict ${r.verdict}`));
  download('verisection_summary.txt', new Blob([L.join('\n')], { type: 'text/plain' }));
}
function copyCursor() { const [z, y, x] = S.cur, k = IDX(z, y, x); const t = `z=${z} y=${y} x=${x} prov=${PROV_NAME[S.vol.prov[k]]} class=${S.P.classNames[S.vol.labels[k] - 1] || '-'}`; (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.resolve()).then(() => toast('Copied: ' + t)); }

/* ---------- modals ---------- */
function modal(title, body, buttons, opts = {}) {
  const bg = h('div', { class: 'modal-bg', onmousedown: e => { if (e.target === bg && !opts.sticky) close(); } });
  const close = () => bg.remove();
  const box = h('div', { class: 'modal', style: opts.width ? 'width:' + opts.width : '' }, h('h3', null, title), h('div', { class: 'mb' }, body),
    h('div', { class: 'mf' }, (buttons || [{ label: 'Close', primary: true }]).map(b => h('button', { class: 'btn' + (b.primary ? ' primary' : ''), onclick: () => { if (!b.act || b.act() !== false) close(); } }, b.label))));
  bg.append(box); $('#modal-root').append(bg); return { close, box };
}
function simDialog(extended = false) {
  if (!window.VS_SERVER) { toast('Simulations need the server: run  verisection ui'); return; }
  const f = {}; const inp = (k, el) => { f[k] = el; return el; };
  const sel = (k, opts, v) => inp(k, h('select', null, opts.map(o => h('option', { value: o[0], selected: o[0] === v }, o[1]))));
  const num = (k, v, min, max, step) => inp(k, h('input', { type: 'number', value: v, min, max, step, style: 'width:90px' }));
  const form = h('div', { class: 'form' },
    h('div', { class: 'fh' }, 'Phantom'),
    h('label', null, 'Tissue model'), sel('preset', [['tissue', 'Tissue: glands, lumen, stroma, immune cells'], ['hard', 'Random-field blobs, hard (rotation, deformation, stain shifts)'], ['easy', 'Random-field blobs, easy']], 'tissue'),
    h('label', null, 'Random seed'), num('seed', 0, 0, 100000, 1),
    h('div', { class: 'fh' }, 'Acquisition'),
    h('label', null, 'Number of sections'), num('n_sections', 20, 6, 40, 1),
    h('label', null, 'Section spacing'), sel('spacing_planes', [[2, '4 µm (contiguous)'], [3, '6 µm'], [4, '8 µm'], [6, '12 µm'], [8, '16 µm']], 3),
    h('div', { class: 'fh' }, 'Reconstruction'),
    h('label', null, 'Registration'), sel('method', [['translation', 'Translation (fast)'], ['rigid', 'Rigid (rotation search)'], ['nonrigid', 'Non-rigid (slow)']], 'translation'),
    h('label', null, 'Registration ensemble size'), num('n_reg', 2, 1, 6, 1),
    h('label', null, 'Segmentation ensemble size'), num('n_seg', 3, 1, 6, 1),
    h('label', null, 'Gap interpolation'), sel('interp_mode', [['kriging', 'Indicator kriging (uncertainty-aware)'], ['linear', 'Linear'], ['nearest', 'Nearest section']], 'kriging'),
    h('label', null, 'Registration-spread multiplier'), num('sigma_scale', 2.2, 1, 10, 0.1),
    h('div', { class: 'fh' }, 'Analysis'),
    h('label', null, 'Extended analysis'), inp('extended', h('input', { type: 'checkbox', checked: extended })), h('span'), h('span', { class: 'note' }, 'adds leave-one-section-out conformal coverage (slower)'));
  modal('New simulation', h('div', null, h('div', { class: 'note' }, 'Synthetic phantom with known truth: results check the software, not real tissue. A 20-section tissue run takes about 30–60 s.'), form), [
    { label: 'Cancel' },
    { label: 'Run simulation', primary: true, act: () => { const p = {}; for (const k in f) p[k] = f[k].type === 'checkbox' ? f[k].checked : f[k].value; startSim(p); } }], { width: '640px' });
}
function progressDialog() {
  const bar = h('i'), msg = h('div', null, 'starting…'), el = h('div', { class: 'tlabel' }, '');
  const m = modal('Running simulation', h('div', null, msg, h('div', { class: 'prog' }, bar), el), [], { sticky: true });
  return { set(p, t, s) { bar.style.width = (p * 100).toFixed(0) + '%'; msg.textContent = t; el.textContent = s.toFixed(0) + ' s elapsed'; }, close: m.close };
}
async function startSim(params) {
  const r = await fetch('/api/simulate', { method: 'POST', body: JSON.stringify(params) });
  if (!r.ok) { toast('Could not start: ' + (await r.text())); return; }
  const dlg = progressDialog();
  const timer = setInterval(async () => {
    try {
      const st = await (await fetch('/api/status')).json(); dlg.set(st.progress, st.message, st.elapsed);
      if (st.state === 'done') { clearInterval(timer); const pj = await (await fetch('/api/result')).json(); dlg.close(); hideWelcome(); loadPayload(pj); }
      else if (st.state === 'error') { clearInterval(timer); dlg.close(); modal('Simulation failed', h('pre', { class: 'm' }, st.error)); }
    } catch (e) { clearInterval(timer); dlg.close(); toast('Lost connection to the server'); }
  }, 600);
}
function openVolDialog() {
  const inp = h('input', { type: 'text', style: 'width:100%', placeholder: 'C:\\path\\to\\volume.npz' });
  modal('Open exported volume', h('div', null, h('div', { class: 'note' }, 'Path of a volume.npz written by “verisection run-stack” (read on this machine only).'), inp), [{ label: 'Cancel' }, {
    label: 'Open', primary: true, act: () => { fetch('/api/open', { method: 'POST', body: JSON.stringify({ path: inp.value.trim() }) }).then(async r => { const j = await r.json(); if (!r.ok) modal('Cannot open', h('div', null, j.error)); else { hideWelcome(); loadPayload(j); } }); } }]);
}
function shortcutsDialog() {
  const rows = [['C / D / R', 'Cursor / Distance / ROI tool'], ['1 2 3 4', 'Select axial / coronal / sagittal / 3D'], ['↑ ↓  or  PgUp PgDn', 'Scroll slice in the active view'], ['wheel / Ctrl+wheel', 'Scroll slice / zoom (2D)'], ['drag / right-drag', '3D orbit / pan'], ['[  ]', 'Previous / next observed plane'],
    ['U  P  G  F', 'Toggle uncertainty / provenance / segmentation / truth-contour overlays'], ['M', 'Cycle 3D render mode'], ['A', 'Auto-rotate 3D'], ['L', 'Cycle layout'], ['Ctrl+N', 'New simulation'], ['Ctrl+Shift+A', 'New image audit'], ['Esc / ← →', 'Close tour / previous, next tour step'], ['Ctrl+Shift+E', 'Export image (300–1200 dpi)'], ['F11', 'Full screen'], ['?', 'This list']];
  modal('Keyboard shortcuts', h('table', { class: 't' }, rows.map(r => h('tr', null, h('td', null, h('kbd', null, r[0])), h('td', null, r[1])))));
}
function limitsDialog() {
  modal('Intended use and limitations', h('div', null, h('p', null, h('b', null, 'Research use only.'), ' This is not a medical device and must not be used for diagnosis, prognosis or treatment decisions.'),
    h('ul', null, ['Simulations use a synthetic phantom with known truth; they check the software, not real tissue.', 'Interpolated voxels are model inference with quantified (but imperfect) uncertainty; they are never observations.', 'Measurement intervals can under-cover (54% at nominal 90% on the hard simulator); verdicts reflect that only partly.', 'Registration here is 2D and sequential; content drift between distant sections is not corrected.', 'Segmentation is a simple pixel classifier standing in for a trained model.', 'Texture features across sections are valid only where registration and spacing allow (see the Texture tab).', 'Image audit works on independent 2D patches: it is not a 3D reconstruction, and descriptor separability is not diagnostic accuracy.', 'Public patch collections differ in stain, scanner and processing; results on one collection need not transfer to another (see Separability).'].map(t => h('li', null, t))),
    h('p', null, 'See the project README and the paper for measured results and caveats.')), null, { width: '620px' });
}
function aboutDialog() { modal('About VeriSection', h('div', null, h('p', null, h('b', null, 'VeriSection Workstation'), ' — uncertainty-aware 3D reconstruction of serial sections with explicit provenance.'), h('p', null, 'Linked orthogonal views, WebGL2 ray-marched volume rendering, and a multi-tab numerical analysis dock.'), h('p', { class: 'note' }, 'Research prototype. Not a medical device.'))); }
function quickDialog() { modal('Quick start', h('div', null, h('p', { class: 'note' }, 'Two workspaces share one shell: Reconstruction (serial sections to a 3D volume with uncertainty) and Image audit (independent 2D patches: stain, quality, texture). Switch with the toolbar control or View ▸ Workspace. Help ▸ Guided tour walks through every region.'), h('ol', null, ['File ▸ New simulation… to generate and reconstruct a tissue phantom.', 'Click in any slice to move the crosshair; scroll to change slices. Orange hatching marks interpolated voxels.', 'Volume ▸ Render mode switches the 3D view (class composite, uncertainty, provenance, isosurface…).', 'Layers toggles overlays; Tools measures distances and ROI statistics.', 'The Analysis dock has tabs for segmentation, calibration, uncertainty, registration, texture, measurements and the planner.', 'Image audit: File ▸ New image audit… reads a folder of class sub-folders; click a thumbnail to inspect a patch; tabs show stain, quality, texture, descriptor separability, stability and interpolation bias.'].map(t => h('li', null, t))))); }
function prefsDialog() {
  const th = h('select', { onchange: e => setTheme(e.target.value) }, ['dark', 'light'].map(t => h('option', { value: t, selected: t === S.theme }, t)));
  modal('Preferences', h('div', { class: 'form' }, h('label', null, 'Theme'), th, h('label', null, '3D quality'), h('select', { onchange: e => setR('steps', +e.target.value) }, [96, 192, 320, 512].map(v => h('option', { value: v, selected: v === S.R.steps }, v + ' steps')))));
}

/* ---------- welcome ---------- */
function showWelcome() {
  if ($('#welcome') || S.ws === 'audit') return;
  const card2 = (title, text, btns) => h('div', { class: 'wcard' }, h('h3', null, title), h('p', null, text), h('div', { class: 'row' }, btns));
  const w = h('div', { id: 'welcome', style: 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:var(--panel);border:1px solid var(--line);border-radius:var(--r);z-index:20;overflow:auto' },
    h('div', { style: 'max-width:760px;padding:16px' }, h('h2', { style: 'margin:0 0 4px;text-align:center' }, 'VeriSection Workstation'),
      h('div', { class: 'note', style: 'margin:0 auto 16px;text-align:center' }, 'Uncertainty-aware 3D histopathology. Choose a workspace.'),
      h('div', { class: 'welcome-grid' },
        card2('Reconstruction', 'Serial sections to a 3D volume. Every voxel is labeled observed, interpolated or synthetic and carries its uncertainty. Linked slice views, WebGL2 volume rendering, calibration and measurement analysis.',
          [h('button', { class: 'btn primary', onclick: () => simDialog() }, 'New simulation…'), h('button', { class: 'btn', onclick: openVolDialog }, 'Open exported volume…')]),
        card2('Image audit', 'Independent 2D patches from public collections. Stain and optical-density diagnostics, image quality, physical-scale texture, descriptor stability and the effect of interpolation. No 3D claim.',
          [h('button', { class: 'btn primary', onclick: () => (S.A ? setWorkspace('audit') : auditDialog()) }, S.A ? 'Open image audit' : 'New image audit…')])),
      h('div', { style: 'display:flex;gap:10px;justify-content:center' }, h('button', { class: 'btn', onclick: () => tour() }, 'Guided tour'), h('button', { class: 'btn', onclick: quickDialog }, 'Quick start'), h('button', { class: 'btn', onclick: limitsDialog }, 'Intended use'))));
  $('#center').style.position = 'relative'; $('#center').append(w);
}
function hideWelcome() { const w = $('#welcome'); if (w) w.remove(); }

/* ---------- menus ---------- */
let MENU_OPEN = -1;
const radio = (label, cur, set) => ({ label, radio: cur, act: set });
function layerMenu(title, key, extra) {
  const L = S.L[key];
  return { label: title, sub: [{ label: 'Visible', chk: () => L.on, act: () => { L.on = !L.on; buildLayersPanel(); refreshAll(); } }, '-', { label: 'Opacity', sub: [0.25, 0.5, 0.75, 1].map(o => ({ label: (o * 100) + '%', radio: () => Math.abs(L.op - o) < 0.01, act: () => { L.op = o; buildLayersPanel(); refreshAll(); } })) }].concat(extra || []) };
}
function menuDefs() {
  const names = () => (S.P ? S.P.classNames : []);
  const tabs = () => $$('#tabs .tab').map(t => t.textContent);
  return [
    { label: 'File', items: [
      { label: 'New simulation…', key: 'Ctrl+N', act: () => simDialog(), dis: () => !window.VS_SERVER },
      { label: 'New image audit…', key: 'Ctrl+Shift+A', act: () => auditDialog(), dis: () => !window.VS_SERVER },
      { label: 'Open exported volume…', act: openVolDialog, dis: () => !window.VS_SERVER }, '-',
      { label: 'Export image…', key: 'Ctrl+Shift+E', act: () => exportDialog(), dis: () => !(S.P || S.A) },
      { label: 'Quick export', dis: () => !(S.P || S.A), sub: [
        { label: 'Entire window · 300 dpi', act: () => runExport('window', 300, {}) }, { label: 'Entire window · 600 dpi', act: () => runExport('window', 600, {}) }, '-',
        { label: 'Active view figure · 600 dpi', act: () => runExport('active-image', 600, {}) }, { label: 'Active view figure · 1200 dpi', act: () => runExport('active-image', 1200, {}) }, '-',
        { label: 'Current analysis tab · 300 dpi', act: () => runExport('tab', 300, {}) }, { label: 'Current tab charts (ZIP) · 600 dpi', act: () => runExport('tab-charts', 600, { svg: true }) }, { label: 'Every analysis tab (ZIP) · 300 dpi', act: () => runExport('all-tabs', 300, {}) }] },
      { label: 'Export', dis: () => !(S.P || S.A), sub: [{ label: 'Analysis (JSON)', act: exportJSON }, { label: 'Current tab tables (CSV)', act: exportCSV }, { label: 'Summary (TXT)', act: exportTXT }] }, '-',
      { label: 'Print report…', key: 'Ctrl+P', act: () => window.print() }] },
    { label: 'Edit', items: [{ label: 'Copy cursor info', act: copyCursor, dis: () => !S.P }, '-', { label: 'Clear measurements', act: () => { S.measures = []; S.pendingMeasure = []; refreshAll(); }, dis: () => !S.P }, { label: 'Clear ROI', act: () => { S.roi = null; refreshAll(); }, dis: () => !S.P }, '-', { label: 'Preferences…', act: prefsDialog }] },
    { label: 'View', items: [
      { label: 'Workspace', sub: [radio('Reconstruction', () => S.ws !== 'audit', () => setWorkspace('volume')), radio('Image audit', () => S.ws === 'audit', () => setWorkspace('audit'))] }, '-',
      { label: 'Layout', sub: [['quad', 'Quad: 3 slice views + 3D'], ['focus3d', '3D focus + 3 slice views'], ['mprrow', 'Slice row + large 3D'], ['mpr', 'Slice views only'], ['single', 'Single active view']].map(([k, t]) => radio(t, () => S.layout === k, () => setLayout(k))) },
      { label: 'Active view', sub: [['axial', 'Axial (x–y)', '1'], ['coronal', 'Coronal (x–z)', '2'], ['sagittal', 'Sagittal (y–z)', '3'], ['3d', '3D', '4']].map(([k, t, key]) => Object.assign(radio(t, () => S.active === k, () => { S.active = k; markActive(); if (S.layout === 'single') setLayout('single'); }), { key })) }, '-',
      { label: 'Crosshair', chk: () => S.view.cross, act: () => toggleView('cross') }, { label: 'Scale bars', chk: () => S.view.scale, act: () => toggleView('scale') }, { label: 'Axis labels', chk: () => S.view.labels, act: () => toggleView('labels') },
      { label: 'Provenance ruler (z axis)', chk: () => S.view.ruler, act: () => toggleView('ruler') }, { label: 'Smooth 2D interpolation', chk: () => S.view.smooth, act: () => toggleView('smooth') }, '-',
      { label: 'Zoom (2D views)', sub: [{ label: 'Reset all', act: () => { S.zoom = { axial: 1, coronal: 1, sagittal: 1 }; S.pan = { axial: [0, 0], coronal: [0, 0], sagittal: [0, 0] }; renderAllSlices(); } }, { label: 'Zoom in active', act: () => { if (S.zoom[S.active]) { S.zoom[S.active] *= 1.25; renderAllSlices(); } } }, { label: 'Zoom out active', act: () => { if (S.zoom[S.active]) { S.zoom[S.active] /= 1.25; renderAllSlices(); } } }] },
      { label: 'Go to', sub: [{ label: 'Center of volume', act: () => { S.cur = S.dims.map(d => d >> 1); refreshAll(); } }, { label: 'First plane', act: () => { S.cur[0] = 0; refreshAll(); } }, { label: 'Last plane', act: () => { S.cur[0] = S.dims[0] - 1; refreshAll(); } }, '-', { label: 'Next observed plane', key: ']', act: () => jumpPlane(1, 1) }, { label: 'Previous observed plane', key: '[', act: () => jumpPlane(-1, 1) }, { label: 'Next interpolated plane', act: () => jumpPlane(1, 2) }, { label: 'Most uncertain plane', act: jumpUncertain }] }, '-',
      { label: 'Theme', sub: [radio('Dark', () => S.theme === 'dark', () => setTheme('dark')), radio('Light', () => S.theme === 'light', () => setTheme('light'))] }, { label: 'Full screen', key: 'F11', act: fullscreen }] },
    { label: 'Layers', dis: () => !S.P || S.ws === 'audit', items: [
      layerMenu('Image (registered sections)', 'base'),
      layerMenu('Segmentation', 'labels', [{ label: 'Classes', sub: names().map((n, c) => ({ label: n, chk: () => S.classOn[c], act: () => { S.classOn[c] = !S.classOn[c]; buildLayersPanel(); refreshAll(); } })) }]),
      layerMenu('Uncertainty', 'unc', [{ label: 'Colormap', sub: ['magma', 'inferno', 'viridis', 'turbo', 'gray'].map(c => radio(c, () => S.L.unc.cmap === c, () => { S.L.unc.cmap = c; buildLayersPanel(); refreshAll(); })) }]),
      layerMenu('Provenance hatch', 'prov'), S.P && S.P.hasTruth ? layerMenu('Truth contour', 'truth') : null, S.P && S.P.hasTruth ? layerMenu('Error map', 'err') : null, layerMenu('Registration variance', 'varReg'), '-',
      { label: 'Segmentation overlay only', act: () => { Object.keys(S.L).forEach(k => S.L[k].on = (k === 'base' || k === 'labels')); buildLayersPanel(); refreshAll(); } },
      { label: 'Uncertainty overlay only', act: () => { Object.keys(S.L).forEach(k => S.L[k].on = (k === 'base' || k === 'unc')); buildLayersPanel(); refreshAll(); } },
      { label: 'Reset layers', act: () => { Object.assign(S.L.base, { on: true, op: 1 }); Object.assign(S.L.labels, { on: true, op: .5 }); S.L.unc.on = false; S.L.prov.on = true; S.L.truth.on = false; S.L.err.on = false; S.L.varReg.on = false; buildLayersPanel(); refreshAll(); } }] },
    { label: 'Volume', dis: () => !S.P || S.ws === 'audit', items: [
      { label: 'Render mode', sub: MODES.map((m, i) => Object.assign(radio(m, () => S.R.mode === i, () => setR('mode', i)), { key: i === 0 ? 'M cycles' : '' })) },
      { label: 'Transfer function', sub: Object.keys(TFS).map(k => ({ label: k, act: () => { applyTF(k); setR('mode', 0); } })) },
      { label: 'Isosurface class', sub: names().map((n, c) => radio(n, () => S.R.cls === c, () => { setR('cls', c); setR('mode', 4); })) },
      { label: 'Isosurface threshold', sub: [0.3, 0.5, 0.7, 0.9].map(v => radio(String(v), () => Math.abs(S.R.iso - v) < 0.01, () => setR('iso', v))) },
      { label: 'Uncertainty MIP threshold', sub: [0.1, 0.25, 0.4, 0.6].map(v => radio(String(v), () => Math.abs(S.R.thr - v) < 0.01, () => setR('thr', v))) },
      { label: 'Density', sub: [0.5, 1, 2, 4].map(v => radio(v + '×', () => S.R.density === v, () => setR('density', v))) },
      { label: 'Lighting', sub: [['Off', 0], ['Medium', 0.5], ['Strong', 0.85]].map(([t, v]) => radio(t, () => Math.abs(S.R.light - v) < 0.01, () => setR('light', v))) },
      { label: 'Quality', sub: [['Low (96 steps)', 96], ['Medium (192)', 192], ['High (320)', 320], ['Ultra (512)', 512]].map(([t, v]) => radio(t, () => S.R.steps === v, () => setR('steps', v))) }, '-',
      { label: 'Cut-away at crosshair', chk: () => S.R.cut, act: () => setR('cut', !S.R.cut) },
      { label: 'Clip box', sub: [{ label: 'Reset', act: () => setR('clip', [0, 0, 0, 1, 1, 1]) }, { label: 'Front half in z', act: () => setR('clip', [0, 0, 0, 1, 1, .5]) }, { label: 'Back half in z', act: () => setR('clip', [0, 0, .5, 1, 1, 1]) }, { label: 'Left half in x', act: () => setR('clip', [0, 0, 0, .5, 1, 1]) }, { label: 'Upper half in y', act: () => setR('clip', [0, 0, 0, 1, .5, 1]) }] },
      { label: 'Observed section slabs', chk: () => S.R.slabs, act: () => setR('slabs', !S.R.slabs) }, { label: 'Slice frames at crosshair', chk: () => S.R.frames, act: () => setR('frames', !S.R.frames) }, { label: 'Auto-rotate', key: 'A', chk: () => S.R.auto, act: () => setR('auto', !S.R.auto) }, '-',
      { label: 'Camera', sub: [['Isometric', 'iso'], ['Front', 'front'], ['Back', 'back'], ['Left', 'left'], ['Right', 'right'], ['Top', 'top'], ['Bottom', 'bottom']].map(([t, k]) => ({ label: t, act: () => GL3D.setView(k) })).concat([{ label: 'Reset camera', act: () => GL3D.resetCamera() }]) },
      { label: 'Uncertainty colormap', sub: ['magma', 'inferno', 'viridis', 'turbo', 'gray'].map(c => radio(c, () => S.R.cmap === c, () => setR('cmap', c))) }] },
    { label: 'Analysis', dis: () => !(S.P || S.A), items: tabs().map((t, i) => ({ label: t, key: i < 9 ? 'Alt+' + (i + 1) : '', radio: () => S.tab === t, act: () => openTab(t) })).concat(['-',
      { label: 'Extended analysis (conformal)…', act: () => simDialog(true), dis: () => !window.VS_SERVER },
      { label: 'Dock height', sub: [['Small (25%)', 0.25], ['Medium (36%, default)', 0.36], ['Large (50%)', 0.5], ['Maximise (72%)', 0.72]].map(([t, f]) => ({ label: t, act: () => dockHeight(Math.round(innerHeight * f)) })) },
      { label: 'Show analysis dock', chk: () => !$('#dockB').classList.contains('hidden'), act: () => { $('#dockB').classList.toggle('hidden'); $('#splitter').classList.toggle('hidden'); setTimeout(fitCanvases, 30); } },
      { label: 'Export current tab as image…', act: () => exportDialog({ scope: 'tab' }) },
      { label: 'Export current tab tables (CSV)', act: exportCSV }]) },
    { label: 'Tools', dis: () => !S.P || S.ws === 'audit', items: [
      radio('Cursor (crosshair)', () => S.tool === 'cursor', () => setTool('cursor')), Object.assign(radio('Distance (µm)', () => S.tool === 'measure', () => setTool('measure')), { key: 'D' }), Object.assign(radio('ROI statistics', () => S.tool === 'roi', () => setTool('roi')), { key: 'R' }), '-',
      { label: 'Clear measurements', act: () => { S.measures = []; S.pendingMeasure = []; refreshAll(); } }, { label: 'Clear ROI', act: () => { S.roi = null; refreshAll(); } }, '-',
      { label: 'Export image…', key: 'Ctrl+Shift+E', act: () => exportDialog() }] },
    { label: 'Window', items: [{ label: 'Reset layout', act: resetLayout }, '-',
      { label: 'Layers panel', chk: () => !$('#dockL').classList.contains('hidden'), act: () => { $('#dockL').classList.toggle('hidden'); setTimeout(fitCanvases, 30); } },
      { label: 'Inspector panel', chk: () => !$('#dockR').classList.contains('hidden'), act: () => { $('#dockR').classList.toggle('hidden'); setTimeout(fitCanvases, 30); } }] },
    { label: 'Help', items: [{ label: 'Guided tour', act: () => tour() }, { label: 'Quick start', act: quickDialog }, { label: 'Keyboard shortcuts', key: '?', act: shortcutsDialog }, { label: 'Intended use and limitations', act: limitsDialog }, '-', { label: 'About VeriSection', act: aboutDialog }] },
  ];
}
function buildPop(items) {
  const pop = h('div', { class: 'menu-pop' });
  items.filter(Boolean).forEach(it => {
    if (it === '-') { pop.append(h('div', { class: 'msep' })); return; }
    const dis = it.dis && it.dis();
    const mark = it.chk ? (it.chk() ? '✓' : '') : it.radio ? (it.radio() ? '●' : '') : '';
    const row = h('div', { class: 'mi' + (dis ? ' dis' : '') }, h('span', { class: 'chk' }, mark), it.label, it.sub ? h('span', { class: 'arrow' }, '▸') : (it.key ? h('span', { class: 'key' }, it.key) : null));
    if (it.sub && !dis) {
      let sp = null;
      row.addEventListener('mouseenter', () => { $$('.menu-pop .menu-pop', pop).forEach(x => x.remove()); sp = buildPop(it.sub); row.append(sp); });
    } else {
      row.addEventListener('mouseenter', () => { $$(':scope > .mi > .menu-pop', pop).forEach(x => x.remove()); });
      if (!dis && it.act) row.addEventListener('click', e => { e.stopPropagation(); closeMenus(); it.act(); });
    }
    pop.append(row);
  });
  return pop;
}
function closeMenus() { MENU_OPEN = -1; $$('.menu-pop').forEach(x => x.remove()); $$('.menu-top').forEach(x => x.classList.remove('open')); }
function openMenu(i) {
  closeMenus(); MENU_OPEN = i; const defs = menuDefs(), top = $$('.menu-top')[i], d = defs[i];
  top.classList.add('open'); const pop = buildPop(d.items.filter(Boolean)); pop.style.left = top.offsetLeft + 'px'; $('#menubar').append(pop);
}
function renderMenusOpenState() { if (MENU_OPEN >= 0) openMenu(MENU_OPEN); }
function buildMenubar() {
  const mb = $('#menubar'); mb.innerHTML = '';
  menuDefs().forEach((d, i) => {
    const t = h('div', { class: 'menu-top' }, d.label);
    t.addEventListener('mousedown', e => { e.stopPropagation(); if (MENU_OPEN === i) closeMenus(); else openMenu(i); });
    t.addEventListener('mouseenter', () => { if (MENU_OPEN >= 0 && MENU_OPEN !== i) openMenu(i); });
    mb.append(t);
  });
  mb.append(h('div', { style: 'margin-left:auto;padding:0 12px;line-height:28px;color:var(--dim)', id: 'mb-title' }, ''));
  document.addEventListener('mousedown', e => { if (!e.target.closest('#menubar')) closeMenus(); });
}

/* ---------- toolbar ---------- */
function buildToolbar() {
  // groups: [simulation] [tools] [view: layout · 3D mode · class · quality] [overlays] ··· [export · theme]
  const tb = $('#toolbar'); tb.innerHTML = '';
  const btn = (id, label, title, fn, cls = '') => h('button', { class: cls, id, title, 'aria-label': title, onclick: fn }, label);
  const sel = (id, opts, change, label) => h('select', { id, title: label, 'aria-label': label, onchange: e => change(e.target.value) }, opts.map(o => h('option', { value: o[0] }, o[1])));
  const grp = (...kids) => h('div', { class: 'tgroup vol' }, ...kids), audg = (...kids) => h('div', { class: 'tgroup aud' }, ...kids), always = (...kids) => h('div', { class: 'tgroup' }, ...kids);
  const toggleOverlay = key => { S.L[key].on = !S.L[key].on; buildLayersPanel(); refreshAll(); syncToolbar(); };
  tb.append(
    always(h('div', { class: 'seg', role: 'group', 'aria-label': 'Workspace' }, btn('tb-ws-volume', h('span', null, '◫', h('span', { class: 'p2' }, ' Reconstruction')), 'Workspace: serial-section reconstruction', () => setWorkspace('volume')), btn('tb-ws-audit', h('span', null, '▦', h('span', { class: 'p2' }, ' Image audit')), 'Workspace: audit of independent 2D patches', () => setWorkspace('audit')))),
    audg(btn('tb-aud', '▶ New image audit', 'File ▸ New image audit', () => auditDialog(), 'tb primary')),
    grp(btn('tb-sim', '▶ New simulation', 'File ▸ New simulation (Ctrl+N)', () => simDialog(), 'tb primary')),
    grp(h('div', { class: 'seg', role: 'group', 'aria-label': 'Tool' },
      btn('tb-cursor', '✛ Cursor', 'Cursor (C)', () => setTool('cursor')), btn('tb-measure', '↔ Distance', 'Distance (D)', () => setTool('measure')), btn('tb-roi', '▭ ROI', 'ROI statistics (R)', () => setTool('roi')))),
    grp(h('span', { class: 'tcap p1' }, 'Layout'), sel('tb-layout', [['quad', 'Quad'], ['focus3d', '3D focus'], ['mprrow', 'Slices + 3D'], ['mpr', 'Slices only'], ['single', 'Single']], v => setLayout(v), 'Layout'),
      h('span', { class: 'tcap p1' }, '3D'), sel('tb-mode', MODES.map((m, i) => [i, m]), v => setR('mode', +v), '3D render mode'),
      h('span', { class: 'tcap p1' }, 'Class'), h('select', { id: 'tb-cls', title: 'Isosurface class', 'aria-label': 'Isosurface class', onchange: e => setR('cls', +e.target.value) }),
      h('span', { class: 'tcap p1 p2' }, 'Quality'), h('span', { class: 'p2' }, sel('tb-q', [[96, 'Low'], [192, 'Medium'], [320, 'High'], [512, 'Ultra']], v => setR('steps', +v), '3D quality'))),
    grp(h('span', { class: 'tcap p1' }, 'Overlays'), h('div', { class: 'seg', role: 'group', 'aria-label': 'Overlays' },
      btn('tb-uncl', 'Unc', 'Uncertainty overlay (U)', () => toggleOverlay('unc')), btn('tb-provl', 'Prov', 'Provenance hatch (P)', () => toggleOverlay('prov')),
      btn('tb-segl', 'Seg', 'Segmentation overlay (G)', () => toggleOverlay('labels')), btn('tb-truth', 'Truth', 'Truth contour (F)', () => toggleOverlay('truth')))),
    h('div', { class: 'tsp' }),
    always(btn('tb-tour', h('span', null, '?', h('span', { class: 'p2' }, ' Tour')), 'Guided tour of the interface (Help ▸ Guided tour)', () => tour(), 'tb'), btn('tb-shot', h('span', null, '⎙', h('span', { class: 'p2' }, ' Export')), 'Export image at 300–1200 dpi (Ctrl+Shift+E)', () => { if (S.P || S.A) exportDialog(); }, 'tb'),
      btn('tb-theme', '◐', 'Toggle dark / light theme', () => setTheme(S.theme === 'dark' ? 'light' : 'dark'), 'tb')));
}
function syncToolbar() {
  [['volume', S.ws !== 'audit'], ['audit', S.ws === 'audit']].forEach(([k, on]) => { const b = $('#tb-ws-' + k); if (b) b.classList.toggle('on', on); });
  ['cursor', 'measure', 'roi'].forEach(t => { const b = $('#tb-' + t); if (b) b.classList.toggle('on', S.tool === t); });
  [['tb-uncl', S.L.unc.on], ['tb-provl', S.L.prov.on], ['tb-segl', S.L.labels.on], ['tb-truth', S.L.truth.on]].forEach(([id, on]) => { const b = $('#' + id); if (b) b.classList.toggle('on', !!on); });
  const m = $('#tb-mode'); if (m) m.value = S.R.mode; const q = $('#tb-q'); if (q) q.value = S.R.steps; const l = $('#tb-layout'); if (l) l.value = S.layout;
  const c = $('#tb-cls'); if (c && S.P) { if (c.options.length !== S.P.classNames.length) c.replaceChildren(...S.P.classNames.map((n, i) => h('option', { value: i }, n))); c.value = S.R.cls; }
  const tr = $('#tb-truth'); if (tr) tr.classList.toggle('hidden', !(S.P && S.P.hasTruth));
  const t = $('#mb-title'); if (t) t.textContent = S.ws === 'audit' && S.A ? S.A.title : (S.P ? S.P.title : '');
}

/* ---------- keyboard ---------- */
document.addEventListener('keydown', e => {
  if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
  const k = e.key;
  if (k === 'Escape') { closeMenus(); $$('.modal-bg').forEach(x => x.remove()); endTour(); return; }
  if ($('#tour')) { if (k === 'ArrowRight' || k === 'Enter') { e.preventDefault(); tourStep(1); } else if (k === 'ArrowLeft') { e.preventDefault(); tourStep(-1); } return; }
  if (e.ctrlKey && e.shiftKey && k.toLowerCase() === 'a') { e.preventDefault(); auditDialog(); return; }
  if (e.ctrlKey && k.toLowerCase() === 'n') { e.preventDefault(); simDialog(); return; }
  if (e.ctrlKey && e.shiftKey && k.toLowerCase() === 'e') { e.preventDefault(); if (S.P || S.A) exportDialog(); return; }
  if (e.altKey && /^[1-9]$/.test(k)) { const t = $$('#tabs .tab')[+k - 1]; if (t) openTab(t.textContent); return; }
  if (!S.P || S.ws === 'audit') { if (k === '?') shortcutsDialog(); return; }
  const act = S.active, scroll = d => { if (act === '3d') return; setSliceIndex(act, sliceIndex(act) + d); refreshAll(); };
  const map = {
    c: () => setTool('cursor'), d: () => setTool('measure'), r: () => setTool('roi'), a: () => setR('auto', !S.R.auto), m: () => setR('mode', (S.R.mode + 1) % MODES.length),
    u: () => { S.L.unc.on = !S.L.unc.on; buildLayersPanel(); refreshAll(); syncToolbar(); }, p: () => { S.L.prov.on = !S.L.prov.on; buildLayersPanel(); refreshAll(); syncToolbar(); },
    g: () => { S.L.labels.on = !S.L.labels.on; buildLayersPanel(); refreshAll(); syncToolbar(); }, f: () => { if (S.P.hasTruth) { S.L.truth.on = !S.L.truth.on; buildLayersPanel(); refreshAll(); syncToolbar(); } },
    l: () => { const order = ['quad', 'focus3d', 'mprrow', 'mpr', 'single']; setLayout(order[(order.indexOf(S.layout) + 1) % order.length]); syncToolbar(); },
    '1': () => { S.active = 'axial'; markActive(); }, '2': () => { S.active = 'coronal'; markActive(); }, '3': () => { S.active = 'sagittal'; markActive(); }, '4': () => { S.active = '3d'; markActive(); },
    ArrowUp: () => scroll(-1), ArrowDown: () => scroll(1), PageUp: () => scroll(-5), PageDown: () => scroll(5), '[': () => jumpPlane(-1, 1), ']': () => jumpPlane(1, 1), '?': shortcutsDialog,
  };
  const f = map[k.toLowerCase()] || map[k]; if (f) { e.preventDefault(); f(); }
});

/* ---------- init ---------- */
function init() {
  window.addEventListener('error', e => { const m = (e.message || '') + ' @' + (e.lineno || ''); console.error(m); setStatus('Error: ' + m); const d = $('#errbox') || h('div', { id: 'errbox', style: 'position:fixed;right:8px;bottom:30px;max-width:520px;background:#5a1a1a;color:#fff;padding:6px 10px;font:11px monospace;z-index:999;white-space:pre-wrap' }); d.textContent = m; document.body.append(d); });
  buildMenubar(); buildToolbar(); buildStatus(); buildViewports(); setLayout('quad'); buildLayersPanel(); buildInspector(); buildTabs();
  GL3D.init(); syncToolbar(); markActive();
  const sp = $('#splitter'); let sd = null;
  sp.addEventListener('mousedown', e => { sd = { y: e.clientY, h: $('#dockB').getBoundingClientRect().height }; e.preventDefault(); });
  window.addEventListener('mousemove', e => { if (sd) { dockHeight(clamp(sd.h - (e.clientY - sd.y), 80, innerHeight * 0.8)); } });
  window.addEventListener('mouseup', () => { sd = null; });
  if (window.VS_PAYLOAD) { loadPayload(window.VS_PAYLOAD); syncToolbar(); }
  if (window.VS_AUDIT) { S.A = null; loadAudit(window.VS_AUDIT); } else if (!window.VS_PAYLOAD) { showWelcome(); }
  window.VS = { S, loadPayload, loadAudit, setWorkspace, tour, openTab, setLayout, setR, GL3D, audit: layoutAudit };
  applyHash();
}
function applyHash() {
  if (S.ws === 'audit' || (location.hash && /ws=audit/.test(location.hash))) { applyAuditHash(); return; }
  // #tab=Calibration&layout=focus3d&mode=4&cls=2&cur=30,48,48&layers=unc,truth&menu=2&sub=1&theme=light&dock=520
  if (!S.P || !location.hash) return;
  const q = new URLSearchParams(location.hash.slice(1));
  if (q.get('theme')) setTheme(q.get('theme'));
  if (q.get('layout')) setLayout(q.get('layout'));
  if (q.get('dock')) dockHeight(+q.get('dock'));
  if (q.get('cur')) { S.cur = q.get('cur').split(',').map(Number); }
  if (q.get('layers')) {
    const want = q.get('layers').split(',');
    Object.keys(S.L).forEach(k => { S.L[k].on = want.includes(k) || k === 'base'; });
    buildLayersPanel();
  }
  if (q.get('tf')) applyTF(Object.keys(TFS)[+q.get('tf')]);
  if (q.get('cls')) S.R.cls = +q.get('cls');
  if (q.get('mode')) setR('mode', +q.get('mode'));
  if (q.get('cut')) S.R.cut = true;
  if (q.get('az')) { GL3D.cam.az = +q.get('az'); GL3D.cam.el = +(q.get('el') || 0.5); }
  if (q.get('tab')) openTab(q.get('tab'));
  refreshAll(); syncToolbar();
  if (q.get('menu')) {
    openMenu(+q.get('menu'));
    if (q.get('sub')) { const r = $$('.menu-pop > .mi')[+q.get('sub')]; if (r) r.dispatchEvent(new MouseEvent('mouseenter')); }
  }
  if (q.get('dialog') === 'sim') simDialog();
  if (q.get('dialog') === 'export') exportDialog(q.get('scope') ? { scope: q.get('scope'), dpi: +(q.get('dpi') || 300) } : {});
  if (q.get('roi')) { const a = q.get('roi').split(','); S.dragRoi = { plane: a[0], i0: +a[1], j0: +a[2], i1: +a[3], j1: +a[4] }; finishRoi(); }
  if (q.get('meas')) {
    const a = q.get('meas').split(',').map(Number), px = S.P.pixelUm, dx = (a[5] - a[2]) * px, dy = (a[4] - a[1]) * px, dz = S.P.zUm[a[3]] - S.P.zUm[a[0]];
    S.measures.push({ p: [{ z: a[0], y: a[1], x: a[2] }, { z: a[3], y: a[4], x: a[5] }], dx, dy, dz, d: Math.hypot(dx, dy, dz) }); refreshAll();
  }
  // test hooks: #hold=PORT keeps the load event pending until the page reports completion to a local server
  const hp = q.get('hold');
  window.VS_RELEASE = () => { if (hp) fetch(`http://127.0.0.1:${hp}/done`, { mode: 'no-cors' }).catch(() => {}); };
  if (hp) document.body.append(h('img', { src: `http://127.0.0.1:${hp}/hold`, style: 'display:none', alt: '' }));
  if (q.get('autoexport')) { window.VS_TEST_NODL = true; const [sc, dp] = q.get('autoexport').split(','); runExport(sc, +dp || 300, {}); }   // #autoexport=active-image,300 runs the dialog's Export-button path
  if (q.get('selftest') === 'export') {   // #selftest=export : real exports, facts written to a hidden element (used by the tests)
    exportSelfTest(q.get('spec') === 'full'
      ? [['window', 300, {}, null], ['active-image', 300, {}, null, 'axial'], ['active-image', 300, {}, null, '3d'], ['active', 300, {}, null], ['tab', 300, {}, null], ['tab-charts', 300, { svg: true }, null], ['window', 600, {}, [0, 0, 1400, 420]]]
      : [['window', 300, {}, null], ['active-image', 300, {}, null, 'axial'], ['active', 300, {}, null, 'coronal'], ['tab', 300, {}, null], ['tab-charts', 300, { svg: true }, null], ['window', 600, {}, [0, 0, 1400, 420]]]).catch(e => { document.body.append(h('pre', { id: 'selftest', style: 'display:none' }, JSON.stringify({ fatal: String(e && e.stack || e) }))); })
      .finally(() => window.VS_RELEASE());
  }
  if (q.has('audit')) setTimeout(writeAudit, 900);   // #audit=1 writes the layout audit into a hidden element (used by the tests)
}

/* ---------- layout audit: panels must not touch, nothing may clip, no sliders ---------- */
function layoutAudit() {
  const MIN_GAP = 6, vis = e => e.getClientRects().length > 0;
  const surf = $$('#toolbar,#dockL,#dockR,#dockB,#statusbar,.vp').filter(vis).map(e => ({ name: e.id || e.className.split(' ')[0], r: e.getBoundingClientRect() }));
  const dist = (a, b) => { const dx = Math.max(a.left - b.right, b.left - a.right), dy = Math.max(a.top - b.bottom, b.top - a.bottom); return (dx > 0 && dy > 0) ? Math.hypot(dx, dy) : Math.max(dx, dy); };
  let minGap = Infinity; const tooClose = [];
  for (let i = 0; i < surf.length; i++) for (let j = i + 1; j < surf.length; j++) {
    const d = dist(surf[i].r, surf[j].r); if (d > 40) continue;   // only neighbors
    minGap = Math.min(minGap, d); if (d < MIN_GAP) tooClose.push([surf[i].name, surf[j].name, +d.toFixed(1)]);
  }
  const clipped = $$('#toolbar,#statusbar,.vhead,#tabs').filter(vis).filter(e => e.id !== 'tabs' && e.scrollWidth > e.clientWidth + 1).map(e => e.id || e.className.split(' ')[0]);
  const offscreen = surf.filter(s => s.r.right > innerWidth + 1 || s.r.bottom > innerHeight + 1 || s.r.left < -1 || s.r.top < -1).map(s => s.name);
  return { viewport: [innerWidth, innerHeight], surfaces: surf.length, minGapPx: minGap === Infinity ? null : +minGap.toFixed(1), tooClose, clipped, offscreen, sliders: $$('input').filter(i => i.type === 'range').length };
}
function writeAudit() { const p = h('pre', { id: 'audit', style: 'display:none' }, JSON.stringify(layoutAudit())); const o = $('#audit'); if (o) o.remove(); document.body.append(p); }
init();
