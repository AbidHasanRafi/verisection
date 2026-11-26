/* ===== analysis dock: multi-tab numerical analysis with tables and charts ===== */
const merge = (a, b) => { const o = Object.assign({}, a); for (const k in b) o[k] = (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k] && typeof a[k] === 'object') ? merge(a[k], b[k]) : b[k]; return o; };
function plotTheme() {
  const dark = document.body.classList.contains('dark'), grid = dark ? '#2a313b' : '#e1e6ee', axis = dark ? '#3a4350' : '#c9d0db';
  return { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: dark ? '#d8dde6' : '#1e2530', size: 11, family: 'Segoe UI,system-ui,sans-serif' },
    margin: { l: 54, r: 18, t: 8, b: 44 }, xaxis: { gridcolor: grid, zeroline: false, linecolor: axis, automargin: true }, yaxis: { gridcolor: grid, zeroline: false, linecolor: axis, automargin: true },
    legend: { orientation: 'h', y: -0.3, font: { size: 11 } }, hoverlabel: { font: { size: 11 } } };
}
// charts export at 3x for figures; the modebar appears on hover only
function plot(div, traces, layout) { Plotly.react(div, traces, merge(plotTheme(), layout || {}), { displaylogo: false, responsive: true, displayModeBar: 'hover', toImageButtonOptions: { format: 'png', scale: 3, filename: 'verisection_chart' } }); }
// what each chart shows and how to read it (keyed by chart title)
const HELP = {
  'Class composition (true vs reconstructed)': 'Share of sampled voxels in each class. Bars that differ from the truth mean the reconstruction over- or under-counts a class.',
  'Provenance of sampled voxels': 'How much of the volume is observed, interpolated or synthetic. Interpolated voxels are model inference.',
  'Confusion matrix (row-normalized: true → predicted)': 'Each row sums to 100%: how voxels of a true class were labeled. Off-diagonal cells are confusions.',
  'Per-class Dice / precision / recall': 'Overlap quality per class (1 is perfect). Low recall = missed voxels; low precision = false additions.',
  'Reliability diagram (top-label)': 'If confidence were calibrated, points would sit on the dashed diagonal. Below it = over-confident.',
  'Confidence histogram': 'How many voxels fall at each confidence level (log scale), by provenance.',
  'Entropy and error along depth': 'Predictive entropy (left axis) and error rate (right axis) per z-plane. Blue bands = observed planes, orange = interpolated.',
  'Sparsification curve': 'Error of the remaining voxels as the most uncertain ones are removed. Close to the oracle means uncertainty ranks errors well.',
  'Entropy distribution by provenance': 'Histogram of entropy for observed vs interpolated voxels.',
  'Variance attribution (MAP-class probability)': 'Registration and segmentation contributions are reported separately; they are not summed into one number.',
  'Expected entropy vs mutual information': 'Expected entropy = class overlap within a sample; mutual information = disagreement across registration × segmentation samples.',
  'Error and entropy by gap distance': 'Error and entropy as a function of distance to the nearest observed slab.',
  'Target registration error and ensemble spread per section': 'True error (simulation) against the ensemble spread. The spread is a ranking signal and under-states error unless recalibrated.',
  'Post-alignment NCC and failed samples': 'Normalized cross-correlation after alignment; bars count ensemble members flagged as failed.',
  'Focus (variance of Laplacian) and noise σ per section': 'Per-section focus and noise; outliers are flagged by robust z-scores.',
  'Intensity continuity to the previous section': 'Distribution shift relative to the previous section, colored by QC verdict.',
  'Class volume fractions with 90% intervals': 'Estimate with 90% interval against simulation truth. Real studies have no truth line.',
  'Estimation sd vs number of sections': 'Expected standard deviation of the class fraction as sections are added.',
  'Recommended additional section positions': 'Where an extra section would reduce the estimation variance the most.',
  'Provenance along depth': 'One bar per z-plane: blue observed, orange interpolated, gray unsampled.',
  'Distance to the nearest observed slab': 'Gap distance per z-plane; uncertainty tends to grow with it.',
  contrast: 'GLCM contrast per class: truth vs observed planes only vs naive interpolation.',
  entropy: 'GLCM entropy per class: truth vs observed planes only vs naive interpolation.',
  homogeneity: 'GLCM homogeneity per class: truth vs observed planes only vs naive interpolation.',
};
const head = t => [h('h4', null, t), HELP[t] ? h('p', { class: 'help' }, HELP[t]) : null];
const later = [];
function plotDiv(title) { const d = h('div', { class: 'plot' }); return { d, title }; }
function tbl(headers, rows, name) {
  S.tables[S.tab] = S.tables[S.tab] || []; S.tables[S.tab].push({ name: name || S.tab, headers, rows: rows.map(r => r.map(c => (c && c.nodeType) ? c.textContent : c)) });
  return h('div', { class: 'tblwrap' }, h('table', { class: 't' }, h('tr', null, headers.map(x => h('th', null, x))),
    rows.map(r => h('tr', null, r.map(c => h('td', { class: typeof c === 'number' ? 'n' : '' }, typeof c === 'number' ? fmt(c, 4) : c))))));
}
const card = (l, v, sub) => h('div', { class: 'card' }, h('div', { class: 'v' }, v), h('div', { class: 'l' }, l), sub ? h('div', { class: 'tlabel' }, sub) : null);
const chip = (txt, kind) => h('span', { class: 'chip ' + (kind || '') }, txt);
function seg(options, current, onpick) {
  return h('div', { class: 'seg' }, options.map(o => h('button', { class: o === current ? 'on' : '', onclick: () => onpick(o) }, o)));
}
const CLS = () => S.P.classNames;
const COL = () => S.P.classColors;

function buildTabs() {
  if (S.ws === 'audit') { buildAuditTabs(); return; }
  const bar = $('#tabs'); bar.innerHTML = '';
  if (!S.P) { $('#tabpane').replaceChildren(h('div', { class: 'note' }, 'The analysis dock fills in after a volume is loaded.')); return; }
  const A = S.P.analysis || {}, names = ['Summary'];
  if (A.segmentation) names.push('Segmentation');
  if (A.calibration) names.push('Calibration');
  if (A.uncertainty) names.push('Uncertainty');
  if (A.registration) names.push('Registration');
  if (A.qc) names.push('Sections QC');
  if (A.texture) names.push('Texture');
  if (A.measurements) names.push('Measurements');
  if (A.planner) names.push('Planner');
  if (A.conformal) names.push('Conformal');
  if (A.provenance) names.push('Provenance');
  if (A.manifest) names.push('Manifest');
  if (!names.includes(S.tab)) S.tab = 'Summary';
  names.forEach(n => bar.append(h('div', { class: 'tab' + (n === S.tab ? ' on' : ''), onclick: () => openTab(n) }, n)));
  openTab(S.tab);
}
function openTab(name) {
  if (S.ws === 'audit') { openAuditTab(name); return; }
  S.tab = name; S.tables[name] = [];
  $$('#tabs .tab').forEach(t => t.classList.toggle('on', t.textContent === name));
  const pane = $('#tabpane'); pane.innerHTML = '';
  const fn = TAB[name]; later.length = 0;
  try { pane.append(fn ? fn(S.P.analysis || {}) : h('div', null, 'n/a')); } catch (e) { console.error(e); pane.append(h('div', { class: 'note warn' }, 'This tab could not be drawn: ' + e.message)); }
  requestAnimationFrame(() => later.forEach(f => { try { f(); } catch (e) { console.error(e); } }));
}
function twoPlots(pairs) { const grid = h('div', { class: 'grid2' }); pairs.forEach(([title, fn]) => { const { d } = plotDiv(); grid.append(h('div', null, head(title), d)); later.push(() => fn(d)); }); return grid; }

const TAB = {
  Summary(A) {
    const s = A.summary || {}, pf = s.provenance_fraction || {};
    const root = h('div', null);
    root.append(h('div', { class: 'card-row' },
      card('voxels', (s.dims || []).join(' × ')), card('sections', fmt(s.n_sections, 0), `spacing ${fmt(s.section_spacing_um, 1)} µm · thickness ${fmt(s.section_thickness_um, 1)} µm`),
      card('observed', pct(pf.observed, 0), 'of sampled voxels'), card('interpolated', pct(pf.interpolated, 0), 'model inference'),
      card('mean entropy', fmt(s.mean_entropy_bits, 3) + ' bits'),
      A.segmentation ? card('accuracy vs truth', pct(A.segmentation.all.accuracy, 1), 'balanced ' + pct(A.segmentation.all.balanced_accuracy, 1)) : null,
      A.calibration ? card('ECE', fmt(A.calibration.all.ece, 3), 'lower is better') : null,
      A.registration ? card('mean TRE', fmt(A.registration.mean_tre_um, 1) + ' µm', A.registration.method) : null));
    root.append(h('div', { class: 'note' }, A.disclaimer || ''));
    if (s.true_class_fraction) {
      root.append(twoPlots([['Class composition (true vs reconstructed)', d => plot(d, [
        { type: 'bar', name: 'truth', x: CLS(), y: s.true_class_fraction, marker: { color: '#7f8ea3' } },
        { type: 'bar', name: 'reconstruction', x: CLS(), y: s.predicted_class_fraction, marker: { color: COL().map(c => c) } }], { barmode: 'group', yaxis: { title: 'fraction of sampled voxels' } })],
      ['Provenance of sampled voxels', d => plot(d, [{ type: 'bar', x: ['observed', 'interpolated', 'synthetic'], y: [pf.observed, pf.interpolated, pf.synthetic], marker: { color: ['#2b7bba', '#f0a030', '#b04ab0'] } }], { yaxis: { title: 'fraction', range: [0, 1] } })]]));
    }
    const p = s.params || {};
    if (Object.keys(p).length) root.append(h('h4', null, 'Run parameters'), tbl(['parameter', 'value'], Object.entries(p).map(([k, v]) => [k, String(v)])));
    return root;
  },
  Segmentation(A) {
    const root = h('div', null); let key = S.segKey || 'all';
    const draw = () => {
      root.innerHTML = '';
      const keys = Object.keys(A.segmentation);
      root.append(seg(keys, key, k => { key = S.segKey = k; draw(); }));
      const g = A.segmentation[key], cm = g.confusion, C = cm.length;
      const norm = cm.map(r => { const s = r.reduce((a, b) => a + b, 0) || 1; return r.map(v => v / s); });
      root.append(h('div', { class: 'card-row' }, card('accuracy', pct(g.accuracy, 1)), card('balanced accuracy', pct(g.balanced_accuracy, 1)), card('sampled voxels', fmt(cm.flat().reduce((a, b) => a + b, 0), 0))));
      const grid = h('div', { class: 'grid2' }), p1 = plotDiv(), p2 = plotDiv();
      grid.append(h('div', null, head('Confusion matrix (row-normalized: true → predicted)'), p1.d), h('div', null, head('Per-class Dice / precision / recall'), p2.d));
      root.append(grid);
      root.append(h('h4', null, 'Per-class metrics'), tbl(['class', 'support', 'predicted', 'precision', 'recall', 'Dice', 'IoU'], g.per_class.map(r => [CLS()[r.class], r.support, r.predicted, r.precision, r.recall, r.dice, r.iou]), 'segmentation ' + key));
      requestAnimationFrame(() => {
        plot(p1.d, [{ type: 'heatmap', z: norm, x: CLS(), y: CLS(), colorscale: 'Blues', zmin: 0, zmax: 1, text: norm.map(r => r.map(v => (v * 100).toFixed(0) + '%')), texttemplate: '%{text}', hovertemplate: 'true %{y} → %{x}: %{z:.2f}<extra></extra>' }],
          { xaxis: { title: 'predicted' }, yaxis: { title: 'true', autorange: 'reversed' } });
        plot(p2.d, [['dice', '#4da3ff'], ['precision', '#f0a030'], ['recall', '#5dd39e']].map(([m, c]) => ({ type: 'bar', name: m, x: CLS(), y: g.per_class.map(r => r[m]), marker: { color: c } })), { barmode: 'group', yaxis: { range: [0, 1] } });
      });
    };
    draw(); return root;
  },
  Calibration(A) {
    const cal = A.calibration, keys = Object.keys(cal), colors = { all: '#4da3ff', observed: '#2b7bba', interpolated: '#f0a030' };
    const root = h('div', null);
    root.append(h('div', { class: 'note' }, 'Reliability diagram: if confidence were calibrated, points would lie on the diagonal. Below the diagonal = over-confident.'));
    root.append(twoPlots([['Reliability diagram (top-label)', d => plot(d, [{ x: [0, 1], y: [0, 1], mode: 'lines', line: { dash: 'dash', color: '#8d96a6' }, name: 'perfect' }].concat(keys.map(k => ({ x: cal[k].bins.map(b => b.conf), y: cal[k].bins.map(b => b.acc), mode: 'lines+markers', name: k, line: { color: colors[k] }, marker: { size: cal[k].bins.map(b => 5 + Math.min(12, Math.sqrt(b.n) / 12)) } }))), { xaxis: { title: 'mean confidence', range: [0, 1] }, yaxis: { title: 'accuracy', range: [0, 1] } })],
      ['Confidence histogram', d => plot(d, keys.map(k => ({ type: 'bar', name: k, x: cal[k].bins.map(b => b.conf), y: cal[k].bins.map(b => b.n), marker: { color: colors[k] }, opacity: 0.8 })), { barmode: 'overlay', xaxis: { title: 'confidence' }, yaxis: { title: 'voxels', type: 'log' } })]]));
    root.append(h('h4', null, 'Calibration metrics by provenance'), tbl(['subset', 'voxels', 'accuracy', 'mean confidence', 'ECE', 'Brier', 'NLL'], keys.map(k => [k, cal[k].n, cal[k].accuracy, cal[k].mean_confidence, cal[k].ece, cal[k].brier, cal[k].nll]), 'calibration'));
    return root;
  },
  Uncertainty(A) {
    const U = A.uncertainty, root = h('div', null), pz = U.per_plane, z = pz.map(r => r.z_um), prv = pz.map(r => r.provenance);
    const shapes = pz.map((r, i) => ({ type: 'rect', xref: 'x', yref: 'paper', x0: z[i] - (S.P.dzUm / 2), x1: z[i] + (S.P.dzUm / 2), y0: 0, y1: 1, fillcolor: r.provenance === 1 ? 'rgba(43,123,186,0.16)' : 'rgba(240,160,48,0.10)', line: { width: 0 }, layer: 'below' }));
    const va = U.variance_attribution;
    root.append(h('div', { class: 'card-row' }, card('AUSE', fmt(U.sparsification.ause, 3), 'lower is better'), card('Spearman(unc., error)', fmt(U.sparsification.spearman, 3)),
      card('mean expected entropy', fmt(va.mean_expected_entropy_bits, 3) + ' bits'), card('mean mutual information', fmt(va.mean_mutual_information_bits, 3) + ' bits')));
    root.append(twoPlots([
      ['Entropy and error along depth', d => plot(d, [{ x: z, y: pz.map(r => r.entropy), name: 'entropy (bits)', line: { color: '#f58518' } }, { x: z, y: pz.map(r => r.error), name: 'error rate', yaxis: 'y2', line: { color: '#e5645f' } }], { shapes, xaxis: { title: 'z (µm)' }, yaxis: { title: 'entropy (bits)' }, yaxis2: { overlaying: 'y', side: 'right', title: 'error rate', rangemode: 'tozero', showgrid: false } })],
      ['Sparsification curve', d => plot(d, [{ x: U.sparsification.fraction_removed, y: U.sparsification.by_uncertainty, name: 'by uncertainty', line: { color: '#4da3ff' } }, { x: U.sparsification.fraction_removed, y: U.sparsification.oracle, name: 'oracle', line: { color: '#5dd39e', dash: 'dot' } }, { x: U.sparsification.fraction_removed, y: U.sparsification.random, name: 'random', line: { color: '#8d96a6', dash: 'dash' } }], { xaxis: { title: 'fraction removed' }, yaxis: { title: 'error of remaining voxels' } })],
      ['Entropy distribution by provenance', d => plot(d, Object.entries(U.entropy_histogram).map(([k, v]) => ({ type: 'bar', name: k, x: v.edges.slice(0, -1).map((e, i) => (e + v.edges[i + 1]) / 2), y: v.counts, marker: { color: k === 'observed' ? '#2b7bba' : '#f0a030' }, opacity: 0.8 })), { barmode: 'overlay', xaxis: { title: 'entropy (bits)' }, yaxis: { title: 'voxels' } })],
      ['Variance attribution (MAP-class probability)', d => plot(d, [{ type: 'bar', x: ['total', 'registration (total effect)', 'segmentation (total effect)', 'registration (first order)', 'segmentation (first order)', 'interaction'], y: [va.total, va.registration_total_effect, va.segmentation_total_effect, va.registration_first_order, va.segmentation_first_order, va.interaction], marker: { color: ['#8d96a6', '#4da3ff', '#f0a030', '#4da3ff', '#f0a030', '#b04ab0'] } }], { yaxis: { title: 'mean variance' }, xaxis: { tickangle: -25 } })],
      ['Expected entropy vs mutual information', d => plot(d, [{ x: z, y: pz.map(r => r.expected_entropy), name: 'expected entropy', stackgroup: 'a', line: { color: '#f58518' } }, { x: z, y: pz.map(r => r.mutual_information), name: 'mutual information', stackgroup: 'a', line: { color: '#b04ab0' } }], { shapes, xaxis: { title: 'z (µm)' }, yaxis: { title: 'bits' } })],
      ['Error and entropy by gap distance', d => plot(d, [{ type: 'bar', x: U.by_gap.map(r => r.gap_um), y: U.by_gap.map(r => r.error), name: 'error rate', marker: { color: '#e5645f' } }, { type: 'scatter', x: U.by_gap.map(r => r.gap_um), y: U.by_gap.map(r => r.entropy), name: 'entropy (bits)', yaxis: 'y2', line: { color: '#f58518' } }], { xaxis: { title: 'gap distance (µm)' }, yaxis: { title: 'error rate' }, yaxis2: { overlaying: 'y', side: 'right', title: 'entropy', showgrid: false } })]]));
    root.append(h('h4', null, 'By gap distance'), tbl(['gap µm', 'voxels', 'error', 'entropy (bits)'], U.by_gap.map(r => [r.gap_um, r.n, r.error, r.entropy]), 'by gap'));
    root.append(h('div', { class: 'note' }, 'Entropy is split into expected entropy (class overlap within each sample) and mutual information (disagreement across registration × segmentation samples). Variance attribution reports registration and segmentation separately; they are not summed into one number.'));
    return root;
  },
  Registration(A) {
    const R = A.registration, s = R.sections, root = h('div', null);
    root.append(h('div', { class: 'card-row' }, card('method', R.method), card('mean TRE', fmt(R.mean_tre_um, 1) + ' µm'), card('max TRE', fmt(R.max_tre_um, 1) + ' µm'), card('mean ensemble spread', fmt(R.mean_sigma_um, 2) + ' µm', `spread/error ${fmt(R.sigma_to_error_ratio, 2)}`), card('rank corr(spread, error)', fmt(R.spearman_sigma_tre, 2)), card('ensemble samples', R.n_samples)));
    root.append(h('div', { class: 'note warn' }, 'TRE is measured against the simulation\'s true transform. The ensemble spread is a ranking signal; it under-states the true error unless recalibrated.'));
    root.append(twoPlots([
      ['Target registration error and ensemble spread per section', d => plot(d, [{ type: 'bar', x: s.map(r => r.index), y: s.map(r => r.tre_um), name: 'true TRE (µm)', marker: { color: '#e5645f' } }, { type: 'scatter', x: s.map(r => r.index), y: s.map(r => r.sigma_um), name: 'ensemble spread (µm)', line: { color: '#4da3ff' } }], { xaxis: { title: 'section index' }, yaxis: { title: 'µm' } })],
      ['Post-alignment NCC and failed samples', d => plot(d, [{ type: 'scatter', x: s.map(r => r.index), y: s.map(r => r.ncc), name: 'median NCC', line: { color: '#5dd39e' } }, { type: 'bar', x: s.map(r => r.index), y: s.map(r => r.failed_samples), name: 'failed samples', yaxis: 'y2', marker: { color: '#f0a030' } }], { xaxis: { title: 'section index' }, yaxis: { title: 'NCC', range: [0, 1.05] }, yaxis2: { overlaying: 'y', side: 'right', title: 'failed', showgrid: false, dtick: 1 } })]]));
    root.append(h('h4', null, 'Per section'), tbl(['#', 'z µm', 'TRE µm', 'spread µm', 'NCC', 'failed samples'], s.map(r => [r.index, r.z_um, r.tre_um, r.sigma_um, r.ncc, r.failed_samples]), 'registration'));
    return root;
  },
  'Sections QC'(A) {
    const q = A.qc, root = h('div', null);
    root.append(h('div', { class: 'note' }, 'Rule-based v0 QC (robust-z outliers on focus, noise, continuity). Flags are diagnostics, not validated decisions.'));
    root.append(twoPlots([['Focus (variance of Laplacian) and noise σ per section', d => plot(d, [{ x: q.map(r => r.index), y: q.map(r => r.focus), name: 'focus', line: { color: '#4da3ff' } }, { x: q.map(r => r.index), y: q.map(r => r.noise), name: 'noise σ', yaxis: 'y2', line: { color: '#e5645f' } }], { xaxis: { title: 'section index' }, yaxis: { title: 'focus' }, yaxis2: { overlaying: 'y', side: 'right', showgrid: false, title: 'noise σ' } })],
      ['Intensity continuity to the previous section', d => plot(d, [{ type: 'bar', x: q.map(r => r.index), y: q.map(r => r.continuity), marker: { color: q.map(r => r.verdict === 'accept' ? '#5dd39e' : r.verdict === 'flag' ? '#f0a030' : '#e5645f') } }], { xaxis: { title: 'section index' }, yaxis: { title: 'Wasserstein / pooled sd' } })]]));
    root.append(tbl(['#', 'z µm', 'focus', 'noise σ', 'mean', 'sd', 'continuity', 'verdict', 'reasons'], q.map(r => [r.index, r.z_um, r.focus, r.noise, r.mean, r.sd, r.continuity, chip(r.verdict, r.verdict === 'accept' ? 'good' : r.verdict === 'flag' ? 'warn' : 'bad'), r.reasons.join('; ')]), 'section QC'));
    return root;
  },
  Texture(A) {
    const T = A.texture, root = h('div', null);
    root.append(h('div', { class: 'note' }, T.note));
    const rows = T.per_class.map(c => [c.class, c.n_sections, ...['contrast', 'entropy', 'homogeneity', 'correlation'].map(f => c.features[f] ? `${fmt(c.features[f].mean, 3)} ± ${fmt(c.features[f].sd, 3)}` : '—'), chip(c.z_feature_mode, c.z_feature_mode === '3D' ? 'good' : c.z_feature_mode === 'masked' ? 'bad' : 'warn'), c.z_feature_reasons.join('; '), c.min_scale_um]);
    root.append(h('h4', null, 'GLCM texture per class (observed registered sections; mean ± sd across sections)'), tbl(['class', 'sections', 'contrast', 'entropy', 'homogeneity', 'correlation', 'across-section (z) features', 'why', 'min scale µm'], rows, 'texture'));
    root.append(h('div', { class: 'note' }, 'Across-section (z) texture is allowed only where the observed fraction is high enough, the registration-limited scale is below the feature scale, and the section spacing is within the tissue correlation length; otherwise features fall back to per-section 2.5D or are masked.'));
    const B = T.observed_vs_interpolated_bias;
    if (B && B.length) {
      root.append(h('h4', null, 'Does interpolation distort texture? (simulation truth)'));
      const feats = ['contrast', 'entropy', 'homogeneity'], grid = h('div', { class: 'grid2' });
      feats.forEach(f => { const { d } = plotDiv(); grid.append(h('div', null, head(f), d)); later.push(() => plot(d, ['truth', 'observed_only', 'naive_interpolated'].map((k, i) => ({ type: 'bar', name: k.replace('_', ' '), x: B.map(r => r.class), y: B.map(r => r[f][k]), marker: { color: ['#7f8ea3', '#2b7bba', '#f0a030'][i] } })), { barmode: 'group' })); });
      root.append(grid, tbl(['class', 'feature', 'truth', 'observed planes only', 'naively interpolated', 'interp. bias vs truth'], B.flatMap(r => feats.map(f => [r.class, f, r[f].truth, r[f].observed_only, r[f].naive_interpolated, r[f].truth ? (r[f].naive_interpolated / r[f].truth - 1) : null])), 'texture bias'));
    }
    return root;
  },
  Measurements(A) {
    const M = A.measurements, root = h('div', null), rows = M.rows;
    root.append(h('div', { class: 'note warn' }, M.note));
    root.append(twoPlots([['Class volume fractions with 90% intervals', d => plot(d, [{ type: 'scatter', mode: 'markers', x: rows.map(r => r.class), y: rows.map(r => r.estimate), name: 'estimate', error_y: { type: 'data', symmetric: false, array: rows.map(r => r.hi90 - r.estimate), arrayminus: rows.map(r => r.estimate - r.lo90), color: '#4da3ff' }, marker: { size: 10, color: '#4da3ff' } }, { type: 'scatter', mode: 'markers', x: rows.map(r => r.class), y: rows.map(r => r.truth), name: 'truth', marker: { symbol: 'diamond', size: 11, color: '#f0a030' } }], { yaxis: { title: 'volume fraction' } })]]));
    root.append(tbl(['class', 'estimate', '90% lo', '90% hi', 'truth', 'covered?', '|error|', 'rel. half-width', 'MC sd', 'sectioning sd', 'observed frac.', 'verdict', 'reasons'],
      rows.map(r => [r.class, r.estimate, r.lo90, r.hi90, r.truth, chip(r.covered ? 'inside' : 'OUTSIDE', r.covered ? 'good' : 'bad'), r.abs_error, r.rel_halfwidth, r.sd_mc, r.sd_sectioning, r.observed_fraction, chip(r.verdict.replace(/_/g, ' '), r.verdict === 'reportable' ? 'good' : r.verdict === 'insufficient_evidence' ? 'bad' : 'warn'), r.reasons.join('; ')]), 'measurements'));
    root.append(h('div', { class: 'note' }, '“covered?” compares the interval with the simulation truth, which a real study would not have. The verdict uses only evidence available without truth (observed fraction, interval width, calibration status).'));
    return root;
  },
  Planner(A) {
    const P = A.planner, root = h('div', null);
    root.append(h('div', { class: 'card-row' }, card('target', P.target_class), card('current estimation sd', fmt(P.current_sd, 4)), card('sections now', P.current_sections)));
    root.append(h('div', { class: 'note' }, P.note));
    root.append(twoPlots([['Estimation sd vs number of sections', d => plot(d, [{ x: P.uniform_designs.map(r => r.n), y: P.uniform_designs.map(r => r.sd), name: 'uniform design', line: { color: '#8d96a6' } }, { x: P.greedy_sequence.map(r => P.current_sections + r.added), y: P.greedy_sequence.map(r => r.sd), name: 'current + planner additions', mode: 'lines+markers', line: { color: '#5dd39e' } }], { xaxis: { title: 'number of sections' }, yaxis: { title: 'sd of estimated volume fraction' } })],
      ['Recommended additional section positions', d => plot(d, [{ type: 'bar', x: P.recommendations.map(r => r.z_um), y: P.recommendations.map(r => r.relative_sd_reduction), marker: { color: '#4da3ff' } }], { xaxis: { title: 'z (µm)' }, yaxis: { title: 'relative sd reduction' } })]]));
    root.append(tbl(['rank', 'z µm', 'variance reduction', 'relative sd reduction', 'cost', 'EVOI/cost', 'method'], P.recommendations.map(r => [r.rank, r.z_um, r.variance_reduction, r.relative_sd_reduction, r.cost, r.evoi_per_cost, r.method]), 'planner'));
    return root;
  },
  Conformal(A) {
    const C = A.conformal, root = h('div', null);
    root.append(h('div', { class: 'note' }, C.note));
    root.append(tbl(['direction', 'nominal', 'coverage (mean over sections)', 'worst section', 'sd over sections', 'mean set size', 'qhat'], Object.entries(C.directions).map(([k, d]) => [k, d.nominal, d.coverage_mean, d.coverage_min_section, d.coverage_sd_sections, d.mean_set_size, d.qhat]), 'conformal'));
    return root;
  },
  Provenance(A) {
    const P = A.provenance, root = h('div', null), pl = P.planes;
    root.append(h('div', { class: 'card-row' }, Object.entries(P.fraction).map(([k, v]) => card(k, pct(v, 1)))));
    root.append(twoPlots([['Provenance along depth', d => plot(d, [{ type: 'bar', x: pl.map(r => r.z_um), y: pl.map(() => 1), marker: { color: pl.map(r => ({ 0: '#555', 1: '#2b7bba', 2: '#f0a030', 3: '#b04ab0' }[r.provenance])) }, customdata: pl.map(r => r.gap_um), hovertemplate: 'z %{x} µm · gap %{customdata} µm<extra></extra>' }], { xaxis: { title: 'z (µm)' }, yaxis: { visible: false }, bargap: 0 })],
      ['Distance to the nearest observed slab', d => plot(d, [{ x: pl.map(r => r.z_um), y: pl.map(r => r.gap_um), mode: 'lines', fill: 'tozeroy', line: { color: '#f0a030' } }], { xaxis: { title: 'z (µm)' }, yaxis: { title: 'µm' } })]]));
    root.append(h('div', { class: 'note' }, 'Every voxel carries a provenance label: observed (inside a physical section slab), interpolated (gap, model inference with quantified uncertainty), synthetic (generated; Mode B) or unsampled. Exports without provenance are refused by the software.'));
    return root;
  },
  Manifest(A) {
    const M = A.manifest, root = h('div', null);
    root.append(h('div', { class: 'card-row' }, card('schema', M.schema_version), card('seed', M.seed), card('intended use', M.intended_use), card('config hash', (M.config_hash || '').slice(0, 10))));
    root.append(h('h4', null, 'Stages'), tbl(['stage', 'parameters'], M.stages.map(s => [s.name, JSON.stringify(s.params)]), 'stages'));
    root.append(h('h4', null, 'Configuration'), h('pre', { class: 'm' }, JSON.stringify(M.config, null, 2)));
    if (A.log) root.append(h('h4', null, 'Run log'), h('pre', { class: 'm' }, A.log.join('\n')));
    return root;
  },
};
