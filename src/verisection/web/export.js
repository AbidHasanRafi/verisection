/* ===== export: DPI-aware PNG of the window, panels and analysis tabs; ZIP bundles of charts and tabs =====
   Resolution model: one on-screen CSS pixel is 1/96 inch, so "N dpi" means scale = N/96 and the physical size equals the
   on-screen size. The DPI is written into the PNG (pHYs chunk) so layout and journal tools read the true print size.
   Everything is rendered, not stretched: the interface is re-rasterised from the DOM in tiles (vector text), slice views are
   re-drawn at the target resolution, the 3D view is ray-marched tile by tile, and charts are re-rendered by Plotly. */
const EXPORT_DPIS = [300, 600, 900, 1200], CSS_DPI = 96, EXPORT_MAX_MP = 150, EXPORT_HEAVY_MP = 100, EXPORT_MAX_SIDE = 16384, CAPTION_PX = 40;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const stage = s => { document.body.dataset.xstage = s; };   // current export stage, visible to the tests when a render stalls
const slug = s =>String(s).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 60) || 'item';

/* ---------- plan: what resolution is achievable for a region of w x h CSS px ---------- */
function exportPlan(w, h, dpi, caption = true) {   // caption = a research-use strip is added below the image (all scopes except the full window)
  const k = dpi / CSS_DPI, hh = h + (caption ? CAPTION_PX : 0), kmax = Math.min(EXPORT_MAX_SIDE / Math.max(w, hh), Math.sqrt(EXPORT_MAX_MP * 1e6 / (w * hh)));
  const eff = Math.min(dpi, Math.floor(kmax * CSS_DPI + 1e-6)), ke = eff / CSS_DPI;
  return { reqDpi: dpi, dpi: eff, k: ke, W: Math.round(w * ke), H: Math.round(h * ke), mp: w * ke * h * ke / 1e6, capped: eff < dpi, inches: [w / CSS_DPI, h / CSS_DPI] };
}

/* ---------- PNG: write the DPI; ZIP: minimal store-only writer ---------- */
const CRC_T = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(u8) { let c = 0xFFFFFFFF; for (let i = 0; i < u8.length; i++) c = CRC_T[(c ^ u8[i]) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
async function pngWithDpi(blob, dpi) {
  const head = new Uint8Array(await blob.slice(0, 33).arrayBuffer());   // signature (8) + IHDR chunk (25)
  if (![137, 80, 78, 71, 13, 10, 26, 10].every((v, i) => head[i] === v) || String.fromCharCode(...head.slice(12, 16)) !== 'IHDR') throw new Error('the browser produced an invalid PNG');
  const ppm = Math.round(dpi / 0.0254), chunk = new Uint8Array(21), dv = new DataView(chunk.buffer);
  dv.setUint32(0, 9); chunk.set([0x70, 0x48, 0x59, 0x73], 4); dv.setUint32(8, ppm); dv.setUint32(12, ppm); chunk[16] = 1;   // pHYs: pixels per metre, unit = metre
  dv.setUint32(17, crc32(chunk.subarray(4, 17)));
  return new Blob([head, chunk, blob.slice(33)], { type: 'image/png' });
}
const canvasToBlob = c => new Promise((res, rej) => c.toBlob(b => b ? res(b) : rej(new Error(`the browser could not encode a ${c.width}×${c.height} px image (out of memory?)`)), 'image/png'));
function dataUrlToBlob(url) { const [meta, b64] = url.split(','), bin = atob(b64), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Blob([u], { type: meta.slice(5, meta.indexOf(';')) }); }
async function zipBlob(files) {
  const enc = new TextEncoder(), parts = [], central = [], now = new Date(); let off = 0;
  const dosTime = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1), dosDate = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
  for (const f of files) {
    const data = new Uint8Array(await f.blob.arrayBuffer()), name = enc.encode(f.name), crc = crc32(data);
    const lh = new DataView(new ArrayBuffer(30)); lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0x0800, true); lh.setUint16(10, dosTime, true); lh.setUint16(12, dosDate, true);
    lh.setUint32(14, crc, true); lh.setUint32(18, data.length, true); lh.setUint32(22, data.length, true); lh.setUint16(26, name.length, true);
    const ch = new DataView(new ArrayBuffer(46)); ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true); ch.setUint16(8, 0x0800, true); ch.setUint16(12, dosTime, true); ch.setUint16(14, dosDate, true);
    ch.setUint32(16, crc, true); ch.setUint32(20, data.length, true); ch.setUint32(24, data.length, true); ch.setUint16(28, name.length, true); ch.setUint32(42, off, true);
    parts.push(lh.buffer, name, data); central.push(ch.buffer, name); off += 30 + name.length + data.length;
  }
  const csize = central.reduce((a, b) => a + b.byteLength, 0), end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true); end.setUint32(12, csize, true); end.setUint32(16, off, true);
  return new Blob([...parts, ...central, end.buffer], { type: 'application/zip' });
}
function tablesCsv(tab) {
  const tabs = S.tables[tab] || []; if (!tabs.length) return '';
  const esc = v => { v = v === null || v === undefined ? '' : String(v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  return tabs.map(t => '# ' + t.name + '\n' + [t.headers, ...t.rows].map(r => r.map(esc).join(',')).join('\n')).join('\n\n');
}

/* ---------- DOM rasterisation (SVG foreignObject, drawn in tiles) ---------- */
function exportCss() {
  // Every active rule, re-rooted from <body> to .vsroot. Inside an SVG image, viewport units and @media queries would be
  // evaluated against the SVG's own size, not the window: so media queries are resolved here (matching ones are flattened,
  // others dropped) and vh/vw are converted to px. @container queries stay: they depend on the element, not the viewport.
  const root = sel => sel.replace(/(^|[\s,>+~])body(?![\w-])/g, '$1.vsroot');
  const text = r => {
    if (r.type === 1) return root(r.selectorText) + '{' + r.style.cssText + '}';
    if (r.type === 4) return matchMedia(r.media.mediaText).matches ? Array.from(r.cssRules).map(text).join('') : '';
    if (r.cssRules && r.cssText.indexOf('{') > 0) return r.cssText.slice(0, r.cssText.indexOf('{')) + '{' + Array.from(r.cssRules).map(text).join('') + '}';
    return r.cssText;
  };
  let css = ''; for (const sh of Array.from(document.styleSheets)) { try { for (const r of Array.from(sh.cssRules)) css += text(r) + '\n'; } catch (e) { /* cross-origin sheet: skip */ } }
  return css.replace(/(-?\d*\.?\d+)v([hw])\b/g, (m, n, a) => (+n * (a === 'h' ? innerHeight : innerWidth) / 100).toFixed(2) + 'px');
}
function syncFormState(src, dst, noScroll) {   // serialisation reads attributes, not live properties: copy the state across
  const a = [src, ...src.querySelectorAll('*')], b = [dst, ...dst.querySelectorAll('*')], wraps = [];
  a.forEach((s, i) => {
    const d = b[i];
    if (s.tagName === 'INPUT') { d.setAttribute('value', s.value); if (s.type === 'checkbox' || s.type === 'radio') { if (s.checked) d.setAttribute('checked', ''); else d.removeAttribute('checked'); } }
    else if (s.tagName === 'SELECT') Array.from(d.options).forEach((o, j) => { if (j === s.selectedIndex) o.setAttribute('selected', ''); else o.removeAttribute('selected'); });
    if (!noScroll && (s.scrollTop || s.scrollLeft) && getComputedStyle(s).display === 'block') wraps.push([s, d]);
  });
  wraps.forEach(([s, d]) => { const w = document.createElement('div'); w.style.transform = `translate(${-s.scrollLeft}px,${-s.scrollTop}px)`; while (d.firstChild) w.append(d.firstChild); d.append(w); });
}
function cloneForExport(roots, W, H, opts = {}) {
  const root = document.createElement('div');
  root.className = 'vsroot ' + (document.body.classList.contains('dark') ? 'dark' : 'light');
  root.style.cssText = `width:${W}px;height:${H}px;position:relative;overflow:hidden;`;
  const st = document.createElement('style'); st.textContent = exportCss(); root.append(st);
  roots.forEach(src => { const dst = src.cloneNode(true); syncFormState(src, dst, opts.noScroll); if (opts.fit) opts.fit(dst); root.append(dst); });
  return root;
}
function loadSvg(svg) {
  return new Promise((res, rej) => {
    const img = new Image();   // a data: URL keeps the canvas untainted (blob: URLs of foreignObject SVG taint it)
    img.onload = () => res(img);
    img.onerror = () => rej(new Error('the browser could not rasterise the interface at this size'));
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  });
}
async function rasterize(root, W, H, k, progress) {   // the cloned DOM -> a canvas of (W*k) x (H*k) px, tile by tile (<= 4096 px)
  const xhtml = new XMLSerializer().serializeToString(root), out = document.createElement('canvas');
  out.width = Math.round(W * k); out.height = Math.round(H * k);
  const ctx = out.getContext('2d'); if (!ctx) throw new Error(`could not allocate a ${out.width}×${out.height} px canvas`);
  const tile = Math.max(128, Math.floor(4096 / k)), nx = Math.ceil(W / tile), ny = Math.ceil(H / tile);
  for (let ty = 0; ty < ny; ty++) for (let tx = 0; tx < nx; tx++) {
    const X0 = Math.round(tx * tile * k), X1 = Math.min(out.width, Math.round((tx + 1) * tile * k)), Y0 = Math.round(ty * tile * k), Y1 = Math.min(out.height, Math.round((ty + 1) * tile * k));
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${X1 - X0}" height="${Y1 - Y0}" viewBox="${X0 / k} ${Y0 / k} ${(X1 - X0) / k} ${(Y1 - Y0) / k}"><foreignObject x="0" y="0" width="${W}" height="${H}">${xhtml}</foreignObject></svg>`;
    document.body.dataset.xstage = `tile ${ty * nx + tx + 1}/${nx * ny} svg ${svg.length} chars`;   // visible to the tests when a render stalls
    ctx.drawImage(await loadSvg(svg), X0, Y0, X1 - X0, Y1 - Y0);
    if (progress) progress((ty * nx + tx + 1) / (nx * ny));
    await sleep(0);
  }
  return out;
}
async function paintCanvases(out, k, rg, progress) {   // overpaint the canvas panels with true high-resolution renders
  if (!S.P) return [];
  const ctx = out.getContext('2d'), warn = [], items = [];
  PLANES.forEach(pl => { const c = S.canv[pl], vp = $('#vp-' + pl); if (c && vp && vp.getClientRects().length && (!rg.el || rg.el.contains(c))) items.push([pl, c]); });
  const c3 = $('#cv3d'), v3 = $('#vp-3d'); if (c3 && v3 && v3.getClientRects().length && (!rg.el || rg.el.contains(c3))) items.push(['3d', c3]);
  let n = 0;
  for (const [kind, c] of items) {
    const r = c.getBoundingClientRect(), cw = r.width, ch = r.height, x = Math.round((r.left - rg.origin.x) * k), y = Math.round((r.top - rg.origin.y) * k);
    try {
      if (kind === '3d') { const t = await GL3D.renderHiRes(cw, ch, k); ctx.drawImage(t, x, y); t.width = t.height = 0; }
      else { const t = document.createElement('canvas'); t.width = Math.round(cw * k); t.height = Math.round(ch * k); renderSlice(kind, { cv: t, k: t.width / cw, cw, ch }); ctx.drawImage(t, x, y); t.width = t.height = 0; }
    } catch (e) { warn.push(kind + ': ' + e.message); }
    if (progress) progress(++n / items.length);
  }
  return warn;
}
function addCaption(canvas, k, text) {   // research-use caption strip (kept on every export except the full window, which has the banner)
  const w = canvas.width / k, lines = [], fs = 10.5, probe = document.createElement('canvas').getContext('2d');
  probe.font = `${fs}px "Segoe UI",sans-serif`; let line = '';
  text.split(' ').forEach(word => { const t = line ? line + ' ' + word : word; if (probe.measureText(t).width > w - 20 && line) { lines.push(line); line = word; } else line = t; });
  lines.push(line);
  const strip = 10 + lines.length * 14, out = document.createElement('canvas'); out.width = canvas.width; out.height = canvas.height + Math.round(strip * k);
  const c = out.getContext('2d'); if (!c) throw new Error('could not allocate the caption canvas');
  c.drawImage(canvas, 0, 0); c.fillStyle = '#14181e'; c.fillRect(0, canvas.height, out.width, out.height - canvas.height);
  c.scale(k, k); c.fillStyle = '#ffe9a8'; c.font = `${fs}px "Segoe UI",sans-serif`; c.textBaseline = 'middle';
  lines.forEach((l, i) => c.fillText(l, 10, canvas.height / k + 5 + 7 + i * 14));
  return out;
}

/* ---------- regions and scopes ---------- */
const EXPORT_SCOPES = [
  ['window', 'Entire window (every panel, as on screen)'], ['viewports', 'All views (slices + 3D), montage'], ['active', 'Active view with its header'],
  ['active-image', 'Active view, image only (figure)'], ['tab', 'Analysis: current tab, full content'], ['tab-charts', 'Analysis: charts of current tab (ZIP: PNG + SVG + CSV)'], ['all-tabs', 'Analysis: every tab (ZIP of PNGs)'],
];
const scopeLabel = id => (EXPORT_SCOPES.find(s => s[0] === id) || [0, id])[1];
function exportRegion(scope) {
  const R = el => { const r = el.getBoundingClientRect(); return { el, w: Math.round(r.width), h: Math.round(r.height), origin: { x: r.left, y: r.top } }; };
  if (scope === 'window') return { el: null, w: innerWidth, h: innerHeight, origin: { x: 0, y: 0 } };
  if (scope === 'viewports') return R($('#viewports'));
  if (scope === 'active') return R($('#vp-' + S.active));
  if (scope === 'active-image') return R($('#vp-' + S.active + ' .vbody'));
  if (scope === 'tab' || scope === 'all-tabs') { const p = $('#tabpane'), r = p.getBoundingClientRect(); return { el: p, w: p.clientWidth, h: p.scrollHeight, origin: { x: r.left, y: r.top } }; }
  if (scope === 'tab-charts') { const d = $('#tabpane .plot'); return { el: null, w: d ? d.clientWidth : 400, h: d ? d.clientHeight : 300, origin: { x: 0, y: 0 } }; }
  throw new Error('unknown export scope: ' + scope);
}
async function waitForPlots(timeout = 6000) {
  const t0 = performance.now();
  while (performance.now() - t0 < timeout) {
    if ($$('#tabpane .plot').every(p => p.classList.contains('js-plotly-plot') && p.querySelector('.main-svg'))) { await sleep(60); return true; }
    await sleep(50);
  }
  return false;
}
const captionFor = (label, dpi, extra) => `VeriSection · ${S.P ? S.P.title : (S.A ? S.A.title : '')} · ${label}${extra ? ' · ' + extra : ''} · ${dpi} dpi · RESEARCH USE ONLY: not a medical device; not for diagnosis; interpolated voxels are model inference, not observation.`;

async function exportOne(scope, dpiReq, progress, tabName) {
  const rg = exportRegion(scope), plan = exportPlan(rg.w, rg.h, dpiReq, scope !== 'window'), k = plan.k, warn = [];
  let canvas;
  if (scope === 'active-image') {
    if (S.active === '3d') canvas = await GL3D.renderHiRes(rg.w, rg.h, k, p => progress(p * 0.9));
    else { canvas = document.createElement('canvas'); canvas.width = Math.round(rg.w * k); canvas.height = Math.round(rg.h * k); renderSlice(S.active, { cv: canvas, k: canvas.width / rg.w, cw: rg.w, ch: rg.h }); progress(0.9); }
  } else {
    let root;
    if (scope === 'window') root = cloneForExport(Array.from(document.body.children).filter(e => !/^(SCRIPT|STYLE)$/.test(e.tagName) && !['modal-root', 'toast', 'errbox'].includes(e.id)), rg.w, rg.h);
    else if (scope === 'tab') root = cloneForExport([$('#tabpane')], rg.w, rg.h, { noScroll: true, fit: d => Object.assign(d.style, { overflow: 'visible', height: 'auto', flex: 'none', width: rg.w + 'px' }) });
    else root = cloneForExport([rg.el], rg.w, rg.h, { fit: d => Object.assign(d.style, { flex: '1 1 auto', width: '100%', height: '100%' }) });
    stage('rasterize ' + scope); canvas = await rasterize(root, rg.w, rg.h, k, p => progress(p * 0.7));
    stage('paint canvases'); warn.push(...await paintCanvases(canvas, k, rg, p => progress(0.7 + p * 0.25)));
  }
  stage('caption/encode');
  if (scope !== 'window') canvas = addCaption(canvas, k, captionFor(scope === 'tab' || scope === 'all-tabs' ? 'Analysis tab: ' + (tabName || S.tab) : scopeLabel(scope), plan.dpi, scope === 'active-image' || scope === 'active' ? S.active : ''));
  const info = Object.assign({}, plan, { scope, width: canvas.width, height: canvas.height, warnings: warn });
  stage('toBlob ' + canvas.width + 'x' + canvas.height); const png = await canvasToBlob(canvas); stage('pHYs');
  const blob = await pngWithDpi(png, plan.dpi); canvas.width = canvas.height = 0; stage('done ' + scope);
  progress(1);
  return { blob, info, name: `verisection_${slug(tabName ? scope + '_' + tabName : scope === 'tab' ? 'tab_' + S.tab : scope === 'active' || scope === 'active-image' ? scope + '_' + S.active : scope)}_${plan.dpi}dpi.png` };
}
async function exportCharts(dpiReq, opts, progress) {
  const plots = $$('#tabpane .plot').filter(p => p.classList.contains('js-plotly-plot'));
  if (!plots.length) throw new Error('This tab has no charts');
  const bg = document.body.classList.contains('dark') ? getComputedStyle(document.body).getPropertyValue('--panel2').trim() || '#1f252d' : '#ffffff';
  const files = [], infos = [];
  for (let i = 0; i < plots.length; i++) {
    const div = plots[i], head = div.parentElement.querySelector('h4'), title = head ? head.textContent : 'chart ' + (i + 1), w = div.clientWidth, h = div.clientHeight, plan = exportPlan(w, h, dpiReq, false);
    const fig = { data: div.data, layout: merge(div.layout, { paper_bgcolor: bg, plot_bgcolor: bg }) }, base = `${String(i + 1).padStart(2, '0')}_${slug(title)}`;
    files.push({ name: `${base}_${plan.dpi}dpi.png`, blob: await pngWithDpi(dataUrlToBlob(await Plotly.toImage(fig, { format: 'png', width: w, height: h, scale: plan.k })), plan.dpi) });
    if (opts.svg !== false) { const u = await Plotly.toImage(fig, { format: 'svg', width: w, height: h }); files.push({ name: `${base}.svg`, blob: new Blob([decodeURIComponent(u.slice(u.indexOf(',') + 1))], { type: 'image/svg+xml' }) }); }
    infos.push(Object.assign({ title }, plan)); progress((i + 1) / plots.length);
  }
  const csv = tablesCsv(S.tab); if (csv) files.push({ name: `${slug(S.tab)}_tables.csv`, blob: new Blob([csv], { type: 'text/csv' }) });
  return { files: [{ name: `verisection_${slug(S.tab)}_charts.zip`, blob: await zipBlob(files) }], info: { scope: 'tab-charts', charts: infos, entries: files.map(f => f.name), dpi: infos[0].dpi, reqDpi: dpiReq, capped: infos.some(c => c.capped) } };
}
async function exportAllTabs(dpiReq, opts, progress) {
  const names = $$('#tabs .tab').map(t => t.textContent), orig = S.tab, files = [], infos = [];
  try {
    for (let i = 0; i < names.length; i++) {
      openTab(names[i]); await waitForPlots();
      const one = await exportOne('tab', dpiReq, p => progress((i + p) / names.length), names[i]);
      files.push({ name: `${String(i + 1).padStart(2, '0')}_${one.name.replace(/^verisection_tab_/, '')}`, blob: one.blob }); infos.push(one.info);
    }
  } finally { openTab(orig); }
  return { files: [{ name: `verisection_analysis_tabs_${infos[0].dpi}dpi.zip`, blob: await zipBlob(files) }], info: { scope: 'all-tabs', tabs: names, dpi: infos[0].dpi, reqDpi: dpiReq, capped: infos.some(c => c.capped), sizes: infos.map(i => [i.width, i.height]) } };
}
/* the single entry point: returns { files:[{name, blob}], info } and never downloads (so tests can call it) */
async function exportBuild(scope, dpiReq, opts = {}, progress = () => {}) {
  if (!S.P && !S.A && scope !== 'window') throw new Error('Load a volume or an image audit first');
  closeMenus(); GL3D.pause();
  const orig = S.theme, relight = opts.theme === 'light' && orig !== 'light';
  if (relight) { setTheme('light'); await waitForPlots(); }
  try {
    if (scope === 'tab-charts') return await exportCharts(dpiReq, opts, progress);
    if (scope === 'all-tabs') return await exportAllTabs(dpiReq, opts, progress);
    const one = await exportOne(scope, dpiReq, progress, null);
    return { files: [{ name: one.name, blob: one.blob }], info: one.info };
  } finally { if (relight) { setTheme(orig); await waitForPlots(); } fitCanvases(); GL3D.resume(); }
}

/* ---------- UI: dialog with a feasibility table ---------- */
S.exp = { scope: 'window', dpi: 300, theme: 'current', svg: true };
function exportDialog(pre = {}) {
  Object.assign(S.exp, pre);
  const E = S.exp, scopeSel = h('select', { onchange: e => { E.scope = e.target.value; refresh(); } }, EXPORT_SCOPES.map(([id, t]) => h('option', { value: id, selected: id === E.scope }, t)));
  const themeSel = h('select', { onchange: e => { E.theme = e.target.value; } }, [['current', 'Current theme'], ['light', 'Light (print-friendly, white charts)']].map(([v, t]) => h('option', { value: v, selected: v === E.theme }, t)));
  const svgBox = h('input', { type: 'checkbox', checked: E.svg, onchange: e => { E.svg = e.target.checked; } });
  const dpiSeg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Resolution' }, EXPORT_DPIS.map(d => h('button', { class: d === E.dpi ? 'on' : '', onclick: ev => { E.dpi = d; $$('button', ev.currentTarget.parentElement).forEach(b => b.classList.toggle('on', b === ev.currentTarget)); refresh(); } }, d + ' dpi')));
  const table = h('div'), svgRow = h('label', { class: 'crow', style: 'grid-template-columns:14px 1fr' }, svgBox, h('span', null, 'Also include vector SVG for each chart'));
  function refresh() {
    let rg; try { rg = exportRegion(E.scope); } catch (e) { table.replaceChildren(h('div', { class: 'note warn' }, e.message)); return; }
    svgRow.classList.toggle('hidden', E.scope !== 'tab-charts');
    const cap = E.scope !== 'window' && E.scope !== 'tab-charts';
    const rows = EXPORT_DPIS.map(d => { const p = exportPlan(rg.w, rg.h, d, cap); return h('tr', { style: d === E.dpi ? 'font-weight:600' : '' },
      h('td', null, d + ' dpi'), h('td', { class: 'n' }, `${p.W} × ${p.H}`), h('td', { class: 'n' }, p.mp.toFixed(1) + ' MP'),
      h('td', null, p.capped ? h('span', { class: 'chip warn' }, 'capped at ' + p.dpi + ' dpi') : p.mp > EXPORT_HEAVY_MP ? h('span', { class: 'chip warn', title: 'Needs several GB of free memory; close other tabs first' }, 'heavy') : h('span', { class: 'chip good' }, 'full'))); });
    const p0 = exportPlan(rg.w, rg.h, E.dpi, cap);
    table.replaceChildren(h('div', { class: 'tblwrap' }, h('table', { class: 't' }, h('tr', null, ['target', 'pixels', 'size', 'status'].map(x => h('th', null, x))), rows)),
      h('div', { class: 'note' }, `Print size ${p0.inches[0].toFixed(1)} × ${p0.inches[1].toFixed(1)} in (on-screen size at 96 px/in). ` + (E.scope === 'all-tabs' ? 'Sizes shown for the current tab; each tab is sized by its own content. ' : E.scope === 'tab-charts' ? 'Sizes shown for the first chart. ' : '') +
        (p0.capped ? `${E.dpi} dpi exceeds the browser's canvas limit (${EXPORT_MAX_MP} MP, ${EXPORT_MAX_SIDE} px per side) for this scope; it will export at ${p0.dpi} dpi instead. Choose a smaller scope for ${E.dpi} dpi.` : 'The DPI is written into the PNG so it places at the correct physical size.')));
  }
  const body = h('div', null, h('div', { class: 'form' },
    h('label', null, 'What to export'), scopeSel, h('label', null, 'Resolution'), dpiSeg, h('label', null, 'Appearance'), themeSel, h('span'), svgRow),
    table, h('div', { class: 'note warn' }, 'Exports carry a research-use caption (the window export keeps the on-screen banner). Slice views, the 3D view and charts are re-rendered at the target resolution, not enlarged.'));
  refresh();
  modal('Export image', body, [{ label: 'Cancel' }, { label: 'Export', primary: true, act: () => { runExport(E.scope, E.dpi, { theme: E.theme, svg: E.svg }); } }], { width: '640px' });
}
async function runExport(scope, dpi, opts) {
  const bar = h('i'), msg = h('div', null, 'Rendering ' + scopeLabel(scope) + ' at ' + dpi + ' dpi…'), t0 = performance.now();
  const m = modal('Exporting image', h('div', null, msg, h('div', { class: 'prog' }, bar), h('div', { class: 'tlabel' }, 'High resolutions can take a minute. The 3D view is re-rendered tile by tile, so the window may flicker.')), [], { sticky: true });
  try {
    const res = await exportBuild(scope, dpi, opts || {}, p => { bar.style.width = (100 * Math.min(1, p)).toFixed(0) + '%'; });
    res.files.forEach(f => download(f.name, f.blob)); m.close();
    const i = res.info; toast(`Saved ${res.files[0].name} · ${i.width ? i.width + '×' + i.height + ' px · ' : ''}${i.dpi} dpi${i.capped ? ' (requested ' + i.reqDpi + ', capped by browser limits)' : ''} · ${((performance.now() - t0) / 1000).toFixed(0)} s`, 6000);
    if (i.warnings && i.warnings.length) modal('Export finished with warnings', h('ul', null, i.warnings.map(w => h('li', null, w))));
  } catch (e) {
    m.close(); console.error(e);
    modal('Export failed', h('div', null, h('p', null, String(e.message || e)), h('p', { class: 'note' }, 'Try a lower resolution or a smaller scope: browsers limit canvas memory.')));
  } finally { if (window.VS_TEST_NODL && window.VS_RELEASE) window.VS_RELEASE(); }
}

/* ---------- self-test (#selftest=export): runs real exports and records facts for the test-suite ---------- */
function pngFacts(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength), f = { width: dv.getUint32(16), height: dv.getUint32(20), ppmX: null, ppmY: null, chunks: [] };
  let o = 8; while (o + 8 <= u8.length) { const len = dv.getUint32(o), type = String.fromCharCode(...u8.subarray(o + 4, o + 8)); f.chunks.push(type); if (type === 'pHYs') { f.ppmX = dv.getUint32(o + 8); f.ppmY = dv.getUint32(o + 12); } if (type === 'IDAT' || type === 'IEND') break; o += 12 + len; }
  return f;
}
async function previewOf(blob, crop) {
  const bmp = crop ? await createImageBitmap(blob, crop[0], crop[1], Math.min(crop[2], 1e9), Math.min(crop[3], 1e9)) : await createImageBitmap(blob);
  const s = Math.min(1, 1400 / bmp.width), c = document.createElement('canvas'); c.width = Math.max(1, Math.round(bmp.width * s)); c.height = Math.max(1, Math.round(bmp.height * s));
  const x = c.getContext('2d'); x.drawImage(bmp, 0, 0, c.width, c.height);
  const d = x.getImageData(0, 0, c.width, c.height).data; let sum = 0, sq = 0, n = 0; for (let i = 0; i < d.length; i += 4 * 7) { const v = (d[i] + d[i + 1] + d[i + 2]) / 3; sum += v; sq += v * v; n++; }
  return { dataUrl: c.toDataURL('image/png'), std: Math.sqrt(Math.max(0, sq / n - (sum / n) ** 2)), w: bmp.width, h: bmp.height };
}
async function tileCheck() {   // the same 3D image rendered as one tile and as many small tiles must agree (no seams, no offsets)
  const r = $('#cv3d').getBoundingClientRect(), g = GL3D, save = [g.tileSize, g.exportSteps]; let A, B;
  try {
    g.exportSteps = 48; g.tileSize = 4096; A = await g.renderHiRes(r.width, r.height, 1.5);
    g.tileSize = 60; B = await g.renderHiRes(r.width, r.height, 1.5);
  } finally { [g.tileSize, g.exportSteps] = save; }
  const da = A.getContext('2d').getImageData(0, 0, A.width, A.height).data, db = B.getContext('2d').getImageData(0, 0, B.width, B.height).data;
  let sum = 0, mx = 0, nz = 0; for (let i = 0; i < da.length; i += 4) { const d = (Math.abs(da[i] - db[i]) + Math.abs(da[i + 1] - db[i + 1]) + Math.abs(da[i + 2] - db[i + 2])) / 3; sum += d; if (d > mx) mx = d; if (da[i] + da[i + 1] + da[i + 2] > 60) nz++; }
  return { w: A.width, h: A.height, tiles: Math.ceil(A.width / 60) * Math.ceil(A.height / 60), meanAbsDiff: sum / (da.length / 4), maxDiff: mx, litFraction: nz / (da.length / 4) };
}
async function exportSelfTest(spec) {
  GL3D.hold = true; GL3D.pause();
  const out = { plans: [exportPlan(1500, 900, 300), exportPlan(1500, 900, 1200), exportPlan(400, 300, 1200)], cases: [], errors: [] };
  try { out.tileCheck = await tileCheck(); } catch (e) { out.errors.push(['tileCheck', 0, String(e && e.stack || e)]); }
  const b64 = u8 => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000)); return btoa(s); };
  const pre = h('pre', { id: 'selftest', style: 'display:none' }, JSON.stringify(Object.assign({ running: true }, out))); document.body.append(pre);   // partial results stay readable if a later case stalls
  for (const [scope, dpi, opts, cropIt, act] of spec) {
    if (act) { S.active = act; markActive(); }
    try {
      const t0 = performance.now(), r = await exportBuild(scope, dpi, opts), f = r.files[0], u8 = new Uint8Array(await f.blob.arrayBuffer());
      const rec = { scope, dpi, name: f.name, bytes: u8.length, info: r.info, ms: Math.round(performance.now() - t0) };
      if (/\.png$/.test(f.name)) { rec.png = pngFacts(u8); rec.preview = await previewOf(f.blob); if (cropIt) rec.crop = await previewOf(f.blob, cropIt); } else rec.zipB64 = b64(u8);
      out.cases.push(rec);
    } catch (e) { out.errors.push([scope, dpi, String(e && e.stack || e)]); }
    pre.textContent = JSON.stringify(Object.assign({ running: true }, out));
  }
  pre.textContent = JSON.stringify(out);
}
