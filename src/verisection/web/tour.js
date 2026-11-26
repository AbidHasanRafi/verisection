/* ===== guided tour: highlights each region of the interface and says what it is for ===== */
let TOUR = null;
function tourSteps() {
  const au = S.ws === 'audit';
  return [
    { sel: '#banner', title: 'Research use only', text: 'VeriSection is research software. It is not a medical device and nothing it shows is a diagnosis or a treatment recommendation. The full statement, with the known limits, is under Help ▸ Intended use and limitations.' },
    { sel: '#menubar', title: 'Menubar', text: au
      ? 'File runs a new image audit and exports figures and tables. View ▸ Workspace switches between Reconstruction and Image audit. Analysis lists the tabs of the bottom dock. Help holds this tour, shortcuts and the limits.'
      : 'File creates simulations, opens volumes and exports figures (300–1200 dpi) and data (CSV, JSON). View sets layouts, crosshair, zoom and theme. Layers, Volume and Tools control overlays, the 3D renderer and measurements. Analysis lists the tabs of the bottom dock.' },
    { sel: '#toolbar', title: 'Toolbar', text: au
      ? 'Left: the workspace switch and the audit button. Right: this tour and the image export. The gallery has its own QC filter, and coloring is chosen in the left dock.'
      : 'Left: the workspace switch and the simulation button. Then the tool (cursor, distance, region of interest), the layout and 3D render mode, and one-click overlay toggles (uncertainty, provenance, segmentation, truth). Right: image export and theme.' },
    { sel: '#dockL', title: au ? 'Collections' : 'Layers', text: au
      ? 'One card per public collection, one row per class. Untick a class and it disappears from the gallery and from every chart and table. “Color by” switches between class shades and collection hues.'
      : 'One card per layer: image, segmentation, uncertainty, provenance, truth and error, registration variance. Each has visibility and opacity presets. Orange hatching always marks interpolated voxels: model inference, not observation.' },
    { sel: au ? '#audview' : '#viewports', title: au ? 'Gallery' : 'Views', text: au
      ? 'Thumbnails of sampled patches. The frame color is the class; a dot marks a patch flagged or excluded by the quality rule. Click one to inspect it. The filter above the grid shows only accepted, flagged or excluded patches.'
      : 'Three linked orthogonal slice views and a WebGL2 ray-marched 3D view share one crosshair. Scroll to change plane, drag to orbit the 3D view. The ruler beside the axial view shows which planes were observed and which were interpolated.' },
    { sel: '#dockR', title: au ? 'Image inspector' : 'Inspector', text: au
      ? 'The selected patch with its hematoxylin and eosin channels (color deconvolution in the browser), its quality verdict and reason, and its stain, quality and texture numbers.'
      : 'What is under the crosshair: provenance, predicted class, entropy, class posterior bars and the distance to the nearest observed section. Draw a region of interest or measure a 3D distance from here.' },
    { sel: '#dockB', title: 'Analysis dock', text: au
      ? 'One tab per question: Overview, Stain, Quality, Texture, Separability, Explanation, Stability, Interpolation, Method & data. Every chart has a one-line “how to read this”, scatter points are clickable, and each tab’s tables export to CSV.'
      : 'One tab per question: segmentation, calibration, uncertainty, registration, section QC, texture, measurements, planner, provenance, manifest. Each chart has a one-line “how to read this”; tables export to CSV. Drag the handle above to resize.' },
    { sel: '#statusbar', title: 'Status bar', text: 'The last action, the cursor position and provenance (or the selected patch), and the research-use reminder. Errors are reported here rather than silently.' },
  ];
}
function tour(i = 0) {
  endTour();
  const root = h('div', { id: 'tour', role: 'dialog', 'aria-label': 'Guided tour' }, h('div', { class: 'hole' }), h('div', { class: 'pop' }));
  document.body.append(root); TOUR = { i, steps: tourSteps() }; tourDraw();
}
function endTour() { const t = $('#tour'); if (t) t.remove(); TOUR = null; }
function tourStep(d) { if (!TOUR) return; const n = TOUR.i + d; if (n < 0) return; if (n >= TOUR.steps.length) { endTour(); return; } TOUR.i = n; tourDraw(); }
function tourDraw() {
  const st = TOUR.steps[TOUR.i], root = $('#tour'), hole = $('.hole', root), pop = $('.pop', root);
  let el = $(st.sel); if (!el || !el.getClientRects().length) { tourStep(1); return; }
  const r = el.getBoundingClientRect(), pad = 3;
  Object.assign(hole.style, { left: r.left - pad + 'px', top: r.top - pad + 'px', width: r.width + 2 * pad + 'px', height: r.height + 2 * pad + 'px' });
  pop.replaceChildren(h('h3', null, st.title), h('p', null, st.text), h('div', { class: 'nav' }, h('span', { class: 'grow' }, `${TOUR.i + 1} / ${TOUR.steps.length}`),
    h('button', { class: 'btn sm', onclick: endTour }, 'Close'), TOUR.i ? h('button', { class: 'btn sm', onclick: () => tourStep(-1) }, 'Back') : null,
    h('button', { class: 'btn sm primary', onclick: () => tourStep(1) }, TOUR.i === TOUR.steps.length - 1 ? 'Done' : 'Next')));
  const pw = 340, ph = pop.offsetHeight, W = innerWidth, H = innerHeight, gap = 14;
  let x, y;
  if (r.width > W * 0.55) { x = clamp(r.left + r.width / 2 - pw / 2, 8, W - pw - 8); y = r.bottom + gap + ph < H ? r.bottom + gap : Math.max(8, r.top - gap - ph); }
  else if (r.right + gap + pw < W) { x = r.right + gap; y = clamp(r.top + r.height / 2 - ph / 2, 8, H - ph - 8); }
  else { x = Math.max(8, r.left - gap - pw); y = clamp(r.top + r.height / 2 - ph / 2, 8, H - ph - 8); }
  Object.assign(pop.style, { left: x + 'px', top: y + 'px' });
}
window.addEventListener('resize', () => { if (TOUR) tourDraw(); });

/* audit-workspace URL hooks (tests, screenshots): #ws=audit&tab=Stain&sel=12&theme=light&dock=520&tour=3&dialog=audit&menu=0&sub=1 */
function applyAuditHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  if (q.get('ws') === 'audit' && S.ws !== 'audit') setWorkspace('audit');
  if (S.ws !== 'audit') return;
  if (q.get('theme')) setTheme(q.get('theme'));
  if (q.get('dock')) dockHeight(+q.get('dock'));
  if (q.get('color')) { AS.colorBy = q.get('color'); buildAuditDocks(); audGallery(); }
  if (q.get('show')) { AS.verdictFilter = q.get('show'); audGallery(); }
  if (q.get('scen')) AS.scen = q.get('scen');
  if (q.get('kind')) AS.sepKind = q.get('kind');
  if (q.get('sel')) audSelect(+q.get('sel'));
  if (q.get('tab')) openTab(q.get('tab'));
  if (q.get('menu')) { openMenu(+q.get('menu')); if (q.get('sub')) { const r = $$('.menu-pop > .mi')[+q.get('sub')]; if (r) r.dispatchEvent(new MouseEvent('mouseenter')); } }
  if (q.get('dialog') === 'audit') auditDialog();
  if (q.get('dialog') === 'export') exportDialog(q.get('scope') ? { scope: q.get('scope'), dpi: +(q.get('dpi') || 300) } : {});
  if (q.has('tour')) tour(+q.get('tour') || 0);
  if (q.has('audit')) setTimeout(writeAudit, 900);
}
