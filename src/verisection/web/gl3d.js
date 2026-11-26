/* ===== WebGL2 ray-marched 3D view ===== */
const VS = `#version 300 es
in vec3 aPos; uniform mat4 uMVP; uniform vec3 uHalf; out vec3 vObj;
void main(){ vec3 p = aPos * uHalf; vObj = p; gl_Position = uMVP * vec4(p, 1.0); }`;
const FS = `#version 300 es
precision highp float; precision highp sampler3D; precision highp sampler2D;
in vec3 vObj; out vec4 outc;
uniform vec3 uCam, uHalf, uTexel, uClipMin, uClipMax, uCursor;
uniform sampler3D uInt, uLab, uUnc, uProv, uProb, uErr, uTru; uniform sampler2D uLUT;
uniform int uMode, uCut, uCls; uniform float uSteps, uDensity, uLight, uIso, uThr;
uniform vec4 uPal[8]; uniform float uOn[8];
float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
vec3 tex3(vec3 p){ return p / (2.0 * uHalf) + 0.5; }
vec3 lutc(float t){ return texture(uLUT, vec2(clamp(t, 0.0, 1.0), 0.5)).rgb; }
vec3 grad(sampler3D s, vec3 q){ vec3 e = uTexel;
  return vec3(texture(s, q + vec3(e.x,0.,0.)).r - texture(s, q - vec3(e.x,0.,0.)).r,
              texture(s, q + vec3(0.,e.y,0.)).r - texture(s, q - vec3(0.,e.y,0.)).r,
              texture(s, q + vec3(0.,0.,e.z)).r - texture(s, q - vec3(0.,0.,e.z)).r); }
float shade(vec3 g, vec3 rd){ float l = length(g); if (l < 2e-3) return 1.0;
  vec3 n = g / l; float d = abs(dot(n, rd)); float spec = pow(d, 24.0);
  return mix(1.0, 0.30 + 0.70 * d + 0.35 * spec, uLight); }
void main(){
  vec3 rd = normalize(vObj - uCam); vec3 ro = uCam;
  vec3 inv = 1.0 / rd; vec3 t0 = (-uHalf - ro) * inv, t1 = (uHalf - ro) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  float tn = max(max(tmin.x, tmin.y), tmin.z), tf = min(min(tmax.x, tmax.y), tmax.z);
  tn = max(tn, 0.0); if (tf <= tn) discard;
  float dt = (tf - tn) / uSteps; float jit = hash(gl_FragCoord.xy);
  vec3 acc = vec3(0.0); float a = 0.0; float mx = 0.0; float prev = 0.0; bool hit = false; vec3 hq = vec3(0.0);
  for (int i = 0; i < 640; i++){
    if (float(i) >= uSteps) break;
    float tt = tn + (float(i) + jit) * dt; vec3 q = tex3(ro + rd * tt);
    if (any(lessThan(q, uClipMin)) || any(greaterThan(q, uClipMax))) { prev = 0.0; continue; }
    if (uCut == 1 && q.x > uCursor.x && q.y > uCursor.y && q.z > uCursor.z) { prev = 0.0; continue; }
    if (uMode == 0 || uMode == 6){
      int c = int((uMode == 6 ? texture(uTru, q).r : texture(uLab, q).r) * 255.0 + 0.5) - 1; if (c < 0) continue;
      if (uOn[c] < 0.5) continue; vec4 pal = uPal[c]; float s = pal.a * uDensity; if (s <= 0.0) continue;
      vec3 col = pal.rgb * shade(grad(uInt, q), rd);
      float aa = 1.0 - exp(-s * dt * 34.0); acc += (1.0 - a) * aa * col; a += (1.0 - a) * aa;
    } else if (uMode == 1){
      float u = texture(uUnc, q).r; mx = max(mx, u);
    } else if (uMode == 2){
      float pv = texture(uProv, q).r * 255.0;
      if (pv > 1.5 && pv < 2.5){ float aa = 1.0 - exp(-0.55 * uDensity * dt * 34.0); vec3 col = vec3(0.94, 0.63, 0.19) * shade(grad(uInt, q), rd); acc += (1.0 - a) * aa * col; a += (1.0 - a) * aa; }
      else if (pv > 0.5){ float aa = 1.0 - exp(-0.55 * uDensity * dt * 34.0); vec3 col = vec3(0.17, 0.55, 0.85); acc += (1.0 - a) * aa * col; a += (1.0 - a) * aa; }
    } else if (uMode == 3){
      if (texture(uProv, q).r < 0.002) continue;
      float v = texture(uInt, q).r; float s = uDensity * smoothstep(0.30, 0.95, v) * 2.2; if (s <= 0.0) continue;
      vec3 col = lutc(v) * shade(grad(uInt, q), rd);
      float aa = 1.0 - exp(-s * dt * 34.0); acc += (1.0 - a) * aa * col; a += (1.0 - a) * aa;
    } else if (uMode == 4){
      float p = texture(uProb, q).r;
      if (p >= uIso && prev < uIso){ hit = true; hq = q; break; }
      prev = p;
    } else if (uMode == 5){
      float e = texture(uErr, q).r; if (e > 0.5){ float aa = 1.0 - exp(-2.5 * uDensity * dt * 34.0); acc += (1.0 - a) * aa * vec3(0.95, 0.2, 0.2); a += (1.0 - a) * aa; }
      else { int c = int(texture(uLab, q).r * 255.0 + 0.5) - 1; if (c >= 0 && uOn[c] > 0.5){ float aa = 1.0 - exp(-0.10 * uDensity * dt * 34.0); acc += (1.0 - a) * aa * uPal[c].rgb; a += (1.0 - a) * aa; } }
    }
    if (a > 0.985) break;
  }
  if (uMode == 1){ if (mx <= uThr) discard; outc = vec4(lutc(mx) * 0.96, 0.96); return; }
  if (uMode == 4){
    if (!hit) discard;
    vec3 g = grad(uProb, hq); float l = length(g); vec3 n = l > 1e-5 ? -g / l : vec3(0.0, 0.0, 1.0);
    float diff = max(dot(n, -rd), 0.0); float spec = pow(max(dot(reflect(rd, n), -rd), 0.0), 40.0);
    vec3 base = uCls < 8 ? uPal[uCls].rgb : vec3(0.8); float pv = texture(uProv, hq).r * 255.0;
    if (pv > 1.5 && pv < 2.5) base = mix(base, vec3(0.94, 0.63, 0.19), 0.45);
    vec3 col = base * (0.28 + 0.72 * diff) + vec3(0.30) * spec;
    outc = vec4(col, 1.0); return;
  }
  if (a < 0.003) discard;
  outc = vec4(acc, a);
}`;
const LVS = `#version 300 es
in vec3 aPos; in vec4 aCol; uniform mat4 uMVP; out vec4 vCol; void main(){ vCol = aCol; gl_Position = uMVP * vec4(aPos, 1.0); }`;
const LFS = `#version 300 es
precision highp float; in vec4 vCol; out vec4 o; void main(){ o = vec4(vCol.rgb * vCol.a, vCol.a); }`;

const M4 = {
  mul(a, b) { const o = new Float32Array(16); for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s; } return o; },
  persp(fovy, asp, n, f) { const t = 1 / Math.tan(fovy / 2), o = new Float32Array(16); o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f); return o; },
  look(e, c, u) {
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], nrm = a => { const l = Math.hypot(...a) || 1; return a.map(x => x / l); };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const f = nrm(sub(c, e)), s = nrm(cross(f, u)), v = cross(s, f), o = new Float32Array(16);
    o[0] = s[0]; o[4] = s[1]; o[8] = s[2]; o[1] = v[0]; o[5] = v[1]; o[9] = v[2]; o[2] = -f[0]; o[6] = -f[1]; o[10] = -f[2];
    o[12] = -dot(s, e); o[13] = -dot(v, e); o[14] = dot(f, e); o[15] = 1; return o;
  },
};
const GL3D = {
  gl: null, dirty: true, ok: false, cam: { az: 0.85, el: 0.5, dist: 2.5, t: [0, 0, 0] }, tex: {}, half: [0.5, 0.5, 0.5], mvp: null, eye: [0, 0, 0],
  init() {
    const cv = $('#cv3d'); if (!cv) return;
    const gl = cv.getContext('webgl2', { premultipliedAlpha: true, antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.gl = gl;
    if (!gl) { this.msg('WebGL2 is not available in this browser; the 3D view is disabled (2D views still work).'); return; }
    const mk = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { console.error(gl.getShaderInfoLog(s)); this.msg('Shader error: ' + gl.getShaderInfoLog(s)); } return s; };
    const prog = (v, f) => { const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, v)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, f)); gl.bindAttribLocation(p, 0, 'aPos'); gl.bindAttribLocation(p, 1, 'aCol'); gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) { console.error(gl.getProgramInfoLog(p)); this.msg('Link error: ' + gl.getProgramInfoLog(p)); } return p; };
    this.pv = prog(VS, FS); this.pl = prog(LVS, LFS);
    // unit cube (36 vertices of +-1)
    const c = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]];
    const f = [[0, 1, 2, 0, 2, 3], [4, 6, 5, 4, 7, 6], [0, 4, 5, 0, 5, 1], [3, 2, 6, 3, 6, 7], [0, 3, 7, 0, 7, 4], [1, 5, 6, 1, 6, 2]];
    const v = []; f.forEach(t => t.forEach(i => v.push(...c[i])));
    this.cubeBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.cubeBuf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.STATIC_DRAW);
    this.lineBuf = gl.createBuffer(); this.ok = true;
    this.events(cv); this.loop();
  },
  msg(t) { const o = $('#vp-3d .vfoot'); if (o) o.textContent = t; },
  tex3d(unit, data, filter) {
    const gl = this.gl, [Z, Y, X] = S.dims; gl.activeTexture(gl.TEXTURE0 + unit);
    let t = this.tex[unit]; if (!t) t = this.tex[unit] = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, t); gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, X, Y, Z, 0, gl.RED, gl.UNSIGNED_BYTE, data);
    const fl = filter === 'nearest' ? gl.NEAREST : gl.LINEAR;
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, fl); gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, fl);
    ['TEXTURE_WRAP_S', 'TEXTURE_WRAP_T', 'TEXTURE_WRAP_R'].forEach(w => gl.texParameteri(gl.TEXTURE_3D, gl[w], gl.CLAMP_TO_EDGE));
  },
  upload() {
    if (!this.ok || !S.P) return;
    const V = S.vol, [Z, Y, X] = S.dims, P = S.P, e = [X * P.pixelUm, Y * P.pixelUm, Z * P.dzUm], m = Math.max(...e);
    this.half = e.map(v => v / m * 0.5);
    this.tex3d(0, V.intensity, 'linear'); this.tex3d(1, V.labels, 'nearest'); this.tex3d(2, V.unc, 'linear'); this.tex3d(3, V.prov, 'nearest');
    this.tex3d(5, V.err || new Uint8Array(X * Y * Z), 'nearest'); this.tex3d(7, V.truth || new Uint8Array(X * Y * Z), 'nearest'); this.setProbClass(); this.setLUT();
    this.resetCamera(false); this.dirty = true;
  },
  setProbClass() { if (this.ok && S.P) this.tex3d(4, S.vol.probs[clamp(S.R.cls, 0, S.vol.probs.length - 1)], 'linear'); this.dirty = true; },
  setLUT() {
    if (!this.ok) return; const gl = this.gl; gl.activeTexture(gl.TEXTURE0 + 6);
    if (!this.lutTex) this.lutTex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
    const name = S.R.mode === 3 ? 'he' : S.R.cmap; gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, 256, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, CM[name] || CM.magma);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); this.dirty = true;
  },
  resetCamera(draw = true) { this.cam = { az: 0.85, el: 0.5, dist: 2.5, t: [0, 0, 0] }; if (draw) this.dirty = true; },
  setView(name) {
    const v = { front: [Math.PI / 2, 0.0], back: [-Math.PI / 2, 0.0], left: [Math.PI, 0.0], right: [0, 0.0], top: [Math.PI / 2, 1.5], bottom: [Math.PI / 2, -1.5], iso: [0.85, 0.5] }[name];
    if (v) { this.cam.az = v[0]; this.cam.el = v[1]; this.dirty = true; }
  },
  objPos(i, j, k) { const [Z, Y, X] = S.dims, h = this.half; return [((i + .5) / X - .5) * 2 * h[0], ((j + .5) / Y - .5) * 2 * h[1], ((k + .5) / Z - .5) * 2 * h[2]]; },
  lines() {
    const [Z, Y, X] = S.dims, h = this.half, pos = [], col = [];
    const seg = (a, b, c) => { pos.push(...a, ...b); col.push(...c, ...c); };
    const rect = (axis, v, c) => {  // rectangle through the box at object coordinate v along axis
      const p = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
      const pt = (u, w) => axis === 2 ? [u * h[0], w * h[1], v] : axis === 1 ? [u * h[0], v, w * h[2]] : [v, u * h[1], w * h[2]];
      for (let i = 0; i < 4; i++) seg(pt(...p[i]), pt(...p[(i + 1) % 4]), c);
    };
    // box
    const bc = [0.62, 0.68, 0.78, 0.8], co = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]];
    [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(([a, b]) => seg(co[a].map((x, i) => x * h[i]), co[b].map((x, i) => x * h[i]), bc));
    if (S.R.frames) {
      const cp = this.objPos(S.cur[2], S.cur[1], S.cur[0]);
      rect(2, cp[2], [0.30, 0.64, 1.0, 0.95]); rect(1, cp[1], [1.0, 0.42, 0.42, 0.95]); rect(0, cp[0], [0.36, 0.83, 0.62, 0.95]);
    }
    if (S.R.slabs) {
      for (let k = 0; k < Z; k++) if (planeProv(k) === 1) { const z = this.objPos(0, 0, k)[2]; rect(2, z, [0.17, 0.48, 0.73, 0.55]); }
    }
    const verts = new Float32Array(pos.length / 3 * 7);
    for (let i = 0; i < pos.length / 3; i++) { verts.set(pos.slice(i * 3, i * 3 + 3), i * 7); verts.set(col.slice(i * 4, i * 4 + 4), i * 7 + 3); }
    return { verts, n: pos.length / 3 };
  },
  matrices() {
    // this.tile (exports only) = { asp, crop:[x0,y0,x1,y1] in NDC }: render a sub-rectangle of a larger virtual image
    const c = this.cam, cv = $('#cv3d'), o = this.tile, asp = o ? o.asp : cv.width / Math.max(cv.height, 1);
    const eye = [c.t[0] + c.dist * Math.cos(c.el) * Math.cos(c.az), c.t[1] + c.dist * Math.cos(c.el) * Math.sin(c.az), c.t[2] - c.dist * Math.sin(c.el)];
    this.eye = eye;
    const V = M4.look(eye, c.t, [0, 0, -1]), Pm = M4.persp(0.56, asp, 0.05, 20);
    this.mvpFull = M4.mul(Pm, V); this.mvp = this.mvpFull;
    if (o && o.crop) {
      const [x0, y0, x1, y1] = o.crop, C = new Float32Array(16);
      C[0] = 2 / (x1 - x0); C[5] = 2 / (y1 - y0); C[10] = 1; C[15] = 1; C[12] = -(x1 + x0) / (x1 - x0); C[13] = -(y1 + y0) / (y1 - y0);
      this.mvp = M4.mul(C, this.mvpFull);
    }
  },
  draw() {
    if (!this.ok || !S.P) return;
    const gl = this.gl, cv = $('#cv3d'); if (cv.width < 4) return;
    gl.viewport(0, 0, cv.width, cv.height); const bgc = location.hash.includes('dbgbg') ? [0.25, 0.0, 0.35, 1] : [0, 0, 0, 0]; gl.clearColor(...bgc); gl.clear(gl.COLOR_BUFFER_BIT);
    this.matrices();
    const [Z, Y, X] = S.dims, P = S.P, R = S.R;
    gl.useProgram(this.pv);
    const U = n => gl.getUniformLocation(this.pv, n);
    gl.uniformMatrix4fv(U('uMVP'), false, this.mvp); gl.uniform3fv(U('uHalf'), this.half); gl.uniform3fv(U('uCam'), this.eye);
    gl.uniform3f(U('uTexel'), 1 / X, 1 / Y, 1 / Z); gl.uniform3f(U('uClipMin'), R.clip[0], R.clip[1], R.clip[2]); gl.uniform3f(U('uClipMax'), R.clip[3], R.clip[4], R.clip[5]);
    gl.uniform3f(U('uCursor'), (S.cur[2] + .5) / X, (S.cur[1] + .5) / Y, (S.cur[0] + .5) / Z);
    gl.uniform1i(U('uMode'), R.mode); gl.uniform1i(U('uCut'), R.cut ? 1 : 0); gl.uniform1i(U('uCls'), R.cls);
    gl.uniform1f(U('uSteps'), R.steps); gl.uniform1f(U('uDensity'), R.density); gl.uniform1f(U('uLight'), R.light); gl.uniform1f(U('uIso'), R.iso); gl.uniform1f(U('uThr'), R.thr);
    const pal = new Float32Array(32), on = new Float32Array(8);
    S.palette.forEach((c, i) => { if (i < 8) { pal.set([c[0] / 255, c[1] / 255, c[2] / 255, S.rw[i]], i * 4); on[i] = S.classOn[i] ? 1 : 0; } });
    gl.uniform4fv(U('uPal'), pal); gl.uniform1fv(U('uOn'), on);
    ['uInt', 'uLab', 'uUnc', 'uProv', 'uProb', 'uErr', 'uLUT'].forEach((n, i) => gl.uniform1i(U(n), i)); gl.uniform1i(U('uTru'), 7);
    for (let i = 0; i < 6; i++) { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_3D, this.tex[i]); }
    gl.activeTexture(gl.TEXTURE0 + 6); gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
    gl.activeTexture(gl.TEXTURE0 + 7); gl.bindTexture(gl.TEXTURE_3D, this.tex[7]);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cubeBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0); gl.disableVertexAttribArray(1);
    gl.enable(gl.CULL_FACE); gl.cullFace(gl.FRONT); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLES, 0, 36); gl.disable(gl.CULL_FACE);
    // wire overlays
    const L = this.lines(); gl.useProgram(this.pl);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.pl, 'uMVP'), false, this.mvp);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf); gl.bufferData(gl.ARRAY_BUFFER, L.verts, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);
    gl.drawArrays(gl.LINES, 0, L.n);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, this._px || (this._px = new Uint8Array(4)));  // forces completion so the frame is presented/captured
    this.gizmo();
    const names = ['Class composite', 'Uncertainty MIP', 'Provenance', 'Image (DVR)', 'Isosurface P(class)', 'Error map', 'Ground-truth classes'];
    const m = $('#mode3d'); if (m) m.textContent = names[R.mode] + (R.mode === 4 ? ` · ${P.classNames[R.cls]} ≥ ${R.iso}` : '');
    const f = $('#vf-3d'); if (f) f.textContent = 'drag orbit · shift-drag pan · wheel zoom · double-click reset' + (location.hash.includes('dbg') ? ` | canvas ${cv.width}x${cv.height} css ${cv.clientWidth}x${cv.clientHeight} glerr ${gl.getError()} px ${(() => { const b = new Uint8Array(4); gl.readPixels(cv.width >> 1, cv.height >> 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, b); return Array.from(b).join(','); })()} ovnz ${(() => { const o = $('#ov3d'); const d = o.getContext('2d').getImageData(0, 0, o.width, o.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n; })()} draws ${(this.n = (this.n || 0) + 1)} half ${this.half.map(v => v.toFixed(2))} eye ${this.eye.map(v => v.toFixed(2))}` : '');
  },
  project(p) { const m = this.mvpFull || this.mvp, x = p[0], y = p[1], z = p[2], w = m[3] * x + m[7] * y + m[11] * z + m[15]; return [(m[0] * x + m[4] * y + m[8] * z + m[12]) / w, (m[1] * x + m[5] * y + m[9] * z + m[13]) / w]; },
  legendSpec() {
    const R = S.R, P = S.P, C = P.classNames, rgb = a => `rgb(${a.join(',')})`;
    const cls = () => C.map((n, i) => ({ col: P.classColors[i], label: n, off: !S.classOn[i] })).filter(x => !x.off);
    switch (R.mode) {
      case 0: return { title: 'Class (MAP label)', items: cls() };
      case 6: return { title: 'True class (simulation)', items: cls() };
      case 1: return { title: 'Predictive entropy', bar: { cmap: R.cmap, lo: '0', hi: P.uncScale.toFixed(1) + ' bits' }, note: `shown above ${(R.thr * P.uncScale).toFixed(2)} bits` };
      case 2: return { title: 'Provenance', items: [{ col: rgb(PROV_RGB[1]), label: 'observed' }, { col: rgb(PROV_RGB[2]), label: 'interpolated' }] };
      case 3: return { title: 'Image intensity', bar: { cmap: 'he', lo: 'low', hi: 'high (OD-like)' } };
      case 4: return { title: 'Isosurface', items: [{ col: P.classColors[R.cls], label: `${C[R.cls]}: P ≥ ${R.iso}` }, { col: rgb(PROV_RGB[2]), label: 'tinted where interpolated' }] };
      case 5: return { title: 'Error map (simulation)', items: [{ col: '#f23333', label: 'misclassified' }, { col: '#7f8ea3', label: 'class volume (faint)' }] };
      default: return null;
    }
  },
  gizmo() {
    const ov = $('#ov3d'), cv = $('#cv3d'); if (!ov || this.noOverlay) return; ov.width = cv.width; ov.height = cv.height;
    const k = devicePixelRatio || 1, c = ov.getContext('2d'); c.scale(k, k); this.drawOverlay(c, ov.width / k, ov.height / k);
  },
  /* annotations in CSS-pixel units on any 2D context: edge extents, orientation triad, legend */
  drawOverlay(c, w, hh) {
    const hf = this.half, P = S.P, FG = '#e8eef8', DIM = '#9fb0c6', FONT = '"Segoe UI",system-ui,sans-serif';
    const scr = p => { const q = this.project(p); return [(q[0] * .5 + .5) * w, (1 - (q[1] * .5 + .5)) * hh]; };
    const rr = (x, y, ww, h2, r) => { c.beginPath(); if (c.roundRect) c.roundRect(x, y, ww, h2, r); else c.rect(x, y, ww, h2); };
    // edge extents: three labeled box edges, in physical units
    const ext = [S.dims[2] * P.pixelUm, S.dims[1] * P.pixelUm, S.dims[0] * P.dzUm];
    const pill = (txt, p, col) => {
      const q = scr(p); c.font = `11px ${FONT}`; const tw = c.measureText(txt).width + 12;
      c.fillStyle = 'rgba(11,13,16,.78)'; rr(q[0] - tw / 2, q[1] - 9, tw, 18, 9); c.fill(); c.strokeStyle = col; c.lineWidth = 1; c.stroke();
      c.fillStyle = col; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(txt, q[0], q[1] + .5);
    };
    pill(`x  ${fmt(ext[0], 0)} µm`, [0, -hf[1], -hf[2]], '#ff9e5e'); pill(`y  ${fmt(ext[1], 0)} µm`, [-hf[0], 0, -hf[2]], '#8dd37e'); pill(`z  ${fmt(ext[2], 0)} µm (depth)`, [-hf[0], -hf[1], 0], '#7fb6ff');
    c.textAlign = 'left'; c.textBaseline = 'alphabetic';
    // orientation triad (bottom-left): axis directions as seen from the camera
    const o = scr([0, 0, 0]), cx = 40, cy = hh - 64;
    c.fillStyle = 'rgba(11,13,16,.55)'; c.beginPath(); c.arc(cx, cy, 30, 0, 7); c.fill();
    [['x', [0.2, 0, 0], '#ff9e5e'], ['y', [0, 0.2, 0], '#8dd37e'], ['z', [0, 0, 0.2], '#7fb6ff']].forEach(([t, v, col]) => {
      const q = scr(v), dx = q[0] - o[0], dy = q[1] - o[1], l = Math.hypot(dx, dy); if (l < 1e-3) return;
      const ex = cx + dx / l * 21, ey = cy + dy / l * 21; c.strokeStyle = col; c.lineWidth = 2; c.beginPath(); c.moveTo(cx, cy); c.lineTo(ex, ey); c.stroke();
      c.fillStyle = col; c.font = `bold 11px ${FONT}`; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(t, cx + dx / l * 27, cy + dy / l * 27);
    });
    c.textAlign = 'left'; c.textBaseline = 'alphabetic';
    // legend (bottom-right)
    const L = this.legendSpec();
    if (L) {
      c.font = `600 11px ${FONT}`; const rows = L.items || [], bw = 140;
      let ww = c.measureText(L.title).width; c.font = `11px ${FONT}`; rows.forEach(r => { ww = Math.max(ww, c.measureText(r.label).width + 20); }); if (L.bar) ww = Math.max(ww, bw); if (L.note) ww = Math.max(ww, c.measureText(L.note).width);
      const bh = 30 + rows.length * 17 + (L.bar ? 26 : 0) + (L.note ? 15 : 0), bx = w - ww - 30, by = hh - bh - 30;
      c.fillStyle = 'rgba(11,13,16,.78)'; rr(bx, by, ww + 20, bh, 7); c.fill(); c.strokeStyle = 'rgba(255,255,255,.14)'; c.lineWidth = 1; c.stroke();
      c.fillStyle = FG; c.font = `600 11px ${FONT}`; c.fillText(L.title, bx + 10, by + 17);
      c.font = `11px ${FONT}`; let y = by + 33;
      rows.forEach(r => { c.fillStyle = r.col; rr(bx + 10, y - 9, 10, 10, 2); c.fill(); c.fillStyle = FG; c.fillText(r.label, bx + 26, y); y += 17; });
      if (L.bar) {
        const lut = CM[L.bar.cmap] || CM.magma, g = c.createLinearGradient(bx + 10, 0, bx + 10 + bw, 0);
        for (let i = 0; i <= 8; i++) { const j = Math.min(255, i * 32); g.addColorStop(i / 8, `rgb(${lut[j * 3]},${lut[j * 3 + 1]},${lut[j * 3 + 2]})`); }
        c.fillStyle = g; rr(bx + 10, y - 6, bw, 9, 3); c.fill(); c.fillStyle = DIM; c.fillText(L.bar.lo, bx + 10, y + 14); c.textAlign = 'right'; c.fillText(L.bar.hi, bx + 10 + bw, y + 14); c.textAlign = 'left'; y += 26;
      }
      if (L.note) { c.fillStyle = DIM; c.fillText(L.note, bx + 10, y + 2); }
    }
  },
  /* High-resolution render of the 3D view: the image (cssW x cssH CSS px at k device px each) is rendered in equal tiles
     of at most 2048 px through a cropped projection, so neither canvas-size limits nor GPU time-outs apply. Returns a 2D canvas. */
  async renderHiRes(cssW, cssH, k, progress) {
    const cv = $('#cv3d'); if (!this.ok || !S.P || !cv) throw new Error('3D view unavailable (WebGL2)');
    const W = Math.round(cssW * k), H = Math.round(cssH * k), TILE = this.tileSize || 2048, nx = Math.ceil(W / TILE), ny = Math.ceil(H / TILE), tw = Math.ceil(W / nx), th = Math.ceil(H / ny);
    const out = document.createElement('canvas'); out.width = W; out.height = H; const c = out.getContext('2d');
    if (!c) throw new Error('could not allocate a ' + W + '×' + H + ' canvas');
    c.fillStyle = '#0b0d10'; c.fillRect(0, 0, W, H);
    const save = { w: cv.width, h: cv.height, steps: S.R.steps }; this.noOverlay = true; S.R.steps = this.exportSteps || Math.max(S.R.steps, 384);   // finer ray sampling for magnified output
    try {
      for (let ty = 0; ty < ny; ty++) for (let tx = 0; tx < nx; tx++) {
        const x0 = tx * tw, y0 = ty * th, w = Math.min(tw, W - x0), hh = Math.min(th, H - y0);
        cv.width = w; cv.height = hh;
        this.tile = { asp: W / H, crop: [2 * x0 / W - 1, 1 - 2 * (y0 + hh) / H, 2 * (x0 + w) / W - 1, 1 - 2 * y0 / H] };
        this.draw(); c.drawImage(cv, x0, y0);
        if (progress) progress((ty * nx + tx + 1) / (nx * ny));
        await new Promise(r => setTimeout(r, 0));
      }
      this.tile = { asp: W / H, crop: null }; this.matrices();   // true-aspect matrices for the annotations
      c.save(); c.scale(k, k); this.drawOverlay(c, cssW, cssH); c.restore();
    } finally { this.tile = null; this.noOverlay = false; S.R.steps = save.steps; cv.width = save.w; cv.height = save.h; this.dirty = true; }
    return out;
  },
  loop() {
    if (this.running) return; this.running = true;
    const step = () => { if (this.paused) { this.running = false; return; } if (S.R.auto) { this.cam.az += 0.006; this.dirty = true; } if (this.dirty) { this.dirty = false; try { this.draw(); } catch (e) { console.error(e); } } requestAnimationFrame(step); };
    requestAnimationFrame(step);
  },
  pause() { this.paused = true; },                       // exports pause the render loop (it would compete for the GPU and the main thread)
  resume() { if (this.hold) return; this.paused = false; this.dirty = true; this.loop(); },   // hold = keep paused (self-test: idle frames would burn the headless time budget)
  events(cv) {
    let drag = null;
    cv.addEventListener('mousedown', e => { drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey || e.button === 1, c: JSON.parse(JSON.stringify(this.cam)) }; e.preventDefault(); });
    window.addEventListener('mouseup', () => { drag = null; });
    window.addEventListener('mousemove', e => {
      if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (drag.pan) { const k = this.cam.dist * 0.0016; const az = drag.c.az; this.cam.t = [drag.c.t[0] - dx * k * -Math.sin(az) - dy * 0, drag.c.t[1] + dx * k * Math.cos(az), drag.c.t[2] + dy * k]; }
      else { this.cam.az = drag.c.az - dx * 0.008; this.cam.el = clamp(drag.c.el + dy * 0.008, -1.5, 1.5); }
      this.dirty = true;
    });
    cv.addEventListener('wheel', e => { e.preventDefault(); this.cam.dist = clamp(this.cam.dist * (e.deltaY > 0 ? 1.1 : 1 / 1.1), 0.9, 8); this.dirty = true; }, { passive: false });
    cv.addEventListener('dblclick', () => this.resetCamera());
    cv.addEventListener('contextmenu', e => e.preventDefault());
  },
};
