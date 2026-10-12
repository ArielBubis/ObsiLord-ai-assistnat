// Renderer: draws the head off-screen with three.js, then one fragment shader
// redraws that image as dither, halftone or ASCII marks, plus the backdrop
// (dotted band, voice ring, orbs, specks), glitch slices and the shutter flash.
//
// layer: 'full' draws head and backdrop together; 'head' draws only the head;
// 'backdrop' draws only the backdrop, laid out around an anchor (the head's
// position on the page), so it can sit behind other page content.

import * as THREE from './vendor/three/build/three.module.min.js';

const RAMP = ' .:-=+*#%@';
const MODE_INDEX = { dither: 0, halftone: 1, ascii: 2 };

// Orbs drift around the head; at rest they sit in a row on the left.
const ORBS = [
  { A: 1.52, y: -.06, r: .09,  p: 0 },
  { A: 1.33, y: -.12, r: .1,   p: .12 },
  { A: 1.06, y: -.05, r: .085, p: .26 },
  { A: .92,  y: -.16, r: .095, p: .38 },
  { A: 1.22, y: .24,  r: .05,  p: -.1 },
];

const VERTEX = /* glsl */`
  void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const FRAGMENT = /* glsl */`
  precision highp float;
  uniform sampler2D tScene, tGlyphs;
  uniform vec2 uRes, uCell;
  uniform float uGlyphN, uDot, uTime, uStep, uGlitch, uContrast, uBright;
  uniform float uBand, uRing, uSpecks, uRipple, uScan, uVoice, uFlash, uTransparent;
  uniform int uMode;
  uniform vec3 uInk, uHead;
  uniform vec4 uOrbs[5];

  float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
  float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3. - 2. * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
  }
  float fbm(vec2 p) { float s = 0., a = .5; for (int i = 0; i < 4; i++) { s += a * noise(p); p *= 2.03; a *= .5; } return s; }
  float bayer2(vec2 a) { a = floor(a); return fract(dot(a, vec2(.5, a.y * .75))); }
  float bayer4(vec2 a) { return bayer2(.5 * a) * .25 + bayer2(a); }
  float bayer8(vec2 a) { return bayer4(.5 * a) * .25 + bayer2(a); }
  float smin(float a, float b, float k) { float h = clamp(.5 + .5 * (b - a) / k, 0., 1.); return mix(b, a, h) - k * h * (1. - h); }
  float taper(vec2 p, vec2 a, vec2 b, float ra, float rb) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0., 1.);
    return length(pa - ba * h) - mix(ra, rb, h);
  }

  // Scene colour averaged over one cell.
  vec4 sceneAt(vec2 c) {
    vec2 p = (c + .5) * uCell, o = uCell * .25;
    return .25 * (texture(tScene, (p - o) / uRes) + texture(tScene, (p + vec2(o.x, -o.y)) / uRes)
                + texture(tScene, (p + vec2(-o.x, o.y)) / uRes) + texture(tScene, (p + o) / uRes));
  }

  // Coverage of one cell's mark at tone v.
  float mark(float v, vec2 c, vec2 l, vec2 off) {
    if (uMode == 0) {
      float sq = step(off.x, uDot - .5) * step(off.y, uDot - .5);
      return sq * step(bayer8(c), v - .002);
    }
    if (uMode == 1) {
      float r = sqrt(clamp(v, 0., 1.)) * .64;
      float aa = 1. / uCell.x;
      return 1. - smoothstep(r - aa, r + aa, length(l - .5));
    }
    float i = floor(clamp(v, 0., 1.) * (uGlyphN - 1.) + .5);
    return texture(tGlyphs, vec2((i + l.x) / uGlyphN, l.y)).a;
  }

  void main() {
    vec2 px = gl_FragCoord.xy;
    vec2 cell = floor(px / uCell);
    vec2 l = fract(px / uCell);
    vec2 off = floor(mod(px, uCell));
    vec2 hr = uHead.z / uCell;
    vec2 hc = floor(uHead.xy / uCell);
    float g = uGlitch;

    // glitch: horizontal slice shifts
    vec2 sc = cell;
    float bA = floor(cell.y / 3.), bB = floor(cell.y / 10.);
    if (hash(vec2(bA, uStep)) < g * .22) sc.x += floor((hash(vec2(bA + 7.1, uStep)) - .5) * hr.x * .45);
    if (hash(vec2(bB + 3.3, uStep * 1.7)) < g * .12) sc.x += floor((hash(vec2(bB, uStep + 9.)) - .5) * hr.x * .9);

    // glitch: ruled-line blocks and displaced chunks near the head
    float ruled = -1.;
    for (int k = 0; k < 7; k++) {
      float f = float(k);
      if (hash(vec2(f * 3.1, uStep + 5.)) > g * .7) continue;
      vec2 ctr = hc + (vec2(hash(vec2(f, uStep * 1.3)), hash(vec2(f + 9., uStep * 1.3))) - .5) * vec2(1.7, 2.1) * hr;
      vec2 hs = vec2(mix(.08, .5, hash(vec2(f, uStep + 11.))) * hr.x, mix(1., 3.5, hash(vec2(f, uStep + 17.))));
      if (all(lessThan(abs(cell - ctr), hs))) {
        if (k < 3) ruled = mod(cell.y, 2.);
        else sc += floor((vec2(hash(vec2(f, uStep + 21.)), hash(vec2(f, uStep + 23.))) - .5) * vec2(.5, .3) * hr);
      }
    }

    vec4 s = sceneAt(sc);
    float lum = pow(max(dot(s.rgb, vec3(.2126, .7152, .0722)), 0.), 1. / 2.2);
    lum = clamp((lum - .5) * uContrast + .5 + uBright, 0., 1.);
    // thinking: a bright scan line rolling down the face
    float scanY = uHead.y + uHead.z * (1.15 - 2.3 * fract(uTime * .5));
    float rowY = (cell.y + .5) * uCell.y;
    lum = min(1., lum + uScan * .6 * (1. - smoothstep(0., uCell.y * 2.2, abs(rowY - scanY))));

    vec3 col = vec3(0.);
    bool onHead = s.a > .5;
    if (onHead) {
      if (uMode == 2) lum = max(lum, .12);
      col = uInk * mark(lum, cell, l, off);
    } else {
      // backdrop, laid out in head-radius units
      vec2 q = ((cell + .5) * uCell - uHead.xy) / uHead.z;
      float inRect = step(-1.62, q.x) * step(q.x, 1.95) * step(-.52, q.y) * step(q.y, .56) * uBand;
      vec2 rc = vec2(1.78, .2);
      float rr = .29 + .025 * uVoice;
      float dC = length(q - rc) - rr;
      float dS = smin(dC, taper(q, rc, vec2(.82, .27), .17, .04), .14);

      float smoke = fbm(vec2(q.x * 2.6 - uTime * .12, q.y * 5.5 + uTime * .03));
      float b = inRect * (.34 + .5 * smoothstep(.45, .85, smoke) * smoothstep(-.4, 1.4, q.x));
      float dist = length(q);
      b += inRect * uRipple * .45 * pow(.5 + .5 * sin(dist * 16. + uTime * 6.), 8.) * smoothstep(3.2, .9, dist);
      float dv = length(q - rc);
      b += inRect * uRing * uVoice * .55 * pow(.5 + .5 * sin(dv * 18. - uTime * 9.), 8.) * smoothstep(2.2, rr, dv);
      b *= .75 + .5 * hash(cell);
      if (uRing > .5 && dS < 0.) b = 0.;

      float rimP = exp(-abs(dC) * (70. - 30. * uVoice)) * (.7 + .5 * uVoice) * uRing;
      bool rimDot = hash(cell + floor(uTime * 12.) * 13.7) < rimP * .9;
      bool speck = uSpecks > .5 && hash(cell * .913 + floor(uTime * .8) * 17.) > .9993;

      if (rimDot) col = uInk * mark(uMode == 0 ? 1. : uMode == 1 ? .45 : .6, cell, l, off);
      else if (speck) col = uInk * mark(uMode == 0 ? 1. : uMode == 1 ? .3 : .12, cell, l, off) * .6;
      else if (b > 0.) {
        if (uMode == 0) col = uInk * mark(1., cell, l, off) * b * .5;
        else if (uMode == 1) col = uInk * mark(b * .6, cell, l, off) * .45;
        else col = uInk * mark(b * .45, cell, l, off) * .4;
      }
    }

    // drifting black orbs; negative depth means behind the head
    for (int i = 0; i < 5; i++) {
      vec4 o = uOrbs[i];
      if (o.z <= 0. || (onHead && o.w < 0.)) continue;
      vec2 d = (cell + .5) * uCell - o.xy;
      float dl = length(d);
      if (dl < o.z) {
        col = vec3(0.);
        float lit = dot(d / max(dl, 1e-3), vec2(-.6, .8));
        if (o.z - dl < uCell.x * 1.2 && lit > .2 && hash(cell + floor(uTime * 6.)) < lit * .8)
          col = uInk * mark(uMode == 0 ? 1. : .35, cell, l, off) * .5;
      }
    }

    if (ruled >= 0.) col = uInk * ruled * step(l.y, .6);
    col = mix(col, uInk - col, uFlash);
    // transparent layers let the page show through wherever nothing is drawn
    float a = uTransparent > .5 ? step(.002, max(col.r, max(col.g, col.b))) : 1.;
    gl_FragColor = vec4(col * a, a);
  }
`;

function makeGlyphAtlas() {
  const cw = 48, ch = 80;
  const c = document.createElement('canvas');
  c.width = cw * RAMP.length; c.height = ch;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.font = `700 ${Math.round(ch * 0.78)}px ui-monospace, Consolas, "Courier New", monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  [...RAMP].forEach((g, i) => ctx.fillText(g, cw * i + cw / 2, ch * 0.54));
  const tex = new THREE.CanvasTexture(c);
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearFilter;
  return tex;
}

export function createRenderer(container, { layer = 'full', transparent = false } = {}) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'display:block;width:100%;height:100%';
  container.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: transparent, powerPreference: 'high-performance' });
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(dpr);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(26, 1, 0.1, 100);
  const rig = new THREE.Group();
  scene.add(rig);
  const hemi = new THREE.HemisphereLight(0xffffff, 0x000000, 0.12);
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.7);
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.15);
  fillLight.position.set(3, -0.5, 2);
  const rimLight = new THREE.DirectionalLight(0xffffff, 2.4);
  rimLight.position.set(1.5, 2, -3);
  scene.add(hemi, keyLight, fillLight, rimLight);

  const sceneRT = new THREE.WebGLRenderTarget(1, 1);
  const post = new THREE.ShaderMaterial({
    depthTest: false, depthWrite: false,
    vertexShader: VERTEX, fragmentShader: FRAGMENT,
    uniforms: {
      tScene: { value: sceneRT.texture },
      tGlyphs: { value: makeGlyphAtlas() },
      uGlyphN: { value: RAMP.length },
      uRes: { value: new THREE.Vector2(1, 1) },
      uCell: { value: new THREE.Vector2(3, 3) },
      uDot: { value: 2 },
      uMode: { value: 0 },
      uTime: { value: 0 },
      uStep: { value: 0 },
      uGlitch: { value: 0 },
      uContrast: { value: 1.45 },
      uBright: { value: 0 },
      uInk: { value: new THREE.Color('#ededed') },
      uHead: { value: new THREE.Vector3(0, 0, 100) },
      uBand: { value: 1 },
      uRing: { value: 1 },
      uSpecks: { value: 1 },
      uRipple: { value: 0 },
      uScan: { value: 0 },
      uVoice: { value: 0 },
      uFlash: { value: 0 },
      uTransparent: { value: transparent ? 1 : 0 },
      uOrbs: { value: Array.from({ length: 5 }, () => new THREE.Vector4()) },
    },
  });
  const u = post.uniforms;
  const postScene = new THREE.Scene();
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), post);
  quad.frustumCulled = false;
  postScene.add(quad);
  const postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  let look = null;
  let W = 1, H = 1;
  let anchor = { x: 0, y: 0, r: 100 }; // backdrop layer: CSS px relative to this canvas
  const tmp = new THREE.Vector3();

  function applyCell() {
    if (!look) return;
    const b = Math.max(1, Math.round(look.dot * dpr));
    const m = MODE_INDEX[look.mode] ?? 0;
    if (m === 0) u.uCell.value.set(b, b);
    else if (m === 1) u.uCell.value.set(Math.round(b * 2.3), Math.round(b * 2.3));
    else u.uCell.value.set(Math.round(b * 2.4), Math.round(b * 4));
    u.uDot.value = Math.max(1, Math.round(b * 0.67));
    u.uMode.value = m;
  }

  function frame() {
    const aspect = camera.aspect;
    const zoom = look ? look.framing.zoom : 1;
    const lift = look ? look.framing.lift : 0.28;
    // head height as a share of the view height
    const frac = (aspect >= 1 ? Math.min(0.64, aspect * 0.36) : Math.min(0.5, aspect * 0.8)) * zoom;
    const viewH = 2 / frac;
    const dist = viewH / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
    camera.position.set(0, -lift, dist);
    camera.lookAt(0, -lift, 0);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    if (layer === 'backdrop') return;
    tmp.set(0, 0, 0).project(camera);
    const cx = (tmp.x * .5 + .5) * W, cy = (tmp.y * .5 + .5) * H;
    tmp.set(0, 1, 0).project(camera);
    u.uHead.value.set(cx, cy, (tmp.y * .5 + .5) * H - cy);
  }

  function resize() {
    const r = container.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
    renderer.setSize(w, h, false);
    W = Math.floor(w * dpr); H = Math.floor(h * dpr);
    sceneRT.setSize(Math.ceil(W / 2), Math.ceil(H / 2));
    camera.aspect = w / h;
    u.uRes.value.set(W, H);
    frame();
    applyCell();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  function setLook(next) {
    look = next;
    // the shader writes this colour straight to the screen, so keep the hex value as is
    u.uInk.value.setStyle(look.ink, THREE.LinearSRGBColorSpace);
    const L = look.light;
    const az = THREE.MathUtils.degToRad(L.azimuth), el = THREE.MathUtils.degToRad(L.elevation);
    keyLight.position.set(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)).multiplyScalar(4);
    keyLight.intensity = L.key;
    rimLight.intensity = L.rim;
    fillLight.intensity = L.fill;
    hemi.intensity = L.ambient;
    frame();
    applyCell();
  }

  function setModel(object) {
    rig.clear();
    if (object) rig.add(object);
  }

  // Head centre and radius in CSS px relative to the page.
  function headScreen() {
    const r = canvas.getBoundingClientRect();
    const [x, y, z] = u.uHead.value.toArray();
    return { x: r.left + x / dpr, y: r.top + (H - y) / dpr, r: z / dpr };
  }

  // Backdrop layer: where the head is, in CSS px relative to the page.
  function setAnchor(a) {
    const r = canvas.getBoundingClientRect();
    anchor = { x: a.x - r.left, y: a.y - r.top, r: a.r };
  }

  // f: per-frame values from the animator.
  function render(f) {
    if (layer === 'backdrop') u.uHead.value.set(anchor.x * dpr, H - anchor.y * dpr, Math.max(1, anchor.r * dpr));
    const [cx, cy, R] = u.uHead.value.toArray();
    const showBackdrop = layer !== 'head';
    const orbsOn = showBackdrop && look.backdrop.orbs;
    ORBS.forEach((o, i) => {
      const a = Math.PI + o.p + f.orbit * (1 + i * 0.23);
      const depth = Math.sin(a);
      u.uOrbs.value[i].set(
        cx + Math.cos(a) * o.A * R,
        cy + (o.y + depth * 0.1 + Math.sin(f.time * 0.6 + i) * 0.015) * R,
        orbsOn ? o.r * R * (1 + 0.25 * depth) : 0,
        depth,
      );
    });
    u.uTime.value = f.time;
    u.uStep.value = Math.floor(f.time * 14);
    u.uGlitch.value = Math.min(1, f.glitch * look.glitch);
    u.uContrast.value = look.contrast;
    u.uBright.value = look.bright;
    u.uBand.value = showBackdrop && look.backdrop.band ? 1 : 0;
    u.uRing.value = showBackdrop && look.backdrop.ring ? 1 : 0;
    u.uSpecks.value = showBackdrop && look.backdrop.specks ? 1 : 0;
    u.uRipple.value = f.ripple;
    u.uScan.value = f.scan;
    u.uVoice.value = f.voice;
    // a full-page flash is harsh, so the backdrop layer gets half of it
    u.uFlash.value = layer === 'backdrop' ? f.flash * 0.5 : f.flash;

    renderer.setRenderTarget(sceneRT);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    if (layer !== 'backdrop') renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.render(postScene, postCam);
  }

  function dispose() {
    ro.disconnect();
    sceneRT.dispose();
    post.dispose();
    u.tGlyphs.value.dispose();
    quad.geometry.dispose();
    renderer.dispose();
    canvas.remove();
  }

  return { canvas, renderer, scene, camera, rig, layer, setLook, setModel, render, resize, dispose, headScreen, setAnchor };
}
