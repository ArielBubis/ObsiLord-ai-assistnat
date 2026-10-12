// ai-face: plays a character file as an animated dot-matrix head.
//
//   import { createFace } from './ai-face/ai-face.js';
//   const face = await createFace(container, { character: './ai-face/characters/default.face.glb' });
//   face.setState('thinking');      // idle | listening | thinking | speaking | looking
//   face.setMouth(0.6);             // 0..1 from speech audio, null for built-in babble
//   face.setExpression('happy');    // a pose saved in the character, null to clear
//
// Character files: .face.glb from the creator, or any plain .glb / .vrm (framed
// automatically, expressions detected by name). Compressed files (Draco, KTX2,
// meshopt) need the creator to convert them.
//
// The container must have a size; the canvas fills it and follows its resizes.
// Pass backdrop: { container } to draw the backdrop on a second, transparent
// canvas (for example a fixed full-page layer behind the page content), laid
// out around wherever the head is on screen.

import * as THREE from './vendor/three/build/three.module.min.js';
import { GLTFLoader } from './vendor/three/examples/jsm/loaders/GLTFLoader.js';
import { createRenderer } from './render.js';
import { createAnimator, detectRig, mergeCharacter, DEFAULT_CHARACTER, STATE_NAMES, EYE_PATTERNS } from './animate.js';
import { CharacterError, inspectGlb, prepareModel, normalizeModel, turnAround } from './normalize.js';

export {
  THREE, GLTFLoader, detectRig, mergeCharacter, DEFAULT_CHARACTER, STATE_NAMES, EYE_PATTERNS,
  CharacterError, inspectGlb, prepareModel, normalizeModel, turnAround,
};

// src: URL string, URL, File/Blob or ArrayBuffer.
// options.loader: a GLTFLoader with Draco/KTX2/meshopt decoders attached (the
// creator passes one); without it compressed files are rejected with a clear message.
// options.name: display name for plain models (defaults to "Unnamed").
export async function loadCharacterFile(src, options = {}) {
  let buffer;
  if (src instanceof ArrayBuffer) buffer = src;
  else if (src instanceof Blob) buffer = await src.arrayBuffer();
  else {
    let res;
    try { res = await fetch(String(src)); } catch { res = null; }
    if (!res || !res.ok) throw new CharacterError('not-found', `The character file could not be downloaded (${res ? res.status : 'network error'}).`);
    buffer = await res.arrayBuffer();
  }
  const info = inspectGlb(buffer);
  if (info.compressed.length && !options.loader) {
    throw new CharacterError('compressed', `This file uses ${info.compressed.join(' and ')} compression. Open it in the face creator and export a .face.glb.`);
  }
  let gltf;
  try { gltf = await (options.loader || new GLTFLoader()).parseAsync(buffer, ''); }
  catch (err) { throw new CharacterError('load-failed', `The file could not be read as a 3D model: ${err.message}`); }
  const settings = gltf.scene.userData?.aiFace || null;
  if (settings) return { object: gltf.scene, settings, info };
  prepareModel(gltf.scene);
  return {
    object: normalizeModel(gltf.scene, { turn: info.vrm === '0' }),
    settings: { name: options.name || 'Unnamed' },
    info,
  };
}

export async function createFace(container, options = {}) {
  const bgOpts = options.backdrop?.container ? options.backdrop : null;
  const view = createRenderer(container, { layer: bgOpts ? 'head' : 'full' });
  const bg = bgOpts ? createRenderer(bgOpts.container, { layer: 'backdrop', transparent: bgOpts.transparent ?? true }) : null;
  const anim = createAnimator();
  let character = mergeCharacter(DEFAULT_CHARACTER, options.settings || {});
  let model = null, rig = null;
  const maps = new Map(); // material -> its texture, for the skin toggle
  let mouth = null, lookAt = null, pointer = null;
  let follow = options.followPointer ?? true;
  let running = true, raf = 0, last = performance.now(), t = 0;
  let loadToken = 0;

  const onPointer = (e) => {
    const r = container.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const cl = (v) => Math.max(-1.5, Math.min(1.5, v));
    pointer = {
      x: cl((e.clientX - (r.left + r.width / 2)) / (r.width / 2)),
      y: cl(-(e.clientY - (r.top + r.height / 2)) / (r.height / 2)),
    };
  };
  window.addEventListener('pointermove', onPointer);

  function applySkin() {
    for (const [m, map] of maps) {
      const want = character.look.skin ? map : null;
      if (m.map !== want) { m.map = want; m.needsUpdate = true; }
    }
  }

  function applyLook() {
    view.setLook(character.look);
    bg?.setLook(character.look);
    applySkin();
  }

  // settings: the file's aiFace block (replaces the character), or null to keep
  // the current settings and only re-detect the rig (swapping faces in the creator).
  function setModel(object, settings = null) {
    anim.setRig(null);
    maps.clear();
    if (settings) character = mergeCharacter(DEFAULT_CHARACTER, settings);
    model = object;
    view.setModel(object);
    rig = object ? detectRig(object, settings ? character.rig : null) : null;
    if (rig) character.rig = rig.saved;
    object?.traverse((o) => {
      if (!o.isMesh) return;
      for (const m of [].concat(o.material)) if (m?.map && !maps.has(m)) maps.set(m, m.map);
    });
    anim.setRig(rig);
    applyLook();
  }

  function tick(now) {
    if (!running) return;
    raf = requestAnimationFrame(tick);
    const dt = Math.min(Math.max(0, (now - last) / 1000), 0.05);
    last = now;
    t += dt;
    const f = anim.update({ dt, t, character, pointer, follow, lookAt, mouth });
    view.rig.rotation.set(f.head.x, f.head.y, f.head.z);
    view.rig.position.y = f.head.bob;
    view.render(f);
    if (bg) {
      // Size the backdrop from the head, but keep the band and ring on the page.
      const h = view.headScreen();
      const pageW = bgOpts.container.clientWidth || window.innerWidth;
      const fit = Math.min((pageW - 16 - h.x) / 2.1, (h.x - 16) / 1.65);
      bg.setAnchor({ x: h.x, y: h.y, r: Math.max(h.r, Math.min(h.r * (bgOpts.scale ?? 3.6), fit)) });
      bg.render(f);
    }
  }

  const api = {
    THREE,
    get canvas() { return view.canvas; },
    get renderer() { return view.renderer; },
    get character() { return character; },
    get model() { return model; },
    get rig() { return rig; },
    get state() { return anim.state; },

    setModel,
    // Load a character file (see loadCharacterFile). If a newer load starts
    // before this one finishes, the older result is dropped.
    async load(src, opts = {}) {
      const token = ++loadToken;
      const { object, settings } = await loadCharacterFile(src, opts);
      if (token === loadToken) setModel(object, settings);
      return api;
    },
    // Merge a partial settings object (live edits).
    applyCharacter(patch) {
      character = mergeCharacter(character, patch);
      applyLook();
    },
    // Replace all settings, keeping the current model's rig.
    replaceCharacter(next) {
      character = mergeCharacter(DEFAULT_CHARACTER, next);
      if (rig) character.rig = rig.saved;
      applyLook();
    },
    setState(name) {
      if (!STATE_NAMES.includes(name)) { console.warn('ai-face: unknown state', name); return false; }
      anim.setState(name, t);
      return true;
    },
    setMouth(level) { mouth = level == null ? null : Number(level) || 0; },
    setExpression(nameOrMap, opts = {}) { anim.setExpression(nameOrMap, character, opts.fadeMs ?? 300); },
    lookAt(x, y = 0) { lookAt = x == null ? null : { x, y }; },
    followPointer(on) { follow = !!on; },
    blink() { anim.blink(); },
    glance() { anim.glance(); },
    flash(amount = 1) { anim.flash(amount); },
    // Neutral pose with textures restored, for exporting. The next frame animates again;
    // call applyCharacter({}) afterwards to re-apply the skin toggle.
    resetPose() {
      anim.resetPose();
      for (const [m, map] of maps) { m.map = map; m.needsUpdate = true; }
      view.rig.rotation.set(0, 0, 0);
      view.rig.position.set(0, 0, 0);
    },
    resize() { view.resize(); bg?.resize(); },
    dispose() {
      running = false;
      cancelAnimationFrame(raf);
      window.removeEventListener('pointermove', onPointer);
      anim.setRig(null);
      view.dispose();
      bg?.dispose();
    },
  };

  applyLook();
  if (options.state) api.setState(options.state);
  raf = requestAnimationFrame(tick);
  if (options.character) await api.load(options.character, { name: options.name });
  return api;
}
