// Animation: character settings, rig detection (which morphs and eye nodes a
// model has), and the per-frame animator (states, eyes, mouth, expressions).

export const STATE_NAMES = ['idle', 'listening', 'thinking', 'speaking', 'looking'];
export const EYE_PATTERNS = ['steady', 'drift', 'dart', 'scan'];

// Everything a character file stores besides the mesh. Missing keys fall back
// to these values, so older files keep working when new settings are added.
export const DEFAULT_CHARACTER = {
  version: 1,
  name: 'Unnamed',
  look: {
    mode: 'dither',
    dot: 3,
    contrast: 1.45,
    bright: 0.04,
    ink: '#ededed',
    glitch: 1,
    skin: true,
    light: { azimuth: -50, elevation: 35, key: 2.7, rim: 2.4, fill: 0.15, ambient: 0.12 },
    backdrop: { band: true, orbs: true, ring: true, specks: true },
    framing: { zoom: 1, lift: 0.28 },
    flashOnLook: true,
  },
  personality: {
    resting: { smile: 0.05, brow: 0, open: 0 },
    mouth: { open: 0.45, round: 0.35 },
    states: {
      idle:      { glitch: .015, bursts: .25, tilt: 0,    motion: 1,   brow: 0,    smile: 0,    open: 0,    scan: 0,  ripple: 0, orbit: 0 },
      listening: { glitch: .01,  bursts: .05, tilt: .07,  motion: .6,  brow: .35,  smile: .07,  open: .45,  scan: 0,  ripple: 1, orbit: .08 },
      thinking:  { glitch: .3,   bursts: 1.4, tilt: -.04, motion: .5,  brow: -.35, smile: -.05, open: -.3,  scan: 1,  ripple: 0, orbit: .9 },
      speaking:  { glitch: .05,  bursts: .5,  tilt: 0,    motion: 1.2, brow: .12,  smile: .13,  open: .1,   scan: 0,  ripple: 0, orbit: .15 },
      looking:   { glitch: .02,  bursts: .1,  tilt: 0,    motion: .2,  brow: -.2,  smile: 0,    open: -.45, scan: .5, ripple: 0, orbit: 0 },
    },
  },
  eyes: {
    follow: 0.7,
    headFollow: 0.32,
    speed: 18,
    blink: { min: 2, max: 6.5, duration: 0.16, double: 0.15 },
    glance: { min: 3, max: 8, amount: 0.25, hold: 0.7 },
    states: {
      idle:      { x: 0,    y: 0,   pattern: 'drift',  follow: 1 },
      listening: { x: 0,    y: .05, pattern: 'steady', follow: 1 },
      thinking:  { x: -.55, y: .5,  pattern: 'drift',  follow: 0 },
      speaking:  { x: 0,    y: 0,   pattern: 'dart',   follow: .6 },
      looking:   { x: 0,    y: 0,   pattern: 'scan',   follow: 0 },
    },
  },
  // Named poses: morph name -> weight, blended on top of the state.
  expressions: {
    happy: { mouthSmile_L: .7, mouthSmile_R: .7, cheekSquint_L: .45, cheekSquint_R: .45, eyeSquint_L: .25, eyeSquint_R: .25, browInnerUp: .15 },
    sad: { mouthFrown_L: .6, mouthFrown_R: .6, browInnerUp: .7, browDown_L: .2, browDown_R: .2, mouthPucker: .2, eyeLookDown_L: .2, eyeLookDown_R: .2 },
    surprised: { eyeWide_L: .8, eyeWide_R: .8, browInnerUp: .8, browOuterUp_L: .7, browOuterUp_R: .7, jawOpen: .35, mouthFunnel: .3 },
  },
  // Filled from the model on load: which morphs drive each channel, which nodes are eyes.
  rig: { channels: null, eyes: null },
};

const isObj = (v) => v && typeof v === 'object' && !Array.isArray(v);

// Deep merge for plain objects. Arrays and primitives replace. Expressions are
// replaced wholesale when given, so deleting one in the creator sticks.
export function mergeCharacter(base, patch) {
  const out = structuredClone(base);
  (function merge(dst, src, path) {
    for (const [k, v] of Object.entries(src || {})) {
      if (path === '' && k === 'expressions' && isObj(v)) { dst[k] = structuredClone(v); continue; }
      if (isObj(v) && isObj(dst[k])) merge(dst[k], v, path + '.' + k);
      else dst[k] = structuredClone(v);
    }
  })(out, patch, '');
  return out;
}

// Channel -> name patterns, tried in order. ARKit first, then Oculus visemes,
// VRoid (VRM) and common generic names.
const CHANNEL_PATTERNS = {
  blink:     [/^eyeBlink/i, /^fcl_eye_close$/i, /^blink$/i, /^eyes?_?closed?$/i],
  wide:      [/^eyeWide/i, /^fcl_eye_surprised$/i],
  squint:    [/^eyeSquint/i],
  browUp:    [/^browInnerUp/i, /^fcl_brw_surprised$/i, /^brows?_?up$/i],
  browOuter: [/^browOuterUp/i],
  browDown:  [/^browDown/i, /^fcl_brw_angry$/i],
  smile:     [/^mouthSmile/i, /^fcl_mth_fun$/i, /^smile$/i, /^happy$/i],
  frown:     [/^mouthFrown/i, /^fcl_mth_angry$/i, /^sad$/i],
  jaw:       [/^jawOpen$/i, /^mouthOpen$/i, /^viseme_aa$/i, /^fcl_mth_a$/i, /^aa$/i],
  funnel:    [/^mouthFunnel$/i, /^viseme_O$/i, /^fcl_mth_o$/i, /^oh?$/i],
  lookUp:    [/^eyeLookUp/i],
  lookDown:  [/^eyeLookDown/i],
  lookInL:   [/^eyeLookIn_?L/i],
  lookOutL:  [/^eyeLookOut_?L/i],
  lookInR:   [/^eyeLookIn_?R/i],
  lookOutR:  [/^eyeLookOut_?R/i],
};
const EYE_NODE_PATTERNS = {
  left:  [/^grp_eyeLeft$/i, /^(mixamorig:)?LeftEye$/i, /^J_Adj_L_FaceEye$/i, /^eye[._ ]?l(eft)?$/i, /^eye_?left$/i],
  right: [/^grp_eyeRight$/i, /^(mixamorig:)?RightEye$/i, /^J_Adj_R_FaceEye$/i, /^eye[._ ]?r(ight)?$/i, /^eye_?right$/i],
};
// "Face_Blendshape.Fcl_EYE_Close" or "blendShape1.jawOpen" -> the part after the last separator.
const baseName = (n) => n.replace(/^.*[.|/]/, '');
const CHANNEL_LABELS = { blink: 'blink', jaw: 'jaw', smile: 'smile', frown: 'frown', browUp: 'brows', wide: 'wide eyes', squint: 'squint', funnel: 'round mouth' };

// saved: character.rig from a file (channel names and eye node names), or null to detect.
export function detectRig(object, saved = null) {
  const meshes = [];
  const names = new Set();
  const nodes = new Map();
  object.traverse((o) => {
    if (o.name) nodes.set(o.name, o);
    if (o.isMesh && o.morphTargetDictionary && o.morphTargetInfluences) {
      const entries = Object.entries(o.morphTargetDictionary);
      if (!entries.length) return;
      meshes.push({ mesh: o, entries });
      entries.forEach(([n]) => names.add(n));
    }
  });
  const morphNames = [...names].sort();
  // expressions saved with bare ARKit names also reach prefixed morphs
  const alias = {};
  for (const n of morphNames) (alias[baseName(n)] ||= []).push(n);

  const channels = {};
  for (const [ch, patterns] of Object.entries(CHANNEL_PATTERNS)) {
    const fromFile = saved?.channels?.[ch]?.filter((n) => names.has(n));
    if (fromFile?.length) { channels[ch] = fromFile; continue; }
    for (const p of patterns) {
      const hit = morphNames.filter((n) => p.test(baseName(n)));
      if (hit.length) { channels[ch] = hit; break; }
    }
  }

  const findEye = (side) => {
    const byName = saved?.eyes?.[side] && nodes.get(saved.eyes[side]);
    if (byName) return byName;
    for (const p of EYE_NODE_PATTERNS[side]) for (const [n, o] of nodes) if (p.test(n)) return o;
    return null;
  };
  const eyes = { left: findEye('left'), right: findEye('right') };
  const eyeBase = new Map();
  for (const e of [eyes.left, eyes.right]) if (e) eyeBase.set(e, e.rotation.clone());

  const found = Object.keys(CHANNEL_LABELS).filter((c) => channels[c]).map((c) => CHANNEL_LABELS[c]);
  return {
    meshes, morphNames, channels, eyes, eyeBase, found, alias,
    eyesTracked: !!(eyes.left || eyes.right || channels.lookInL),
    saved: {
      channels,
      eyes: { left: eyes.left?.name || null, right: eyes.right?.name || null },
    },
  };
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const rand = (a, b) => a + Math.random() * (b - a);

export function createAnimator() {
  let rig = null;
  let state = 'idle';
  let stateEnteredAt = 0;
  const cur = {};
  const look = { x: 0, y: 0 };
  let headYaw = 0, burst = 0, env = 0, orbit = 0, flash = 0;
  let blinkAt = 1.2, blinkStart = -1, secondBlinkAt = -1, forcedBlink = false;
  let glanceAt = 4, glanceUntil = -1, forcedGlance = false;
  const glanceOff = { x: 0, y: 0 };
  let dartAt = 0;
  const dartOff = { x: 0, y: 0 };
  let wordOn = true, wordFlipAt = 0;
  const expr = { current: {}, target: {}, rate: 10 };
  const weights = {};

  function setRig(next) {
    if (rig) resetPose();
    rig = next;
  }

  function resetPose() {
    if (!rig) return;
    for (const { mesh } of rig.meshes) mesh.morphTargetInfluences.fill(0);
    for (const [e, base] of rig.eyeBase) e.rotation.copy(base);
  }

  function setState(name, t) {
    if (name === state) return;
    state = name;
    stateEnteredAt = t;
  }

  function setExpression(nameOrMap, character, fadeMs = 300) {
    const map = typeof nameOrMap === 'string' ? character.expressions[nameOrMap] : nameOrMap;
    expr.target = { ...(map || {}) };
    expr.rate = 3000 / Math.max(30, fadeMs);
  }

  function add(channel, v) {
    if (!rig || v <= 0) return;
    const list = rig.channels[channel];
    if (list) for (const n of list) weights[n] = (weights[n] || 0) + v;
  }

  // input: { dt, t, character, pointer: {x,y}|null, follow: bool, lookAt: {x,y}|null, mouth: 0..1|null }
  function update(input) {
    const { dt, t, character: C } = input;
    const P = C.personality, E = C.eyes;
    const target = P.states[state] || P.states.idle;
    const ease = 1 - Math.exp(-dt * 5);
    for (const k in target) cur[k] = (cur[k] ?? target[k]) + (target[k] - (cur[k] ?? target[k])) * ease;
    const R = P.resting;

    // glitch bursts
    if (Math.random() < target.bursts * dt) burst = 0.5 + Math.random() * 0.6;
    burst *= Math.exp(-dt * 7);
    flash *= Math.exp(-dt * 7);

    // mouth: external level (speech audio) or babble with short word gaps
    let mouthTarget = 0;
    if (input.mouth != null) mouthTarget = clamp01(input.mouth);
    else if (state === 'speaking') {
      if (t > wordFlipAt) { wordOn = !wordOn; wordFlipAt = t + (wordOn ? rand(0.18, 0.55) : rand(0.05, 0.12)); }
      const syl = 0.5 + 0.5 * Math.sin(t * 15 + Math.sin(t * 2.3) * 3);
      mouthTarget = wordOn ? 0.2 + 0.8 * syl * syl : 0;
    }
    env += (mouthTarget - env) * (1 - Math.exp(-dt * (mouthTarget > env ? 30 : 14)));

    // eyes: state target + pattern + glance + pointer, or an explicit lookAt
    const ES = E.states[state] || E.states.idle;
    let gx = ES.x, gy = ES.y;
    if (ES.pattern === 'drift') {
      gx += Math.sin(t * 0.31) * 0.06 + Math.sin(t * 0.73 + 1.3) * 0.03;
      gy += Math.sin(t * 0.27 + 2) * 0.04 + Math.sin(t * 0.61) * 0.02;
    } else if (ES.pattern === 'dart') {
      if (t > dartAt) { dartOff.x = rand(-0.18, 0.18); dartOff.y = rand(-0.08, 0.1); dartAt = t + rand(0.6, 1.6); }
      gx += dartOff.x; gy += dartOff.y;
    } else if (ES.pattern === 'scan') {
      gx += Math.sin(t * 2.4) * 0.45;
    }
    const glances = ES.pattern === 'drift' || ES.pattern === 'dart';
    if (t > glanceAt) {
      if ((glances || forcedGlance) && E.glance.amount > 0) {
        const a = Math.random() * Math.PI * 2;
        glanceOff.x = Math.cos(a) * E.glance.amount;
        glanceOff.y = Math.sin(a) * E.glance.amount * 0.6;
        glanceUntil = t + E.glance.hold;
      }
      glanceAt = t + rand(E.glance.min, Math.max(E.glance.min, E.glance.max));
      forcedGlance = false;
    }
    if (t < glanceUntil) { gx += glanceOff.x; gy += glanceOff.y; }
    if (input.follow && input.pointer) {
      gx += input.pointer.x * E.follow * ES.follow;
      gy += input.pointer.y * E.follow * ES.follow;
    }
    if (input.lookAt) { gx = input.lookAt.x; gy = input.lookAt.y; }
    const gEase = 1 - Math.exp(-dt * E.speed);
    look.x += (gx - look.x) * gEase;
    look.y += (gy - look.y) * gEase;

    // head follows the gaze, plus idle sway scaled by motion
    const m = cur.motion;
    headYaw += ((look.x * E.headFollow + Math.sin(t * 0.37) * 0.04 * m) - headYaw) * (1 - Math.exp(-dt * 3));
    const head = {
      x: -look.y * 0.16 + Math.sin(t * 0.8) * 0.012 * m,
      y: headYaw,
      z: cur.tilt + Math.sin(t * 0.29) * 0.02 * m,
      bob: Math.sin(t * 1.1) * 0.012 * m,
    };

    // blink (with an optional quick second blink)
    if (t > blinkAt || forcedBlink) {
      blinkStart = t;
      blinkAt = t + rand(E.blink.min, Math.max(E.blink.min, E.blink.max));
      secondBlinkAt = !forcedBlink && Math.random() < E.blink.double ? t + E.blink.duration + 0.08 : -1;
      forcedBlink = false;
    }
    if (secondBlinkAt > 0 && t > secondBlinkAt) { blinkStart = t; secondBlinkAt = -1; }
    const bt = (t - blinkStart) / Math.max(0.05, E.blink.duration);
    const blink = bt >= 0 && bt < 1 ? Math.sin(bt * Math.PI) : 0;

    // looking: shutter flash on entry
    if (state === 'looking' && C.look.flashOnLook && t - stateEnteredAt < dt * 1.5) flash = 1;

    if (rig) {
      for (const k in weights) delete weights[k];
      const open = cur.open + R.open, brow = cur.brow + R.brow, smile = cur.smile + R.smile;
      add('blink', Math.max(blink, Math.max(0, -open) * 0.25));
      add('wide', Math.max(0, open) * (1 - blink));
      add('squint', Math.max(0, -open));
      add('browUp', Math.max(0, brow));
      add('browOuter', Math.max(0, brow) * 0.6);
      add('browDown', Math.max(0, -brow));
      add('smile', Math.max(0, smile));
      add('frown', Math.max(0, -smile));
      add('jaw', env * P.mouth.open);
      add('funnel', env * (0.5 + 0.5 * Math.sin(t * 3.7)) * P.mouth.round);
      const hasEyeNodes = rig.eyeBase.size > 0;
      const lidFollow = hasEyeNodes ? 0.5 : 1;
      add('lookUp', Math.max(0, look.y) * lidFollow);
      add('lookDown', Math.max(0, -look.y) * lidFollow);
      if (!hasEyeNodes) {
        add('lookOutL', Math.max(0, look.x)); add('lookInR', Math.max(0, look.x));
        add('lookInL', Math.max(0, -look.x)); add('lookOutR', Math.max(0, -look.x));
      }
      // expressions crossfade, then add raw morph weights
      const k = 1 - Math.exp(-dt * expr.rate);
      for (const n of new Set([...Object.keys(expr.current), ...Object.keys(expr.target)])) {
        const v = (expr.current[n] || 0) + ((expr.target[n] || 0) - (expr.current[n] || 0)) * k;
        if (Math.abs(v) < 1e-3 && !expr.target[n]) delete expr.current[n];
        else expr.current[n] = v;
      }
      for (const [n, v] of Object.entries(expr.current)) {
        if (v <= 0) continue;
        const targets = rig.alias[n] && !rig.morphNames.includes(n) ? rig.alias[n] : [n];
        for (const m of targets) weights[m] = (weights[m] || 0) + v;
      }

      for (const { mesh, entries } of rig.meshes) {
        const inf = mesh.morphTargetInfluences;
        inf.fill(0);
        for (const [n, i] of entries) { const w = weights[n]; if (w) inf[i] = clamp01(w); }
      }
      for (const [e, base] of rig.eyeBase) e.rotation.set(base.x - look.y * 0.3, base.y + look.x * 0.4, base.z);
    }

    orbit += dt * cur.orbit;
    return {
      time: t,
      head,
      glitch: cur.glitch + burst * 0.5,
      ripple: cur.ripple,
      scan: cur.scan,
      voice: state === 'speaking' ? env : 0,
      orbit,
      flash: Math.min(1, flash),
    };
  }

  return {
    setRig, resetPose, setState, setExpression, update,
    get state() { return state; },
    blink() { forcedBlink = true; },
    glance() { glanceAt = 0; forcedGlance = true; },
    flash(a = 1) { flash = Math.max(flash, a); },
  };
}
