// Turns any .glb or .vrm into something the face renderer can use: checks the
// file, converts geometry and materials, and frames the head at height 2,
// centred at the origin. Files exported by the creator (.face.glb) are already
// framed and skip this step.

import * as THREE from './vendor/three/build/three.module.min.js';

export class CharacterError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'CharacterError';
    this.code = code; // 'not-glb' | 'compressed' | 'not-found' | 'load-failed'
  }
}

const COMPRESSION = {
  KHR_draco_mesh_compression: 'Draco',
  EXT_meshopt_compression: 'meshopt',
  KHR_texture_basisu: 'KTX2',
};

// Reads the glTF header and JSON chunk without decoding the file.
export function inspectGlb(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 20 || view.getUint32(0, true) !== 0x46546c67) {
    throw new CharacterError('not-glb', 'This is not a binary glTF file. Use a .glb or .vrm file.');
  }
  if (view.getUint32(4, true) !== 2) throw new CharacterError('not-glb', 'Only glTF 2.0 files are supported.');
  const jsonLength = view.getUint32(12, true);
  if (view.getUint32(16, true) !== 0x4e4f534a || 20 + jsonLength > buffer.byteLength) {
    throw new CharacterError('not-glb', 'The file is damaged: its glTF header is incomplete.');
  }
  let json;
  try { json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLength))); }
  catch { throw new CharacterError('not-glb', 'The file is damaged: its glTF data is not readable.'); }
  const used = new Set([...(json.extensionsUsed || []), ...(json.extensionsRequired || [])]);
  const scene = json.scenes?.[json.scene ?? 0];
  return {
    compressed: Object.entries(COMPRESSION).filter(([ext]) => used.has(ext)).map(([, label]) => label),
    vrm: used.has('VRMC_vrm') ? '1' : used.has('VRM') ? '0' : null,
    aiFace: scene?.extras?.aiFace || null,
  };
}

// Plain float geometry (meshopt/quantized files decode to normalized ints or
// interleaved buffers) and lit materials, so export and the dot shader behave.
export function prepareModel(root) {
  const FLOAT_ATTRS = /^(position|normal|tangent|uv\d*|color)$/;
  root.traverse((o) => {
    if (!o.isMesh) return;
    const g = o.geometry;
    for (const [name, a] of Object.entries(g.attributes)) if (FLOAT_ATTRS.test(name)) g.setAttribute(name, toFloat(a));
    for (const [name, list] of Object.entries(g.morphAttributes)) {
      g.morphAttributes[name] = list.map((a) => { const f = toFloat(a); f.name = a.name; return f; });
    }
    o.material = [].concat(o.material).map(litMaterial);
    if (o.material.length === 1) o.material = o.material[0];
  });
}

function toFloat(a) {
  if (!a.isInterleavedBufferAttribute && a.array instanceof Float32Array && !a.normalized) return a;
  const k = a.itemSize, out = new Float32Array(a.count * k);
  const get = [a.getX, a.getY, a.getZ, a.getW];
  for (let i = 0; i < a.count; i++) for (let j = 0; j < k; j++) out[i * k + j] = get[j].call(a, i);
  return new THREE.BufferAttribute(out, k);
}

function litMaterial(m) {
  if (m.isMeshStandardMaterial) { m.metalness = 0; m.roughness = Math.max(0.6, m.roughness ?? 0.7); return m; }
  const s = new THREE.MeshStandardMaterial({
    name: m.name, color: m.color ?? 0xffffff, map: m.map ?? null,
    transparent: m.transparent, alphaTest: m.alphaTest, side: m.side, roughness: 0.75, metalness: 0,
  });
  m.dispose();
  return s;
}

// Wraps the model so it sits at head height 2, centred. Rigged characters with
// a body below the head are framed on the head itself: the vertices mostly
// skinned to the Head bone or its jaw/eye/face children. VRM 0.x models face
// -Z, so they are turned around.
export function normalizeModel(root, { turn = false } = {}) {
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root, true);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  let headHeight = size.y;
  let headBone = null;
  root.traverse((o) => { if (!headBone && o.isBone && /head$/i.test(o.name)) headBone = o; });
  if (headBone) {
    const headY = headBone.getWorldPosition(new THREE.Vector3()).y;
    if (headY - box.min.y > 0.6 * size.y) {
      const headBox = boxOfBone(root, headBone);
      if (!headBox.isEmpty()) {
        headBox.getCenter(center);
        headHeight = headBox.getSize(new THREE.Vector3()).y;
      }
    }
  }
  const wrapper = new THREE.Group();
  wrapper.name = 'aiFaceRoot';
  wrapper.scale.setScalar(2 / (headHeight || 1));
  if (turn) wrapper.rotation.y = Math.PI;
  root.position.sub(center);
  wrapper.add(root);
  return wrapper;
}

// Hair and accessory bones are left out so long hair does not stretch the box.
function boxOfBone(root, bone) {
  const owned = new Set([bone]);
  bone.traverse((o) => { if (o !== bone && /eye|jaw|teeth|tongue|face|brow|lid/i.test(o.name)) owned.add(o); });
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  root.traverse((o) => {
    if (!o.isMesh) return;
    if (!o.isSkinnedMesh) {
      let p = o.parent, under = false;
      while (p) { if (owned.has(p)) { under = true; break; } p = p.parent; }
      if (under) box.expandByObject(o, true);
      return;
    }
    const idx = o.geometry.attributes.skinIndex, wts = o.geometry.attributes.skinWeight;
    if (!idx || !wts) return;
    const bones = o.skeleton.bones;
    for (let i = 0; i < idx.count; i++) {
      let best = 0, bi = -1;
      for (let k = 0; k < 4; k++) { const w = wts.getComponent(i, k); if (w > best) { best = w; bi = idx.getComponent(i, k); } }
      if (bi < 0 || !owned.has(bones[bi])) continue;
      o.getVertexPosition(i, v);
      box.expandByPoint(v.applyMatrix4(o.matrixWorld));
    }
  });
  return box;
}

// Turn the face a quarter turn (exports can face any direction).
export function turnAround(object) {
  object.rotation.y = Math.round(object.rotation.y / (Math.PI / 2) + 1) % 4 * (Math.PI / 2);
}
