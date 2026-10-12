// obsi-lord skin: draws the assistant with ai-face (a 3D head redrawn as dots)
// instead of the 64x48 code skin. It has the same interface as skin_code.js:
//
//   draw(ctx, frame)
//   frame = { state, blink: "open"|"half"|"closed", glance: {dx, dy}, bobY,
//             mouth: 0..3, shutter: 0..1, dots: 0..3 }
//
// To use it, change one import in web/face.js:
//   import * as skin from "./ai-face/obsi-lord-skin.js";
//
// face.js calls draw() only when the frame changes. The 3D face runs its own
// animation loop (idle motion, blinks, glitch), so draw() just forwards the
// latest state, mouth and shutter values.
//
// The character file comes from the backend: obsi-lord serves the file named
// in config.toml ([face] character) at options.character. When that route is
// missing or the file fails, the bundled character is shown instead. If WebGL
// fails too, the module falls back to ../skin_code.js.

import { createFace } from "./ai-face.js";

export const options = {
  // The configured character, served by the backend.
  character: "/face/character.glb",
  // Shown when the configured character is missing or does not load.
  fallbackCharacter: new URL("./characters/default.face.glb", import.meta.url).href,
  // "box": head and backdrop together inside the face box. "off": head only.
  // "page": head in the face box, backdrop on a full-page layer behind the HUD.
  backdrop: "box",
  // Backdrop size relative to the head on screen (page mode only).
  backdropScale: 3.6,
  // false: blinks and glances come from the character's eye settings.
  // true: follow obsi-lord's idle loop (frame.blink and frame.glance) instead.
  useFrameEyes: false,
  // Eyes follow the mouse pointer when true.
  followPointer: false,
};

let face = null;
let fallback = null;
let mounting = false;
let lastFrame = null;
let lastBlink = "open";

export function draw(ctx, frame) {
  lastFrame = frame;
  if (fallback) { fallback.draw(ctx, frame); return; }
  if (face) { apply(frame); return; }
  if (!mounting) mount(ctx);
}

export function getFace() { return face; }

// Show another character file (a URL). Rejects if it does not load; the
// current character stays on screen in that case.
export async function setCharacter(url) {
  if (!face) throw new Error("The face is not ready yet.");
  await face.load(url);
  adapt();
}

// Settings the adapter owns, re-applied after every character load.
function adapt() {
  // obsi-lord drives the shutter itself, so the character's own flash is off.
  face.applyCharacter({ look: { flashOnLook: false } });
  if (options.backdrop === "off") {
    face.applyCharacter({ look: { backdrop: { band: false, orbs: false, ring: false, specks: false } } });
  }
  if (options.useFrameEyes) {
    face.applyCharacter({ eyes: { blink: { min: 1e6, max: 1e6, double: 0 }, glance: { amount: 0 } } });
  }
}

async function mount(ctx) {
  mounting = true;
  const canvas = ctx.canvas;
  const host = canvas.parentElement;
  if (getComputedStyle(host).position === "static") host.style.position = "relative";
  const box = document.createElement("div");
  box.className = "ai-face";
  box.style.cssText = "position:absolute;inset:0;";
  host.append(box);
  let backdropBox = null;
  if (options.backdrop === "page") {
    backdropBox = document.createElement("div");
    backdropBox.className = "ai-face-backdrop";
    backdropBox.style.cssText = "position:fixed;inset:0;z-index:-1;pointer-events:none;";
    document.body.prepend(backdropBox);
  }
  // The 2D canvas stays in the layout so face.js keeps sizing the frame.
  canvas.style.visibility = "hidden";

  try {
    face = await createFace(box, {
      followPointer: options.followPointer,
      backdrop: backdropBox ? { container: backdropBox, scale: options.backdropScale } : null,
    });
    try {
      await setCharacter(options.character);
    } catch (err) {
      if (err.code !== "not-found") console.error("ai-face: the configured character did not load, showing the bundled one.", err.message);
      await setCharacter(options.fallbackCharacter);
    }
    if (lastFrame) apply(lastFrame);
  } catch (err) {
    console.error("ai-face: the 3D face did not start, using the code skin.", err);
    face?.dispose();
    face = null;
    box.remove();
    backdropBox?.remove();
    canvas.style.visibility = "";
    try {
      fallback = await import("../skin_code.js");
      if (lastFrame) fallback.draw(ctx, lastFrame);
    } catch {
      console.error("ai-face: no fallback skin found next to the ai-face folder");
    }
  }
}

function apply(frame) {
  if (face.state !== frame.state) face.setState(frame.state);
  face.setMouth(Math.max(0, Math.min(3, frame.mouth || 0)) / 3);
  if (frame.shutter > 0) face.flash(frame.shutter);
  if (options.useFrameEyes) {
    if (frame.blink !== "open" && lastBlink === "open") face.blink();
    lastBlink = frame.blink;
    const g = frame.glance || { dx: 0, dy: 0 };
    face.lookAt(g.dx || g.dy ? g.dx * 0.15 : null, -(g.dy || 0) * 0.15);
  }
}
