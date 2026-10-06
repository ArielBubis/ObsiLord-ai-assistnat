// Code skin: draws a frame description onto the 64x48 logical canvas from
// pixel grids defined in this file. No image assets.
//
// Skin interface: a skin module exports draw(ctx, frame), where frame is the
// object built by buildFrame() in idle.js:
//   { state, blink: "open"|"half"|"closed", glance: {dx, dy}, bobY,
//     mouth: 0..3, shutter: 0..1, dots: 0..3 }
// A skin only reads the frame; it holds no timers or state. A future
// skin_image.js would implement the same draw(ctx, frame) with drawImage
// calls from aligned PNG layers (eyes per state and blink phase, pupils,
// mouth frames), and face.js would import it instead of this module.
//
// Grid legend: "#" feature color, "o" tongue color, "." transparent.

import { FACE_CONFIG } from "./config.js";

const W = FACE_CONFIG.logicalWidth;
const H = FACE_CONFIG.logicalHeight;

const COLORS = {
  bg: "#0b1020",
  fg: "#8ef6e4",
  tongue: "#ff6b8b",
  rec: "#ff3b3b",
};

const EYE_LEFT = { cx: 20, cy: 18 };
const EYE_RIGHT = { cx: 44, cy: 18 };
const MOUTH_CX = 32;
const MOUTH_TOP = 33;

const EYE_NORMAL = [
  "..######..",
  ".########.",
  "##########",
  "##########",
  "##########",
  "##########",
  "##########",
  "##########",
  ".########.",
  "..######..",
];

// Listening: eyes wide.
const EYE_WIDE = [
  "...######...",
  ".##########.",
  ".##########.",
  "############",
  "############",
  "############",
  "############",
  "############",
  "############",
  ".##########.",
  ".##########.",
  "...######...",
];

// Looking: narrowed eyes.
const EYE_NARROW = [
  ".########.",
  "##########",
  "##########",
  ".########.",
];

const PUPIL = [
  "####",
  "####",
  "####",
  "####",
];

// Looking: a smaller pupil so the narrowed eye stays one piece.
const PUPIL_SMALL = [
  "##",
  "##",
];

const MOUTHS = [
  // 0 closed (slight smile)
  [
    ".#..........#.",
    "..##########..",
  ],
  // 1 small
  [
    "...########...",
    "..#........#..",
    "...########...",
  ],
  // 2 medium
  [
    "..##########..",
    ".#..........#.",
    ".#..........#.",
    ".#...oooo...#.",
    "..##########..",
  ],
  // 3 wide
  [
    ".############.",
    "#............#",
    "#............#",
    "#............#",
    "#...oooooo...#",
    "#..oooooooo..#",
    ".############.",
  ],
];

// Speaking with a closed mouth (a pause between words) uses a flat line
// instead of the smile, so speaking never looks identical to idle.
const MOUTH_SPEAKING_CLOSED = [
  "..##########..",
];

export const STATE_STYLE = {
  idle: { eye: EYE_NORMAL, pupilGrid: PUPIL, pupil: [0, 0], brows: null },
  speaking: { eye: EYE_NORMAL, pupilGrid: PUPIL, pupil: [0, 0], brows: null },
  listening: { eye: EYE_WIDE, pupilGrid: PUPIL, pupil: [0, 0], brows: { left: -4, right: -4 } },
  thinking: { eye: EYE_NORMAL, pupilGrid: PUPIL, pupil: [2, -1], brows: { left: -2, right: -4 } },
  looking: { eye: EYE_NARROW, pupilGrid: PUPIL_SMALL, pupil: [0, 0], brows: { left: -2, right: -2 } },
};

const PUPIL_MAX_OFFSET = 3;
const BROW_WIDTH = 8;

// Squash an eye grid for blinking. "half" keeps the top and bottom quarter
// rows so the rounded outline survives; "closed" is one line.
function blinkGrid(grid, blink) {
  if (blink === "open") return grid;
  const w = grid[0].length;
  if (blink === "closed") {
    return ["." + "#".repeat(w - 2) + "."];
  }
  const keep = Math.max(1, Math.floor(grid.length / 4));
  return grid.slice(0, keep).concat(grid.slice(grid.length - keep));
}

function drawGrid(ctx, grid, x0, y0) {
  for (let r = 0; r < grid.length; r++) {
    const row = grid[r];
    for (let c = 0; c < row.length; c++) {
      const ch = row[c];
      if (ch === ".") continue;
      ctx.fillStyle = ch === "o" ? COLORS.tongue : COLORS.fg;
      ctx.fillRect(x0 + c, y0 + r, 1, 1);
    }
  }
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// Pupil offset on one axis: state base plus glance, clamped so the pupil
// stays inside the eye. If clamping would swallow a glance entirely, the
// pupil still moves 1 px in the glance direction, so glances stay visible in
// states with an offset base (thinking).
export function pupilOffset(base, glance, max = PUPIL_MAX_OFFSET) {
  const v = clamp(base + glance, -max, max);
  if (glance !== 0 && v === base) {
    return clamp(base + Math.sign(glance), -max - 1, max + 1);
  }
  return v;
}

function drawEye(ctx, center, style, frame) {
  const grid = blinkGrid(style.eye, frame.blink);
  const w = grid[0].length;
  const h = grid.length;
  const x0 = center.cx - Math.floor(w / 2);
  const y0 = center.cy - Math.floor(h / 2) + frame.bobY;
  drawGrid(ctx, grid, x0, y0);
  if (frame.blink === "closed") return;

  const dx = pupilOffset(style.pupil[0], frame.glance.dx);
  const dy = pupilOffset(style.pupil[1], frame.glance.dy);
  const pupil = style.pupilGrid;
  const px0 = center.cx - Math.floor(pupil[0].length / 2) + dx;
  const py0 = center.cy - Math.floor(pupil.length / 2) + dy + frame.bobY;
  ctx.fillStyle = COLORS.bg;
  for (let r = 0; r < pupil.length; r++) {
    for (let c = 0; c < pupil[r].length; c++) {
      if (pupil[r][c] === ".") continue;
      // Only cut the pupil where the eye is drawn, so it never leaks out.
      const gx = px0 + c - x0;
      const gy = py0 + r - y0;
      if (gy < 0 || gy >= h || gx < 0 || gx >= w || grid[gy][gx] === ".") continue;
      ctx.fillRect(px0 + c, py0 + r, 1, 1);
    }
  }
}

function drawBrow(ctx, center, eyeGrid, lift, bobY) {
  const x0 = center.cx - BROW_WIDTH / 2;
  const y = center.cy - Math.floor(eyeGrid.length / 2) + lift + bobY;
  ctx.fillStyle = COLORS.fg;
  ctx.fillRect(x0, y, BROW_WIDTH, 1);
}

export function draw(ctx, frame) {
  const style = STATE_STYLE[frame.state] || STATE_STYLE.idle;

  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, W, H);

  drawEye(ctx, EYE_LEFT, style, frame);
  drawEye(ctx, EYE_RIGHT, style, frame);

  if (style.brows) {
    drawBrow(ctx, EYE_LEFT, style.eye, style.brows.left, frame.bobY);
    drawBrow(ctx, EYE_RIGHT, style.eye, style.brows.right, frame.bobY);
  }

  let mouth = MOUTHS[frame.mouth] || MOUTHS[0];
  if (frame.state === "speaking" && frame.mouth === 0) mouth = MOUTH_SPEAKING_CLOSED;
  drawGrid(ctx, mouth, MOUTH_CX - Math.floor(mouth[0].length / 2), MOUTH_TOP + frame.bobY);

  // Thinking dots rise up and to the right of the face.
  ctx.fillStyle = COLORS.fg;
  const dotPositions = [[52, 9], [56, 6], [60, 3]];
  for (let i = 0; i < frame.dots && i < dotPositions.length; i++) {
    const [x, y] = dotPositions[i];
    ctx.fillRect(x, y + frame.bobY, 2, 2);
  }

  if (frame.state === "looking") {
    // Camera-active marker, top left. Not bobbed: it is a status light.
    ctx.fillStyle = COLORS.rec;
    ctx.fillRect(2, 2, 3, 3);
  }

  if (frame.shutter > 0) {
    ctx.fillStyle = `rgba(255, 255, 255, ${frame.shutter})`;
    ctx.fillRect(0, 0, W, H);
  }
}
