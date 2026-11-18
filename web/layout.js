// Three-column page: Settings (left), the card (middle), Snapshots (right).
// face.js asks how wide the face may be once the side panels are counted.
//
// The numbers below are mirrored in web/style.css (the body grid columns, the
// column gap, and the (width < 1300px) stacking query). SIDE_PANEL_PX is the
// panels' minimum width (they grow to 640px in CSS when the window is wide).
// They are constants, not measured widths, so the face size depends on the
// window only and never on how wide the panels happen to render. Change them
// in both places together.
export const SIDE_PANEL_PX = 360;
export const COLUMN_GAP_PX = 16;
export const WIDE_MIN_PX = 1300;

// CSS px the face may fill. `sides` is the width (plus gaps) taken by the side
// panels beside the card: 0 when they are stacked below it or hidden.
export function faceWidth(innerWidth, sides, reserveX) {
  return Math.max(0, innerWidth - reserveX - sides);
}

// Width the two side panels and their gaps take from the window. Wide windows
// (at least `breakpoint` px) put both panels beside the card; narrower ones
// stack them below it, so nothing is taken. In demo mode the panels are hidden,
// so nothing is taken at any width.
export function sideReserve(
  innerWidth,
  panelWidth = SIDE_PANEL_PX,
  gap = COLUMN_GAP_PX,
  breakpoint = WIDE_MIN_PX,
  demo = false,
) {
  if (demo) return 0;
  return innerWidth >= breakpoint ? 2 * (panelWidth + gap) : 0;
}
