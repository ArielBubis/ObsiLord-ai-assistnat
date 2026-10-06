// Front-end constants. These mirror the [face] section of config.example.toml;
// keep them in sync by hand until the backend serves config to the page.

export const FACE_CONFIG = Object.freeze({
  logicalWidth: 64,
  logicalHeight: 48,
  // RMS cutoffs for mouth frames 1, 2, 3. Unused until Phase 3 (voice out).
  mouthThresholds: Object.freeze([0.05, 0.15, 0.3]),
});
