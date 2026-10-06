import { test } from "node:test";
import assert from "node:assert/strict";

import { pupilOffset, STATE_STYLE } from "../../web/skin_code.js";
import { STATES } from "../../web/state.js";

const GLANCES = [-2, -1, 1, 2];

test("every state has a style", () => {
  for (const s of STATES) assert.ok(STATE_STYLE[s], s);
});

test("a nonzero glance always moves the pupil in every state", () => {
  for (const s of STATES) {
    const [bx, by] = STATE_STYLE[s].pupil;
    for (const g of GLANCES) {
      for (const [base, label] of [[bx, "x"], [by, "y"]]) {
        const v = pupilOffset(base, g);
        assert.notEqual(v, base, `${s} ${label} glance ${g}`);
        assert.equal(Math.sign(v - base), Math.sign(g), `${s} ${label} glance ${g} direction`);
        assert.ok(Math.abs(v) <= 3, `${s} ${label} glance ${g} offset ${v}`);
      }
    }
    assert.equal(pupilOffset(bx, 0), bx);
    assert.equal(pupilOffset(by, 0), by);
  }
});
