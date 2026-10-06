import { test } from "node:test";
import assert from "node:assert/strict";

import { STATES, isValidState, createStateMachine } from "../../web/state.js";

test("five states", () => {
  assert.deepEqual([...STATES], ["idle", "listening", "thinking", "speaking", "looking"]);
});

test("starts idle by default", () => {
  const m = createStateMachine();
  assert.equal(m.state, "idle");
});

test("any state can go to any other state", () => {
  for (const from of STATES) {
    for (const to of STATES) {
      if (from === to) continue;
      const m = createStateMachine(from);
      assert.equal(m.set(to, 5), true);
      assert.equal(m.state, to);
      assert.equal(m.enteredAt, 5);
    }
  }
});

test("setting the current state is a no-op", () => {
  const m = createStateMachine("thinking", 10);
  let calls = 0;
  m.subscribe(() => calls++);
  assert.equal(m.set("thinking", 99), false);
  assert.equal(m.enteredAt, 10);
  assert.equal(calls, 0);
});

test("unknown names are rejected and state is unchanged", () => {
  const m = createStateMachine("speaking");
  for (const bad of ["sleeping", "", "IDLE", null, undefined, 3, {}]) {
    assert.throws(() => m.set(bad), /unknown face state/);
    assert.equal(m.state, "speaking");
    assert.equal(isValidState(bad), false);
  }
  assert.throws(() => createStateMachine("nope"), /unknown face state/);
});

test("listeners get new and previous state, and can unsubscribe", () => {
  const m = createStateMachine();
  const seen = [];
  const off = m.subscribe((cur, prev) => seen.push([prev, cur]));
  m.set("listening");
  m.set("looking");
  off();
  m.set("idle");
  assert.deepEqual(seen, [["idle", "listening"], ["listening", "looking"]]);
});
