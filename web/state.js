// Face state machine. Pure: no DOM, no timers.
// Any state may go to any other state. Unknown names are rejected.

export const STATES = Object.freeze(["idle", "listening", "thinking", "speaking", "looking"]);

export function isValidState(name) {
  return typeof name === "string" && STATES.includes(name);
}

// nowMs is injected so callers (and tests) control time.
export function createStateMachine(initial = "idle", nowMs = 0) {
  if (!isValidState(initial)) {
    throw new Error(`unknown face state: ${initial}`);
  }
  let current = initial;
  let enteredAt = nowMs;
  const listeners = new Set();

  return {
    get state() {
      return current;
    },
    get enteredAt() {
      return enteredAt;
    },
    // Returns true if the state changed, false if it was already current.
    // Throws on an unknown name.
    set(name, now = 0) {
      if (!isValidState(name)) {
        throw new Error(`unknown face state: ${name}`);
      }
      if (name === current) {
        return false;
      }
      const prev = current;
      current = name;
      enteredAt = now;
      for (const fn of listeners) {
        fn(current, prev);
      }
      return true;
    },
    // Returns an unsubscribe function.
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}
