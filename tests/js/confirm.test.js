import { test } from "node:test";
import assert from "node:assert/strict";

import {
  cardFrom, nextCard, answerMessage, secondsLeft, initConfirm, DEFAULT_TIMEOUT_S, MAX_TIMEOUT_S,
} from "../../web/confirm.js";

const confirm = {
  type: "confirm", id: "c1", tool: "look", label: "Look", question: "Take a photo?", details: "front camera", timeout_s: 20,
};

test("a confirm message makes a card with a local deadline", () => {
  const card = cardFrom(confirm, 1000);
  assert.equal(card.id, "c1");
  assert.equal(card.question, "Take a photo?");
  assert.equal(card.details, "front camera");
  assert.equal(card.deadlineMs, 21000);
});

test("a bad or missing timeout falls back, a huge one is capped", () => {
  assert.equal(cardFrom({ ...confirm, timeout_s: undefined }, 0).timeoutS, DEFAULT_TIMEOUT_S);
  assert.equal(cardFrom({ ...confirm, timeout_s: -3 }, 0).timeoutS, DEFAULT_TIMEOUT_S);
  assert.equal(cardFrom({ ...confirm, timeout_s: 9999 }, 0).timeoutS, MAX_TIMEOUT_S);
  assert.equal(cardFrom({ type: "state" }, 0), null);
  assert.equal(cardFrom({ type: "confirm" }, 0), null);
});

test("the card shows on confirm and clears on confirm_done for its id", () => {
  let card = nextCard(null, confirm, 0);
  assert.equal(card.id, "c1");
  assert.equal(nextCard(card, { type: "confirm_done", id: "other", answer: "no" }, 0), card);
  assert.equal(nextCard(card, { type: "state", state: "thinking" }, 0), card);
  assert.equal(nextCard(card, { type: "confirm_done", id: "c1", answer: "yes" }, 0), null);
});

test("Yes and No answers", () => {
  assert.deepEqual(JSON.parse(answerMessage("c1", true)), { type: "confirm_answer", id: "c1", yes: true });
  assert.deepEqual(JSON.parse(answerMessage("c1", false)), { type: "confirm_answer", id: "c1", yes: false });
});

test("the countdown rounds up and stops at 0", () => {
  const card = cardFrom(confirm, 0);
  assert.equal(secondsLeft(card, 0), 20);
  assert.equal(secondsLeft(card, 500), 20);
  assert.equal(secondsLeft(card, 19001), 1);
  assert.equal(secondsLeft(card, 20000), 0);
  assert.equal(secondsLeft(card, 99999), 0);
  assert.equal(secondsLeft(null, 0), 0);
});

function fakeEl() {
  const mk = () => {
    const listeners = {};
    return { hidden: false, textContent: "", addEventListener: (t, fn) => { listeners[t] = fn; }, click: () => listeners.click() };
  };
  return { card: mk(), question: mk(), details: mk(), countdown: mk(), yes: mk(), no: mk() };
}

test("initConfirm shows, answers by button and clears on confirm_done", () => {
  const el = fakeEl();
  const sent = [];
  let t = 0;
  const c = initConfirm(el, (s) => sent.push(s), () => t);
  assert.equal(el.card.hidden, true);
  assert.equal(c.active(), false);
  el.yes.click();
  assert.deepEqual(sent, []);
  c.handle(confirm);
  assert.equal(el.card.hidden, false);
  assert.equal(c.active(), true);
  assert.equal(el.question.textContent, "Take a photo?");
  assert.equal(el.countdown.textContent, "20 s");
  t = 5000;
  el.no.click();
  el.yes.click();
  assert.deepEqual(sent.map((s) => JSON.parse(s).yes), [false, true]);
  assert.equal(c.handle({ type: "confirm_done", id: "c1", answer: "yes" }), true);
  assert.equal(el.card.hidden, true);
  assert.equal(c.active(), false);
  c.clear();
});

test("clear removes the card (disconnect)", () => {
  const el = fakeEl();
  const c = initConfirm(el, () => {}, () => 0);
  c.handle(confirm);
  c.clear();
  assert.equal(el.card.hidden, true);
  assert.equal(c.active(), false);
});
