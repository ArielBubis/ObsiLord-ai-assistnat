// The confirm card: the server asks before a tool in Ask mode runs. The card
// shows the question and details, Yes and No buttons and a countdown. The
// server decides (click, typed or spoken answer, or timeout) and ends it with
// confirm_done, which clears the card. The turn keeps running meanwhile.
//
// The pure helpers at the top are tested with node --test; the DOM code at
// the bottom is thin.

export const DEFAULT_TIMEOUT_S = 20;
export const MAX_TIMEOUT_S = 120;
const MAX_TEXT = 600;

function clip(value) {
  return typeof value === "string" ? value.slice(0, MAX_TEXT) : "";
}

// Card state from a server confirm message, or null. The deadline is local
// (now + timeout_s), only for the countdown.
export function cardFrom(msg, nowMs) {
  if (!msg || typeof msg !== "object" || msg.type !== "confirm") return null;
  if (typeof msg.id !== "string" && !Number.isFinite(msg.id)) return null;
  const t = Number(msg.timeout_s);
  const timeoutS = Number.isFinite(t) && t > 0 ? Math.min(t, MAX_TIMEOUT_S) : DEFAULT_TIMEOUT_S;
  return {
    id: msg.id,
    tool: clip(msg.tool),
    label: clip(msg.label),
    question: clip(msg.question),
    details: clip(msg.details),
    timeoutS,
    deadlineMs: nowMs + timeoutS * 1000,
  };
}

// The card after a server message: a confirm shows (replacing any older one),
// a confirm_done for the shown id clears it, anything else changes nothing.
export function nextCard(card, msg, nowMs) {
  if (!msg || typeof msg !== "object") return card;
  if (msg.type === "confirm") return cardFrom(msg, nowMs) || card;
  if (msg.type === "confirm_done" && card && msg.id === card.id) return null;
  return card;
}

export function answerMessage(id, yes) {
  return JSON.stringify({ type: "confirm_answer", id, yes: Boolean(yes) });
}

// Whole seconds left, never below 0.
export function secondsLeft(card, nowMs) {
  if (!card) return 0;
  return Math.max(0, Math.ceil((card.deadlineMs - nowMs) / 1000));
}

// el: { card, question, details, countdown, yes, no }. send(text) sends if
// the socket is open. now() is in ms. Returns { handle, clear, active }.
export function initConfirm(el, send, now = () => Date.now()) {
  let card = null;
  let timer = null;

  function tick() {
    if (card) el.countdown.textContent = `${secondsLeft(card, now())} s`;
  }

  function render() {
    el.card.hidden = card === null;
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
    if (!card) return;
    el.question.textContent = card.question || card.label;
    el.details.textContent = card.details;
    el.details.hidden = !card.details;
    tick();
    timer = setInterval(tick, 250);
  }

  function answer(yes) {
    if (card) send(answerMessage(card.id, yes));
  }
  el.yes.addEventListener("click", () => answer(true));
  el.no.addEventListener("click", () => answer(false));
  render();

  return {
    // A server message; true if the card changed.
    handle(msg) {
      const next = nextCard(card, msg, now());
      if (next === card) return false;
      card = next;
      render();
      return true;
    },
    clear() {
      card = null;
      render();
    },
    active() {
      return card !== null;
    },
  };
}
