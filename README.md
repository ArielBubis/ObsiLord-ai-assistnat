# Obsi-Lord

Desktop prototype of a voice assistant with an 8-bit animated face. Everything runs on one PC with a webcam. The camera only takes a photo during a conversation turn: when you ask ("what am I holding?") or when the assistant decides to look, limited by a cooldown and an hourly cap.

Status: working prototype. Text and voice chat, the animated face (pixel and 3D skins), look() with a local vision model, and an optional online lookup for a cut-out of the frame.

## How it works

- A browser page is the face, the microphone input, and the speaker output.
- A Python backend does speech-to-text, the LLM call, text-to-speech (Piper), and the webcam grab.
- The chat model is Gemini or a local model, picked in the page's Settings panel. Local means any server with an OpenAI-compatible API on your PC: LM Studio (default address `http://127.0.0.1:1234/v1`, turn on just-in-time model loading so the page can switch models) or Ollama, set in `[llm_local]`. The panel lists the server's models, and a note after each switch says what changes: with a local model the chat stays on the PC and costs nothing, but general knowledge is weaker, and a small model can miss a request to look. Models that LM Studio marks as tool-capable call the camera tools themselves, like Gemini; for models without tool support a short routing step picks the tool before each reply (`[llm_local] tool_mode` overrides the choice). Compare models with `python -m obsilord.llm.routecheck --model <id> --repeat 3` (add `--mode native` or `--mode router` to compare both ways).
- Camera frames stay on the PC by default: a small local vision model (Qwen3-VL-2B on llama.cpp, started only while it looks) describes the frame, and the chat model gets that text. Download it once with `python -m obsilord.vision.fetch` (add `--sample` for the photo the live tests use).
- Optional, off by default: web image search (`[image_search]` in config.toml) for things the local model cannot name, such as an album cover. The local model finds the thing in the frame, the app cuts it out, and only that cut-out is uploaded: to Gemini (the default, same API key as the chat) or to Google Cloud Vision web detection (needs its own key). This is the one feature that sends part of a camera frame off the PC. The assistant is told to use it only when you ask it to identify something, and it has hourly and monthly caps.
- The Snapshots switch on the page shows the frame the camera took and the cut-out that was sent to the search. The pictures are kept in memory only.
- They talk over a WebSocket bound to 127.0.0.1 only.

## Files

- planning/ (local only, gitignored): PLAN.md (phases with tasks and "done when" checks; work one phase at a time), ARCHITECTURE.md (components, WebSocket protocol, state machine, interfaces, config keys), and handoff specs.
- CLAUDE.md (local only, gitignored): conventions the coding agent must follow.
- .claude/agents/ (local only, gitignored): subagents (backend-dev, face-dev, reviewer).
- prompts/system.md: the assistant's system prompt. prompts/vision.md: instructions for the local vision model.
- config.example.toml, .env.example, requirements.in, .gitignore: setup files.

## Using this with a coding agent

1. Move this folder to your projects directory and run `git init`.
2. Open Claude Code in the folder.
3. Paste the kickoff prompt below.

Kickoff prompt:

```
Read CLAUDE.md, planning/PLAN.md and planning/ARCHITECTURE.md. Outline the plan for Phase 0 in a few lines and wait for my OK. Then do Phase 0 only. Use the backend-dev subagent for implementation and the reviewer subagent before committing. Look up current versions of all dependencies before pinning them.
```

For later phases: "Continue with Phase N. Same rules."

## Agent harness

A single agent that reads CLAUDE.md and planning/PLAN.md is enough for a solo prototype. The subagents are optional. They help mainly by keeping the main context clean and by giving you an independent review pass before each commit.

- Main session: orchestrator. Reads the plan, delegates, integrates, commits once per phase.
- backend-dev: everything under src/ and tests/.
- face-dev: everything under web/.
- reviewer: read-only. Runs after each phase, reports findings, edits nothing.

Delegation per phase:

- Phase 0: backend-dev
- Phase 1: face-dev
- Phase 2: backend-dev (LLM, sentence splitter) and face-dev (subtitles, fake mouth) in parallel, since the WebSocket protocol is already fixed in planning/ARCHITECTURE.md
- Phase 3 and 4: both, in parallel, same reason
- Phase 5: backend-dev, plus face-dev for the "looking" animation
- Phase 6 (nice to have): both
- Reviewer after every phase, before the commit
