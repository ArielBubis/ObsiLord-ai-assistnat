# Obsi-Lord

Desktop prototype of a voice assistant with an 8-bit animated face. Everything runs on one PC with a webcam. The camera only takes a photo during a conversation turn: when you ask ("what am I holding?") or when the assistant decides to look, limited by a cooldown and an hourly cap.

Status: working prototype. Text and voice chat, the animated face (pixel and 3D skins), and look() with a local vision model.

## How it works

- A browser page is the face, the microphone input, and the speaker output.
- A Python backend does speech-to-text, the LLM call (Gemini), text-to-speech (Piper), and the webcam grab.
- Camera frames never leave the PC: a small local vision model (Qwen3-VL-2B on llama.cpp, started only while it looks) describes the frame, and Gemini gets that text. Download it once with `python -m obsilord.vision.fetch` (add `--sample` for the photo the live tests use).
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
