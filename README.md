# Pixel Buddy

Desktop prototype of a voice assistant with an 8-bit animated face. Everything runs on one PC with a webcam. The camera only fires on command ("what am I holding?").

Status: planning. No code yet.

## How it works

- A browser page is the face, the microphone input, and the speaker output.
- A Python backend does speech-to-text, the LLM call (Gemini), text-to-speech (Piper), and the webcam grab.
- They talk over a WebSocket bound to 127.0.0.1 only.

## Files

- PLAN.md: phases with tasks and "done when" checks. Work one phase at a time.
- ARCHITECTURE.md: components, WebSocket protocol, state machine, interfaces, config keys.
- CLAUDE.md: conventions the coding agent must follow.
- .claude/agents/: subagents (backend-dev, face-dev, reviewer).
- prompts/system.md: the assistant's system prompt.
- config.example.toml, .env.example, requirements.in, .gitignore: setup files.

## Using this with a coding agent

1. Move this folder to your projects directory and run `git init`.
2. Open Claude Code in the folder.
3. Paste the kickoff prompt below.

Kickoff prompt:

```
Read CLAUDE.md, PLAN.md and ARCHITECTURE.md. Outline the plan for Phase 0 in a few lines and wait for my OK. Then do Phase 0 only. Use the backend-dev subagent for implementation and the reviewer subagent before committing. Look up current versions of all dependencies before pinning them.
```

For later phases: "Continue with Phase N. Same rules."

## Agent harness

A single agent that reads CLAUDE.md and PLAN.md is enough for a solo prototype. The subagents are optional. They help mainly by keeping the main context clean and by giving you an independent review pass before each commit.

- Main session: orchestrator. Reads the plan, delegates, integrates, commits once per phase.
- backend-dev: everything under src/ and tests/.
- face-dev: everything under web/.
- reviewer: read-only. Runs after each phase, reports findings, edits nothing.

Delegation per phase:

- Phase 0: backend-dev
- Phase 1: face-dev
- Phase 2: backend-dev (LLM, sentence splitter) and face-dev (subtitles, fake mouth) in parallel, since the WebSocket protocol is already fixed in ARCHITECTURE.md
- Phase 3 and 4: both, in parallel, same reason
- Phase 5: backend-dev, plus face-dev for the "looking" animation
- Phase 6 (nice to have): both
- Reviewer after every phase, before the commit
