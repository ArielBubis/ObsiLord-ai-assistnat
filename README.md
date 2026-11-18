# Obsi-Lord

A voice assistant with an animated 3D face that lives on your desktop, sees through your webcam when it needs to, and keeps your conversations on your own PC.

Obsi-Lord is a personal prototype. You talk to it (or type), it answers out loud with a short, dry reply, and its face reacts while it listens, thinks, looks and speaks, with the mouth following its voice. It can run fully offline on a local model, or use Gemini when you want a smarter chat model.

**Status:** working prototype. Voice and text chat, the animated face, camera looks, an optional image lookup, and conversation memory all work today.

## Why

Most assistants live in the cloud, listen all the time, and forget you between sessions. Obsi-Lord tries the opposite:

- **Local first.** Speech recognition, speech output, vision and (optionally) the chat model all run on your machine.
- **The camera is not a stream.** It takes one photo only during a turn: when you ask ("what am I holding?") or when the assistant decides to look, within a cooldown and an hourly cap. The photo is described by a local vision model, and the chat model only ever gets that text.
- **It remembers, locally.** Conversations are kept in one SQLite file on your PC, and the assistant can search them when you refer to something from before.
- **It has a character.** Short, spoken replies with a bit of attitude, not a wall of markdown.

## Features

- **Talk or type.** Hold to talk, or type in the chat box. Replies stream as subtitles and are spoken as they arrive.
- **An animated face.** A 3D character rendered with three.js and redrawn as dots (dither, halftone or ASCII), with a backdrop of a voice ring and drifting orbs. It has idle, listening, thinking, looking and speaking states, blinks and glances, a mouth driven by the speech audio, and expressions saved in the character. Bring your own character as a .glb or .vrm file. A 64x48 8-bit face is the alternative skin, and the fallback when WebGL is not available.
- **Pick your brain.** Gemini, or any local model behind an OpenAI-compatible server (LM Studio, Ollama), switchable from the page. Models with tool support call the camera themselves; for the rest, a small routing step picks the tool.
- **Looks when it matters.** `look()` takes one photo. A local chat model that sees images gets the photo itself; for Gemini and text-only models, Qwenny, a small local vision model (Qwen3-VL-2B on llama.cpp), describes it. Settings, Photos chooses whether photos are dropped after the reply, kept for follow-ups ("what color was it?"), or saved to disk.
- **Identifies things (opt-in).** For things a description can't name, like an album cover or a product, the local model cuts the object out of the photo and only that cut-out goes to an online lookup. Off by default, with hourly and monthly caps.
- **Conversation memory.** Recent turns go with every request, a reload within 30 minutes picks up where you left off (older conversations can be continued from Settings), and with a local model the assistant can search older conversations ("what did we talk about yesterday?").
- **Editable memory.** In Settings you can edit, rename, delete or add turns in stored conversations.
- **Cost guards built in.** Usage logging, a daily token cap, and look() limits exist before any model call does.

## How it works

```
 browser page                         Python backend (127.0.0.1 only)
 ------------                         -------------------------------
 face, mic, speaker  <-- WebSocket -->  speech to text   (faster-whisper)
                                        chat model       (Gemini or local)
                                        tools            look, search_image, recall
                                        vision           (Qwen3-VL-2B, llama.cpp)
                                        text to speech   (Piper)
                                        memory           (SQLite + FTS5)
```

One turn: you speak, the backend transcribes it, the chat model answers or calls a tool (take a photo, search memory), and the reply is split into sentences and spoken while the rest is still being written. The face follows the state of the turn.

## Privacy

- The server binds to 127.0.0.1 only. No open ports, no port forwarding.
- Camera frames are never logged, and they are written to disk only if you set Photos to "Save to disk" in Settings (into `data/photos/`, gitignored). By default a photo is used for its reply and then dropped. The Snapshots view in the page keeps the last photo in memory only.
- The only feature that sends part of a photo off the PC is the opt-in image lookup, and it sends only the cut-out.
- Conversation memory is text only and stays in `data/` (gitignored). Older conversations are searchable by local models only, so they are not sent to Gemini that way.

## Built with

Python 3.12, FastAPI, faster-whisper, Piper, llama.cpp, OpenCV, SQLite, plain JavaScript with three.js and WebGL shaders for the face, and the Gemini API or any OpenAI-compatible local server.

## Running it

Windows PC with a webcam and a microphone, Python 3.12.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy config.example.toml config.toml
copy .env.example .env            # add GEMINI_API_KEY to use Gemini
python -m obsilord.vision.fetch   # downloads llama.cpp and the vision model once
python -m obsilord
```

Then open http://127.0.0.1:8765. All settings are in `config.toml`, with comments for each key. Set `demo = true` under `[face]` to show only the face, subtitles and chat row.

## Roadmap

- Obsidian: conversations and notes in a vault.
- A Telegram front end, using long polling so no inbound port is needed.
