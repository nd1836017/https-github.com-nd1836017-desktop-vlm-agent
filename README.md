# Desktop VLM Agent

A Python desktop automation agent that uses a **Gemini Flash** VLM as its "brain" to interact with the OS. It reads step-by-step natural-language instructions from `tasks.txt`, captures screenshots, asks the VLM what to do, parses a command from the response, executes it with `pyautogui`, and verifies the outcome with a second VLM call.

## Features

- **Visual-Action Loop** — screenshot → VLM plan → regex parse → `pyautogui` execute → screenshot → VLM verify.
- **Three commands**: `CLICK [X,Y]`, `PRESS [KEY]`, `TYPE [TEXT]`.
- **Scaling correction** — VLM emits coordinates on a normalized 0–1000 grid; the agent auto-detects the native screen resolution and maps to pixels.
- **Animation buffer** — a mandatory `time.sleep(1.5)` after any `CLICK` or `PRESS [win]` so OS UI animations can finish before the next screenshot.
- **Regex-protected parsing** — tolerates conversational text and common bracket omissions; if a command cannot be parsed, the step is retried once.
- **Short-term memory** — a rolling window of recent `(step, action, verdict)` records is fed back into each plan prompt so the VLM can reason about what has already happened. Tunable via `HISTORY_WINDOW`.
- **Replan-on-failure with budget** — when the verifier reports FAIL, the agent tells the VLM what went wrong and asks for a different action, up to `MAX_REPLANS_PER_STEP` times before halting. Avoids both premature halts and runaway retry loops.
- **Checkpoint + resume** — after every verified step the progress is atomically written to a JSON state file. Long runs can be resumed from the last successful step with `python -m agent --resume`.
- **Two-stage CLICK refinement** (optional) — when enabled, after the planner emits a coarse `CLICK [x,y]` the agent crops a `TWO_STAGE_CROP_SIZE_PX` × `TWO_STAGE_CROP_SIZE_PX` region around that point, asks the VLM to list all plausible targets matching the step, and (if multiple are returned) a second VLM call disambiguates by picking the right one. Dramatically cuts "missed the address bar"-style failures on dense / multi-hitbox UIs. Toggle per run with `--two-stage-click` / `--no-two-stage-click`, or globally via `ENABLE_TWO_STAGE_CLICK`.
- **Verification + halt** — after every action, a second VLM call checks that the screen state matches the goal; if the replan budget is exhausted, execution halts and the user is notified via the terminal.
- **Failsafe** — `pyautogui.FAILSAFE = True` (move the mouse to a screen corner to abort).

## Architecture

```
tasks.txt ──▶ agent.run ──┐
                          │
                          ▼
           ┌──── capture_screenshot ─────┐
           │                             │
           ▼                             │
   GeminiClient.plan_action              │
           │                             │
           ▼                             │
      parse_command (regex)              │
           │                             │
           ▼                             │
         execute (pyautogui)             │
           │                             │
           ▼                             │
           └──── capture_screenshot ─────┘
                          │
                          ▼
              GeminiClient.verify ──▶ PASS ▶ next step
                          │
                          └────────── FAIL ▶ halt + notify
```

## Install

Requires Python 3.10+ and a desktop environment with a display server (X11 / Wayland on Linux, native on Windows/macOS).

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Linux, `pyautogui` needs a couple of extras:

```bash
sudo apt-get install -y python3-tk python3-dev scrot
```

## Configure

```bash
cp .env.example .env
# then edit .env and paste your Gemini API key
```

Get a key at https://aistudio.google.com/app/apikey.

Env vars (all optional except `GEMINI_API_KEY`):

| Variable | Default | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | **Required.** API key for Google AI Studio. |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite-preview` | Any Gemini vision-capable model (e.g. `gemini-3.1-flash-lite-preview`, `gemini-2.5-flash-lite`, `gemini-2.5-flash`). Lite = cheaper + higher free-tier quota. |
| `TASKS_FILE` | `tasks.txt` | Path to the instructions file. |
| `ANIMATION_BUFFER_SECONDS` | `1.5` | Sleep after CLICK / PRESS [win]. |
| `MAX_STEP_RETRIES` | `1` | Retries per attempt on unparseable VLM responses. |
| `MAX_REPLANS_PER_STEP` | `2` | Replan attempts after a verifier FAIL before halting. |
| `HISTORY_WINDOW` | `5` | Number of recent steps fed back to the VLM as short-term memory (0 = disabled). |
| `STATE_FILE` | `.agent_state.json` | Checkpoint file written atomically after each verified step. |
| `ENABLE_TWO_STAGE_CLICK` | `true` | Enable two-stage CLICK refinement (crop + VLM-refined pick + optional disambiguation). Overridable per run via `--two-stage-click` / `--no-two-stage-click`. |
| `TWO_STAGE_CROP_SIZE_PX` | `300` | Side length (px) of the square cropped around the coarse click point for refinement. |
| `MAX_CLICK_CANDIDATES` | `5` | Cap on the number of refined candidates to consider before disambiguation. |
| `LOG_LEVEL` | `INFO` | Standard Python logging level. |

## Write a task file

`tasks.txt` is a plain text file. One natural-language instruction per line. Blank lines and lines starting with `#` are ignored.

```
Press the Windows key to open the Start menu
Type "notepad" to search for Notepad
Press Enter to open Notepad
Type "Hello from the Desktop VLM Agent!"
```

## Run

```bash
python -m agent                      # normal run, starts from step 1
python -m agent --resume             # continue from the last verified step
python -m agent --reset              # delete the checkpoint before running

# Two-stage CLICK refinement (crop + VLM-refined pick) — toggle per run:
python -m agent --two-stage-click    # force ON for this run (safer, +API calls)
python -m agent --no-two-stage-click # force OFF for this run (faster, cheaper)
```

Logs go to stdout. On verification failure after the replan budget is exhausted, the agent halts with exit code 1, prints the reason, and suggests `--resume` so you can continue after fixing the blocker.

**Abort at any time** by slamming your mouse into any corner of the screen — that triggers `pyautogui`'s failsafe.

## Command reference

The VLM is instructed to respond with exactly one of:

| Command | Example | Meaning |
| --- | --- | --- |
| `CLICK [X,Y]` | `CLICK [500,250]` | Left-click at normalized (0–1000) coordinates. |
| `PRESS [KEY]` | `PRESS [win]`, `PRESS [ctrl+c]` | Press a single key or `+`-separated hotkey. |
| `TYPE [TEXT]` | `TYPE [hello world]` | Type literal text. |

The parser is lenient: it will still recover if the VLM omits brackets, wraps the command in prose, or uses parentheses instead of brackets.

## Safety notes

- The agent clicks, types, and presses keys on your **actual desktop**. Close anything sensitive before running it.
- `pyautogui.FAILSAFE = True` is always on.
- The verifier halts immediately on any ambiguous or negative verdict — it errs on the side of stopping.
- Keys in `tasks.txt` are sent as literal keystrokes; never put secrets or passwords in it.

## Project layout

```
desktop-vlm-agent/
├── agent/
│   ├── __init__.py
│   ├── __main__.py      # entry point (python -m agent)
│   ├── agent.py         # main run-loop
│   ├── config.py        # .env loading
│   ├── executor.py      # pyautogui wrapper + animation buffer
│   ├── parser.py        # regex command parser (crash-proof)
│   ├── screen.py        # screenshot + 0-1000 <-> pixel scaling
│   └── vlm.py           # Gemini 2.0 Flash client
├── tests/
│   ├── test_parser.py
│   └── test_screen.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── tasks.txt
```
