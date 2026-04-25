# Desktop VLM Agent

A Python desktop automation agent that uses **Google Gemini** (a Vision-Language Model) as its "brain" to drive the OS. It reads plain-English instructions from `tasks.txt`, takes a screenshot, asks Gemini what to do, executes the action with `pyautogui`, and verifies the result with a second Gemini call.

> **Quick start:** jump to [Quick start (Windows)](#quick-start-windows) if you just want to run it. No Python experience required.

## Table of contents

- [Quick start (Windows)](#quick-start-windows)
- [Troubleshooting](#troubleshooting)
- [Features](#features)
- [Developer setup (Linux / macOS)](#developer-setup-linux--macos)
- [Configure](#configure)
- [Write a task file](#write-a-task-file)
- [Run](#run)
- [Command reference](#command-reference)
- [Architecture](#architecture)
- [Safety notes](#safety-notes)
- [Project layout](#project-layout)

---

## Quick start (Windows)

Takes ~10 minutes.

### 1. Install Python 3.10+

1. Go to https://www.python.org/downloads/.
2. Click the big yellow **Download Python 3.x** button.
3. Run the installer.
4. **Important:** on the first screen, tick **"Add python.exe to PATH"**. Then click **Install Now**.

Verify (open **Command Prompt**: <kbd>Win</kbd> → type `cmd` → Enter):

```cmd
python --version
```

You should see `Python 3.10.x` or newer. If it says *"not recognized"*, re-run the installer and tick the PATH box.

### 2. Install Git (or skip and download the ZIP)

Install from https://git-scm.com/download/win (defaults through every screen are fine), then **open a new Command Prompt** so PATH updates.

Or — if you'd rather not install Git — just download the repo as a ZIP:
1. Open https://github.com/nd1836017/https-github.com-nd1836017-desktop-vlm-agent in your browser.
2. Click green **Code** → **Download ZIP**.
3. Extract it to `C:\Users\<you>\Documents\` and rename the folder to `desktop-vlm-agent`.

### 3. Get a free Gemini API key

1. Go to https://aistudio.google.com/app/apikey.
2. Sign in with Google.
3. Click **Create API key** → **Create API key in new project**.
4. Copy the key (looks like `AIzaSy...`). You'll paste it in step 5.

Free tier allows up to 500 requests/day on `gemini-3.1-flash-lite-preview` — plenty for several long tasks a day.

### 4. Clone the repo

```cmd
cd %USERPROFILE%\Documents
git clone https://github.com/nd1836017/https-github.com-nd1836017-desktop-vlm-agent.git desktop-vlm-agent
cd desktop-vlm-agent
```

(Skip the `git clone` line if you downloaded the ZIP.)

### 5. Create a virtual environment and install dependencies

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

The last line will download ~100 MB and takes 1–2 minutes. You'll know the venv is active when your prompt starts with `(.venv)`.

> **Remember:** every time you reopen Command Prompt, `cd` into the project folder and run `.venv\Scripts\activate` again.

### 6. Create your `.env` file

```cmd
copy .env.example .env
notepad .env
```

In the Notepad window that opens, replace `your_api_key_here` on the `GEMINI_API_KEY=` line with the key from step 3. Save (<kbd>Ctrl+S</kbd>) and close.

### 7. Write a task file

```cmd
notepad tasks.txt
```

Start simple:

```
Press the Windows key to open the Start menu
Type "notepad" to search for Notepad
Press Enter to open Notepad
Type "Hello from the VLM agent"
```

Save and close.

### 8. Run it

```cmd
python -m agent
```

The agent starts taking screenshots, calling Gemini, and clicking around. You'll see step-by-step logs in the terminal.

**Abort at any time** by slamming your mouse into any corner of the screen — that triggers `pyautogui`'s failsafe.

If the agent hits a 2FA prompt or a "Verify it's you" screen, it prints `[!] PAUSE:` and waits for input. Handle the prompt manually (tap your phone, enter a code, etc.), then press <kbd>Enter</kbd> in the terminal to resume. Type `q`+<kbd>Enter</kbd> to abort.

### 9. Resume or reset

If the agent halts partway through (quota, network, verifier fail), restart from the last verified step:

```cmd
python -m agent --resume
```

To wipe the checkpoint and start fresh:

```cmd
python -m agent --reset
```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `'python' is not recognized as an internal or external command` | Rerun the Python installer and tick **"Add python.exe to PATH"**. Then open a **new** Command Prompt. |
| `'git' is not recognized` | You skipped step 2. Either install Git from https://git-scm.com/download/win (then open a new cmd window) or download the repo as a ZIP. |
| `RuntimeError: GEMINI_API_KEY is not set` | You forgot to edit `.env`. Run `notepad .env` and paste your key. |
| Clicks land on the wrong pixel | Windows display scaling ≠ 100%. Right-click `python.exe` → Properties → Compatibility → **Change high DPI settings** → tick **Override high DPI scaling behavior** → set to **Application**. Or set Windows display scaling to 100% temporarily. |
| Chrome keeps asking "Verify it's you" | The agent's Chrome profile is separate from your normal one. Use a persistent profile: `mkdir %USERPROFILE%\.chrome-agent-profile` then launch Chrome with `--user-data-dir="%USERPROFILE%\.chrome-agent-profile"` once and sign in. Future runs will reuse the session. |
| `Retry budget exhausted` / 429 errors | You've hit Gemini's free daily quota. Wait 24h, or switch to a different model by setting `GEMINI_MODEL=gemini-2.5-flash-lite` in `.env`. |
| Agent halts mid-run | Use `python -m agent --resume` to continue from the last verified step. |
| Want to debug a flaky step | Set `SAVE_RUN_ARTIFACTS=true` in `.env`. Each run writes `runs/<timestamp>/step_NNN_*` with before/after screenshots, raw VLM responses, and a `summary.json`. |

---

## Features

- **Visual-Action Loop** — screenshot → VLM plan → parse → `pyautogui` execute → screenshot → VLM verify.
- **Rich command set** — `CLICK` / `DOUBLE_CLICK` / `RIGHT_CLICK` / `MOVE_TO`, `CLICK_TEXT` (OCR-anchored), `PRESS`, `TYPE` (clipboard-based for Unicode), `SCROLL`, `DRAG`, `WAIT`, `PAUSE`.
- **CSV-driven loops** — `FOR_EACH_ROW [data.csv]` … `END_FOR_EACH_ROW` repeats a block per row with `{{row.<field>}}` substitution. See [CSV-driven loops](#csv-driven-loops).
- **Two-stage CLICK** — after a coarse coordinate, the agent crops around the target and asks the VLM to refine + disambiguate. Cuts misclicks dramatically.
- **Normalized grid** — VLM emits coordinates on a 0–1000 grid; agent auto-detects native resolution and maps to pixels.
- **Short-term memory + replan** — rolling window of recent (step, action, verdict) rows fed back into the planner; on verifier FAIL the agent replans up to a configurable budget before halting.
- **Checkpoint / resume** — `.agent_state.json` is written atomically after each verified step. `--resume` picks up where the last run halted.
- **PAUSE on 2FA / CAPTCHA** — the VLM is taught to emit `PAUSE [reason]` when it sees a manual-approval screen. Agent halts cleanly without side-effects, then resumes on <kbd>Enter</kbd>.
- **RPD guard** — tracks Gemini requests per day; warns at 75% of your free-tier quota and halts cleanly at 95% so `--resume` works later.
- **Retry-with-backoff** — transient 429 / 5xx errors are retried with exponential + jittered backoff. No silent model fallback.
- **Run artifacts** — per-step before/after screenshots, raw VLM output, verdicts, and `summary.json` for postmortems (`SAVE_RUN_ARTIFACTS=true`).
- **Log redaction** — `TYPE` contents are logged as `<REDACTED, N chars>` by default, so passwords never hit disk.
- **Failsafe** — mouse-to-corner kill switch is always on.

---

## Developer setup (Linux / macOS)

Requires Python 3.10+ and a desktop environment with a display server (X11 / Wayland on Linux, native on macOS).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Linux, `pyautogui` + `pyperclip` need a few extras:

```bash
sudo apt-get install -y python3-tk python3-dev scrot xclip
```

(`xclip` is needed so `pyperclip` can paste non-ASCII text via the OS clipboard. macOS ships `pbcopy`/`pbpaste` so no extra install is needed.)

---

## Configure

```bash
cp .env.example .env   # Windows: copy .env.example .env
# then edit .env and paste your Gemini API key
```

Get a key at https://aistudio.google.com/app/apikey.

Common env vars (full list in [`.env.example`](./.env.example)):

| Variable | Default | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | **Required.** Google AI Studio API key. |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite-preview` | Any Gemini vision-capable model. |
| `TASKS_FILE` | `tasks.txt` | Path to the instructions file. |
| `ANIMATION_BUFFER_SECONDS` | `1.5` | Sleep after CLICK / PRESS so UI animations settle. |
| `MAX_REPLANS_PER_STEP` | `2` | Replans allowed per step after a verifier FAIL. |
| `MAX_TOTAL_REPLANS` | `10` | Global replan cap across the whole run. |
| `HISTORY_WINDOW` | `5` | Recent (step, action, verdict) rows fed into each plan prompt. |
| `STATE_FILE` | `.agent_state.json` | Checkpoint file for `--resume`. |
| `ENABLE_TWO_STAGE_CLICK` | `true` | Toggle coarse→refined CLICK. |
| `SAVE_RUN_ARTIFACTS` | `false` | Dump per-step before/after + VLM output + `summary.json` under `runs/<timestamp>/`. |
| `RPD_LIMIT` | `500` | Gemini requests/day before HALT. Matches the 3.1-flash-lite-preview free tier. |
| `RPD_WARN_THRESHOLD` / `RPD_HALT_THRESHOLD` | `0.75` / `0.95` | WARN and HALT fractions of `RPD_LIMIT`. |
| `LOG_REDACT_TYPE` | `true` | Log `TYPE <REDACTED, N chars>` instead of literal text. |
| `LOG_LEVEL` | `INFO` | Python logging level. |

---

## Write a task file

`tasks.txt` is a plain text file. One natural-language instruction per line. Blank lines and lines starting with `#` are ignored.

```
# open notepad
Press the Windows key to open the Start menu
Type "notepad" to search for Notepad
Press Enter to open Notepad
Type "Hello from the Desktop VLM Agent!"

# open a website
Open Chrome
Click the address bar
Type "https://example.com" and press Enter
```

Instructions are free-form English. The agent is not parsing them directly — it feeds them to Gemini as goals, one at a time, and Gemini emits concrete commands.

### CSV-driven loops

For repetitive work — filling a form once per row, sending bulk emails, etc. — wrap a block of steps in `FOR_EACH_ROW [data.csv]` … `END_FOR_EACH_ROW` and use `{{row.<field>}}` placeholders to pull values from the CSV header:

```
# open the form once
Open Chrome
Click the address bar
Type "https://example.com/contact" and press Enter

FOR_EACH_ROW [data.csv]
    Click the "First name" field
    Type "{{row.first_name}}"
    Click the "Email" field
    Type "{{row.email}}"
    Click the "Message" field
    Type "{{row.message|Hello!}}"   # default value if cell is empty
    Click Submit
END_FOR_EACH_ROW
```

With a CSV like:

```
first_name,email,message
Alice,alice@example.com,Following up on our chat
Bob,bob@example.com,
```

the block above runs the inner steps twice — once with Alice's row, once with Bob's. Empty cells fall back to the `|default` value (`Hello!` here for Bob's missing message).

- Field names match the CSV header **exactly** (case-sensitive).
- The CSV path is resolved relative to the tasks file (or override at runtime with `python -m agent --csv real_data.csv`).
- Nesting `FOR_EACH_ROW` is not supported.
- A working example is in [`examples/tasks_csv_demo.txt`](./examples/tasks_csv_demo.txt) + [`examples/data_demo.csv`](./examples/data_demo.csv).

---

## Run

```bash
python -m agent                # run from tasks.txt (fresh or continued from checkpoint)
python -m agent --resume       # force resume from last verified step
python -m agent --reset        # delete checkpoint and start fresh
python -m agent --csv data.csv # override the FOR_EACH_ROW data file at runtime
python -m agent --two-stage-click     # force two-stage CLICK on (overrides .env)
python -m agent --no-two-stage-click  # force two-stage CLICK off (faster, less safe)
```

Logs go to stdout. On a halting verifier FAIL the agent exits with a non-zero code and prints the reason.

**Abort at any time** by slamming your mouse into any screen corner — `pyautogui`'s failsafe.

---

## Command reference

The VLM is instructed to respond with exactly one of the following:

| Command | Example | Meaning |
| --- | --- | --- |
| `CLICK [X,Y]` | `CLICK [500,250]` | Left-click at normalized (0–1000) coordinates. |
| `DOUBLE_CLICK [X,Y]` | `DOUBLE_CLICK [400,400]` | Double-click (e.g. open files/folders). |
| `RIGHT_CLICK [X,Y]` | `RIGHT_CLICK [700,300]` | Right-click (context menu). |
| `MOVE_TO [X,Y]` | `MOVE_TO [500,500]` | Move mouse without clicking (hover tooltips/menus). |
| `CLICK_TEXT [label]` | `CLICK_TEXT [Sign in]` | OCR-anchored click on the nearest matching visible text. |
| `PRESS [KEY]` | `PRESS [enter]`, `PRESS [ctrl+c]` | Press a single key or `+`-separated hotkey. |
| `TYPE [TEXT]` | `TYPE [hello world]` | Type literal text. Non-ASCII is pasted via clipboard. |
| `SCROLL [DIR, AMOUNT]` | `SCROLL [down, 5]` | Scroll up or down by AMOUNT wheel clicks. Both fields required. |
| `DRAG [X1,Y1,X2,Y2]` | `DRAG [100,200,400,500]` | Click-and-drag from (X1,Y1) to (X2,Y2) on the 0–1000 grid. |
| `WAIT [SECONDS]` | `WAIT [3]` | Pause execution for N seconds (no VLM call). |
| `PAUSE [reason]` | `PAUSE [2FA prompt needs manual approval]` | Emitted by the VLM when it sees a 2FA / CAPTCHA / "Verify it's you" screen. Agent halts cleanly with zero side-effects and waits for <kbd>Enter</kbd>. |
| `DOWNLOAD [url, filename]` | `DOWNLOAD [https://example.com/inv.pdf, inv.pdf]` | Fetches a URL via HTTPS and stores it in the run's file workspace. `filename` is optional; when missing it's derived from the URL. Honors the run's file mode (temp/save/feed). |
| `ATTACH_FILE [filename]` | `ATTACH_FILE [inv.pdf]` | Types a path into the focused OS file-picker (`Ctrl+L` → path → Enter). The agent must already have clicked the dialog's Browse button. `filename` is resolved against the workspace first, then the disk. |
| `CAPTURE_FOR_AI [filename]` | `CAPTURE_FOR_AI` | Buffers an image for the next plan call. Without an arg, captures the current screen; with one, reads it from the workspace or disk. Useful for "look at this PDF and tell me the invoice number". |

The parser is lenient — it recovers from conversational wrapping, missing brackets, or parentheses in place of brackets.

### File-handling modes

When the tasks file uses `DOWNLOAD`, `ATTACH_FILE`, or `CAPTURE_FOR_AI`, the agent asks at run start how to persist captured files:

```
File handling for this run:
  [t] temp  — auto-cleanup on success, kept on failure for debugging
  [s] save  — persist all downloads to a directory
  [f] feed  — never write to disk, only show to the VLM
Choose [t/s/f]:
```

- **temp** *(default)* — files go to an OS temp dir, wiped automatically when the run succeeds, preserved on failure so you can inspect what the agent grabbed.
- **save** — files persist to `--workdir` (or `WORKDIR` from `.env`, defaulting to `./agent_files`).
- **feed** — files never touch disk; bytes are streamed straight into the next VLM call.

Tasks files that don't use any file primitives skip this prompt entirely. For unattended runs (`.exe`, scheduled jobs), set `FILE_MODE` / `WORKDIR` in `.env` or pass `--mode` / `--workdir` on the CLI.

Inside `FOR_EACH_ROW`, file names are suffixed with `(rowN)` to prevent collisions: `invoice.pdf` on row 50 becomes `invoice(row50).pdf`.

---

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
      parse_command (regex + JSON)       │
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
                          │
                          └────────── FAIL ▶ replan (budget) ▶ halt + checkpoint
```

---

## Safety notes

- The agent clicks, types, and presses keys on your **actual desktop**. Close anything sensitive before running it.
- `pyautogui.FAILSAFE = True` is always on (mouse-to-corner aborts).
- The verifier halts immediately on any ambiguous or negative verdict — it errs on the side of stopping.
- `TYPE` contents are redacted from logs by default. Still, avoid putting production passwords directly in `tasks.txt` — prefer a password manager that you copy-paste out of at `PAUSE` time.

---

## Project layout

```
desktop-vlm-agent/
├── agent/
│   ├── __init__.py
│   ├── __main__.py      # CLI entry (python -m agent)
│   ├── agent.py         # main run-loop + replan + checkpoint
│   ├── artifacts.py     # per-step run artifacts (SAVE_RUN_ARTIFACTS)
│   ├── config.py        # .env loading + validation
│   ├── cost.py          # RPD guard (requests-per-day quota tracker)
│   ├── executor.py      # pyautogui wrapper + human-like cadence
│   ├── history.py       # rolling short-term memory
│   ├── ocr.py           # pytesseract-backed CLICK_TEXT
│   ├── parser.py        # command parser (regex + JSON, lenient)
│   ├── screen.py        # screenshot + 0-1000 ↔ pixel scaling + overlays
│   ├── state.py         # checkpoint file (atomic writes)
│   ├── tasks_loader.py  # tasks.txt parser + FOR_EACH_ROW + CSV templating
│   └── vlm.py           # Gemini client (plan + verify + retry)
├── examples/
│   ├── tasks_csv_demo.txt
│   └── data_demo.csv
├── scripts/
│   └── launch-chrome.sh # persistent-profile Chrome launcher (Linux)
├── tests/               # pytest unit tests
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── tasks.txt
```
