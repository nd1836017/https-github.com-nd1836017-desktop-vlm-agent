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
| `VLM_IMAGE_MAX_DIM` | `1280` | Long-edge cap (px) applied before sending each screenshot to Gemini. `0` disables. |
| `VLM_IMAGE_QUALITY` | `80` | JPEG quality (1–100) used for VLM screenshots. Defaults are zero-accuracy-risk. |
| `VLM_SKIP_IDENTICAL_FRAMES` | `false` | When on, the planner gets a "screen unchanged" text marker instead of a fresh image when consecutive frames hash identically. |
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
| `DOWNLOAD [url, filename]` | `DOWNLOAD [https://example.com/inv.pdf, inv.pdf]` | Fetches a URL via HTTPS and stores it in the run's file workspace. `filename` is optional; when missing it's derived from the URL. The URL/FILENAME delimiter is **comma followed by at least one space** (so URLs like `?ids=1,2,3` aren't truncated). Honors the run's file mode (temp/save/feed). |
| `ATTACH_FILE [filename]` | `ATTACH_FILE [inv.pdf]` | Pastes a path into the focused OS file-picker (`Ctrl+L` → paste → Enter) via the system clipboard, so non-ASCII filenames work. The agent must already have clicked the dialog's Browse button. `filename` is resolved against the workspace first, then the disk. |
| `CAPTURE_FOR_AI [filename]` | `CAPTURE_FOR_AI []` | Buffers an image for the next plan call. **Brackets are required** (empty `[]` = grab the current screen; otherwise reads `filename` from the workspace or disk). Useful for "look at this PDF and tell me the invoice number". |
| `REMEMBER [name]` | `REMEMBER [order_id]` | Reads the value labeled `name` off the current screen (via VLM) and stores it as a run variable. Use when later steps need a value (order id, confirmation number, etc.) that is currently visible. The variable name must match `[A-Za-z_][A-Za-z0-9_]*`. |
| `REMEMBER [name = value]` | `REMEMBER [order_id = ND12345]` | Stores `value` literally as `name` with no VLM extraction — useful when the value came from a prior CAPTURE_FOR_AI. |
| `RECALL [name]` | `RECALL [order_id]` | Types the stored value of `name` into the focused field. Equivalent to `TYPE [{{var.name}}]`. Fails loudly if `name` is unset. |

You can also reference variables anywhere in step text via the `{{var.<name>}}` placeholder — for example, `open the page about {{var.order_id}}`. Use `{{var.<name>|default}}` to fall back when the variable is unset. Substitution happens at step run-time, so a step can use a variable set by an earlier `REMEMBER` step. Variables persist across `--resume` (snapshotted in the checkpoint).

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

### Reliability: timeouts and stuck-step detection

Two safeguards stop a single bad step from burning the whole replan budget (or the daily RPD quota):

- **Per-step wall-clock timeout** — `STEP_TIMEOUT_SECONDS` (default `180`). When the elapsed wall time on a step exceeds the budget the agent stops trying and fails the step with a clear reason. Includes parse retries, replans, and `PAUSE` waits. Set to `0` to disable.
- **Stuck-step detection** — `STUCK_STEP_THRESHOLD` (default `3`). After each failed attempt the agent fingerprints the post-action screenshot. If the last `N` attempts produced **identical** screen state, the step bails early instead of replanning into the same dead-end. Set to `0` to disable.

Both safeguards halt cleanly between steps with the checkpoint intact — fix the underlying problem (a frozen modal, an offline page) and `python -m agent --resume` continues from where it stopped.

### Smart screenshots (token / bandwidth saver)

Every screenshot we send to Gemini goes through a tiny optimization pipeline first. The defaults are zero-accuracy-risk and on by default — they just stop wasting bytes the model would discard anyway.

- **Downsample to `VLM_IMAGE_MAX_DIM`** (default `1280` px on the long edge). Gemini's vision tokenizer downscales internally — sending raw 4K screenshots burns request size for pixels that get thrown away.
- **JPEG-encode at `VLM_IMAGE_QUALITY`** (default `80`). UI text and icons stay sharp; the wire payload shrinks ~10–30× versus the equivalent PNG.
- **Identical-frame skip (opt-in)** — set `VLM_SKIP_IDENTICAL_FRAMES=true` to drop the screenshot from `plan_action` calls when the new frame fingerprints identically to the previous one (i.e. the last action did nothing visible). The planner gets a "screen unchanged" marker instead, and is asked to pick a *different* action. Verify / disambiguate / refine still always get a fresh image.

If you ever need to send the original full-resolution PNGs again (debugging an accuracy regression, comparing model behavior), set `VLM_IMAGE_MAX_DIM=0` and `VLM_IMAGE_QUALITY=100` — the encoded JPEG is then effectively lossless.

### Smart task router (TASK_ROUTING)

The router runs **once at run start** — a single Gemini call decomposes your raw `tasks.txt` and tags each step's complexity. The planner then sees that tag (and an optional suggested literal command) as advisory context. Goal: stop randomizing features and steer the agent to the cheapest primitive that fits.

Three complexity tags:

| Tag | Cost | Used for | Example |
|---|---|---|---|
| `browser-fast` | ~0 vision tokens | URLs / known selectors | `BROWSER_GO [https://youtube.com]` |
| `browser-vlm` | full vision | browser, vision-required | "click the third video" |
| `desktop-vlm` | full vision | native apps, OS dialogs | "open Notepad" |

Three modes (`TASK_ROUTING` env var):

- `auto` (default) — runs the router. Browser fast-path is auto-enabled when Chrome's CDP is reachable; otherwise `browser-fast` decisions are downgraded to `browser-vlm` so the planner doesn't try to emit doomed commands.
- `manual` — skips the Gemini call. You annotate lines yourself with `[browser-fast]`, `[browser-vlm]`, `[desktop-vlm]` (or aliases `[browser]`, `[vlm]`, `[desktop]`).
- `off` — no router at all. Safety rollback if anything goes wrong.

Manual annotations beat the auto router — the user's explicit `[tag]` is always more authoritative than the model's classification.

Example tasks file with mixed annotation:

```
# Auto-classified (router decides)
open youtube
search for justin bieber
click the first video

# Manual override
[vlm] take a screenshot of the comments section
[browser-fast] BROWSER_GO [https://gmail.com]
```

The router fails open: any error (network, schema, timeout) drops back to no-routing behavior and logs a warning. A flaky router never blocks a run from starting.

### Task decomposition (TASK_DECOMPOSITION)

Runs **before** the router. One extra Gemini call at run start splits compound natural-language steps into atomic substeps the agent can verify one at a time. Without it, a single line like `play the 2nd video on youtube` reaches the planner as one step — the verifier then has to judge the whole compound goal off a single screenshot, which can produce false positives (e.g. autoplay running on YouTube's homepage being read as "the 2nd video is playing").

Two modes (`TASK_DECOMPOSITION` env var):

- `auto` (default) — runs Gemini; on any error the original step list is returned unchanged. Single-action lines pass through 1:1; compound lines emit 2-5 atomic substeps.
- `off` — no decomposition. Use as a rollback or when you've already split your task by hand.

Example expansion (auto mode):

```
play the 2nd video on youtube
```

becomes:

```
open a new Chrome tab
go to youtube.com
scroll the video grid into view
click the second video result
```

Row metadata (`FOR_EACH_ROW`) and inline routing annotations (`[browser-fast]` etc.) are preserved across decomposition.

### Run replay dashboard

Read-only HTTP UI over the artifact directories saved by `SAVE_RUN_ARTIFACTS=true`. Launch with:

```
python -m agent --serve-dashboard
```

It binds to `127.0.0.1:8000` by default (localhost only — screenshots can contain sensitive content). Use `--dashboard-host 0.0.0.0` to expose, or `--dashboard-port` to change the port.

The dashboard reads `RUN_ARTIFACTS_DIR` (default `runs/`) and shows a per-step timeline: the BEFORE screenshot the planner saw, the action it emitted, the AFTER screenshot, and the verifier's verdict. Failing steps are highlighted; you can click any screenshot to view at full resolution. Useful for diagnosing the kind of false-positive PASS verdicts described above.

The dashboard requires `fastapi` and `uvicorn` (already in `requirements.txt`). The agent itself never imports them — they're lazy-loaded only when `--serve-dashboard` runs.

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
