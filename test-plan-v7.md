# Test Plan — PR #7 (Tier 3: run artifacts + PAUSE + RPD guard)

## What changed
Three independent features wired end-to-end for surviving long/complex runs:
1. **`SAVE_RUN_ARTIFACTS=true`** → every step writes `runs/<ts>/step_NNN_{before,after}.png + plan.txt + verdict.txt`, plus a rolling `summary.json`.
2. **`PAUSE [REASON]`** planner primitive → planner auto-emits it on 2FA / "Verify it's you" / CAPTCHA; agent halts, blocks on stdin, resumes on Enter (or aborts on `q`). Does **not** consume the replan budget.
3. **RPD quota guard** — one-shot WARNING at `RPD_WARN_THRESHOLD` (default 75%), clean halt at `RPD_HALT_THRESHOLD` (default 95%) between steps with the checkpoint intact.

Code refs:
- `agent/artifacts.py` (new)
- `agent/cost.py` (new)
- `agent/parser.py` — `PauseCommand` + `_PAUSE_RE` ordered before `_CLICK_TEXT_RE`
- `agent/vlm.py` — `reason` field on `PlanResponseModel`; `PAUSE` branch in `plan_response_to_command`; `RpdGuard.record()` wired into `_generate()`
- `agent/agent.py` lines 534-620 — `RpdGuard` + `ArtifactWriter` init; PAUSE handling in `_attempt_step` → `run_step`; `rpd_guard.should_halt()` between steps

## Scope
**ONE combined live run + ONE targeted live run** — designed so a broken implementation of any of the three features produces visibly different output.

---

## Run 1 — Run-artifacts + RPD halt (combined, single invocation)

### Setup
- Empty `runs/` (on disk, not committed)
- `.env` override for this run only:
  - `SAVE_RUN_ARTIFACTS=true`
  - `RPD_LIMIT=2`
  - `RPD_WARN_THRESHOLD=0.5`
  - `RPD_HALT_THRESHOLD=0.95`
  - `MAX_REPLANS_PER_STEP=0` (no replans — keep math predictable)
- `tasks.txt`:
  ```
  Wait for 0.5 seconds.
  Wait for 0.5 seconds.
  Wait for 0.5 seconds.
  ```
  (3 WAIT-only steps — no mouse/keyboard side effects; planner should emit `WAIT [0.5]` each time)

### Expected behavior (why broken impl looks different)
With `RPD_LIMIT=2`, WARN at 50%, HALT at 95%:
- Step 1 plan → `calls=1`. `1 ≥ 0.5*2 = 1.0` → **WARN fires**.
- Step 1 verify → `calls=2`.
- End of step 1: `should_halt` → `2 ≥ 0.95*2 = 1.9` → **HALT between steps**.
- Step 2 and 3 never run.
- Artifact writer captures ONLY step 1 (before + after + plan + verdict + 1-entry summary).

If RPD guard is broken (never halts): steps 2 and 3 also run, summary.json has 3 entries, exit 0.
If artifact writer is broken: `runs/` never appears or files are missing.
If WARN threshold is broken: no `approaching daily quota` log line.

### Assertions (Run 1)
| # | Assertion | Expected | Would differ if broken? |
|---|---|---|---|
| A1 | Exit code | `1` | Yes — would be `0` if RPD halt never fires |
| A2 | stderr last line | contains `RPD halt threshold reached: 2 / 2` and `Checkpoint saved` | Yes — absent if halt path not taken |
| A3 | Log contains `approaching daily quota` exactly once | Yes, with `1 / 2 (50%)` | Yes — absent if warn broken |
| A4 | `runs/<single-ts>/` exists on disk | Yes | Yes — won't exist if artifact writer off |
| A5 | `runs/<ts>/step_001_before.png` has PNG magic bytes `89 50 4E 47` and size > 1KB | Yes | Yes — broken writer → absent/empty |
| A6 | `runs/<ts>/step_001_after.png` has PNG magic bytes and size > 1KB | Yes | Yes |
| A7 | `runs/<ts>/step_001_plan.txt` contains both `# action_text` and `# raw VLM response` | Yes | Yes — broken formatter → missing sections |
| A8 | `runs/<ts>/step_001_verdict.txt` starts with `VERDICT:` line | Yes (`VERDICT: PASS` expected) | Yes — broken writer → absent |
| A9 | `runs/<ts>/summary.json` parses as JSON, is a list, has **exactly 1** entry with `step_idx=1`, `passed=true` | Yes | Yes — would have 3 entries if halt broken; 0 entries if append_summary broken |
| A10 | No `step_002_*` or `step_003_*` files in `runs/<ts>/` | Yes | Yes — presence proves RPD halt didn't fire |
| A11 | `.agent_state.json` written with `last_completed_step=1` | Yes | Yes — broken checkpoint-before-halt path |

---

## Run 2 — PAUSE live trigger + user abort

### Setup
- Local HTML file `pause_demo.html` with content that mimics a real 2FA / verify-it's-you screen:
  ```html
  <h1>Verify it's you</h1>
  <p>Google sent a notification to your phone. Tap Yes, then enter 75 when prompted.</p>
  <p>2-Step Verification — waiting for device approval.</p>
  ```
- Opened in Chrome in full-screen.
- `.env` restored to defaults (`RPD_LIMIT=500`, `SAVE_RUN_ARTIFACTS=false`).
- `tasks.txt`:
  ```
  Proceed past the verification screen shown in the browser.
  ```
- Invocation: `echo "q" | python -m agent 2>&1 | tee run2.log` — feeds `q` into the PAUSE prompt to exercise the **abort** path.

### Expected behavior (why broken impl looks different)
- VLM sees "Verify it's you" + "2-Step Verification" → emits `PAUSE [Verify it's you prompt]` (or similar reason).
- Agent detects `PauseCommand`, returns `PauseRequested`.
- `_handle_pause()` prints `[!] PAUSE: <reason>` to stderr and calls `input(">>> Resume? [Enter / q]: ")`.
- Stdin delivers `q` → handler returns `False` → verdict becomes `User aborted at PAUSE: <reason>` → exit 1.
- Crucially: NO mouse/keyboard action fired against the browser (PAUSE never runs `execute()`).

If PAUSE detection broken (VLM wires prompt wrong): planner returns CLICK or TYPE, agent actually clicks something on the page. Log shows `Pre-click human delay` or `Action: CLICK` instead of `PAUSE`.
If parser precedence broken: `CLICK_TEXT` might match "Verify" first — log shows `CLICK_TEXT [Verify]`, not `PAUSE`.
If stdin handler broken: process hangs indefinitely (timeout = fail).

### Assertions (Run 2)
| # | Assertion | Expected | Would differ if broken? |
|---|---|---|---|
| B1 | stderr contains `[!] PAUSE:` line | Yes | Yes — absent if VLM didn't emit PAUSE |
| B2 | stderr contains `>>> Resume? [Enter / q]:` prompt | Yes | Yes — absent if handler broken |
| B3 | Log contains `Agent paused — waiting for human:` WARNING | Yes | Yes |
| B4 | Exit code | `1` | Yes |
| B5 | Final HALT message contains `User aborted at PAUSE` | Yes | Yes — would say something else if abort path broken |
| B6 | Log does **NOT** contain `Pre-click human delay` or `Action: CLICK` (proves no click was fired) | Yes | Yes — broken PAUSE routing → real click fires |
| B7 | Log does **NOT** contain `CLICK_TEXT [` (proves parser precedence correct) | Yes | Yes |
| B8 | Chrome URL bar still shows `file://.../pause_demo.html` (page unchanged, no navigation) | Yes | Yes — proves zero side-effects from PAUSE path |
| B9 | Process terminates within 30 seconds of stdin receiving `q` | Yes | Yes — timeout if stdin handler hangs |

---

## Regression (not recorded, not main focus)
- `pytest -q` → expect 171 passing (same as PR-creation-time run)
- `ruff check .` → expect clean

Labeled explicitly as **Regression** — will include terse one-line result only.

---

## Recording plan
- Single recording covering Run 1 + Run 2.
- Annotations:
  - `setup` before each run
  - `test_start` at start of each assertion group (A*, B*)
  - `assertion` with `test_result` at each pass/fail

Will NOT record the pytest regression (shell-only).

---

## Known caveat
Gemini `gemini-3.1-flash-lite-preview` has been intermittently returning 503s recently. The retry-with-backoff from PR #4 will handle transient failures, but if the model is fully unavailable the runs will halt with `Retry budget exhausted`. That would be a Gemini outage, not a PR #7 bug — will flag clearly in the report if it happens and retry later.

The agent is pinned to 3.1-flash-lite-preview per user instruction; NO fallback to 2.5-flash-lite.
