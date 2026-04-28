# Test Report — PR #1 (desktop VLM agent)

**PR**: https://github.com/nd1836017/https-github.com-nd1836017-desktop-vlm-agent/pull/1
**Session**: https://app.devin.ai/sessions/241ab3b350814b07ba833a4658f39d56
**Recording**: attached (`rec-5a3f3a0a-...-subtitled.mp4`)

## TL;DR

All three planned test flows executed. The agent works end-to-end: it drove Chrome from a blank new tab to https://example.com using a real Gemini VLM loop, with every step independently verified by a second VLM call. It also halts correctly when a step's outcome doesn't match the goal — proven live during a failed-click attempt.

## Escalations

1. **Gemini API quota** — the provided `GEMINI_API_KEY` has **0 free-tier quota for `gemini-2.0-flash`** (`RESOURCE_EXHAUSTED`, `limit: 0`). For the duration of testing I overrode `GEMINI_MODEL=gemini-2.5-flash` in `.env` (same SDK, same 0–1000 coord convention) — all other code paths are unchanged. You'll want to either enable billing on the Google Cloud project backing this key, or switch `GEMINI_MODEL` to `gemini-2.5-flash` in your `.env` when running it. [No code change needed.]
2. **Run-1 coordinate miss** — on the first live run (tasks: "Click the address bar, type example.com, press Enter"), the VLM emitted `CLICK [400,36]` → pixel `(640, 43)`, which is **above** the actual address bar on Chrome's title strip. The agent correctly detected this via the verification call (`VERDICT: FAIL — The Chrome address bar is not focused; no cursor or highlight is visible.`), halted, and exited with code 1. This is the intended halt-on-mismatch behavior, but it means the primary flow had to be re-run with a more robust task phrasing that used `Ctrl+L` (keyboard shortcut) instead of a coordinate-precision click. That is arguably a weakness of VLM coordinate grounding on small UI chrome — not an agent bug — but worth knowing when writing tasks.

## Test results

| # | Test | Result |
| --- | --- | --- |
| T1a | Live end-to-end — "click address bar, type, enter" | **Halted** by verify — correct halt-on-mismatch behavior (but did not reach example.com) |
| T1b | Live end-to-end — "Ctrl+L, type, enter" | **Passed** — navigated to example.com, 3/3 `VERDICT: PASS` |
| T2  | Retry-on-parse-failure harness (deterministic, 2 cases) | **Passed** |
| T3  | Verification-halt harness (deterministic, 2 cases) | **Passed** |
| Unit | Parser + coordinate scaling | **Passed** (21/21) |
| CI  | Lint + tests on Python 3.10 / 3.11 / 3.12 | **Passed** (3/3 green) |

**Full count after T2/T3 added**: 25/25 pytest pass.

## T1b — Live end-to-end (primary flow)

### Setup
```
GEMINI_MODEL=gemini-2.5-flash          # .env override, see Escalation 1
```
`tasks.txt`:
```
Press Ctrl+L to focus the browser's address bar
Type example.com
Press the Enter key to navigate to the URL
```

### Evidence (log excerpts)

| | |
| --- | --- |
| Geometry | `Detected screen geometry: 1600x1200` |
| Step 1 | `PRESS 'ctrl+l'` → `Verify: VERDICT: PASS — The address bar is focused, indicated by the blinking cursor.` |
| Step 2 | `TYPE 'example.com'` → `Verify: VERDICT: PASS — "example.com" has been typed into the address bar.` |
| Step 3 | `PRESS 'enter'` → `Verify: VERDICT: PASS — The browser successfully navigated to example.com.` |
| Exit | `[ok] All 3 step(s) completed successfully.` / exit code 0 |

### Before / After

| 🟢 Before (new tab) | 🟢 After (navigated to example.com) |
| --- | --- |
| ![before: Google new tab](https://app.devin.ai/attachments/4736268c-feeb-4b8d-8d19-3ee8d1b18e45/screenshot_96350d286a14432b818dc7f0c384e228.png) | ![after: example.com loaded](https://app.devin.ai/attachments/40f4e671-e85b-4125-b4e4-8b4631110057/screenshot_4c2927d7942b4d01b931075d50383492.png) |
| Chrome on `google.com` homepage | Tab title "Example Domain", URL bar `example.com`, page body "Example Domain ... This domain is for use in documentation examples..." |

### Assertion ledger (from the test plan)

| # | Expectation | Result |
| --- | --- | --- |
| A1 | Startup log reports real screen geometry | **PASS** — `1600x1200` (matches `xdpyinfo`) |
| A2 | Address bar focused after step 1 | **PASS** — verifier observed blinking cursor |
| A3 | `example.com` appears in address bar after step 2 | **PASS** — verifier quoted the literal string |
| A4 | Step 3 navigates to example.com | **PASS** — tab title "Example Domain", URL `example.com` |
| A5 | Exit code 0 | **PASS** |
| A6 | Three `VERDICT: PASS` lines in log | **PASS** — 3/3 |
| A7 | 1.5 s animation buffer honored after CLICK | **PASS (from T1a)** — `CLICK` at 19:09:59 → next `plan_action` at 19:10:01 = 2.0 s gap, consistent with 1.5 s buffer + ~0.5 s screenshot/SDK overhead. |

## T1a — Live halt-on-mismatch (side evidence)

`tasks.txt` (original, more adversarial — asked VLM to CLICK the small address bar):
```
Click on the Chrome address bar at the top of the window to focus it
Type the URL example.com
Press the Enter key to navigate
```

Log excerpt:
```
Step 1/3: Click on the Chrome address bar at the top of the window to focus it
CLICK normalized=(400,36) -> pixels=(640,43)
Verify: VERDICT: FAIL — The Chrome address bar is not focused; no cursor or highlight is visible.
[!] HALT at step 1/3: Click on the Chrome address bar at the top of the window to focus it
    Reason: VERDICT: FAIL — The Chrome address bar is not focused; no cursor or highlight is visible.
    The agent has stopped to prevent runaway actions.
Halting execution: VERDICT: FAIL — ...
```
Exit code: **1**. Steps 2 and 3 never ran.

![halt state: address bar unfocused](https://app.devin.ai/attachments/6f72558c-1744-4b5d-942a-c3966180cb52/screenshot_0b0b507abc904aeead5cc43a321b95f3.png)

This is exactly the spec requirement: "If the VLM responds that the state does not match the goal (e.g., the wrong app opened), the script must execution-halt and notify the user via the terminal."

## T2 — Retry on parse failure (deterministic)

Two cases in `tests/integration/test_retry_harness.py`:

1. **Recovers after one failed parse** — fake VLM returns prose first, `CLICK [100,200]` second.
   - `plan_action` called exactly 2× ✓
   - `pyautogui.click(192, 216)` called exactly once (correct 0–1000 → 1920×1080 scaling) ✓
   - Step result `passed=True` ✓
2. **Both attempts unparseable** — fake VLM returns prose both times.
   - `plan_action` called exactly 2× ✓
   - `pyautogui.click` never called ✓
   - Step result `passed=False`, reason contains `"parse failure"` ✓

## T3 — Verification halt (deterministic)

Two cases in the same file:

1. **Verdict FAIL on step 1/2** — run sees `plan_action` called only once; stderr contains `HALT at step 1/2` and the fake reason `wrong app opened`; exit code 1; step 2 never reached. ✓
2. **All PASS sanity** — with two passing steps, `run()` returns 0 and both steps are executed. ✓

## Raw artifacts

- Screen recording: `rec-5a3f3a0a-5992-4e96-845c-8729b4f75493-subtitled.mp4` (attached)
- Agent log (failed run): `test-run-halt.log` in repo
- Agent log (successful run): `test-run-success.log` in repo
- Harness: `tests/integration/test_retry_harness.py` in repo (temporary test artifact; remove before merge or keep as integration-test suite — your call)

## Would a broken implementation look identical?

No:
- Broken scaling → `pyautogui.click` in T2 would get wrong args → assertion fails.
- Broken retry → `plan_action` call count in T2 would be 1 or ≥ 3 → assertion fails.
- Broken halt → T3 exit code would be 0, stderr empty, `plan_action` called 2× → three assertions fail.
- Broken verifier wiring → T1b would either never halt (any garbage "passes") or always halt (can't pass 3 steps). Observing 3 legit PASS lines on the success run and 1 legit FAIL line on the halt run rules out both failure modes.
