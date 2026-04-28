# Test Plan — desktop VLM agent PR #1

**PR**: https://github.com/nd1836017/https-github.com-nd1836017-desktop-vlm-agent/pull/1
**Session**: https://app.devin.ai/sessions/241ab3b350814b07ba833a4658f39d56
**CI status at plan time**: 3/3 green (test 3.10 / 3.11 / 3.12), 0 PR comments.

## What changed (user-visible)

A brand-new Python agent that reads `tasks.txt`, captures a screenshot, asks Gemini 2.0 Flash what to do, parses `CLICK [X,Y] | PRESS [KEY] | TYPE [TEXT]` from the response, executes it via `pyautogui`, then verifies the result with a second VLM call. Halts on any verification mismatch.

## Testing environment

| | |
| --- | --- |
| OS | Ubuntu 22.04 / live X display, Chrome already running on the Google homepage (1024×768) |
| Python | 3.10 venv with `pip install -r requirements.txt pytest ruff` already done |
| VLM | Gemini 2.0 Flash via the live `google-genai` SDK using `$GEMINI_API_KEY` |
| Target app | Google Chrome (already open) — used purely as a visible target that the VLM can interact with through its address bar |

## Tests

### T1 — End-to-end: agent drives Chrome's address bar via Gemini 2.0 Flash (PRIMARY)

**Goal**: prove the full visual-action loop works live — screenshot → VLM → parse → pyautogui → screenshot → VLM verify — with all three command types exercised.

`tasks.txt` for this test:
```
Click on the Chrome address bar at the top of the window
Type the URL example.com
Press the Enter key to navigate
```

**Steps** (each step takes ~5–15 s including two Gemini calls):

1. Maximize the Chrome window (`wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz`).
2. Write the `tasks.txt` above, place `$GEMINI_API_KEY` in `.env`, start screen recording.
3. `python -m agent` in the foreground.

**Concrete assertions** (all must pass):

| # | Expectation | How I'll verify |
| --- | --- | --- |
| A1 | Startup log line `Detected screen geometry: WxH` matches the result of `xrandr`/`wmctrl` (not hard-coded, not 0×0). | grep stdout for "Detected screen geometry:" and sanity-check against display size. |
| A2 | Step 1 ("click address bar") results in a `CLICK [x,y]` command AND the address bar actually becomes focused (blue outline / text cursor visible in the screenshot). | Screenshot after step 1 must show a focused, highlighted address bar. |
| A3 | Step 2 ("type example.com") actually types `example.com` into the focused address bar. | Screenshot after step 2 must show the literal string `example.com` in the URL bar (not elsewhere). |
| A4 | Step 3 ("press Enter") navigates to `example.com`. | After step 3 completes, the URL bar shows `example.com` and the page title contains `Example Domain` (visible as "Example Domain" header). |
| A5 | The agent exits with code 0. | `echo $?` is `0`. |
| A6 | After each step, a VLM verify call returns `VERDICT: PASS` (check agent log). | stdout contains three `Verify: VERDICT: PASS` lines. |
| A7 | The spec's 1.5-s animation buffer is honored after CLICK. | Log timestamps between the CLICK log line and the next plan_action log line are ≥ 1.5 s apart. |

**Would this look identical if the code were broken?** No:
- If scaling were wrong → click would miss the address bar (A2 fails).
- If the parser were broken → no command executed (A2/A3 fail).
- If FAILSAFE or the typing path were broken → `example.com` would be missing or corrupted (A3 fails).
- If the verifier were wired backwards → agent would halt (A5/A6 fail).
- If the animation buffer were removed → screenshots would be taken before the address bar focus animation settled, and the VLM would likely verify FAIL on step 1 (A6 fails).

### T2 — Parse-failure retry harness (DETERMINISTIC)

**Goal**: prove the "retry step once on parse failure" behaviour required by the spec.

**Setup**: a tiny Python harness (`tests/integration/test_retry_harness.py`, **temporary, not committed**) that runs `agent.agent.run_step` with a `FakeGeminiClient` whose `plan_action` returns:
  1. First call: `"I'm not sure what to do here, but maybe try clicking?"` (no bracketed command — unparseable).
  2. Second call: `"CLICK [100,200]"`.
And whose `verify` returns `VerificationResult(passed=True, reason="VERDICT: PASS — ok")`.

Execution intercepts `pyautogui` via monkeypatch so no real clicks happen.

**Concrete assertions**:

| # | Expectation | How I'll verify |
| --- | --- | --- |
| B1 | `plan_action` is called exactly **twice** (one failed parse, one successful). | Counter in the fake; expected value `2`. |
| B2 | `pyautogui.click` is called exactly **once** with arguments `(round(100 * w / 1000), round(200 * h / 1000))` where `w,h` are the harness-provided geometry (e.g. 1920×1080 → `(192, 216)`). | Mock capture; exact tuple compared. |
| B3 | The returned `VerificationResult.passed` is `True`. | Direct assertion. |

### T3 — Verification-halt harness (DETERMINISTIC)

**Goal**: prove the agent halts and notifies the user when the verifier says the state doesn't match the goal.

**Setup**: same harness style as T2, but `plan_action` returns a valid `CLICK [500,500]` and `verify` returns `VerificationResult(passed=False, reason="VERDICT: FAIL — wrong app opened")`.

**Concrete assertions**:

| # | Expectation | How I'll verify |
| --- | --- | --- |
| C1 | `agent.run` returns exit code `1`. | Direct return value. |
| C2 | stderr contains the substring `HALT at step 1/1` and the reason `VERDICT: FAIL — wrong app opened`. | Capture stderr via `capsys`. |
| C3 | Subsequent steps are **not** executed even if present. | Add a second step to `tasks.txt` and confirm `plan_action` was called only once. |

## Out of scope

- Windows/macOS behaviour (agent only runs on Linux in this environment).
- DPI scaling correctness on a HiDPI monitor (the spec target is the 1024×768 test display, and the scaling math is covered by unit tests already passing in CI).
- `pyautogui.FAILSAFE` itself — documented but not executable in this session (would require physically slamming the mouse into a corner and crashing the agent, which is in the human reviewer's checklist in the PR body).

## Reporting

One GitHub comment on PR #1 with a collapsible summary of T1 / T2 / T3 results, pre-expanded screenshot panels for T1 (before/after the address bar is populated), and a link to this session.
