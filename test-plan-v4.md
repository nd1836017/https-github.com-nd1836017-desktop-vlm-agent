# PR #4 — Live test plan: YouTube login (bot-detection dodge)

## Scope
Drive the agent to log in to YouTube with the secondary Google account
(`GOOGLE_TEST_EMAIL` / `GOOGLE_TEST_PASSWORD`, both user-scope secrets) and
prove that the new v4 features are active and working:

1. **Retry-with-backoff on 429/5xx** for every Gemini call.
   - No fallback to another model. Pinned to `gemini-3.1-flash-lite-preview`.
2. **Pre-CLICK human delay** — random sleep `[0.8s, 2.0s]` before every click.
3. **Per-keystroke TYPE jitter** — `[0.03s, 0.12s]` between every character.

The primary metric is "does Google's sign-in page NOT trip a bot-detection
challenge when the agent drives the click and typing?". Secondary metrics are
log-visible evidence that the three features are firing (or, for retry,
unit-test coverage + opportunistic live evidence only).

## Out-of-scope (do NOT include)
- Opening email, inbox search, file lookup — deferred per user instruction.
- Saving browser profile — user explicitly declined until personal machine.
- Regression of PR #1–#3 features (covered by prior live tests + 93-test suite).

## Pre-conditions (already satisfied)
- Chrome is running in the VM, currently on `youtube.com` with a visible
  "Sign in" button in the top-right (confirmed logged-out). Screenshot:
  `~/screenshots/screenshot_620559d197404d4f902831e4f0c29215.png`.
- `GOOGLE_TEST_EMAIL` + `GOOGLE_TEST_PASSWORD` present in env.
- `.env` pins `GEMINI_MODEL=gemini-3.1-flash-lite-preview`.
- PR #4 CI green on Python 3.10/3.11/3.12 (3/3).

## Fixture preparation (at run-time, not committed)
A rendered task file `tasks_yt_login.txt` is generated right before the run
with the email substituted in but the password NOT written to disk — the
password step will be typed via a tasks step that says literally
`Type the password from the GOOGLE_TEST_PASSWORD env var`, which the VLM
cannot resolve by itself. So instead:

- **Two runs**, each purpose-built:
  - **Run A (primary, live)**: render a tasks file that contains only
    `Click the 'Sign in' button in the top-right corner` and
    `Type the email: <email>` and `Click the 'Next' button`. Stop after the
    Next click that submits email, since the password screen typically
    redirects to a different "Not secure browser" interstitial that we want
    to inspect independently. Covers all three new features (CLICK ×2,
    TYPE ×1).
  - **Run B (secondary, deferred)**: Only if Run A completes cleanly and no
    bot-detection modal fires. Render a tasks file with the password step
    added, type the password, click Next, verify signed-in.

`tasks_yt_login.txt` is git-ignored (added to `.gitignore`).

## Assertions (adversarial — each designed to fail visibly if the feature is broken)

### A1 — Pre-CLICK human delay fires on every CLICK
**What we check:** the agent's log stream contains exactly one
`Pre-click human delay: X.YZs` line immediately before every `CLICK pixels=`
line, and X.YZ is in the range `[0.8, 2.0]`.

**Why it's adversarial:**
- If v4 is reverted / the delay is disabled / the range is zeroed, the log
  line would be absent or would report `0.00s`.
- If the delay is applied unconditionally but ignores the env range (e.g.
  hard-coded `0.1s`), the value would be outside `[0.8, 2.0]` and A1 fails.
- If the delay is applied at a different time (e.g. AFTER the click instead
  of before), the log ordering flips relative to `CLICK pixels=`.

**Pass criteria:**
- N CLICK commands ⇒ exactly N `Pre-click human delay:` log lines.
- Each delay value is ≥ 0.80 AND ≤ 2.00.
- Each delay log appears BEFORE the next `CLICK pixels=` line.

### A2 — TYPE cadence is jittered (per-character loop)
**What we check:** the wall-clock duration of the TYPE step for the email
is consistent with a per-character jittered loop, i.e. it takes at least
`len(email) * TYPE_MIN_INTERVAL_SECONDS` = `len(email) * 0.03` seconds.

**Why it's adversarial:**
- If v4 is reverted, the old code path uses `pyautogui.typewrite(text,
  interval=0.02)` which is uniform and much faster (~`len * 0.02` s).
  For a 20-char email the delta is ~0.2s vs ~1.5s — easily measurable at
  log timestamp resolution (ms).
- If the loop is present but the jitter range is bugged (both 0), total
  would be ~0 and A2 fails.

**Pass criteria:**
- Measure `ts(plan_action for next step) - ts(TYPE <email> log line)`.
  Expected: between `len(email) * 0.03 + 0.25` (lower bound, min jitter +
  post-TYPE sleep) and `len(email) * 0.12 + 0.25 + 2.0` (upper bound).
- Accept any value ≥ `len(email) * 0.03 + 0.25`. This distinguishes the
  new loop from the old uniform path.

### A3 — Login flow completes without bot-detection challenge
**What we check:** after the "Next" click that submits the email, the
resulting page is one of:
- Google's normal password entry page (URL still `accounts.google.com/...`,
  page shows a password input + "Forgot password" link + a Next button).

And is NOT one of:
- "Couldn't sign you in — This browser or app may not be secure".
- "Verify it's you" CAPTCHA / phone challenge.
- A silent redirect back to youtube.com signed-out.

**Why it's adversarial:**
- This is the feature's end-user metric. If v4's human cadence changes are
  insufficient to dodge Google's automation heuristics, A3 fails and the
  recording will visibly show the interstitial.
- A broken change that skipped the delay entirely would likely trigger this
  more often than not (Google flags perfectly-uniform input cadence).

**Pass criteria:**
- Final screenshot shows a password field AND the text "Welcome" or
  "Enter your password" AND the entered email as the account identifier.
- DOM inspection confirms: an `input[type=password]` exists.

### A4 — Retry-with-backoff fires on transient Gemini error (OPPORTUNISTIC)
**What we check:** IF a 429 or 5xx is returned during the run, the log
stream contains `plan_action: 503 ...` or `... 429 ...` followed by
`retry N/M after Z.Zs` followed by eventual success OR halt on exhaustion.

**Why it's opportunistic:** we cannot force the server to 503 without
breaking the API contract. This assertion is marked UNTESTED in the report
if no transient error occurs during the run. Unit tests in
`tests/test_retry_backoff.py` cover 8 code paths including exhaustion.

**Pass criteria:** if triggered, logs show the retry pattern AND the run
either completes or halts with a clean message (not a raw Python traceback).

### A5 — No fallback to `gemini-2.5-flash-lite`
**What we check:** `grep -c 'gemini-2.5-flash-lite' run.log` → 0.
All `plan_action:` and `verify:` log lines must reference
`gemini-3.1-flash-lite-preview` or no model name at all.

**Why it's adversarial:** the user explicitly forbade falling back. Even
on 503 exhaustion, the agent must halt with the primary model name in
the error message, not switch.

**Pass criteria:** zero occurrences of `2.5-flash-lite` in the run log.

## Execution plan
1. `cd /home/ubuntu/repos/desktop-vlm-agent && git fetch origin && git checkout devin/1777013535-retry-click-delay` (PR #4 branch).
2. Render `tasks_yt_login.txt` from an env var template.
3. Navigate Chrome to `https://www.youtube.com/` (fresh tab — NOT the existing watch page, so we start from a clean header with a clickable "Sign in"). Wait for page load.
4. Start screen recording (`computer record_start`).
5. Run the agent: `python -m agent 2>&1 | tee ~/run_pr4.log`.
6. Observe in real time.
7. Stop recording once the flow settles (either A3 pass or fail).
8. Post-process the log, run assertions A1–A5 via small `grep` one-liners.
9. Take a final screenshot of the page state.
10. Render a report (`test-report-v4.md`) with screenshot evidence + log excerpts + pass/fail per assertion.
11. Post ONE comment on PR #4 with the summary, pre-expanded "Assertions" section.

## Failure branches and recovery
- **Run fails on A3 (bot challenge)**: report FAIL, attach screenshot of
  the challenge, STOP. Do NOT retry (would burn quota and potentially
  lock the account). User decides next steps (e.g. bump delays higher,
  add mouse-move jitter, use residential IP).
- **Run fails on A1 (pre-click delay missing)**: bug in v4. Exit test
  mode, re-open the code, fix, push, re-test.
- **Run dies on 429 exhaustion (A4 triggers but fails)**: mark A4 as
  "fired but exhausted, not a v4 bug — quota pressure". User can bump
  `GEMINI_RETRY_MAX_ATTEMPTS`. Do NOT retry immediately.
- **Parse/verify replan loop eats budget**: expected from PR #2; halt is
  the correct behavior, not a v4 regression.
