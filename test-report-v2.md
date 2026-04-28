# Test Report — PR #2 (v2: memory + replan + checkpoint/resume)

**Status: PASS** (all three adversarial assertions met)

## Environment
- Model used for live runs: `gemini-2.5-flash-lite` (see "Notes" below for why I didn't use `gemini-3.1-flash-lite-preview` this time)
- `LOG_LEVEL=DEBUG`, `HISTORY_WINDOW=5`, `MAX_REPLANS_PER_STEP=2`, `STATE_FILE=.agent_state.json`
- Tasks file (`tasks.txt`):
  1. Press Ctrl+L to focus the browser address bar
  2. Type "example.com" into the address bar
  3. Press Enter to navigate to example.com

## A1 — Short-term memory is injected into step 2+ plan prompts

**How measured**: temporarily added `log.debug("plan_action prompt: %r", prompt)` in `agent/vlm.py` right before the `generate_content` call (reverted before finishing).

**Evidence** (from `run1.log`):

- Step 1's plan prompt:
  ```
  'Current step: Press Ctrl+L to focus the browser address bar\n\nRespond with ONE command only.'
  ```
  → no `Recent action history` block (correct: history is empty at start).

- Step 2's plan prompt:
  ```
  'Recent action history (most recent last):\n1. step=\'Press Ctrl+L to focus the browser address bar\' action=PRESS [ctrl+l] -> PASS (VERDICT: PASS — The browser address bar is focused, ...)\n\nCurrent step: Type "example.com" into the address bar\n\nRespond with ONE command only.'
  ```
  → contains `Recent action history`, the step-1 text, the literal action `PRESS [ctrl+l]`, and the PASS verdict.

- Step 3's plan prompt contains both prior records (step 1 and step 2) with their actions and verdicts.

**Verdict: PASS** — memory is threaded through the planner exactly as designed.

## A2 — Checkpoint file reflects last verified step

**Step 1 PASS, process killed before step 2 action** — inspected `.agent_state.json`:
```json
{
  "version": 1,
  "tasks_file": "tasks.txt",
  "total_steps": 3,
  "last_completed_step": 1
}
```

**After resumed run completes** — inspected again:
```json
{
  "version": 1,
  "tasks_file": "tasks.txt",
  "total_steps": 3,
  "last_completed_step": 3
}
```

**Verdict: PASS** — atomic JSON state is written right after verdict PASS, not before, and the 1 → 3 transition is correctly observed.

## A3 — `--resume` continues from the checkpoint, skips completed steps

**Evidence** (from `run3_resume.log`):

```
05:12:10 [INFO] agent.agent: Resuming from checkpoint: step 2/3 (file=.agent_state.json)
05:12:11 [INFO] agent.agent: Step 2/3: Type "example.com" into the address bar
05:12:12 [INFO] agent.agent: Action: TYPE [example.com]
05:12:14 [INFO] agent.agent: Verify: VERDICT: PASS — "example.com" was typed into the address bar ...
05:12:14 [INFO] agent.agent: Step 3/3: Press Enter to navigate to example.com
05:12:15 [INFO] agent.agent: Action: PRESS [enter]
05:12:17 [INFO] agent.agent: Verify: VERDICT: PASS — The screenshot shows the "Example Domain" website loaded ...
05:12:17 [INFO] agent.agent: All 3 step(s) completed successfully.
[ok] All 3 step(s) completed successfully.
```

Counts (`grep -c`):
- `"Step 1/3"` → **0** (step 1 correctly skipped)
- `"Step 2/3"` → 1
- `"Step 3/3"` → 1

**Verdict: PASS** — resume honored the checkpoint, step 1 was not re-executed, steps 2+3 completed cleanly.

## Regression — unit tests

- `pytest -q` → **49 passed** (covers parse-retry, replan budget, state I/O, etc.).
- Replan-budget exhaustion is covered by `tests/test_agent_loop.py::test_replan_budget_exhausts_and_halts`.
  Not attempted live because forcing a deterministic FAIL against real Gemini is flaky and would just burn quota.

## Notes / issues

1. **Model fallback**: my first two attempts at `gemini-3.1-flash-lite-preview` returned `503 UNAVAILABLE ("high demand")` from Google, so I switched the run to `gemini-2.5-flash-lite`. The three v2 features are model-agnostic — no code path depends on which model is used. You can switch `.env` back to `gemini-3.1-flash-lite-preview` any time.
2. **Free-tier 10 RPM cap hit once**: during my kill-and-resume sequence I made ~8 calls in under a minute and got a 429; waited ~60 s and resumed, completed cleanly. No retries or backoff are built into the agent today — if you want me to add tenacity-style backoff on 429/503 in a follow-up PR, say so.
3. **Temporary debug line reverted**: `log.debug("plan_action prompt: %r", prompt)` was added in-session to make A1 observable, then removed. No changes landed in the committed tree beyond what was already in PR #2.

## Artifacts

- Full run1 log lives at `run1.log` (not committed — gitignored by `*.log`, see `.gitignore`).
- Recording of the kill-and-resume flow attached to the PR message.
