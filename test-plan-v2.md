# Test Plan — PR #2 (v2: memory + replan + checkpoint/resume)

## What changed (user-visible)

1. **Short-term memory** — each subsequent `plan_action` prompt now contains a rolling summary of prior `(step, action, verdict)` records.
2. **Replan-on-failure with budget** — verifier FAIL triggers up to `MAX_REPLANS_PER_STEP` (default 2) additional plan attempts, each primed with the previous failure reason, before halting.
3. **Checkpoint + resume** — `.agent_state.json` is written atomically after each verified step; `python -m agent --resume` continues from `last_completed_step + 1`.

## Primary live end-to-end flow (one recording)

Same browser-based driver flow as PR #1 testing:
1. Task file (3 steps):
   - `Press Ctrl+L to focus the browser address bar`
   - `Type "example.com" into the address bar`
   - `Press Enter to navigate to example.com`

2. Run: `python -m agent` on the real desktop with real Gemini calls (`GEMINI_MODEL=gemini-3.1-flash-lite-preview`, `LOG_LEVEL=DEBUG`, `HISTORY_WINDOW=5`). Capture full stdout/stderr to a log file.

3. Kill the process with Ctrl-C **after step 2 is logged `Verify: VERDICT: PASS …`** but **before step 3 begins**. (Step 3 triggers navigation which is destructive for the tab — killing earlier lets us demo resume on that step.)

4. Re-run with `python -m agent --resume`. Capture the second log.

## Adversarial assertions

Each assertion below is worded so the test **would visibly fail if the change were broken**.

### A1 — Short-term memory is injected into step 2's plan prompt
**Evidence source**: first log file.
**Expected**: grep for `plan_action prompt:` logs.
- The **first** matching log line (step 1's plan prompt) **must NOT** contain `"Recent action history"`.
- The **second** matching log line (step 2's plan prompt) **must** contain `"Recent action history (most recent last):"` **and** the literal text `"Press Ctrl+L to focus the browser address bar"` (the step-1 text) **and** `"PRESS [ctrl+l]"` (step-1's action).

> Why this is adversarial: if the v2 memory plumbing were broken (e.g. history not threaded, window=0 off-by-one, or summary empty after a PASS), the step-2 prompt would be identical to step-1's. That mismatch is directly visible in the log grep.

### A2 — Checkpoint file reflects the last verified step
**Evidence source**: `.agent_state.json` on disk, inspected at two points.

- **After step 1 PASS, before killing step 3**: the file must contain `"last_completed_step": 2` and `"total_steps": 3`. (I wait for step 2's PASS before killing so we have two completed steps banked.)
- **After the resumed run completes**: the file must contain `"last_completed_step": 3`.

> Why this is adversarial: if save_state were misplaced (e.g. called before verification, called only at the end, or the index were wrong), the numbers in the JSON would be visibly off. Hard-coded expected values.

### A3 — `--resume` starts at the correct step, skips the completed ones
**Evidence source**: second log file (after `--resume`).

- The startup log must contain the literal line `"Resuming from checkpoint: step 3/3"`.
- The log must **NOT** contain `"Step 1/3:"` or `"Step 2/3:"` anywhere.
- The log **must** contain exactly one `"Step 3/3:"` block.

> Why this is adversarial: if resume were broken (e.g. it ignored the checkpoint, started over, or mis-offset the index), you'd see `Step 1/3:` reappear, which I'm asserting is absent.

## Regression sanity (quick)

- **R1**: `pytest -q` on the current branch reports `49 passed`. (Proves parse-retry / replan budget / state I/O unit tests still work — no need to re-demo these in the live recording.)

## What I won't do in the live recording

- **Replan budget exhaustion live**: the verifier FAIL path is already deterministically tested by `test_replan_budget_exhausts_and_halts` (and 3 related tests). Trying to force a live FAIL against real Gemini is flaky and would just burn quota. I'll cite the unit test result in the report and skip a live demo unless the agent happens to hit a real miss-then-recover during the primary flow (in which case I'll annotate it).
- Login / setup steps (already configured; `.env` has the key).

## Temporary testing-only diff

I will add ONE line to `agent/vlm.py`:

```python
log.debug("plan_action prompt: %r", prompt)
```

right before the `generate_content` call. This is needed to make assertion A1 verifiable without mocking. I will revert this line before exiting test mode.

## Recording plan

- Start a screen recording covering: the terminal running the agent (side-by-side with the Chrome window), the kill, the `cat .agent_state.json` check, and the resume run.
- Annotate at: run start, step 2 verify PASS, Ctrl-C, JSON inspection, resume start, resume success.
