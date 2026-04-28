# PR #3 — Two-stage CLICK test plan

## What changed (user-visible)
- When the planner VLM emits a coarse `CLICK [x,y]`, the agent no longer clicks it blindly. It crops a 300×300 region around that point, asks the VLM to list all plausible targets in the crop, and (if more than one) runs a second "disambiguate" call that picks the target matching the step. The final pixel is computed from the crop — not from the coarse point.
- Toggle per run with `--two-stage-click` / `--no-two-stage-click`. `.env` default is ON.

Evidence from code (files informing the plan):
- `agent/agent.py:50-105` (`_execute_click_two_stage`) — logs `Two-stage CLICK: coarse norm=...; crop origin=...`, `Refine found K candidate(s): [...]`, and `Disambiguator picked index N / K` when K>1. Never falls back to the coarse pixel if refine is empty or PICK is 0 — it synthesises a FAIL verdict.
- `agent/screen.py:101-145` (`crop_around`) — edge-clamping.
- `agent/executor.py:35-49` (`execute_click_pixels`) — logs `CLICK pixels=(px,py)`.
- `agent/__main__.py:27-47` — CLI flags override env.

## Primary flow (single end-to-end test on the real stack)

Launch the agent on `tasks_youtube.txt` with two-stage CLICK **enabled** (default). The task:
1. Focus the address bar (PRESS Ctrl+L — does NOT trigger two-stage).
2. Type `youtube.com` (TYPE — does NOT trigger two-stage).
3. Press Enter to navigate (PRESS — does NOT trigger two-stage).
4. Click the first video thumbnail on the home grid (CLICK — **triggers two-stage; the crop will contain multiple thumbnail candidates and force the disambiguation path**).

Then immediately re-run the same task with `--no-two-stage-click` once, to confirm the flag actually disables the path.

## Assertions (each chosen so a broken implementation produces visibly different output)

### Run A — two-stage ON (default)

Environment: `ENABLE_TWO_STAGE_CLICK=true` (default). No CLI flag.

| # | Assertion | Expected observable | Would look different if broken? |
|---|---|---|---|
| A1 | Two-stage path fires on the CLICK step | Log contains `Two-stage CLICK: coarse norm=(` followed by `crop origin=` on step 4 | If `_execute_click_two_stage` isn't wired, the line never appears. |
| A2 | Refine call returns ≥1 candidate for the thumbnail grid | Log contains `Refine found` with a number ≥ 1 and a non-empty `[(x,y), …]` list | If refine fails to parse or returns 0, the agent FAILs the step — test would fail visibly. |
| A3 | Disambiguator fires when K>1 on the grid (the normal case) | Log contains `Disambiguator picked index N / K` with `K >= 2` and `1 <= N <= K` | If wiring is wrong and disambig is skipped, this line is absent entirely. |
| A4 | Executor clicks at the refined pixel, NOT the coarse point | Log contains `CLICK pixels=(px,py)`. The `(px,py)` must differ from the naive coarse-pixel that `CLICK [x,y]` would have produced on a 1600×1120 screen | If two-stage is a no-op, `px,py` == `coarse_x * 1.6, coarse_y * 1.12` exactly. We'll compute both and confirm inequality. |
| A5 | A YouTube video actually opens | Post-run, the Chrome URL (via `google-chrome` window-title / DOM) starts with `https://www.youtube.com/watch?v=` | If the agent clicks something non-video (logo, sidebar item), URL is `/` or `/channel/...` and this fails. |
| A6 | Agent exits 0 with 4× `VERDICT: PASS` | exit code 0; `grep -c 'VERDICT: PASS' run_a.log == 4` | If any step fails and the replan budget is exhausted, exit code 1. |

### Run B — two-stage OFF (`--no-two-stage-click`)

Same task, same `.env`, CLI override only.

| # | Assertion | Expected observable | Would look different if broken? |
|---|---|---|---|
| B1 | Two-stage path is skipped entirely | `grep -c 'Two-stage CLICK' run_b.log == 0` AND `grep -c 'Refine found' run_b.log == 0` AND `grep -c 'Disambiguator picked' run_b.log == 0` | If the CLI flag doesn't override the env, these lines will appear. |
| B2 | Legacy coarse-click path is used | Log contains a line of the form `CLICK normalized=(X,Y) -> pixels=(px,py)` from `executor.execute` (the pre-PR-3 code path) | If `_execute_click_two_stage` is accidentally called anyway, we'd see `CLICK pixels=` instead. |

### Regression (shell-only)

| # | Assertion | Expected observable |
|---|---|---|
| R1 | Full unit suite green on this branch | `pytest -q` returns exit 0 and `75 passed` |
| R2 | `ruff check` clean | exit 0, `All checks passed!` |

## Out of scope (explicit)
- Login / email / email-file lookup — user said "dont do the rest until i told you to".
- Perf / latency benchmarks.
- Headless CI runs (already green on GitHub Actions).

## Pass/fail decision rule
The test run is **PASSED** only if A1–A6 AND B1–B2 AND R1–R2 all pass. Any single failure → report as FAILED with the failing assertion called out first.
