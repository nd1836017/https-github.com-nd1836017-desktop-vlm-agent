# Test Report — PR #7 (Tier 3: artifacts + PAUSE + RPD guard)

## Headline
**All 20 assertions PASS, both live runs clean, zero side-effects.**
- Run 1 (artifacts + RPD halt): 11/11 pass, exit 1, halt fires between step 1 and step 2 as designed.
- Run 2 (PAUSE live): 9/9 pass, exit 1, VLM emitted `PAUSE` on 2FA-like page, stdin abort cleaned up.
- Regression: `pytest -q` = **171 passed**, `ruff check .` clean.

No bot detection, no model fallbacks. Both runs on `gemini-3.1-flash-lite-preview` (pinned per user instruction).

---

## Run 1 — Run-artifacts + RPD halt (combined)

### Setup
- `RPD_LIMIT=2`, `RPD_WARN_THRESHOLD=0.5`, `RPD_HALT_THRESHOLD=0.95`
- `SAVE_RUN_ARTIFACTS=true`, `MAX_REPLANS_PER_STEP=0`
- `tasks_run1.txt`: 3× `Wait for 0.5 seconds.`

### Key log excerpts
```
12:11:40 [WARNING] agent.cost: RPD usage at 1 / 2 (50%) — approaching daily quota.
12:11:51 [INFO] agent.agent: Action: WAIT [0.5]
12:11:52 [INFO] agent.executor: WAIT 0.50s
12:12:04 [INFO] agent.agent: Verify: The screen remains in the expected state after the wait period.

[!] RPD halt threshold reached: 2 / 2 (100%). Checkpoint saved — resume tomorrow or raise RPD_LIMIT.
12:12:04 [ERROR] agent.agent: Halting execution: RPD halt threshold reached: 2 / 2 (100%)
```

Python exit code: **1** (confirmed on a second clean invocation — `EXIT=1`).

### Artifact directory (`runs/20260424-121139/`)
| File | Size | Notes |
|---|---|---|
| `step_001_before.png` | 235,348 B | valid PNG |
| `step_001_after.png` | 235,371 B | valid PNG |
| `step_001_plan.txt` | 81 B | contains both `# action_text` and `# raw VLM response` |
| `step_001_verdict.txt` | 86 B | `VERDICT: PASS` |
| `summary.json` | 208 B | **exactly 1 entry** (not 3) — proves halt between steps |

**No `step_002_*` or `step_003_*` files exist** — filesystem-side proof the RPD halt fires *between* step 1 and step 2, not after the whole run.

`summary.json`:
```json
[
  {
    "step_idx": 1,
    "step": "Wait for 0.5 seconds.",
    "action": "<see step_NNN_plan.txt>",
    "passed": true,
    "reason": "The screen remains in the expected state after the wait period."
  }
]
```

`.agent_state_run1.json`:
```json
{"version": 1, "tasks_file": "tasks_run1.txt", "total_steps": 3, "last_completed_step": 1}
```

### Assertions (Run 1)
| # | Assertion | Result |
|---|---|---|
| A1 | Exit code = 1 | PASS |
| A2 | stderr has `RPD halt threshold reached: 2 / 2` + `Checkpoint saved` | PASS |
| A3 | Log has `approaching daily quota` exactly once | PASS (at call 1, 50%) |
| A4 | `runs/<ts>/` exists | PASS |
| A5 | `step_001_before.png` valid PNG > 1KB | PASS (235,348 B) |
| A6 | `step_001_after.png` valid PNG > 1KB | PASS (235,371 B) |
| A7 | `step_001_plan.txt` has both section headers | PASS |
| A8 | `step_001_verdict.txt` starts with `VERDICT:` | PASS (`VERDICT: PASS`) |
| A9 | `summary.json` has exactly 1 entry, `passed=true` | PASS |
| A10 | No `step_002_*` / `step_003_*` files | PASS |
| A11 | `.agent_state.json` has `last_completed_step=1` | PASS |

### Screenshots captured by the artifact writer itself
(These came from `runs/20260424-121139/` — the writer captured what the VLM saw during its own run.)

| Before step 1 | After step 1 |
|---|---|
| ![step_001_before.png](https://app.devin.ai/attachments/c6c40026-5635-4603-af52-9c4b2995ebf8/step_001_before.png) | ![step_001_after.png](https://app.devin.ai/attachments/665802a8-57ec-4182-8055-78152b576233/step_001_after.png) |

---

## Run 2 — PAUSE live trigger + user abort

### Setup
- Opened `pause_demo.html` locally in Chrome (mimics a real "Verify it's you" + "2-Step Verification" page with the number `75` and the iPhone-approval copy).
- `.env` restored to defaults (`RPD_LIMIT=500`, `SAVE_RUN_ARTIFACTS=false`).
- `tasks_run2.txt`: `Proceed past the verification screen shown in the browser.`
- Invocation: `echo "q" | python -m agent`

### Key log excerpts
```
12:14:06 [INFO] agent.agent: Planner emitted PAUSE: A 2-Step Verification prompt
  requires manual approval on the mobile device to proceed.
12:14:06 [WARNING] agent.agent: Agent paused — waiting for human: A 2-Step Verification
  prompt requires manual approval on the mobile device to proceed.

============================================================
[!] PAUSE: A 2-Step Verification prompt requires manual approval on the mobile device
    to proceed.
    Resolve the prompt above (e.g. approve on your phone, solve
    the captcha, etc.) and then press Enter to resume.
    Type 'q' + Enter to abort the run instead.
============================================================
>>> Resume? [Enter / q]:
[!] HALT at step 1/1: Proceed past the verification screen shown in the browser.
    Reason: User aborted at PAUSE: A 2-Step Verification prompt requires manual approval
    on the mobile device to proceed.
12:14:06 [ERROR] agent.agent: Halting execution: User aborted at PAUSE
```

Python exit code via `PIPESTATUS`: **1**.

### Zero side-effects on the page (B6/B7/B8)
```
$ grep -c "Pre-click human delay" run2.log       → 0
$ grep -c "CLICK_TEXT \[" run2.log               → 0
$ grep -c "Action: CLICK" run2.log               → 0
$ grep -cE "executor: CLICK|executor: TYPE|executor: PRESS" run2.log → 0
```

URL bar after PAUSE → abort: still `file:///home/ubuntu/repos/desktop-vlm-agent/pause_demo.html` (unchanged).

![Post-abort Chrome state — URL bar unchanged](https://app.devin.ai/attachments/c74aac6a-99e7-4e34-8920-af298e701d7c/screenshot_f42c802d368d498e9629a7e19f8b5b17.png)

### Assertions (Run 2)
| # | Assertion | Result |
|---|---|---|
| B1 | stderr has `[!] PAUSE:` | PASS |
| B2 | stderr has `>>> Resume? [Enter / q]:` | PASS |
| B3 | Log has `Agent paused — waiting for human:` WARNING | PASS |
| B4 | Exit code = 1 | PASS (PIPESTATUS=0 1 0) |
| B5 | HALT contains `User aborted at PAUSE` | PASS |
| B6 | No `Pre-click human delay` in log | PASS (0 matches) |
| B7 | No `CLICK_TEXT [` in log | PASS (0 matches) |
| B8 | Chrome URL unchanged (still on `pause_demo.html`) | PASS |
| B9 | Process terminated within 30s of stdin `q` | PASS (~0.5s total after VLM response) |

---

## Regression
- `pytest -q` → **171 passed** in 1.39s
- `ruff check .` → **All checks passed!**

---

## Notes
- Zero retries needed. `gemini-3.1-flash-lite-preview` returned HTTP 200 on every request today.
- No model fallback occurred. Pin to 3.1-flash-lite-preview held.
- Artifact writer + PAUSE + RPD guard are all wired into the same `run()` function and all three fired in the expected order during Run 1 (artifacts every step, RPD halt between steps) and Run 2 (PAUSE before any execute call).
