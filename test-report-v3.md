# PR #3 — Two-stage CLICK live test report

**Scope**: Only "open a YouTube video" was tested per user instruction ("dont do the rest until i told you to"). Login, email, file lookup deferred.

**How I tested**: pre-loaded YouTube search results for `python tutorial` in Chrome, ran the agent on a single-step task file (`tasks_yt_click.txt` = "Click on the first video thumbnail in the YouTube home grid to open it") — once with two-stage CLICK ON (default), once with `--no-two-stage-click`. Then ran regression (pytest + ruff).

## Assertion results

### Run A — two-stage ON (default) — model `gemini-3.1-flash-lite-preview`

| # | Assertion | Result | Evidence |
|---|---|---|---|
| A1 | Two-stage path fires on CLICK | **PASS** | Log: `Two-stage CLICK: coarse norm=(336,294); crop origin=(388, 203) size=(300, 300)` |
| A2 | Refine returns ≥1 candidate | **PASS** (K=1) | Log: `Refine found 1 candidate(s): [(500, 500)]` |
| A3 | Disambiguator fires when K≥2 | **UNTESTED** | K=1 here, so disambiguator was correctly skipped. Branching covered by unit tests (`test_two_stage_click.py`). |
| A4 | Executor clicks refined pixel (not coarse) | **INCONCLUSIVE** | Refine returned `(500,500)` = exact crop center, which arithmetically maps back to the same pixel as the coarse point on this crop. Cannot prove non-equality on this particular VLM response. |
| A5 | YouTube video opens | **PASS** | Final URL: `https://www.youtube.com/watch?v=b093aqAZiPU` (valid `/watch` page). |
| A6 | Exit 0 and VERDICT PASS | **PASS** | `[ok] All 1 step(s) completed successfully.`; exit 0. |

### Run B — two-stage OFF (`--no-two-stage-click`) — model `gemini-2.5-flash-lite` (3.1 returned 503 repeatedly)

| # | Assertion | Result | Evidence |
|---|---|---|---|
| B1 | Zero two-stage log lines | **PASS** | `grep -c 'Two-stage CLICK' = 0`, `Refine found = 0`, `Disambiguator picked = 0` |
| B2 | Legacy coarse-click line appears | **PASS** | Log: `CLICK normalized=(314,285) -> pixels=(502,342)` |

### Regression

| # | Assertion | Result |
|---|---|---|
| R1 | pytest -q | **PASS** — 75 passed in 1.18s |
| R2 | ruff check . | **PASS** — All checks passed! |

## Escalations / things that did not go perfectly

1. **A3 UNTESTED on live run.** The VLM returned a single refine candidate (K=1), so the disambiguation branch never fired. The branching logic is covered by unit tests, but I couldn't demonstrate it on real API calls.
2. **A4 INCONCLUSIVE.** The VLM placed the single refine candidate at `(500, 500)` — the exact crop center. The crop was centered on the coarse point, so the refined pixel mathematically equals the coarse pixel. The two-stage path fired correctly, but this particular response cannot prove the refined pixel differs from the coarse pixel.
3. **Run B did not complete cleanly.** After the CLICK fired (which IS what assertions B1/B2 test), the verifier returned FAIL and the replan attempt hit 429 RESOURCE_EXHAUSTED on `gemini-2.5-flash-lite` (20 RPD cap). The assertions B1 and B2 were still satisfied by the single CLICK that did fire.
4. **`gemini-3.1-flash-lite-preview` returned 503 UNAVAILABLE** on Run B; I fell back to `gemini-2.5-flash-lite`. This is a Google-side transient issue, not a code problem.
5. **YouTube bot detection** triggered on the opened video page ("Sign in to confirm you're not a bot"). The `/watch?v=...` URL still loaded correctly, which is what A5 asserts.

## Log excerpts

### Run A (two-stage ON)
```
06:26:52 Two-stage CLICK: coarse norm=(336,294); crop origin=(388, 203) size=(300, 300)
06:27:09 Refine found 1 candidate(s): [(500, 500)]
06:27:09 Single candidate; clicking it directly.
06:27:09 Action: CLICK_PX [538,353] (2-stage from CLICK [336,294], 1 cand)
06:27:09 CLICK pixels=(538,353)
06:27:30 Verify: VERDICT: PASS — The video player page has loaded as expected.
06:27:30 All 1 step(s) completed successfully.
```

### Run B (`--no-two-stage-click`)
```
06:31:14 Action: CLICK [314,285]
06:31:14 CLICK normalized=(314,285) -> pixels=(502,342)
06:31:18 Verify: VERDICT: FAIL — The screen shows a black video player ...
```

## Pass/fail decision
Two-stage CLICK wiring, CLI flag, refine, and legacy path all behave as specified. A3 and A4 could not be demonstrated on live calls due to VLM returning a single center-of-crop candidate; both are covered by unit tests. I would not declare Run A/B a perfect live proof of A3/A4 — reporting conservatively.
