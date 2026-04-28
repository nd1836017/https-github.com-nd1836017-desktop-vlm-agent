# Test Plan — PR #11 (6 post-merge bug fixes)

## What changed (user-visible)

1. `history.py`: `History.__iter__` annotation corrected from `Iterable[StepRecord]` → `Iterator[StepRecord]`.
2. `vlm.py`: markdown-fence stripping now handles any language tag case (`json`/`JSON`/`Json`/`yaml`/none) and no longer scrubs backticks from inside the JSON payload.
3. `executor.py`: TYPE now routes non-ASCII chars (accented, CJK, emoji) through the system clipboard + Ctrl+V instead of silently dropping them.
4. `agent.py`: PAUSE no longer consumes a replan-budget slot — a PAUSE followed by a real attempt still gets its full allowance.
5. `config.py`: `Config.load()` raises `ValueError` if `RPD_WARN_THRESHOLD >= RPD_HALT_THRESHOLD` instead of silently producing a config where the warn log can never fire.
6. `screen.py`: candidate-number badge is always rendered 34px tall, even when the candidate sits at the top edge of the image (`top ≤ 33`).

## What I will test

Three live tests + targeted unit-test runs for the logic-only bugs.

- **Live T1 (Bug #3)** — type `café résumé 你好 😀 Test!` into a Chrome `<textarea>` via `executor.execute(TypeCommand(...))` directly (no VLM). Verify the textarea contains the **exact** string, character-for-character. On broken code this would produce `cafrsum  Test!` (non-ASCII silently dropped).
- **Live T2 (Bug #5)** — three invocations of `Config.load()` via `python -c`:
  - `warn=0.95, halt=0.95` → must exit non-zero with `ValueError: RPD_WARN_THRESHOLD (0.95) must be less than RPD_HALT_THRESHOLD (0.95)`
  - `warn=0.99, halt=0.95` → must exit non-zero with matching message
  - `warn=0.75, halt=0.95` → must exit zero (sane config still works)
- **Live T3 (Bug #6)** — render two annotated candidate overlays at `(y=0)` and `(y=30)` via `screen.annotate_candidates`, save PNGs, and programmatically assert the badge rectangle contains non-white (red border) pixels in both `y in [0,33]` bands. On broken code the badge collapses to zero height and the pixel check for `y=0` finds only background.
- **Unit T4 (Bug #1)** — `pytest tests/test_bug_regressions.py::test_history_iter_returns_iterator_not_iterable -v`
- **Unit T5 (Bug #2)** — `pytest tests/test_bug_regressions.py -k "fence or markdown or backtick" -v`
- **Unit T6 (Bug #4)** — `pytest tests/test_bug_regressions.py::test_pause_does_not_starve_replan_budget -v`

## Key assertions (each would fail visibly on a broken impl)

### Bug #3 — non-ASCII TYPE
| # | Expected | Pass/Fail criterion |
|---|---|---|
| A1 | Textarea contains the literal string `café résumé 你好 😀 Test!` after TYPE completes | `textarea.value == "café résumé 你好 😀 Test!"` via JS read; any missing char = FAIL |
| A2 | Log contains exactly one `TYPE '...'` line (not redacted since `log_redact_type=False` in test) | grep count == 1 |
| A3 | No `WARNING: TYPE: non-ASCII char ... dropped` line (pyperclip is installed + xclip is installed) | grep count == 0 |
| A4 | `_is_typewrite_safe('é')` returns `False`, `_is_typewrite_safe('a')` returns `True` | direct assertion |

### Bug #5 — RPD validation
| # | Input env | Expected stderr substring |
|---|---|---|
| A5 | `RPD_WARN_THRESHOLD=0.95 RPD_HALT_THRESHOLD=0.95` | `ValueError: RPD_WARN_THRESHOLD (0.95) must be less than RPD_HALT_THRESHOLD (0.95)`; exit code ≠ 0 |
| A6 | `RPD_WARN_THRESHOLD=0.99 RPD_HALT_THRESHOLD=0.95` | `ValueError: RPD_WARN_THRESHOLD (0.99) must be less than RPD_HALT_THRESHOLD (0.95)`; exit code ≠ 0 |
| A7 | `RPD_WARN_THRESHOLD=0.75 RPD_HALT_THRESHOLD=0.95` | no error, exits 0, `cfg.rpd_warn_threshold == 0.75` |

### Bug #6 — badge height
| # | Candidate y | Expected |
|---|---|---|
| A8 | `y_norm` such that pixel `top == 0` | Saved PNG has at least one red (r>200, g<80, b<80) pixel in band `y=0..33` at the badge column | 
| A9 | `y_norm` such that pixel `top == 30` | Same check — red pixels in `y=0..33` |
| A10 | Badge visually legible in `out_top0.png` and `out_top30.png` | side-by-side attachment in report |

## Evidence I'll capture

- `test-report-v11.md` with 2-column 🔴-before/🟢-after layout where applicable.
- Screenshots: Chrome textarea after TYPE; `annotate_candidates` output PNGs for `top=0` and `top=30`.
- Command output for the 3 `Config.load()` invocations and the 3 targeted pytest runs.
- Single PR comment on #11 with a `<details>` per bug + session link.
- Recording of the Chrome TYPE test only (the rest is shell — text evidence is enough).

## What I'm NOT testing live and why

- **Bug #1 (Iterator annotation)**: pure static type hint — no runtime behavior change possible to observe. Lock-in via the new `test_history_iter_returns_iterator_not_iterable` test.
- **Bug #2 (markdown fences)**: deterministic text transformation. Unit tests exercise every failure mode (lowercase/UPPERCASE/mixed/no-tag/internal-backtick). A live test would just call the same function. No additional signal.
- **Bug #4 (PAUSE budget)**: triggering a real PAUSE + replan cycle needs a real Gemini-backed run with a real 2FA-ish page and costs multiple real VLM calls. The `test_pause_does_not_starve_replan_budget` test mocks `_attempt_step` with a sequence `[PAUSE, PAUSE, pass]` and asserts the final attempt lands on a 1-replan budget — which is exactly the starvation scenario the fix targets. Documented as a known gap.

## Cost

- ~0 Gemini calls (tests never hit the API).
- ~1 xclip install (done).
- ~1 Chrome tab (textarea data URL).
