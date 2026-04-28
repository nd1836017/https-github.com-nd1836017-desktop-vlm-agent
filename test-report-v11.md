# Test Report — PR #11 (6 post-merge bug fixes)

**Summary**: all 6 bug fixes verified. 3 live end-to-end tests + 25 regression unit tests pass. Bug #4 (PAUSE replan-budget) has no live-observable behavior without a real Gemini-backed 2FA flow; locked in via dedicated unit test.

Session: https://app.devin.ai/sessions/241ab3b350814b07ba833a4658f39d56

## Escalations

None. One test-environment note: pyautogui's `import mouseinfo` requires tkinter, which is missing on this pyenv-built Python. A one-line `sys.modules.setdefault("mouseinfo", ...)` stub in the test script works around this. Production environments (the user's Windows box, the CI ubuntu image with system python) bundle tkinter — this does not affect the shipped code.

## Assertions

| # | Bug | Assertion | Result |
|---|---|---|---|
| A1 | #3 | Textarea contains exactly `café résumé 你好 😀 Test!` after `TypeCommand` runs | ✅ passed |
| A2 | #3 | `_is_typewrite_safe` returns `True` for ASCII chars, `False` for `é`, `你`, `好`, `😀` | ✅ passed |
| A3 | #3 | Non-ASCII chars routed to clipboard-paste path (no silent drop) | ✅ passed |
| A5 | #5 | `warn=0.95, halt=0.95` → `ValueError: RPD_WARN_THRESHOLD (0.95) must be less than RPD_HALT_THRESHOLD (0.95)` | ✅ passed |
| A6 | #5 | `warn=0.99, halt=0.95` → matching `ValueError` | ✅ passed |
| A7 | #5 | `warn=0.75, halt=0.95` → exits 0, loads config | ✅ passed |
| A8 | #6 | Candidate at `top=-30` produces 34px-tall badge with red border pixels | ✅ passed (344 red pixels) |
| A9 | #6 | Candidate at `top=0` produces 34px-tall badge with red border pixels | ✅ passed (344 red pixels) |
| A10 | #6 | Candidate at `top=570` (mid-image) still produces 34px-tall badge | ✅ passed (344 red pixels) |
| U1 | #1 | `History.__iter__()` returns an `Iterator` (supports `next()`) | ✅ unit test passed |
| U2 | #2 | `_strip_markdown_fences` handles `json`, `JSON`, `Json`, no-tag, `yaml`, internal backticks | ✅ 8 unit tests passed |
| U3 | #4 | PAUSE+PAUSE+real attempt on 1-replan budget lands the real attempt (no starvation) | ✅ unit test passed |

**Total**: 215 tests pass (190 pre-existing + 25 new regression), `ruff check .` clean, CI 3/3.

## Evidence

### Bug #3 — non-ASCII TYPE

| 🔴 BEFORE — textarea empty, about to TYPE | 🟢 AFTER — exact match, every char landed |
|---|---|
| ![Empty textarea](/tmp/bug3_before.png) | ![Full non-ASCII string typed](/tmp/bug3_after.png) |

Classifier spot-check output:
```
'a' -> True     ' ' -> True     '!' -> True     'T' -> True
'é' -> False    '你' -> False   '好' -> False    '😀' -> False
```

Executor typing log (abbreviated):
```
[ 0] 'c' safe=True        [ 3] 'é' safe=False  ← clipboard path
[ 6] 'é' safe=False       [12] '你' safe=False
[13] '好' safe=False      [15] '😀' safe=False
[17] 'T' safe=True        [21] '!' safe=True
DONE
```

### Bug #5 — RPD threshold validation

```
=== A5: warn==halt (0.95, 0.95), should raise ===
ValueError: RPD_WARN_THRESHOLD (0.95) must be less than RPD_HALT_THRESHOLD (0.95)
exit=1

=== A6: warn > halt (0.99, 0.95), should raise ===
ValueError: RPD_WARN_THRESHOLD (0.99) must be less than RPD_HALT_THRESHOLD (0.95)
exit=1

=== A7: warn < halt (0.75, 0.95), sane ===
warn= 0.75 halt= 0.95
exit=0
```

### Bug #6 — candidate badge at image top

| 🟢 top=-30 (pre-fix: invisible) | 🟢 top=0 (pre-fix: zero-height) | 🟢 top=570 (regression) |
|---|---|---|
| ![Badge at top=-30](/tmp/bug6_edge0_top-30.png) | ![Badge at top=0](/tmp/bug6_near_edge_top0.png) | ![Badge at top=570](/tmp/bug6_middle_top570.png) |

All three show the same 34px red-bordered white badge with a visible "1" label. Under the pre-fix formula, the first two would have rendered as zero-height invisible boxes.

### Bug #4 — PAUSE does not starve replan budget

Unit test `test_pause_does_not_starve_replan_budget` (mocked `_attempt_step`):

```
Sequence: PAUSE → PAUSE → pass
Budget:   1 replan allowed
Expected: real attempt fires on the 3rd iteration (PAUSEs don't count)
Actual:   passed — _attempt_step called 3 times, final VerificationResult.passed=True
```

Cannot be live-tested without a real Gemini-backed 2FA page + real VLM calls; deferred as a known gap per test plan.

## Recording

Bug #3 live recording (Chrome + executor): attached to the user-facing message.

## Not tested live (intentional)

- **Bug #1**: type annotation — no runtime effect. Unit test locked in.
- **Bug #2**: deterministic string transformation — every failure mode covered by 8 parametric unit tests. Live call would be a tautology.
- **Bug #4**: PAUSE+replan requires a real Gemini-backed run with a real 2FA page. Would consume ~3–5 real VLM calls from the daily quota. Locked in via mocked-step unit test that exactly exercises the starvation scenario the fix targets.
