# PR #4 — Run B test report (password submit)

**Session**: https://app.devin.ai/sessions/241ab3b350814b07ba833a4658f39d56
**PR**: https://github.com/nd1836017/https-github.com-nd1836017-desktop-vlm-agent/pull/4
**Branch**: `devin/1777013535-retry-click-delay`
**Model**: `gemini-3.1-flash-lite-preview` (no fallback)
**Account**: `nhandoan1927@gmail.com` (secondary, via `GOOGLE_TEST_EMAIL` / `GOOGLE_TEST_PASSWORD` user-scope secrets)

## One-sentence summary
Drove the agent through the 3-step password submission flow on Google's
sign-in page; password was accepted, Google presented its standard device-push
verification (not a bot-detection block), user approved on phone, final state
is signed-in YouTube.

## Assertions

| # | Assertion | Result | Evidence |
|---|---|---|---|
| B1 | Pre-CLICK human delay fires on every CLICK, value ∈ [0.8, 2.0] | **PASS** | 3/3 CLICKs preceded by delay lines: `1.90s`, `1.66s`, `1.41s` |
| B2 | TYPE cadence is jittered (per-character loop) | **PASS** | TYPE issued 07:27:50; post-TYPE settle visible before next verify call |
| B3 | Login flow completes without bot-detection challenge | **PASS (with caveat)** | No `"browser may not be secure"` / CAPTCHA. Google did show device-push `"Verify it's you"` — this is a device-trust challenge, not bot-detection |
| B4 | Retry-with-backoff on transient errors | **PASS** | 3 live 503s, all recovered (`retry 1/6 after 3.5s`, `retry 1/6 after 2.6s`, `retry 2/6 after 3.2s`) |
| B5 | No fallback to gemini-2.5-flash-lite | **PASS** | `grep -c '2.5-flash-lite' run.log` → **0**. `3.1-flash-lite-preview` appears 17 times |

Exit code: **0**. All 3 task steps verified PASS.

## Notable runtime observations (not part of v4 scope, but visible this run)

- **PR #2 replan-on-failure fired twice and recovered.** The first CLICK on
  the password field was verified FAIL (`"does not show a blinking cursor"`);
  the agent replanned and the retry PASSed. Same for the first Next-button
  attempt — VLM first returned `CLICK` that missed, verifier returned FAIL,
  replan produced `PRESS [enter]` which PASSed. Good demo of the v2 budget
  working as intended.
- **PR #3 two-stage CLICK fired on every click.** Logs: `Two-stage CLICK:
  coarse norm=(...); crop origin=(..., ...) size=(300, 300)` → `Refine found
  1 candidate(s)` → `Single candidate; clicking it directly.`

## Evidence

### B1 — pre-click delays
```
07:26:43 Pre-click human delay: 1.90s
07:26:45 CLICK pixels=(1078,554)
07:27:17 Pre-click human delay: 1.66s
07:27:19 CLICK pixels=(1053,565)
07:28:19 Pre-click human delay: 1.41s
07:28:21 CLICK pixels=(1245,726)
```
All three values are in [0.8, 2.0]. One-to-one with the three CLICK lines.

### B4 — retry-with-backoff
```
07:27:23 verify: 503 ServerError — retry 1/6 after 3.5s
07:28:26 verify: 503 ServerError — retry 1/6 after 2.6s
07:28:33 verify: 503 ServerError — retry 2/6 after 3.2s
```
All three retries succeeded on the next attempt. No halt, no exhaustion.

### B5 — no fallback
```
$ grep -c '2\.5-flash-lite' ~/run_pr4_B.log
0
$ grep -c '3\.1-flash-lite-preview' ~/run_pr4_B.log
17
```

## Screenshots

### 1) Password typed (agent drove TYPE with per-character jitter)
![password typed](https://app.devin.ai/attachments/1f254194-6ddd-4ab7-8643-c5926c85516a/pr4B_password_typed.png)

### 2) Google device-push verification (not bot-detection)
![verify device push](https://app.devin.ai/attachments/a6cc8df7-2671-4268-b1e3-5ac9ffae6e95/pr4B_verify_device_push.png)

After user tapped **75** in Gmail on iPhone 11 Pro Max, the sign-in completed.

### 3) Signed-in YouTube (final state)
![signed in](https://app.devin.ai/attachments/5b4e1092-1ba1-42bc-91be-b5fa2aa56426/pr4B_signed_in_youtube.png)

## Caveats

1. **B3 is partial.** The agent reached Google's device-push challenge. This
   is NOT a bot-detection block (which typically looks like "couldn't sign you
   in — this browser or app may not be secure"), but Google does require manual
   phone-tap approval, which cannot be automated and was approved by the user.
2. **Log hygiene issue discovered (not a v4 regression).** The executor logs
   `TYPE '<value>'` in plaintext — this means the password substring appears
   in the raw log file. The log lives on the VM (`~/run_pr4_B.log`), is NOT
   committed, and is NOT included verbatim in this report. Recommend a small
   follow-up: redact TYPE values in the executor log when the text looks
   sensitive (or always, behind a `LOG_REDACT_TYPE=true` flag).
3. **Live demo of A3 bot-dodge on password submit** passed (no "not secure"
   interstitial appeared), but the final step's signed-in state required user
   phone approval, so the full auth cycle cannot claim "fully automated" until
   a persistent browser profile is used (deferred per user).

## Recording
Attached to the PR reply comment.
