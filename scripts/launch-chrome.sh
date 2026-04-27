#!/usr/bin/env bash
# Launch Chrome with a persistent profile directory so that logins,
# cookies, and "Verify it's you" trust decisions survive across agent runs.
#
# Usage:
#     ./scripts/launch-chrome.sh                  # default profile + default URL
#     ./scripts/launch-chrome.sh https://...     # default profile, custom URL
#     PROFILE_DIR=/tmp/x ./scripts/launch-chrome.sh
#
# The agent itself does NOT launch the browser — pair this script with
# `python -m agent` to drive a real Chrome session that retains state
# between runs.
set -euo pipefail

PROFILE_DIR="${PROFILE_DIR:-${HOME}/.cache/desktop-vlm-agent/chrome-profile}"
mkdir -p "${PROFILE_DIR}"

START_URL="${1:-about:blank}"

# Pick the first Chrome-family binary that exists.
for cand in google-chrome google-chrome-stable chromium chromium-browser; do
    if command -v "${cand}" >/dev/null 2>&1; then
        CHROME="${cand}"
        break
    fi
done

if [[ -z "${CHROME:-}" ]]; then
    echo "error: no Chrome/Chromium binary found in PATH" >&2
    exit 1
fi

CDP_PORT="${CDP_PORT:-29229}"

echo "[launch-chrome] profile=${PROFILE_DIR} url=${START_URL} cdp-port=${CDP_PORT}"
# --remote-debugging-port exposes Chrome DevTools Protocol on localhost so
# the agent's BROWSER_GO / BROWSER_CLICK / BROWSER_FILL fast-path can drive
# the active tab without going through the VLM. Harmless when the agent
# isn't using the fast path; only listens on localhost so it isn't a
# remote-attack surface. Set CDP_PORT=0 to disable explicitly.
#
# --remote-allow-origins is REQUIRED on Chrome 111+ — without it Chrome
# rejects the websocket handshake with "403 Forbidden" unless the
# connecting client omits the Origin header entirely. We use
# http://localhost:<port> as the narrowest allow-list that still lets
# the bridge connect. (The bridge ALSO passes suppress_origin=True for
# older / non-script-launched Chromes, but setting this flag here
# matches documented Chromium guidance.)
DEBUG_FLAG=""
if [[ "${CDP_PORT}" != "0" ]]; then
    DEBUG_FLAG="--remote-debugging-port=${CDP_PORT} --remote-allow-origins=http://localhost:${CDP_PORT}"
fi
exec "${CHROME}" \
    --user-data-dir="${PROFILE_DIR}" \
    --no-first-run \
    --no-default-browser-check \
    ${DEBUG_FLAG} \
    "${START_URL}"
