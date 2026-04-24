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

echo "[launch-chrome] profile=${PROFILE_DIR} url=${START_URL}"
exec "${CHROME}" \
    --user-data-dir="${PROFILE_DIR}" \
    --no-first-run \
    --no-default-browser-check \
    "${START_URL}"
