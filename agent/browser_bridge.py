"""Chrome DevTools Protocol bridge for the browser-fast-path optimization.

When the user is driving a Chrome browser with ``--remote-debugging-port``
exposed, three new primitives let the planner act on the page DOM directly
instead of going through the VLM-OCR-pyautogui pipeline:

    BROWSER_GO   [url]                — Page.navigate
    BROWSER_CLICK[selector]           — querySelector + element.click()
    BROWSER_FILL [selector, value]    — querySelector + .value = ... + input event

A single CDP roundtrip is essentially free (a few hundred bytes of JSON over
localhost) where the visual path costs ~15k Gemini vision tokens for the
same logical operation. On browser-heavy runs this cuts plan_action token
spend by ~5x.

Design constraints:

- Best-effort. If CDP is not available (port closed, Chrome killed,
  selector misses), every operation returns a clear error string and the
  caller falls back to the ordinary visual flow. Nothing here may crash
  a run.
- Synchronous-looking API. The executor loop is sync; we use
  ``websocket-client`` (a tiny sync websocket library) and a per-call
  short timeout to keep the planner's perception of latency stable.
- Lazy import of ``websocket`` so test environments without the package
  installed (and runs that don't enable the fast path) still import this
  module successfully.
- One websocket connection per ``BrowserBridge`` instance; the executor
  reuses the bridge across the run so we don't pay reconnect cost on
  every step.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


DEFAULT_CDP_HOST = "localhost"
DEFAULT_CDP_PORT = 29229
# Per-call CDP timeout. Most ops (navigate, click, fill) complete in <100ms
# on localhost; if Chrome is hung we want to bail out quickly and let the
# fallback (visual path) take over instead of stalling the agent.
CDP_TIMEOUT_SECONDS = 10.0
# Discovery roundtrip is a plain HTTP GET — should be near-instant. Use a
# tighter budget so a wedged debug port doesn't add seconds of latency to
# every step.
CDP_DISCOVERY_TIMEOUT_SECONDS = 2.0


class BrowserBridgeUnavailable(RuntimeError):
    """Raised when the CDP target cannot be reached or used."""


@dataclass
class _CdpTarget:
    """A single page target advertised by Chrome's /json endpoint."""

    target_id: str
    url: str
    websocket_url: str
    title: str = ""

    @property
    def is_real_page(self) -> bool:
        """Skip Chrome's own tooling tabs (devtools, extensions popup, etc.).

        Deny-list rather than allow-list: anything that's not a Chrome
        internal tooling surface is treated as drivable. This is
        important because a freshly-launched Chrome with no manual
        navigation lands on ``chrome://newtab/``, and the previous
        allow-list rejected it — meaning ``connect()`` returned False
        on a brand-new browser.
        """
        u = self.url
        return not (
            u.startswith("devtools://") or u.startswith("chrome-extension://")
        )


class BrowserBridge:
    """Drive Chrome via CDP for fast browser actions.

    Lifecycle:

        bridge = BrowserBridge()
        if bridge.connect():
            bridge.navigate("https://youtube.com")
            bridge.fill("input[type=search]", "baby")
            bridge.click("button#search-icon-legacy")
        bridge.close()

    All public methods return ``(ok: bool, message: str)`` so callers can
    log the result and fall back to the visual path on failure without
    catching exceptions.
    """

    def __init__(
        self,
        host: str = DEFAULT_CDP_HOST,
        port: int = DEFAULT_CDP_PORT,
        *,
        timeout_seconds: float = CDP_TIMEOUT_SECONDS,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout_seconds
        self._ws = None  # websocket.WebSocket once connected
        self._next_id = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public

    def connect(self) -> bool:
        """Connect to the active page tab. Returns True on success.

        Never raises — failures are logged at INFO so a missing CDP port
        doesn't spam ERROR logs on every run.
        """
        try:
            target = self._discover_active_target()
        except BrowserBridgeUnavailable as exc:
            log.info("BrowserBridge: CDP unavailable (%s)", exc)
            return False
        try:
            self._open_websocket(target.websocket_url)
        except BrowserBridgeUnavailable as exc:
            log.info(
                "BrowserBridge: failed to open CDP websocket (%s)", exc
            )
            return False
        log.info(
            "BrowserBridge: connected to %s (%s)", target.url or "<blank>", target.target_id[:8]
        )
        return True

    def is_connected(self) -> bool:
        return self._ws is not None

    def close(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is None:
            return
        try:
            ws.close()
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("BrowserBridge: close raised %s", exc)

    def navigate(self, url: str) -> tuple[bool, str]:
        """Navigate the active tab to ``url``. URL must be absolute."""
        if not self.is_connected():
            return False, "BrowserBridge not connected"
        if not url:
            return False, "BROWSER_GO: empty URL"
        if not (url.startswith("http://") or url.startswith("https://")):
            return False, f"BROWSER_GO: URL must start with http:// or https:// (got {url!r})"
        try:
            result = self._send("Page.navigate", {"url": url})
        except BrowserBridgeUnavailable as exc:
            return False, f"BROWSER_GO failed: {exc}"
        # Chrome surfaces navigation failures (DNS, connection refused,
        # ERR_NAME_NOT_RESOLVED, etc.) via an ``errorText`` field on the
        # result dict — NOT through the protocol-level ``error`` key that
        # ``_send`` already raises on. Without this check, a typo'd URL
        # gets marked PASS, the checkpoint advances, and the next step
        # runs against the wrong page (or no page at all).
        error_text = result.get("errorText") if isinstance(result, dict) else None
        if error_text:
            return False, f"BROWSER_GO [{url}] — navigation error: {error_text}"
        return True, f"BROWSER_GO [{url}]"

    def click(self, selector: str) -> tuple[bool, str]:
        """Click the first element matching ``selector`` via Runtime.evaluate.

        Uses ``element.click()`` which dispatches a synthetic click event;
        this works for buttons, links, and most form controls. For elements
        that need a real mouse event (e.g. some custom drag handles) the
        caller should use the visual CLICK path.
        """
        if not self.is_connected():
            return False, "BrowserBridge not connected"
        if not selector:
            return False, "BROWSER_CLICK: empty selector"
        # Embed the selector as a JSON string so quotes inside it are
        # escaped properly. JSON.stringify-style encoding via json.dumps.
        expr = (
            f"(() => {{ const el = document.querySelector({json.dumps(selector)}); "
            f"if (!el) return 'NOT_FOUND'; el.click(); return 'OK'; }})()"
        )
        try:
            result = self._send(
                "Runtime.evaluate",
                {"expression": expr, "returnByValue": True},
            )
        except BrowserBridgeUnavailable as exc:
            return False, f"BROWSER_CLICK failed: {exc}"
        value = _extract_eval_value(result)
        if value == "OK":
            return True, f"BROWSER_CLICK [{selector}]"
        if value == "NOT_FOUND":
            return False, f"BROWSER_CLICK [{selector}] — selector not found"
        return False, f"BROWSER_CLICK [{selector}] — unexpected result {value!r}"

    def fill(self, selector: str, value: str) -> tuple[bool, str]:
        """Set ``value`` on the first input matching ``selector``.

        Sets ``element.value`` and then dispatches an ``input`` event so
        React/Vue/etc. controlled-component handlers see the change. Most
        modern frameworks listen for ``input``; a few want ``change`` too,
        which we also dispatch as a belt-and-suspenders measure.

        ``value`` is privacy-redacted in the returned message — the
        history/log layer relies on action_text never containing user-typed
        secrets. The CDP payload itself does carry the literal value (it
        has to, that's the whole point), but it stays on localhost.
        """
        if not self.is_connected():
            return False, "BrowserBridge not connected"
        if not selector:
            return False, "BROWSER_FILL: empty selector"
        expr = (
            f"(() => {{ const el = document.querySelector({json.dumps(selector)}); "
            f"if (!el) return 'NOT_FOUND'; "
            f"el.focus(); el.value = {json.dumps(value)}; "
            f"el.dispatchEvent(new Event('input', {{bubbles: true}})); "
            f"el.dispatchEvent(new Event('change', {{bubbles: true}})); "
            f"return 'OK'; }})()"
        )
        try:
            result = self._send(
                "Runtime.evaluate",
                {"expression": expr, "returnByValue": True},
            )
        except BrowserBridgeUnavailable as exc:
            return False, f"BROWSER_FILL failed: {exc}"
        value_kind = _extract_eval_value(result)
        # Privacy-redact the typed value in the result string — same policy
        # as TYPE in agent/executor.py. Length is preserved so postmortems
        # still distinguish "filled 6 chars" from "filled 200 chars".
        redacted = f"<REDACTED, {len(value)} chars>"
        if value_kind == "OK":
            return True, f"BROWSER_FILL [{selector}, {redacted}]"
        if value_kind == "NOT_FOUND":
            return False, f"BROWSER_FILL [{selector}] — selector not found"
        return (
            False,
            f"BROWSER_FILL [{selector}] — unexpected result {value_kind!r}",
        )

    # ----------------------------------------------------------------- private

    def _discover_active_target(self) -> _CdpTarget:
        """Hit /json on the debug port and pick the first navigatable tab."""
        url = f"http://{self._host}:{self._port}/json"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=CDP_DISCOVERY_TIMEOUT_SECONDS) as resp:
                payload = resp.read().decode("utf-8")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise BrowserBridgeUnavailable(
                f"could not reach {url}: {exc}"
            ) from exc
        try:
            entries = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise BrowserBridgeUnavailable(
                f"malformed JSON from {url}"
            ) from exc
        if not isinstance(entries, list):
            raise BrowserBridgeUnavailable(
                f"expected list from {url}, got {type(entries).__name__}"
            )
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "page":
                continue
            ws_url = entry.get("webSocketDebuggerUrl")
            if not isinstance(ws_url, str) or not ws_url:
                continue
            target = _CdpTarget(
                target_id=str(entry.get("id", "")),
                url=str(entry.get("url", "")),
                websocket_url=ws_url,
                title=str(entry.get("title", "")),
            )
            if target.is_real_page:
                return target
        raise BrowserBridgeUnavailable("no navigatable page tabs available")

    def _open_websocket(self, ws_url: str) -> None:
        try:
            import websocket  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise BrowserBridgeUnavailable(
                "websocket-client package not installed; run pip install -r requirements.txt"
            ) from exc
        try:
            # ``suppress_origin=True`` tells websocket-client not to
            # send the Origin header at all. Chrome 111+ rejects CDP
            # websocket connections whose Origin doesn't match an
            # explicit allow-list, returning HTTP 403. With no Origin
            # header, Chrome treats the connection as "no origin" and
            # accepts it — so the bridge works even when the user
            # launched Chrome without --remote-allow-origins. Belt and
            # suspenders: launch-chrome.{sh,bat} now also pass the
            # flag.
            ws = websocket.create_connection(
                ws_url, timeout=self._timeout, suppress_origin=True
            )
        except Exception as exc:  # noqa: BLE001 — ws library raises broad
            raise BrowserBridgeUnavailable(
                f"could not connect to {ws_url}: {exc}"
            ) from exc
        self._ws = ws

    def _send(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC message and read the matching reply.

        CDP multiplexes many message kinds on one websocket: command
        replies (matched by ``id``) and unsolicited events (no ``id``).
        We loop reading until we see our reply, ignoring events. The
        per-call timeout caps the total wait so a flood of unrelated
        events can't starve us forever.
        """
        ws = self._ws
        if ws is None:
            raise BrowserBridgeUnavailable("BrowserBridge not connected")
        with self._lock:
            self._next_id += 1
            msg_id = self._next_id
            payload = json.dumps(
                {"id": msg_id, "method": method, "params": params}
            )
            try:
                ws.send(payload)
            except Exception as exc:  # noqa: BLE001
                self._ws = None
                raise BrowserBridgeUnavailable(
                    f"CDP send failed ({method}): {exc}"
                ) from exc
            # Read until we get our id back, dropping events.
            while True:
                try:
                    raw = ws.recv()
                except Exception as exc:  # noqa: BLE001
                    self._ws = None
                    raise BrowserBridgeUnavailable(
                        f"CDP recv failed ({method}): {exc}"
                    ) from exc
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise BrowserBridgeUnavailable(
                        f"CDP returned malformed JSON ({method}): {exc}"
                    ) from exc
                if not isinstance(msg, dict):
                    continue
                if msg.get("id") != msg_id:
                    # An unsolicited Page.* / Network.* event — ignore.
                    continue
                if "error" in msg:
                    err = msg["error"]
                    detail = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    raise BrowserBridgeUnavailable(
                        f"CDP {method} error: {detail}"
                    )
                result = msg.get("result")
                if not isinstance(result, dict):
                    raise BrowserBridgeUnavailable(
                        f"CDP {method} reply has no result"
                    )
                return result


def _extract_eval_value(result: dict[str, Any]) -> Any:
    """Pull the JS-side return value out of a Runtime.evaluate reply.

    Reply shape::

        {"result": {"type": "string", "value": "OK"}, ...}

    or, for an exception::

        {"exceptionDetails": {...}}

    We return the raw value when present, ``"<EXCEPTION>"`` when JS
    threw, and ``None`` when the reply has neither (shouldn't happen
    in practice).
    """
    if "exceptionDetails" in result:
        return "<EXCEPTION>"
    inner = result.get("result")
    if isinstance(inner, dict):
        return inner.get("value")
    return None
