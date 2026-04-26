"""Tests for the browser-fast-path primitives + CDP bridge.

Covers:
- Parser correctness for BROWSER_GO / BROWSER_CLICK / BROWSER_FILL.
- ``render_command`` redacts BROWSER_FILL value the same way as TYPE.
- ``BrowserBridge.connect()`` returns False when the CDP port is closed
  rather than raising.
- ``BrowserBridge`` send/receive happy-path with a mock websocket — covers
  navigate / click / fill returning OK, NOT_FOUND, and error replies.
- ``inspect_features`` detects browser hints in tasks text and gates
  ``uses_browser_fast_path``.

The bridge tests use a fake websocket object that records sent payloads
and pre-stages reply messages. We never actually open a real CDP socket
in tests — that would require a running Chrome instance.
"""
from __future__ import annotations

import json
from collections import deque

import pytest

from agent.browser_bridge import (
    BrowserBridge,
    _CdpTarget,
    _extract_eval_value,
)
from agent.files import inspect_features
from agent.history import render_command
from agent.parser import (
    BrowserClickCommand,
    BrowserFillCommand,
    BrowserGoCommand,
    parse_command,
)
from agent.tasks_loader import TaskStep

# ---------------------------------------------------------------------- parser


def test_parse_browser_go_extracts_https_url() -> None:
    cmd = parse_command("BROWSER_GO [https://youtube.com]")
    assert isinstance(cmd, BrowserGoCommand)
    assert cmd.url == "https://youtube.com"


def test_parse_browser_go_rejects_non_http_scheme() -> None:
    """Bare hostnames / file:// paths fall through (regex anchors on http(s))."""
    assert parse_command("BROWSER_GO [youtube.com]") is None
    assert parse_command("BROWSER_GO [file:///tmp/x.html]") is None


def test_parse_browser_click_with_simple_selector() -> None:
    cmd = parse_command("BROWSER_CLICK [#search-icon-legacy]")
    assert isinstance(cmd, BrowserClickCommand)
    assert cmd.selector == "#search-icon-legacy"


def test_parse_browser_click_with_attribute_selector() -> None:
    cmd = parse_command("BROWSER_CLICK [button[aria-label='Search']]")
    assert isinstance(cmd, BrowserClickCommand)
    assert cmd.selector == "button[aria-label='Search']"


def test_parse_browser_fill_splits_on_first_comma_space() -> None:
    cmd = parse_command("BROWSER_FILL [input[type=search], baby]")
    assert isinstance(cmd, BrowserFillCommand)
    assert cmd.selector == "input[type=search]"
    assert cmd.value == "baby"


def test_parse_browser_fill_value_with_internal_commas() -> None:
    """Values may contain commas; only the FIRST ``, `` separates args."""
    cmd = parse_command("BROWSER_FILL [textarea#bio, hello, world, foo]")
    assert isinstance(cmd, BrowserFillCommand)
    assert cmd.selector == "textarea#bio"
    assert cmd.value == "hello, world, foo"


def test_parse_browser_fill_does_not_steal_browser_click() -> None:
    """A BROWSER_CLICK with no comma must not be parsed as BROWSER_FILL."""
    cmd = parse_command("BROWSER_CLICK [button.search]")
    assert isinstance(cmd, BrowserClickCommand)
    assert cmd.selector == "button.search"


def test_parse_browser_commands_dont_get_eaten_by_click() -> None:
    """The generic CLICK [x,y] regex must not consume a BROWSER_CLICK arg."""
    # If parse-order were wrong, ``BROWSER_CLICK [...]`` could match the
    # CLICK [X,Y] coord pattern when the arg happens to be two integers.
    cmd = parse_command("BROWSER_CLICK [div[data-x='100'][data-y='200']]")
    assert isinstance(cmd, BrowserClickCommand)


# ---------------------------------------------------------------------- render


def test_render_browser_go() -> None:
    cmd = BrowserGoCommand(url="https://example.com")
    assert render_command(cmd) == "BROWSER_GO [https://example.com]"


def test_render_browser_click() -> None:
    cmd = BrowserClickCommand(selector="button#go")
    assert render_command(cmd) == "BROWSER_CLICK [button#go]"


def test_render_browser_fill_redacts_value_when_requested() -> None:
    """Same privacy policy as TYPE: value is replaced, length preserved."""
    cmd = BrowserFillCommand(selector="input#email", value="hunter2")
    redacted = render_command(cmd, redact_type=True)
    assert "hunter2" not in redacted
    assert "<REDACTED, 7 chars>" in redacted
    # Selector and command class stay visible.
    assert "BROWSER_FILL" in redacted
    assert "input#email" in redacted


def test_render_browser_fill_keeps_value_when_not_redacting() -> None:
    cmd = BrowserFillCommand(selector="input#q", value="hello")
    assert render_command(cmd) == "BROWSER_FILL [input#q, hello]"


# ------------------------------------------------------------- features detect


def _step(text: str) -> TaskStep:
    return TaskStep(text=text)


def test_inspect_features_detects_browser_go_hint() -> None:
    feats = inspect_features([_step("BROWSER_GO [https://x.com]")])
    assert feats.uses_browser_fast_path is True
    assert feats.browser_go_count == 1
    assert feats.browser_click_count == 0


def test_inspect_features_counts_each_browser_kind() -> None:
    feats = inspect_features(
        [
            _step("BROWSER_GO [https://x.com]"),
            _step("BROWSER_CLICK [#submit]"),
            _step("BROWSER_FILL [#q, hello]"),
            _step("BROWSER_CLICK [.btn]"),
        ]
    )
    assert feats.browser_go_count == 1
    assert feats.browser_click_count == 2
    assert feats.browser_fill_count == 1
    assert feats.uses_browser_fast_path is True


def test_inspect_features_prose_mentions_dont_trigger() -> None:
    """Casual mentions must not opt the run into the CDP bridge."""
    feats = inspect_features(
        [
            _step("open the browser and go to youtube.com"),
            _step("browser_go without brackets is not a hint"),
        ]
    )
    assert feats.uses_browser_fast_path is False
    assert feats.browser_go_count == 0


# ----------------------------------------------------------------- bridge: net


def test_browser_bridge_connect_returns_false_when_port_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """connect() must NOT raise; it must log + return False so the run
    can continue with the visual fallback path."""

    def _fake_urlopen(*_args, **_kwargs):  # noqa: ANN001
        raise OSError("Connection refused")

    monkeypatch.setattr("agent.browser_bridge.urllib.request.urlopen", _fake_urlopen)
    bridge = BrowserBridge(host="localhost", port=29229)
    assert bridge.connect() is False
    assert bridge.is_connected() is False
    bridge.close()  # idempotent


def test_browser_bridge_connect_returns_false_when_no_pages_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty /json response means Chrome is up but has no real page tabs."""

    class _FakeResp:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_a):  # noqa: ANN001
            return False

        def read(self) -> bytes:
            return self._payload

    monkeypatch.setattr(
        "agent.browser_bridge.urllib.request.urlopen",
        lambda *_a, **_k: _FakeResp(b"[]"),
    )
    bridge = BrowserBridge()
    assert bridge.connect() is False


# ---------------------------------------------------------------- bridge: send


class _FakeWebSocket:
    """In-memory CDP websocket double.

    ``replies`` is a list of message dicts to return from ``recv()`` in
    order. ``sent`` accumulates payloads passed to ``send()`` so tests can
    assert the JSON-RPC requests were shaped correctly.
    """

    def __init__(self, replies: list[dict]) -> None:
        self._replies = deque(replies)
        self.sent: list[dict] = []
        self.closed = False

    def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    def recv(self) -> str:
        if not self._replies:
            raise RuntimeError("FakeWebSocket exhausted")
        return json.dumps(self._replies.popleft())

    def close(self) -> None:
        self.closed = True


def _bridge_with_ws(replies: list[dict]) -> tuple[BrowserBridge, _FakeWebSocket]:
    """Build a bridge with the fake websocket pre-attached.

    Skips ``connect()`` entirely — we don't want to hit the network. The
    public surface (navigate / click / fill) only checks ``is_connected()``,
    which is True iff ``self._ws is not None``.
    """
    bridge = BrowserBridge()
    fake = _FakeWebSocket(replies)
    bridge._ws = fake
    return bridge, fake


def test_browser_bridge_navigate_happy_path() -> None:
    bridge, fake = _bridge_with_ws(
        [{"id": 1, "result": {"frameId": "F1"}}]
    )
    ok, msg = bridge.navigate("https://example.com")
    assert ok is True
    assert "https://example.com" in msg
    # The JSON-RPC request must carry the right method + URL.
    assert fake.sent == [
        {
            "id": 1,
            "method": "Page.navigate",
            "params": {"url": "https://example.com"},
        }
    ]


def test_browser_bridge_navigate_rejects_relative_url() -> None:
    bridge, fake = _bridge_with_ws([])
    ok, msg = bridge.navigate("youtube.com")
    assert ok is False
    assert "http://" in msg
    # Must not have sent anything to CDP.
    assert fake.sent == []


def test_browser_bridge_click_returns_not_found_for_missing_selector() -> None:
    """When querySelector returns null, JS returns 'NOT_FOUND' and the
    bridge surfaces this distinct from generic failure so the planner
    knows to try a different selector."""
    bridge, _ = _bridge_with_ws(
        [
            {
                "id": 1,
                "result": {"result": {"type": "string", "value": "NOT_FOUND"}},
            }
        ]
    )
    ok, msg = bridge.click("#does-not-exist")
    assert ok is False
    assert "selector not found" in msg


def test_browser_bridge_click_ignores_unrelated_events_before_reply() -> None:
    """CDP multiplexes Page/Network events on the same socket; the bridge
    must skip messages without a matching id."""
    bridge, fake = _bridge_with_ws(
        [
            # An unsolicited Page.frameStartedLoading event (no id).
            {"method": "Page.frameStartedLoading", "params": {}},
            # Actual reply for our id=1.
            {
                "id": 1,
                "result": {"result": {"type": "string", "value": "OK"}},
            },
        ]
    )
    ok, msg = bridge.click("button.go")
    assert ok is True
    assert "button.go" in msg


def test_browser_bridge_fill_redacts_value_in_action_text() -> None:
    bridge, fake = _bridge_with_ws(
        [
            {
                "id": 1,
                "result": {"result": {"type": "string", "value": "OK"}},
            }
        ]
    )
    ok, msg = bridge.fill("input#password", "supersecret")
    assert ok is True
    # The action_text returned must NOT contain the literal value.
    assert "supersecret" not in msg
    assert "<REDACTED, 11 chars>" in msg
    # The CDP payload sent to Chrome DOES contain the value (it has to —
    # that's what we're asking Chrome to set). It just stays on localhost.
    sent_expr = fake.sent[0]["params"]["expression"]
    assert "supersecret" in sent_expr


def test_browser_bridge_send_reports_cdp_errors() -> None:
    """A CDP error reply (e.g. invalid selector) must surface as ok=False."""
    bridge, _ = _bridge_with_ws(
        [
            {
                "id": 1,
                "error": {"code": -32000, "message": "Invalid arguments"},
            }
        ]
    )
    ok, msg = bridge.click("button[broken")
    assert ok is False
    assert "Invalid arguments" in msg


def test_browser_bridge_methods_fail_clean_when_not_connected() -> None:
    """Every public method must return ok=False instead of raising when
    ``is_connected()`` is False — the executor relies on this."""
    bridge = BrowserBridge()
    assert bridge.is_connected() is False
    assert bridge.navigate("https://x.com") == (
        False,
        "BrowserBridge not connected",
    )
    assert bridge.click("#x") == (False, "BrowserBridge not connected")
    assert bridge.fill("#x", "v") == (False, "BrowserBridge not connected")


# ---------------------------------------------------------------- helpers


def test_extract_eval_value_handles_exception_replies() -> None:
    assert _extract_eval_value({"exceptionDetails": {"text": "boom"}}) == "<EXCEPTION>"


def test_extract_eval_value_returns_inner_value() -> None:
    reply = {"result": {"type": "string", "value": "OK"}}
    assert _extract_eval_value(reply) == "OK"


def test_cdp_target_is_real_page_filters_devtools() -> None:
    real = _CdpTarget(target_id="1", url="https://example.com", websocket_url="ws://x")
    fake = _CdpTarget(
        target_id="2", url="devtools://devtools/x", websocket_url="ws://x"
    )
    assert real.is_real_page is True
    assert fake.is_real_page is False
