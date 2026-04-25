"""Tests for the JSON-schema path of plan_action / verify."""
from __future__ import annotations

from unittest import mock

import pytest
from PIL import Image

from agent.parser import (
    ClickCommand,
    ClickTextCommand,
    DoubleClickCommand,
    DragCommand,
    MoveToCommand,
    PressCommand,
    RightClickCommand,
    ScrollCommand,
    TypeCommand,
    WaitCommand,
)
from agent.vlm import (
    GeminiClient,
    PlanResponseModel,
    VerifyResponseModel,
    _parse_plan_response_json,
    _parse_verify_response_json,
    plan_response_to_command,
)


@pytest.fixture
def json_client():
    """A GeminiClient with JSON output enabled, no real SDK."""
    with mock.patch("agent.vlm.genai.Client"):
        return GeminiClient(
            api_key="fake", model_name="fake-model", enable_json_output=True
        )


def _stub_json_response(client: GeminiClient, *, text: str, parsed=None):
    """Mimic the SDK: `.text` is the raw model output; `.parsed` is the
    SDK-populated pydantic instance (may be None if the SDK couldn't / didn't).
    """
    resp = mock.MagicMock()
    resp.text = text
    resp.parsed = parsed
    client._client.models.generate_content = mock.MagicMock(return_value=resp)


def test_plan_response_click():
    cmd = plan_response_to_command(
        PlanResponseModel(command="CLICK", x=100, y=200)
    )
    assert cmd == ClickCommand(x=100, y=200)


def test_plan_response_double_click():
    cmd = plan_response_to_command(
        PlanResponseModel(command="DOUBLE_CLICK", x=300, y=400)
    )
    assert cmd == DoubleClickCommand(x=300, y=400)


def test_plan_response_right_click():
    cmd = plan_response_to_command(
        PlanResponseModel(command="RIGHT_CLICK", x=50, y=60)
    )
    assert cmd == RightClickCommand(x=50, y=60)


def test_plan_response_move_to():
    cmd = plan_response_to_command(
        PlanResponseModel(command="MOVE_TO", x=500, y=500)
    )
    assert cmd == MoveToCommand(x=500, y=500)


def test_plan_response_press():
    cmd = plan_response_to_command(PlanResponseModel(command="PRESS", key="enter"))
    assert cmd == PressCommand(key="enter")


def test_plan_response_type():
    cmd = plan_response_to_command(
        PlanResponseModel(command="TYPE", text="hello world")
    )
    assert cmd == TypeCommand(text="hello world")


def test_plan_response_scroll():
    cmd = plan_response_to_command(
        PlanResponseModel(command="SCROLL", direction="down", amount=5)
    )
    assert cmd == ScrollCommand(direction="down", amount=5)


def test_plan_response_scroll_rejects_bad_direction():
    assert (
        plan_response_to_command(
            PlanResponseModel(command="SCROLL", direction="sideways", amount=5)
        )
        is None
    )


def test_plan_response_drag():
    cmd = plan_response_to_command(
        PlanResponseModel(command="DRAG", x1=10, y1=20, x2=30, y2=40)
    )
    assert cmd == DragCommand(x1=10, y1=20, x2=30, y2=40)


def test_plan_response_wait():
    cmd = plan_response_to_command(PlanResponseModel(command="WAIT", seconds=1.5))
    assert cmd == WaitCommand(seconds=1.5)


def test_plan_response_click_text():
    cmd = plan_response_to_command(
        PlanResponseModel(command="CLICK_TEXT", label="Sign in")
    )
    assert cmd == ClickTextCommand(label="Sign in")


def test_plan_response_missing_coords_returns_none():
    # CLICK with no x/y should return None, not a default (0,0) click.
    assert plan_response_to_command(PlanResponseModel(command="CLICK")) is None


def test_plan_response_unknown_kind_returns_none():
    assert (
        plan_response_to_command(PlanResponseModel(command="UNKNOWN", x=1, y=2))
        is None
    )


# --- PR #8 regression: manual JSON fallback when SDK doesn't populate .parsed -

def test_parse_plan_response_json_decodes_valid_blob():
    text = '{"command": "CLICK", "x": 500, "y": 600}'
    parsed = _parse_plan_response_json(text)
    assert isinstance(parsed, PlanResponseModel)
    assert parsed.command == "CLICK"
    assert parsed.x == 500 and parsed.y == 600


def test_parse_plan_response_json_strips_markdown_fences():
    text = '```json\n{"command": "PRESS", "key": "enter"}\n```'
    parsed = _parse_plan_response_json(text)
    assert isinstance(parsed, PlanResponseModel)
    cmd = plan_response_to_command(parsed)
    assert cmd == PressCommand(key="enter")


def test_parse_plan_response_json_returns_none_on_invalid_json():
    assert _parse_plan_response_json("not json at all") is None
    assert _parse_plan_response_json("") is None
    # Valid JSON but not an object — we expect a dict at the top level.
    assert _parse_plan_response_json("[1, 2, 3]") is None


def test_parse_plan_response_json_returns_none_on_missing_required_field():
    # PlanResponseModel requires `command`; an empty dict should fail validation.
    assert _parse_plan_response_json("{}") is None


def test_parse_verify_response_json_pass():
    parsed = _parse_verify_response_json(
        '{"verdict": "PASS", "reason": "address bar focused"}'
    )
    assert isinstance(parsed, VerifyResponseModel)
    assert parsed.verdict == "PASS"
    assert parsed.reason == "address bar focused"


def test_parse_verify_response_json_fail_with_fences():
    parsed = _parse_verify_response_json(
        '```\n{"verdict": "FAIL", "reason": "nothing happened"}\n```'
    )
    assert isinstance(parsed, VerifyResponseModel)
    assert parsed.verdict == "FAIL"
    assert parsed.reason == "nothing happened"


def test_parse_verify_response_json_returns_none_on_non_json():
    # A plain "PASS" string would have been handled by the legacy
    # `_parse_verify_text` path — the JSON helper should decline it.
    assert _parse_verify_response_json("VERDICT: PASS (text mode)") is None
    assert _parse_verify_response_json("") is None


# --- End-to-end regression: SDK returns parsed=None but .text is valid JSON ---

def test_plan_action_recovers_when_sdk_parsed_is_none(json_client):
    """Before PR #8 fix: if `response.parsed` was None (old SDK, schema
    quirk, etc.), plan_action would fall through to `parse_command()` on
    a JSON blob and return (text, None) — the caller would see a parse
    failure. After: we decode the JSON ourselves and return a real Command.
    """
    _stub_json_response(
        json_client,
        text='{"command": "CLICK", "x": 250, "y": 750}',
        parsed=None,  # <-- simulates the bug surface
    )
    text, cmd = json_client.plan_action("step", Image.new("RGB", (10, 10)))
    assert cmd == ClickCommand(x=250, y=750)
    assert '"command"' in text  # raw JSON preserved for logs


def test_verify_recovers_when_sdk_parsed_is_none_pass(json_client):
    """Before PR #8 fix: verify() with parsed=None would send the JSON blob
    through `_parse_verify_text` which searches for literal 'VERDICT: PASS'
    — doesn't match — returns an "Unparseable" FAIL. After: decode JSON
    and return the real verdict.
    """
    _stub_json_response(
        json_client,
        text='{"verdict": "PASS", "reason": "address bar focused"}',
        parsed=None,
    )
    result = json_client.verify("click address bar", Image.new("RGB", (10, 10)))
    assert result.passed is True
    assert "address bar focused" in result.reason


def test_verify_recovers_when_sdk_parsed_is_none_fail(json_client):
    _stub_json_response(
        json_client,
        text='{"verdict": "FAIL", "reason": "nothing changed"}',
        parsed=None,
    )
    result = json_client.verify("click thing", Image.new("RGB", (10, 10)))
    assert result.passed is False
    assert "nothing changed" in result.reason


def test_verify_still_works_when_sdk_populates_parsed(json_client):
    """Happy path regression: parsed is populated, JSON fallback is NOT
    exercised, and the verdict still comes through correctly.
    """
    _stub_json_response(
        json_client,
        text='{"verdict": "PASS"}',
        parsed=VerifyResponseModel(verdict="PASS", reason="looks good"),
    )
    result = json_client.verify("step", Image.new("RGB", (10, 10)))
    assert result.passed is True
    assert "looks good" in result.reason


def test_verify_falls_through_to_text_parser_on_non_json(json_client):
    """If the model emits prose (truly non-JSON), the legacy text parser
    is still the last resort. This guards against accidental regression
    on that path too.
    """
    _stub_json_response(
        json_client,
        text="VERDICT: PASS\nThe page loaded as expected.",
        parsed=None,
    )
    result = json_client.verify("step", Image.new("RGB", (10, 10)))
    assert result.passed is True


# -------------------- File primitives in JSON-output mode -----------------------


def test_plan_response_to_command_download():
    from agent.parser import DownloadCommand

    cmd = plan_response_to_command(
        PlanResponseModel(
            command="DOWNLOAD",
            url="https://example.com/foo.pdf",
            filename="foo.pdf",
        )
    )
    assert isinstance(cmd, DownloadCommand)
    assert cmd.url == "https://example.com/foo.pdf"
    assert cmd.filename == "foo.pdf"


def test_plan_response_to_command_download_url_only():
    from agent.parser import DownloadCommand

    cmd = plan_response_to_command(
        PlanResponseModel(command="DOWNLOAD", url="https://x/y.pdf")
    )
    assert isinstance(cmd, DownloadCommand)
    assert cmd.filename == ""


def test_plan_response_to_command_download_missing_url_returns_none():
    cmd = plan_response_to_command(
        PlanResponseModel(command="DOWNLOAD", filename="orphan.pdf")
    )
    assert cmd is None


def test_plan_response_to_command_attach_file():
    from agent.parser import AttachFileCommand

    cmd = plan_response_to_command(
        PlanResponseModel(command="ATTACH_FILE", filename="resume.pdf")
    )
    assert isinstance(cmd, AttachFileCommand)
    assert cmd.filename == "resume.pdf"


def test_plan_response_to_command_attach_file_missing_filename_returns_none():
    cmd = plan_response_to_command(PlanResponseModel(command="ATTACH_FILE"))
    assert cmd is None


def test_plan_response_to_command_capture_for_ai_no_filename():
    from agent.parser import CaptureForAiCommand

    cmd = plan_response_to_command(PlanResponseModel(command="CAPTURE_FOR_AI"))
    assert isinstance(cmd, CaptureForAiCommand)
    assert cmd.filename == ""


def test_plan_response_to_command_capture_for_ai_with_filename():
    from agent.parser import CaptureForAiCommand

    cmd = plan_response_to_command(
        PlanResponseModel(command="CAPTURE_FOR_AI", filename="snapshot.png")
    )
    assert isinstance(cmd, CaptureForAiCommand)
    assert cmd.filename == "snapshot.png"
