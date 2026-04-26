"""Tests for PR D: REMEMBER/RECALL + Tier 4 reliability.

Covers:
- ``GeminiClient.extract_value`` parses text + JSON forms.
- ``image_signature`` is stable across repeat captures.
- REMEMBER literal form populates the variable store.
- REMEMBER screen-extract form falls back to the VLM and stores results.
- RECALL types the stored value via the standard executor.
- ``{{var.<name>}}`` substitution happens at run time, not load time.
- Per-step wall-clock timeout fails the step early.
- Stuck-step detection bails before exhausting the replan budget.
- ``RunFeatures.uses_variables`` gates the VariableStore setup.
- Variables persist across checkpoint save / load.
"""
from __future__ import annotations

import sys
from typing import cast
from unittest import mock

import pytest
from PIL import Image

from agent.agent import _execute_recall, _execute_remember, run, run_step
from agent.config import Config
from agent.files import inspect_features
from agent.history import History
from agent.parser import RecallCommand, RememberCommand
from agent.screen import ScreenGeometry, image_signature
from agent.state import STATE_SCHEMA_VERSION, AgentState, load_state, save_state
from agent.tasks_loader import load_steps
from agent.variables import (
    UnknownVariableError,
    VariableStore,
    substitute_variables,
    text_uses_variables,
)
from agent.vlm import (
    ExtractionResult,
    GeminiClient,
    VerificationResult,
    _parse_extract_response_json,
)

# ---------------------------------------------------------------------- helpers


class _FakeResponse:
    """Mimic the google-genai response object."""

    def __init__(self, text: str, parsed=None):
        self.text = text
        self.parsed = parsed


class _FakePlanOnlyClient:
    """Drop-in fake for GeminiClient used by run_step end-to-end tests."""

    def __init__(self, plan_outputs, verify_outputs, extract_outputs=None):
        self._plan_outputs = list(plan_outputs)
        self._verify_outputs = list(verify_outputs)
        self._extract_outputs = list(extract_outputs or [])
        self.plan_calls = 0
        self.verify_calls = 0
        self.extract_calls = 0

    def plan_action(
        self,
        step,
        screenshot,
        history_summary: str = "",
        previous_failure: str = "",
        extra_images=None,
    ):
        self.plan_calls += 1
        out = self._plan_outputs.pop(0)
        if isinstance(out, tuple):
            return out
        return out, None

    def verify(self, goal, screenshot):
        self.verify_calls += 1
        out = self._verify_outputs.pop(0)
        if isinstance(out, VerificationResult):
            return out
        passed, reason = out
        return VerificationResult(passed=passed, reason=reason)

    def extract_value(self, name, screenshot, hint: str = ""):
        self.extract_calls += 1
        return self._extract_outputs.pop(0)


@pytest.fixture
def fake_geometry():
    return ScreenGeometry(width=1920, height=1080)


@pytest.fixture(autouse=True)
def patch_pyautogui():
    fake_img = Image.new("RGB", (8, 8), color=(0, 0, 0))
    fake_pyautogui = mock.MagicMock()
    fake_pyautogui.size.return_value = mock.MagicMock(width=1920, height=1080)
    fake_pyautogui.screenshot.return_value = fake_img
    fake_pyautogui.FAILSAFE = False
    with mock.patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
        yield fake_pyautogui


def _cfg(tmp_path, tasks_text: str, **overrides) -> Config:
    tasks_file = tmp_path / "tasks.txt"
    tasks_file.write_text(tasks_text, encoding="utf-8")
    defaults = dict(
        gemini_api_key="fake",
        gemini_model="fake-model",
        tasks_file=tasks_file,
        animation_buffer_seconds=0.0,
        max_step_retries=1,
        max_replans_per_step=2,
        history_window=5,
        state_file=tmp_path / ".agent_state.json",
        enable_two_stage_click=False,
        two_stage_crop_size_px=300,
        max_click_candidates=5,
        click_min_delay_seconds=0.0,
        click_max_delay_seconds=0.0,
        type_min_interval_seconds=0.0,
        type_max_interval_seconds=0.0,
        gemini_retry_max_attempts=1,
        gemini_retry_base_delay_seconds=0.0,
        gemini_retry_max_delay_seconds=0.0,
        log_redact_type=True,
        enable_json_output=False,
        max_total_replans=0,
        save_run_artifacts=False,
        run_artifacts_dir=tmp_path / "runs",
        rpd_limit=0,
        rpd_warn_threshold=0.75,
        rpd_halt_threshold=0.95,
        file_mode=None,
        workdir=None,
        log_level="INFO",
        step_timeout_seconds=0.0,
        stuck_step_threshold=0,
        vlm_image_max_dim=1280,
        vlm_image_quality=80,
        vlm_skip_identical_frames=False,
    )
    defaults.update(overrides)
    return Config(**defaults)


# ---------------------------------------------------------- extract_value parsing


def test_parse_extract_response_json_valid():
    parsed = _parse_extract_response_json('{"found": true, "value": "ND12345"}')
    assert parsed is not None
    assert parsed.found is True
    assert parsed.value == "ND12345"


def test_parse_extract_response_json_with_fences():
    parsed = _parse_extract_response_json(
        '```json\n{"found": false, "value": ""}\n```'
    )
    assert parsed is not None
    assert parsed.found is False
    assert parsed.value == ""


def test_parse_extract_response_json_invalid():
    assert _parse_extract_response_json("not json") is None
    assert _parse_extract_response_json("[1,2,3]") is None
    assert _parse_extract_response_json("") is None


def test_extract_text_parser_value_prefix():
    result = GeminiClient._parse_extract_text("VALUE: ND-12345")
    assert result.found is True
    assert result.value == "ND-12345"


def test_extract_text_parser_none():
    result = GeminiClient._parse_extract_text("NONE")
    assert result.found is False
    assert result.value == ""


def test_extract_text_parser_none_with_punctuation():
    """Regression: a slightly verbose model emits ``NONE.`` or ``NONE: ...``
    when it can't find the value. The previous parser only matched bare
    ``NONE`` and ``NONE\\n…``, so these forms were misparsed as a
    successful extraction of the literal string ``NONE.``.
    """
    for variant in (
        "NONE.",
        "NONE: value not found",
        "NONE — couldn't find it",
        "NONE\nreason: blocked by modal",
        "  NONE  ",
        "VALUE: NONE",
        "VALUE: NONE.",
    ):
        result = GeminiClient._parse_extract_text(variant)
        assert result.found is False, f"variant {variant!r} should be NONE"
        assert result.value == ""


def test_extract_text_parser_does_not_swallow_nonexistent_value():
    """The fix uses a word boundary so ``NONEXISTENT`` and other words
    that merely *start* with ``NONE`` are still extractable.
    """
    result = GeminiClient._parse_extract_text("NONEXISTENT-FILE")
    assert result.found is True
    assert result.value == "NONEXISTENT-FILE"

    result2 = GeminiClient._parse_extract_text("VALUE: NONESUCH-12345")
    assert result2.found is True
    assert result2.value == "NONESUCH-12345"


def test_extract_text_parser_freeform_takes_first_line():
    result = GeminiClient._parse_extract_text("ND-99999\nextra junk")
    assert result.found is True
    assert result.value == "ND-99999"


def test_extract_text_parser_empty():
    result = GeminiClient._parse_extract_text("")
    assert result.found is False
    assert result.value == ""


# --------------------------------------------------------- image_signature shape


def test_image_signature_stable_across_calls():
    img = Image.new("RGB", (200, 100), color=(80, 80, 80))
    s1 = image_signature(img)
    s2 = image_signature(img)
    assert s1 == s2
    assert isinstance(s1, str)
    # SHA-1 hex is exactly 40 chars; the function downsamples then hashes.
    assert len(s1) == 40


def test_image_signature_changes_with_content():
    a = Image.new("RGB", (200, 100), color=(0, 0, 0))
    b = Image.new("RGB", (200, 100), color=(255, 255, 255))
    assert image_signature(a) != image_signature(b)


# ---------------------------------------------------------- REMEMBER literal form


def test_execute_remember_literal_stores_value():
    store = VariableStore()
    cmd = RememberCommand(
        name="order_id", literal_value="ND12345", from_screen=False
    )
    img = Image.new("RGB", (4, 4))
    ok, action_text = _execute_remember(
        cmd=cmd, screenshot=img, vlm=mock.MagicMock(), variables=store, step_text=""
    )
    assert ok is True
    assert "REMEMBER" in action_text
    assert store.get("order_id") == "ND12345"


def test_execute_remember_literal_does_not_call_vlm():
    store = VariableStore()
    vlm = mock.MagicMock()
    cmd = RememberCommand(
        name="order_id", literal_value="ND12345", from_screen=False
    )
    img = Image.new("RGB", (4, 4))
    ok, _ = _execute_remember(
        cmd=cmd, screenshot=img, vlm=vlm, variables=store, step_text=""
    )
    assert ok is True
    # extract_value must NOT be called for the literal form.
    vlm.extract_value.assert_not_called()


def test_execute_remember_screen_form_calls_vlm_and_stores():
    store = VariableStore()
    vlm = mock.MagicMock()
    vlm.extract_value.return_value = ExtractionResult(
        found=True, value="ND-99999", raw="raw"
    )
    cmd = RememberCommand(name="order_id", literal_value="", from_screen=True)
    img = Image.new("RGB", (4, 4))
    ok, action_text = _execute_remember(
        cmd=cmd,
        screenshot=img,
        vlm=vlm,
        variables=store,
        step_text="extract the order id",
    )
    assert ok is True
    assert store.get("order_id") == "ND-99999"
    vlm.extract_value.assert_called_once()
    # The step text is passed as a hint for context.
    _args, kwargs = vlm.extract_value.call_args
    assert kwargs.get("hint") == "extract the order id"


def test_execute_remember_fail_when_vlm_returns_not_found():
    store = VariableStore()
    vlm = mock.MagicMock()
    vlm.extract_value.return_value = ExtractionResult(
        found=False, value="", raw=""
    )
    cmd = RememberCommand(name="missing", literal_value="", from_screen=True)
    ok, reason = _execute_remember(
        cmd=cmd,
        screenshot=Image.new("RGB", (4, 4)),
        vlm=vlm,
        variables=store,
        step_text="",
    )
    assert ok is False
    assert "missing" not in store


# ---------------------------------------------------------------- RECALL execute


def test_execute_recall_types_stored_value(fake_geometry):
    store = VariableStore()
    store.set("order_id", "ND-12345")
    cmd = RecallCommand(name="order_id")
    with mock.patch("agent.agent.execute") as mock_exec:
        ok, action_text = _execute_recall(
            cmd=cmd,
            geometry=fake_geometry,
            variables=store,
            animation_buffer=0.0,
            type_min_interval=0.0,
            type_max_interval=0.0,
            log_redact_type=True,
        )
    assert ok is True
    assert "RECALL" in action_text
    assert "8 chars" in action_text
    # The synthetic TypeCommand was passed to the executor.
    assert mock_exec.called
    args, kwargs = mock_exec.call_args
    synthetic = args[0]
    assert synthetic.kind == "TYPE"
    assert synthetic.text == "ND-12345"


def test_execute_recall_fails_when_variable_missing(fake_geometry):
    store = VariableStore()
    cmd = RecallCommand(name="never_set")
    with mock.patch("agent.agent.execute") as mock_exec:
        ok, reason = _execute_recall(
            cmd=cmd,
            geometry=fake_geometry,
            variables=store,
            animation_buffer=0.0,
            type_min_interval=0.0,
            type_max_interval=0.0,
            log_redact_type=True,
        )
    assert ok is False
    assert "unset" in reason
    mock_exec.assert_not_called()


# ----------------------------------------------------- variable substitution time


def test_substitute_variables_simple():
    store = VariableStore()
    store.set("name", "World")
    out = substitute_variables("Hello {{var.name}}!", store)
    assert out == "Hello World!"


def test_substitute_variables_with_default():
    store = VariableStore()
    out = substitute_variables("Hello {{var.name|stranger}}!", store)
    assert out == "Hello stranger!"


def test_substitute_variables_unknown_raises():
    store = VariableStore()
    with pytest.raises(UnknownVariableError):
        substitute_variables("Hello {{var.missing}}!", store)


def test_text_uses_variables_detection():
    assert text_uses_variables("type {{var.name}}") is True
    assert text_uses_variables("type {{ var.name }}") is True
    assert text_uses_variables("type {{ var.name | default }}") is True
    assert text_uses_variables("type Hello") is False
    # {{row.x}} placeholders (CSV) should not trip the variable detector.
    assert text_uses_variables("type {{row.x}}") is False


# -------------------------------------------------- features detection


def test_inspect_features_picks_up_remember_recall_var(tmp_path):
    tasks = tmp_path / "tasks.txt"
    tasks.write_text(
        "REMEMBER [order_id]\n"
        "click on Submit\n"
        "RECALL [order_id]\n"
        "open page about {{var.order_id}}\n",
        encoding="utf-8",
    )
    steps = load_steps(tasks)
    features = inspect_features(steps)
    assert features.uses_variables is True
    assert features.remember_count == 1
    assert features.recall_count == 1
    assert features.var_placeholder_count == 1


def test_inspect_features_no_variables_when_absent(tmp_path):
    tasks = tmp_path / "tasks.txt"
    tasks.write_text(
        "press Win\n"
        "type Notepad\n"
        "press Enter\n",
        encoding="utf-8",
    )
    steps = load_steps(tasks)
    features = inspect_features(steps)
    assert features.uses_variables is False
    assert features.remember_count == 0
    assert features.recall_count == 0
    assert features.var_placeholder_count == 0


def test_inspect_features_does_not_count_prose_remember(tmp_path):
    tasks = tmp_path / "tasks.txt"
    tasks.write_text(
        "please remember to press Submit later\n",
        encoding="utf-8",
    )
    steps = load_steps(tasks)
    features = inspect_features(steps)
    # No bracketed REMEMBER[name] form → not counted.
    assert features.remember_count == 0
    assert features.uses_variables is False


# ----------------------------------------------------------- state migration


def test_state_v1_loads_with_empty_variables(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"version": 1, "tasks_file": "tasks.txt", '
        '"total_steps": 5, "last_completed_step": 2}',
        encoding="utf-8",
    )
    loaded = load_state(state_path)
    assert loaded is not None
    assert loaded.version == 1
    assert loaded.last_completed_step == 2
    assert loaded.variables == {}


def test_state_v2_round_trips_variables(tmp_path):
    state_path = tmp_path / "state.json"
    initial = AgentState.initial(tmp_path / "tasks.txt", total_steps=3)
    initial = initial.with_variables({"order_id": "ND-42", "user": "alice"})
    save_state(state_path, initial)
    loaded = load_state(state_path)
    assert loaded is not None
    assert loaded.version == STATE_SCHEMA_VERSION
    assert loaded.variables == {"order_id": "ND-42", "user": "alice"}


def test_state_advance_carries_variables():
    s = AgentState.initial("/tmp/x", total_steps=3).with_variables({"a": "1"})
    advanced = s.advance()
    assert advanced.last_completed_step == 1
    assert advanced.variables == {"a": "1"}


# ----------------------------------------------------- run_step Tier 4 wiring


def test_run_step_timeout_fails_step_when_exceeded(fake_geometry):
    """If the wall-clock budget expires before any attempt, return a clear FAIL."""
    client = _FakePlanOnlyClient(plan_outputs=["TYPE [hi]"], verify_outputs=[])
    history = History(window=5)
    # Use a tiny timeout and a sleep mock to ensure deadline triggers.
    with mock.patch("agent.agent.time") as fake_time:
        # First check: now=0, deadline=0.5 → no timeout, attempt proceeds...
        # but we rig time.monotonic to immediately exceed the deadline so
        # the very first iteration of the loop bails.
        fake_time.monotonic.side_effect = [0.0, 100.0]
        result = run_step(
            step="press Enter",
            vlm=cast(GeminiClient, client),
            geometry=fake_geometry,
            animation_buffer=0.0,
            max_parse_retries=0,
            max_replans=2,
            history=history,
            enable_two_stage_click=False,
            step_timeout_seconds=0.5,
            stuck_step_threshold=0,
        )
    assert result.passed is False
    assert "wall-clock" in result.reason
    # The step never even had a chance to call the planner.
    assert client.plan_calls == 0


def test_run_step_stuck_step_detection_bails_early(fake_geometry):
    """3 consecutive identical post-action screenshots triggers an early FAIL."""
    # 3 attempts that all FAIL (initial + 2 replans), all with same screenshot.
    plan_outputs = ["TYPE [a]", "TYPE [b]", "TYPE [c]"]
    verify_outputs = [
        VerificationResult(passed=False, reason="nothing changed"),
        VerificationResult(passed=False, reason="nothing changed"),
        VerificationResult(passed=False, reason="nothing changed"),
    ]
    client = _FakePlanOnlyClient(plan_outputs=plan_outputs, verify_outputs=verify_outputs)
    history = History(window=5)
    # Pin image_signature so all 3 produce the same fingerprint.
    with mock.patch("agent.agent.image_signature", return_value="frozen-sig"):
        result = run_step(
            step="press Enter",
            vlm=cast(GeminiClient, client),
            geometry=fake_geometry,
            animation_buffer=0.0,
            max_parse_retries=0,
            max_replans=10,  # plenty of slack — bail must come from stuck check
            history=history,
            enable_two_stage_click=False,
            step_timeout_seconds=0.0,
            stuck_step_threshold=3,
        )
    assert result.passed is False
    assert "stuck" in result.reason.lower()
    # Bail at attempt 3, not 11.
    assert client.plan_calls == 3
    # Stuck-step HALT reason must surface the action that kept failing
    # so a postmortem reader can see WHAT the agent kept trying.
    # (UX upgrade #1 — better stuck message.) TYPE payloads are
    # privacy-redacted to avoid leaking user-typed text into error logs;
    # the action *class* (TYPE) and length are still visible.
    assert "Last action attempted" in result.reason
    assert "TYPE [" in result.reason  # command class survives redaction
    assert "nothing changed" in result.reason


def test_run_step_no_stuck_when_screen_changes(fake_geometry):
    """When successive attempts produce DIFFERENT screens, no early bail."""
    plan_outputs = ["TYPE [a]", "TYPE [b]"]
    verify_outputs = [
        VerificationResult(passed=False, reason="failed"),
        VerificationResult(passed=True, reason="ok"),
    ]
    client = _FakePlanOnlyClient(plan_outputs=plan_outputs, verify_outputs=verify_outputs)
    history = History(window=5)
    sigs = iter(["sig-a", "sig-b", "sig-c"])
    with mock.patch(
        "agent.agent.image_signature", side_effect=lambda *_a, **_kw: next(sigs)
    ):
        result = run_step(
            step="do thing",
            vlm=cast(GeminiClient, client),
            geometry=fake_geometry,
            animation_buffer=0.0,
            max_parse_retries=0,
            max_replans=3,
            history=history,
            enable_two_stage_click=False,
            stuck_step_threshold=3,
        )
    assert result.passed is True


# ----------------------------------------------------- end-to-end run() var substitution


def test_run_substitutes_variables_at_step_time(tmp_path, fake_geometry):
    """A step containing {{var.X}} sees the value set by an earlier REMEMBER."""
    tasks_text = (
        "REMEMBER [name = world]\n"
        "type hello {{var.name}}\n"
    )
    cfg = _cfg(tmp_path, tasks_text)

    # 1st step: REMEMBER [name = world] — parsed by the regex, no VLM call.
    # 2nd step: TYPE [hello world] — TextClient returns this exactly.
    plan_outputs = [
        "REMEMBER [name = world]",
        "TYPE [hello world]",
    ]
    verify_outputs = [
        # REMEMBER doesn't go through the visual verifier (synthesized PASS),
        # so only the TYPE step verifies.
        VerificationResult(passed=True, reason="typed"),
    ]
    fake = _FakePlanOnlyClient(plan_outputs=plan_outputs, verify_outputs=verify_outputs)
    captured_steps: list[str] = []
    real_plan_action = fake.plan_action

    def spy(step, *args, **kwargs):
        captured_steps.append(step)
        return real_plan_action(step, *args, **kwargs)

    fake.plan_action = spy

    with (
        mock.patch("agent.agent.GeminiClient", return_value=fake),
        mock.patch("agent.agent.detect_geometry", return_value=fake_geometry),
        mock.patch("agent.agent.execute"),
    ):
        rc = run(cfg)
    assert rc == 0
    # Second step's `step` argument should already have {{var.name}} resolved.
    assert any("hello world" in s for s in captured_steps)
    assert all("{{var.name}}" not in s for s in captured_steps)


def test_run_halts_on_unresolved_variable(tmp_path, fake_geometry, capsys):
    """A {{var.X}} placeholder with no REMEMBER + no default produces a clean halt."""
    tasks_text = "type hello {{var.never_set}}\n"
    cfg = _cfg(tmp_path, tasks_text)
    fake = _FakePlanOnlyClient(plan_outputs=[], verify_outputs=[])
    with (
        mock.patch("agent.agent.GeminiClient", return_value=fake),
        mock.patch("agent.agent.detect_geometry", return_value=fake_geometry),
    ):
        rc = run(cfg)
    assert rc == 1
    captured = capsys.readouterr()
    assert "unresolved variable" in captured.err.lower()
    # We should NOT have called the planner — the halt happens before it.
    assert fake.plan_calls == 0


def test_run_does_not_setup_variables_when_unused(tmp_path, fake_geometry):
    """If tasks.txt has no REMEMBER/RECALL/{{var}}, no var store is created."""
    tasks_text = "press Win\n"
    cfg = _cfg(tmp_path, tasks_text)
    fake = _FakePlanOnlyClient(
        plan_outputs=["PRESS [Win]"],
        verify_outputs=[VerificationResult(passed=True, reason="ok")],
    )
    with (
        mock.patch("agent.agent.GeminiClient", return_value=fake),
        mock.patch("agent.agent.detect_geometry", return_value=fake_geometry),
        mock.patch("agent.agent.execute"),
    ):
        rc = run(cfg)
    assert rc == 0
    # Sanity check: state checkpoint has empty variables since the run never
    # used them.
    loaded = load_state(cfg.state_file)
    assert loaded is not None
    assert loaded.variables == {}


def test_run_persists_variables_in_checkpoint(tmp_path, fake_geometry):
    """REMEMBER literals survive a restart via the checkpoint file."""
    tasks_text = (
        "REMEMBER [a = first]\n"
        "type hello {{var.a}}\n"
    )
    cfg = _cfg(tmp_path, tasks_text)
    fake = _FakePlanOnlyClient(
        plan_outputs=["REMEMBER [a = first]", "TYPE [hello first]"],
        verify_outputs=[VerificationResult(passed=True, reason="ok")],
    )
    with (
        mock.patch("agent.agent.GeminiClient", return_value=fake),
        mock.patch("agent.agent.detect_geometry", return_value=fake_geometry),
        mock.patch("agent.agent.execute"),
    ):
        rc = run(cfg)
    assert rc == 0
    loaded = load_state(cfg.state_file)
    assert loaded is not None
    assert loaded.variables == {"a": "first"}
