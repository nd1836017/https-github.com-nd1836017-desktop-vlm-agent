"""Tests for the task decomposer.

Covers:
* parse_mode: case + whitespace tolerant; unknown -> AUTO with warning.
* apply_decomposer: mode=off pass-through; mode=auto with stub client;
  graceful fallback on errors; row metadata + manual annotations
  preserved.
* decompose_step_texts validation: rejects out-of-range indices, empty
  texts, missing inputs, and reordered outputs.
* apply_decomposition: builds TaskSteps that inherit row metadata.

Gemini is never actually called — tests use a stub that returns
hand-built ``_DecomposerResponseModel`` instances.
"""
from __future__ import annotations

import logging

import pytest

from agent.task_decomposer import (
    DecomposerUnavailable,
    DecompositionMode,
    _DecomposedStep,
    _DecomposerResponseModel,
    apply_decomposer,
    apply_decomposition,
    decompose_step_texts,
    parse_mode,
)
from agent.task_router import RouterUnavailable, RoutingHint
from agent.tasks_loader import TaskStep


class _StubClient:
    """Minimal stand-in for ``GeminiClient.call_router_raw``."""

    def __init__(
        self,
        *,
        response: _DecomposerResponseModel | None = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        self._response = response
        self._raise_on_call = raise_on_call
        self.call_count = 0
        self.last_user_prompt: str = ""

    def call_router_raw(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type,
    ):
        self.call_count += 1
        self.last_user_prompt = user_prompt
        if self._raise_on_call is not None:
            raise self._raise_on_call
        assert self._response is not None
        return self._response


def _step(text: str, **kwargs) -> TaskStep:
    """Build a TaskStep with sensible defaults (no row, no hint)."""
    return TaskStep(
        text=text,
        row_index=kwargs.pop("row_index", None),
        csv_name=kwargs.pop("csv_name", None),
        routing_hint=kwargs.pop("routing_hint", None),
    )


def _resp(*pairs: tuple[int, str]) -> _DecomposerResponseModel:
    return _DecomposerResponseModel(
        steps=[
            _DecomposedStep(original_index=idx, text=text) for idx, text in pairs
        ]
    )


# ----- parse_mode -----


class TestParseMode:
    def test_none_defaults_to_auto(self) -> None:
        assert parse_mode(None) == DecompositionMode.AUTO

    def test_blank_defaults_to_auto(self) -> None:
        assert parse_mode("") == DecompositionMode.AUTO

    def test_case_insensitive_and_trimmed(self) -> None:
        assert parse_mode("  AUTO  ") == DecompositionMode.AUTO
        assert parse_mode("Off") == DecompositionMode.OFF

    def test_unknown_warns_and_falls_back_to_auto(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="agent.task_decomposer"):
            mode = parse_mode("turbo")
        assert mode == DecompositionMode.AUTO
        assert any("not recognised" in r.message for r in caplog.records)


# ----- apply_decomposer (high-level entry point) -----


class TestApplyDecomposerHighLevel:
    def test_mode_off_returns_input_unchanged(self) -> None:
        steps = [_step("a"), _step("b")]
        client = _StubClient()
        out = apply_decomposer(steps, mode=DecompositionMode.OFF, client=client)
        assert [s.text for s in out] == ["a", "b"]
        assert client.call_count == 0

    def test_empty_steps_returns_empty(self) -> None:
        client = _StubClient()
        out = apply_decomposer(
            [], mode=DecompositionMode.AUTO, client=client
        )
        assert out == []
        assert client.call_count == 0

    def test_no_client_acts_like_off(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        steps = [_step("a")]
        with caplog.at_level(logging.WARNING, logger="agent.task_decomposer"):
            out = apply_decomposer(
                steps, mode=DecompositionMode.AUTO, client=None
            )
        assert [s.text for s in out] == ["a"]
        assert any("no Gemini client" in r.message for r in caplog.records)

    def test_atomic_passthrough_called_once(self) -> None:
        steps = [_step("open chrome"), _step("press enter")]
        client = _StubClient(
            response=_resp((0, "open chrome"), (1, "press enter"))
        )
        out = apply_decomposer(
            steps, mode=DecompositionMode.AUTO, client=client
        )
        assert client.call_count == 1
        assert [s.text for s in out] == ["open chrome", "press enter"]

    def test_compound_expanded_into_atomic_substeps(self) -> None:
        # The classic "play the 2nd video on youtube" case from the bug
        # report. One input -> 4 atomic outputs.
        steps = [_step("play the 2nd video on youtube")]
        client = _StubClient(
            response=_resp(
                (0, "open a new Chrome tab"),
                (0, "go to youtube.com"),
                (0, "scroll the video grid into view"),
                (0, "click the second video result"),
            )
        )
        out = apply_decomposer(
            steps, mode=DecompositionMode.AUTO, client=client
        )
        assert client.call_count == 1
        assert len(out) == 4
        assert out[0].text == "open a new Chrome tab"
        assert out[3].text == "click the second video result"

    def test_row_metadata_preserved_through_decomposition(self) -> None:
        # A FOR_EACH_ROW expansion attaches row_index + csv_name to each
        # TaskStep. The decomposer must preserve those when it expands.
        steps = [
            _step("buy item", row_index=3, csv_name="orders"),
        ]
        client = _StubClient(
            response=_resp(
                (0, "click the buy button"),
                (0, "click confirm"),
            )
        )
        out = apply_decomposer(
            steps, mode=DecompositionMode.AUTO, client=client
        )
        assert len(out) == 2
        for s in out:
            assert s.row_index == 3
            assert s.csv_name == "orders"

    def test_manual_routing_hint_preserved(self) -> None:
        # Manual [browser-fast] annotation must survive decomposition so
        # the router's downstream apply step still respects it.
        hint = RoutingHint(
            complexity="browser-fast", suggested_command="", source="manual"
        )
        steps = [_step("open youtube", routing_hint=hint)]
        client = _StubClient(response=_resp((0, "open youtube")))
        out = apply_decomposer(
            steps, mode=DecompositionMode.AUTO, client=client
        )
        assert len(out) == 1
        assert out[0].routing_hint is not None
        assert out[0].routing_hint.source == "manual"
        assert out[0].routing_hint.complexity == "browser-fast"

    def test_router_unavailable_falls_through_gracefully(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Any exception type — RouterUnavailable, ValueError, RuntimeError —
        # should be caught and logged, never raised.
        steps = [_step("a"), _step("b")]
        client = _StubClient(
            raise_on_call=RouterUnavailable("boom")
        )
        with caplog.at_level(logging.WARNING, logger="agent.task_decomposer"):
            out = apply_decomposer(
                steps, mode=DecompositionMode.AUTO, client=client
            )
        # Same identity-ish pass-through (list copy, same texts).
        assert [s.text for s in out] == ["a", "b"]
        assert any("decomposition failed" in r.message for r in caplog.records)

    def test_arbitrary_exception_falls_through(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        steps = [_step("a")]
        client = _StubClient(raise_on_call=ValueError("schema bad"))
        with caplog.at_level(logging.WARNING, logger="agent.task_decomposer"):
            out = apply_decomposer(
                steps, mode=DecompositionMode.AUTO, client=client
            )
        assert [s.text for s in out] == ["a"]
        assert any("decomposition failed" in r.message for r in caplog.records)


# ----- decompose_step_texts (validation paths) -----


class TestDecomposeStepTextsValidation:
    def test_empty_input_returns_empty(self) -> None:
        client = _StubClient()
        assert decompose_step_texts(client, []) == []
        assert client.call_count == 0

    def test_empty_response_raises(self) -> None:
        client = _StubClient(response=_DecomposerResponseModel(steps=[]))
        with pytest.raises(DecomposerUnavailable, match="empty list"):
            decompose_step_texts(client, ["a"])

    def test_out_of_range_index_raises(self) -> None:
        client = _StubClient(response=_resp((5, "bogus")))
        with pytest.raises(DecomposerUnavailable, match="out of range"):
            decompose_step_texts(client, ["a"])

    def test_negative_index_raises(self) -> None:
        client = _StubClient(response=_resp((-1, "bogus")))
        with pytest.raises(DecomposerUnavailable, match="out of range"):
            decompose_step_texts(client, ["a"])

    def test_empty_text_raises(self) -> None:
        client = _StubClient(response=_resp((0, "   ")))
        with pytest.raises(DecomposerUnavailable, match="empty text"):
            decompose_step_texts(client, ["a"])

    def test_missing_input_index_raises(self) -> None:
        # Two inputs but the response only references index 0.
        client = _StubClient(response=_resp((0, "x")))
        with pytest.raises(DecomposerUnavailable, match="skipped input"):
            decompose_step_texts(client, ["a", "b"])

    def test_reordered_indices_raises(self) -> None:
        # original_index must be non-decreasing across the output.
        client = _StubClient(response=_resp((1, "y"), (0, "x")))
        with pytest.raises(DecomposerUnavailable, match="reordered"):
            decompose_step_texts(client, ["a", "b"])


# ----- apply_decomposition (low-level expansion) -----


class TestApplyDecomposition:
    def test_no_decomposition_returns_input_copy(self) -> None:
        steps = [_step("a")]
        out = apply_decomposition(steps, [])
        # Same content but a new list.
        assert out == steps
        assert out is not steps

    def test_expansion_preserves_metadata_per_original(self) -> None:
        steps = [
            _step("compound", row_index=0, csv_name="data"),
            _step("atomic", row_index=1, csv_name="data"),
        ]
        decomposed = [
            _DecomposedStep(original_index=0, text="part 1"),
            _DecomposedStep(original_index=0, text="part 2"),
            _DecomposedStep(original_index=1, text="atomic"),
        ]
        out = apply_decomposition(steps, decomposed)
        assert [s.text for s in out] == ["part 1", "part 2", "atomic"]
        assert out[0].row_index == 0
        assert out[1].row_index == 0
        assert out[2].row_index == 1
