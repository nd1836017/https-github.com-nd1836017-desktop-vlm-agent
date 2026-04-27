"""Tests for the smart task router.

Covers the public surface of ``agent.task_router`` plus its integration
with ``agent.tasks_loader``:

* ``parse_inline_annotation`` handles tagged + untagged + unknown-tag lines.
* ``apply_router`` honors mode=off / manual / auto branches.
* ``apply_router`` in auto mode calls Gemini once and returns aligned hints.
* ``apply_router`` falls through gracefully when Gemini errors.
* ``apply_router`` downgrades browser-fast suggestions when the bridge
  isn't connected.
* ``attach_routing_hints`` preserves manual annotations even when the
  auto router has a conflicting suggestion (manual wins).
* The end-to-end load path strips ``[browser-fast]`` prefixes from the
  step text and surfaces them as ``RoutingHint(source='manual')``.

Gemini is never actually called — every test uses a stub client whose
``call_router_raw`` returns a hand-built ``_RouterResponseModel``.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent.task_router import (
    RouterUnavailable,
    RoutingComplexity,
    RoutingHint,
    RoutingMode,
    _RouterResponseModel,
    _RouterStepDecision,
    apply_router,
    parse_inline_annotation,
    parse_mode,
)
from agent.tasks_loader import (
    TaskStep,
    attach_routing_hints,
    load_steps,
)


class _StubClient:
    """Minimal stand-in for ``GeminiClient`` exposing only ``call_router_raw``.

    Stores the most recent system + user prompts so tests can assert
    that the bridge-availability note is propagated correctly. Returns
    a pre-staged response, or raises ``RouterUnavailable`` when
    ``raise_on_call`` is set.
    """

    def __init__(
        self,
        *,
        response: _RouterResponseModel | None = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        self._response = response
        self._raise_on_call = raise_on_call
        self.last_system_prompt: str = ""
        self.last_user_prompt: str = ""
        self.call_count = 0

    def call_router_raw(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: type,
    ) -> _RouterResponseModel:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        if self._raise_on_call is not None:
            raise self._raise_on_call
        assert self._response is not None
        return self._response


def _decision(
    index: int,
    complexity: str,
    suggested: str = "",
    reasoning: str = "",
) -> _RouterStepDecision:
    return _RouterStepDecision(
        index=index,
        complexity=complexity,
        suggested_command=suggested,
        reasoning=reasoning,
    )


# ----- parse_inline_annotation -----

class TestInlineAnnotation:
    def test_no_tag_returns_unchanged(self) -> None:
        hint, text = parse_inline_annotation("open youtube")
        assert hint is None
        assert text == "open youtube"

    def test_browser_fast_tag(self) -> None:
        hint, text = parse_inline_annotation("[browser-fast] open youtube")
        assert hint is not None
        assert hint.complexity == RoutingComplexity.BROWSER_FAST
        assert hint.source == "manual"
        assert text == "open youtube"

    def test_vlm_alias_maps_to_desktop_vlm(self) -> None:
        # ``[vlm]`` is the user-friendly shorthand we promised in docs.
        hint, text = parse_inline_annotation("[vlm] post on instagram")
        assert hint is not None
        assert hint.complexity == RoutingComplexity.DESKTOP_VLM
        assert text == "post on instagram"

    def test_browser_alias(self) -> None:
        hint, _ = parse_inline_annotation("[browser] go to gmail")
        assert hint is not None
        assert hint.complexity == RoutingComplexity.BROWSER_FAST

    def test_case_insensitive_tag(self) -> None:
        hint, _ = parse_inline_annotation("[BROWSER-FAST] open YouTube")
        assert hint is not None
        assert hint.complexity == RoutingComplexity.BROWSER_FAST

    def test_unknown_tag_left_in_place(self) -> None:
        # ``Type [admin]`` is legitimate user content; the loader must
        # NOT swallow it as a routing tag.
        hint, text = parse_inline_annotation("Type [admin] in the box")
        assert hint is None
        assert text == "Type [admin] in the box"

    def test_underscore_alias(self) -> None:
        # ``[browser_fast]`` is a syntactically nicer form of
        # ``[browser-fast]`` — both map to the same complexity.
        hint, _ = parse_inline_annotation("[browser_fast] open x")
        assert hint is not None
        assert hint.complexity == RoutingComplexity.BROWSER_FAST


# ----- parse_mode -----

class TestParseMode:
    def test_default_is_auto(self) -> None:
        assert parse_mode("") == RoutingMode.AUTO
        assert parse_mode(None) == RoutingMode.AUTO

    def test_known_modes(self) -> None:
        assert parse_mode("auto") == RoutingMode.AUTO
        assert parse_mode("MANUAL") == RoutingMode.MANUAL
        assert parse_mode("Off") == RoutingMode.OFF

    def test_unknown_falls_back_to_auto(self) -> None:
        # Typo in .env shouldn't crash the agent.
        assert parse_mode("autoo") == RoutingMode.AUTO


# ----- apply_router(mode=off / manual) -----

class TestApplyRouterTrivialModes:
    def test_off_returns_all_none_without_calling_client(self) -> None:
        client = _StubClient()
        hints = apply_router(
            ["open youtube", "click first"],
            mode=RoutingMode.OFF,
            client=client,
            enable_browser_fast_path=True,
        )
        assert hints == [None, None]
        assert client.call_count == 0

    def test_manual_does_not_call_client(self) -> None:
        # Manual mode is implemented at the loader level. apply_router
        # should never hit the client even when mode=manual.
        client = _StubClient()
        hints = apply_router(
            ["[browser-fast] open youtube"],
            mode=RoutingMode.MANUAL,
            client=client,
            enable_browser_fast_path=True,
        )
        assert hints == [None]
        assert client.call_count == 0

    def test_empty_step_list_short_circuits(self) -> None:
        client = _StubClient()
        assert (
            apply_router(
                [], mode=RoutingMode.AUTO, client=client, enable_browser_fast_path=True
            )
            == []
        )
        assert client.call_count == 0


# ----- apply_router(mode=auto) -----

class TestApplyRouterAuto:
    def test_happy_path_aligned_response(self) -> None:
        response = _RouterResponseModel(
            steps=[
                _decision(0, "browser-fast", "BROWSER_GO [https://youtube.com]"),
                _decision(
                    1,
                    "browser-fast",
                    "BROWSER_FILL [input[name=search_query], baby]",
                ),
                _decision(2, "browser-vlm"),
                _decision(3, "desktop-vlm"),
            ]
        )
        client = _StubClient(response=response)

        hints = apply_router(
            [
                "open youtube",
                "search for baby",
                "click the first video",
                "open notepad",
            ],
            mode=RoutingMode.AUTO,
            client=client,
            enable_browser_fast_path=True,
        )

        assert client.call_count == 1
        assert len(hints) == 4
        assert hints[0] is not None
        assert hints[0].complexity == RoutingComplexity.BROWSER_FAST
        assert hints[0].suggested_command == "BROWSER_GO [https://youtube.com]"
        assert hints[0].source == "auto"
        assert hints[1].complexity == RoutingComplexity.BROWSER_FAST
        assert hints[1].suggested_command.startswith("BROWSER_FILL")
        assert hints[2].complexity == RoutingComplexity.BROWSER_VLM
        assert hints[2].suggested_command == ""
        assert hints[3].complexity == RoutingComplexity.DESKTOP_VLM

    def test_bridge_disabled_downgrades_browser_fast(self) -> None:
        # When Chrome isn't connected, BROWSER_* commands are not in
        # the planner's system prompt. A browser-fast hint pointing at
        # BROWSER_GO would get the planner to emit doomed commands.
        response = _RouterResponseModel(
            steps=[
                _decision(0, "browser-fast", "BROWSER_GO [https://x.com]"),
            ]
        )
        client = _StubClient(response=response)

        hints = apply_router(
            ["go to x.com"],
            mode=RoutingMode.AUTO,
            client=client,
            enable_browser_fast_path=False,
        )

        assert hints[0] is not None
        # browser-fast was downgraded to browser-vlm.
        assert hints[0].complexity == RoutingComplexity.BROWSER_VLM
        # The BROWSER_GO suggestion is dropped — the planner can't use it.
        assert hints[0].suggested_command == ""
        # The user prompt should have included the bridge-disabled note
        # so the model is aware in the first place.
        assert "NOT available" in client.last_user_prompt

    def test_browser_command_on_non_browser_fast_tag_is_stripped(self) -> None:
        # Defensive: if the model misclassifies a step as desktop-vlm
        # but still includes a BROWSER_* suggestion, the suggestion
        # would be doomed (the planner would emit BROWSER_GO on a step
        # tagged for the visual planner). Strip it.
        response = _RouterResponseModel(
            steps=[
                _decision(
                    0, "desktop-vlm", "BROWSER_GO [https://oops.com]"
                ),
            ]
        )
        client = _StubClient(response=response)

        hints = apply_router(
            ["open notepad"],
            mode=RoutingMode.AUTO,
            client=client,
            enable_browser_fast_path=True,
        )

        assert hints[0] is not None
        assert hints[0].complexity == RoutingComplexity.DESKTOP_VLM
        assert hints[0].suggested_command == ""

    def test_router_failure_returns_all_none(self) -> None:
        # Network / schema errors must not crash the run. The agent
        # falls through to no-routing behavior.
        client = _StubClient(
            raise_on_call=RouterUnavailable("simulated failure"),
        )
        hints = apply_router(
            ["open youtube", "click first"],
            mode=RoutingMode.AUTO,
            client=client,
            enable_browser_fast_path=True,
        )
        assert hints == [None, None]

    def test_router_unexpected_exception_swallowed(self) -> None:
        # Any exception (not just RouterUnavailable) is converted to
        # the same fallback so the run never aborts because the router
        # is flaky.
        client = _StubClient(raise_on_call=ValueError("kaboom"))
        hints = apply_router(
            ["x", "y"],
            mode=RoutingMode.AUTO,
            client=client,
            enable_browser_fast_path=True,
        )
        assert hints == [None, None]

    def test_length_mismatch_falls_through(self) -> None:
        # Model returned the wrong number of decisions — discard.
        response = _RouterResponseModel(
            steps=[_decision(0, "browser-fast")]  # only 1 of 2
        )
        client = _StubClient(response=response)
        hints = apply_router(
            ["a", "b"],
            mode=RoutingMode.AUTO,
            client=client,
            enable_browser_fast_path=True,
        )
        assert hints == [None, None]

    def test_index_misalignment_falls_through(self) -> None:
        # If the model permutes order, we don't try to fix it — too
        # easy to mis-match a hint to the wrong step.
        response = _RouterResponseModel(
            steps=[
                _decision(1, "browser-fast"),  # index 1 first
                _decision(0, "browser-fast"),
            ]
        )
        client = _StubClient(response=response)
        hints = apply_router(
            ["a", "b"],
            mode=RoutingMode.AUTO,
            client=client,
            enable_browser_fast_path=True,
        )
        assert hints == [None, None]

    def test_no_client_in_auto_mode_falls_through(self) -> None:
        # Defensive: if the run loop somehow calls apply_router(auto)
        # without a client, return None instead of crashing.
        hints = apply_router(
            ["open youtube"],
            mode=RoutingMode.AUTO,
            client=None,
            enable_browser_fast_path=True,
        )
        assert hints == [None]


# ----- attach_routing_hints (manual beats auto) -----

class TestAttachRoutingHints:
    def test_auto_hints_attach_when_no_manual(self) -> None:
        steps = [TaskStep(text="open youtube"), TaskStep(text="click first")]
        hints: list[RoutingHint | None] = [
            RoutingHint(
                complexity=RoutingComplexity.BROWSER_FAST,
                suggested_command="BROWSER_GO [https://youtube.com]",
            ),
            RoutingHint(complexity=RoutingComplexity.BROWSER_VLM),
        ]
        out = attach_routing_hints(steps, hints)
        assert out[0].routing_hint is not None
        assert (
            out[0].routing_hint.complexity == RoutingComplexity.BROWSER_FAST
        )
        assert out[1].routing_hint is not None

    def test_manual_hint_wins_over_auto(self) -> None:
        # User explicitly said [vlm], router said browser-fast: keep manual.
        manual = RoutingHint(
            complexity=RoutingComplexity.DESKTOP_VLM, source="manual"
        )
        steps = [TaskStep(text="x", routing_hint=manual)]
        auto = [
            RoutingHint(
                complexity=RoutingComplexity.BROWSER_FAST,
                suggested_command="BROWSER_GO [https://x.com]",
                source="auto",
            )
        ]
        out = attach_routing_hints(steps, auto)
        assert out[0].routing_hint is manual

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            attach_routing_hints([TaskStep(text="x")], [None, None])

    def test_none_hint_leaves_step_unannotated(self) -> None:
        steps = [TaskStep(text="x")]
        out = attach_routing_hints(steps, [None])
        assert out[0].routing_hint is None


# ----- end-to-end loader integration -----

class TestLoaderInlineAnnotations:
    def test_inline_tag_stripped_at_load(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.txt"
        path.write_text(
            textwrap.dedent(
                """\
                [browser-fast] open youtube
                [vlm] post on instagram
                click the file menu
                """
            )
        )
        steps = load_steps(path)
        assert len(steps) == 3
        assert steps[0].text == "open youtube"
        assert steps[0].routing_hint is not None
        assert (
            steps[0].routing_hint.complexity == RoutingComplexity.BROWSER_FAST
        )
        assert steps[0].routing_hint.source == "manual"

        assert steps[1].text == "post on instagram"
        assert (
            steps[1].routing_hint is not None
            and steps[1].routing_hint.complexity == RoutingComplexity.DESKTOP_VLM
        )

        assert steps[2].text == "click the file menu"
        assert steps[2].routing_hint is None  # no annotation

    def test_unknown_tag_left_intact(self, tmp_path: Path) -> None:
        # ``[admin]`` isn't a routing tag; it's user content. It must
        # NOT be stripped or treated as a hint.
        path = tmp_path / "tasks.txt"
        path.write_text("Type [admin] in the username box\n")
        steps = load_steps(path)
        assert steps[0].text == "Type [admin] in the username box"
        assert steps[0].routing_hint is None


# ----- prompt-rendering ergonomics -----

class TestRoutingHintRendering:
    def test_render_includes_complexity(self) -> None:
        hint = RoutingHint(complexity=RoutingComplexity.BROWSER_FAST)
        assert hint.render_for_prompt() == "complexity=browser-fast"

    def test_render_includes_suggested_command(self) -> None:
        hint = RoutingHint(
            complexity=RoutingComplexity.BROWSER_FAST,
            suggested_command="BROWSER_GO [https://youtube.com]",
        )
        rendered = hint.render_for_prompt()
        assert "complexity=browser-fast" in rendered
        assert "BROWSER_GO [https://youtube.com]" in rendered
