"""Tests for the JSON-schema path of plan_action / verify."""
from __future__ import annotations

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
from agent.vlm import PlanResponseModel, plan_response_to_command


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
