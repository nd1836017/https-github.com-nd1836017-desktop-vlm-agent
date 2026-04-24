"""Unit tests for the per-step artifact writer."""
from __future__ import annotations

import json

from PIL import Image

from agent.artifacts import ArtifactWriter


def _red_image() -> Image.Image:
    return Image.new("RGB", (16, 16), color=(255, 0, 0))


def test_disabled_writer_no_ops(tmp_path):
    w = ArtifactWriter.create(enabled=False, base_dir=tmp_path)
    w.save_before(1, _red_image())
    w.save_after(1, _red_image())
    w.save_plan(1, "raw", "CLICK [100,200]")
    w.save_verdict(1, True, "ok")
    w.append_summary(1, "step", True, "ok", action_text="CLICK [100,200]")
    # Nothing should have been written.
    assert not any(tmp_path.iterdir())


def test_enabled_writer_creates_run_dir_with_timestamp(tmp_path):
    w = ArtifactWriter.create(enabled=True, base_dir=tmp_path)
    assert w.run_dir is not None
    assert w.run_dir.parent == tmp_path
    assert w.run_dir.exists()


def test_enabled_writer_writes_all_artifacts(tmp_path):
    w = ArtifactWriter.create(enabled=True, base_dir=tmp_path)
    w.save_before(3, _red_image())
    w.save_after(3, _red_image())
    w.save_plan(3, "RAW_RESPONSE", "CLICK [100,200]")
    w.save_verdict(3, True, "address bar clicked")
    w.append_summary(3, "click address bar", True, "ok", action_text="CLICK [100,200]")

    run_dir = w.run_dir
    assert (run_dir / "step_003_before.png").exists()
    assert (run_dir / "step_003_after.png").exists()
    plan = (run_dir / "step_003_plan.txt").read_text(encoding="utf-8")
    assert "RAW_RESPONSE" in plan
    assert "CLICK [100,200]" in plan
    verdict = (run_dir / "step_003_verdict.txt").read_text(encoding="utf-8")
    assert "VERDICT: PASS" in verdict
    assert "address bar clicked" in verdict

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 1
    assert summary[0]["step_idx"] == 3
    assert summary[0]["passed"] is True
    assert summary[0]["action"] == "CLICK [100,200]"


def test_summary_accumulates_across_steps(tmp_path):
    w = ArtifactWriter.create(enabled=True, base_dir=tmp_path)
    w.append_summary(1, "step-1", True, "ok", action_text="PRESS [ctrl+l]")
    w.append_summary(2, "step-2", False, "blocked", action_text="TYPE [<REDACTED>]")

    summary = json.loads((w.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert [entry["step_idx"] for entry in summary] == [1, 2]
    assert [entry["passed"] for entry in summary] == [True, False]
    assert summary[1]["reason"] == "blocked"


def test_append_summary_uses_last_saved_action_when_not_provided(tmp_path):
    """`run()` doesn't pass action_text — the writer should recall it
    from the most recent `save_plan(step_idx, ...)` call. This verifies
    the fix for PR #8 medium finding #1: summary.json rows used to be
    hardcoded to the "<see step_NNN_plan.txt>" placeholder.
    """
    w = ArtifactWriter.create(enabled=True, base_dir=tmp_path)
    w.save_plan(1, "raw-1", "PRESS [ctrl+l]")
    w.save_plan(2, "raw-2", "TYPE <REDACTED, 8 chars>")
    # NOTE: no action_text kwarg — simulates run()'s call site.
    w.append_summary(1, "step-1", True, "ok")
    w.append_summary(2, "step-2", False, "blocked")

    summary = json.loads((w.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary[0]["action"] == "PRESS [ctrl+l]"
    assert summary[1]["action"] == "TYPE <REDACTED, 8 chars>"


def test_append_summary_falls_back_to_marker_when_no_plan_saved(tmp_path):
    """Defensive: a verdict with no matching save_plan call (e.g. parse
    failure before any action executed) should still record a row, with
    a clear marker instead of silently logging the wrong action.
    """
    w = ArtifactWriter.create(enabled=True, base_dir=tmp_path)
    w.append_summary(5, "no-plan-step", False, "parse failure")
    summary = json.loads((w.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary[0]["action"] == "<no-action-recorded>"


def test_save_plan_overwrites_latest_action_within_step(tmp_path):
    """If a step replans, the latest action wins for the summary row."""
    w = ArtifactWriter.create(enabled=True, base_dir=tmp_path)
    w.save_plan(4, "raw-a", "CLICK [500,500]")   # first attempt
    w.save_plan(4, "raw-b", "CLICK_TEXT [Sign in]")  # replan
    w.append_summary(4, "login step", True, "ok")
    summary = json.loads((w.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary[0]["action"] == "CLICK_TEXT [Sign in]"


def test_unwritable_base_dir_disables_gracefully(tmp_path):
    # Point base_dir at an existing FILE so mkdir fails — writer should
    # degrade to disabled silently.
    bad = tmp_path / "not_a_dir"
    bad.write_text("just a file", encoding="utf-8")
    w = ArtifactWriter.create(enabled=True, base_dir=bad / "runs")
    # mkdir should have failed → enabled flipped to False.
    assert w.enabled is False
