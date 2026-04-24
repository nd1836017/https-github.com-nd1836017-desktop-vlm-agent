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
    w.append_summary(1, "step", "CLICK [100,200]", True, "ok")
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
    w.append_summary(3, "click address bar", "CLICK [100,200]", True, "ok")

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


def test_summary_accumulates_across_steps(tmp_path):
    w = ArtifactWriter.create(enabled=True, base_dir=tmp_path)
    w.append_summary(1, "step-1", "PRESS [ctrl+l]", True, "ok")
    w.append_summary(2, "step-2", "TYPE [<REDACTED>]", False, "blocked")

    summary = json.loads((w.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert [entry["step_idx"] for entry in summary] == [1, 2]
    assert [entry["passed"] for entry in summary] == [True, False]
    assert summary[1]["reason"] == "blocked"


def test_unwritable_base_dir_disables_gracefully(tmp_path):
    # Point base_dir at an existing FILE so mkdir fails — writer should
    # degrade to disabled silently.
    bad = tmp_path / "not_a_dir"
    bad.write_text("just a file", encoding="utf-8")
    w = ArtifactWriter.create(enabled=True, base_dir=bad / "runs")
    # mkdir should have failed → enabled flipped to False.
    assert w.enabled is False
