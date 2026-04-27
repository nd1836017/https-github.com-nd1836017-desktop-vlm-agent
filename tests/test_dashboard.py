"""Tests for the run replay dashboard.

Covers:
* scan_run / list_runs read existing artifacts correctly.
* Plan + verdict file parsers handle malformed input without crashing.
* FastAPI endpoints work end-to-end via TestClient.
* Path-traversal attempts on /run/<id>/img/<name> are rejected.

These tests build artifact directories on disk in a tmp_path and
exercise the read-only dashboard against them. No Gemini calls or
agent runs are needed.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from agent.dashboard import (
    StepArtifact,
    _parse_plan_file,
    _parse_verdict_file,
    create_app,
    list_runs,
    load_summary,
    scan_run,
)

pytestmark = pytest.mark.filterwarnings(
    # FastAPI / Starlette emit DeprecationWarnings on Python 3.13+ for
    # event-loop policy access; not our concern for these tests.
    "ignore::DeprecationWarning"
)


def _png_bytes(color: tuple[int, int, int] = (200, 50, 50)) -> bytes:
    img = Image.new("RGB", (10, 10), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_run(
    base: Path,
    run_id: str,
    *,
    steps: list[dict] | None = None,
) -> Path:
    """Build a fake run dir with the given steps.

    Each step dict can have keys:
      idx, before, after, action, raw, verdict, reason
    """
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for step in steps or []:
        idx = step["idx"]
        if step.get("before", True):
            (run_dir / f"step_{idx:03d}_before.png").write_bytes(_png_bytes())
        if step.get("after", True):
            (run_dir / f"step_{idx:03d}_after.png").write_bytes(
                _png_bytes((50, 200, 50))
            )
        action = step.get("action")
        if action is not None:
            raw = step.get("raw", "")
            (run_dir / f"step_{idx:03d}_plan.txt").write_text(
                f"# action_text\n{action}\n\n# raw VLM response\n{raw}\n",
                encoding="utf-8",
            )
        verdict = step.get("verdict")
        if verdict is not None:
            (run_dir / f"step_{idx:03d}_verdict.txt").write_text(
                f"VERDICT: {verdict}\nREASON: {step.get('reason', '')}\n",
                encoding="utf-8",
            )
    return run_dir


# ----- parsers -----


class TestParsers:
    def test_plan_file_well_formed(self) -> None:
        text = (
            "# action_text\nCLICK [500,500]\n\n"
            "# raw VLM response\nCLICK [500,500]\n"
        )
        action, raw = _parse_plan_file(text)
        assert action == "CLICK [500,500]"
        assert raw == "CLICK [500,500]"

    def test_plan_file_missing_markers_falls_back_to_raw(self) -> None:
        action, raw = _parse_plan_file("just some text")
        assert action == ""
        assert raw == "just some text"

    def test_verdict_file_pass(self) -> None:
        verdict, reason = _parse_verdict_file(
            "VERDICT: PASS\nREASON: looks right\n"
        )
        assert verdict == "PASS"
        assert reason == "looks right"

    def test_verdict_file_lowercase_normalised(self) -> None:
        verdict, _ = _parse_verdict_file("VERDICT: pass\n")
        assert verdict == "PASS"

    def test_verdict_file_empty(self) -> None:
        verdict, reason = _parse_verdict_file("")
        assert verdict == ""
        assert reason == ""


# ----- scan_run / list_runs -----


class TestScanRun:
    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "empty"
        run_dir.mkdir()
        assert scan_run(run_dir) == []

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert scan_run(tmp_path / "doesnt-exist") == []

    def test_full_run(self, tmp_path: Path) -> None:
        run_dir = _make_run(
            tmp_path,
            "20260101-120000Z",
            steps=[
                {
                    "idx": 1,
                    "action": "open chrome",
                    "verdict": "PASS",
                    "reason": "chrome window visible",
                },
                {
                    "idx": 2,
                    "action": "click foo",
                    "verdict": "FAIL",
                    "reason": "nothing happened",
                },
            ],
        )
        steps = scan_run(run_dir)
        assert len(steps) == 2
        assert isinstance(steps[0], StepArtifact)
        assert steps[0].index == 1
        assert steps[0].verdict == "PASS"
        assert steps[0].plan_action == "open chrome"
        assert steps[0].has_before and steps[0].has_after
        assert steps[1].index == 2
        assert steps[1].verdict == "FAIL"

    def test_list_runs_sorted_newest_first(self, tmp_path: Path) -> None:
        # The writer's UTC timestamp format sorts lexicographically as
        # expected — newer dates sort higher. We simulate by name.
        _make_run(
            tmp_path,
            "20260101-100000Z",
            steps=[{"idx": 1, "action": "a", "verdict": "PASS"}],
        )
        _make_run(
            tmp_path,
            "20260102-100000Z",
            steps=[{"idx": 1, "action": "b", "verdict": "FAIL"}],
        )
        runs = list_runs(tmp_path)
        assert [r.run_id for r in runs] == [
            "20260102-100000Z",
            "20260101-100000Z",
        ]
        assert runs[0].n_fail == 1
        assert runs[1].n_pass == 1

    def test_list_runs_skips_non_run_dirs(self, tmp_path: Path) -> None:
        # A subdirectory without any step_NNN_* files should be ignored.
        (tmp_path / "not-a-run").mkdir()
        _make_run(
            tmp_path,
            "20260101-100000Z",
            steps=[{"idx": 1, "action": "a", "verdict": "PASS"}],
        )
        runs = list_runs(tmp_path)
        assert [r.run_id for r in runs] == ["20260101-100000Z"]

    def test_list_runs_missing_base_returns_empty(self, tmp_path: Path) -> None:
        assert list_runs(tmp_path / "does-not-exist") == []

    def test_load_summary_present(self, tmp_path: Path) -> None:
        run_dir = _make_run(
            tmp_path,
            "20260101-100000Z",
            steps=[{"idx": 1, "action": "a", "verdict": "PASS"}],
        )
        (run_dir / "summary.json").write_text(
            json.dumps([{"step": 1, "passed": True}])
        )
        loaded = load_summary(run_dir)
        assert loaded == [{"step": 1, "passed": True}]

    def test_load_summary_missing(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path, "rid", steps=[])
        assert load_summary(run_dir) is None

    def test_load_summary_corrupt(self, tmp_path: Path) -> None:
        run_dir = _make_run(tmp_path, "rid", steps=[])
        (run_dir / "summary.json").write_text("not json")
        assert load_summary(run_dir) is None


# ----- FastAPI endpoints -----


class TestApp:
    def _client(self, tmp_path: Path):
        from fastapi.testclient import TestClient

        app = create_app(tmp_path)
        return TestClient(app)

    def test_healthz(self, tmp_path: Path) -> None:
        with self._client(tmp_path) as c:
            r = c.get("/healthz")
            assert r.status_code == 200
            assert r.text == "ok"

    def test_index_empty(self, tmp_path: Path) -> None:
        with self._client(tmp_path) as c:
            r = c.get("/")
            assert r.status_code == 200
            assert "No runs found" in r.text

    def test_index_with_runs(self, tmp_path: Path) -> None:
        _make_run(
            tmp_path,
            "20260101-100000Z",
            steps=[
                {"idx": 1, "action": "open", "verdict": "PASS"},
                {"idx": 2, "action": "click", "verdict": "FAIL"},
            ],
        )
        with self._client(tmp_path) as c:
            r = c.get("/")
            assert r.status_code == 200
            assert "20260101-100000Z" in r.text
            # The badge shows pass + fail counts.
            assert ">1<" in r.text  # n_pass=1, n_fail=1

    def test_run_view_renders_steps(self, tmp_path: Path) -> None:
        _make_run(
            tmp_path,
            "rid",
            steps=[
                {
                    "idx": 1,
                    "action": "open chrome",
                    "verdict": "PASS",
                    "reason": "visible",
                },
            ],
        )
        with self._client(tmp_path) as c:
            r = c.get("/run/rid")
            assert r.status_code == 200
            assert "open chrome" in r.text
            assert "PASS" in r.text
            assert "step_001_before.png" in r.text

    def test_run_view_unknown_run_404s(self, tmp_path: Path) -> None:
        with self._client(tmp_path) as c:
            r = c.get("/run/no-such-run")
            assert r.status_code == 404

    def test_run_api_json(self, tmp_path: Path) -> None:
        _make_run(
            tmp_path,
            "rid",
            steps=[{"idx": 1, "action": "x", "verdict": "PASS"}],
        )
        with self._client(tmp_path) as c:
            r = c.get("/run/rid/api")
            assert r.status_code == 200
            data = r.json()
            assert data["run_id"] == "rid"
            assert len(data["steps"]) == 1
            assert data["steps"][0]["verdict"] == "PASS"

    def test_image_passthrough(self, tmp_path: Path) -> None:
        _make_run(
            tmp_path,
            "rid",
            steps=[{"idx": 1, "action": "x", "verdict": "PASS"}],
        )
        with self._client(tmp_path) as c:
            r = c.get("/run/rid/img/step_001_before.png")
            assert r.status_code == 200
            assert r.headers["content-type"] == "image/png"
            assert r.content.startswith(b"\x89PNG")

    def test_image_unknown_filename_404s(self, tmp_path: Path) -> None:
        _make_run(
            tmp_path,
            "rid",
            steps=[{"idx": 1, "action": "x", "verdict": "PASS"}],
        )
        with self._client(tmp_path) as c:
            r = c.get("/run/rid/img/random.png")
            assert r.status_code == 404

    def test_path_traversal_rejected_in_image_name(
        self, tmp_path: Path
    ) -> None:
        # The /img/<name> route's regex restricts to step_NNN_*.png — any
        # other filename, including ones with .. or absolute paths, must
        # 404 before we ever touch disk. We test the most direct
        # bypass attempt: a file that doesn't match the regex.
        _make_run(
            tmp_path,
            "rid",
            steps=[{"idx": 1, "action": "x", "verdict": "PASS"}],
        )
        # Stash a sibling file outside the run dir to make sure we
        # never serve it.
        (tmp_path / "secret.png").write_bytes(_png_bytes())
        with self._client(tmp_path) as c:
            # Filenames not matching ``step_NNN_*.png`` are rejected.
            r = c.get("/run/rid/img/secret.png")
            assert r.status_code == 404
