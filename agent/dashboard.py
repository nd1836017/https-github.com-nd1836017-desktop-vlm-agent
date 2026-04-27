"""Run replay dashboard — read-only HTTP UI over ``runs/<id>/`` artifacts.

Launched via ``python -m agent --serve-dashboard`` (no agent run; just the
viewer). Reads existing artifact directories under
``RUN_ARTIFACTS_DIR`` (default ``runs/``); does NOT write or modify
anything. Safe to point at a long-lived runs/ tree.

Endpoints
---------

GET /                       — index of all runs (newest first)
GET /run/<run_id>           — timeline view of one run
GET /run/<run_id>/api       — JSON dump of the same data (for tooling)
GET /run/<run_id>/img/<name>  — raw PNG passthrough
GET /healthz                — liveness check (returns "ok")

The HTML is plain server-rendered (no JavaScript framework) so the
dashboard works without an internet connection. Styling is minimal; the
goal is "looks like a debugger, not a product page."

Run artifact layout (set by ``agent/artifacts.py``)::

    runs/<run_id>/
      step_001_before.png
      step_001_after.png
      step_001_plan.txt        # has '# action_text\\n<text>\\n\\n# raw VLM response\\n<raw>\\n'
      step_001_verdict.txt     # 'VERDICT: PASS|FAIL\\nREASON: <text>\\n'
      summary.json             # rolling per-step summary
      cost.json (optional)     # if cost telemetry exists
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type-checking
    pass

log = logging.getLogger(__name__)


# Artifact-name regexes shared with ``ArtifactWriter``. Centralising
# these here avoids the dashboard silently misparsing if the writer's
# layout ever changes — at least the failure mode is "dashboard shows
# nothing" instead of "dashboard shows the wrong thing".
_BEFORE_RE = re.compile(r"^step_(\d{3,})_before\.png$")
_AFTER_RE = re.compile(r"^step_(\d{3,})_after\.png$")
_PLAN_RE = re.compile(r"^step_(\d{3,})_plan\.txt$")
_VERDICT_RE = re.compile(r"^step_(\d{3,})_verdict\.txt$")


@dataclass(frozen=True)
class StepArtifact:
    """One step's view for the dashboard."""

    index: int
    has_before: bool
    has_after: bool
    plan_action: str
    plan_raw: str
    verdict: str  # "PASS" / "FAIL" / "" (unknown)
    verdict_reason: str

    @property
    def status_label(self) -> str:
        return self.verdict or "—"

    @property
    def status_class(self) -> str:
        return {"PASS": "pass", "FAIL": "fail"}.get(self.verdict, "unknown")


@dataclass(frozen=True)
class RunSummary:
    """One row in the index page."""

    run_id: str
    path: Path
    n_steps: int
    n_pass: int
    n_fail: int
    has_cost: bool

    @property
    def status_class(self) -> str:
        if self.n_fail > 0:
            return "fail"
        if self.n_pass == self.n_steps and self.n_steps > 0:
            return "pass"
        return "unknown"


def _parse_plan_file(text: str) -> tuple[str, str]:
    """Pull (action_text, raw_response) out of step_<n>_plan.txt.

    The writer format is::

        # action_text
        <action>
        <blank line>
        # raw VLM response
        <raw>

    A best-effort parse: missing markers fall through to empty strings
    rather than crashing.
    """
    action = ""
    raw = ""
    if "# action_text" in text and "# raw VLM response" in text:
        # Split on the second header — everything between the two
        # headers (minus the blank separator) is the action text.
        head, _, rest = text.partition("# action_text")
        body, _, raw_section = rest.partition("# raw VLM response")
        action = body.strip()
        raw = raw_section.strip()
    else:
        # Older / partial files: just dump everything as raw.
        raw = text.strip()
    return action, raw


def _parse_verdict_file(text: str) -> tuple[str, str]:
    """Pull (verdict, reason) out of step_<n>_verdict.txt."""
    verdict = ""
    reason = ""
    for line in text.splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return verdict, reason


def list_runs(base_dir: Path) -> list[RunSummary]:
    """Discover every run directory under ``base_dir``.

    A run is any subdirectory containing at least one ``step_NNN_*``
    artifact. Sorted with newest first by directory name (the writer's
    UTC timestamp format sorts lexicographically as expected).
    """
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    runs: list[RunSummary] = []
    for child in sorted(base_dir.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        steps = scan_run(child)
        if not steps:
            # Probably not a run dir — skip silently. Could be the
            # files-workspace runs/ subtree if the user repurposed the
            # path; we don't want to error out on those.
            continue
        n_pass = sum(1 for s in steps if s.verdict == "PASS")
        n_fail = sum(1 for s in steps if s.verdict == "FAIL")
        runs.append(
            RunSummary(
                run_id=child.name,
                path=child,
                n_steps=len(steps),
                n_pass=n_pass,
                n_fail=n_fail,
                has_cost=(child / "cost.json").exists(),
            )
        )
    return runs


def scan_run(run_dir: Path) -> list[StepArtifact]:
    """Read every step artifact in ``run_dir`` and group by step index."""
    if not run_dir.exists() or not run_dir.is_dir():
        return []

    by_index: dict[int, dict[str, str | bool]] = {}

    def _slot(idx: int) -> dict:
        return by_index.setdefault(
            idx,
            {
                "has_before": False,
                "has_after": False,
                "plan_action": "",
                "plan_raw": "",
                "verdict": "",
                "verdict_reason": "",
            },
        )

    for entry in sorted(run_dir.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        if (m := _BEFORE_RE.match(name)) is not None:
            _slot(int(m.group(1)))["has_before"] = True
        elif (m := _AFTER_RE.match(name)) is not None:
            _slot(int(m.group(1)))["has_after"] = True
        elif (m := _PLAN_RE.match(name)) is not None:
            try:
                txt = entry.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            action, raw = _parse_plan_file(txt)
            slot = _slot(int(m.group(1)))
            slot["plan_action"] = action
            slot["plan_raw"] = raw
        elif (m := _VERDICT_RE.match(name)) is not None:
            try:
                txt = entry.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            verdict, reason = _parse_verdict_file(txt)
            slot = _slot(int(m.group(1)))
            slot["verdict"] = verdict
            slot["verdict_reason"] = reason

    out: list[StepArtifact] = []
    for idx in sorted(by_index.keys()):
        slot = by_index[idx]
        out.append(
            StepArtifact(
                index=idx,
                has_before=bool(slot["has_before"]),
                has_after=bool(slot["has_after"]),
                plan_action=str(slot["plan_action"]),
                plan_raw=str(slot["plan_raw"]),
                verdict=str(slot["verdict"]),
                verdict_reason=str(slot["verdict_reason"]),
            )
        )
    return out


def load_summary(run_dir: Path) -> list[dict] | None:
    """Read ``summary.json`` if present, else None."""
    p = run_dir / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to read %s: %s", p, exc)
        return None


def load_cost(run_dir: Path) -> dict | None:
    """Read ``cost.json`` if present, else None."""
    p = run_dir / "cost.json"
    if not p.exists():
        return None
    try:
        loaded = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to read %s: %s", p, exc)
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


# -------------------------------------------------------------------- HTML rendering


_BASE_STYLE = """
<style>
  :root {
    color-scheme: light dark;
    --pass: #1a7f37;
    --fail: #cf222e;
    --unknown: #6e7781;
    --border: #d0d7de;
    --muted: #57606a;
    --bg-card: #f6f8fa;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    margin: 0;
    padding: 24px;
    line-height: 1.4;
    max-width: 1400px;
    margin: 0 auto;
  }
  h1, h2, h3 {
    margin-top: 0;
  }
  a { color: inherit; }
  table { border-collapse: collapse; width: 100%; }
  th, td {
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }
  th { background: var(--bg-card); }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .badge.pass { background: var(--pass); color: white; }
  .badge.fail { background: var(--fail); color: white; }
  .badge.unknown { background: var(--unknown); color: white; }
  .step {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 16px;
    background: var(--bg-card);
  }
  .step.fail { border-left: 4px solid var(--fail); }
  .step.pass { border-left: 4px solid var(--pass); }
  .step.unknown { border-left: 4px solid var(--unknown); }
  .step-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .step-imgs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 12px 0;
  }
  .step-imgs figure { margin: 0; }
  .step-imgs img {
    width: 100%;
    height: auto;
    border: 1px solid var(--border);
    border-radius: 4px;
    cursor: zoom-in;
  }
  .step-imgs figcaption {
    color: var(--muted);
    font-size: 12px;
    margin-top: 4px;
  }
  .step-detail {
    display: grid;
    grid-template-columns: 100px 1fr;
    gap: 4px 12px;
    font-size: 14px;
    margin-top: 8px;
  }
  .step-detail dt { color: var(--muted); font-weight: 600; }
  .step-detail dd { margin: 0; }
  pre {
    background: white;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px;
    overflow-x: auto;
    font-size: 12px;
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .nav { margin-bottom: 16px; color: var(--muted); }
  .empty {
    color: var(--muted);
    padding: 40px;
    text-align: center;
    border: 1px dashed var(--border);
    border-radius: 6px;
  }
  .search {
    margin-bottom: 16px;
    width: 100%;
    padding: 8px;
    font-size: 14px;
    border: 1px solid var(--border);
    border-radius: 4px;
  }
</style>
<script>
  // Click a screenshot to toggle its size between fit and full-resolution
  // ("fullscreen" is the wrong word — we just remove the width
  // constraint so users can scroll the image in its native size).
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.step-imgs img').forEach(img => {
      img.addEventListener('click', () => {
        if (img.style.width === '100%' || !img.style.width) {
          img.style.width = 'auto';
          img.style.maxWidth = 'none';
          img.style.cursor = 'zoom-out';
        } else {
          img.style.width = '100%';
          img.style.maxWidth = '100%';
          img.style.cursor = 'zoom-in';
        }
      });
    });

    // Live filter on the index page.
    const search = document.querySelector('.search');
    if (search) {
      search.addEventListener('input', () => {
        const q = search.value.toLowerCase();
        document.querySelectorAll('table tbody tr').forEach(tr => {
          const txt = tr.textContent.toLowerCase();
          tr.style.display = txt.includes(q) ? '' : 'none';
        });
      });
    }
  });
</script>
"""


def render_index(runs: list[RunSummary]) -> str:
    """HTML for ``GET /``."""
    if not runs:
        body = (
            '<div class="empty">No runs found. Set <code>SAVE_RUN_ARTIFACTS=true</code> '
            "and run the agent to populate this view.</div>"
        )
    else:
        rows = []
        for r in runs:
            rows.append(
                f"<tr>"
                f"<td><a href='/run/{escape(r.run_id)}'>{escape(r.run_id)}</a></td>"
                f"<td>{r.n_steps}</td>"
                f"<td><span class='badge pass'>{r.n_pass}</span></td>"
                f"<td><span class='badge fail'>{r.n_fail}</span></td>"
                f"<td>{'yes' if r.has_cost else '—'}</td>"
                f"<td><span class='badge {r.status_class}'>{r.status_class}</span></td>"
                f"</tr>"
            )
        rows_html = "\n".join(rows)
        body = (
            "<input class='search' placeholder='Filter runs (e.g. fail, today, step text)' />"
            "<table>"
            "<thead><tr>"
            "<th>Run ID</th><th>Steps</th><th>Pass</th><th>Fail</th>"
            "<th>Cost</th><th>Status</th>"
            "</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
        )
    return (
        "<!doctype html><html><head><title>Run replay dashboard</title>"
        + _BASE_STYLE
        + "</head><body>"
        "<h1>Run replay dashboard</h1>"
        f"<p class='nav'>{len(runs)} run(s) — newest first</p>"
        + body
        + "</body></html>"
    )


def render_run(run_id: str, run: RunSummary, steps: list[StepArtifact]) -> str:
    """HTML for ``GET /run/<id>``."""
    if not steps:
        body = (
            "<div class='empty'>No step artifacts found in this run "
            "directory.</div>"
        )
    else:
        cards = []
        for step in steps:
            before_url = (
                f"/run/{escape(run_id)}/img/step_{step.index:03d}_before.png"
                if step.has_before
                else None
            )
            after_url = (
                f"/run/{escape(run_id)}/img/step_{step.index:03d}_after.png"
                if step.has_after
                else None
            )
            imgs_html = ""
            if before_url or after_url:
                imgs_html = (
                    "<div class='step-imgs'>"
                    + (
                        f"<figure><img src='{before_url}' alt='before'>"
                        "<figcaption>BEFORE (planner saw this)</figcaption></figure>"
                        if before_url
                        else "<div></div>"
                    )
                    + (
                        f"<figure><img src='{after_url}' alt='after'>"
                        "<figcaption>AFTER (verifier saw this)</figcaption></figure>"
                        if after_url
                        else "<div></div>"
                    )
                    + "</div>"
                )
            cards.append(
                f"<div class='step {step.status_class}'>"
                "<div class='step-header'>"
                f"<h3>Step {step.index}</h3>"
                f"<span class='badge {step.status_class}'>{escape(step.status_label)}</span>"
                "</div>"
                "<dl class='step-detail'>"
                f"<dt>Action</dt><dd><pre>{escape(step.plan_action) or '—'}</pre></dd>"
                f"<dt>Reason</dt><dd>{escape(step.verdict_reason) or '—'}</dd>"
                "</dl>"
                + imgs_html
                + (
                    "<details><summary>Raw planner response</summary>"
                    f"<pre>{escape(step.plan_raw) or '(empty)'}</pre>"
                    "</details>"
                    if step.plan_raw
                    else ""
                )
                + "</div>"
            )
        body = "\n".join(cards)
    return (
        "<!doctype html><html><head>"
        f"<title>Run {escape(run_id)}</title>"
        + _BASE_STYLE
        + "</head><body>"
        "<p class='nav'><a href='/'>&larr; All runs</a></p>"
        f"<h1>Run {escape(run_id)}</h1>"
        f"<p>{run.n_steps} step(s) — "
        f"<span class='badge pass'>{run.n_pass} pass</span> "
        f"<span class='badge fail'>{run.n_fail} fail</span></p>"
        + body
        + "</body></html>"
    )


# -------------------------------------------------------------------- FastAPI glue

def create_app(base_dir: Path):  # type: ignore[no-untyped-def]
    """Build the FastAPI app rooted at ``base_dir``.

    Imported lazily so a `pip install` without FastAPI doesn't break
    the rest of the agent (the dashboard is opt-in via a CLI flag).
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import (
            FileResponse,
            HTMLResponse,
            JSONResponse,
            PlainTextResponse,
        )
    except ImportError as exc:  # pragma: no cover - guard, exercised on missing dep
        raise RuntimeError(
            "FastAPI is required for --serve-dashboard. "
            "Install it with: pip install fastapi uvicorn"
        ) from exc

    app = FastAPI(title="VLM agent run replay")
    base_dir = base_dir.resolve()

    def _resolve_run(run_id: str) -> Path:
        # Reject any path traversal attempt — we only accept run IDs that
        # resolve to a direct child of base_dir. base_dir is trusted
        # (set by the operator); user-supplied path components must be
        # rejected if they escape it.
        candidate = (base_dir / run_id).resolve()
        try:
            candidate.relative_to(base_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid run id") from exc
        if not candidate.is_dir():
            raise HTTPException(status_code=404, detail="run not found")
        return candidate

    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz() -> str:
        return "ok"

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        runs = list_runs(base_dir)
        return render_index(runs)

    @app.get("/run/{run_id}", response_class=HTMLResponse)
    def run_view(run_id: str) -> str:
        run_dir = _resolve_run(run_id)
        steps = scan_run(run_dir)
        n_pass = sum(1 for s in steps if s.verdict == "PASS")
        n_fail = sum(1 for s in steps if s.verdict == "FAIL")
        run = RunSummary(
            run_id=run_id,
            path=run_dir,
            n_steps=len(steps),
            n_pass=n_pass,
            n_fail=n_fail,
            has_cost=(run_dir / "cost.json").exists(),
        )
        return render_run(run_id, run, steps)

    @app.get("/run/{run_id}/api")
    def run_api(run_id: str) -> JSONResponse:
        run_dir = _resolve_run(run_id)
        steps = scan_run(run_dir)
        return JSONResponse(
            {
                "run_id": run_id,
                "summary": load_summary(run_dir),
                "cost": load_cost(run_dir),
                "steps": [
                    {
                        "index": s.index,
                        "verdict": s.verdict,
                        "verdict_reason": s.verdict_reason,
                        "action": s.plan_action,
                        "has_before": s.has_before,
                        "has_after": s.has_after,
                    }
                    for s in steps
                ],
            }
        )

    @app.get("/run/{run_id}/img/{name}")
    def run_image(run_id: str, name: str):  # type: ignore[no-untyped-def]
        run_dir = _resolve_run(run_id)
        # Restrict to PNGs we actually generate. Anything else is a 404.
        if not (
            _BEFORE_RE.match(name) or _AFTER_RE.match(name)
        ) or not name.endswith(".png"):
            raise HTTPException(status_code=404, detail="not found")
        # Resolve and validate the image path is inside run_dir — the
        # regex already restricts the format, but we double-check to
        # defend against `..\` or absolute-path components on Windows.
        target = (run_dir / name).resolve()
        try:
            target.relative_to(run_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid name") from exc
        if not target.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(target, media_type="image/png")

    return app


def serve(
    base_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:  # pragma: no cover - runs an HTTP server
    """Start the dashboard. Blocks until interrupted (Ctrl-C).

    Lazy-imports uvicorn so callers that don't actually launch the
    dashboard don't need it installed. Defaults to localhost only —
    the dashboard reads run artifacts which may contain screenshots of
    the user's screen, so we do not expose 0.0.0.0 by default.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required for --serve-dashboard. "
            "Install it with: pip install fastapi uvicorn"
        ) from exc

    app = create_app(base_dir)
    log.info(
        "Starting run replay dashboard at http://%s:%d (artifacts dir: %s)",
        host,
        port,
        base_dir,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")
