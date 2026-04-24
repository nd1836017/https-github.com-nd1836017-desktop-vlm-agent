"""Checkpoint state: tracks which step we last completed so a long run can resume.

The state file is a tiny JSON document. Writes are atomic (write-then-rename)
so a crash mid-write cannot corrupt the checkpoint.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AgentState:
    version: int
    tasks_file: str
    total_steps: int
    last_completed_step: int  # 1-indexed; 0 means "nothing completed yet"

    @classmethod
    def initial(cls, tasks_file: Path, total_steps: int) -> AgentState:
        return cls(
            version=STATE_SCHEMA_VERSION,
            tasks_file=str(tasks_file),
            total_steps=total_steps,
            last_completed_step=0,
        )

    def advance(self) -> AgentState:
        return AgentState(
            version=self.version,
            tasks_file=self.tasks_file,
            total_steps=self.total_steps,
            last_completed_step=self.last_completed_step + 1,
        )


def load_state(path: Path) -> AgentState | None:
    """Return the checkpoint at `path`, or None if it doesn't exist / is unreadable."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentState(
            version=int(data["version"]),
            tasks_file=str(data["tasks_file"]),
            total_steps=int(data["total_steps"]),
            last_completed_step=int(data["last_completed_step"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        log.warning("Ignoring unreadable state file %s: %s", path, exc)
        return None


def save_state(path: Path, state: AgentState) -> None:
    """Atomically persist `state` to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file in the same directory, then rename for atomicity.
    fd, tmp = tempfile.mkstemp(prefix=".agent_state.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        # Best-effort cleanup of the temp file on failure.
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def reset_state(path: Path) -> None:
    """Delete the checkpoint file if it exists."""
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
