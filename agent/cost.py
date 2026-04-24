"""RPD (Requests-Per-Day) usage tracker and threshold guard.

This is intentionally minimal: we count every Gemini API call this run,
log a single WARNING at the 75% mark, and signal `should_halt()` once we
cross 95% so the agent can stop cleanly with the checkpoint intact.

The counter is process-local (per-run) — it does NOT persist across
restarts. A future PR could persist it to disk (`~/.agent_rpd_<date>.json`)
to track usage across multiple `python -m agent` invocations on the
same day.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class RpdGuard:
    """Track Gemini call count against the daily limit."""

    rpd_limit: int = 0  # 0 disables all guards.
    warn_threshold: float = 0.75
    halt_threshold: float = 0.95
    calls: int = 0
    _warned: bool = False

    def record(self) -> None:
        """Account for one outbound API call."""
        self.calls += 1
        if self.rpd_limit <= 0:
            return
        if not self._warned and self.calls >= self.warn_threshold * self.rpd_limit:
            self._warned = True
            log.warning(
                "RPD usage at %d / %d (%.0f%%) — approaching daily quota.",
                self.calls,
                self.rpd_limit,
                100.0 * self.calls / self.rpd_limit,
            )

    def should_halt(self) -> bool:
        """True when the cumulative call count crosses the halt threshold."""
        if self.rpd_limit <= 0:
            return False
        return self.calls >= self.halt_threshold * self.rpd_limit

    def halt_message(self) -> str:
        return (
            f"RPD halt threshold reached: {self.calls} / {self.rpd_limit} "
            f"({100.0 * self.calls / max(self.rpd_limit, 1):.0f}%). "
            "Checkpoint saved — resume tomorrow or raise RPD_LIMIT."
        )
