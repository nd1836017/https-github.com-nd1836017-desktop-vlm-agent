"""Unit tests for the RPD warning / halt guard."""
from __future__ import annotations

import logging

from agent.cost import RpdGuard


def test_rpd_disabled_never_halts():
    g = RpdGuard(rpd_limit=0)
    for _ in range(1_000):
        g.record()
    assert not g.should_halt()


def test_rpd_warn_fires_once(caplog):
    g = RpdGuard(rpd_limit=100, warn_threshold=0.75, halt_threshold=0.95)
    with caplog.at_level(logging.WARNING, logger="agent.cost"):
        for _ in range(74):
            g.record()
        assert "approaching daily quota" not in caplog.text
        # 75th call crosses the threshold.
        g.record()
        assert "approaching daily quota" in caplog.text
        # Further calls don't spam new warnings.
        before_count = caplog.text.count("approaching daily quota")
        for _ in range(10):
            g.record()
        after_count = caplog.text.count("approaching daily quota")
        assert before_count == after_count, "warning should only fire once"


def test_rpd_halt_threshold():
    g = RpdGuard(rpd_limit=100, warn_threshold=0.75, halt_threshold=0.95)
    for _ in range(94):
        g.record()
    assert not g.should_halt()
    g.record()  # 95th call — crosses halt threshold (0.95 * 100 = 95).
    assert g.should_halt()


def test_rpd_halt_message_contains_counts():
    g = RpdGuard(rpd_limit=100, halt_threshold=0.95)
    for _ in range(95):
        g.record()
    msg = g.halt_message()
    assert "95" in msg
    assert "100" in msg
    assert "Checkpoint saved" in msg


def test_rpd_custom_thresholds():
    # 50% warn, 60% halt — extreme but valid.
    g = RpdGuard(rpd_limit=10, warn_threshold=0.5, halt_threshold=0.6)
    for _ in range(5):
        g.record()
    assert not g.should_halt()
    g.record()  # 6th call -> 60%, halt.
    assert g.should_halt()
