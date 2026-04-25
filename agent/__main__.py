"""Entry point: `python -m agent`."""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from .agent import run
from .config import Config, configure_logging
from .state import reset_state


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="agent", description="Desktop VLM automation agent.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last successful step in the checkpoint file (STATE_FILE).",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Delete the checkpoint file before running (ignored with --resume).",
    )
    p.add_argument(
        "--csv",
        dest="csv_override",
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "Override the CSV path used by every FOR_EACH_ROW block in the "
            "tasks file. Useful for swapping demo data for real data without "
            "editing tasks.txt."
        ),
    )
    # Two-stage CLICK toggle. Mutually exclusive; if neither is given, we fall
    # back to the .env / ENABLE_TWO_STAGE_CLICK value.
    click_group = p.add_mutually_exclusive_group()
    click_group.add_argument(
        "--two-stage-click",
        dest="two_stage_click",
        action="store_true",
        default=None,
        help=(
            "Force the two-stage CLICK refinement (crop + VLM-refined pick) ON "
            "for this run, overriding ENABLE_TWO_STAGE_CLICK. Safer, uses more quota."
        ),
    )
    click_group.add_argument(
        "--no-two-stage-click",
        dest="two_stage_click",
        action="store_false",
        default=None,
        help=(
            "Force the two-stage CLICK refinement OFF for this run, overriding "
            "ENABLE_TWO_STAGE_CLICK. Faster / cheaper; use for simple tasks."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = Config.load()
    except RuntimeError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    # Apply CLI overrides on top of the .env config.
    if args.two_stage_click is not None:
        config = dataclasses.replace(config, enable_two_stage_click=args.two_stage_click)

    configure_logging(config.log_level)

    if args.reset and not args.resume:
        reset_state(config.state_file)

    return run(config, resume=args.resume, csv_override=args.csv_override)


if __name__ == "__main__":
    sys.exit(main())
