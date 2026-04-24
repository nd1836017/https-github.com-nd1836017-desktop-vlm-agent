"""Entry point: `python -m agent`."""
from __future__ import annotations

import argparse
import sys

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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = Config.load()
    except RuntimeError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    configure_logging(config.log_level)

    if args.reset and not args.resume:
        reset_state(config.state_file)

    return run(config, resume=args.resume)


if __name__ == "__main__":
    sys.exit(main())
