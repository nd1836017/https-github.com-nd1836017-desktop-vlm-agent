"""Entry point: `python -m agent`."""
from __future__ import annotations

import sys

from .agent import run
from .config import Config, configure_logging


def main() -> int:
    try:
        config = Config.load()
    except RuntimeError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    configure_logging(config.log_level)
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
