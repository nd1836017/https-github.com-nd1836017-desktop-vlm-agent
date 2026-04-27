"""Entry point: `python -m agent`."""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from .agent import run
from .config import Config, configure_logging
from .files import FileMode
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
    p.add_argument(
        "--mode",
        dest="file_mode",
        choices=[m.value for m in FileMode],
        default=None,
        help=(
            "How files captured during the run (DOWNLOAD / CAPTURE_FOR_AI) "
            "should be persisted. 'temp' wipes them on success and keeps "
            "them on failure; 'save' persists everything to --workdir; "
            "'feed' never writes to disk and feeds bytes straight to the "
            "VLM. If omitted you'll be asked at run start (unless FILE_MODE "
            "is set in .env)."
        ),
    )
    p.add_argument(
        "--workdir",
        dest="workdir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Where downloads land when --mode save is selected. Created if "
            "it doesn't exist. Defaults to ./agent_files when not supplied."
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
    # Run replay dashboard. Mutually exclusive with running the agent —
    # this just serves the existing artifacts directory and exits when
    # interrupted.
    p.add_argument(
        "--serve-dashboard",
        action="store_true",
        help=(
            "Serve the read-only run replay dashboard on localhost:8000 "
            "instead of running the agent. Reads existing run artifacts "
            "from RUN_ARTIFACTS_DIR (default 'runs/'). Requires fastapi "
            "and uvicorn to be installed."
        ),
    )
    p.add_argument(
        "--dashboard-host",
        default="127.0.0.1",
        metavar="HOST",
        help=(
            "Bind address for --serve-dashboard. Defaults to 127.0.0.1 "
            "(localhost only) so screenshots in the artifacts dir aren't "
            "exposed on the network. Set to 0.0.0.0 to expose."
        ),
    )
    p.add_argument(
        "--dashboard-port",
        type=int,
        default=8000,
        metavar="PORT",
        help="Port for --serve-dashboard. Defaults to 8000.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = Config.load()
    except (RuntimeError, ValueError) as exc:
        # ValueError covers FILE_MODE / RPD threshold validation; RuntimeError
        # covers missing/blank GEMINI_API_KEY etc. Both surface as "[config error]".
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    # Apply CLI overrides on top of the .env config.
    if args.two_stage_click is not None:
        config = dataclasses.replace(config, enable_two_stage_click=args.two_stage_click)

    configure_logging(config.log_level)

    if args.serve_dashboard:
        # Lazy-import the dashboard so the regular agent path doesn't pull
        # in FastAPI on every invocation.
        from .dashboard import serve as serve_dashboard

        try:
            serve_dashboard(
                config.run_artifacts_dir,
                host=args.dashboard_host,
                port=args.dashboard_port,
            )
        except RuntimeError as exc:
            print(f"[dashboard error] {exc}", file=sys.stderr)
            return 2
        return 0

    if args.reset and not args.resume:
        reset_state(config.state_file)

    cli_file_mode = FileMode(args.file_mode) if args.file_mode else None

    return run(
        config,
        resume=args.resume,
        csv_override=args.csv_override,
        cli_file_mode=cli_file_mode,
        cli_workdir=args.workdir,
    )


if __name__ == "__main__":
    sys.exit(main())
