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
    # Skill library helpers — these short-circuit the run loop so they
    # work even when GEMINI_API_KEY isn't set (handy for first-time
    # users authoring skills before connecting an API key).
    p.add_argument(
        "--list-skills",
        action="store_true",
        help=(
            "List every skill available under SKILLS_DIR (with a "
            "one-line preview) and exit. Useful for discovering what "
            "skills are already available before authoring new ones."
        ),
    )
    p.add_argument(
        "--new-skill",
        dest="new_skill",
        default=None,
        metavar="NAME",
        help=(
            "Scaffold a starter skill at SKILLS_DIR/<NAME>.txt and "
            "exit. The starter file includes a header comment plus an "
            "example BROWSER_GO so you can run it immediately. "
            "Refuses to overwrite an existing skill unless "
            "--overwrite-skill is also supplied."
        ),
    )
    p.add_argument(
        "--overwrite-skill",
        action="store_true",
        help=(
            "When used with --new-skill, replace any existing skill "
            "file with the same name. No effect on its own."
        ),
    )
    # Run-memory inspection helpers — short-circuit the run loop so
    # they work even when GEMINI_API_KEY isn't set.
    p.add_argument(
        "--list-memory",
        action="store_true",
        help=(
            "List every saved run-memory entry under RUN_MEMORY_DIR "
            "(with task signature, age, summary preview) and exit."
        ),
    )
    p.add_argument(
        "--clear-memory",
        action="store_true",
        help=(
            "Wipe every saved run-memory entry under RUN_MEMORY_DIR "
            "and exit. Cannot be undone."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Run-memory helpers run BEFORE Config.load() — they only need
    # RUN_MEMORY_DIR (or its default ./memory/) and a working JSON
    # store. Lets the user inspect / clear memory without an API key
    # configured.
    if args.list_memory or args.clear_memory:
        import os

        from .run_memory import RunMemoryStore

        memory_dir = Path(os.getenv("RUN_MEMORY_DIR", "memory")).expanduser()
        store_path = memory_dir / "run_memory.json"
        store = RunMemoryStore(store_path)
        store.load()
        if args.clear_memory:
            store.clear()
            try:
                store.save()
            except OSError as exc:
                print(f"[memory error] could not write {store_path}: {exc}", file=sys.stderr)
                return 2
            print(f"Cleared all run-memory entries at {store_path}.")
            return 0
        # --list-memory
        entries = store.all_entries()
        if not entries:
            print(
                f"No run-memory entries at {store_path}. "
                f"Successful runs will record entries here automatically."
            )
            return 0
        print(f"Run-memory entries at {store_path}:")
        # Newest first.
        for e in sorted(entries, key=lambda x: x.recorded_at, reverse=True):
            age = e.age_days()
            when = (
                f"{age:.1f}d ago" if age >= 1 else
                f"{age * 24:.1f}h ago" if age * 24 >= 1 else
                "just now"
            )
            preview = e.summary if len(e.summary) <= 80 else e.summary[:77] + "..."
            print(
                f"  signature={e.signature[:8]}  steps={e.step_count}  "
                f"actions={len(e.actions)}  recorded={when}\n"
                f"    summary: {preview}"
            )
        return 0

    # Skill helpers run BEFORE Config.load() so they work without a
    # configured GEMINI_API_KEY. They only need SKILLS_DIR (or its
    # default ./skills/), which we resolve directly here.
    if args.list_skills or args.new_skill is not None:
        from .config import _env_skills_dir
        from .skills import (
            DEFAULT_SKILLS_DIR_NAME,
            SkillError,
            list_skills,
            scaffold_skill,
        )

        skills_dir = _env_skills_dir() or Path(DEFAULT_SKILLS_DIR_NAME)
        if args.list_skills:
            infos = list_skills(skills_dir)
            if not infos:
                print(
                    f"No skills found in {skills_dir}. "
                    f"Create one with: python -m agent --new-skill <name>"
                )
                return 0
            print(f"Skills in {skills_dir}:")
            name_width = max((len(i.name) for i in infos), default=10)
            for info in infos:
                print(
                    f"  {info.name:<{name_width}}  "
                    f"({info.line_count} lines)  "
                    f"{info.preview}"
                )
            print(
                "\nUse a skill from any tasks file with `USE <name>`."
            )
            return 0
        # --new-skill <name>
        try:
            target = scaffold_skill(
                skills_dir,
                args.new_skill,
                overwrite=args.overwrite_skill,
            )
        except SkillError as exc:
            print(f"[skill error] {exc}", file=sys.stderr)
            return 2
        print(f"Scaffolded new skill: {target}")
        print(
            f"Edit it, then add `USE {args.new_skill}` to any tasks file."
        )
        return 0

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
