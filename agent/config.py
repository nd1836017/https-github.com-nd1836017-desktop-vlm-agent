"""Configuration loading from environment variables."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .files import FileMode


@dataclass(frozen=True)
class Config:
    gemini_api_key: str
    gemini_model: str
    tasks_file: Path
    animation_buffer_seconds: float
    max_step_retries: int
    max_replans_per_step: int
    history_window: int
    state_file: Path
    enable_two_stage_click: bool
    two_stage_crop_size_px: int
    max_click_candidates: int
    click_min_delay_seconds: float
    click_max_delay_seconds: float
    type_min_interval_seconds: float
    type_max_interval_seconds: float
    gemini_retry_max_attempts: int
    gemini_retry_base_delay_seconds: float
    gemini_retry_max_delay_seconds: float
    log_redact_type: bool
    enable_json_output: bool
    max_total_replans: int
    save_run_artifacts: bool
    run_artifacts_dir: Path
    rpd_limit: int
    rpd_warn_threshold: float
    rpd_halt_threshold: float
    file_mode: FileMode | None
    workdir: Path | None
    log_level: str
    # Tier 4 reliability: per-step wall-clock timeout and stuck-step detection.
    step_timeout_seconds: float
    stuck_step_threshold: int
    # Smart-screenshot (PR S Phase 1): downsample + JPEG-encode every
    # screenshot before sending to Gemini, and optionally skip the
    # planner image entirely when consecutive frames are identical.
    vlm_image_max_dim: int
    vlm_image_quality: int
    vlm_skip_identical_frames: bool
    # Browser fast-path: when true and Chrome is reachable on the CDP
    # debug port, the planner is told it can emit BROWSER_GO /
    # BROWSER_CLICK / BROWSER_FILL primitives that bypass vision-tokens.
    # Default OFF until the user has tested with their actual Chrome
    # setup; flip to true after launching Chrome with
    # ``--remote-debugging-port=29229``.
    browser_fast_path: bool
    browser_cdp_host: str
    browser_cdp_port: int
    # Smart task router (TASK_ROUTING). One Gemini call at run start
    # decomposes natural-language instructions into atomic actions and
    # tags each as 'browser-fast' / 'browser-vlm' / 'desktop-vlm'. The
    # planner sees the tag + suggested command as advisory context. Three
    # values: ``auto`` (default; runs Gemini; graceful fallback on
    # error), ``manual`` (only honors inline ``[tag]`` prefixes you
    # write yourself), ``off`` (no router at all).
    task_routing_mode: str
    # Task decomposition (TASK_DECOMPOSITION). One Gemini call at run
    # start that splits compound natural-language steps ("play the 2nd
    # video on youtube") into atomic substeps the planner can verify
    # one at a time. Two values: ``auto`` (default; runs Gemini;
    # graceful fallback on error), ``off`` (no decomposition).
    task_decomposition_mode: str
    # Skill library: SKILLS_DIR points at a folder of .txt skill files
    # so a tasks file can ``USE login_to_gmail`` and inline that skill's
    # contents at load time. ``None`` disables the directive (USE in a
    # tasks file becomes a load-time error).
    skills_dir: Path | None
    # Conditional logic: WAIT_UNTIL polling cadence + total budget. The
    # IF/ELSE/END_IF directives reuse WAIT_UNTIL_TIMEOUT_SECONDS only
    # for their single screenshot capture (no polling) — distinct knobs
    # left in case we want to evolve them independently later.
    wait_until_timeout_seconds: float
    wait_until_poll_seconds: float
    # Smart step-skip: when ``smart_skip_enabled`` is True, after a
    # step exhausts its replan budget the run does a 2-tier diagnosis
    # before halting:
    #   Tier 2 — three yes/no VLM checks (already done? previous-state
    #            still on screen? unrelated screen?)
    #   Tier 3 — open-ended classify-and-jump (describe screen, then
    #            ask which step matches; jump ahead if a future step
    #            is recognized).
    # ``smart_skip_max_tier`` clamps how aggressive the escalation
    # gets: 1 = retry only (legacy behavior), 2 = retry + Tier 2 only,
    # 3 = full escalation (default).
    smart_skip_enabled: bool
    smart_skip_max_tier: int
    # Skill auto-use: when ``True`` (default), at run start the agent
    # scans every step's text and auto-replaces any whose text matches
    # a known skill's ``# TRIGGERS:`` keyword list with that skill's
    # full content. Manual ``USE skill`` directives still work and are
    # always processed first. Set ``SKILL_AUTO_USE=off`` to disable
    # auto-detection (manual USE remains available).
    skill_auto_use_enabled: bool
    # Run memory (self-correction across runs). When ``run_memory_enabled``
    # is True, after a SUCCESSFUL run the agent summarises what worked
    # and saves a hint under a signature derived from the tasks-file
    # content. On the next run with the same signature, the planner is
    # seeded with that hint via ``plan_action(prior_run_hint=...)`` so
    # repeat runs become more reliable. Disabled with ``RUN_MEMORY=off``.
    # ``run_memory_dir`` is the directory holding the JSON store
    # (default ``memory/`` next to ``tasks.txt``). ``max_per_signature``
    # caps how many entries we keep per task; ``max_age_days`` evicts
    # old entries on load.
    run_memory_enabled: bool
    run_memory_dir: Path
    run_memory_max_per_signature: int
    run_memory_max_age_days: float

    @classmethod
    def load(cls) -> Config:
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )

        cfg = cls(
            gemini_api_key=api_key,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview").strip(),
            tasks_file=Path(os.getenv("TASKS_FILE", "tasks.txt")).expanduser(),
            animation_buffer_seconds=float(
                os.getenv("ANIMATION_BUFFER_SECONDS", "1.5")
            ),
            max_step_retries=int(os.getenv("MAX_STEP_RETRIES", "1")),
            max_replans_per_step=int(os.getenv("MAX_REPLANS_PER_STEP", "2")),
            history_window=int(os.getenv("HISTORY_WINDOW", "5")),
            state_file=Path(os.getenv("STATE_FILE", ".agent_state.json")).expanduser(),
            enable_two_stage_click=_env_bool("ENABLE_TWO_STAGE_CLICK", default=True),
            two_stage_crop_size_px=int(os.getenv("TWO_STAGE_CROP_SIZE_PX", "300")),
            max_click_candidates=int(os.getenv("MAX_CLICK_CANDIDATES", "5")),
            click_min_delay_seconds=float(os.getenv("CLICK_MIN_DELAY_SECONDS", "0.8")),
            click_max_delay_seconds=float(os.getenv("CLICK_MAX_DELAY_SECONDS", "2.0")),
            type_min_interval_seconds=float(os.getenv("TYPE_MIN_INTERVAL_SECONDS", "0.03")),
            type_max_interval_seconds=float(os.getenv("TYPE_MAX_INTERVAL_SECONDS", "0.12")),
            gemini_retry_max_attempts=int(os.getenv("GEMINI_RETRY_MAX_ATTEMPTS", "6")),
            gemini_retry_base_delay_seconds=float(
                os.getenv("GEMINI_RETRY_BASE_DELAY_SECONDS", "5.0")
            ),
            gemini_retry_max_delay_seconds=float(
                os.getenv("GEMINI_RETRY_MAX_DELAY_SECONDS", "300.0")
            ),
            log_redact_type=_env_bool("LOG_REDACT_TYPE", default=True),
            enable_json_output=_env_bool("ENABLE_JSON_OUTPUT", default=True),
            max_total_replans=int(os.getenv("MAX_TOTAL_REPLANS", "10")),
            save_run_artifacts=_env_bool("SAVE_RUN_ARTIFACTS", default=False),
            run_artifacts_dir=Path(
                os.getenv("RUN_ARTIFACTS_DIR", "runs")
            ).expanduser(),
            rpd_limit=int(os.getenv("RPD_LIMIT", "500")),
            rpd_warn_threshold=float(os.getenv("RPD_WARN_THRESHOLD", "0.75")),
            rpd_halt_threshold=float(os.getenv("RPD_HALT_THRESHOLD", "0.95")),
            file_mode=_env_file_mode(),
            workdir=_env_workdir(),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            step_timeout_seconds=float(
                os.getenv("STEP_TIMEOUT_SECONDS", "180.0")
            ),
            stuck_step_threshold=int(
                os.getenv("STUCK_STEP_THRESHOLD", "3")
            ),
            vlm_image_max_dim=int(os.getenv("VLM_IMAGE_MAX_DIM", "1280")),
            vlm_image_quality=int(os.getenv("VLM_IMAGE_QUALITY", "80")),
            vlm_skip_identical_frames=_env_bool(
                "VLM_SKIP_IDENTICAL_FRAMES", default=False
            ),
            browser_fast_path=_env_bool("BROWSER_FAST_PATH", default=False),
            browser_cdp_host=os.getenv("BROWSER_CDP_HOST", "localhost").strip(),
            browser_cdp_port=int(os.getenv("BROWSER_CDP_PORT", "29229")),
            task_routing_mode=os.getenv("TASK_ROUTING", "auto").strip().lower(),
            task_decomposition_mode=os.getenv(
                "TASK_DECOMPOSITION", "auto"
            ).strip().lower(),
            skills_dir=_env_skills_dir(),
            wait_until_timeout_seconds=float(
                os.getenv("WAIT_UNTIL_TIMEOUT_SECONDS", "30.0")
            ),
            wait_until_poll_seconds=float(
                os.getenv("WAIT_UNTIL_POLL_SECONDS", "2.0")
            ),
            smart_skip_enabled=_env_bool("SMART_SKIP", default=True),
            smart_skip_max_tier=max(
                1, min(3, int(os.getenv("SMART_SKIP_MAX_TIER", "3")))
            ),
            skill_auto_use_enabled=_env_bool("SKILL_AUTO_USE", default=True),
            run_memory_enabled=_env_bool("RUN_MEMORY", default=True),
            run_memory_dir=Path(
                os.getenv("RUN_MEMORY_DIR", "memory")
            ).expanduser(),
            run_memory_max_per_signature=max(
                1, int(os.getenv("RUN_MEMORY_MAX_PER_SIGNATURE", "3"))
            ),
            run_memory_max_age_days=max(
                0.0, float(os.getenv("RUN_MEMORY_MAX_AGE_DAYS", "30"))
            ),
        )
        if cfg.rpd_warn_threshold >= cfg.rpd_halt_threshold:
            raise ValueError(
                f"RPD_WARN_THRESHOLD ({cfg.rpd_warn_threshold}) must be "
                f"less than RPD_HALT_THRESHOLD ({cfg.rpd_halt_threshold})"
            )
        if not 1 <= cfg.vlm_image_quality <= 100:
            raise ValueError(
                f"VLM_IMAGE_QUALITY must be in [1, 100]; got {cfg.vlm_image_quality}"
            )
        if cfg.task_routing_mode not in {"auto", "manual", "off"}:
            raise ValueError(
                f"TASK_ROUTING must be one of auto/manual/off; "
                f"got {cfg.task_routing_mode!r}"
            )
        if cfg.task_decomposition_mode not in {"auto", "off"}:
            raise ValueError(
                f"TASK_DECOMPOSITION must be one of auto/off; "
                f"got {cfg.task_decomposition_mode!r}"
            )
        return cfg


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_file_mode() -> FileMode | None:
    """Parse ``FILE_MODE`` (case-insensitive). ``None`` means "ask interactively"."""
    raw = os.getenv("FILE_MODE", "").strip().lower()
    if not raw:
        return None
    try:
        return FileMode(raw)
    except ValueError as exc:
        raise ValueError(
            f"FILE_MODE={raw!r} is not one of "
            f"{[m.value for m in FileMode]}"
        ) from exc


def _env_workdir() -> Path | None:
    raw = os.getenv("WORKDIR", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _env_skills_dir() -> Path | None:
    """Resolve ``SKILLS_DIR``. Defaults to ``./skills`` if that folder exists.

    Returning ``None`` (when SKILLS_DIR is unset and ``./skills`` doesn't
    exist) makes ``USE skill_name`` raise a clear "skills directory not
    configured" error rather than silently doing nothing.
    """
    raw = os.getenv("SKILLS_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    default = Path("skills")
    if default.exists() and default.is_dir():
        return default
    return None


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
