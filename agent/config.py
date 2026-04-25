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
        )
        if cfg.rpd_warn_threshold >= cfg.rpd_halt_threshold:
            raise ValueError(
                f"RPD_WARN_THRESHOLD ({cfg.rpd_warn_threshold}) must be "
                f"less than RPD_HALT_THRESHOLD ({cfg.rpd_halt_threshold})"
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


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
