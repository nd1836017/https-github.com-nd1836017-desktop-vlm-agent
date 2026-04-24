"""Configuration loading from environment variables."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


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
    log_level: str

    @classmethod
    def load(cls) -> Config:
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )

        return cls(
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
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
