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
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip(),
            tasks_file=Path(os.getenv("TASKS_FILE", "tasks.txt")).expanduser(),
            animation_buffer_seconds=float(
                os.getenv("ANIMATION_BUFFER_SECONDS", "1.5")
            ),
            max_step_retries=int(os.getenv("MAX_STEP_RETRIES", "1")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
