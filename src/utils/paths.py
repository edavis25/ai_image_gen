from __future__ import annotations

import os
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = SRC_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
PACKAGES_DIR = PROJECT_ROOT / "packages"


def ensure_project_dirs() -> None:
    """Ensure key project directories exist."""
    for p in (MODELS_DIR, OUTPUT_DIR, PACKAGES_DIR, SRC_DIR):
        p.mkdir(parents=True, exist_ok=True)


def resolve_path(p: str | os.PathLike) -> Path:
    return (PROJECT_ROOT / Path(p)).resolve() if not str(p).startswith("/") else Path(p).resolve()


__all__ = [
    "PROJECT_ROOT",
    "SRC_DIR",
    "MODELS_DIR",
    "OUTPUT_DIR",
    "PACKAGES_DIR",
    "ensure_project_dirs",
    "resolve_path",
]
