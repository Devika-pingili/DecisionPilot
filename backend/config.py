"""Shared local/demo data path configuration for the DecisionPilot service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DATA_ROOT_ENV = "DECISIONPILOT_DATA_ROOT"


@dataclass(frozen=True)
class DataPaths:
    root: Path
    raw_dir: Path
    index_path: Path
    model_path: Path
    build_index_if_missing: bool


def resolve_data_paths(data_root: str | Path | None = None) -> DataPaths:
    """Resolve data locations, defaulting to the existing local project layout."""
    configured_root = data_root
    if configured_root is None:
        configured_root = os.getenv(DATA_ROOT_ENV)

    build_index_if_missing = configured_root is not None and bool(str(configured_root).strip())
    if configured_root is None or not str(configured_root).strip():
        root = DEFAULT_DATA_ROOT
    else:
        root = Path(configured_root).expanduser()
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        root = root.resolve()

    return DataPaths(
        root=root,
        raw_dir=root / "raw" / "instacart",
        index_path=root / "processed" / "history.sqlite",
        model_path=root / "processed" / "ml_dev" / "model.joblib",
        build_index_if_missing=build_index_if_missing,
    )
