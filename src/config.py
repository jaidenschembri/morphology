"""Central configuration for paths, defaults, and reproducibility settings.

This module centralizes file-system paths and basic constants used across the
pipeline. Import from here instead of hardcoding paths or defaults in
scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = OUTPUT_DIR / "logs"

DEFAULT_NETWORK_TYPE = "drive"
DEFAULT_SEED = 42


@dataclass(frozen=True)
class PlotDefaults:
    """Default plotting parameters for visuals."""

    dpi: int = 300
    bgcolor: str = "#0b0c10"
    edge_color: str = "#f5f6f7"
    node_color: str = "#f5f6f7"
    cmap: str = "viridis"


def ensure_output_dirs(base_dir: Optional[Path] = None) -> None:
    """Create expected output subdirectories if missing."""

    root = Path(base_dir) if base_dir is not None else OUTPUT_DIR
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "embeddings").mkdir(parents=True, exist_ok=True)
    visuals = root / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)
    (visuals / "single_city").mkdir(parents=True, exist_ok=True)
    (visuals / "comparisons").mkdir(parents=True, exist_ok=True)


def data_path(*parts: str, base_dir: Optional[Path] = None) -> Path:
    """Return a path under the data directory."""

    root = Path(base_dir) if base_dir is not None else DATA_DIR
    return root.joinpath(*parts)


def output_path(*parts: str, base_dir: Optional[Path] = None) -> Path:
    """Return a path under the output directory."""

    root = Path(base_dir) if base_dir is not None else OUTPUT_DIR
    return root.joinpath(*parts)
