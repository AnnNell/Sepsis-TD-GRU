"""
Configuration loader and global reproducibility helpers.

Usage
-----
>>> from src.config import load_config, set_global_seed
>>> cfg = load_config()
>>> set_global_seed(cfg["split"]["random_seed"])
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the central YAML config.

    Parameters
    ----------
    path : str, optional
        Override path. If None, looks for SEPSIS_CONFIG env var,
        else falls back to ../configs/config.yaml relative to this file.
    """
    if path is None:
        path = os.environ.get("SEPSIS_CONFIG")
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / "configs" / "config.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve a path key inside cfg['paths'] to an absolute Path."""
    root = Path(cfg["paths"]["data_root"])
    sub = cfg["paths"][key]
    return root / sub


def ensure_dir(p: Path) -> Path:
    """Create directory if missing; return Path."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_global_seed(seed: int) -> None:
    """Set seeds across Python, NumPy, and PyTorch for reproducibility.

    Note: full determinism in CUDA convolutions also requires
    `torch.use_deterministic_algorithms(True)` and may slow training.
    We do NOT enforce that here; we set seeds and note the limitation
    in the manuscript.
    """
    import torch  # local import so data-only workflows don't pay the cost
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return CUDA if available, else CPU."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
