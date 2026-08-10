"""Project paths and parameter loading.

Every script reads its settings through :func:`load_params` so that
``params.yaml`` stays the only place a value is defined.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# src/config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Read ``params.yaml`` and return it as a plain dict.

    Parameters
    ----------
    path
        Override for the params file. Defaults to ``<project root>/params.yaml``.

    Raises
    ------
    FileNotFoundError
        If the params file does not exist.
    """
    params_path = Path(path) if path is not None else PARAMS_PATH
    if not params_path.is_file():
        raise FileNotFoundError(f"params file not found: {params_path}")

    with params_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(relative: str | Path) -> Path:
    """Turn a repo-relative path from params.yaml into an absolute path.

    Keeps the pipeline runnable from any working directory, which matters
    because DVC executes stages from the repo root but pytest does not.
    """
    relative = Path(relative)
    return relative if relative.is_absolute() else PROJECT_ROOT / relative
