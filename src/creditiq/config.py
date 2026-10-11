"""Configuration loading, project-root resolution and logging setup."""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT_ENV_VAR = "CREDITIQ_ROOT"
CONFIG_ENV_VAR = "CREDITIQ_CONFIG"
DEFAULT_CONFIG_RELPATH = Path("config") / "config.yaml"

logger = logging.getLogger(__name__)


def find_project_root(start: Path | None = None) -> Path:
    """Return the project root.

    Resolution order: ``CREDITIQ_ROOT`` env var, then the first parent of ``start``
    (default: this file) that contains both ``pyproject.toml`` and ``config/``.
    """
    env_root = os.environ.get(ROOT_ENV_VAR)
    if env_root:
        return Path(env_root).expanduser().resolve()
    start = (start or Path(__file__)).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "config").is_dir():
            return candidate
    # Fall back to the current working directory (e.g. when installed non-editable).
    cwd = Path.cwd().resolve()
    if (cwd / "config").is_dir():
        return cwd
    raise FileNotFoundError(
        f"Could not locate the project root. Set the {ROOT_ENV_VAR} environment variable."
    )


@dataclass(frozen=True)
class Config:
    """Thin wrapper around the YAML configuration with path helpers."""

    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        """Resolve a ``paths.<key>`` entry relative to the project root."""
        value = self.raw["paths"][key]
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["random_seed"])

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a copy with nested ``overrides`` deep-merged into the raw config."""
        merged = _deep_merge(copy.deepcopy(self.raw), overrides)
        return Config(raw=merged, root=self.root)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, root: Path | None = None) -> Config:
    """Load the YAML config. ``path`` defaults to ``CREDITIQ_CONFIG`` or ``config/config.yaml``."""
    root = root or find_project_root()
    if path is None:
        path = os.environ.get(CONFIG_ENV_VAR) or (root / DEFAULT_CONFIG_RELPATH)
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping.")
    return Config(raw=raw, root=root)


def setup_logging(level: str | int | None = None) -> None:
    """Configure structured, timestamped console logging once per process."""
    level = level or os.environ.get("CREDITIQ_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    for noisy in ("matplotlib", "PIL", "numba", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
