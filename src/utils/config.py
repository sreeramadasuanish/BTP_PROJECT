"""Configuration loader and validator for FED4FED."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml(file_path: Path | str) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_base_dir() -> Path:
    """Return root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent


def load_main_config() -> Dict[str, Any]:
    """Load top-level config.yaml."""
    cfg_path = get_base_dir() / "config.yaml"
    return load_yaml(cfg_path)


def load_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Load dataset specific configuration."""
    clean_name = dataset_name.lower().replace("-", "_")
    cfg_path = get_base_dir() / "configs" / f"{clean_name}.yaml"
    return load_yaml(cfg_path)
