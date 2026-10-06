"""Configuration helpers shared by the ML and API layers."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    """Load the repository configuration using paths relative to the repo."""
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def root_path(value: str) -> Path:
    """Resolve a configured relative path from the repository root."""
    return ROOT / value
