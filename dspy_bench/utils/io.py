"""I/O utilities for dspy-bench."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        path: Output path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """Load data from JSON file.

    Args:
        path: Input path

    Returns:
        Loaded data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r") as f:
        return json.load(f)


def save_yaml(data: Any, path: Union[str, Path]) -> None:
    """Save data to YAML file.

    Args:
        data: Data to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def load_yaml(path: Union[str, Path]) -> Any:
    """Load data from YAML file.

    Args:
        path: Input path

    Returns:
        Loaded data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r") as f:
        return yaml.safe_load(f)


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """Copy a file from source to destination.

    Args:
        src: Source path
        dst: Destination path
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, dst)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str) -> str:
    """Make filename safe for filesystem.

    Args:
        filename: Original filename

    Returns:
        Safe filename
    """
    import re

    # Replace unsafe characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing dots and spaces
    safe_name = safe_name.strip('. ')
    # Limit length
    if len(safe_name) > 255:
        safe_name = safe_name[:255]

    return safe_name or "unnamed"