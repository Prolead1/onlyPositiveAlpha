"""Path handling and validation utilities."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_workspace_root() -> Path:
    """Get workspace root directory.

    Returns
    -------
    Path
        Absolute path to workspace root.
    """
    # Assuming utils/ is at workspace root
    return Path(__file__).parent.parent


def get_storage_path(subpath: str = "") -> Path:
    """Get path within data/cached directory.

    Parameters
    ----------
    subpath : str
        Optional subdirectory path within data/cached.

    Returns
    -------
    Path
        Absolute path to storage location.
    """
    root = get_workspace_root()
    base = root / "data" / "cached"
    if subpath:
        return base / subpath
    return base


def ensure_directory(path: Path | str, *, parents: bool = True, exist_ok: bool = True) -> Path:
    """Ensure directory exists, create if needed.

    Parameters
    ----------
    path : Path | str
        Directory path to create.
    parents : bool
        Create parent directories if needed.
    exist_ok : bool
        Don't raise error if directory already exists.

    Returns
    -------
    Path
        The created/validated directory path.
    """
    path = Path(path)
    path.mkdir(parents=parents, exist_ok=exist_ok)
    return path


def validate_path_exists(path: Path | str, error_msg: str | None = None) -> Path:
    """Validate that a path exists, raise ValueError if not.

    Parameters
    ----------
    path : Path | str
        Path to validate.
    error_msg : str | None
        Custom error message. If None, uses default message.

    Returns
    -------
    Path
        Validated path.

    Raises
    ------
    ValueError
        If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        msg = error_msg or f"Path does not exist: {path}"
        raise ValueError(msg)
    return path
