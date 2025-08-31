from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, TextIO
import os

# Resolve a single, absolute logs directory at repo root
_here = os.path.abspath(os.path.dirname(__file__))
_repo_root = os.path.abspath(os.path.join(_here, os.pardir, os.pardir))
_default_logs_dir = os.path.join(_repo_root, "logs")

_log_file: Optional[TextIO] = None
_log_directory = _default_logs_dir
_log_counter_file = os.path.join(_log_directory, "log_counter.txt")
_log_number: Optional[int] = None
_verbose_mode = False
_original_print = print


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_print(*args, **kwargs) -> None:
    timestamp = get_timestamp()
    message = " ".join(str(arg) for arg in args)
    timestamped = f"[{timestamp}] {message}"
    _original_print(timestamped, **kwargs)
    if _log_file is not None:
        _log_file.write(timestamped + "\n")
        _log_file.flush()


def debug_print(*args, **kwargs) -> None:
    if _verbose_mode:
        log_print("[DEBUG]", *args, **kwargs)


def set_verbose_mode(value: bool) -> None:
    global _verbose_mode
    _verbose_mode = value


def get_log_directory() -> str:
    """Return the absolute path to the unified logs directory."""
    return _log_directory


def get_next_log_number() -> int:
    global _log_counter_file, _log_number
    if _log_number is not None:
        return _log_number
    os.makedirs(os.path.dirname(_log_counter_file), exist_ok=True)
    counter = 1
    try:
        if os.path.exists(_log_counter_file):
            with open(_log_counter_file, "r") as f:
                counter = int(f.read().strip())
    except Exception:
        counter = 1
    try:
        with open(_log_counter_file, "w") as f:
            f.write(str(counter + 1))
    except Exception:
        pass
    _log_number = counter
    return counter


def initialize_logging(log_directory: Optional[str] = None) -> str:
    """Initialize logging to the unified repo_root/logs unless overridden.

    If a relative path is provided, it is resolved against the repo root to
    avoid scattering logs across different working directories.
    """
    global _log_directory, _log_counter_file
    if not log_directory:
        _log_directory = _default_logs_dir
    else:
        # Resolve relative paths to repo root; keep absolute paths as-is
        _log_directory = (
            os.path.join(_repo_root, log_directory)
            if not os.path.isabs(log_directory)
            else log_directory
        )
    os.makedirs(_log_directory, exist_ok=True)
    _log_counter_file = os.path.join(_log_directory, "log_counter.txt")
    log_number = get_next_log_number()
    return os.path.join(_log_directory, f"IMO{log_number}.log")


def set_log_file(log_file_path: str) -> bool:
    global _log_file
    if log_file_path:
        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            _log_file = open(log_file_path, "w", encoding="utf-8")
            return True
        except Exception as e:
            log_print(f"Error opening log file {log_file_path}: {e}")
            return False
    return True


def close_log_file() -> None:
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None
