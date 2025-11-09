"""Lightweight logging helpers with optional Rich integration."""

from __future__ import annotations

import logging
from typing import Optional


def _build_handler() -> logging.Handler:
    """Create a Rich handler when available, else fall back."""

    try:
        from rich.logging import RichHandler  # type: ignore

        return RichHandler(
            rich_tracebacks=True,
            show_time=False,
            show_path=False,
            markup=True,
        )
    except Exception:  # pragma: no cover
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        return handler


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger."""

    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    handler = _build_handler()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
