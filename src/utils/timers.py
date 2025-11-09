"""Timing utilities for experiments."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Dict, Iterator


@dataclass
class Timer:
    """Context manager recording elapsed wall-clock time."""

    name: str
    sink: Dict[str, float]

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        end = time.perf_counter()
        self.sink[self.name] = end - self.start


@contextlib.contextmanager
def timed(name: str, sink: Dict[str, float]) -> Iterator[None]:
    """Convenient wrapper for ad-hoc timing."""

    timer = Timer(name=name, sink=sink)
    timer.__enter__()
    try:
        yield
    finally:
        timer.__exit__(None, None, None)
