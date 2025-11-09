"""Simple helpers to capture time and memory complexity proxies."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ComplexityTracker:
    """Tracks wall-clock time for labeled sections."""

    timings: Dict[str, float] = field(default_factory=dict)
    start_times: Dict[str, float] = field(default_factory=dict)

    def start(self, name: str) -> None:
        self.start_times[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if name not in self.start_times:
            raise KeyError(f"Timer {name} was not started.")
        elapsed = time.perf_counter() - self.start_times.pop(name)
        self.timings[name] = self.timings.get(name, 0.0) + elapsed

    def summary(self) -> Dict[str, float]:
        return dict(self.timings)
