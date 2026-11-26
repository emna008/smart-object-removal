"""Mesure des images par seconde (moyenne glissante)."""
import time
from collections import deque


class FPSCounter:
    def __init__(self, window: int = 30):
        self._times = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
