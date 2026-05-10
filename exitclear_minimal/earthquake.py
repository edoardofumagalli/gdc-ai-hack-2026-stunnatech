from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .config import EarthquakeConfig

GRAVITY_MPS2 = 9.81


@dataclass(frozen=True)
class EarthquakeReading:
    vibration_mps2: float
    triggered: bool


class EarthquakeDetector:
    def __init__(self, config: EarthquakeConfig) -> None:
        self.threshold_mps2 = config.threshold_mps2
        self.min_duration_s = config.min_duration_s
        self._above_since_s: float | None = None
        self._alerted = False

    def update(self, ax: float, ay: float, az: float) -> EarthquakeReading:
        magnitude = math.sqrt(ax**2 + ay**2 + az**2)
        vibration = abs(magnitude - GRAVITY_MPS2)

        now_s = time.monotonic()
        triggered = False
        if vibration >= self.threshold_mps2:
            if self._above_since_s is None:
                self._above_since_s = now_s
            elif (
                not self._alerted
                and now_s - self._above_since_s >= self.min_duration_s
            ):
                self._alerted = True
                triggered = True
        else:
            self._above_since_s = None
            self._alerted = False

        return EarthquakeReading(
            vibration_mps2=vibration,
            triggered=triggered,
        )
