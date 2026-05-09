from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np

from exitclear.models import FramePacket


@dataclass(frozen=True)
class FakeScenario:
    name: str
    frames: int


class FakeDepthSource:
    def __init__(
        self,
        frame_shape: tuple[int, int] = (120, 160),
        fps: float = 5.0,
        baseline_depth_mm: int = 2200,
        start_time: datetime | None = None,
    ) -> None:
        self.frame_shape = frame_shape
        self.fps = fps
        self.baseline_depth_mm = baseline_depth_mm
        self._start = start_time or datetime(
            2026, 5, 9, 11, 42, 0, tzinfo=timezone(timedelta(hours=2))
        )
        self._index = 0

    def calibration_frames(self, count: int) -> list[FramePacket]:
        return [
            FramePacket(
                timestamp=self._start - timedelta(seconds=(count - index) / self.fps),
                depth_mm=self._blank_depth(),
                scenario="baseline_clear",
            )
            for index in range(count)
        ]

    def frames(self) -> Iterable[FramePacket]:
        scenarios = [
            FakeScenario("clear_path", 5),
            FakeScenario("object_outside_zone", 5),
            FakeScenario("object_inside_zone", 1),
            FakeScenario("object_persists", 14),
            FakeScenario("violation", 5),
            FakeScenario("object_removed", 1),
            FakeScenario("clear_again", 4),
        ]
        for scenario in scenarios:
            for _ in range(scenario.frames):
                yield self._packet(scenario.name)

    def _packet(self, scenario: str) -> FramePacket:
        depth = self._blank_depth()
        if scenario == "object_outside_zone":
            depth[72:102, 8:28] = 1450
        elif scenario in {"object_inside_zone", "object_persists", "violation"}:
            depth[48:104, 62:100] = 1450

        timestamp = self._start + timedelta(seconds=self._index / self.fps)
        self._index += 1
        return FramePacket(timestamp=timestamp, depth_mm=depth, scenario=scenario)

    def _blank_depth(self) -> np.ndarray:
        return np.full(self.frame_shape, self.baseline_depth_mm, dtype=np.uint16)
