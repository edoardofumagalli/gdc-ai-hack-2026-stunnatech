from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Protocol

import numpy as np


@dataclass
class FramePacket:
    timestamp: datetime
    rgb_frame: np.ndarray | None
    depth_frame: np.ndarray
    intrinsics: np.ndarray
    earthquake_triggered: bool = False
    earthquake_vibration_mps2: float | None = None


class DepthSource(Protocol):
    fps: float

    def frames(self) -> Iterator[FramePacket]:
        ...

    def close(self) -> None:
        ...
