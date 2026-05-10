from __future__ import annotations

from collections import deque
import warnings

import numpy as np


class BaselineBuilder:
    def __init__(self, frame_count: int, min_valid_depth_mm: int) -> None:
        self.frame_count = frame_count
        self.min_valid_depth_mm = min_valid_depth_mm
        self._frames: deque[np.ndarray] = deque(maxlen=frame_count)

    @property
    def progress(self) -> int:
        return len(self._frames)

    @property
    def ready(self) -> bool:
        return len(self._frames) >= self.frame_count

    def add(self, depth_frame: np.ndarray) -> None:
        depth = depth_frame.astype(np.float32, copy=True)
        invalid = (~np.isfinite(depth)) | (depth < self.min_valid_depth_mm)
        depth[invalid] = np.nan
        self._frames.append(depth)

    def compute(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("Not enough frames to compute baseline")

        stack = np.stack(list(self._frames), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmedian(stack, axis=0).astype(np.float32)
