from __future__ import annotations

import numpy as np

from .models import BoundsCameraMm, ZoneConfig


class ZoneGeometry:
    def __init__(
        self, zone: ZoneConfig, frame_shape: tuple[int, int], bounds: BoundsCameraMm
    ) -> None:
        self.zone = zone
        self.bounds = bounds
        self.height, self.width = frame_shape
        self.x0 = int(round(zone.image_roi.x_min_pct * self.width))
        self.x1 = int(round(zone.image_roi.x_max_pct * self.width))
        self.y0 = int(round(zone.image_roi.y_min_pct * self.height))
        self.y1 = int(round(zone.image_roi.y_max_pct * self.height))
        self.x0 = max(0, min(self.width, self.x0))
        self.x1 = max(self.x0 + 1, min(self.width, self.x1))
        self.y0 = max(0, min(self.height, self.y0))
        self.y1 = max(self.y0 + 1, min(self.height, self.y1))
        self.mask = self._build_mask()

    def _build_mask(self) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[self.y0 : self.y1, self.x0 : self.x1] = True
        return mask

    def pixel_to_camera_mm(self, centroid_px: tuple[float, float], z_mm: float) -> list[int]:
        x_px, y_px = centroid_px
        x_ratio = _clamp((x_px - self.x0) / max(1, self.x1 - self.x0), 0.0, 1.0)
        y_ratio = _clamp((y_px - self.y0) / max(1, self.y1 - self.y0), 0.0, 1.0)
        x_mm = self.bounds.x_min + x_ratio * (self.bounds.x_max - self.bounds.x_min)
        y_mm = self.bounds.y_min + y_ratio * (self.bounds.y_max - self.bounds.y_min)
        return [int(round(x_mm)), int(round(y_mm)), int(round(z_mm))]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
