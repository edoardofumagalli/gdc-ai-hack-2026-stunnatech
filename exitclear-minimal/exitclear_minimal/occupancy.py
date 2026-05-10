from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import MonitoringConfig, RoiPx
from .volume import MonitoredVolume, VolumeBounds, xyz_from_depth


@dataclass
class OccupancyResult:
    raw_occupancy_pct: float
    occupancy_pct: float
    valid_pixels: int
    occupied_pixels: int
    occupied_mask: np.ndarray
    volume_mask: np.ndarray
    roi_px: RoiPx
    volume_bounds: VolumeBounds


class OccupancyMonitor:
    def __init__(self, config: MonitoringConfig, volume: MonitoredVolume) -> None:
        self.config = config
        self.volume = volume
        self._recent: deque[float] = deque(maxlen=config.smoothing_frames)

    def evaluate(
        self,
        depth_frame: np.ndarray,
        baseline_depth: np.ndarray,
        intrinsics: np.ndarray,
    ) -> OccupancyResult:
        if depth_frame.shape != baseline_depth.shape:
            raise ValueError(
                f"Depth frame shape {depth_frame.shape} does not match baseline "
                f"shape {baseline_depth.shape}"
            )

        height, width = depth_frame.shape[:2]
        current = depth_frame.astype(np.float32, copy=False)
        baseline = baseline_depth.astype(np.float32, copy=False)
        volume_mask = self.volume.projection_mask((height, width), intrinsics)
        roi = self.volume.projected_roi((height, width), intrinsics)

        min_depth = self.config.min_valid_depth_mm
        current_valid = np.isfinite(current) & (current >= min_depth)
        baseline_valid = (
            np.isfinite(baseline) & (baseline >= min_depth)
        )
        denominator = volume_mask & baseline_valid

        x_mm, y_mm, z_mm = xyz_from_depth(current, intrinsics)
        bounds = self.volume.bounds
        current_inside_volume = (
            current_valid
            & (x_mm >= bounds.x_min_mm)
            & (x_mm <= bounds.x_max_mm)
            & (y_mm >= bounds.y_min_mm)
            & (y_mm <= bounds.y_max_mm)
            & (z_mm >= bounds.z_min_mm)
            & (z_mm <= bounds.z_max_mm)
        )

        occupied_mask = (
            denominator
            & current_inside_volume
            & ((baseline - current) > float(self.config.depth_delta_mm))
        )
        valid_pixels = int(denominator.sum())
        occupied_pixels = int(occupied_mask.sum())
        raw_pct = (
            0.0
            if valid_pixels == 0
            else float(occupied_pixels / valid_pixels * 100.0)
        )

        self._recent.append(raw_pct)
        occupancy_pct = float(np.mean(self._recent)) if self._recent else raw_pct

        return OccupancyResult(
            raw_occupancy_pct=raw_pct,
            occupancy_pct=occupancy_pct,
            valid_pixels=valid_pixels,
            occupied_pixels=occupied_pixels,
            occupied_mask=occupied_mask,
            volume_mask=volume_mask,
            roi_px=roi,
            volume_bounds=bounds,
        )
