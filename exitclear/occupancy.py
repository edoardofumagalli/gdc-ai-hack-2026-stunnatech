from __future__ import annotations

from collections import deque

import numpy as np

from .models import Component, OccupancyResult, ZoneConfig
from .zone import ZoneGeometry


class OccupancyEngine:
    def __init__(self, zone: ZoneConfig, frame_shape: tuple[int, int]) -> None:
        self.zone = zone
        self.geometry = ZoneGeometry(zone, frame_shape)
        self._recent_occupancy_pct: deque[float] = deque(
            maxlen=max(1, zone.occupancy.smoothing_frames)
        )

    def evaluate(
        self, depth_mm: np.ndarray, baseline_depth_mm: np.ndarray
    ) -> OccupancyResult:
        if depth_mm.shape != baseline_depth_mm.shape:
            raise ValueError("Depth frame and baseline must have the same shape")

        bounds = self.zone.bounds_camera_mm
        current_depth = depth_mm.astype(np.float32, copy=False)
        baseline_depth = baseline_depth_mm.astype(np.float32, copy=False)
        zone_mask = self.geometry.mask
        zone_pixel_count = int(zone_mask.sum())

        current_valid = (
            zone_mask
            & np.isfinite(current_depth)
            & (current_depth > bounds.z_min)
            & (current_depth < bounds.z_max)
        )
        baseline_valid = (
            zone_mask
            & np.isfinite(baseline_depth)
            & (baseline_depth > bounds.z_min)
            & (baseline_depth < bounds.z_max)
        )
        valid = current_valid & baseline_valid
        depth_valid_pct = (
            0.0 if zone_pixel_count == 0 else float(valid.sum() / zone_pixel_count * 100.0)
        )

        closer_than_baseline = (
            (baseline_depth - current_depth) > self.zone.occupancy.depth_delta_mm
        )
        occupied_raw = valid & closer_than_baseline
        occupied, components = _filter_components(
            occupied_raw,
            current_depth,
            min_area_px=self.zone.occupancy.min_component_area_px,
        )

        valid_pixel_count = int(valid.sum())
        occupied_pixel_count = int(occupied.sum())
        raw_occupancy_pct = (
            0.0
            if valid_pixel_count == 0
            else float(occupied_pixel_count / valid_pixel_count * 100.0)
        )
        if raw_occupancy_pct == 0.0 and not components:
            self._recent_occupancy_pct.clear()
        self._recent_occupancy_pct.append(raw_occupancy_pct)
        occupancy_pct = float(np.mean(self._recent_occupancy_pct))

        occupied_bin_flags, total_bins = self._lateral_occupancy_bins(occupied)
        occupied_bins = [
            idx for idx, is_occupied in enumerate(occupied_bin_flags) if is_occupied
        ]
        longest_free_run = _longest_false_run(occupied_bin_flags)
        measured_width = int(
            round(
                longest_free_run
                / max(1, total_bins)
                * self.zone.required_clear_width_mm
            )
        )
        severity = _severity(
            occupancy_pct=occupancy_pct,
            measured_free_width_mm=measured_width,
            required_clear_width_mm=self.zone.required_clear_width_mm,
            violation_threshold=self.zone.occupancy.min_occupied_pct_violation,
        )
        reason = _reason(occupancy_pct, measured_width, self.zone)

        return OccupancyResult(
            zone_id=self.zone.id,
            current_occupancy_pct=raw_occupancy_pct,
            occupancy_pct=occupancy_pct,
            measured_free_width_mm=measured_width,
            occupied_bin_count=len(occupied_bins),
            occupied_bins=occupied_bins,
            total_bins=total_bins,
            zone_pixel_count=zone_pixel_count,
            valid_pixel_count=valid_pixel_count,
            occupied_pixel_count=occupied_pixel_count,
            depth_valid_pct=depth_valid_pct,
            baseline_ready=True,
            components=components,
            severity=severity,
            reason=reason,
        )

    def _lateral_occupancy_bins(self, occupied: np.ndarray) -> tuple[list[bool], int]:
        bins = max(1, self.zone.occupancy.lateral_bins)
        roi = occupied[
            self.geometry.y0 : self.geometry.y1,
            self.geometry.x0 : self.geometry.x1,
        ]
        if roi.size == 0:
            return [False] * bins, bins

        bin_edges = np.linspace(0, roi.shape[1], bins + 1, dtype=int)
        occupied_bins = []
        for idx in range(bins):
            x0, x1 = bin_edges[idx], bin_edges[idx + 1]
            cell = roi[:, x0:x1]
            if cell.size == 0:
                occupied_bins.append(False)
                continue
            pct = float(cell.sum() / cell.size * 100.0)
            occupied_bins.append(pct >= self.zone.occupancy.min_occupied_pct_watch)
        return occupied_bins, bins


def _filter_components(
    occupied: np.ndarray, depth_mm: np.ndarray, min_area_px: int
) -> tuple[np.ndarray, list[Component]]:
    filtered = np.zeros_like(occupied, dtype=bool)
    visited = np.zeros_like(occupied, dtype=bool)
    height, width = occupied.shape
    components: list[Component] = []

    for y in range(height):
        for x in range(width):
            if not occupied[y, x] or visited[y, x]:
                continue
            pixels = _flood_fill(occupied, visited, x, y)
            if len(pixels) < min_area_px:
                continue
            ys = np.array([item[0] for item in pixels], dtype=np.int32)
            xs = np.array([item[1] for item in pixels], dtype=np.int32)
            filtered[ys, xs] = True
            depths = depth_mm[ys, xs]
            components.append(
                Component(
                    area_px=len(pixels),
                    centroid_px=(float(xs.mean()), float(ys.mean())),
                    median_depth_mm=float(np.median(depths)),
                )
            )
    return filtered, components


def _flood_fill(
    occupied: np.ndarray, visited: np.ndarray, start_x: int, start_y: int
) -> list[tuple[int, int]]:
    height, width = occupied.shape
    stack = [(start_y, start_x)]
    pixels: list[tuple[int, int]] = []
    visited[start_y, start_x] = True

    while stack:
        y, x = stack.pop()
        pixels.append((y, x))
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if visited[ny, nx] or not occupied[ny, nx]:
                continue
            visited[ny, nx] = True
            stack.append((ny, nx))
    return pixels


def _longest_false_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _severity(
    occupancy_pct: float,
    measured_free_width_mm: int,
    required_clear_width_mm: int,
    violation_threshold: float,
) -> float:
    occupancy_score = occupancy_pct / max(1.0, violation_threshold)
    width_score = 1.0 - (measured_free_width_mm / max(1, required_clear_width_mm))
    return float(max(0.0, min(1.0, max(occupancy_score, width_score))))


def _reason(
    occupancy_pct: float,
    measured_free_width_mm: int,
    zone: ZoneConfig,
) -> str:
    if measured_free_width_mm < zone.required_clear_width_mm:
        return "clearance_reduction_detected"
    if occupancy_pct < zone.occupancy.min_occupied_pct_watch:
        return "clearance_volume_free"
    return "depth_occupancy_watch"
