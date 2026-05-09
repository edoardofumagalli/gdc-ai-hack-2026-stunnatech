from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RoiPx, VolumeConfig


@dataclass(frozen=True)
class SpatialPoint:
    x_mm: float
    y_mm: float
    z_mm: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x_mm": round(self.x_mm, 1),
            "y_mm": round(self.y_mm, 1),
            "z_mm": round(self.z_mm, 1),
        }


@dataclass(frozen=True)
class VolumeBounds:
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    z_min_mm: float
    z_max_mm: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x_min_mm": round(self.x_min_mm, 1),
            "x_max_mm": round(self.x_max_mm, 1),
            "y_min_mm": round(self.y_min_mm, 1),
            "y_max_mm": round(self.y_max_mm, 1),
            "z_min_mm": round(self.z_min_mm, 1),
            "z_max_mm": round(self.z_max_mm, 1),
        }


@dataclass(frozen=True)
class MonitoredVolume:
    anchor_label: str
    anchor: SpatialPoint
    config: VolumeConfig
    bounds: VolumeBounds

    @classmethod
    def from_anchor(
        cls, anchor_label: str, anchor: SpatialPoint, config: VolumeConfig
    ) -> "MonitoredVolume":
        half_width = config.width_mm / 2.0
        bounds = VolumeBounds(
            x_min_mm=anchor.x_mm - half_width,
            x_max_mm=anchor.x_mm + half_width,
            y_min_mm=anchor.y_mm - config.height_below_anchor_mm,
            y_max_mm=anchor.y_mm,
            z_min_mm=anchor.z_mm - config.depth_before_anchor_mm,
            z_max_mm=anchor.z_mm,
        )
        return cls(
            anchor_label=anchor_label,
            anchor=anchor,
            config=config,
            bounds=bounds,
        )

    def as_event_dict(self) -> dict:
        return {
            "anchor_label": self.anchor_label,
            "anchor_xyz_mm": self.anchor.as_dict(),
            "volume_mm": self.config.as_dict(),
            "volume_bounds_mm": self.bounds.as_dict(),
        }

    def projection_mask(
        self, depth_shape: tuple[int, int], intrinsics: np.ndarray
    ) -> np.ndarray:
        height, width = depth_shape
        fx, fy = float(intrinsics[0][0]), float(intrinsics[1][1])
        cx, cy = float(intrinsics[0][2]), float(intrinsics[1][2])

        u = np.arange(width, dtype=np.float32)[None, :]
        v = np.arange(height, dtype=np.float32)[:, None]
        ray_x = (u - cx) / fx
        ray_y = -((v - cy) / fy)

        z_min = np.full((height, width), self.bounds.z_min_mm, dtype=np.float32)
        z_max = np.full((height, width), self.bounds.z_max_mm, dtype=np.float32)

        z_min, z_max, valid_x = _intersect_axis(
            z_min, z_max, ray_x, self.bounds.x_min_mm, self.bounds.x_max_mm
        )
        z_min, z_max, valid_y = _intersect_axis(
            z_min, z_max, ray_y, self.bounds.y_min_mm, self.bounds.y_max_mm
        )

        return valid_x & valid_y & (z_max >= z_min) & (z_max > 0)

    def projected_roi(
        self, depth_shape: tuple[int, int], intrinsics: np.ndarray
    ) -> RoiPx:
        mask = self.projection_mask(depth_shape, intrinsics)
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return RoiPx(0, 0, 0, 0)
        return RoiPx(
            x_min=int(xs.min()),
            y_min=int(ys.min()),
            x_max=int(xs.max()) + 1,
            y_max=int(ys.max()) + 1,
        )


def xyz_from_depth(depth_mm: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, ...]:
    height, width = depth_mm.shape[:2]
    fx, fy = float(intrinsics[0][0]), float(intrinsics[1][1])
    cx, cy = float(intrinsics[0][2]), float(intrinsics[1][2])

    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    z = depth_mm.astype(np.float32, copy=False)
    x = (u - cx) * z / fx
    y = -((v - cy) * z / fy)
    return x, y, z


def _intersect_axis(
    z_min: np.ndarray,
    z_max: np.ndarray,
    ray_axis: np.ndarray,
    axis_min: float,
    axis_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-6
    parallel = np.abs(ray_axis) < eps
    contains_origin = axis_min <= 0.0 <= axis_max
    valid_parallel = parallel & contains_origin

    with np.errstate(divide="ignore", invalid="ignore"):
        a = axis_min / ray_axis
        b = axis_max / ray_axis

    axis_z_min = np.minimum(a, b)
    axis_z_max = np.maximum(a, b)

    updated_z_min = np.where(parallel, z_min, np.maximum(z_min, axis_z_min))
    updated_z_max = np.where(parallel, z_max, np.minimum(z_max, axis_z_max))
    valid = valid_parallel | (~parallel & (updated_z_max >= updated_z_min))
    return updated_z_min, updated_z_max, valid
