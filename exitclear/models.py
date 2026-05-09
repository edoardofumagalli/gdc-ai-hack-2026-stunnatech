from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np


class ComplianceState(str, Enum):
    COMPLIANT = "COMPLIANT"
    WATCH = "WATCH"
    BLOCKED_PENDING = "BLOCKED_PENDING"
    VIOLATION = "VIOLATION"
    CLEARED = "CLEARED"


@dataclass(frozen=True)
class DeviceConfig:
    id: str
    mode: str
    privacy_mode_default: bool


@dataclass(frozen=True)
class ImageRoi:
    x_min_pct: float
    x_max_pct: float
    y_min_pct: float
    y_max_pct: float


@dataclass(frozen=True)
class BoundsCameraMm:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int


@dataclass(frozen=True)
class OccupancyConfig:
    depth_delta_mm: int
    min_occupied_pct_watch: float
    min_occupied_pct_violation: float
    smoothing_frames: int
    min_component_area_px: int
    lateral_bins: int


@dataclass(frozen=True)
class ZoneConfig:
    id: str
    label: str
    type: str
    required_clear_width_mm: int
    monitored_height_mm: int
    persistence_threshold_s: float
    transient_person_grace_s: float
    image_roi: ImageRoi
    bounds_camera_mm: BoundsCameraMm
    occupancy: OccupancyConfig


@dataclass(frozen=True)
class OutputsConfig:
    event_log_path: str
    emit_json: bool
    preview_enabled: bool
    preview_fps: int


@dataclass(frozen=True)
class AppConfig:
    device: DeviceConfig
    zones: list[ZoneConfig]
    outputs: OutputsConfig


@dataclass
class Detection:
    class_name: str
    confidence: Optional[float]
    bbox_xyxy: tuple[int, int, int, int]


@dataclass
class FramePacket:
    timestamp: datetime
    depth_mm: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    scenario: str = ""


@dataclass
class Component:
    area_px: int
    centroid_px: tuple[float, float]
    median_depth_mm: float


@dataclass
class ObstacleTrack:
    track_id: int
    class_name: str
    centroid_xyz_mm: list[int]
    speed_mps: float
    inside_zone: bool
    confidence: Optional[float] = None

    def status_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "centroid_xyz_mm": self.centroid_xyz_mm,
            "speed_mps": round(self.speed_mps, 3),
            "inside_zone": self.inside_zone,
        }

    def event_dict(self) -> dict[str, Any]:
        payload = self.status_dict()
        payload["confidence"] = self.confidence
        return payload


@dataclass
class OccupancyResult:
    zone_id: str
    current_occupancy_pct: float
    occupancy_pct: float
    measured_free_width_mm: int
    occupied_bin_count: int
    occupied_bins: list[int]
    total_bins: int
    zone_pixel_count: int
    valid_pixel_count: int
    occupied_pixel_count: int
    depth_valid_pct: float
    baseline_ready: bool
    components: list[Component]
    severity: float
    reason: str


@dataclass
class ComplianceStatus:
    timestamp: datetime
    zone_id: str
    state: ComplianceState
    severity: float
    required_clear_width_mm: int
    measured_free_width_mm: int
    occupancy_pct: float
    persistence_s: float
    reason: str
    obstacles: list[ObstacleTrack]
    zone_pixel_count: int
    occupied_pixel_count: int
    occupied_bins: list[int]
    baseline_ready: bool
    depth_valid_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": format_timestamp(self.timestamp),
            "zone_id": self.zone_id,
            "state": self.state.value,
            "severity": round(self.severity, 3),
            "required_clear_width_mm": self.required_clear_width_mm,
            "measured_free_width_mm": self.measured_free_width_mm,
            "occupancy_pct": round(self.occupancy_pct, 1),
            "persistence_s": round(self.persistence_s, 1),
            "reason": self.reason,
            "obstacles": [obstacle.status_dict() for obstacle in self.obstacles],
            "zone_pixel_count": self.zone_pixel_count,
            "occupied_pixel_count": self.occupied_pixel_count,
            "occupied_bins": self.occupied_bins,
            "baseline_ready": self.baseline_ready,
            "depth_valid_pct": round(self.depth_valid_pct, 1),
        }


@dataclass
class ComplianceEvent:
    event_type: str
    device_id: str
    timestamp: datetime
    zone_id: str
    zone_type: str
    previous_state: ComplianceState
    state: ComplianceState
    severity: float
    reason: str
    required_clear_width_mm: int
    measured_free_width_mm: int
    occupancy_pct: float
    persistence_s: float
    obstacles: list[ObstacleTrack]
    privacy_mode: bool
    raw_video_streamed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "device_id": self.device_id,
            "timestamp": format_timestamp(self.timestamp),
            "zone_id": self.zone_id,
            "zone_type": self.zone_type,
            "previous_state": self.previous_state.value,
            "state": self.state.value,
            "severity": round(self.severity, 3),
            "reason": self.reason,
            "required_clear_width_mm": self.required_clear_width_mm,
            "measured_free_width_mm": self.measured_free_width_mm,
            "occupancy_pct": round(self.occupancy_pct, 1),
            "persistence_s": round(self.persistence_s, 1),
            "obstacles": [obstacle.event_dict() for obstacle in self.obstacles],
            "privacy_mode": self.privacy_mode,
            "raw_video_streamed": self.raw_video_streamed,
        }


# Backward-compatible aliases for early Phase 1 code and imports.
Obstacle = ObstacleTrack
Status = ComplianceStatus


def format_timestamp(value: datetime) -> str:
    return value.astimezone().isoformat(timespec="milliseconds")


def dataclass_to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return dataclass_to_plain(asdict(value))
    if isinstance(value, dict):
        return {key: dataclass_to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [dataclass_to_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return format_timestamp(value)
    return value
