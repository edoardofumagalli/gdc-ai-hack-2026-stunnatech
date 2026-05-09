from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import dist

from .models import Component, ObstacleTrack
from .zone import ZoneGeometry


@dataclass
class _Track:
    track_id: int
    xyz_mm: list[int]
    centroid_px: tuple[float, float]
    timestamp: datetime
    missed_frames: int = 0


class SimpleCentroidTracker:
    def __init__(self, geometry: ZoneGeometry, max_match_distance_px: float = 35.0) -> None:
        self.geometry = geometry
        self.max_match_distance_px = max_match_distance_px
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(
        self, components: list[Component], timestamp: datetime
    ) -> list[ObstacleTrack]:
        unmatched = set(self._tracks.keys())
        obstacles: list[ObstacleTrack] = []

        for component in sorted(components, key=lambda item: item.area_px, reverse=True):
            xyz = self.geometry.pixel_to_camera_mm(
                component.centroid_px, component.median_depth_mm
            )
            track = self._match(component.centroid_px, unmatched)
            if track is None:
                track_id = self._next_id
                self._next_id += 1
                speed_mps = 0.0
            else:
                track_id = track.track_id
                unmatched.discard(track_id)
                dt = max(1e-6, (timestamp - track.timestamp).total_seconds())
                speed_mps = dist(xyz, track.xyz_mm) / dt / 1000.0

            self._tracks[track_id] = _Track(
                track_id=track_id,
                xyz_mm=xyz,
                centroid_px=component.centroid_px,
                timestamp=timestamp,
            )
            obstacles.append(
                ObstacleTrack(
                    track_id=track_id,
                    class_name="unknown_static_obstruction",
                    confidence=None,
                    centroid_xyz_mm=xyz,
                    speed_mps=speed_mps,
                    inside_zone=True,
                )
            )

        for track_id in unmatched:
            track = self._tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > 5:
                del self._tracks[track_id]

        return obstacles

    def _match(
        self, centroid_px: tuple[float, float], candidates: set[int]
    ) -> _Track | None:
        best: tuple[float, _Track] | None = None
        for track_id in candidates:
            track = self._tracks[track_id]
            distance_px = dist(centroid_px, track.centroid_px)
            if distance_px > self.max_match_distance_px:
                continue
            if best is None or distance_px < best[0]:
                best = (distance_px, track)
        return None if best is None else best[1]
