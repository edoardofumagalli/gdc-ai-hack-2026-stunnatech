from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime
import re
from threading import Lock

import numpy as np

from .config import AppConfig
from .state_machine import State, StateStatus

HEATMAP_WIDTH = 48
HEATMAP_HEIGHT = 32
HEATMAP_HISTORY_SIZE = 16
HEATMAP_HISTORY_INTERVAL_S = 1.0


class DashboardStatusStore:
    def __init__(self, config: AppConfig, anchor_label: str) -> None:
        self.config = config
        self.exit_identity = _exit_identity(anchor_label)
        self._lock = Lock()
        self._last_status = StateStatus(
            timestamp=datetime.now().astimezone(),
            state=State.NO_BASELINE,
            occupancy_pct=0.0,
            persistence_s=0.0,
        )
        self._earthquake_started_at: datetime | None = None
        self._earthquake_vibration_mps2: float | None = None
        self._earthquake_audio_url: str | None = None
        self._earthquake_audio_sequence: list[str] | None = None
        self._earthquake_audio_pause_ms: int | None = None
        self._people_current = 0
        self._heatmap: dict | None = None
        self._heatmap_history = deque(maxlen=HEATMAP_HISTORY_SIZE)
        self._last_heatmap_history_ts: datetime | None = None
        self._snapshot = self._build_snapshot(
            timestamp=self._last_status.timestamp,
            status=self._last_status,
        )

    def update(
        self,
        status: StateStatus,
        *,
        people_count: float | None = None,
        people_density_map=None,
        exit_location: dict | None = None,
        camera_location: dict | None = None,
    ) -> None:
        with self._lock:
            self._last_status = status
            if people_count is not None:
                self._people_current = max(0, int(round(people_count)))
            self._update_heatmap(
                people_density_map,
                status.timestamp,
                exit_location=exit_location,
                camera_location=camera_location,
            )
            self._snapshot = self._build_snapshot(
                timestamp=status.timestamp,
                status=status,
            )

    def trigger_earthquake(
        self,
        *,
        timestamp: datetime,
        vibration_mps2: float | None,
        audio_url: str | None = None,
        audio_sequence: list[str] | None = None,
        audio_pause_ms: int | None = None,
        people_density_map=None,
        exit_location: dict | None = None,
        camera_location: dict | None = None,
    ) -> bool:
        with self._lock:
            newly_triggered = self._earthquake_started_at is None
            if newly_triggered:
                self._earthquake_started_at = timestamp
            self._earthquake_vibration_mps2 = vibration_mps2
            if audio_url is not None:
                self._earthquake_audio_url = audio_url
            if audio_sequence is not None:
                self._earthquake_audio_sequence = audio_sequence
            if audio_pause_ms is not None:
                self._earthquake_audio_pause_ms = audio_pause_ms
            self._update_heatmap(
                people_density_map,
                timestamp,
                exit_location=exit_location,
                camera_location=camera_location,
            )
            self._snapshot = self._build_snapshot(
                timestamp=timestamp,
                status=self._last_status,
            )
            return newly_triggered

    def set_earthquake_audio(
        self,
        *,
        audio_url: str,
        audio_sequence: list[str] | None = None,
        audio_pause_ms: int | None = None,
    ) -> None:
        with self._lock:
            self._earthquake_audio_url = audio_url
            self._earthquake_audio_sequence = audio_sequence
            self._earthquake_audio_pause_ms = audio_pause_ms
            timestamp = self._earthquake_started_at or datetime.now().astimezone()
            self._snapshot = self._build_snapshot(
                timestamp=timestamp,
                status=self._last_status,
            )

    def get(self) -> dict:
        with self._lock:
            return deepcopy(self._snapshot)

    def _build_snapshot(
        self,
        *,
        timestamp: datetime,
        status: StateStatus,
    ) -> dict:
        room = self.config.dashboard.room
        monitoring = self.config.monitoring
        exit_status = (
            State.CLEAR if status.state == State.NO_BASELINE else status.state
        )
        emergency_active = self._earthquake_started_at is not None

        snapshot = {
            "state": (
                "emergency" if emergency_active else _dashboard_state(status.state)
            ),
            "room": {
                "name": room.name,
                "deviceId": room.device_id,
                "capacity": room.capacity,
            },
            "people": {"current": self._people_current},
            "averageExitTimeSeconds": _estimate_average_exit_time_seconds(
                people_count=self._people_current,
                clear_exits=1 if exit_status == State.CLEAR else 0,
            ),
            "alerts": [],
            "exits": [
                {
                    **self.exit_identity,
                    "status": exit_status.value,
                    "occupancy": round(float(status.occupancy_pct), 1),
                    "occupancyThreshold": monitoring.occupancy_threshold_pct,
                }
            ],
            "updatedAt": timestamp.astimezone().isoformat(timespec="milliseconds"),
        }

        if emergency_active:
            started_at = self._earthquake_started_at
            vibration = self._earthquake_vibration_mps2
            audio_url = self._earthquake_audio_url
            audio_sequence = self._earthquake_audio_sequence
            audio_pause_ms = self._earthquake_audio_pause_ms
            description = "OAK IMU detected sustained vibration above threshold."
            if vibration is not None:
                description = (
                    "OAK IMU detected sustained vibration above threshold "
                    f"({vibration:.2f} m/s^2)."
                )
            alert = {
                "severity": "emergency",
                "title": "Earthquake detected",
                "description": description,
            }
            evacuation = {
                "primaryExitId": self.exit_identity["id"],
                "route": self.exit_identity["name"],
                "arrow": "←",
                "startedAt": started_at.astimezone().isoformat(
                    timespec="milliseconds"
                ),
                "label": f"Use {self.exit_identity['name']}",
            }
            if audio_url is not None:
                alert["audioUrl"] = audio_url
                evacuation["audioUrl"] = audio_url
            if audio_sequence:
                alert["audioSequence"] = audio_sequence
                evacuation["audioSequence"] = audio_sequence
            if audio_pause_ms is not None:
                alert["audioPauseMs"] = audio_pause_ms
                evacuation["audioPauseMs"] = audio_pause_ms
            if self._heatmap is not None:
                snapshot["heatmap"] = self._heatmap
                snapshot["heatmapHistory"] = list(self._heatmap_history)
            snapshot["alerts"] = [alert]
            snapshot["evacuation"] = evacuation

        return snapshot

    def _update_heatmap(
        self,
        density_map,
        timestamp: datetime,
        *,
        exit_location: dict | None = None,
        camera_location: dict | None = None,
    ) -> None:
        heatmap = _serialize_heatmap(
            density_map,
            timestamp,
            exit_location=exit_location,
            camera_location=camera_location,
        )
        if heatmap is None:
            return

        self._heatmap = heatmap
        if (
            self._last_heatmap_history_ts is None
            or (
                timestamp - self._last_heatmap_history_ts
            ).total_seconds()
            >= HEATMAP_HISTORY_INTERVAL_S
        ):
            self._heatmap_history.append(heatmap)
            self._last_heatmap_history_ts = timestamp


def _dashboard_state(state: State) -> str:
    if state == State.TRIGGERED:
        return "danger"
    if state == State.OCCUPIED_PENDING:
        return "caution"
    return "safe"


def _exit_identity(anchor_label: str) -> dict[str, str]:
    exit_type = _exit_type(anchor_label)
    exit_number = 1
    readable_type = exit_type.replace("_", " ").title()
    return {
        "id": f"{exit_type}_{exit_number}",
        "name": f"{readable_type} Exit {exit_number}",
        "type": exit_type,
    }


def _exit_type(anchor_label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", anchor_label.lower()).strip("_")
    if not slug:
        return "exit"
    if slug.startswith("emergency"):
        return "emergency"
    return slug


def _estimate_average_exit_time_seconds(
    *,
    people_count: int,
    clear_exits: int,
) -> int:
    if people_count <= 0:
        return 0

    usable_exits = max(1, clear_exits)
    people_per_second_per_exit = 1.3
    nominal_travel_seconds = 8
    average_queue_seconds = people_count / (
        2 * usable_exits * people_per_second_per_exit
    )
    return round(nominal_travel_seconds + average_queue_seconds)


def _serialize_heatmap(
    density_map,
    timestamp: datetime,
    *,
    exit_location: dict | None = None,
    camera_location: dict | None = None,
) -> dict | None:
    if density_map is None:
        return None

    try:
        density = np.asarray(density_map, dtype=np.float32).squeeze()
    except Exception:
        return None

    if density.ndim != 2 or density.size == 0:
        return None

    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    density = np.maximum(density, 0.0)
    resized = _resize_nearest(density, HEATMAP_HEIGHT, HEATMAP_WIDTH)
    peak = float(resized.max())
    total = float(resized.sum())
    if peak > 0:
        values = np.rint((resized / peak) * 100.0).astype(np.uint8)
    else:
        values = np.zeros((HEATMAP_HEIGHT, HEATMAP_WIDTH), dtype=np.uint8)

    heatmap = {
        "active": True,
        "width": HEATMAP_WIDTH,
        "height": HEATMAP_HEIGHT,
        "values": values.flatten().tolist(),
        "peak": round(peak, 4),
        "total": round(total, 4),
        "updatedAt": timestamp.astimezone().isoformat(timespec="milliseconds"),
    }
    if exit_location is not None:
        heatmap["exit"] = exit_location
    if camera_location is not None:
        heatmap["camera"] = camera_location
    return heatmap


def _resize_nearest(array: np.ndarray, height: int, width: int) -> np.ndarray:
    y_indices = np.linspace(0, array.shape[0] - 1, height).round().astype(int)
    x_indices = np.linspace(0, array.shape[1] - 1, width).round().astype(int)
    return array[np.ix_(y_indices, x_indices)]
