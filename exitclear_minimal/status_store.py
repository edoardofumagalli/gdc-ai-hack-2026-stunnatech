from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import re
from threading import Lock

from .config import AppConfig
from .state_machine import State, StateStatus


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
        self._snapshot = self._build_snapshot(
            timestamp=self._last_status.timestamp,
            status=self._last_status,
        )

    def update(self, status: StateStatus) -> None:
        with self._lock:
            self._last_status = status
            self._snapshot = self._build_snapshot(
                timestamp=status.timestamp,
                status=status,
            )

    def trigger_earthquake(
        self,
        *,
        timestamp: datetime,
        vibration_mps2: float | None,
    ) -> bool:
        with self._lock:
            newly_triggered = self._earthquake_started_at is None
            if newly_triggered:
                self._earthquake_started_at = timestamp
            self._earthquake_vibration_mps2 = vibration_mps2
            self._snapshot = self._build_snapshot(
                timestamp=timestamp,
                status=self._last_status,
            )
            return newly_triggered

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
            "people": {"current": 0},
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
            description = "OAK IMU detected sustained vibration above threshold."
            if vibration is not None:
                description = (
                    "OAK IMU detected sustained vibration above threshold "
                    f"({vibration:.2f} m/s^2)."
                )
            snapshot["alerts"] = [
                {
                    "severity": "emergency",
                    "title": "Earthquake detected",
                    "description": description,
                }
            ]
            snapshot["evacuation"] = {
                "primaryExitId": self.exit_identity["id"],
                "route": self.exit_identity["name"],
                "arrow": "←",
                "startedAt": started_at.astimezone().isoformat(
                    timespec="milliseconds"
                ),
                "label": f"Use {self.exit_identity['name']}",
            }

        return snapshot


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
