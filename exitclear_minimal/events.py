from __future__ import annotations

import json
from pathlib import Path

from .config import AppConfig, RoiPx
from .state_machine import State, StateStatus
from .volume import MonitoredVolume


class EventWriter:
    def __init__(
        self, path: str | Path, config: AppConfig, volume: MonitoredVolume
    ) -> None:
        self.path = Path(path)
        self.config = config
        self.volume = volume
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit_state_change(
        self, status: StateStatus, previous_state: State, projected_roi_px: RoiPx
    ) -> dict | None:
        if not self.config.output.write_events_jsonl:
            return None

        if status.state == State.TRIGGERED:
            event_type = "volume_occupancy_triggered"
        elif status.state == State.CLEAR and previous_state in {
            State.OCCUPIED_PENDING,
            State.TRIGGERED,
        }:
            event_type = "volume_occupancy_cleared"
        else:
            return None

        monitoring = self.config.monitoring
        event = {
            "event_type": event_type,
            "device_id": self.config.device.id,
            "zone_id": monitoring.zone_id,
            "timestamp": status.timestamp.astimezone().isoformat(timespec="milliseconds"),
            "state": status.state.value,
            "occupancy_pct": round(status.occupancy_pct, 1),
            "occupancy_threshold_pct": monitoring.occupancy_threshold_pct,
            "persistence_s": round(status.persistence_s, 1),
            "persistence_threshold_s": monitoring.persistence_threshold_s,
            "projected_roi_px": projected_roi_px.as_dict(),
            **self.volume.as_event_dict(),
            "depth_delta_mm": monitoring.depth_delta_mm,
        }

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return event
