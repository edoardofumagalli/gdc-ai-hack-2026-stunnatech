from __future__ import annotations

import json
from pathlib import Path

from .models import ComplianceEvent, ComplianceState, ComplianceStatus, ZoneConfig


class EventLogger:
    def __init__(
        self,
        path: str | Path,
        device_id: str,
        zone: ZoneConfig,
        privacy_mode: bool,
        reset: bool = True,
    ) -> None:
        self.path = Path(path)
        self.device_id = device_id
        self.zone = zone
        self.privacy_mode = privacy_mode
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_text("", encoding="utf-8")

    def emit_state_change(
        self, status: ComplianceStatus, previous_state: ComplianceState
    ) -> ComplianceEvent:
        event = ComplianceEvent(
            event_type="compliance_state_change",
            device_id=self.device_id,
            timestamp=status.timestamp,
            zone_id=status.zone_id,
            zone_type=self.zone.type,
            previous_state=previous_state,
            state=status.state,
            severity=status.severity,
            reason=status.reason,
            required_clear_width_mm=status.required_clear_width_mm,
            measured_free_width_mm=status.measured_free_width_mm,
            occupancy_pct=status.occupancy_pct,
            persistence_s=status.persistence_s,
            obstacles=status.obstacles,
            privacy_mode=self.privacy_mode,
            raw_video_streamed=False,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        return event

    def read_events(self, limit: int | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            lines = lines[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
