from __future__ import annotations

from datetime import datetime

from .models import (
    ComplianceState,
    ComplianceStatus,
    OccupancyResult,
    ObstacleTrack,
    ZoneConfig,
)


class ComplianceStateMachine:
    def __init__(self, zone: ZoneConfig) -> None:
        self.zone = zone
        self.state = ComplianceState.COMPLIANT
        self._blocked_since: datetime | None = None
        self._clear_latched = False

    def update(
        self,
        timestamp: datetime,
        occupancy: OccupancyResult,
        obstacles: list[ObstacleTrack],
    ) -> tuple[ComplianceStatus, ComplianceState | None]:
        previous_state = self.state
        blocked = (
            occupancy.current_occupancy_pct
            >= self.zone.occupancy.min_occupied_pct_watch
            and occupancy.measured_free_width_mm < self.zone.required_clear_width_mm
        )
        watched = (
            occupancy.current_occupancy_pct
            >= self.zone.occupancy.min_occupied_pct_watch
        )

        if blocked:
            if self._blocked_since is None:
                self._blocked_since = timestamp
            persistence_s = max(0.0, (timestamp - self._blocked_since).total_seconds())
            self._clear_latched = False
            if persistence_s == 0.0 and previous_state == ComplianceState.COMPLIANT:
                next_state = ComplianceState.WATCH
            elif persistence_s < self.zone.persistence_threshold_s:
                next_state = ComplianceState.BLOCKED_PENDING
            else:
                next_state = ComplianceState.VIOLATION
        elif watched:
            self._blocked_since = None
            self._clear_latched = False
            persistence_s = 0.0
            next_state = ComplianceState.WATCH
        else:
            self._blocked_since = None
            persistence_s = 0.0
            if previous_state in {
                ComplianceState.WATCH,
                ComplianceState.BLOCKED_PENDING,
                ComplianceState.VIOLATION,
            } and not self._clear_latched:
                next_state = ComplianceState.CLEARED
                self._clear_latched = True
            else:
                next_state = ComplianceState.COMPLIANT
                self._clear_latched = False

        reason = occupancy.reason
        if next_state == ComplianceState.VIOLATION:
            reason = "persistent_clearance_reduction"
        elif next_state == ComplianceState.CLEARED:
            reason = "clearance_restored"

        self.state = next_state
        status = ComplianceStatus(
            timestamp=timestamp,
            zone_id=self.zone.id,
            state=next_state,
            severity=occupancy.severity,
            required_clear_width_mm=self.zone.required_clear_width_mm,
            measured_free_width_mm=occupancy.measured_free_width_mm,
            occupancy_pct=occupancy.occupancy_pct,
            persistence_s=persistence_s,
            reason=reason,
            obstacles=obstacles,
            zone_pixel_count=occupancy.zone_pixel_count,
            occupied_pixel_count=occupancy.occupied_pixel_count,
            occupied_bins=occupancy.occupied_bins,
            baseline_ready=occupancy.baseline_ready,
            depth_valid_pct=occupancy.depth_valid_pct,
        )
        changed_from = previous_state if previous_state != next_state else None
        return status, changed_from
