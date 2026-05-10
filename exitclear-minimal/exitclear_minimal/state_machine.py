from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .config import MonitoringConfig


class State(str, Enum):
    NO_BASELINE = "NO_BASELINE"
    CLEAR = "CLEAR"
    OCCUPIED_PENDING = "OCCUPIED_PENDING"
    TRIGGERED = "TRIGGERED"


@dataclass(frozen=True)
class StateStatus:
    timestamp: datetime
    state: State
    occupancy_pct: float
    persistence_s: float


class OccupancyStateMachine:
    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self.state = State.NO_BASELINE
        self._above_threshold_since: datetime | None = None

    def update(
        self, timestamp: datetime, occupancy_pct: float
    ) -> tuple[StateStatus, State | None]:
        previous_state = self.state
        above_threshold = occupancy_pct >= self.config.occupancy_threshold_pct

        if not above_threshold:
            self._above_threshold_since = None
            next_state = State.CLEAR
            persistence_s = 0.0
        else:
            if self._above_threshold_since is None:
                self._above_threshold_since = timestamp
            persistence_s = max(
                0.0, (timestamp - self._above_threshold_since).total_seconds()
            )
            if persistence_s >= self.config.persistence_threshold_s:
                next_state = State.TRIGGERED
            else:
                next_state = State.OCCUPIED_PENDING

        self.state = next_state
        changed_from = previous_state if previous_state != next_state else None
        return (
            StateStatus(
                timestamp=timestamp,
                state=next_state,
                occupancy_pct=occupancy_pct,
                persistence_s=persistence_s,
            ),
            changed_from,
        )
