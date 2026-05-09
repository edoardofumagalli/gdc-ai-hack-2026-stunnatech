from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig
from .events import EventLogger
from .models import ComplianceEvent, ComplianceState, ComplianceStatus, FramePacket
from .occupancy import OccupancyEngine
from .state_machine import ComplianceStateMachine
from .tracker import SimpleCentroidTracker

StatusChangeCallback = Callable[
    [ComplianceStatus, ComplianceEvent, ComplianceState, str], None
]


class ExitClearRuntime:
    def __init__(
        self,
        root: Path,
        config: AppConfig,
        source: Any,
        source_name: str,
        baseline_frames: int,
        append_events: bool,
        status_change_callback: StatusChangeCallback | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.zone = config.zones[0]
        self.source = source
        self.source_name = source_name
        self.baseline_frames = baseline_frames
        self.status_change_callback = status_change_callback

        self.event_logger = EventLogger(
            root / config.outputs.event_log_path,
            device_id=config.device.id,
            zone=self.zone,
            privacy_mode=config.device.privacy_mode_default,
            reset=not append_events,
        )

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline_depth: np.ndarray | None = None
        self._occupancy_engine: OccupancyEngine | None = None
        self._tracker: SimpleCentroidTracker | None = None
        self._state_machine: ComplianceStateMachine | None = None
        self._latest_status: ComplianceStatus | None = None
        self._latest_scenario: str | None = None

    @property
    def event_log_path(self) -> Path:
        return self.event_logger.path

    def calibrate_baseline(self) -> ComplianceStatus:
        packets = self.source.calibration_frames(self.baseline_frames)
        baseline_depth = np.median(
            np.stack([packet.depth_mm for packet in packets], axis=0), axis=0
        ).astype(np.uint16)
        with self._lock:
            self._baseline_depth = baseline_depth
            self._occupancy_engine = OccupancyEngine(self.zone, baseline_depth.shape)
            self._tracker = SimpleCentroidTracker(self._occupancy_engine.geometry)
            self._state_machine = ComplianceStateMachine(self.zone)
            status, _ = self._process_packet_locked(packets[-1])
            return status

    def process_packet(
        self, packet: FramePacket
    ) -> tuple[ComplianceStatus, ComplianceEvent | None, ComplianceState | None]:
        with self._lock:
            status, previous_state = self._process_packet_locked(packet)
            event = None
            if previous_state is not None:
                event = self.event_logger.emit_state_change(status, previous_state)
            self._latest_scenario = packet.scenario

        if event is not None and previous_state is not None:
            self._notify_status_change(status, event, previous_state, packet.scenario)
        return status, event, previous_state

    def latest_status_dict(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._latest_status is None else self._latest_status.to_dict()

    def events(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return self.event_logger.read_events(limit=limit)

    def health(self) -> dict[str, Any]:
        with self._lock:
            state = (
                None
                if self._latest_status is None
                else self._latest_status.state.value
            )
            return {
                "ok": True,
                "source": self.source_name,
                "device_id": self.config.device.id,
                "zone_id": self.zone.id,
                "state": state,
                "event_log_path": str(self.event_log_path),
            }

    def start_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run_forever, name="exitclear-runtime", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        close = getattr(self.source, "close", None)
        if callable(close):
            close()

    def run_forever(self) -> None:
        while not self._stop_event.is_set():
            for packet in self.source.frames():
                if self._stop_event.is_set():
                    return
                self.process_packet(packet)
                time.sleep(1.0 / max(1.0, float(self.source.fps)))

    def _process_packet_locked(
        self, packet: FramePacket
    ) -> tuple[ComplianceStatus, ComplianceState | None]:
        if (
            self._baseline_depth is None
            or self._occupancy_engine is None
            or self._tracker is None
            or self._state_machine is None
        ):
            raise RuntimeError("Baseline has not been calibrated")

        occupancy = self._occupancy_engine.evaluate(
            packet.depth_mm, self._baseline_depth
        )
        obstacles = self._tracker.update(occupancy.components, packet.timestamp)
        status, previous_state = self._state_machine.update(
            packet.timestamp, occupancy, obstacles
        )
        self._latest_status = status
        self._latest_scenario = packet.scenario
        return status, previous_state

    def _notify_status_change(
        self,
        status: ComplianceStatus,
        event: ComplianceEvent,
        previous_state: ComplianceState,
        scenario: str,
    ) -> None:
        if self.status_change_callback is not None:
            self.status_change_callback(status, event, previous_state, scenario)
