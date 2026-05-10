from __future__ import annotations

import argparse
from pathlib import Path
import time

from exitclear_minimal.api import ApiServer
from exitclear_minimal.audio import EmergencyAudioService
from exitclear_minimal.baseline import BaselineBuilder
from exitclear_minimal.config import load_config
from exitclear_minimal.events import EventWriter
from exitclear_minimal.oak_depth_source import OakDepthSource
from exitclear_minimal.occupancy import OccupancyMonitor
from exitclear_minimal.preview import LivePreview
from exitclear_minimal.sign_detection import run_sign_detection
from exitclear_minimal.state_machine import OccupancyStateMachine, State
from exitclear_minimal.state_machine import StateStatus
from exitclear_minimal.status_store import DashboardStatusStore
from exitclear_minimal.volume import MonitoredVolume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExitClear minimal depth MVP")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml. Defaults to config.yaml in the project root.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional override for sign_detection.model_path.",
    )
    parser.add_argument(
        "--api-host",
        default="0.0.0.0",
        help="Host for the status API. Defaults to 0.0.0.0.",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for the status API. Defaults to 8000.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    config_root = config_path.parent

    events_path = Path(config.output.events_path)
    if not events_path.is_absolute():
        events_path = config_root / events_path

    model_path = Path(args.model or config.sign_detection.model_path)
    if not model_path.is_absolute():
        model_path = config_root / model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Sign detection model not found: {model_path}. "
            "Update sign_detection.model_path in config.yaml."
        )
    anchor_label, anchor = run_sign_detection(config.sign_detection, model_path)
    volume = MonitoredVolume.from_anchor(
        anchor_label=anchor_label,
        anchor=anchor,
        config=config.monitoring.volume,
    )
    print("\n=== Monitoring volume ===")
    print(f"Anchor [{anchor_label}] {volume.anchor.as_dict()}")
    print(f"Bounds {volume.bounds.as_dict()}")

    source = OakDepthSource(config)
    preview = LivePreview(config)
    baseline = BaselineBuilder(
        frame_count=config.monitoring.baseline_frames,
        min_valid_depth_mm=config.monitoring.min_valid_depth_mm,
    )
    occupancy_monitor = OccupancyMonitor(config.monitoring, volume)
    state_machine = OccupancyStateMachine(config.monitoring)
    event_writer = EventWriter(events_path, config, volume)
    status_store = DashboardStatusStore(config, anchor_label)
    audio_service = EmergencyAudioService(config.audio, config_root)
    api_server = ApiServer(
        status_store,
        host=args.api_host,
        port=args.api_port,
        audio_dir=audio_service.output_dir if audio_service.enabled else None,
    )
    api_server.start()
    print(f"Status API available at {api_server.url}")

    baseline_depth = None
    last_status_print = 0.0

    try:
        for packet in source.frames():
            depth_shape = packet.depth_frame.shape[:2]
            volume_corners_px = volume.projected_corners(
                depth_shape, packet.intrinsics
            )
            anchor_px = volume.project_point(
                volume.anchor, depth_shape, packet.intrinsics
            )
            volume_center_px = volume.project_point(
                volume.center(), depth_shape, packet.intrinsics
            )
            if packet.earthquake_triggered:
                newly_triggered = status_store.trigger_earthquake(
                    timestamp=packet.timestamp,
                    vibration_mps2=packet.earthquake_vibration_mps2,
                )
                if newly_triggered:
                    vibration = packet.earthquake_vibration_mps2
                    audio_service.request_earthquake_audio(
                        event_label="earthquake",
                        exit_id=status_store.exit_identity["id"],
                        exit_name=status_store.exit_identity["name"],
                        public_url_for_filename=api_server.audio_url,
                        on_ready=lambda audio: status_store.set_earthquake_audio(
                            audio_url=audio.audio_url,
                            audio_sequence=audio.sequence_urls,
                            audio_pause_ms=audio.pause_ms,
                        ),
                    )
                    if vibration is None:
                        print("\nEarthquake detected: evacuation triggered.")
                    else:
                        print(
                            "\nEarthquake detected: "
                            f"vibration={vibration:.2f} m/s^2. "
                            "Evacuation triggered."
                        )

            if baseline_depth is None:
                baseline.add(packet.depth_frame)
                status_store.update(
                    StateStatus(
                        timestamp=packet.timestamp,
                        state=State.NO_BASELINE,
                        occupancy_pct=0.0,
                        persistence_s=0.0,
                    )
                )
                if config.output.print_status:
                    print(
                        f"Calibrating baseline: {baseline.progress} / "
                        f"{baseline.frame_count}",
                        end="\r",
                        flush=True,
                    )

                keep_running = preview.show(
                    packet=packet,
                    state=State.NO_BASELINE,
                    volume_corners_px=volume_corners_px,
                    anchor_px=anchor_px,
                    volume_center_px=volume_center_px,
                    anchor_label=anchor_label,
                    baseline_progress=baseline.progress,
                    baseline_total=baseline.frame_count,
                )
                if not keep_running:
                    break

                if baseline.ready:
                    baseline_depth = baseline.compute()
                    if config.output.print_status:
                        print(
                            f"\nBaseline ready from {baseline.frame_count} frames."
                        )
                continue

            result = occupancy_monitor.evaluate(
                packet.depth_frame, baseline_depth, packet.intrinsics
            )
            status, previous_state = state_machine.update(
                packet.timestamp, result.occupancy_pct
            )
            status_store.update(status)
            if previous_state is not None:
                event = event_writer.emit_state_change(
                    status, previous_state, result.roi_px
                )
                if event is not None:
                    print(f"\nEvent written: {event['event_type']} -> {events_path}")

            now = time.monotonic()
            if config.output.print_status and (
                now - last_status_print > 0.5 or previous_state is not None
            ):
                print(
                    "State: "
                    f"{status.state.value} | "
                    f"occupancy={status.occupancy_pct:.1f}% "
                    f"(raw={result.raw_occupancy_pct:.1f}%) | "
                    f"persistence={status.persistence_s:.1f}s | "
                    f"valid={result.valid_pixels} | "
                    f"occupied={result.occupied_pixels}"
                )
                last_status_print = now

            keep_running = preview.show(
                packet=packet,
                state=status.state,
                occupancy_pct=status.occupancy_pct,
                persistence_s=status.persistence_s,
                occupied_mask=result.occupied_mask,
                volume_corners_px=volume_corners_px,
                anchor_px=anchor_px,
                volume_center_px=volume_center_px,
                anchor_label=anchor_label,
            )
            if not keep_running:
                break

    except KeyboardInterrupt:
        print("\nInterrupted, exiting cleanly.")
    finally:
        api_server.stop()
        source.close()
        preview.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
