from __future__ import annotations

import argparse
import time
from pathlib import Path

from fake_source import FakeDepthSource
from exitclear.config import load_config
from exitclear.models import ComplianceEvent, ComplianceState, ComplianceStatus
from exitclear.runtime import ExitClearRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExitClear backend compliance engine")
    parser.add_argument("--source", choices=["fake", "oak"], default="fake")
    parser.add_argument("--config", default="config/zones.yaml")
    parser.add_argument("--baseline-frames", type=int, default=20)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--append-events", action="store_true")
    parser.add_argument(
        "--verbose-status",
        action="store_true",
        help="Print every frame status instead of only state changes.",
    )
    parser.add_argument("--server", action="store_true", help="Run the HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="Optional OAK MXID/IP for --source oak")
    parser.add_argument("--oak-width", type=int, default=640)
    parser.add_argument("--oak-height", type=int, default=400)
    parser.add_argument("--oak-fps", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)
    zone = config.zones[0]

    if args.source == "oak":
        from oak_pipeline import OakDepthSource

        source = OakDepthSource(
            device_id=args.device,
            frame_shape=(args.oak_width, args.oak_height),
            fps=args.oak_fps,
        )
    else:
        source = FakeDepthSource()

    runtime = ExitClearRuntime(
        root=root,
        config=config,
        source=source,
        source_name=args.source,
        baseline_frames=args.baseline_frames,
        append_events=args.append_events,
        status_change_callback=print_status_change,
    )
    runtime.calibrate_baseline()

    print(
        f"ExitClear source={args.source} zone={zone.id} "
        f"baseline_frames={args.baseline_frames} events={runtime.event_log_path}"
    )
    if args.server:
        from server import run_server

        runtime.start_background()
        run_server(runtime=runtime, host=args.host, port=args.port)
        return

    printed_initial_status = False
    try:
        for index, packet in enumerate(source.frames(), start=1):
            if args.frame_limit is not None and index > args.frame_limit:
                break

            status, _, previous_state = runtime.process_packet(packet)
            if previous_state is None and (
                args.verbose_status or not printed_initial_status
            ):
                print(format_status_line("status", status, packet.scenario))
                printed_initial_status = True
            if args.real_time:
                time.sleep(1.0 / max(1.0, float(source.fps)))
    except KeyboardInterrupt:
        print("ExitClear stopped.")
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()


def print_status_change(
    status: ComplianceStatus,
    event: ComplianceEvent,
    previous_state: ComplianceState,
    scenario: str,
) -> None:
    print(format_status_line("state_change", status, scenario, previous_state))


def format_status_line(
    prefix: str,
    status: ComplianceStatus,
    scenario: str,
    previous_state: ComplianceState | None = None,
) -> str:
    if previous_state is None:
        transition = status.state.value
    else:
        transition = f"{previous_state.value}->{status.state.value}"
    return (
        f"{prefix} {transition} "
        f"scenario={scenario} "
        f"severity={status.severity:.2f} "
        f"occupancy={status.occupancy_pct:.1f}% "
        f"free_width={status.measured_free_width_mm}mm "
        f"persistence={status.persistence_s:.1f}s "
        f"reason={status.reason} "
        f"occupied_px={status.occupied_pixel_count} "
        f"bins={status.occupied_bins}"
    )


if __name__ == "__main__":
    main()
