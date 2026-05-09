from __future__ import annotations

from typing import Any

import numpy as np

from .models import BoundsCameraMm, ComplianceStatus


def render_preview_jpeg(
    depth_mm: np.ndarray,
    rgb: np.ndarray | None,
    occupied_mask: np.ndarray | None,
    status: ComplianceStatus,
    bounds: BoundsCameraMm,
    event_active: bool,
) -> bytes | None:
    try:
        import cv2
    except Exception:
        return None

    frame = _rgb_preview(rgb, cv2) if rgb is not None else None
    if frame is None:
        frame = _depth_preview(depth_mm, bounds, cv2)

    if occupied_mask is not None and occupied_mask.shape[:2] == frame.shape[:2]:
        frame = _overlay_occupied(frame, occupied_mask, cv2)

    _draw_hud(frame, status, event_active, cv2)
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    )
    if not ok:
        return None
    return encoded.tobytes()


def _rgb_preview(rgb: np.ndarray, cv2: Any) -> np.ndarray | None:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    frame = rgb[:, :, :3]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _depth_preview(depth_mm: np.ndarray, bounds: BoundsCameraMm, cv2: Any) -> np.ndarray:
    depth = depth_mm.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > bounds.z_min) & (depth < bounds.z_max)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth, bounds.z_min, bounds.z_max)
        normalized = (
            (1.0 - (clipped - bounds.z_min) / max(1, bounds.z_max - bounds.z_min))
            * 255.0
        ).astype(np.uint8)
        normalized[~valid] = 0
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def _overlay_occupied(frame: np.ndarray, occupied_mask: np.ndarray, cv2: Any) -> np.ndarray:
    overlay = frame.copy()
    overlay[occupied_mask] = (0, 0, 255)
    return cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)


def _draw_hud(
    frame: np.ndarray,
    status: ComplianceStatus,
    event_active: bool,
    cv2: Any,
) -> None:
    color = _state_color(status.state.value)
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1] - 1, frame.shape[0] - 1),
        color,
        2,
    )

    lines = [
        f"ExitClear {status.state.value}",
        f"occupancy {status.occupancy_pct:.1f}%  width {status.measured_free_width_mm}mm",
        f"persistence {status.persistence_s:.1f}s  valid depth {status.depth_valid_pct:.1f}%",
    ]
    x, y = 12, 22
    line_height = 22
    box_height = line_height * len(lines) + 14
    cv2.rectangle(frame, (6, 6), (520, box_height), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if event_active:
        cv2.circle(frame, (frame.shape[1] - 24, 24), 10, (0, 0, 255), -1)
        cv2.putText(
            frame,
            "EVENT",
            (frame.shape[1] - 96, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _state_color(state: str) -> tuple[int, int, int]:
    if state == "VIOLATION":
        return (0, 0, 255)
    if state == "BLOCKED_PENDING":
        return (0, 165, 255)
    if state in {"WATCH", "CLEARED"}:
        return (0, 255, 255)
    return (0, 220, 0)
