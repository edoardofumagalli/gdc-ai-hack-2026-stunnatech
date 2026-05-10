from __future__ import annotations

import cv2
import numpy as np

from .config import AppConfig
from .depth_source import FramePacket
from .state_machine import State


class LivePreview:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.enabled = config.output.live_view
        self.show_mask = config.output.show_occupied_mask
        self.window_name = "ExitClear Minimal"

    def show(
        self,
        packet: FramePacket,
        state: State,
        occupancy_pct: float = 0.0,
        persistence_s: float = 0.0,
        occupied_mask: np.ndarray | None = None,
        volume_corners_px: list[tuple[int, int] | None] | None = None,
        anchor_px: tuple[int, int] | None = None,
        volume_center_px: tuple[int, int] | None = None,
        anchor_label: str | None = None,
        baseline_progress: int | None = None,
        baseline_total: int | None = None,
    ) -> bool:
        if not self.enabled:
            return True

        frame = self._display_frame(packet)
        depth_shape = packet.depth_frame.shape[:2]
        if self.show_mask:
            self._draw_occupied_mask_overlay(frame, occupied_mask)
        self._draw_volume_cuboid(frame, volume_corners_px, depth_shape)
        self._draw_debug_point(
            frame,
            anchor_px,
            depth_shape,
            label="anchor",
            color=(255, 0, 255),
        )
        self._draw_debug_point(
            frame,
            volume_center_px,
            depth_shape,
            label="volume center",
            color=(255, 255, 0),
        )
        self._draw_overlay(
            frame=frame,
            state=state,
            occupancy_pct=occupancy_pct,
            persistence_s=persistence_s,
            people_count=packet.people_count,
            anchor_label=anchor_label,
            baseline_progress=baseline_progress,
            baseline_total=baseline_total,
        )

        cv2.imshow(self.window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def close(self) -> None:
        if self.enabled:
            cv2.destroyAllWindows()

    def _display_frame(self, packet: FramePacket) -> np.ndarray:
        if packet.rgb_frame is not None:
            frame = packet.rgb_frame
            if frame.ndim == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            return frame.copy()

        depth = packet.depth_frame.astype(np.float32, copy=False)
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            normalized = np.zeros(depth.shape, dtype=np.uint8)
        else:
            near = np.nanpercentile(depth[valid], 5)
            far = np.nanpercentile(depth[valid], 95)
            normalized = np.clip((depth - near) / max(1.0, far - near), 0.0, 1.0)
            normalized = (255 - normalized * 255).astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

    def _draw_occupied_mask_overlay(
        self,
        frame: np.ndarray,
        occupied_mask: np.ndarray | None,
    ) -> None:
        if occupied_mask is None:
            return

        mask = occupied_mask.astype(np.uint8, copy=False)
        display_h, display_w = frame.shape[:2]
        if mask.shape[:2] != (display_h, display_w):
            mask = cv2.resize(
                mask,
                (display_w, display_h),
                interpolation=cv2.INTER_NEAREST,
            )

        occupied = mask.astype(bool)
        if not occupied.any():
            return

        overlay = frame.copy()
        overlay[occupied] = (0, 0, 255)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        contour_mask = (occupied.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(
            contour_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    def _draw_volume_cuboid(
        self,
        frame: np.ndarray,
        corners: list[tuple[int, int] | None] | None,
        depth_shape: tuple[int, int],
    ) -> None:
        if (
            corners is None
            or len(corners) != 8
            or any(point is None for point in corners)
        ):
            return

        points = [
            self._scale_point(point, depth_shape, frame.shape[:2])
            for point in corners
        ]

        # Corner order from MonitoredVolume.corners():
        # 0..3 = near face, toward the camera (z_min)
        # 4..7 = far face, on the sign/door side (z_max)
        near_face = [points[index] for index in (0, 1, 3, 2)]
        far_face = [points[index] for index in (4, 5, 7, 6)]

        self._fill_poly(frame, far_face, color=(80, 220, 80), alpha=0.14)
        self._fill_poly(frame, near_face, color=(0, 150, 255), alpha=0.18)

        far_edges = ((4, 5), (5, 7), (7, 6), (6, 4))
        near_edges = ((0, 1), (1, 3), (3, 2), (2, 0))
        depth_edges = ((0, 4), (1, 5), (2, 6), (3, 7))

        for start, end in depth_edges:
            cv2.line(frame, points[start], points[end], (0, 255, 255), 2, cv2.LINE_AA)
        for start, end in far_edges:
            cv2.line(frame, points[start], points[end], (80, 220, 80), 2, cv2.LINE_AA)
        for start, end in near_edges:
            cv2.line(frame, points[start], points[end], (0, 150, 255), 3, cv2.LINE_AA)

        far_center = _mean_point(far_face)
        near_center = _mean_point(near_face)
        cv2.arrowedLine(
            frame,
            far_center,
            near_center,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.12,
        )

        cv2.putText(
            frame,
            "sign plane",
            (far_center[0] + 8, far_center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 220, 80),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "toward camera",
            (near_center[0] + 8, near_center[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for point in points:
            cv2.circle(frame, point, 4, (0, 255, 255), -1, cv2.LINE_AA)

    def _draw_debug_point(
        self,
        frame: np.ndarray,
        point: tuple[int, int] | None,
        depth_shape: tuple[int, int],
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        if point is None:
            return

        depth_h, depth_w = depth_shape
        display_h, display_w = frame.shape[:2]
        sx = display_w / max(1, depth_w)
        sy = display_h / max(1, depth_h)
        x = int(round(point[0] * sx))
        y = int(round(point[1] * sy))

        cv2.drawMarker(
            frame,
            (x, y),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
        cv2.putText(
            frame,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    def _draw_overlay(
        self,
        frame: np.ndarray,
        state: State,
        occupancy_pct: float,
        persistence_s: float,
        people_count: float | None,
        anchor_label: str | None,
        baseline_progress: int | None,
        baseline_total: int | None,
    ) -> None:
        monitoring = self.config.monitoring
        lines = [
            f"State: {state.value}",
            f"Occupancy: {occupancy_pct:.1f}% / {monitoring.occupancy_threshold_pct:.1f}%",
            f"Persistence: {persistence_s:.1f}s / {monitoring.persistence_threshold_s:.1f}s",
            f"People: {_format_people_count(people_count)}",
        ]
        if anchor_label:
            lines.append(f"Anchor: {anchor_label}")
        if baseline_progress is not None and baseline_total is not None:
            lines.append(f"Calibrating baseline: {baseline_progress} / {baseline_total}")
        lines.append("Press q to quit")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        padding = 10
        line_gap = 8
        sizes = [
            cv2.getTextSize(line, font, font_scale, thickness)[0]
            for line in lines
        ]
        text_width = max(width for width, _ in sizes)
        line_height = max(height for _, height in sizes)
        box_width = min(frame.shape[1], text_width + padding * 2)
        box_height = (
            padding * 2
            + len(lines) * line_height
            + (len(lines) - 1) * line_gap
        )
        x0 = 8
        y0 = max(8, frame.shape[0] - box_height - 8)
        x1 = min(frame.shape[1] - 1, x0 + box_width)
        y1 = min(frame.shape[0] - 1, y0 + box_height)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (
                    x0 + padding,
                    y0 + padding + line_height + index * (line_height + line_gap),
                ),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    @staticmethod
    def _scale_point(
        point: tuple[int, int],
        depth_shape: tuple[int, int],
        display_shape: tuple[int, int],
    ) -> tuple[int, int]:
        depth_h, depth_w = depth_shape
        display_h, display_w = display_shape
        sx = display_w / max(1, depth_w)
        sy = display_h / max(1, depth_h)
        return int(round(point[0] * sx)), int(round(point[1] * sy))

    @staticmethod
    def _fill_poly(
        frame: np.ndarray,
        points: list[tuple[int, int]],
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        overlay = frame.copy()
        polygon = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _mean_point(points: list[tuple[int, int]]) -> tuple[int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys)))


def _format_people_count(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}"
