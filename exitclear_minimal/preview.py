from __future__ import annotations

import cv2
import numpy as np

from .config import AppConfig, RoiPx
from .depth_source import FramePacket
from .state_machine import State


class LivePreview:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.enabled = config.output.live_view
        self.show_mask = config.output.show_occupied_mask
        self.window_name = "ExitClear Minimal"
        self.mask_window_name = "ExitClear Occupied Mask"

    def show(
        self,
        packet: FramePacket,
        state: State,
        occupancy_pct: float = 0.0,
        persistence_s: float = 0.0,
        occupied_mask: np.ndarray | None = None,
        roi_px: RoiPx | None = None,
        anchor_label: str | None = None,
        baseline_progress: int | None = None,
        baseline_total: int | None = None,
    ) -> bool:
        if not self.enabled:
            return True

        frame = self._display_frame(packet)
        depth_shape = packet.depth_frame.shape[:2]
        roi = (
            self._scale_roi(roi_px, depth_shape, frame.shape[:2])
            if roi_px is not None and roi_px.x_max > roi_px.x_min
            else None
        )

        if roi is not None:
            cv2.rectangle(
                frame,
                (roi.x_min, roi.y_min),
                (roi.x_max, roi.y_max),
                (0, 220, 255),
                2,
            )
        self._draw_overlay(
            frame=frame,
            state=state,
            occupancy_pct=occupancy_pct,
            persistence_s=persistence_s,
            anchor_label=anchor_label,
            baseline_progress=baseline_progress,
            baseline_total=baseline_total,
        )

        cv2.imshow(self.window_name, frame)
        if self.show_mask:
            self._show_mask(packet.depth_frame.shape[:2], occupied_mask, roi_px)

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

    def _draw_overlay(
        self,
        frame: np.ndarray,
        state: State,
        occupancy_pct: float,
        persistence_s: float,
        anchor_label: str | None,
        baseline_progress: int | None,
        baseline_total: int | None,
    ) -> None:
        monitoring = self.config.monitoring
        lines = [
            f"State: {state.value}",
            f"Occupancy: {occupancy_pct:.1f}% / {monitoring.occupancy_threshold_pct:.1f}%",
            f"Persistence: {persistence_s:.1f}s / {monitoring.persistence_threshold_s:.1f}s",
        ]
        if anchor_label:
            lines.append(f"Anchor: {anchor_label}")
        if baseline_progress is not None and baseline_total is not None:
            lines.append(f"Calibrating baseline: {baseline_progress} / {baseline_total}")
        lines.append("Press q to quit")

        overlay = frame.copy()
        height = 28 + 24 * len(lines)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (12, 28 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _show_mask(
        self,
        depth_shape: tuple[int, int],
        occupied_mask: np.ndarray | None,
        roi_px: RoiPx | None,
    ) -> None:
        if occupied_mask is None:
            mask = np.zeros(depth_shape, dtype=np.uint8)
        else:
            mask = (occupied_mask.astype(np.uint8) * 255)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if roi_px is not None and roi_px.x_max > roi_px.x_min:
            roi = roi_px.clipped(width=depth_shape[1], height=depth_shape[0])
            cv2.rectangle(
                mask_bgr,
                (roi.x_min, roi.y_min),
                (roi.x_max, roi.y_max),
                (0, 220, 255),
                1,
            )
        cv2.imshow(self.mask_window_name, mask_bgr)

    @staticmethod
    def _scale_roi(
        roi: RoiPx, depth_shape: tuple[int, int], display_shape: tuple[int, int]
    ) -> RoiPx:
        depth_h, depth_w = depth_shape
        display_h, display_w = display_shape
        sx = display_w / max(1, depth_w)
        sy = display_h / max(1, depth_h)
        return RoiPx(
            x_min=int(round(roi.x_min * sx)),
            y_min=int(round(roi.y_min * sy)),
            x_max=int(round(roi.x_max * sx)),
            y_max=int(round(roi.y_max * sy)),
        ).clipped(width=display_w, height=display_h)
