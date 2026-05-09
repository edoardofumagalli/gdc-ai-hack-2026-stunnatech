from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np

from exitclear.models import FramePacket


class OakDepthSource:
    """Real OAK stereo-depth source using DepthAI v3.

    This intentionally emits only depth frames for Phase 2. Object detections and
    RGB preview stay out of the compliance path until the depth loop is stable.
    """

    def __init__(
        self,
        device_id: str | None = None,
        frame_shape: tuple[int, int] = (640, 400),
        fps: float = 5.0,
        queue_size: int = 4,
    ) -> None:
        try:
            import depthai as dai
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DepthAI is required for --source oak. Install project requirements "
                "and verify the OAK device is connected."
            ) from exc

        self.dai = dai
        self.device_id = device_id
        self.frame_shape = frame_shape
        self.fps = fps
        self.queue_size = queue_size
        self._validate_frame_shape()

        self.device = self._open_device()
        self.fx, self.fy = self._read_camera_intrinsics()
        try:
            self.device.setIrLaserDotProjectorIntensity(1.0)
        except Exception:
            pass
        self.pipeline = self.dai.Pipeline(self.device)
        self.depth_queue: Any | None = None
        self.rgb_queue: Any | None = None
        self._build_pipeline()
        self.pipeline.start()

        info = self.device.getDeviceInfo()
        print(f"OAK depth source connected: {info}")

    def calibration_frames(self, count: int) -> list[FramePacket]:
        return [self._read_depth_packet("oak_baseline") for _ in range(count)]

    def frames(self) -> Iterable[FramePacket]:
        while self.pipeline.isRunning():
            yield self._read_depth_packet("oak_depth")

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _open_device(self):
        if self.device_id:
            return self.dai.Device(self.dai.DeviceInfo(self.device_id))
        return self.dai.Device()

    def _build_pipeline(self) -> None:
        left_socket, right_socket = self._stereo_sockets()
        left = self.pipeline.create(self.dai.node.Camera).build(left_socket)
        right = self.pipeline.create(self.dai.node.Camera).build(right_socket)

        left_out = left.requestOutput(
            self.frame_shape,
            type=self.dai.ImgFrame.Type.NV12,
            fps=self.fps,
        )
        right_out = right.requestOutput(
            self.frame_shape,
            type=self.dai.ImgFrame.Type.NV12,
            fps=self.fps,
        )

        stereo = self.pipeline.create(self.dai.node.StereoDepth).build(
            left=left_out,
            right=right_out,
            presetMode=self.dai.node.StereoDepth.PresetMode.DEFAULT,
        )
        stereo.setLeftRightCheck(True)
        stereo.setRectification(True)

        self.depth_queue = stereo.depth.createOutputQueue(
            maxSize=self.queue_size,
            blocking=True,
        )
        self._try_build_rgb_preview()

    def _try_build_rgb_preview(self) -> None:
        try:
            color = self.pipeline.create(self.dai.node.Camera).build(
                self.dai.CameraBoardSocket.CAM_A
            )
            color_out = color.requestOutput(
                self.frame_shape,
                type=self.dai.ImgFrame.Type.RGB888i,
                fps=self.fps,
            )
            self.rgb_queue = color_out.createOutputQueue(maxSize=2, blocking=False)
        except Exception as exc:
            self.rgb_queue = None
            print(f"OAK RGB preview unavailable: {exc}")

    def _validate_frame_shape(self) -> None:
        width, height = self.frame_shape
        if width % 128 != 0:
            raise ValueError(
                "OAK StereoDepth input width must be divisible by 128 on this "
                f"device. Got width={width}. Try --oak-width 640 --oak-height 400."
            )
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid OAK frame shape: {self.frame_shape}")

    def _read_camera_intrinsics(self) -> tuple[float, float]:
        width, height = self.frame_shape
        try:
            calib = self.device.readCalibration()
            intrinsics = calib.getCameraIntrinsics(
                self.dai.CameraBoardSocket.CAM_B, width, height
            )
        except Exception:
            try:
                calib = self.device.readCalibration()
                intrinsics = calib.getCameraIntrinsics(self.dai.CameraBoardSocket.CAM_B)
            except Exception:
                return 500.0, 500.0
        return float(intrinsics[0][0]), float(intrinsics[1][1])

    def _stereo_sockets(self):
        fallback = (
            self.dai.CameraBoardSocket.CAM_B,
            self.dai.CameraBoardSocket.CAM_C,
        )
        try:
            features = self.device.getConnectedCameraFeatures()
        except Exception:
            return fallback

        mono_sockets = [
            feature.socket
            for feature in features
            if self.dai.CameraSensorType.MONO in feature.supportedTypes
        ]
        if fallback[0] in mono_sockets and fallback[1] in mono_sockets:
            return fallback
        if len(mono_sockets) >= 2:
            return mono_sockets[0], mono_sockets[1]
        return fallback

    def _read_depth_packet(self, scenario: str) -> FramePacket:
        if self.depth_queue is None:
            raise RuntimeError("Depth output queue has not been created")
        message = self.depth_queue.get()
        depth_mm = np.asarray(message.getFrame()).astype(np.uint16, copy=False)
        rgb = self._latest_rgb_frame()
        return FramePacket(
            timestamp=datetime.now().astimezone(),
            depth_mm=depth_mm,
            rgb=rgb,
            detections=[],
            scenario=scenario,
        )

    def _latest_rgb_frame(self) -> np.ndarray | None:
        if self.rgb_queue is None:
            return None
        latest = None
        while True:
            message = self.rgb_queue.tryGet()
            if message is None:
                break
            latest = message
        if latest is None:
            return None
        try:
            return np.asarray(latest.getCvFrame()).copy()
        except Exception:
            return None
