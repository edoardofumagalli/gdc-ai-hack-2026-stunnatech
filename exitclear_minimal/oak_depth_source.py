from __future__ import annotations

from datetime import datetime
from typing import Iterator

import numpy as np

from .config import AppConfig
from .depthai_helpers import configure_stereo, stereo_preset
from .depth_source import FramePacket
from .earthquake import EarthquakeDetector


class OakDepthSource:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.fps = config.monitoring.fps
        self.frame_size = (
            config.monitoring.frame_width,
            config.monitoring.frame_height,
        )
        self._running = True

    def frames(self) -> Iterator[FramePacket]:
        try:
            import depthai as dai
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "depthai is required. Install requirements.txt first."
            ) from exc

        device = dai.Device()
        if not device.setIrLaserDotProjectorIntensity(1):
            print(
                "Failed to set IR laser projector intensity. "
                "The device may not support this feature."
            )

        with dai.Pipeline(device) as pipeline:
            print("Creating OAK pipeline...")
            platform = device.getPlatform()
            calibration = device.readCalibration()
            intrinsics = np.array(
                calibration.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_A,
                    self.frame_size[0],
                    self.frame_size[1],
                ),
                dtype=np.float32,
            )

            rgb_cam = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_A
            )
            left_cam = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_B
            )
            right_cam = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_C
            )

            rgb_out = rgb_cam.requestOutput(
                self.frame_size, type=dai.ImgFrame.Type.BGR888i, fps=self.fps
            )
            left_out = left_cam.requestOutput(
                self.frame_size, type=dai.ImgFrame.Type.NV12, fps=self.fps
            )
            right_out = right_cam.requestOutput(
                self.frame_size, type=dai.ImgFrame.Type.NV12, fps=self.fps
            )

            stereo = pipeline.create(dai.node.StereoDepth).build(
                left=left_out,
                right=right_out,
                presetMode=stereo_preset(dai, self.config.monitoring.stereo_preset),
            )
            configure_stereo(
                stereo,
                dai,
                subpixel=self.config.monitoring.subpixel,
                median_filter_name=self.config.monitoring.median_filter,
            )

            # OAK 4 / RVC4 uses ImageAlign for depth-to-RGB alignment.
            if platform == dai.Platform.RVC4:
                align = pipeline.create(dai.node.ImageAlign)
                stereo.depth.link(align.input)
                rgb_out.link(align.inputAlignTo)
                depth_out = align.outputAligned
            else:
                stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
                try:
                    stereo.setOutputSize(*self.frame_size)
                except AttributeError:
                    pass
                depth_out = stereo.depth

            earthquake_detector = None
            imu_queue = None
            if self.config.earthquake.enabled:
                earthquake_detector = EarthquakeDetector(self.config.earthquake)
                imu = pipeline.create(dai.node.IMU)
                imu.enableIMUSensor(
                    dai.IMUSensor.ACCELEROMETER_RAW,
                    self.config.earthquake.sample_rate_hz,
                )
                imu.setBatchReportThreshold(
                    self.config.earthquake.batch_threshold
                )
                imu.setMaxBatchReports(10)
                imu_queue = imu.out.createOutputQueue(blocking=False, maxSize=4)

            rgb_queue = rgb_out.createOutputQueue(blocking=False, maxSize=1)
            depth_queue = depth_out.createOutputQueue(blocking=True, maxSize=1)

            print("OAK pipeline created.")
            pipeline.start()

            last_rgb: np.ndarray | None = None
            while self._running and pipeline.isRunning():
                depth_msg = depth_queue.get()
                if rgb_queue.has():
                    last_rgb = rgb_queue.get().getCvFrame()

                earthquake_triggered = False
                earthquake_vibration = None
                if imu_queue is not None and earthquake_detector is not None:
                    while imu_queue.has():
                        imu_data = imu_queue.get()
                        for packet in imu_data.packets:
                            accel = packet.acceleroMeter
                            reading = earthquake_detector.update(
                                accel.x, accel.y, accel.z
                            )
                            earthquake_vibration = reading.vibration_mps2
                            earthquake_triggered = (
                                earthquake_triggered or reading.triggered
                            )

                depth = depth_msg.getFrame()
                if depth.ndim == 3:
                    depth = depth.squeeze()

                yield FramePacket(
                    timestamp=datetime.now().astimezone(),
                    rgb_frame=last_rgb,
                    depth_frame=depth.astype(np.float32, copy=False),
                    intrinsics=intrinsics,
                    earthquake_triggered=earthquake_triggered,
                    earthquake_vibration_mps2=earthquake_vibration,
                )

    def close(self) -> None:
        self._running = False
