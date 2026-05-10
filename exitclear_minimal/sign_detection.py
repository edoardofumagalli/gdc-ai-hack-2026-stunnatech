from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from .config import SignDetectionConfig
from .depthai_helpers import configure_stereo, stereo_preset
from .volume import SpatialPoint

ANCHOR_SAMPLE_COUNT = 10


class DetectionStore:
    def __init__(self) -> None:
        self._data: dict[str, SpatialPoint] = {}

    def update(self, label: str, xyz: tuple[float, float, float]) -> None:
        self._data[label] = SpatialPoint(*xyz)

    def get(self, label: str) -> SpatialPoint | None:
        return self._data.get(label)

    def first(self) -> tuple[str, SpatialPoint] | None:
        for label, xyz in self._data.items():
            return label, xyz
        return None

    def all(self) -> dict[str, SpatialPoint]:
        return dict(self._data)

    def __repr__(self) -> str:
        lines = [
            f"  [{label}] X={point.x_mm:.0f} Y={point.y_mm:.0f} Z={point.z_mm:.0f} mm"
            for label, point in self._data.items()
        ]
        return "DetectionStore:\n" + "\n".join(lines)


def run_sign_detection(
    config: SignDetectionConfig, model_path: Path
) -> tuple[str, SpatialPoint]:
    store = DetectionStore()

    try:
        import depthai as dai
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "depthai is required for sign detection. Install requirements.txt first."
        ) from exc

    with dai.Pipeline() as pipeline:
        platform = pipeline.getDefaultDevice().getPlatform()

        camera_node = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A, sensorFps=config.fps
        )
        mono_left = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B, sensorFps=config.fps
        )
        mono_right = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C, sensorFps=config.fps
        )

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(stereo_preset(dai, config.stereo_preset))
        configure_stereo(
            stereo,
            dai,
            subpixel=config.subpixel,
            median_filter_name=config.median_filter,
        )
        if platform == dai.Platform.RVC2:
            stereo.setOutputSize(config.stereo_width, config.stereo_height)
        mono_left.requestOutput((config.stereo_width, config.stereo_height)).link(
            stereo.left
        )
        mono_right.requestOutput((config.stereo_width, config.stereo_height)).link(
            stereo.right
        )

        nn_archive = dai.NNArchive(str(model_path))
        detection_network = pipeline.create(dai.node.DetectionNetwork).build(
            camera_node, nn_archive
        )
        try:
            detection_network.setConfidenceThreshold(config.confidence_threshold)
        except AttributeError:
            pass
        label_map = detection_network.getClasses()

        rgb_queue = detection_network.passthrough.createOutputQueue()
        detections_queue = detection_network.out.createOutputQueue()

        if platform == dai.Platform.RVC4:
            align = pipeline.create(dai.node.ImageAlign)
            stereo.depth.link(align.input)
            detection_network.passthrough.link(align.inputAlignTo)
            depth_out = align.outputAligned
        else:
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            try:
                stereo.setOutputSize(config.stereo_width, config.stereo_height)
            except AttributeError:
                pass
            depth_out = stereo.depth

        depth_queue = depth_out.createOutputQueue()

        pipeline.start()

        device = pipeline.getDefaultDevice()
        calibration = device.readCalibration()
        intrinsics_cache = {}

        def get_intrinsics(socket, width: int, height: int):
            key = (socket, width, height)
            if key not in intrinsics_cache:
                intrinsics_cache[key] = calibration.getCameraIntrinsics(
                    socket, width, height
                )
            return intrinsics_cache[key]

        frame = None
        detections = []
        depth_frame = None
        start_time = time.monotonic()
        last_target_warning = start_time
        counter = 0
        should_stop = False
        selected: tuple[str, SpatialPoint] | None = None
        seen_labels: set[str] = set()
        anchor_samples: list[tuple[str, SpatialPoint]] = []

        def frame_norm(display_frame, bbox):
            norm_vals = np.full(len(bbox), display_frame.shape[0])
            norm_vals[::2] = display_frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

        def display_frame(name: str, display):
            color = (255, 0, 0)
            for det in detections:
                bbox = frame_norm(display, (det.xmin, det.ymin, det.xmax, det.ymax))
                label = label_map[det.label] if label_map else str(det.label)
                cv2.putText(
                    display,
                    label,
                    (bbox[0] + 10, bbox[1] + 20),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )
                cv2.putText(
                    display,
                    f"{int(det.confidence * 100)}%",
                    (bbox[0] + 10, bbox[1] + 40),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )
                xyz = store.get(label)
                if xyz is not None:
                    cv2.putText(
                        display,
                        f"X:{int(xyz.x_mm)} Y:{int(xyz.y_mm)} Z:{int(xyz.z_mm)}",
                        (bbox[0] + 10, bbox[1] + 60),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        255,
                    )
                cv2.rectangle(
                    display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2
                )
            cv2.imshow(name, display)

        print("=== Detecting emergency sign anchor ===")
        while pipeline.isRunning() and not should_stop:
            in_rgb = rgb_queue.get()
            in_detections = detections_queue.get()
            in_depth = depth_queue.tryGet()

            if in_depth is not None:
                depth_frame = in_depth.getFrame()

            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                fps = counter / max(0.001, time.monotonic() - start_time)
                cv2.putText(
                    frame,
                    f"NN fps: {fps:.2f}",
                    (2, frame.shape[0] - 4),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.4,
                    (255, 255, 255),
                )

            if in_detections is not None:
                detections = in_detections.detections
                counter += 1

                if depth_frame is not None:
                    depth_h, depth_w = depth_frame.shape[:2]
                    intrinsics = get_intrinsics(
                        dai.CameraBoardSocket.CAM_A, depth_w, depth_h
                    )

                    for det in detections:
                        label = label_map[det.label] if label_map else str(det.label)
                        seen_labels.add(label)
                        xyz = calc_spatial_coords(
                            depth_frame,
                            (det.xmin, det.ymin, det.xmax, det.ymax),
                            intrinsics,
                            depth_w,
                            depth_h,
                            bbox_scale=config.bbox_scale,
                            depth_lower_mm=config.depth_lower_mm,
                            depth_upper_mm=config.depth_upper_mm,
                        )
                        if xyz is None:
                            print(
                                f"[{label}] conf={int(det.confidence * 100)}%  XYZ=N/A"
                            )
                            continue

                        store.update(label, xyz)
                        x, y, z = xyz
                        print(
                            f"[{label}] conf={int(det.confidence * 100)}%  "
                            f"X={x:7.0f} mm  Y={y:7.0f} mm  Z={z:7.0f} mm"
                        )

                        point = store.get(label)
                        if point is None:
                            continue
                        if config.target_label and label == config.target_label:
                            anchor_samples.append((label, point))
                            print(
                                f"Anchor samples: {len(anchor_samples)} / "
                                f"{ANCHOR_SAMPLE_COUNT}"
                            )
                            if len(anchor_samples) >= ANCHOR_SAMPLE_COUNT:
                                selected = _median_anchor(anchor_samples)
                                should_stop = True
                            break
                        if not config.target_label and selected is None:
                            anchor_samples.append((label, point))
                            print(
                                f"Anchor samples: {len(anchor_samples)} / "
                                f"{ANCHOR_SAMPLE_COUNT}"
                            )
                            if len(anchor_samples) >= ANCHOR_SAMPLE_COUNT:
                                selected = _median_anchor(anchor_samples)
                                should_stop = True
                            break

                    now = time.monotonic()
                    if (
                        config.target_label
                        and selected is None
                        and seen_labels
                        and now - last_target_warning > 2.0
                    ):
                        print(
                            "Waiting for target label "
                            f"'{config.target_label}'. Seen labels: "
                            f"{', '.join(sorted(seen_labels))}"
                        )
                        last_target_warning = now

            if frame is not None:
                display_frame("sign detection", frame)

            if cv2.waitKey(1) == ord("q"):
                should_stop = True

        pipeline.stop()

    cv2.destroyAllWindows()

    if selected is not None:
        label, point = selected
        print(
            f"\nSelected anchor [{label}] "
            f"X={point.x_mm:.0f} Y={point.y_mm:.0f} Z={point.z_mm:.0f} mm"
        )
        return selected

    if config.target_label:
        raise RuntimeError(
            f"No valid XYZ found for target label '{config.target_label}'. "
            "Update sign_detection.target_label or press q only after detection."
        )

    first = store.first()
    if first is None:
        raise RuntimeError("No valid sign detection was acquired.")
    return first


def _median_anchor(samples: list[tuple[str, SpatialPoint]]) -> tuple[str, SpatialPoint]:
    labels = [label for label, _ in samples]
    label = max(set(labels), key=labels.count)
    points = np.array(
        [
            [point.x_mm, point.y_mm, point.z_mm]
            for sample_label, point in samples
            if sample_label == label
        ],
        dtype=np.float32,
    )
    x_mm, y_mm, z_mm = np.median(points, axis=0)
    return label, SpatialPoint(float(x_mm), float(y_mm), float(z_mm))


def calc_spatial_coords(
    depth_frame: np.ndarray,
    bbox_norm,
    intrinsics,
    depth_w: int,
    depth_h: int,
    bbox_scale: float,
    depth_lower_mm: int,
    depth_upper_mm: int,
) -> tuple[float, float, float] | None:
    xmin, ymin, xmax, ymax = bbox_norm
    cx_n = (xmin + xmax) / 2.0
    cy_n = (ymin + ymax) / 2.0
    half_w = (xmax - xmin) * bbox_scale / 2.0
    half_h = (ymax - ymin) * bbox_scale / 2.0

    x1 = int(np.clip(cx_n - half_w, 0, 1) * depth_w)
    x2 = int(np.clip(cx_n + half_w, 0, 1) * depth_w)
    y1 = int(np.clip(cy_n - half_h, 0, 1) * depth_h)
    y2 = int(np.clip(cy_n + half_h, 0, 1) * depth_h)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = depth_frame[y1:y2, x1:x2]
    valid = roi[(roi >= depth_lower_mm) & (roi <= depth_upper_mm)]
    if valid.size == 0:
        return None

    z = float(np.median(valid))
    u = (xmin + xmax) / 2.0 * depth_w
    v = (ymin + ymax) / 2.0 * depth_h
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]
    return (u - cx) * z / fx, -((v - cy) * z / fy), z
