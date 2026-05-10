from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_FPS = 15
DEFAULT_FRAME_SIZE = [1280, 800]
DEFAULT_STEREO_PRESET = "HIGH_DETAIL"
DEFAULT_SUBPIXEL = True
DEFAULT_MEDIAN_FILTER = "KERNEL_7x7"
DEFAULT_BBOX_SCALE = 0.5
DEFAULT_DEPTH_LOWER_MM = 100
DEFAULT_DEPTH_UPPER_MM = 10000
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MIN_VALID_DEPTH_MM = 100
DEFAULT_EARTHQUAKE_ENABLED = False
DEFAULT_EARTHQUAKE_SAMPLE_RATE_HZ = 400
DEFAULT_EARTHQUAKE_BATCH_THRESHOLD = 20
DEFAULT_EARTHQUAKE_THRESHOLD_MPS2 = 0.98
DEFAULT_EARTHQUAKE_MIN_DURATION_S = 0.05
DEFAULT_AUDIO_ENABLED = False
DEFAULT_AUDIO_OUTPUT_DIR = "generated_audio"
DEFAULT_AUDIO_ALARM_PATH = "assets/alarm.mp3"
DEFAULT_AUDIO_REPEAT_COUNT = 3
DEFAULT_AUDIO_PAUSE_MS = 650
DEFAULT_AUDIO_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_AUDIO_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_AUDIO_OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_AUDIO_STABILITY = 0.35
DEFAULT_AUDIO_SIMILARITY_BOOST = 0.8
DEFAULT_AUDIO_STYLE = 0.6
DEFAULT_AUDIO_SPEED = 1.05
DEFAULT_AUDIO_USE_SPEAKER_BOOST = True
DEFAULT_EARTHQUAKE_AUDIO_MESSAGE = (
    "Attention. {event} detected. Evacuate immediately from {exit_name}."
)


@dataclass(frozen=True)
class DeviceConfig:
    id: str


@dataclass(frozen=True)
class RoiPx:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def as_dict(self) -> dict[str, int]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }

    def clipped(self, width: int, height: int) -> "RoiPx":
        return RoiPx(
            x_min=max(0, min(width, self.x_min)),
            y_min=max(0, min(height, self.y_min)),
            x_max=max(0, min(width, self.x_max)),
            y_max=max(0, min(height, self.y_max)),
        )


@dataclass(frozen=True)
class SignDetectionConfig:
    model_path: str
    target_label: str
    fps: int
    stereo_width: int
    stereo_height: int
    stereo_preset: str
    subpixel: bool
    median_filter: str
    bbox_scale: float
    depth_lower_mm: int
    depth_upper_mm: int
    confidence_threshold: float


@dataclass(frozen=True)
class VolumeConfig:
    width_mm: int
    height_below_anchor_mm: int
    depth_before_anchor_mm: int

    def as_dict(self) -> dict[str, int]:
        return {
            "width_mm": self.width_mm,
            "height_below_anchor_mm": self.height_below_anchor_mm,
            "depth_before_anchor_mm": self.depth_before_anchor_mm,
        }


@dataclass(frozen=True)
class MonitoringConfig:
    zone_id: str
    fps: int
    frame_width: int
    frame_height: int
    stereo_preset: str
    subpixel: bool
    median_filter: str
    volume: VolumeConfig
    depth_delta_mm: int
    occupancy_threshold_pct: float
    persistence_threshold_s: float
    baseline_frames: int
    min_valid_depth_mm: int
    smoothing_frames: int


@dataclass(frozen=True)
class OutputConfig:
    print_status: bool
    write_events_jsonl: bool
    events_path: str
    live_view: bool
    show_occupied_mask: bool


@dataclass(frozen=True)
class DashboardRoomConfig:
    name: str
    device_id: str
    capacity: int


@dataclass(frozen=True)
class DashboardConfig:
    room: DashboardRoomConfig


@dataclass(frozen=True)
class EarthquakeConfig:
    enabled: bool
    sample_rate_hz: int
    batch_threshold: int
    threshold_mps2: float
    min_duration_s: float


@dataclass(frozen=True)
class AudioConfig:
    enabled: bool
    output_dir: str
    alarm_path: str
    repeat_count: int
    pause_ms: int
    voice_id: str
    model_id: str
    output_format: str
    stability: float
    similarity_boost: float
    style: float
    speed: float
    use_speaker_boost: bool
    earthquake_message_template: str


@dataclass(frozen=True)
class AppConfig:
    device: DeviceConfig
    sign_detection: SignDetectionConfig
    monitoring: MonitoringConfig
    output: OutputConfig
    dashboard: DashboardConfig
    earthquake: EarthquakeConfig
    audio: AudioConfig


def load_config(path: str | Path) -> AppConfig:
    raw = _load_yaml(Path(path))

    device = raw.get("device", {})
    sign_detection = raw.get("sign_detection", {})
    monitoring = raw.get("monitoring", {})
    volume = monitoring.get("volume_mm", {})
    monitoring_frame_size = monitoring.get("frame_size", DEFAULT_FRAME_SIZE)
    output = raw.get("output", {})
    dashboard = raw.get("dashboard", {})
    dashboard_room = dashboard.get("room", {})
    earthquake = raw.get("earthquake", {})
    audio = raw.get("audio", {})
    audio_messages = audio.get("messages", {})
    voice_settings = audio.get("voice_settings", {})
    stereo_size = sign_detection.get("stereo_size", DEFAULT_FRAME_SIZE)

    config = AppConfig(
        device=DeviceConfig(id=str(device["id"])),
        sign_detection=SignDetectionConfig(
            model_path=str(sign_detection["model_path"]),
            target_label=str(sign_detection.get("target_label", "")),
            fps=int(sign_detection.get("fps", DEFAULT_FPS)),
            stereo_width=int(stereo_size[0]),
            stereo_height=int(stereo_size[1]),
            stereo_preset=str(
                sign_detection.get("stereo_preset", DEFAULT_STEREO_PRESET)
            ),
            subpixel=bool(sign_detection.get("subpixel", DEFAULT_SUBPIXEL)),
            median_filter=str(
                sign_detection.get("median_filter", DEFAULT_MEDIAN_FILTER)
            ),
            bbox_scale=float(sign_detection.get("bbox_scale", DEFAULT_BBOX_SCALE)),
            depth_lower_mm=int(
                sign_detection.get("depth_lower_mm", DEFAULT_DEPTH_LOWER_MM)
            ),
            depth_upper_mm=int(
                sign_detection.get("depth_upper_mm", DEFAULT_DEPTH_UPPER_MM)
            ),
            confidence_threshold=float(
                sign_detection.get(
                    "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD
                )
            ),
        ),
        monitoring=MonitoringConfig(
            zone_id=str(monitoring["zone_id"]),
            fps=int(monitoring.get("fps", DEFAULT_FPS)),
            frame_width=int(monitoring_frame_size[0]),
            frame_height=int(monitoring_frame_size[1]),
            stereo_preset=str(monitoring.get("stereo_preset", DEFAULT_STEREO_PRESET)),
            subpixel=bool(monitoring.get("subpixel", DEFAULT_SUBPIXEL)),
            median_filter=str(monitoring.get("median_filter", DEFAULT_MEDIAN_FILTER)),
            volume=VolumeConfig(
                width_mm=int(volume["width_mm"]),
                height_below_anchor_mm=int(volume["height_below_anchor_mm"]),
                depth_before_anchor_mm=int(volume["depth_before_anchor_mm"]),
            ),
            depth_delta_mm=int(monitoring["depth_delta_mm"]),
            occupancy_threshold_pct=float(monitoring["occupancy_threshold_pct"]),
            persistence_threshold_s=float(monitoring["persistence_threshold_s"]),
            baseline_frames=int(monitoring["baseline_frames"]),
            min_valid_depth_mm=int(
                monitoring.get("min_valid_depth_mm", DEFAULT_MIN_VALID_DEPTH_MM)
            ),
            smoothing_frames=int(monitoring["smoothing_frames"]),
        ),
        output=OutputConfig(
            print_status=bool(output.get("print_status", True)),
            write_events_jsonl=bool(output.get("write_events_jsonl", True)),
            events_path=str(output.get("events_path", "events.jsonl")),
            live_view=bool(output.get("live_view", True)),
            show_occupied_mask=bool(output.get("show_occupied_mask", True)),
        ),
        dashboard=DashboardConfig(
            room=DashboardRoomConfig(
                name=str(dashboard_room.get("name", "Aula 4")),
                device_id=str(dashboard_room.get("device_id", "OAK-4D")),
                capacity=int(dashboard_room.get("capacity", 100)),
            )
        ),
        earthquake=EarthquakeConfig(
            enabled=bool(earthquake.get("enabled", DEFAULT_EARTHQUAKE_ENABLED)),
            sample_rate_hz=int(
                earthquake.get(
                    "sample_rate_hz", DEFAULT_EARTHQUAKE_SAMPLE_RATE_HZ
                )
            ),
            batch_threshold=int(
                earthquake.get(
                    "batch_threshold", DEFAULT_EARTHQUAKE_BATCH_THRESHOLD
                )
            ),
            threshold_mps2=float(
                earthquake.get(
                    "threshold_mps2", DEFAULT_EARTHQUAKE_THRESHOLD_MPS2
                )
            ),
            min_duration_s=float(
                earthquake.get(
                    "min_duration_s", DEFAULT_EARTHQUAKE_MIN_DURATION_S
                )
            ),
        ),
        audio=AudioConfig(
            enabled=bool(audio.get("enabled", DEFAULT_AUDIO_ENABLED)),
            output_dir=str(audio.get("output_dir", DEFAULT_AUDIO_OUTPUT_DIR)),
            alarm_path=str(audio.get("alarm_path", DEFAULT_AUDIO_ALARM_PATH)),
            repeat_count=int(audio.get("repeat_count", DEFAULT_AUDIO_REPEAT_COUNT)),
            pause_ms=int(audio.get("pause_ms", DEFAULT_AUDIO_PAUSE_MS)),
            voice_id=str(audio.get("voice_id", DEFAULT_AUDIO_VOICE_ID)),
            model_id=str(audio.get("model_id", DEFAULT_AUDIO_MODEL_ID)),
            output_format=str(
                audio.get("output_format", DEFAULT_AUDIO_OUTPUT_FORMAT)
            ),
            stability=float(
                voice_settings.get("stability", DEFAULT_AUDIO_STABILITY)
            ),
            similarity_boost=float(
                voice_settings.get(
                    "similarity_boost", DEFAULT_AUDIO_SIMILARITY_BOOST
                )
            ),
            style=float(voice_settings.get("style", DEFAULT_AUDIO_STYLE)),
            speed=float(voice_settings.get("speed", DEFAULT_AUDIO_SPEED)),
            use_speaker_boost=bool(
                voice_settings.get(
                    "use_speaker_boost", DEFAULT_AUDIO_USE_SPEAKER_BOOST
                )
            ),
            earthquake_message_template=str(
                audio_messages.get("earthquake", DEFAULT_EARTHQUAKE_AUDIO_MESSAGE)
            ),
        ),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    if not config.device.id.strip():
        raise ValueError("device.id must not be empty")

    monitoring = config.monitoring
    if not monitoring.zone_id.strip():
        raise ValueError("monitoring.zone_id must not be empty")
    if monitoring.fps <= 0:
        raise ValueError("monitoring.fps must be positive")
    if monitoring.frame_width <= 0 or monitoring.frame_height <= 0:
        raise ValueError("monitoring.frame_size must be positive")
    if not monitoring.stereo_preset.strip():
        raise ValueError("monitoring.stereo_preset must not be empty")
    if not monitoring.median_filter.strip():
        raise ValueError("monitoring.median_filter must not be empty")

    sign_detection = config.sign_detection
    if not sign_detection.model_path.strip():
        raise ValueError("sign_detection.model_path must not be empty")
    if sign_detection.fps <= 0:
        raise ValueError("sign_detection.fps must be positive")
    if sign_detection.stereo_width <= 0 or sign_detection.stereo_height <= 0:
        raise ValueError("sign_detection.stereo_size must be positive")
    if not sign_detection.stereo_preset.strip():
        raise ValueError("sign_detection.stereo_preset must not be empty")
    if not sign_detection.median_filter.strip():
        raise ValueError("sign_detection.median_filter must not be empty")
    if not 0.0 < sign_detection.bbox_scale <= 1.0:
        raise ValueError("sign_detection.bbox_scale must be between 0 and 1")
    if sign_detection.depth_lower_mm <= 0:
        raise ValueError("sign_detection.depth_lower_mm must be positive")
    if sign_detection.depth_lower_mm >= sign_detection.depth_upper_mm:
        raise ValueError(
            "sign_detection.depth_lower_mm must be smaller than depth_upper_mm"
        )
    if not 0.0 <= sign_detection.confidence_threshold <= 1.0:
        raise ValueError("sign_detection.confidence_threshold must be between 0 and 1")

    volume = monitoring.volume
    if volume.width_mm <= 0:
        raise ValueError("monitoring.volume_mm.width_mm must be positive")
    if volume.height_below_anchor_mm <= 0:
        raise ValueError(
            "monitoring.volume_mm.height_below_anchor_mm must be positive"
        )
    if volume.depth_before_anchor_mm <= 0:
        raise ValueError(
            "monitoring.volume_mm.depth_before_anchor_mm must be positive"
        )
    if monitoring.depth_delta_mm <= 0:
        raise ValueError("monitoring.depth_delta_mm must be positive")
    if not 0.0 <= monitoring.occupancy_threshold_pct <= 100.0:
        raise ValueError("monitoring.occupancy_threshold_pct must be between 0 and 100")
    if monitoring.persistence_threshold_s < 0.0:
        raise ValueError("monitoring.persistence_threshold_s must be non-negative")
    if monitoring.baseline_frames < 1:
        raise ValueError("monitoring.baseline_frames must be at least 1")
    if monitoring.min_valid_depth_mm < 0:
        raise ValueError("monitoring.min_valid_depth_mm must be non-negative")
    if monitoring.smoothing_frames < 1:
        raise ValueError("monitoring.smoothing_frames must be at least 1")

    if not config.output.events_path.strip():
        raise ValueError("output.events_path must not be empty")

    room = config.dashboard.room
    if not room.name.strip():
        raise ValueError("dashboard.room.name must not be empty")
    if not room.device_id.strip():
        raise ValueError("dashboard.room.device_id must not be empty")
    if room.capacity <= 0:
        raise ValueError("dashboard.room.capacity must be positive")

    earthquake = config.earthquake
    if earthquake.sample_rate_hz <= 0:
        raise ValueError("earthquake.sample_rate_hz must be positive")
    if earthquake.batch_threshold <= 0:
        raise ValueError("earthquake.batch_threshold must be positive")
    if earthquake.threshold_mps2 <= 0.0:
        raise ValueError("earthquake.threshold_mps2 must be positive")
    if earthquake.min_duration_s < 0.0:
        raise ValueError("earthquake.min_duration_s must be non-negative")

    audio = config.audio
    if not audio.output_dir.strip():
        raise ValueError("audio.output_dir must not be empty")
    if audio.repeat_count < 1:
        raise ValueError("audio.repeat_count must be at least 1")
    if audio.pause_ms < 0:
        raise ValueError("audio.pause_ms must be non-negative")
    if not audio.model_id.strip():
        raise ValueError("audio.model_id must not be empty")
    if not audio.output_format.strip():
        raise ValueError("audio.output_format must not be empty")
    if not 0.0 <= audio.stability <= 1.0:
        raise ValueError("audio.voice_settings.stability must be between 0 and 1")
    if not 0.0 <= audio.similarity_boost <= 1.0:
        raise ValueError(
            "audio.voice_settings.similarity_boost must be between 0 and 1"
        )
    if not 0.0 <= audio.style <= 1.0:
        raise ValueError("audio.voice_settings.style must be between 0 and 1")
    if not 0.7 <= audio.speed <= 1.2:
        raise ValueError("audio.voice_settings.speed must be between 0.7 and 1.2")
    if not audio.earthquake_message_template.strip():
        raise ValueError("audio.messages.earthquake must not be empty")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to read config.yaml. Install requirements.txt first."
        ) from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at top level in {path}")
    return data
