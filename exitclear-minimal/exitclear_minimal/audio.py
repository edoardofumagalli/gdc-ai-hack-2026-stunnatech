from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import re
import shutil
import subprocess
import wave
from threading import Lock, Thread
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import AudioConfig

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


@dataclass(frozen=True)
class EmergencyAudioResult:
    audio_url: str
    sequence_urls: list[str]
    pause_ms: int


class EmergencyAudioService:
    def __init__(self, config: AudioConfig, project_root: Path) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = project_root / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        alarm_path = config.alarm_path.strip()
        self.alarm_path = Path(alarm_path) if alarm_path else None
        if self.alarm_path is not None and not self.alarm_path.is_absolute():
            self.alarm_path = project_root / self.alarm_path
        self._pending: set[str] = set()
        self._lock = Lock()
        self._warned_missing_credentials = False
        self._warned_missing_ffmpeg = False

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def request_earthquake_audio(
        self,
        *,
        event_label: str,
        exit_id: str,
        exit_name: str,
        public_url_for_filename: Callable[[str], str],
        on_ready: Callable[[EmergencyAudioResult], None],
    ) -> None:
        if not self.enabled:
            return

        stem = f"{_slug(event_label)}_{_slug(exit_id)}"
        filename = f"{stem}_alarm_loop.mp3"
        path = self.output_dir / filename
        audio_url = public_url_for_filename(filename)
        if path.exists() and path.stat().st_size > 0:
            on_ready(
                EmergencyAudioResult(
                    audio_url=audio_url,
                    sequence_urls=[audio_url],
                    pause_ms=0,
                )
            )
            return

        with self._lock:
            if filename in self._pending:
                return
            self._pending.add(filename)

        text = self.config.earthquake_message_template.format(
            event=event_label,
            exit_name=exit_name,
        )
        thread = Thread(
            target=self._generate_background,
            kwargs={
                "filename": filename,
                "path": path,
                "voice_path": self.output_dir / f"{stem}_voice.mp3",
                "text": text,
                "audio_url": audio_url,
                "public_url_for_filename": public_url_for_filename,
                "on_ready": on_ready,
            },
            daemon=True,
        )
        thread.start()

    def _generate_background(
        self,
        *,
        filename: str,
        path: Path,
        voice_path: Path,
        text: str,
        audio_url: str,
        public_url_for_filename: Callable[[str], str],
        on_ready: Callable[[EmergencyAudioResult], None],
    ) -> None:
        try:
            generated = (
                voice_path.exists()
                and voice_path.stat().st_size > 0
                or self._generate_voice_audio(voice_path, text)
            )
            if not generated:
                alarm_sequence = self._alarm_sequence_urls(
                    public_url_for_filename=public_url_for_filename
                )
                on_ready(
                    EmergencyAudioResult(
                        audio_url=alarm_sequence[0],
                        sequence_urls=alarm_sequence,
                        pause_ms=self.config.pause_ms,
                    )
                )
                print("Emergency audio fallback ready: alarm only")
                return

            composed = self._compose_or_fallback(path, voice_path)
            result_audio_url = (
                audio_url if composed else public_url_for_filename(voice_path.name)
            )
            on_ready(
                EmergencyAudioResult(
                    audio_url=result_audio_url,
                    sequence_urls=self._audio_sequence_urls(
                        composed=composed,
                        audio_url=audio_url,
                        voice_path=voice_path,
                        public_url_for_filename=public_url_for_filename,
                    ),
                    pause_ms=0 if composed else self.config.pause_ms,
                )
            )
            if composed:
                print(f"Emergency audio generated: {path}")
            else:
                print("Emergency audio sequence ready: alarm + voice fallback")
        except Exception as exc:
            print(f"Emergency audio generation failed: {exc}")
        finally:
            with self._lock:
                self._pending.discard(filename)

    def _generate_voice_audio(self, path: Path, text: str) -> bool:
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        voice_id = self.config.voice_id.strip()
        if not api_key or not voice_id:
            if not self._warned_missing_credentials:
                print(
                    "Emergency audio disabled: set ELEVENLABS_API_KEY and "
                    "audio.voice_id to generate files."
                )
                self._warned_missing_credentials = True
            return False

        payload = {
            "text": text,
            "model_id": self.config.model_id,
            "voice_settings": {
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost,
                "style": self.config.style,
                "speed": self.config.speed,
                "use_speaker_boost": self.config.use_speaker_boost,
            },
        }
        url = (
            ELEVENLABS_TTS_URL.format(voice_id=quote(voice_id, safe=""))
            + f"?output_format={quote(self.config.output_format, safe='')}"
        )
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "xi-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                data = response.read()
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ElevenLabs returned HTTP {exc.code}: {details}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"ElevenLabs request failed: {exc.reason}") from exc

        if not data:
            raise RuntimeError("ElevenLabs returned an empty audio response")
        path.write_bytes(data)
        return True

    def _compose_or_fallback(self, path: Path, voice_path: Path) -> bool:
        if path.exists() and path.stat().st_size > 0:
            return True

        if self._compose_emergency_loop(path, voice_path):
            return True

        return False

    def _compose_emergency_loop(self, path: Path, voice_path: Path) -> bool:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            if not self._warned_missing_ffmpeg:
                print(
                    "Emergency audio loop disabled: install ffmpeg to compose "
                    "alarm + repeated voice. Using single voice file."
                )
                self._warned_missing_ffmpeg = True
            return False

        segments = self._loop_segments(voice_path)
        if len(segments) <= 1:
            return False

        temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
        command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        filter_inputs = []

        for segment in segments:
            if segment == "silence":
                pause_s = self.config.pause_ms / 1000.0
                command.extend(
                    [
                        "-f",
                        "lavfi",
                        "-t",
                        f"{pause_s:.3f}",
                        "-i",
                        "anullsrc=channel_layout=stereo:sample_rate=44100",
                    ]
                )
            else:
                command.extend(["-i", str(segment)])
            index = len(filter_inputs)
            filter_inputs.append(
                f"[{index}:a:0]"
                "aresample=44100,"
                "aformat=sample_fmts=fltp:channel_layouts=stereo"
                f"[a{index}]"
            )

        concat_inputs = "".join(f"[a{index}]" for index in range(len(segments)))
        filter_complex = ";".join(
            [
                *filter_inputs,
                f"{concat_inputs}concat=n={len(segments)}:v=0:a=1[outa]",
            ]
        )
        command.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[outa]",
                "-vn",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "128k",
                str(temp_path),
            ]
        )

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"Emergency audio composition failed: {exc}")
            if temp_path.exists():
                temp_path.unlink()
            return False

        if temp_path.exists() and temp_path.stat().st_size > 0:
            temp_path.replace(path)
            return True
        return False

    def _loop_segments(self, voice_path: Path) -> list[Path | str]:
        alarm_path = self._served_alarm_path()
        segments: list[Path | str] = []
        for index in range(self.config.repeat_count):
            if alarm_path is not None:
                segments.append(alarm_path)
            segments.append(voice_path)
            if self.config.pause_ms > 0 and index < self.config.repeat_count - 1:
                segments.append("silence")
        return segments

    def _served_alarm_path(self) -> Path:
        generated_alarm = self.output_dir / "default_alarm.wav"
        if (
            self.alarm_path is not None
            and self.alarm_path.exists()
            and self.alarm_path.stat().st_size > 0
        ):
            served_alarm = self.output_dir / f"alarm{self.alarm_path.suffix}"
            if self.alarm_path.resolve() != served_alarm.resolve():
                shutil.copyfile(self.alarm_path, served_alarm)
            return served_alarm

        if not generated_alarm.exists() or generated_alarm.stat().st_size == 0:
            alarm_source = self.alarm_path or "no configured alarm path"
            print(
                f"Emergency alarm sound not found: {alarm_source}. "
                f"Generated fallback alarm: {generated_alarm}"
            )
            _write_fallback_alarm(generated_alarm)
        return generated_alarm

    def _audio_sequence_urls(
        self,
        *,
        composed: bool,
        audio_url: str,
        voice_path: Path,
        public_url_for_filename: Callable[[str], str],
    ) -> list[str]:
        if composed:
            return [audio_url]

        alarm_path = self._served_alarm_path()
        alarm_url = public_url_for_filename(alarm_path.name)
        voice_url = public_url_for_filename(voice_path.name)
        sequence = []
        for _ in range(self.config.repeat_count):
            sequence.append(alarm_url)
            sequence.append(voice_url)
        return sequence

    def _alarm_sequence_urls(
        self,
        *,
        public_url_for_filename: Callable[[str], str],
    ) -> list[str]:
        alarm_path = self._served_alarm_path()
        alarm_url = public_url_for_filename(alarm_path.name)
        return [alarm_url for _ in range(self.config.repeat_count)]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "audio"


def _write_fallback_alarm(path: Path) -> None:
    sample_rate = 44100
    duration_s = 0.85
    samples = int(sample_rate * duration_s)
    amplitude = 0.45

    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        frames = bytearray()
        for index in range(samples):
            t = index / sample_rate
            frequency = 880 if int(t * 8) % 2 == 0 else 1320
            envelope = min(1.0, t / 0.04, (duration_s - t) / 0.08)
            value = int(
                32767
                * amplitude
                * max(0.0, envelope)
                * math.sin(2.0 * math.pi * frequency * t)
            )
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))
        audio.writeframes(bytes(frames))
