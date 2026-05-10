# ExitClear Minimal

Minimal hackathon MVP for depth-based volume-change detection with a Luxonis OAK camera.

The program first detects an emergency-exit sign and uses its XYZ position as the anchor for the monitored clearance volume. It then captures an empty-scene depth baseline, monitors the configured 3D volume around that anchor, and marks pixels as occupied when the current depth is closer than the baseline by at least `depth_delta_mm`. If smoothed occupancy stays above `occupancy_threshold_pct` for `persistence_threshold_s`, it enters `TRIGGERED`, writes a JSONL event, and updates a local status API for the frontend dashboard. The same OAK pipeline can also listen to the IMU for earthquake detection and run a lightweight people-counting NN for the frontend room occupancy metric.

This version uses object detection only for the first sign-localization step. The clearance decision itself remains depth-based and does not use segmentation, tracking, people counting, image streaming, cloud services, or custom models beyond the sign detector. Optional dashboard metrics and emergency audio are handled separately.

## Install

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For ElevenLabs emergency audio, set the API key in the shell before running.
To compose alarm + repeated voice into a single MP3, install `ffmpeg` too:

```bash
export ELEVENLABS_API_KEY="your-key"
# macOS, if needed:
brew install ffmpeg
```

## Run

Put the sign detection model archive next to `config.yaml` or update `sign_detection.model_path`, connect the OAK 4 D, then run:

```bash
python main.py
```

You can also override the model path without editing YAML:

```bash
python main.py --model /path/to/yolo.rvc4.tar.xz
```

Press `q` in the OpenCV window or `Ctrl+C` in the terminal to quit cleanly.

The process also starts a local status API:

```text
GET http://localhost:8000/api/status
GET http://localhost:8000/health
```

If needed, override the bind address:

```bash
python main.py --api-host 0.0.0.0 --api-port 8000
```

## Tune `config.yaml`

Key fields:

- `sign_detection.model_path`: local YOLO archive used to find the emergency sign.
- `sign_detection.target_label`: label to use as the monitored volume anchor. If empty, the first valid detection is used.
- `monitoring.volume_mm.width_mm`: monitored width around the anchor, half left and half right.
- `monitoring.volume_mm.height_below_anchor_mm`: monitored height below the anchor Y coordinate.
- `monitoring.volume_mm.depth_before_anchor_mm`: monitored depth in front of the anchor Z coordinate, toward the camera.
- `monitoring.depth_delta_mm`: a pixel is occupied when it is this much closer than baseline.
- `monitoring.occupancy_threshold_pct`: percentage of valid projected-volume pixels required to start pending.
- `monitoring.persistence_threshold_s`: seconds above threshold before entering `TRIGGERED`.
- `monitoring.baseline_frames`: empty-scene frames used for the median baseline.
- `monitoring.smoothing_frames`: rolling average window for occupancy percent.
- `output.events_path`: JSONL output path for trigger and clear events.
- `output.show_occupied_mask`: overlays occupied pixels in the main OpenCV window.
- `dashboard.room.name`: room name shown by the frontend.
- `dashboard.room.device_id`: frontend-facing device label.
- `dashboard.room.capacity`: room capacity shown by the frontend.
- `earthquake.enabled`: enables IMU-based earthquake detection.
- `earthquake.threshold_mps2`: vibration threshold in m/s^2 after subtracting gravity.
- `earthquake.min_duration_s`: sustained vibration time before evacuation is triggered.
- `people_counter.enabled`: enables the DM-Count camera node used for `people.current`.
- `people_counter.model_name`: Luxonis model zoo name for people counting.
- `people_counter.raw_scale`: density-map sum divisor used to convert raw model output to estimated people.
- `people_counter.smoothing_frames`: rolling average window for people count.
- `audio.enabled`: enables emergency audio generation through ElevenLabs.
- `audio.output_dir`: folder where generated MP3 files are cached.
- `audio.alarm_path`: optional local alarm MP3 to prepend before each voice message. Put your demo alarm at `assets/alarm.mp3` or update this path. If missing, a simple fallback alarm WAV is generated.
- `audio.repeat_count`: number of alarm/voice repetitions in the final emergency MP3.
- `audio.pause_ms`: silence between repeated emergency messages.
- `audio.voice_id`: ElevenLabs voice ID.
- `audio.voice_settings`: ElevenLabs voice settings used to make the announcement more urgent.
- `audio.messages.earthquake`: template for the earthquake evacuation message.

Console status, event writing, and live preview are enabled by default. They can still be overridden with `output.print_status`, `output.write_events_jsonl`, and `output.live_view` if needed, but they are intentionally left out of the default YAML.

For example, if the detected anchor is:

```text
X=0 mm, Y=2000 mm, Z=10000 mm
```

and the configured volume is:

```yaml
volume_mm:
  width_mm: 1500
  height_below_anchor_mm: 2000
  depth_before_anchor_mm: 1000
```

the monitored volume becomes:

```text
X: -750..750 mm
Y: 0..2000 mm
Z: 9000..10000 mm
```

For demo tuning, start with the provided values, place the camera where it will run, make sure the sign is detected, keep the clearance volume empty during baseline calibration, then adjust volume dimensions and thresholds.

The code uses high-quality camera defaults internally: `1280x800`, `15 FPS`, `HIGH_DETAIL`, subpixel depth, and `KERNEL_7x7` median filtering. These are hidden from the normal YAML to keep hackathon tuning focused on the monitored volume and trigger thresholds. If the device rejects the resolution or runs too slowly, add `frame_size: [640, 400]` under `monitoring` and `stereo_size: [640, 400]` under `sign_detection`.

## Live View

The main OpenCV window shows:

- RGB preview when available, otherwise a depth colormap.
- Orange front face of the monitored volume, closer to the camera.
- Green back face of the monitored volume, on the sign/door plane.
- Yellow depth edges and arrow showing the direction from the sign plane toward the camera.
- Red overlay on pixels currently considered occupied, if `show_occupied_mask` is enabled.
- Magenta cross at the detected sign anchor.
- Cyan cross at the projected center of the monitored volume.
- A compact status box with current state, smoothed occupancy percentage, threshold, persistence seconds, selected anchor label, and baseline calibration progress.
- `Press q to quit`.

The preview uses one OpenCV window only. `show_occupied_mask` controls whether occupied pixels are drawn on top of the live RGB feed.

## Status API

`GET /api/status` returns the latest in-memory snapshot for the frontend. It does not read from `events.jsonl`; events remain an append-only history.

The exit identity is derived from the detected sign label. For the current `emergency` label, the API returns `id: emergency_1`, `name: Emergency Exit 1`, and `type: emergency`.

`people.current` is updated from the OAK people-counter node when `people_counter.enabled` is true. If the model is unavailable or no NN result has arrived yet, it remains at the last known value, initially `0`.

`averageExitTimeSeconds` is included in every status, including `safe`, and is estimated from the current people count plus the number of clear exits.

Generated emergency audio is served from:

```text
GET /audio/<filename>
```

State mapping:

- `NO_BASELINE` and `CLEAR` -> dashboard `state: safe`, exit `status: CLEAR`.
- `OCCUPIED_PENDING` -> dashboard `state: caution`, exit `status: OCCUPIED_PENDING`.
- `TRIGGERED` -> dashboard `state: danger`, exit `status: TRIGGERED`.
- IMU earthquake trigger -> dashboard `state: emergency`, with an evacuation payload. This is latched until the backend is restarted.

When audio is enabled and `ELEVENLABS_API_KEY` is available, the earthquake trigger generates one cached voice MP3 with ElevenLabs, then composes a final emergency MP3 from the optional alarm sound plus the repeated voice message. The final `audioUrl`, normally similar to `/audio/earthquake_emergency_1_alarm_loop.mp3`, is added to both `alerts[0]` and `evacuation`.

If `assets/alarm.mp3` is missing, the backend generates `generated_audio/default_alarm.wav`. If `ffmpeg` is missing, the API also returns `audioSequence`, allowing the frontend to play alarm and voice clips in order without a pre-composed MP3. If ElevenLabs credentials are missing and no cached voice file exists, the backend still exposes an alarm-only sequence so evacuation has an audible fallback.

Example:

```json
{
  "state": "danger",
  "room": {
    "name": "Aula 4",
    "deviceId": "OAK-4D",
    "capacity": 100
  },
  "people": {
    "current": 13
  },
  "averageExitTimeSeconds": 13,
  "alerts": [],
  "exits": [
    {
      "id": "emergency_1",
      "name": "Emergency Exit 1",
      "type": "emergency",
      "status": "TRIGGERED",
      "occupancy": 23.6,
      "occupancyThreshold": 15.0
    }
  ],
  "updatedAt": "2026-05-10T02:30:00.000+02:00"
}
```

An earthquake evacuation response keeps the exit status visible and adds an alert:

```json
{
  "state": "emergency",
  "room": {
    "name": "Aula 4",
    "deviceId": "OAK-4D",
    "capacity": 100
  },
  "people": {
    "current": 0
  },
  "alerts": [
    {
      "severity": "emergency",
      "title": "Earthquake detected",
      "description": "OAK IMU detected sustained vibration above threshold (1.12 m/s^2).",
      "audioUrl": "http://localhost:8000/audio/earthquake_emergency_1_voice.mp3",
      "audioSequence": [
        "http://localhost:8000/audio/default_alarm.wav",
        "http://localhost:8000/audio/earthquake_emergency_1_voice.mp3",
        "http://localhost:8000/audio/default_alarm.wav",
        "http://localhost:8000/audio/earthquake_emergency_1_voice.mp3",
        "http://localhost:8000/audio/default_alarm.wav",
        "http://localhost:8000/audio/earthquake_emergency_1_voice.mp3"
      ],
      "audioPauseMs": 650
    }
  ],
  "exits": [
    {
      "id": "emergency_1",
      "name": "Emergency Exit 1",
      "type": "emergency",
      "status": "CLEAR",
      "occupancy": 0.0,
      "occupancyThreshold": 15.0
    }
  ],
  "evacuation": {
    "primaryExitId": "emergency_1",
    "route": "Emergency Exit 1",
    "arrow": "←",
    "startedAt": "2026-05-10T02:30:00.000+02:00",
    "label": "Use Emergency Exit 1",
    "audioUrl": "http://localhost:8000/audio/earthquake_emergency_1_voice.mp3",
    "audioSequence": [
      "http://localhost:8000/audio/default_alarm.wav",
      "http://localhost:8000/audio/earthquake_emergency_1_voice.mp3",
      "http://localhost:8000/audio/default_alarm.wav",
      "http://localhost:8000/audio/earthquake_emergency_1_voice.mp3",
      "http://localhost:8000/audio/default_alarm.wav",
      "http://localhost:8000/audio/earthquake_emergency_1_voice.mp3"
    ],
    "audioPauseMs": 650
  },
  "updatedAt": "2026-05-10T02:30:00.000+02:00"
}
```

## Events

Events are appended to `events.jsonl` by default. A triggered event looks like:

```json
{"anchor_label":"emergency","anchor_xyz_mm":{"x_mm":0.0,"y_mm":2000.0,"z_mm":10000.0},"depth_delta_mm":150,"device_id":"oak4d-exitclear-minimal-01","event_type":"volume_occupancy_triggered","occupancy_pct":25.7,"occupancy_threshold_pct":15.0,"persistence_s":3.0,"persistence_threshold_s":3.0,"projected_roi_px":{"x_max":362,"x_min":279,"y_max":201,"y_min":89},"state":"TRIGGERED","timestamp":"2026-05-09T12:34:56.789+02:00","volume_bounds_mm":{"x_max_mm":750.0,"x_min_mm":-750.0,"y_max_mm":2000.0,"y_min_mm":0.0,"z_max_mm":10000.0,"z_min_mm":9000.0},"volume_mm":{"depth_before_anchor_mm":1000,"height_below_anchor_mm":2000,"width_mm":1500},"zone_id":"exit_clearance_volume"}
```

A clear event uses `event_type: volume_occupancy_cleared` when occupancy returns below threshold after pending or triggered.

## Known Limitations

- One ROI only.
- The sign anchor is detected once at startup.
- Clearance monitoring only detects depth changes inside the configured 3D volume.
- People counting is an approximate density-map estimate and depends on `people_counter.raw_scale` calibration.
- Baseline is captured once at startup and is not automatically refreshed.
- Camera motion after baseline will produce false occupancy.
- Reflective, transparent, dark, or very distant surfaces can produce invalid depth.
- The projected volume uses camera intrinsics and assumes aligned depth/RGB output.
- Earthquake evacuation is latched and currently clears only on backend restart.
