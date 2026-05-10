# SeeCure — Technical Writeup

## Emergency-exit detector fine-tuning

This part of the project focused on teaching the system to recognize emergency exits. Standard object detectors such as YOLO are usually trained on COCO, which covers 80 common classes but does not include doors or emergency exits. We first tried the model YOLO word-testing, but that was not reliable enough for this use case. To solve the problem, we fine-tuned YOLOv8n in Google Colab on a custom emergency-exit dataset labeled in Roboflow.

Workflow summary

- Train a baseline YOLOv8n model on a custom dataset of emergency exits.
- Export the trained weights after fine-tuning.
- Upload the resulting model to the OAK device through Luxonis Hub so the detector runs on-device.
- Keep inference on the camera to avoid sending heavy video data to the PC.

Relevant training settings used in Colab

- Base model: `yolov8n.pt`
- Dataset: custom emergency-exit dataset from Roboflow (`data.yaml`)
- Epochs: `150`
- Image size: `640`
- Batch size: `8`
- Frozen layers: `6`
- Device: `0`
- Learning rate: `lr0=0.001`, `lrf=0.01`, `warmup_epochs=3`
- Optimizer: `AdamW`
- Weight decay: `0.0005`
- Augmentation: `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `degrees=10`, `translate=0.1`, `scale=0.5`, `mosaic=0.5`, `copy_paste=0.3`
- Disabled augmentations: `fliplr=0.0`, `flipud=0.0`, `mixup=0.0`
- Checkpointing and stopping: `save_period=10`, `patience=30`, `val=True`

Notes:
- Smaller batch size and lower learning rate helped the model adapt to a small custom dataset without becoming unstable.
- Moderate augmentation improved robustness to lighting and viewpoint changes while preserving the appearance of exit signs.
- Fine-tuning on the edge device reduced latency and removed the need to stream video to a PC for exit detection.


## Status API — detailed behavior

This document contains the full API examples, event formats, and runtime notes referenced from the main README.

Endpoints

- `GET /api/status` — returns the latest in-memory snapshot used by the dashboard. It includes room, people, exits, alerts, evacuation state, and `updatedAt` timestamp.
- `GET /health` — basic health endpoint.
- `GET /audio/<filename>.mp3` — serves generated or cached audio files when audio is enabled.

State mapping

- `NO_BASELINE` / `CLEAR` -> dashboard `state: safe`, exit `status: CLEAR`.
- `OCCUPIED_PENDING` -> dashboard `state: caution`, exit `status: OCCUPIED_PENDING`.
- `TRIGGERED` -> dashboard `state: danger`, exit `status: TRIGGERED`.
- IMU earthquake trigger -> dashboard `state: emergency` with an `evacuation` payload. Earthquake evacuation is latched until backend restart.

Example `/api/status` response

```json
{
  "state": "danger",
  "room": {
    "name": "Aula 4",
    "deviceId": "OAK-4D",
    "capacity": 100
  },
  "people": {
    "current": 0
  },
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

Earthquake evacuation example

See the full example in the repository README; the `evacuation` object includes route, audioUrl, audioSequence and `startedAt` timestamp.

## Events format (`events.jsonl`)

Events are appended to the file configured by `output.events_path`. A triggered event example:

```json
{"anchor_label":"emergency","anchor_xyz_mm":{"x_mm":0.0,"y_mm":2000.0,"z_mm":10000.0},"depth_delta_mm":150,"device_id":"oak4d-exitclear-minimal-01","event_type":"volume_occupancy_triggered","occupancy_pct":25.7,"occupancy_threshold_pct":15.0,"persistence_s":3.0,"persistence_threshold_s":3.0,"projected_roi_px":{"x_max":362,"x_min":279,"y_max":201,"y_min":89},"state":"TRIGGERED","timestamp":"2026-05-09T12:34:56.789+02:00","volume_bounds_mm":{"x_max_mm":750.0,"x_min_mm":-750.0,"y_max_mm":2000.0,"y_min_mm":0.0,"z_max_mm":10000.0,"z_min_mm":9000.0},"volume_mm":{"depth_before_anchor_mm":1000,"height_below_anchor_mm":2000,"width_mm":1500},"zone_id":"exit_clearance_volume"}
```

Clear events use `event_type: volume_occupancy_cleared`.

## Audio generation and behavior

- Voice TTS: See `ELEVENLABS_API_KEY` environment variable. When set and `audio.enabled` is true, the service requests a voice audio file from ElevenLabs and caches it.
- Composition: If `ffmpeg` is available on PATH, the service composes alarm + voice into a single MP3 loop using `ffmpeg`.
- Fallback: If `ffmpeg` is missing, the API returns an `audioSequence` (alarm + voice files) and a pause duration; the frontend can play them in order.

Runtime notes and troubleshooting

- If the sign detection model archive is missing, `main.py` raises `FileNotFoundError` with instructions to update `sign_detection.model_path` or pass `--model`.
- People-count models are fetched from the Luxonis model zoo at runtime. Pre-download archives to avoid runtime network fetches.
- On headless servers, OpenCV preview may fail; disable `output.live_view`.
- If audio generation fails due to missing API key, the system prints a warning and serves fallback alarm audio.

## Where to look in the code

- API server implementation: `exitclear_minimal/api.py`
- Event writer: `exitclear_minimal/events.py`
- Audio generation: `exitclear_minimal/audio.py`
- Main entry point: `main.py`


## Events

Events are appended to `events.jsonl` by default. A triggered event looks like:

```json
{"anchor_label":"emergency","anchor_xyz_mm":{"x_mm":0.0,"y_mm":2000.0,"z_mm":10000.0},"depth_delta_mm":150,"device_id":"oak4d-exitclear-minimal-01","event_type":"volume_occupancy_triggered","occupancy_pct":25.7,"occupancy_threshold_pct":15.0,"persistence_s":3.0,"persistence_threshold_s":3.0,"projected_roi_px":{"x_max":362,"x_min":279,"y_max":201,"y_min":89},"state":"TRIGGERED","timestamp":"2026-05-09T12:34:56.789+02:00","volume_bounds_mm":{"x_max_mm":750.0,"x_min_mm":-750.0,"y_max_mm":2000.0,"y_min_mm":0.0,"z_max_mm":10000.0,"z_min_mm":9000.0},"volume_mm":{"depth_before_anchor_mm":1000,"height_below_anchor_mm":2000,"width_mm":1500},"zone_id":"exit_clearance_volume"}
```

A clear event uses `event_type: volume_occupancy_cleared` when occupancy returns below threshold after pending or triggered.