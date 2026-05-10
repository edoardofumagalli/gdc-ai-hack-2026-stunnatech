
# SeeCure - Spatial Safety Intellligence

SeeCure — an intelligent, edge-first safety monitoring system

SeeCure is an intelligent, edge-first safety monitoring system that reasons in three dimensions. Using depth-capable cameras and on-device compute, SeeCure automatically detects emergency exits, observes the clearance volume beneath each exit for obstructions, counts people in the room, and estimates evacuation time. It fuses these signals and makes real-time safety judgments, including earthquake detection and alarm generation, without sending data to the cloud. The system summarizes its assessment into four safety levels — Safe, Hazard, Danger, Evacuate — so operators get immediate, actionable guidance.

High-level behaviors (non-technical):

- Safe: exits are unobstructed, occupancy is within safe limits, and no earthquake is detected.
- Alert: a new object has been detected within an exit clearance volume that may hinder evacuation, and/or the number of people in the room is at least 90% of the room’s maximum capacity.
- Danger: an obstruction has persisted beneath an exit for a sustained period, and/or the number of people in the room exceeds the room’s maximum capacity.
- Evacuate: an earthquake is detected and evacuation procedures should begin.

The goal of SeeCure is to reduce the chance that a catastrophic event becomes worse because exits are blocked or occupancy prevents effective evacuation. The remainder of this README explains the repository structure, how to run the demo, example outputs, and where to find the technical writeup for implementation details.

## Project structure — main components

- `main.py`: entry point; parses args, loads `config.yaml`, runs sign detection, builds the monitored volume, and wires together the pipeline (device, baseline, occupancy monitor, API, preview, audio, events).
- `config.yaml`: central configuration for volumes, thresholds, people counter, earthquake detector, audio, and outputs.
- `exitclear_minimal/`: primary package containing implementation modules:
  - `oak_depth_source.py`: OAK device pipeline and frame producer (depth, RGB, IMU, people NN).
  - `people_counter.py`: converts NN output into a smooth people count and density map.
  - `earthquake.py`: IMU-based vibration detector and trigger logic.
  - `sign_detection.py`: runs the sign detector to find the exit anchor.
  - `baseline.py`: captures empty-scene depth baseline.
  - `occupancy.py` and `state_machine.py`: compute occupancy percent and map it to Safe/Hazard/Danger states.
  - `preview.py`: OpenCV live view and overlays.
  - `api.py`: lightweight status API for the dashboard.
  - `audio.py`: emergency audio generation and composition (ElevenLabs + ffmpeg fallback).
  - `events.py`: append-only event writer for triggers and clears.
- `assets/` and `generated_audio/`: optional demo media and runtime audio cache.
- `events.jsonl`: append-only event history (configurable path).


## Setup & Run

This section describes how to prepare an environment and run the system so external users can reproduce the demo.

Requirements

- Python 3.10 or newer.
- A Luxonis OAK device for full functionality (depth, IMU, people counter). The code uses the `depthai` package to talk to the device.
- Optional: `ffmpeg` to compose alarm + repeated voice MP3s. If `ffmpeg` is missing the app falls back to a single voice file + alarm sequence.
- Internet access at first run to fetch people-counter NN models from the Luxonis model zoo (or pre-download them if running offline).

Create the Python environment and install dependencies:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install system `ffmpeg` (optional):

- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
- Windows: install `ffmpeg` and add it to `PATH` (choco or manual download)

Configure the sign detector model

- The project expects the sign detector archive referenced by `sign_detection.model_path` in `config.yaml` (default: `yolo.rvc4.tar.xz`).
- Put the model file next to `config.yaml` or edit `sign_detection.model_path` to point to the archive.
- You can override the model path at runtime with `--model` (see examples below). If the model file is missing, `main.py` will raise an informative FileNotFoundError.

Environment variables (optional)

- `ELEVENLABS_API_KEY`: set this to enable ElevenLabs TTS for emergency voice audio.
  - Bash/macOS: `export ELEVENLABS_API_KEY="your-key"`
  - PowerShell: `$env:ELEVENLABS_API_KEY = 'your-key'`

Run the system

```bash
# default: reads config.yaml in project root
python main.py

# useful options
python main.py --config config.yaml            # alternate config path
python main.py --model /path/to/model.tar.xz  # override sign detector archive
python main.py --api-host 0.0.0.0 --api-port 8000
```

What to expect

- The app connects to an OAK device via `depthai`. If `depthai` is not installed or the device is not connected, the program will raise an informative error.
- `events.jsonl` (configured in `config.yaml`) will be appended with trigger and clear events.
- A local status API is started (default `http://0.0.0.0:8000`) and serves `/api/status` and audio files if generated.
- If `ffmpeg` is missing, the audio service will fall back to a voice + alarm sequence and print a warning.

Troubleshooting & notes

- People counting uses a model from the Luxonis model zoo. Ensure internet access or pre-download the model archive to avoid runtime fetches.
- On headless machines the OpenCV preview window may not work; disable live preview in `config.yaml` if needed (`output.live_view: false`).
- If you see `Sign detection model not found`, either download the model archive, update `config.yaml`, or pass `--model`.
- For device-specific errors (enumeration failures, missing firmware), consult the `depthai` / OAK documentation and ensure the device is connected and supported.

## Tune `config.yaml`

The `config.yaml` file exposes the parameters that control monitored volume geometry, depth sensitivity, persistence, people-count smoothing and audio behavior. Adjust these values to match the physical room, camera position, and acceptable false-positive rate. For step-by-step tuning guidance and example values, keep a copy of observed runs (preview, `events.jsonl`, and `/api/status`) and iterate the parameters below. 

Config options and tuning details are documented in the technical file, `TECHNICAL.md`.



## Status API

See the `TECHNICAL.md` file for full API examples, schemas, and event formats. Quick checks:

- `GET /api/status` — returns the latest dashboard snapshot.
- `GET /health` — basic health check.
- Generated audio is served from `/audio/<filename>.mp3` when audio is enabled.



## Known Limitations

- One ROI only.
- The sign anchor is detected once at startup.
- Clearance monitoring only detects depth changes inside the configured 3D volume.
- Baseline is captured once at startup and is not automatically refreshed.
- Camera motion after baseline will produce false occupancy.
- Reflective, transparent, dark, or very distant surfaces can produce invalid depth.
- The projected volume uses camera intrinsics and assumes aligned depth/RGB output.
- Earthquake evacuation is latched and currently clears only on backend restart.
