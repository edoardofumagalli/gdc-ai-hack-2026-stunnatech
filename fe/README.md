# SeeCure FE

Small React/Vite frontend for the SeeCure room dashboard.

## Run

```bash
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

The frontend calls:

```text
http://localhost:8000/api/status
```

You can override the endpoint without changing code:

```text
http://localhost:5173?api=http://localhost:8000/api/status
```

The dashboard state is primarily controlled by the backend payload. The frontend also promotes room-over-capacity conditions so the people metric has visible impact during the demo. There are no local controls to switch between `safe`, `caution`, `danger`, and `emergency`.

## API payload

The FE accepts a compact status payload. Exit objects use occupancy-based fields:

```json
{
  "state": "safe",
  "room": {
    "name": "Aula 4",
    "deviceId": "OAK-4D",
    "capacity": 100
  },
  "people": {
    "current": 54
  },
  "alerts": [],
  "exits": [
    {
      "id": "emergency_1",
      "name": "Emergency Exit 1",
      "type": "emergency",
      "status": "CLEAR",
      "occupancy": 12,
      "occupancyThreshold": 15
    }
  ],
  "evacuation": {
    "primaryExitId": "L",
    "route": "Corridor -> Scala C",
    "arrow": "←",
    "startedAt": "2026-05-09T20:45:00Z",
    "audioUrl": "http://localhost:8000/audio/earthquake_emergency_1_voice.mp3",
    "audioSequence": [
      "http://localhost:8000/audio/default_alarm.wav",
      "http://localhost:8000/audio/earthquake_emergency_1_voice.mp3"
    ],
    "audioPauseMs": 650
  }
}
```

Accepted `state` values are `safe`, `caution`, `alarm`, `danger`, and `emergency`.

The frontend also applies local room-occupancy thresholds based on `room.capacity`:

- `people.current > room.capacity * 0.9` promotes `safe` to `caution`.
- `people.current > room.capacity` promotes the panel to `danger`.

Metric colors are independent from the global state. For example, if the room is in `caution` only because people are near capacity, a clear exit remains green. If an exit is pending but people are below the capacity threshold, the people card remains green.

Accepted exit `status` values are `CLEAR`, `OCCUPIED_PENDING`, and `TRIGGERED`. `TRIGGERED` is rendered as the current blocked/danger condition. If an exit status is missing, the FE can only infer `CLEAR` or `OCCUPIED_PENDING` from `occupancy >= occupancyThreshold`; the time-based transition to `TRIGGERED` must be decided by the backend.

If `state` is missing or invalid, the FE rejects the payload and keeps waiting for a valid backend status.

`capacity` belongs to `room`; the FE derives the internal `people.capacity` value from `room.capacity` for the `current / capacity` metric.

The third metric card shows average exit time instead of alert count. If the backend provides `averageExitTimeS`, `averageExitTimeSeconds`, `exitTime.averageSeconds`, or `evacuation.averageExitTimeSeconds`, the FE uses it. Otherwise it estimates the value from `people.current` and the number of clear exits.

When `evacuation.audioSequence` or `alerts[0].audioSequence` is present, the FE plays those URLs in order. Otherwise it falls back to `audioUrl`. The FE arms audio playback on the first click or keypress anywhere on the page, then starts evacuation audio automatically when `EVACUATE` arrives. Browser autoplay rules can still block fully unattended playback if the page never receives any user interaction.
