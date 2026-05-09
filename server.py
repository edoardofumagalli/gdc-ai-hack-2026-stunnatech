from __future__ import annotations

import asyncio
import json
from typing import Any

from exitclear.runtime import ExitClearRuntime


def create_app(runtime: ExitClearRuntime):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="ExitClear Backend", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return runtime.health()

    @app.get("/status")
    def status() -> dict[str, Any]:
        latest = runtime.latest_status_dict()
        if latest is None:
            raise HTTPException(status_code=503, detail="No status available yet")
        return latest

    @app.get("/events")
    def events(
        limit: int = Query(
            50,
            ge=1,
            le=500,
            description="Return the latest N events from the JSONL event log.",
        )
    ) -> list[dict[str, Any]]:
        return runtime.events(limit=limit)

    @app.post("/calibrate/baseline")
    def calibrate_baseline() -> dict[str, Any]:
        status_after_calibration = runtime.calibrate_baseline()
        return {
            "ok": True,
            "baseline_frames": runtime.baseline_frames,
            "status": status_after_calibration.to_dict(),
        }

    @app.get("/events/stream")
    async def events_stream() -> StreamingResponse:
        async def stream():
            next_index = len(runtime.events())
            while True:
                current_events = runtime.events()
                for event in current_events[next_index:]:
                    yield f"event: compliance_state_change\n"
                    yield f"data: {json.dumps(event, sort_keys=True)}\n\n"
                next_index = len(current_events)
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def run_server(runtime: ExitClearRuntime, host: str, port: int) -> None:
    import uvicorn

    app = create_app(runtime)
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        runtime.stop()
