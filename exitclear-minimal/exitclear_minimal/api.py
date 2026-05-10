from __future__ import annotations

from pathlib import Path
from threading import Thread
from urllib.parse import quote

from .status_store import DashboardStatusStore


def create_app(
    status_store: DashboardStatusStore,
    *,
    audio_dir: Path | None = None,
):
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FastAPI is required for the status API. Install requirements.txt first."
        ) from exc

    app = FastAPI(title="ExitClear Minimal API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/status")
    def status() -> dict:
        return status_store.get()

    if audio_dir is not None:
        audio_dir.mkdir(parents=True, exist_ok=True)
        app.mount("/audio", StaticFiles(directory=str(audio_dir)), name="audio")

    return app


class ApiServer:
    def __init__(
        self,
        status_store: DashboardStatusStore,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        audio_dir: Path | None = None,
    ) -> None:
        self.status_store = status_store
        self.host = host
        self.port = port
        self.audio_dir = audio_dir
        self._server = None
        self._thread: Thread | None = None

    @property
    def url(self) -> str:
        display_host = "localhost" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{display_host}:{self.port}/api/status"

    def audio_url(self, filename: str) -> str:
        display_host = "localhost" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{display_host}:{self.port}/audio/{quote(filename)}"

    def start(self) -> None:
        try:
            import uvicorn
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "uvicorn is required for the status API. Install requirements.txt first."
            ) from exc

        app = create_app(self.status_store, audio_dir=self.audio_dir)
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
