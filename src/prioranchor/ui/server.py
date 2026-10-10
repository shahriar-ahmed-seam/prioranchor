"""FastAPI application factory for PriorAnchor telemetry dashboard and API."""

from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="PriorAnchor Telemetry Console",
        version="0.1.0",
        description="Dual-Objective Prior-Preservation Telemetry and Evaluation Workbench",
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "prioranchor"}

    return app
