"""
app.py — FastAPI application entry point.

Startup sequence:
  1. Load MLP model + scaler into inference module singletons.
  2. Create all database tables.
  3. Mount all route modules and the WebSocket endpoint.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import create_tables
from .inference import init_inference
from .middleware import RequestLoggingMiddleware, RateLimitMiddleware, setup_cors
from .routes.auth_routes import router as auth_router
from .routes.exam_routes import router as exam_router
from .routes.admin_routes import router as admin_router
from .routes.flag_routes import router as flag_router
from .websocket_handler import proctor_websocket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle handler."""
    logger.info("=== InvigilAI Proctoring Backend starting up ===")

    # Load ML models — resolve paths relative to this file's location so
    # the server works regardless of which directory uvicorn is launched from.
    _backend_dir = os.path.dirname(os.path.abspath(__file__))
    model_path  = os.path.normpath(os.path.join(_backend_dir, settings.model_path))
    scaler_path = os.path.normpath(os.path.join(_backend_dir, settings.scaler_path))
    logger.info("Loading MLP model from %s", model_path)
    logger.info("Loading scaler from %s", scaler_path)
    try:
        init_inference(model_path, scaler_path)
        logger.info("Models loaded successfully ✓")
    except Exception as exc:
        logger.error("Failed to load models: %s — inference will return defaults", exc)

    # Create DB tables
    create_tables()
    logger.info("Database tables created/verified ✓")

    # Ensure upload directories exist
    os.makedirs(os.path.join(settings.upload_dir, "screenshots"), exist_ok=True)

    yield

    logger.info("=== InvigilAI Proctoring Backend shutting down ===")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="InvigilAI Proctoring API",
    description="Real-time AI-powered online exam proctoring system",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware (order matters — CORS must come first)
setup_cors(app)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=200)

# REST routes
app.include_router(auth_router)
app.include_router(exam_router)
app.include_router(admin_router)
app.include_router(flag_router)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/proctor/{session_id}")
async def websocket_proctor(websocket: WebSocket, session_id: int) -> None:
    """
    WebSocket endpoint for real-time proctoring.

    Connect: ws://host:8000/ws/proctor/{session_id}?token=<jwt>
    """
    await proctor_websocket(websocket, session_id)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """Return server health status and model load state."""
    from .inference import _model, _scaler
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "scaler_loaded": _scaler is not None,
    }
