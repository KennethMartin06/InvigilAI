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
from .ensemble_inference import init_ensemble, get_ensemble
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
        logger.info("MLP singleton loaded ✓")
    except Exception as exc:
        logger.error("Failed to load MLP: %s — falling back to ensemble only", exc)

    # Initialise the full multi-model ensemble (MLP + RF + SVM + LGB + XGB + ONNX)
    models_dir = os.path.dirname(model_path)
    try:
        ensemble = init_ensemble(models_dir)
        status = ensemble.get_status()
        logger.info(
            "Ensemble ready: %d models (%s)",
            status["ensemble_size"],
            ", ".join(status["models_loaded"]) or "none",
        )
    except Exception as exc:
        logger.error("Ensemble init failed: %s", exc)

    # Create DB tables
    create_tables()
    logger.info("Database tables created/verified ✓")

    # Seed demo accounts + sample data (only when SEED_DEMO_DATA=true)
    if os.getenv("SEED_DEMO_DATA", "false").lower() == "true":
        try:
            from .seed_demo import seed
            from .database import get_db
            db = next(get_db())
            seed(db)
        except Exception as exc:
            logger.warning("Demo seeding failed (non-fatal): %s", exc)

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
    ensemble = get_ensemble()
    ensemble_status = ensemble.get_status() if ensemble else {
        "ensemble_size": 0, "models_loaded": [],
    }
    return {
        "status": "ok",
        "mlp_loaded": _model is not None,
        "scaler_loaded": _scaler is not None,
        "ensemble_size": ensemble_status["ensemble_size"],
        "models_loaded": ensemble_status["models_loaded"],
    }


@app.get("/api/predict", tags=["inference"])
async def predict_endpoint(
    gaze_yaw: float = 0.0, gaze_pitch: float = 5.0,
    head_yaw: float = 0.0, head_pitch: float = 0.0, head_roll: float = 0.0,
    face_count: float = 1.0, gaze_deviation_ratio: float = 0.05,
    face_embedding_norm: float = 1.0,
    keystroke_rate: float = 3.0, mean_dwell_time: float = 100.0,
    mean_flight_time: float = 150.0, burst_coefficient: float = 1.0,
    cursor_velocity: float = 180.0, click_frequency: float = 0.5,
    idle_ratio: float = 0.10, trajectory_linearity: float = 0.55,
) -> dict:
    """
    One-shot REST prediction — useful for testing without a WebSocket.

    All 16 features are optional query parameters with sensible defaults.
    Returns the same shape as the WebSocket status message.
    """
    from .feature_fusion import fuse_and_predict
    visual = dict(
        gaze_yaw=gaze_yaw, gaze_pitch=gaze_pitch,
        head_yaw=head_yaw, head_pitch=head_pitch, head_roll=head_roll,
        face_count=face_count, gaze_deviation_ratio=gaze_deviation_ratio,
        face_embedding_norm=face_embedding_norm,
    )
    behavioral = dict(
        keystroke_rate=keystroke_rate, mean_dwell_time=mean_dwell_time,
        mean_flight_time=mean_flight_time, burst_coefficient=burst_coefficient,
        cursor_velocity=cursor_velocity, click_frequency=click_frequency,
        idle_ratio=idle_ratio, trajectory_linearity=trajectory_linearity,
    )
    return fuse_and_predict(visual, behavioral)
