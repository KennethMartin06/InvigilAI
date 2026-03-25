"""
websocket_handler.py — WebSocket endpoint for real-time proctoring.

Message protocol (all JSON):

Frontend → Backend:
  {"type": "frame",    "data": "<base64 JPEG>"}
  {"type": "behavior", "keystrokes": [...], "mouse": [...]}

Backend → Frontend:
  {"type": "status",  "cheating_probability": 0.12, "class_name": "Normal",
   "is_cheating": false, "class_probabilities": {...}, "timestamp": "..."}
  {"type": "flag",    "flag_id": 42, "cheating_probability": 0.87,
   "class_name": "Gaze/Distraction", "timestamp": "..."}
  {"type": "error",   "message": "..."}
  {"type": "ping",    "timestamp": "..."}
"""

import asyncio
import base64
import json
import logging
import os
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from .behavior_processor import BehaviorProcessor
from .config import settings
from .database import SessionLocal
from .feature_fusion import fuse_and_predict
from .frame_processor import FrameProcessor
from .models_db import ExamSession, Flag
from .auth import get_user_from_token_string

logger = logging.getLogger(__name__)

# Inference cycle interval in seconds (process & respond every N seconds)
_INFERENCE_INTERVAL = 2.0


async def proctor_websocket(websocket: WebSocket, session_id: int) -> None:
    """
    Handle a proctoring WebSocket connection for one exam session.

    Authenticates the user via `?token=<jwt>` query parameter.  Maintains
    per-connection FrameProcessor and BehaviorProcessor instances.  Runs an
    inference cycle on a 2-second background task loop.

    Parameters
    ----------
    websocket : WebSocket
    session_id : int — ExamSession.id
    """
    await websocket.accept()

    # ── Authenticate ──────────────────────────────────────────────────────────
    token = websocket.query_params.get("token")
    db: Session = SessionLocal()

    try:
        user = get_user_from_token_string(token, db) if token else None
        if user is None:
            await websocket.send_json({"type": "error", "message": "Unauthorised"})
            await websocket.close(code=4001)
            return

        session = db.query(ExamSession).filter(
            ExamSession.id == session_id,
            ExamSession.user_id == user.id,
        ).first()

        if session is None:
            await websocket.send_json({"type": "error", "message": "Session not found"})
            await websocket.close(code=4004)
            return

        logger.info(
            "WS connected: user=%d session=%d", user.id, session_id
        )

        # ── Per-connection state ──────────────────────────────────────────────
        frame_proc = FrameProcessor()
        behavior_proc = BehaviorProcessor()
        latest_frame_b64: list[str] = [None]  # mutable container
        last_inference_time: list[float] = [0.0]
        running: list[bool] = [True]

        # ── Inference task ────────────────────────────────────────────────────
        async def inference_loop() -> None:
            """Run inference every _INFERENCE_INTERVAL seconds while connected."""
            while running[0]:
                await asyncio.sleep(_INFERENCE_INTERVAL)
                if not running[0]:
                    break

                frame_b64 = latest_frame_b64[0]
                if frame_b64 is None:
                    continue

                try:
                    # Extract features
                    visual = frame_proc.process_frame(frame_b64)
                    behavioral = behavior_proc.compute_and_clear()
                    result = fuse_and_predict(visual, behavioral)

                    ts = datetime.utcnow().isoformat()

                    # Send status update to frontend
                    await websocket.send_json({
                        "type": "status",
                        "cheating_probability": result["cheating_probability"],
                        "class_name": result["class_name"],
                        "is_cheating": result["is_cheating"],
                        "class_probabilities": result["class_probabilities"],
                        "timestamp": ts,
                    })

                    # Persist flag if cheating detected
                    if result["is_cheating"]:
                        screenshot_path = await _save_screenshot(
                            frame_b64, session_id, ts
                        )
                        flag = Flag(
                            session_id=session_id,
                            timestamp=datetime.utcnow(),
                            cheating_probability=result["cheating_probability"],
                            predicted_class=result["predicted_class"],
                            class_name=result["class_name"],
                            screenshot_path=screenshot_path,
                            visual_features=visual,
                            behavioral_features=behavioral,
                        )
                        db.add(flag)

                        # Update session counters
                        session.total_flags = (session.total_flags or 0) + 1
                        session.max_cheating_score = max(
                            session.max_cheating_score or 0.0,
                            result["cheating_probability"],
                        )
                        db.commit()
                        db.refresh(flag)

                        await websocket.send_json({
                            "type": "flag",
                            "flag_id": flag.id,
                            "cheating_probability": result["cheating_probability"],
                            "class_name": result["class_name"],
                            "timestamp": ts,
                        })

                except Exception as exc:
                    logger.exception("Inference loop error: %s", exc)

        task = asyncio.create_task(inference_loop())

        # ── Message receive loop ──────────────────────────────────────────────
        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)

                if msg.get("type") == "frame":
                    latest_frame_b64[0] = msg.get("data", "")

                elif msg.get("type") == "behavior":
                    behavior_proc.add_keystroke_events(msg.get("keystrokes", []))
                    behavior_proc.add_mouse_events(msg.get("mouse", []))

        except WebSocketDisconnect:
            logger.info("WS disconnected: session=%d", session_id)
        finally:
            running[0] = False
            task.cancel()
            frame_proc.reset()
            behavior_proc.reset()

            # Mark session as completed if still active
            try:
                active_session = db.query(ExamSession).filter(
                    ExamSession.id == session_id
                ).first()
                if active_session and active_session.status == "active":
                    active_session.status = "completed"
                    active_session.end_time = datetime.utcnow()
                    db.commit()
            except Exception:
                pass

    finally:
        db.close()


async def _save_screenshot(base64_frame: str, session_id: int, ts: str) -> str:
    """
    Save a base64 JPEG frame as a file and return the relative path.

    Parameters
    ----------
    base64_frame : str — base64-encoded JPEG.
    session_id : int
    ts : str — ISO timestamp (used as filename).

    Returns
    -------
    str — relative path like "uploads/screenshots/42/2024-01-01T12:00:00.jpg"
    """
    try:
        safe_ts = ts.replace(":", "-").replace(".", "-")
        session_dir = os.path.join(settings.upload_dir, "screenshots", str(session_id))
        os.makedirs(session_dir, exist_ok=True)
        filename = f"{safe_ts}.jpg"
        filepath = os.path.join(session_dir, filename)

        # Strip data URI prefix if present
        if "," in base64_frame:
            base64_frame = base64_frame.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_frame)

        with open(filepath, "wb") as f:
            f.write(img_bytes)

        return filepath
    except Exception as exc:
        logger.warning("Failed to save screenshot: %s", exc)
        return ""
