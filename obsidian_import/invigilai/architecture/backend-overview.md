---
title: Real-Time Backend Overview — InvigilAI
project: invigilai
tags: [backend, deployment, websockets, architecture]
source: proctoring_app/backend/
---

# Real-Time Backend Overview

Flask + WebSocket server that accepts live exam session streams (video frames, audio, keystroke events) and returns cheating risk scores in real time.

## File map

| File | Role |
|------|------|
| `backend_app.py` | Flask app factory, route registration, CORS, DB init |
| `backend_feature_fusion.py` | Aggregates per-modality scores → unified suspicion index |
| `backend_inference.py` | Loads trained models (RF, MLP, Ensemble, ONNX) for inference |
| `backend_websocket_handler.py` | Live session streaming: receives frames, runs inference, pushes alerts |
| `frame_processor.py` | Extracts face landmarks + gaze from webcam frames (MediaPipe) |
| `behavior_processor.py` | Computes keystroke dwell/flight, cursor velocity features |
| `auth.py` + `auth_routes.py` | JWT session auth |
| `models_db.py` + `database.py` | SQLAlchemy models for sessions/flags/users |
| `schemas.py` | Pydantic request/response validation |
| `routes/admin_routes.py` | Admin dashboard endpoints |
| `routes/exam_routes.py` | Student exam session endpoints |
| `routes/flag_routes.py` | Flag review + appeal endpoints |

## Request flow

```
Browser ──webcam+keys──▶ WebSocket ──▶ frame_processor + behavior_processor
                                             │
                                             ▼
                                      feature_fusion (32-dim vector)
                                             │
                                             ▼
                                      inference.predict_proba
                                             │
                                   ┌─────────┴──────────┐
                                   ▼                    ▼
                              risk score          Grad-CAM/SHAP
                                   │                    │
                                   └──────┬─────────────┘
                                          ▼
                             WebSocket ──▶ Admin Dashboard
                                          │
                                          ▼
                                      DB flag insert
```

## Deployment notes

- ONNX model is preferred in production (~5× faster than PyTorch on CPU)
- Scaler must be loaded alongside the model and applied to features before inference
- Use the same `add_derived_features` function as during training (16 → 32 dims)
- WebSocket frame rate: 2 fps is sufficient; 30 fps wastes compute

## Key design decision

**Late fusion over early fusion.** Each modality (video, audio, screen, keystroke) runs its own scorer; scores are combined downstream. This means:
- Any modality can be swapped / upgraded independently
- A missing modality (e.g. muted mic) doesn't break inference
- Weights (w_m) are tunable post-hoc without retraining
