<div align="center">

# InvigilAI

**Multi-Modal AI-Powered Online Exam Proctoring System**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*Detect cheating in real-time using fused visual and behavioural AI signals — no human invigilator required.*

[Features](#-features) · [Architecture](#-architecture) · [ML Pipeline](#-ml-pipeline) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Deployment](#-deployment)

</div>

---

## Overview

InvigilAI is a full-stack, production-ready online exam proctoring platform that uses a trained **multi-modal machine learning model** to detect cheating behaviour in real-time. It fuses **8 visual signals** captured from the webcam (gaze direction, head pose, face count) with **8 behavioural signals** (keystroke dynamics, mouse trajectories) to produce a per-window cheating probability score.

When a student sits an exam, a **React frontend** streams webcam frames and input events over WebSocket to a **FastAPI backend** every 2 seconds. The backend runs inference through a trained **PyTorch MLP** and responds instantly — flagging suspicious sessions, storing screenshots, and surfacing everything on a live **Admin Dashboard**.

---

## Features

| Category | Capabilities |
|---|---|
| **AI Detection** | 5-class MLP (99.4% test accuracy), SVM and Random Forest fallbacks, binary cheating threshold at 70% |
| **Visual Analysis** | MediaPipe Face Mesh gaze estimation, solvePnP head pose, face count, rolling gaze deviation ratio |
| **Behavioural Analysis** | Keystroke rate, dwell time, flight time, burst coefficient, cursor velocity, click frequency, idle ratio, trajectory linearity |
| **Real-Time** | WebSocket inference cycle every 2 seconds, live status badge (🟢/🟡/🔴), in-exam warning banners |
| **Admin Tools** | Live session grid, full session review, flag timeline with screenshots, per-flag feature breakdown, one-click acknowledge |
| **Auth** | JWT (HS256) + bcrypt, role-based access (student / admin), 24-hour token expiry |
| **Storage** | PostgreSQL (production) / SQLite (local), JPEG screenshot archive per session |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (Student)                        │
│                                                                 │
│  ┌──────────────┐   frame every 1.5s  ┌─────────────────────┐  │
│  │  CameraFeed  │ ──────────────────► │                     │  │
│  │  (WebRTC)    │                     │   WebSocket         │  │
│  └──────────────┘  behavior every 2s  │  /ws/proctor/{id}   │  │
│  ┌──────────────┐ ──────────────────► │                     │  │
│  │  Keystroke + │                     └──────────┬──────────┘  │
│  │  Mouse hooks │ ◄──── status / flag ───────────┘             │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
                              │ REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│                                                                 │
│  frame_processor.py        behavior_processor.py               │
│  ┌─────────────────┐       ┌──────────────────────┐            │
│  │ MediaPipe Face  │       │  Keystroke/Mouse     │            │
│  │ Mesh → 8 visual │       │  events → 8 behav.   │            │
│  │ features        │       │  features            │            │
│  └────────┬────────┘       └──────────┬───────────┘            │
│           │                           │                         │
│           └──────────┬────────────────┘                        │
│                      ▼                                          │
│              feature_fusion.py                                  │
│          ┌───────────────────────┐                              │
│          │ 16-dim feature vector │                              │
│          │  [8 visual + 8 behav] │                              │
│          └──────────┬────────────┘                              │
│                     ▼                                           │
│               inference.py                                      │
│          ┌───────────────────────┐                              │
│          │  PyTorch MLP          │   → predicted_class          │
│          │  16→128→64→5          │   → cheating_probability     │
│          │  + StandardScaler     │   → is_cheating (θ=0.70)    │
│          └───────────────────────┘                              │
│                                                                 │
│  SQLAlchemy ORM  │  JWT Auth  │  Screenshot storage             │
│  PostgreSQL / SQLite          │  uploads/screenshots/           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Browser (Admin)                             │
│                                                                 │
│  AdminDashboard → StatsOverview → LiveSessionCard               │
│  SessionReview  → FlagTimeline → Probability chart              │
└─────────────────────────────────────────────────────────────────┘
```

---

## ML Pipeline

The ML component (`cheating_detection/`) is a standalone, fully trained pipeline.

### Dataset

240 synthetic exam sessions across 5 categories, totalling **5,411 time windows**:

| Label | Category | Sessions | Windows |
|---|---|---|---|
| 0 | Normal Behaviour | 102 | 2,303 |
| 1 | Gaze / Distraction | 58 | 1,314 |
| 2 | External Device Use | 42 | 909 |
| 3 | Multi-Person / Impersonation | 24 | 558 |
| 4 | Abnormal Keystroke Behaviour | 14 | 327 |

### Feature Vector (16 dimensions)

**Visual (indices 0–7)** — extracted from webcam frames via MediaPipe Face Mesh:

| Feature | Description |
|---|---|
| `gaze_yaw` | Horizontal gaze angle (degrees) |
| `gaze_pitch` | Vertical gaze angle (degrees) |
| `head_yaw` | Head horizontal rotation via solvePnP |
| `head_pitch` | Head vertical tilt |
| `head_roll` | Head roll angle |
| `face_count` | Number of detected faces in frame |
| `gaze_deviation_ratio` | Fraction of frames where \|gaze_yaw\| > 25° (rolling 30-frame window) |
| `face_embedding_norm` | L2 norm of facial embedding (impersonation proxy) |

**Behavioural (indices 8–15)** — computed from raw keystroke/mouse event streams:

| Feature | Description |
|---|---|
| `keystroke_rate` | Keys per second |
| `mean_dwell_time` | Average key-hold duration (ms) |
| `mean_flight_time` | Average inter-key interval (ms) |
| `burst_coefficient` | Variance / mean of inter-key intervals (irregularity index) |
| `cursor_velocity` | Mean cursor speed (px/s) |
| `click_frequency` | Mouse clicks per second |
| `idle_ratio` | Fraction of window with no input activity |
| `trajectory_linearity` | Straight-line / total path ratio (copy-paste proxy) |

### Model Performance (held-out test set, 813 windows)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **MLP (PyTorch)** | **99.38%** | 99.63% | 98.64% | 99.12% |
| Random Forest | 99.38% | 99.72% | 98.81% | 99.24% |
| SVM (RBF) | 99.38% | 99.72% | 98.81% | 99.24% |

**Ablation study** — why fusion matters:

| Feature Set | F1 Score |
|---|---|
| Visual Only (8 features) | 69.9% |
| Behavioural Only (8 features) | 67.7% |
| **Multi-Modal Fused (16 features)** | **99.1%** |

Binary cheating detection: **ROC AUC = 0.9997** · **Average Precision = 0.9998**

---

## Project Structure

```
InvigilAI/
├── cheating_detection/              # ML pipeline (standalone)
│   ├── data/
│   │   ├── generate_dataset.py      # Synthetic data generator (240 sessions)
│   │   ├── synthetic_dataset.npz    # Compressed numpy dataset
│   │   └── synthetic_dataset.csv   # Human-readable dataset
│   ├── preprocessing/
│   │   └── preprocess.py            # StandardScaler, 70/15/15 stratified split
│   ├── features/
│   │   ├── visual_features.py       # Gaze deviation, embedding norm extraction
│   │   └── behavioral_features.py  # KR, MDT, MFT, BC, CV, CF, IR, TL
│   ├── models/
│   │   ├── train.py                 # SVM + GridSearch, RF, PyTorch MLP training
│   │   ├── evaluate.py              # Metrics, ablation, threshold sweep, ROC/PR
│   │   ├── model_utils.py           # Save/load helpers, cheating_probability()
│   │   ├── mlp_model.pth            # Trained MLP weights
│   │   ├── rf_model.joblib          # Trained Random Forest
│   │   ├── svm_model.joblib         # Trained SVM
│   │   └── scaler.joblib            # Fitted StandardScaler
│   ├── outputs/                     # Training curves, confusion matrix, ROC plots
│   ├── config.py                    # All hyperparameters
│   └── main.py                      # End-to-end: generate → train → evaluate
│
└── proctoring_app/                  # Full-stack web application
    ├── backend/                     # FastAPI (Python)
    │   ├── app.py                   # Entry point + lifespan handler
    │   ├── config.py                # Settings from .env (pydantic-settings)
    │   ├── database.py              # SQLAlchemy engine (PostgreSQL + SQLite)
    │   ├── models_db.py             # ORM: User, ExamSession, Flag, Question
    │   ├── schemas.py               # Pydantic request/response schemas
    │   ├── auth.py                  # JWT HS256 + bcrypt
    │   ├── inference.py             # MLP loader + predict()
    │   ├── frame_processor.py       # MediaPipe → 8 visual features
    │   ├── behavior_processor.py    # Event buffers → 8 behavioural features
    │   ├── feature_fusion.py        # Merge → 16-dim → inference
    │   ├── websocket_handler.py     # Real-time 2s inference loop
    │   ├── middleware.py            # CORS, logging, rate limiting
    │   ├── routes/                  # auth, exam, admin, flags endpoints
    │   ├── seed_data.py             # Demo users + exam questions
    │   └── requirements.txt
    └── frontend/                    # React 18 + Vite + Tailwind CSS
        └── src/
            ├── pages/               # Login, Signup, ExamLobby, ExamPage,
            │                        # ExamComplete, AdminDashboard, SessionReview
            ├── components/          # CameraFeed, StatusIndicator, FlagTimeline,
            │                        # ExamTimer, QuestionPanel, WarningBanner,
            │                        # LiveSessionCard, StatsOverview, Navbar
            ├── hooks/               # useCamera, useWebSocket, useKeystrokes,
            │                        # useMouseTracker, useAuth
            ├── services/            # Axios API client, ProctorSocket class
            ├── context/             # AuthContext (JWT persistence)
            └── utils/               # featureHelpers, statusColor, formatTime
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+

### 1. Clone & train the ML models

```bash
git clone https://github.com/kennethmartin06/invigilai.git
cd invigilai

pip install -r cheating_detection/requirements.txt
python -m cheating_detection.main
# Generates dataset, trains all 3 models, saves to cheating_detection/models/
```

### 2. Start the backend

```bash
cd proctoring_app/backend

pip install -r requirements.txt

cp .env.example .env
# Edit .env — set a strong JWT_SECRET and your DATABASE_URL

python seed_data.py          # Creates demo users + 10 exam questions
uvicorn proctoring_app.backend.app:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### 3. Start the frontend

```bash
cd proctoring_app/frontend

npm install
npm run dev
# Open http://localhost:5173
```

### Default credentials

| Role | Email | Password |
|---|---|---|
| Admin | `admin@proctor.ai` | `admin123` |
| Student | `student1@proctor.ai` | `student123` |
| Student | `student2@proctor.ai` | `student123` |

---

## Demo Flow

```
1. Log in as student1@proctor.ai
2. ExamLobby → allow camera → select "General Knowledge" → Start Exam
3. ExamPage → answer questions
   └── Camera streams every 1.5s → inference every 2s
   └── Status badge updates live: 🟢 Normal / 🟡 Suspicious / 🔴 Cheating
   └── Warning banner slides in when a flag is raised
4. Submit Exam → ExamComplete summary

5. Log out → log in as admin@proctor.ai
6. AdminDashboard → live stats, session table, flag distribution
7. Click any session → SessionReview → flag timeline + probability chart
```

---

## API Reference

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/signup` | Register new user, returns JWT |
| `POST` | `/api/auth/login` | Login, returns JWT |
| `GET` | `/api/auth/me` | Current user profile |

### Exam (Student)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/exam/start` | Create session, get questions |
| `POST` | `/api/exam/answer` | Save / update answer |
| `POST` | `/api/exam/end/{session_id}` | Submit exam, get summary |
| `GET` | `/api/exam/questions/{title}` | Fetch questions for an exam |

### Flags

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/flags/{session_id}` | All flags for a session |
| `GET` | `/api/flags/student/{student_id}` | All flags for a student |
| `PATCH` | `/api/flags/{flag_id}/acknowledge` | Mark flag reviewed (admin) |

### Admin

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/admin/sessions` | All sessions (filterable) |
| `GET` | `/api/admin/sessions/active` | Currently live sessions |
| `GET` | `/api/admin/stats` | Aggregate dashboard stats |
| `GET` | `/api/admin/student/{id}/history` | Student session history |

### WebSocket

```
ws://host:8000/ws/proctor/{session_id}?token=<jwt>
```

**Client → Server:**
```jsonc
// Camera frame (every 1.5s)
{ "type": "frame", "data": "<base64 JPEG>" }

// Behavioural events (every 2s)
{ "type": "behavior", "keystrokes": [...], "mouse": [...] }
```

**Server → Client:**
```jsonc
// Inference result (every 2s)
{ "type": "status", "cheating_probability": 0.12, "class_name": "Normal",
  "is_cheating": false, "class_probabilities": {...}, "timestamp": "..." }

// When cheating detected
{ "type": "flag", "flag_id": 42, "cheating_probability": 0.91,
  "class_name": "Gaze/Distraction", "timestamp": "..." }
```

---

## Deployment

### Frontend → Vercel

```bash
cd proctoring_app/frontend
# Connect repo to Vercel
# Build command:  npm run build
# Output dir:     dist
# Environment:    VITE_WS_BASE=wss://your-backend.railway.app
```

### Backend → Railway

```bash
# Add PostgreSQL add-on in Railway dashboard, then set env vars:
DATABASE_URL=postgresql://...   # from Railway add-on
JWT_SECRET=<secrets.token_hex(32)>
CORS_ORIGINS=https://your-app.vercel.app
MODEL_PATH=../../cheating_detection/models/mlp_model.pth
SCALER_PATH=../../cheating_detection/models/scaler.joblib
```

```toml
# railway.toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn proctoring_app.backend.app:app --host 0.0.0.0 --port $PORT"
```

> **Note:** The ML models (`mlp_model.pth`, `scaler.joblib`) must be present at deploy time. Run `python -m cheating_detection.main` locally first, then commit the `cheating_detection/models/` directory, or upload to object storage and set the paths accordingly.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Training | Python, PyTorch, scikit-learn, NumPy, pandas |
| Visual Processing | MediaPipe Face Mesh, OpenCV |
| Backend | FastAPI, SQLAlchemy, pydantic-settings |
| Auth | python-jose (JWT), passlib (bcrypt) |
| Database | PostgreSQL (production), SQLite (development) |
| Real-Time | WebSocket (native FastAPI), asyncio |
| Frontend | React 18, Vite, Tailwind CSS, React Router v6 |
| HTTP Client | Axios with JWT interceptor |
| Deployment | Vercel (frontend), Railway (backend) |

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with PyTorch · FastAPI · React · MediaPipe

</div>
