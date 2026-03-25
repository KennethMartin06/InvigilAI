# InvigilAI — Full-Stack AI Proctoring Application

Real-time online exam proctoring system that uses the trained ML models from
`cheating_detection/` to detect cheating behaviour live via webcam and
keyboard/mouse tracking.

---

## Architecture

```
Browser (React + Vite)
  │
  ├── REST  → POST /api/auth/*, /api/exam/*, /api/admin/*, /api/flags/*
  │
  └── WebSocket  ws://host:8000/ws/proctor/{session_id}?token=<jwt>
        │
        ├── Frontend sends:
        │     {"type":"frame",    "data":"<base64 JPEG>"}   every 1.5 s
        │     {"type":"behavior", "keystrokes":[...], "mouse":[...]}  every 2 s
        │
        └── Backend responds:
              {"type":"status", "cheating_probability":0.12, "class_name":"Normal", ...}
              {"type":"flag",   "flag_id":42, "class_name":"Gaze/Distraction", ...}

FastAPI Backend
  ├── inference.py      — loads MLP (.pth) + scaler (.joblib), runs predict()
  ├── frame_processor.py — MediaPipe Face Mesh → 8 visual features
  ├── behavior_processor.py — keystroke/mouse events → 8 behavioral features
  ├── feature_fusion.py — 16-dim vector → inference.predict()
  ├── websocket_handler.py — orchestrates 2-second inference cycle
  ├── auth.py           — JWT (HS256), bcrypt passwords
  └── SQLite via SQLAlchemy ORM
        Users, ExamSessions, Flags, ExamQuestions, StudentAnswers
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- The trained models in `../cheating_detection/models/`:
  - `mlp_model.pth`
  - `scaler.joblib`

---

## Backend Setup

```bash
cd proctoring_app/backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — set JWT_SECRET to a random string
# Verify MODEL_PATH and SCALER_PATH point to the correct files

# Seed demo data (admin + students + 10 questions)
python seed_data.py

# Start the server
uvicorn proctoring_app.backend.app:app --reload --port 8000
# OR from the proctoring_app/backend directory:
# uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## Frontend Setup

```bash
cd proctoring_app/frontend

npm install
npm run dev
```

The app will be available at `http://localhost:5173`.

---

## Default Credentials

| Role    | Email                   | Password    |
|---------|-------------------------|-------------|
| Admin   | admin@proctor.ai        | admin123    |
| Student | student1@proctor.ai     | student123  |
| Student | student2@proctor.ai     | student123  |
| Student | student3@proctor.ai     | student123  |

---

## End-to-End Demo Flow

1. Open `http://localhost:5173`
2. Log in as `student1@proctor.ai / student123`
3. ExamLobby — allow camera permission, select "General Knowledge", click Start
4. ExamPage — answer questions; the proctoring status badge updates in real-time
5. Submit the exam → ExamComplete summary page
6. Log out → log in as `admin@proctor.ai / admin123`
7. AdminDashboard — see stats, completed sessions, flag distribution
8. Click any session → SessionReview — view flag timeline, probability chart, feature details

---

## API Reference

| Method | Endpoint                          | Auth     | Description                    |
|--------|-----------------------------------|----------|--------------------------------|
| POST   | /api/auth/signup                  | —        | Register new user              |
| POST   | /api/auth/login                   | —        | Login, get JWT                 |
| GET    | /api/auth/me                      | Bearer   | Current user info              |
| POST   | /api/exam/start                   | Student  | Create session + get questions |
| POST   | /api/exam/answer                  | Student  | Save/update answer             |
| POST   | /api/exam/end/{session_id}        | Student  | Submit exam                    |
| GET    | /api/exam/questions/{title}       | Bearer   | Get questions for exam         |
| GET    | /api/flags/{session_id}           | Bearer   | Flags for a session            |
| PATCH  | /api/flags/{flag_id}/acknowledge  | Admin    | Mark flag reviewed             |
| GET    | /api/admin/sessions               | Admin    | All sessions (filterable)      |
| GET    | /api/admin/sessions/active        | Admin    | Live sessions only             |
| GET    | /api/admin/stats                  | Admin    | Aggregate stats                |
| GET    | /api/admin/student/{id}/history   | Admin    | All sessions for a student     |
| WS     | /ws/proctor/{session_id}?token=   | JWT      | Real-time proctoring stream    |
| GET    | /health                           | —        | Server health + model status   |

---

## Project Structure

```
proctoring_app/
├── backend/
│   ├── app.py                  FastAPI entry point + lifespan
│   ├── config.py               Settings from .env
│   ├── database.py             SQLAlchemy engine + session
│   ├── models_db.py            ORM models
│   ├── schemas.py              Pydantic schemas
│   ├── auth.py                 JWT + bcrypt
│   ├── inference.py            MLP loader + predict()
│   ├── frame_processor.py      MediaPipe → 8 visual features
│   ├── behavior_processor.py   Events → 8 behavioral features
│   ├── feature_fusion.py       16-dim fusion → inference
│   ├── websocket_handler.py    Real-time WS inference loop
│   ├── middleware.py           CORS, logging, rate limiting
│   ├── routes/
│   │   ├── auth_routes.py
│   │   ├── exam_routes.py
│   │   ├── admin_routes.py
│   │   └── flag_routes.py
│   ├── seed_data.py
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx             Router + auth guards
    │   ├── pages/              LoginPage, SignupPage, ExamLobby,
    │   │                       ExamPage, ExamComplete,
    │   │                       AdminDashboard, SessionReview
    │   ├── components/         CameraFeed, StatusIndicator,
    │   │                       FlagTimeline, ExamTimer,
    │   │                       QuestionPanel, WarningBanner,
    │   │                       LiveSessionCard, StatsOverview,
    │   │                       ProtectedRoute, Navbar
    │   ├── hooks/              useCamera, useWebSocket,
    │   │                       useKeystrokes, useMouseTracker, useAuth
    │   ├── services/           api.js (Axios), websocket.js
    │   ├── context/            AuthContext
    │   └── utils/              featureHelpers.js
    ├── package.json
    ├── vite.config.js
    └── tailwind.config.js
```
