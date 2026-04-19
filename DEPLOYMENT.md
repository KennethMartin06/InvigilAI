# InvigilAI — Production Deployment Guide

End-to-end guide for getting InvigilAI running in production with the full
multi-model ensemble (MLP + RF + SVM + LightGBM + XGBoost + ONNX).

---

## What gets deployed

```
┌────────────────┐    HTTPS    ┌─────────────────┐    WebSocket    ┌──────────────┐
│ React SPA      │◄───────────►│ nginx (proxy)   │◄───────────────►│ FastAPI      │
│ (exam UI +     │             │ frontend:80     │                 │ + ensemble   │
│  admin dash)   │             │                 │                 │ backend:8000 │
└────────────────┘             └─────────────────┘                 └──────┬───────┘
                                                                          │
                                                                   ┌──────┴──────┐
                                                                   │ PostgreSQL  │
                                                                   │ postgres:5432│
                                                                   └─────────────┘
```

- **Frontend**: React 18 + Vite, served through nginx (port 80), reverse
  proxies `/api` and `/ws` to the backend.
- **Backend**: FastAPI with the multi-model ensemble. Loads every trained
  model found in `cheating_detection/models/` at startup. Gracefully skips
  missing ones.
- **Database**: PostgreSQL 16 (replaces the dev SQLite).

---

## Prerequisites

- Docker 24+ and Docker Compose v2
- ~2 GB free disk
- Trained model files in `cheating_detection/models/`. At minimum you need:
  - `mlp_model.pth`
  - `scaler.joblib`
  - Optionally: `rf_model.joblib`, `svm_model.joblib`, `lgb_model.joblib`,
    `xgb_model.joblib`, `mlp_model.onnx`

---

## Option A — Local Docker stack (recommended)

One command:

```bash
./deploy.sh up
```

Opens:
- **Frontend**: http://localhost
- **Backend**: http://localhost:8000
- **Health**: http://localhost:8000/health
- **API docs**: http://localhost:8000/docs

Other commands:

```bash
./deploy.sh logs      # tail backend logs
./deploy.sh test      # hit /health + /api/predict with sample inputs
./deploy.sh down      # stop stack
./deploy.sh rebuild   # rebuild from scratch (no cache)
```

Custom config (optional) — create a `.env` file in the repo root:

```env
POSTGRES_PASSWORD=your_strong_db_password
JWT_SECRET=long_random_secret_at_least_32_chars
CHEATING_THRESHOLD=0.70
```

---

## Option B — Render.com (free cloud deploy)

1. Push the repo to GitHub.
2. Go to https://dashboard.render.com/blueprints.
3. Connect your GitHub repo — Render reads `render.yaml` and provisions:
   - PostgreSQL DB
   - Backend web service
   - Frontend web service
4. Wait ~5 min for first build.
5. Frontend URL: `https://invigilai-frontend.onrender.com`.

Render auto-injects `DATABASE_URL` and generates a `JWT_SECRET`. Update
`CORS_ORIGINS` env var on the backend service once the frontend URL is known.

---

## Option C — AWS EC2 (production, full control)

1. **Launch** a `t3.medium` or larger Ubuntu 22.04 instance with ports 80/443
   open.
2. **SSH in** and install Docker:
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   newgrp docker
   ```
3. **Clone** the repo and `cd` into it.
4. **Create** a `.env` with strong secrets.
5. **Deploy**:
   ```bash
   ./deploy.sh up
   ```
6. (Optional) Add Let's Encrypt via `certbot` + nginx SSL termination for HTTPS.

---

## Verifying deployment

### 1. Health check
```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "ok",
  "mlp_loaded": true,
  "scaler_loaded": true,
  "ensemble_size": 3,
  "models_loaded": ["mlp", "random_forest", "svm"]
}
```

### 2. One-shot prediction
```bash
curl "http://localhost:8000/api/predict?gaze_yaw=35&gaze_deviation_ratio=0.8"
```

Returns per-class probabilities + per-model votes from every loaded model.

### 3. Sign up + login
```bash
# Signup
curl -X POST http://localhost:8000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"t@x.com","name":"Test","password":"pass1234","role":"student"}'

# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"t@x.com","password":"pass1234"}'
```

### 4. Frontend smoke test
Open http://localhost, sign in, join an exam. The `StatusIndicator` should
update every ~2 s with a cheating probability from the ensemble.

---

## Adding more trained models

When you finish training LightGBM / XGBoost / Transformer / TCN / GNN, just
drop the `.joblib` / `.pth` / `.onnx` files into
`cheating_detection/models/` and rebuild:

```bash
./deploy.sh rebuild
```

The ensemble auto-discovers every known filename at startup. Check
`/health` to confirm it picked them up.

---

## Production improvements checklist

| Area          | Action                                                       |
|---------------|--------------------------------------------------------------|
| **TLS/HTTPS** | Put Cloudflare / Let's Encrypt in front of port 80           |
| **Secrets**   | Use a vault (AWS SSM, Doppler) instead of `.env`             |
| **Scaling**   | Set `--workers 4` on uvicorn; add a load balancer            |
| **Logging**   | Pipe Docker logs to CloudWatch / Loki                         |
| **Metrics**   | Add `prometheus-fastapi-instrumentator` to `backend/app.py`  |
| **Rate limit**| Already on (200 req/min via middleware)                      |
| **Model opt** | Export ONNX with `python -m cheating_detection.models.export`|
| **GPU**       | Use `nvidia/cuda:*` base image + `--gpus all` in compose     |
| **DB backup** | `pg_dump` nightly via cron                                    |
| **Frontend**  | Point `VITE_API_BASE` to HTTPS endpoint in production build  |

---

## Troubleshooting

**"No models loaded"** → The `cheating_detection/models/` directory is empty
in the Docker image. Run `python3 cheating_detection/main.py` locally first
to generate model files, then rebuild.

**WebSocket disconnects after 60s** → Nginx `proxy_read_timeout` is set to
3600s already; check your cloud load balancer timeout too.

**High CPU usage** → Switch MLP to ONNX (`mlp_model.onnx`) — it's ~5× faster.
Run `python -m cheating_detection.models.export` to generate it.

**Camera / mic permissions blocked** → Production must run over HTTPS;
browsers block `getUserMedia` on HTTP (except localhost).
