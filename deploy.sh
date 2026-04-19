#!/usr/bin/env bash
# deploy.sh — one-command local deploy of InvigilAI via docker-compose.
#
# Usage:
#   ./deploy.sh up        # build & run
#   ./deploy.sh down      # stop & remove containers
#   ./deploy.sh logs      # tail backend logs
#   ./deploy.sh test      # hit /health and /api/predict to verify
#   ./deploy.sh rebuild   # force rebuild (--no-cache)

set -euo pipefail

cd "$(dirname "$0")"

cmd="${1:-up}"

case "$cmd" in
  up)
    echo ">>> Building and starting InvigilAI stack..."
    docker compose up -d --build
    echo ""
    echo ">>> Stack is up!"
    echo "    Frontend:  http://localhost"
    echo "    Backend:   http://localhost:8000"
    echo "    Health:    http://localhost:8000/health"
    echo "    API docs:  http://localhost:8000/docs"
    ;;

  rebuild)
    echo ">>> Rebuilding without cache..."
    docker compose build --no-cache
    docker compose up -d
    ;;

  down)
    echo ">>> Stopping stack..."
    docker compose down
    ;;

  logs)
    docker compose logs -f backend
    ;;

  test)
    echo ">>> Health check..."
    curl -fsS http://localhost:8000/health | python3 -m json.tool
    echo ""
    echo ">>> Prediction test (default values)..."
    curl -fsS "http://localhost:8000/api/predict" | python3 -m json.tool
    echo ""
    echo ">>> Prediction test (suspicious input — high gaze deviation)..."
    curl -fsS "http://localhost:8000/api/predict?gaze_yaw=35&gaze_deviation_ratio=0.8&face_count=2" \
      | python3 -m json.tool
    ;;

  *)
    echo "Unknown command: $cmd"
    echo "Usage: $0 {up|down|rebuild|logs|test}"
    exit 1
    ;;
esac
