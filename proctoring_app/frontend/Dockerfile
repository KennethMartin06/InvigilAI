# syntax=docker/dockerfile:1.6
#
# Frontend image — build React app then serve via nginx with reverse proxy
# to the backend.

# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY proctoring_app/frontend/package*.json ./
RUN npm ci --prefer-offline --no-audit

COPY proctoring_app/frontend/ ./

# Build-time API base (injected at build via --build-arg VITE_API_BASE=...)
ARG VITE_API_BASE=/api
ARG VITE_WS_BASE=
ENV VITE_API_BASE=$VITE_API_BASE
ENV VITE_WS_BASE=$VITE_WS_BASE

RUN npm run build

# ── Stage 2: serve via nginx ──────────────────────────────────────────────────
FROM nginx:1.27-alpine

COPY proctoring_app/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
