# syntax=docker/dockerfile:1.6
FROM node:20-alpine AS builder
WORKDIR /app
COPY proctoring_app/frontend/package*.json ./
RUN npm ci --prefer-offline --no-audit
COPY proctoring_app/frontend/ ./
ARG VITE_API_BASE=/api
ARG VITE_WS_BASE=
ENV VITE_API_BASE=$VITE_API_BASE VITE_WS_BASE=$VITE_WS_BASE
RUN npm run build

FROM nginx:1.27-alpine
EXPOSE 80
# Generate nginx config inline to avoid COPY issues
RUN cat > /etc/nginx/conf.d/default.conf <<'EOF'
upstream backend { server backend:8000; }
server {
  listen 80; server_name _;
  root /usr/share/nginx/html; index index.html;
  gzip on; gzip_types text/plain text/css application/json application/javascript;
  location = /health { access_log off; return 200 "ok\n"; add_header Content-Type text/plain; }
  location /api/ { proxy_pass http://backend; proxy_http_version 1.1; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; proxy_set_header X-Forwarded-Proto $scheme; }
  location /ws/ { proxy_pass http://backend; proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade"; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; }
  location / { try_files $uri $uri/ /index.html; }
  location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff2?)$ { expires 1y; add_header Cache-Control "public, immutable"; }
}
EOF
COPY --from=builder /app/dist /usr/share/nginx/html
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD wget -qO- http://localhost/health || exit 1
CMD ["nginx", "-g", "daemon off;"]

