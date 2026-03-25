"""
middleware.py — CORS, request logging, and simple rate limiting.
"""

import logging
import time
from collections import defaultdict

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings

logger = logging.getLogger(__name__)


def setup_cors(app) -> None:
    """
    Add CORSMiddleware to the FastAPI app with origins from settings.

    Parameters
    ----------
    app : FastAPI
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log method, path, status code, and response time for every HTTP request.
    WebSocket upgrades are skipped to avoid noise.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d  (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter: max requests_per_minute per client IP.

    Not suitable for multi-process deployments — use Redis in production.
    """

    def __init__(self, app, requests_per_minute: int = 120) -> None:
        super().__init__(app)
        self._limit = requests_per_minute
        self._counts: dict = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip WebSocket upgrades
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60.0

        # Purge old entries
        self._counts[client_ip] = [
            t for t in self._counts[client_ip] if t > window_start
        ]

        if len(self._counts[client_ip]) >= self._limit:
            return Response(
                content='{"detail":"Too many requests"}',
                status_code=429,
                media_type="application/json",
            )

        self._counts[client_ip].append(now)
        return await call_next(request)
