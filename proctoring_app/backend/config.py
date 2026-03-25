"""
config.py — Application settings loaded from .env via pydantic-settings.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./proctoring.db"

    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_secret: str = "changeme-use-a-long-random-string-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24

    # ── ML Models ─────────────────────────────────────────────────────────────
    model_path: str = "../cheating_detection/models/mlp_model.pth"
    scaler_path: str = "../cheating_detection/models/scaler.joblib"
    cheating_threshold: float = 0.70

    # ── File storage ──────────────────────────────────────────────────────────
    upload_dir: str = "./uploads"

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance (singleton pattern)."""
    return Settings()


settings = get_settings()
