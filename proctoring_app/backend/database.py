"""
database.py — SQLAlchemy engine, session factory, and declarative Base.

Supports both PostgreSQL (production) and SQLite (local development).
The driver is selected automatically from DATABASE_URL:
  postgresql://...  → psycopg2, connection pooling enabled
  sqlite:///...     → SQLite, check_same_thread disabled
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .config import settings


def _build_engine():
    """
    Create a SQLAlchemy engine tuned for the configured database backend.

    PostgreSQL: uses NullPool-safe defaults with a pre-ping to recover
    from stale connections (common on Railway / Render after idle periods).

    SQLite: disables the same-thread check required for FastAPI's async
    request handling.

    Returns
    -------
    sqlalchemy.engine.Engine
    """
    url = settings.database_url

    if url.startswith("postgresql") or url.startswith("postgres"):
        # Normalise legacy "postgres://" scheme (Railway uses this)
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return create_engine(
            url,
            pool_pre_ping=True,   # detect stale connections automatically
            pool_size=5,          # max persistent connections
            max_overflow=10,      # burst connections above pool_size
            echo=False,
        )

    # SQLite fallback — local development only
    return create_engine(
        url,
        connect_args={"check_same_thread": False},
        echo=False,
    )


engine = _build_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def get_db():
    """
    FastAPI dependency that yields a database session and ensures cleanup.

    Yields
    ------
    sqlalchemy.orm.Session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all tables defined in ORM models if they don't exist."""
    from . import models_db  # noqa: F401 — registers models with Base
    Base.metadata.create_all(bind=engine)
