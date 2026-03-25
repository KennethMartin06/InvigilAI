"""
database.py — SQLAlchemy engine, session factory, and declarative Base.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .config import settings


# SQLite: connect_args needed for check_same_thread
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {},
    echo=False,
)

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
