"""Production-grade async SQLite database engine.

Uses SQLAlchemy 2.0 async API with aiosqlite for non-blocking
database I/O compatible with FastAPI's async request lifecycle.
"""

from __future__ import annotations

import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)

# ── Database path ────────────────────────────────────────────────────────────
DB_DIR = Path(__file__).resolve().parent / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "jobsearch.db"

DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# ── Async Engine ─────────────────────────────────────────────────────────────
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False},
)

# ── Session Factory ──────────────────────────────────────────────────────────
async_session_maker = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── ORM Base ─────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ── Lifecycle ────────────────────────────────────────────────────────────────
async def init_db() -> None:
    """Create all tables if they don't exist (startup hook)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables initialized at %s", DB_PATH)


async def close_db() -> None:
    """Dispose of the engine connection pool (shutdown hook)."""
    await engine.dispose()
    logger.info("🔌 Database engine disposed")


# ── Dependency Injection ─────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session.

    Usage:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
