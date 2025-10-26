"""
Database connection and session management.
Handles PostgreSQL connection with pgvector extension.
"""

from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from loguru import logger

from config.settings import settings
from .models import Base


# Global engine and session factory
_engine = None
_SessionLocal = None


def init_db() -> None:
    """
    BUSINESS_RULE: Initialize database connection and create tables.
    Creates pgvector extension if not exists.
    Should be called once at application startup.
    """
    global _engine, _SessionLocal

    try:
        # Create engine with connection pooling
        _engine = create_engine(
            settings.database.url,
            poolclass=QueuePool,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_pre_ping=True,  # Verify connections before using
            echo=settings.api.debug,
        )

        # Create session factory
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

        # Create pgvector extension
        with _engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logger.info("✓ pgvector extension created/verified")

        # Create all tables
        Base.metadata.create_all(bind=_engine)
        logger.info("✓ Database tables created/verified")

    except Exception as e:
        logger.error(f"✗ Failed to initialize database: {e}")
        raise


def close_db() -> None:
    """
    BUSINESS_RULE: Close database connections.
    Should be called at application shutdown.
    """
    global _engine
    if _engine:
        _engine.dispose()
        logger.info("✓ Database connections closed")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    BUSINESS_RULE: Get database session as context manager.
    Ensures proper session lifecycle and transaction management.
    
    Usage:
        with get_db() as db:
            results = db.query(Document).all()
    """
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    db = _SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database transaction error: {e}")
        raise
    finally:
        db.close()


def get_db_session() -> Session:
    """
    BUSINESS_RULE: Get database session for dependency injection (FastAPI).
    Used in API endpoints for automatic session management.
    
    Usage in FastAPI:
        @app.get("/")
        def endpoint(db: Session = Depends(get_db_session)):
            return db.query(Document).all()
    """
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection() -> bool:
    """
    VALIDATION: Test database connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with get_db() as db:
            result = db.execute(text("SELECT 1")).scalar()
            return result == 1
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def get_db_connection():
    """
    BUSINESS_RULE: Get raw database connection for direct SQL queries.
    Used for products query handler.
    
    Returns:
        Raw database connection
    """
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=settings.database.host,
            port=settings.database.port,
            database=settings.database.database,
            user=settings.database.user,
            password=settings.database.password
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to get database connection: {e}")
        raise

