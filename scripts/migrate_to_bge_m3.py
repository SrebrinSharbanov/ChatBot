#!/usr/bin/env python3
"""
Migration script to update database schema for BGE-M3 model.
Changes vector dimension from 384 to 1024 and rebuilds index.
"""

import sys
import os
sys.path.append('/app')

from sqlalchemy import text
from loguru import logger
from src.database.db import get_db

def migrate_to_bge_m3():
    """
    BUSINESS_RULE: Migrate database schema for BGE-M3 model.
    
    Steps:
    1. Drop existing vector index
    2. Alter column dimension from 384 to 1024
    3. Recreate vector index with new dimension
    """
    logger.info("Starting migration to BGE-M3 (1024 dimensions)")
    
    try:
        with get_db() as db:
            # Step 1: Drop existing index
            logger.info("Dropping existing vector index...")
            db.execute(text("DROP INDEX IF EXISTS idx_segment_embedding;"))
            db.commit()
            
            # Step 2: Alter column dimension
            logger.info("Altering embedding column to 1024 dimensions...")
            db.execute(text("ALTER TABLE document_segments ALTER COLUMN embedding TYPE vector(1024);"))
            db.commit()
            
            # Step 3: Recreate index
            logger.info("Creating new vector index for 1024 dimensions...")
            db.execute(text("""
                CREATE INDEX idx_segment_embedding
                ON document_segments USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """))
            db.commit()
            
            logger.success("✓ Migration completed successfully!")
            logger.info("Database schema updated for BGE-M3 model")
            
    except Exception as e:
        logger.error(f"✗ Migration failed: {e}")
        raise

if __name__ == "__main__":
    migrate_to_bge_m3()
