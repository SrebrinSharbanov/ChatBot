#!/usr/bin/env python3
"""
Direct database migration for BGE-M3 using psycopg2.
"""

import psycopg2
from loguru import logger
import os

def migrate_database():
    """Direct migration using psycopg2 connection."""
    
    # Database connection parameters
    conn_params = {
        'host': 'postgres',
        'port': 5432,
        'database': 'rag_chatbot',
        'user': 'raguser',
        'password': 'ragpassword'
    }
    
    try:
        logger.info("Connecting to database...")
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Step 1: Drop existing index
        logger.info("Dropping existing vector index...")
        cursor.execute("DROP INDEX IF EXISTS idx_segment_embedding;")
        
        # Step 2: Alter column dimension
        logger.info("Altering embedding column to 1024 dimensions...")
        cursor.execute("ALTER TABLE document_segments ALTER COLUMN embedding TYPE vector(1024);")
        
        # Step 3: Recreate index
        logger.info("Creating new vector index for 1024 dimensions...")
        cursor.execute("""
            CREATE INDEX idx_segment_embedding
            ON document_segments USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        logger.success("✓ Migration completed successfully!")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"✗ Migration failed: {e}")
        raise

if __name__ == "__main__":
    migrate_database()
