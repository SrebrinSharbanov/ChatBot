-- Initialize PostgreSQL database with pgvector extension
-- This script should run automatically on first database creation

-- Create pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Create custom functions for vector operations (if needed)

-- Function to calculate cosine similarity (alternative to built-in)
CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
RETURNS float
AS $$
BEGIN
    RETURN 1 - (a <=> b);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- Function to calculate normalized score (0-100) from similarity
CREATE OR REPLACE FUNCTION similarity_to_score(
    similarity float,
    min_sim float DEFAULT 0.2,
    max_sim float DEFAULT 0.8
)
RETURNS integer
AS $$
BEGIN
    -- Clamp similarity to [min_sim, max_sim]
    IF similarity <= min_sim THEN
        RETURN 0;
    ELSIF similarity >= max_sim THEN
        RETURN 100;
    ELSE
        -- Linear mapping to [0, 100]
        RETURN ROUND(((similarity - min_sim) / (max_sim - min_sim)) * 100)::integer;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- Create index for efficient vector search (will be created by SQLAlchemy model, but kept here for reference)
-- CREATE INDEX IF NOT EXISTS idx_segment_embedding 
-- ON document_segments 
-- USING ivfflat (embedding vector_cosine_ops) 
-- WITH (lists = 100);

-- Grant permissions (adjust username as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO raguser;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO raguser;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Database initialized successfully with pgvector extension';
END $$;

