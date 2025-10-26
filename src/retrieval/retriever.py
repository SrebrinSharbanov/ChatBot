"""
Document retrieval using pgvector similarity search.
Main RAG retrieval logic with scoring and ranking.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy import text
from loguru import logger  # pyright: ignore[reportMissingImports]

from config.settings import settings
from src.database.db import get_db
from src.database.models import Document, DocumentSegment
from src.indexing.embeddings import get_embedding_model
from .scoring import (
    calculate_score,
    should_answer,
    format_sources,
    get_context_window
)


class Retriever:
    """
    BUSINESS_RULE: Main retrieval class for RAG system.
    Performs semantic search using pgvector and ranks results by similarity.
    """
    
    def __init__(self):
        """Initialize retriever with embedding model."""
        self.embedding_model = get_embedding_model()
        logger.info("Retriever initialized")
    
    def search(
        self,
        query: str,
        top_k: int = None,
        min_similarity: float = None
    ) -> List[Dict[str, Any]]:
        """
        BUSINESS_RULE: Search for relevant documents using vector similarity.
        
        Process:
        1. Encode query to embedding vector
        2. Perform pgvector similarity search (cosine distance)
        3. Return top-k most similar segments with metadata
        
        Args:
            query: User query text
            top_k: Number of results to return (default from config)
            min_similarity: Minimum similarity threshold (optional)
        
        Returns:
            List of retrieved segments with similarity scores
        """
        top_k = top_k or settings.rag.top_k
        min_similarity = min_similarity or 0.0
        
        # PERFORMANCE_CRITICAL: Generate query embedding
        logger.debug(f"Encoding query: {query[:100]}...")
        query_embedding = self.embedding_model.encode_query(query)
        
        # PERFORMANCE_CRITICAL: Vector similarity search using pgvector
        with get_db() as db:
            # Use pgvector's <=> operator for cosine distance
            # cosine_similarity = 1 - cosine_distance
            sql_query = text("""
                SELECT 
                    ds.id,
                    ds.segment_id,
                    ds.text,
                    ds.position,
                    d.source_id,
                    d.source_table,
                    d.title,
                    1 - (ds.embedding <=> :query_embedding) as similarity
                FROM document_segments ds
                JOIN documents d ON ds.document_id = d.id
                WHERE ds.embedding IS NOT NULL
                ORDER BY ds.embedding <=> :query_embedding
                LIMIT :top_k
            """)
            
            # Convert embedding to string format for pgvector
            embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
            
            result = db.execute(
                text(f"""
                    SELECT 
                        ds.id,
                        ds.segment_id,
                        ds.text,
                        ds.position,
                        d.source_id,
                        d.source_table,
                        d.title,
                        1 - (ds.embedding <=> '{embedding_str}'::vector) as similarity
                    FROM document_segments ds
                    JOIN documents d ON ds.document_id = d.id
                    WHERE ds.embedding IS NOT NULL
                    ORDER BY ds.embedding <=> '{embedding_str}'::vector
                    LIMIT {top_k}
                """)
            ).fetchall()
            
            # Format results
            results = []
            for row in result:
                similarity = float(row.similarity)
                
                # Filter by minimum similarity if specified
                if similarity < min_similarity:
                    continue
                
                results.append({
                    "id": row.id,
                    "segment_id": row.segment_id,
                    "text": row.text,
                    "position": row.position,
                    "source_id": row.source_id,
                    "source_table": row.source_table,
                    "title": row.title,
                    "similarity": similarity
                })
            
            logger.debug(f"Found {len(results)} relevant segments")
            return results
    
    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        BUSINESS_RULE: Complete retrieval pipeline with scoring.
        
        Returns:
        - Retrieved segments
        - Confidence score (0-100)
        - Formatted sources
        - Context window for LLM
        - Decision: should_answer (True/False)
        
        Args:
            query: User query text
            top_k: Number of results to retrieve
        
        Returns:
            Dictionary with retrieval results and metadata
        """
        # Search for relevant segments
        results = self.search(query, top_k=top_k)
        
        # Calculate confidence score
        score = calculate_score(results)
        
        # Determine if we should answer
        can_answer = should_answer(score)
        
        # Format sources
        sources = format_sources(results)
        
        # Build context window
        context = get_context_window(results) if can_answer else ""
        
        return {
            "query": query,
            "results": results,
            "score": score,
            "should_answer": can_answer,
            "sources": sources,
            "context": context
        }
    
    def test_retrieval(self, query: str) -> None:
        """
        VALIDATION: Test retrieval and print results.
        Useful for debugging and verification.
        
        Args:
            query: Test query
        """
        logger.info(f"Testing retrieval for: '{query}'")
        
        retrieval_result = self.retrieve(query)
        
        logger.info(f"Score: {retrieval_result['score']}/100")
        logger.info(f"Should answer: {retrieval_result['should_answer']}")
        logger.info(f"Found {len(retrieval_result['results'])} segments")
        
        logger.info("\nTop results:")
        for i, result in enumerate(retrieval_result['results'][:3], 1):
            logger.info(f"{i}. [{result['source_table']}:{result['source_id']}] {result['title']}")
            logger.info(f"   Similarity: {result['similarity']:.4f}")
            logger.info(f"   Text: {result['text'][:150]}...")
        
        if retrieval_result['sources']:
            logger.info("\nSources:")
            for source in retrieval_result['sources']:
                logger.info(f"  - {source['source_table']}:{source['source_id']} ({source['segment_id']})")

