"""
Hybrid Search Retriever
Combines vector similarity with keyword/BM25 search for better retrieval quality.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import text
from loguru import logger  # pyright: ignore[reportMissingImports]

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings
from src.database.db import get_db
from src.indexing.embeddings import EmbeddingModel


class HybridRetriever:
    """
    Hybrid search retriever combining vector similarity and keyword search.
    
    BUSINESS_RULE: Use both semantic (vector) and lexical (keyword) search to improve
    retrieval quality and reduce noise from high vector similarity but irrelevant content.
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initialize hybrid retriever.
        
        Args:
            embedding_model: Pre-loaded embedding model for vector search
        """
        self.embedding_model = embedding_model
        self.alpha = 0.6  # Weight for vector search (0.6 = 60% vector, 40% keyword)
        logger.info("HybridRetriever initialized")
    
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        min_similarity: Optional[float] = None,
        filter_tables: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: User query text
            top_k: Number of results to return
            alpha: Weight for vector search (0.0 = keyword only, 1.0 = vector only)
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of retrieved segments with hybrid scores
        """
        top_k = top_k or settings.rag.top_k
        alpha = alpha or self.alpha
        min_similarity = min_similarity or 0.0
        
        logger.debug(f"Hybrid search: query='{query[:50]}...', top_k={top_k}, alpha={alpha}")
        
        # Step 1: Vector search
        vector_results = self._vector_search(query, top_k * 2, filter_tables)  # Get more for combination
        
        # Step 2: Keyword search  
        keyword_results = self._keyword_search(query, top_k * 2, filter_tables)
        
        # Step 3: Combine results with hybrid scoring
        combined_results = self._combine_results(
            vector_results, 
            keyword_results, 
            alpha, 
            min_similarity
        )
        
        # Step 4: Return top-k results
        final_results = combined_results[:top_k]
        
        logger.info(f"Hybrid search returned {len(final_results)} results")
        return final_results
    
    def _vector_search(self, query: str, top_k: int, filter_tables: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        logger.debug("Performing vector search...")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode_query(query)
        embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
        
        # Build filter condition
        filter_condition = ""
        if filter_tables:
            table_list = "', '".join(filter_tables)
            filter_condition = f"AND d.source_table IN ('{table_list}')"
        
        with get_db() as db:
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
                    WHERE ds.embedding IS NOT NULL {filter_condition}
                    ORDER BY ds.embedding <=> '{embedding_str}'::vector
                    LIMIT {top_k}
                """)
            ).fetchall()
            
            results = []
            for row in result:
                results.append({
                    "id": row.id,
                    "segment_id": row.segment_id,
                    "text": row.text,
                    "position": row.position,
                    "source_id": row.source_id,
                    "source_table": row.source_table,
                    "title": row.title,
                    "vector_score": float(row.similarity),
                    "keyword_score": 0.0  # Will be updated in combination
                })
            
            logger.debug(f"Vector search found {len(results)} results")
            return results
    
    def _keyword_search(self, query: str, top_k: int, filter_tables: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform keyword search using PostgreSQL full-text search."""
        logger.debug("Performing keyword search...")
        
        # Clean query for full-text search
        clean_query = query.strip().replace("'", "''")  # Escape single quotes
        
        # Build filter condition
        filter_condition = ""
        if filter_tables:
            table_list = "', '".join(filter_tables)
            filter_condition = f"AND d.source_table IN ('{table_list}')"
        
        with get_db() as db:
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
                        ts_rank(to_tsvector('simple', ds.text), plainto_tsquery('simple', :query)) as rank
                    FROM document_segments ds
                    JOIN documents d ON ds.document_id = d.id
                    WHERE to_tsvector('simple', ds.text) @@ plainto_tsquery('simple', :query) {filter_condition}
                    ORDER BY rank DESC
                    LIMIT {top_k}
                """),
                {"query": clean_query}
            ).fetchall()
            
            results = []
            for row in result:
                # Normalize rank to 0-1 scale
                normalized_score = min(float(row.rank) * 2, 1.0)  # Scale and cap at 1.0
                
                results.append({
                    "id": row.id,
                    "segment_id": row.segment_id,
                    "text": row.text,
                    "position": row.position,
                    "source_id": row.source_id,
                    "source_table": row.source_table,
                    "title": row.title,
                    "vector_score": 0.0,  # Will be updated in combination
                    "keyword_score": normalized_score
                })
            
            logger.debug(f"Keyword search found {len(results)} results")
            return results
    
    def _combine_results(
        self, 
        vector_results: List[Dict[str, Any]], 
        keyword_results: List[Dict[str, Any]], 
        alpha: float,
        min_similarity: float
    ) -> List[Dict[str, Any]]:
        """Combine vector and keyword results with hybrid scoring."""
        logger.debug("Combining search results...")
        
        # Create combined dictionary
        combined = {}
        
        # Add vector results
        for result in vector_results:
            result_id = result["id"]
            combined[result_id] = result.copy()
        
        # Add/update keyword results
        for result in keyword_results:
            result_id = result["id"]
            if result_id in combined:
                # Update existing result with keyword score
                combined[result_id]["keyword_score"] = result["keyword_score"]
            else:
                # Add new result from keyword search
                combined[result_id] = result.copy()
        
        # Calculate hybrid scores
        for result_id, result in combined.items():
            vector_score = result.get("vector_score", 0.0)
            keyword_score = result.get("keyword_score", 0.0)
            
            # Hybrid score: weighted combination
            hybrid_score = alpha * vector_score + (1 - alpha) * keyword_score
            result["hybrid_score"] = hybrid_score
            result["similarity"] = hybrid_score  # For compatibility with existing code
        
        # Filter by minimum similarity
        filtered_results = [
            result for result in combined.values() 
            if result["hybrid_score"] >= min_similarity
        ]
        
        # Sort by hybrid score (descending)
        sorted_results = sorted(
            filtered_results, 
            key=lambda x: x["hybrid_score"], 
            reverse=True
        )
        
        logger.debug(f"Combined {len(combined)} results, filtered to {len(sorted_results)}")
        return sorted_results
