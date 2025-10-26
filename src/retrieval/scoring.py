"""
Scoring logic for RAG system.
Converts similarity scores to confidence scores (0-100).
"""

from typing import List, Dict, Any
from config.settings import settings


def similarity_to_score(similarity: float) -> int:
    """
    BUSINESS_RULE: Convert cosine similarity to confidence score (0-100).
    
    Formula:
    - similarity ≤ min_similarity (0.2) → score = 0
    - similarity ≥ max_similarity (0.8) → score = 100
    - Linear interpolation between min and max
    
    This calibration is based on empirical testing with multilingual-e5-small model.
    The thresholds (0.2 and 0.8) work well for Bulgarian text retrieval.
    
    Args:
        similarity: Cosine similarity in range [-1, 1]
    
    Returns:
        Confidence score in range [0, 100]
    """
    min_sim = settings.rag.min_similarity
    max_sim = settings.rag.max_similarity
    
    # Clamp to boundaries
    if similarity <= min_sim:
        return 0
    if similarity >= max_sim:
        return 100
    
    # Linear mapping from [min_sim, max_sim] to [0, 100]
    score = ((similarity - min_sim) / (max_sim - min_sim)) * 100
    return int(round(score))


def calculate_score(results: List[Dict[str, Any]]) -> int:
    """
    BUSINESS_RULE: Calculate overall confidence score from retrieval results.
    Uses the maximum similarity among top-k results as the primary indicator.
    
    Rationale:
    - If at least one highly relevant document is found, we can answer confidently
    - The top result's similarity is the best indicator of answer quality
    - We use a threshold of 80 to ensure high-quality answers only
    
    Args:
        results: List of retrieval results with 'similarity' field
    
    Returns:
        Confidence score in range [0, 100]
    """
    if not results:
        return 0
    
    # Use maximum similarity as score indicator
    max_similarity = max(r['similarity'] for r in results)
    return similarity_to_score(max_similarity)


def should_answer(score: int, threshold: int = None) -> bool:
    """
    BUSINESS_RULE: Determine if system should generate answer or decline.
    
    If score < threshold:
        Return: "Това не е в моята компетенция."
    Else:
        Generate answer from retrieved context
    
    Args:
        score: Confidence score (0-100)
        threshold: Minimum score threshold (default from config)
    
    Returns:
        True if should answer, False if should decline
    """
    threshold = threshold or settings.rag.score_threshold
    return score >= threshold


def format_sources(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    BUSINESS_RULE: Format source citations for response.
    
    Returns source identifiers in format: "{table}:{id}"
    Example: "policy:2", "faq:5", "product:42"
    
    Args:
        results: List of retrieval results
    
    Returns:
        List of source dictionaries with formatted identifiers
    """
    sources = []
    seen = set()
    
    for result in results:
        source_key = f"{result['source_table']}:{result['source_id']}"
        
        # Avoid duplicate sources
        if source_key not in seen:
            sources.append({
                "source_id": result['source_id'],
                "source_table": result['source_table'],
                "segment_id": result['segment_id'],
                "title": result.get('title', ''),
                "similarity": result['similarity']
            })
            seen.add(source_key)
    
    return sources


def get_context_window(
    results: List[Dict[str, Any]],
    max_length: int = None
) -> str:
    """
    BUSINESS_RULE: Build context window for LLM from retrieval results.
    Combines top-k segments into single context string.
    
    Args:
        results: List of retrieval results
        max_length: Maximum context length (default from config)
    
    Returns:
        Formatted context string for LLM
    """
    max_length = max_length or settings.rag.max_context_length
    
    context_parts = []
    current_length = 0
    
    for i, result in enumerate(results, 1):
        title = result.get('title', '')
        text = result.get('text', '')
        
        # Format: [Source] Title\nText
        source_label = f"[{result['source_table']}:{result['source_id']}]"
        part = f"{source_label} {title}\n{text}"
        
        # Check if adding this part would exceed limit
        if current_length + len(part) > max_length:
            # Try to add at least first result
            if i == 1:
                context_parts.append(part[:max_length])
            break
        
        context_parts.append(part)
        current_length += len(part) + 4  # +4 for separator
    
    return "\n\n---\n\n".join(context_parts)

