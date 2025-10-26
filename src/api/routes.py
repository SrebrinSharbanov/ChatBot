"""
FastAPI routes for RAG chatbot API.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from loguru import logger  # pyright: ignore[reportMissingImports]
import time

from config.settings import settings
from src.database.db import get_db_session
from src.database.models import Document, DocumentSegment, QueryLog
from src.chatbot.rag import RAGChatbot
from src.retrieval.hybrid_retriever import HybridRetriever
from src.indexing.embeddings import EmbeddingModel
from .models import QueryRequest, QueryResponse, HealthResponse, StatsResponse

# Create router
router = APIRouter()

# Global instances (initialized on startup)
_chatbot: RAGChatbot = None
_hybrid_retriever: HybridRetriever = None
_embedding_model: EmbeddingModel = None


def get_chatbot() -> RAGChatbot:
    """Get global chatbot instance."""
    global _chatbot
    if _chatbot is None:
        _chatbot = RAGChatbot()
    return _chatbot


def get_hybrid_retriever() -> HybridRetriever:
    """Get global hybrid retriever instance."""
    global _hybrid_retriever, _embedding_model
    if _hybrid_retriever is None:
        if _embedding_model is None:
            _embedding_model = EmbeddingModel()
        _hybrid_retriever = HybridRetriever(_embedding_model)
    return _hybrid_retriever


@router.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query_chatbot(
    request: QueryRequest,
    db: Session = Depends(get_db_session)
) -> QueryResponse:
    """
    BUSINESS_RULE: Main chatbot query endpoint.
    
    Process:
    1. Receive user question
    2. Retrieve relevant documents
    3. Calculate confidence score
    4. Generate answer (if score >= threshold)
    5. Return answer + sources + score
    6. Log query for analytics
    
    Returns:
    - answer: Generated text or "Това не е в моята компетенция"
    - score: Confidence (0-100)
    - sources: List of source citations
    """
    start_time = time.time()
    
    try:
        # Get chatbot instance
        chatbot = get_chatbot()
        
        # Process query
        logger.info(f"API query: {request.q}")
        result = chatbot.ask(
            query=request.q,
            top_k=request.k,
            threshold=request.threshold
        )
        
        # Log query to database
        response_time_ms = int((time.time() - start_time) * 1000)
        query_log = QueryLog(
            query_text=request.q,
            score=result['score'],
            answer=result['answer'],
            sources=str(result['sources']),
            response_time_ms=response_time_ms
        )
        db.add(query_log)
        db.commit()
        
        logger.info(f"Query completed in {response_time_ms}ms (score: {result['score']})")
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """
    VALIDATION: Check API and component health.
    
    Returns health status of:
    - Retriever (database + embeddings)
    - Generator (LLM)
    - Overall system
    """
    try:
        chatbot = get_chatbot()
        health = chatbot.health_check()
        
        return HealthResponse(
            status=health.get("overall", "unknown"),
            retriever=health.get("retriever", "unknown"),
            generator=health.get("generator", "unknown"),
            version=settings.__version__ if hasattr(settings, '__version__') else "1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="error",
            retriever="unknown",
            generator="unknown",
            version="1.0.0"
        )


@router.get("/stats", response_model=StatsResponse, summary="Get statistics")
async def get_stats(db: Session = Depends(get_db_session)):
    """
    Get system statistics:
    - Number of indexed documents
    - Number of segments
    - Total queries processed
    """
    try:
        doc_count = db.query(Document).count()
        segment_count = db.query(DocumentSegment).count()
        query_count = db.query(QueryLog).count()
        
        return StatsResponse(
            documents=doc_count,
            segments=segment_count,
            total_queries=query_count
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.get("/", summary="API information")
async def root():
    """API root endpoint with basic information."""
    return {
        "name": "Mini RAG Chatbot API",
        "version": "1.0.0",
        "description": "RAG-based chatbot with scoring and source citation",
        "endpoints": {
            "POST /api/query": "Ask a question",
            "GET /api/health": "Health check",
            "GET /api/stats": "Statistics",
            "GET /api/faq": "Get FAQ data",
            "GET /api/frontend": "Frontend interface",
            "GET /docs": "API documentation (Swagger UI)",
            "GET /redoc": "API documentation (ReDoc)"
        }
    }


@router.get("/faq", summary="Get FAQ data")
async def get_faq(db: Session = Depends(get_db_session)):
    """Get FAQ data from the database."""
    try:
        # Query FAQ data from documents table
        from sqlalchemy import text
        faq_data = db.execute(
            text("SELECT title as question, content as answer FROM documents WHERE source_table = 'faq' ORDER BY id")
        ).fetchall()
        
        return [
            {
                "question": row.question,
                "answer": row.answer
            }
            for row in faq_data
        ]
    except Exception as e:
        logger.error(f"FAQ retrieval error: {e}")
        return []


@router.get("/frontend", response_class=HTMLResponse, summary="Frontend interface")
async def get_frontend():
    """Serve the frontend interface."""
    try:
        with open("src/frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Please check if src/frontend/index.html exists.</p>",
            status_code=404
        )


@router.post("/query/hybrid", response_model=QueryResponse, summary="Hybrid search query")
async def query_hybrid(
    request: QueryRequest,
    db: Session = Depends(get_db_session)
) -> QueryResponse:
    """
    Hybrid search endpoint combining vector and keyword search.
    
    Args:
        request: Query request with optional search parameters
        db: Database session
        
    Returns:
        QueryResponse with hybrid search results
    """
    start_time = time.time()
    
    try:
        # Get hybrid retriever
        hybrid_retriever = get_hybrid_retriever()
        
        # Extract search parameters from request
        alpha = getattr(request, 'alpha', 0.6)
        search_type = getattr(request, 'search_type', 'hybrid')
        
        logger.info(f"Hybrid query: {request.q} (type: {search_type}, alpha: {alpha})")
        
        # Perform hybrid search
        if search_type == 'hybrid':
            results = hybrid_retriever.search(
                query=request.q,
                top_k=request.k,
                alpha=alpha
            )
        elif search_type == 'vector':
            results = hybrid_retriever._vector_search(request.q, request.k)
        elif search_type == 'keyword':
            results = hybrid_retriever._keyword_search(request.q, request.k)
        else:
            raise ValueError(f"Invalid search_type: {search_type}")
        
        # Calculate overall score
        if results:
            max_score = max(result.get('hybrid_score', result.get('similarity', 0)) for result in results)
            overall_score = int(max_score * 100)
        else:
            overall_score = 0
        
        # Generate answer if score is high enough
        if overall_score >= (request.threshold or settings.rag.score_threshold):
            # Use chatbot for answer generation
            chatbot = get_chatbot()
            answer = chatbot.generate_answer(request.q, results)
        else:
            answer = "Това не е в моята компетенция."
        
        # Format sources
        sources = []
        for result in results:
            sources.append({
                "segment_id": result.get("segment_id", ""),
                "source_id": result.get("source_id", ""),
                "source_table": result.get("source_table", ""),
                "title": result.get("title", ""),
                "text": result.get("text", ""),
                "similarity": result.get("hybrid_score", result.get("similarity", 0))
            })
        
        # Log query
        response_time_ms = int((time.time() - start_time) * 1000)
        query_log = QueryLog(
            query_text=request.q,
            score=overall_score,
            answer=answer,
            sources=str(sources),
            response_time_ms=response_time_ms
        )
        db.add(query_log)
        db.commit()
        
        logger.info(f"Hybrid query completed in {response_time_ms}ms (score: {overall_score})")
        
        return QueryResponse(
            answer=answer,
            score=overall_score,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Hybrid query error: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

