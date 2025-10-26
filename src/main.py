"""
Mini RAG Chatbot - FastAPI Application
Main entry point for the API server.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.database.db import init_db, close_db
from src.api.routes import router


# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format=settings.logging.format if hasattr(settings, 'logging') else "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level=settings.logging.level if hasattr(settings, 'logging') else "INFO"
)

# Add file logger if configured
if hasattr(settings, 'logging') and hasattr(settings.logging, 'log_file'):
    log_file = Path(settings.logging.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_file),
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        level=settings.logging.level
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    BUSINESS_RULE: Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 80)
    logger.info("Mini RAG Chatbot - Starting up...")
    logger.info("=" * 80)
    
    try:
        # Initialize database
        logger.info("Initializing database connection...")
        init_db()
        logger.success("✓ Database initialized")
        
        # Pre-load embedding model (lazy loading alternative)
        logger.info("Pre-loading embedding model...")
        from src.indexing.embeddings import get_embedding_model
        model = get_embedding_model()
        logger.success(f"✓ Embedding model loaded: {model.model_name}")
        
        # Initialize chatbot
        logger.info("Initializing RAG chatbot...")
        from src.chatbot.rag import RAGChatbot
        chatbot = RAGChatbot()
        logger.success("✓ RAG chatbot initialized")
        
        logger.info("=" * 80)
        logger.success("✓ Application started successfully")
        logger.info(f"   API: http://{settings.api.host}:{settings.api.port}")
        logger.info(f"   Docs: http://{settings.api.host}:{settings.api.port}/docs")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        logger.exception(e)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    close_db()
    logger.info("✓ Application shut down successfully")


# Create FastAPI application
app = FastAPI(
    title="Mini RAG Chatbot API",
    description="RAG-based chatbot with pgvector, multilingual embeddings, and Ollama LLM",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["chatbot"])


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Mini RAG Chatbot",
        "version": "1.0.0",
        "status": "running",
        "api_docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers,
        log_level=settings.logging.level.lower() if hasattr(settings, 'logging') else "info"
    )

