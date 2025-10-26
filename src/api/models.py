"""
API request/response models (Pydantic schemas).
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for chatbot query"""
    q: str = Field(..., description="User question", min_length=1, max_length=1000)
    k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    threshold: Optional[int] = Field(None, description="Score threshold (0-100)", ge=0, le=100)
    filter_tables: Optional[List[str]] = Field(None, description="Filter by source tables (policies, faq, products)")
    search_type: Optional[str] = Field("hybrid", description="Search type: hybrid, vector, keyword")
    alpha: Optional[float] = Field(0.6, description="Hybrid search weight (0.0=keyword only, 1.0=vector only)", ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "q": "Каква е политиката за връщане на продукти?",
                "k": 5
            }
        }


class SourceModel(BaseModel):
    """Source citation model"""
    source_id: str = Field(..., description="Source document ID")
    source_table: str = Field(..., description="Source table name")
    segment_id: str = Field(..., description="Segment identifier")
    title: str = Field("", description="Document title")
    similarity: float = Field(..., description="Similarity score")


class QueryResponse(BaseModel):
    """Response model for chatbot query"""
    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer or decline message")
    score: int = Field(..., description="Confidence score (0-100)", ge=0, le=100)
    sources: List[SourceModel] = Field(default_factory=list, description="Source citations")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")
    intent: Optional[str] = Field(None, description="Recognized intent category")
    intent_confidence: Optional[float] = Field(None, description="Intent recognition confidence")
    processed_query: Optional[str] = Field(None, description="Processed/expanded query")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Каква е политиката за връщане?",
                "answer": "Можете да върнете продукти в рамките на 30 дни...",
                "score": 92,
                "sources": [
                    {
                        "source_id": "1",
                        "source_table": "policies",
                        "segment_id": "policies:1:seg0",
                        "title": "Политика за връщане",
                        "similarity": 0.87
                    }
                ],
                "confidence": "high"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall health status")
    retriever: str = Field(..., description="Retriever health")
    generator: str = Field(..., description="Generator health")
    version: str = Field(..., description="Application version")


class StatsResponse(BaseModel):
    """Statistics response"""
    documents: int = Field(..., description="Total documents in index")
    segments: int = Field(..., description="Total segments in index")
    total_queries: int = Field(..., description="Total queries processed")

