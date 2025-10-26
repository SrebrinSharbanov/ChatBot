"""
Database models for Mini RAG Chatbot.
Uses SQLAlchemy with pgvector extension for vector similarity search.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Document(Base):
    """
    BUSINESS_RULE: Represents source documents (policies, FAQ, products).
    Each document can have multiple segments for better retrieval.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(String(100), nullable=False, index=True, comment="Original ID from source (e.g., policy:2)")
    source_table = Column(String(50), nullable=False, index=True, comment="Source table (policy, faq, product)")
    title = Column(String(500), nullable=False, comment="Document title/name")
    content = Column(Text, nullable=False, comment="Full document content")
    meta_data = Column(Text, nullable=True, comment="Additional metadata (JSON)")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    segments = relationship("DocumentSegment", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, source_id='{self.source_id}', table='{self.source_table}')>"


class DocumentSegment(Base):
    """
    BUSINESS_RULE: Represents text segments for vector search.
    Documents are split into segments for better retrieval granularity.
    Each segment has its own embedding vector.
    """
    __tablename__ = "document_segments"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    segment_id = Column(String(100), nullable=False, unique=True, index=True, comment="Unique segment identifier")
    text = Column(Text, nullable=False, comment="Segment text content")
    embedding = Column(Vector(1024), nullable=True, comment="Vector embedding (1024 dimensions for BGE-M3)")
    position = Column(Integer, nullable=False, default=0, comment="Segment position in document")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="segments")

    # Indexes for efficient vector search
    __table_args__ = (
        Index('idx_segment_embedding', 'embedding', postgresql_using='ivfflat', 
              postgresql_with={'lists': 100}, postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )

    def __repr__(self):
        return f"<DocumentSegment(id={self.id}, segment_id='{self.segment_id}', doc_id={self.document_id})>"


# Query tracking for monitoring and analytics (optional)
class QueryLog(Base):
    """
    BUSINESS_RULE: Tracks all queries for analytics and improvement.
    Logs query, score, sources, and user feedback.
    """
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False, comment="User query")
    score = Column(Integer, nullable=False, comment="Confidence score (0-100)")
    answer = Column(Text, nullable=True, comment="Generated answer")
    sources = Column(Text, nullable=True, comment="Retrieved sources (JSON)")
    response_time_ms = Column(Integer, nullable=True, comment="Response time in milliseconds")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self):
        return f"<QueryLog(id={self.id}, score={self.score}, query='{self.query_text[:50]}...')>"

