"""
Build vector index in PostgreSQL with pgvector.
Generates embeddings and populates database with documents and segments.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger  # pyright: ignore[reportMissingImports]
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.database.db import init_db, get_db
from src.database.models import Document, DocumentSegment
from src.indexing.embeddings import get_embedding_model


def load_segments(segments_file: Path) -> List[Dict[str, Any]]:
    """
    Load segments from JSONL file.
    
    Args:
        segments_file: Path to segments JSONL file
    
    Returns:
        List of segment dictionaries
    """
    segments = []
    with open(segments_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                segments.append(json.loads(line))
    return segments


def populate_database(segments: List[Dict[str, Any]], embedding_model):
    """
    BUSINESS_RULE: Populate database with documents and segments including embeddings.
    
    Args:
        segments: List of segment dictionaries
        embedding_model: Embedding model instance
    """
    with get_db() as db:
        # Group segments by document
        documents_map = {}
        for segment in segments:
            doc_key = f"{segment['source_table']}:{segment['source_id']}"
            if doc_key not in documents_map:
                documents_map[doc_key] = {
                    'source_id': segment['source_id'],
                    'source_table': segment['source_table'],
                    'title': segment['title'],
                    'segments': []
                }
            documents_map[doc_key]['segments'].append(segment)
        
        logger.info(f"Processing {len(documents_map)} documents with {len(segments)} segments")
        
        # Clear existing data (optional - comment out to preserve data)
        logger.info("Clearing existing data...")
        db.query(DocumentSegment).delete()
        db.query(Document).delete()
        db.commit()
        
        # Process each document
        for doc_key, doc_data in tqdm(documents_map.items(), desc="Indexing documents"):
            # Combine all segments into full content
            full_content = "\n\n".join([s['text'] for s in doc_data['segments']])
            
            # Create document
            document = Document(
                source_id=doc_data['source_id'],
                source_table=doc_data['source_table'],
                title=doc_data['title'],
                content=full_content,
                metadata=json.dumps(doc_data['segments'][0].get('metadata', {}))
            )
            db.add(document)
            db.flush()  # Get document ID
            
            # Generate embeddings for all segments at once (batch processing)
            segment_texts = [s['text'] for s in doc_data['segments']]
            embeddings = embedding_model.encode_documents(
                segment_texts,
                batch_size=settings.embedding.batch_size,
                show_progress=False
            )
            
            # Create document segments with embeddings
            for i, segment_data in enumerate(doc_data['segments']):
                segment = DocumentSegment(
                    document_id=document.id,
                    segment_id=segment_data['segment_id'],
                    text=segment_data['text'],
                    embedding=embeddings[i].tolist(),  # Convert numpy array to list
                    position=segment_data['position']
                )
                db.add(segment)
        
        db.commit()
        logger.success(f"✓ Indexed {len(documents_map)} documents with {len(segments)} segments")


def verify_index(embedding_model):
    """
    VALIDATION: Verify index by performing test query.
    
    Args:
        embedding_model: Embedding model instance
    """
    with get_db() as db:
        # Count documents and segments
        doc_count = db.query(Document).count()
        segment_count = db.query(DocumentSegment).count()
        
        logger.info(f"\nIndex statistics:")
        logger.info(f"  Documents: {doc_count}")
        logger.info(f"  Segments: {segment_count}")
        
        # Test query
        if segment_count > 0:
            test_query = "политика за връщане"
            logger.info(f"\nTest query: '{test_query}'")
            
            query_embedding = embedding_model.encode_query(test_query)
            
            # Find similar segments using pgvector
            from sqlalchemy import text
            # Convert embedding to string format for pgvector
            embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
            
            result = db.execute(
                text(f"""
                    SELECT 
                        ds.segment_id,
                        d.title,
                        ds.text,
                        1 - (ds.embedding <=> '{embedding_str}'::vector) as similarity
                    FROM document_segments ds
                    JOIN documents d ON ds.document_id = d.id
                    ORDER BY ds.embedding <=> '{embedding_str}'::vector
                    LIMIT 3
                """)
            ).fetchall()
            
            logger.info(f"\nTop 3 results:")
            for i, (segment_id, title, text, similarity) in enumerate(result, 1):
                logger.info(f"  {i}. {title} (similarity: {similarity:.4f})")
                logger.info(f"     Segment: {segment_id}")
                logger.info(f"     Text: {text[:100]}...")


def main():
    """
    Main function to build vector index.
    """
    logger.info("=" * 80)
    logger.info("Mini RAG Chatbot - Building Vector Index")
    logger.info("=" * 80)
    logger.info("")
    
    # Check segments file
    segments_file = Path(settings.data.segments_output)
    if not segments_file.exists():
        logger.error(f"Segments file not found: {segments_file}")
        logger.info("Run scripts/prepare_data.py first to generate segments")
        return
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        logger.info("")
        
        # Load embedding model
        logger.info("Loading embedding model...")
        embedding_model = get_embedding_model()
        logger.info("")
        
        # Load segments
        logger.info(f"Loading segments from: {segments_file}")
        segments = load_segments(segments_file)
        logger.info(f"Loaded {len(segments)} segments")
        logger.info("")
        
        # Populate database
        logger.info("Generating embeddings and populating database...")
        populate_database(segments, embedding_model)
        logger.info("")
        
        # Verify index
        logger.info("Verifying index...")
        verify_index(embedding_model)
        logger.info("")
        
        logger.success("=" * 80)
        logger.success("✓ Vector index built successfully!")
        logger.success("=" * 80)
        logger.success("")
        logger.success("Next steps:")
        logger.success("  1. Start API: python src/main.py")
        logger.success("  2. Test chatbot: python scripts/test_client.py")
        logger.success("")
        
    except Exception as e:
        logger.error(f"✗ Failed to build index: {e}")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()

