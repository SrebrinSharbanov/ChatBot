"""
Text preprocessing and segmentation.
Splits documents into segments for better retrieval granularity.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger  # pyright: ignore[reportMissingImports]

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def segment_text(
    text: str,
    max_length: int = 1000,
    overlap: int = 100
) -> List[str]:
    """
    BUSINESS_RULE: Segment text into chunks with overlap.
    Uses paragraph boundaries when possible, falls back to character-based splitting.
    
    Args:
        text: Input text to segment
        max_length: Maximum segment length in characters
        overlap: Overlap between segments in characters
    
    Returns:
        List of text segments
    """
    # Split by paragraphs first (double newline)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    
    segments = []
    current_segment = ""
    
    for paragraph in paragraphs:
        # If paragraph fits in current segment
        if len(current_segment) + len(paragraph) + 2 <= max_length:
            if current_segment:
                current_segment += "\n\n"
            current_segment += paragraph
        else:
            # Save current segment if not empty
            if current_segment:
                segments.append(current_segment)
            
            # If paragraph itself is too long, split by sentences
            if len(paragraph) > max_length:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_segment = ""
                
                for sentence in sentences:
                    if len(temp_segment) + len(sentence) + 1 <= max_length:
                        if temp_segment:
                            temp_segment += " "
                        temp_segment += sentence
                    else:
                        if temp_segment:
                            segments.append(temp_segment)
                        
                        # If single sentence is too long, force split
                        if len(sentence) > max_length:
                            for i in range(0, len(sentence), max_length - overlap):
                                segments.append(sentence[i:i + max_length])
                        else:
                            temp_segment = sentence
                
                current_segment = temp_segment
            else:
                current_segment = paragraph
    
    # Add last segment
    if current_segment:
        segments.append(current_segment)
    
    return segments


def create_segments(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    BUSINESS_RULE: Create document segments for vector indexing.
    Each segment gets unique ID and maintains link to source document.
    
    Args:
        documents: List of normalized documents
    
    Returns:
        List of document segments
    """
    all_segments = []
    
    for doc in documents:
        text = doc["text"]
        source_id = doc["source_id"]
        source_table = doc["source_table"]
        title = doc["title"]
        
        # Segment the text
        segments = segment_text(
            text,
            max_length=settings.data.max_segment_length,
            overlap=settings.data.overlap
        )
        
        # Create segment objects
        for i, segment_content in enumerate(segments):
            all_segments.append({
                "segment_id": f"{source_table}:{source_id}:seg{i}",
                "source_id": source_id,
                "source_table": source_table,
                "title": title,
                "text": segment_content,
                "position": i,
                "metadata": doc.get("metadata", {})
            })
    
    return all_segments


def main():
    """
    Main function to preprocess documents and create segments.
    """
    # Input/output paths
    input_file = Path(settings.data.corpus_output)
    output_file = Path(settings.data.segments_output)
    
    if not input_file.exists():
        logger.error(f"Corpus file not found: {input_file}")
        logger.info("Run sql_to_json.py first to generate corpus.jsonl")
        return
    
    # Read documents
    logger.info(f"Reading documents from: {input_file}")
    documents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Create segments
    logger.info("Creating text segments...")
    segments = create_segments(documents)
    logger.info(f"Created {len(segments)} segments")
    
    # Write segments to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in segments:
            f.write(json.dumps(segment, ensure_ascii=False) + '\n')
    
    logger.success(f"✓ Written {len(segments)} segments to {output_file}")
    
    # Statistics
    avg_length = sum(len(s["text"]) for s in segments) / len(segments) if segments else 0
    logger.info(f"\nStatistics:")
    logger.info(f"  Total segments: {len(segments)}")
    logger.info(f"  Average segment length: {avg_length:.0f} characters")
    logger.info(f"  Segments per document: {len(segments) / len(documents):.1f}")
    
    # Print sample
    if segments:
        logger.info(f"\nSample segment:")
        sample = segments[0]
        logger.info(f"  ID: {sample['segment_id']}")
        logger.info(f"  Source: {sample['source_table']}:{sample['source_id']}")
        logger.info(f"  Text: {sample['text'][:200]}...")


if __name__ == "__main__":
    main()

