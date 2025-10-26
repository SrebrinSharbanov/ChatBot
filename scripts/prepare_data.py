"""
Complete data preparation pipeline.
Runs SQL parsing, normalization, segmentation, and database import.
"""

import sys
from pathlib import Path
from loguru import logger  # pyright: ignore[reportMissingImports]

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import individual scripts
import sql_to_json
import preprocess


def main():
    """
    BUSINESS_RULE: Complete data preparation pipeline.
    1. Parse SQL seed file
    2. Normalize documents
    3. Create text segments
    4. Ready for embedding generation
    """
    logger.info("=" * 80)
    logger.info("Mini RAG Chatbot - Data Preparation Pipeline")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Step 1: SQL to JSON
        logger.info("Step 1/2: Parsing SQL seed file...")
        sql_to_json.main()
        logger.info("")
        
        # Step 2: Preprocess and segment
        logger.info("Step 2/2: Creating text segments...")
        preprocess.main()
        logger.info("")
        
        logger.success("=" * 80)
        logger.success("✓ Data preparation completed successfully!")
        logger.success("=" * 80)
        logger.success("")
        logger.success("Next steps:")
        logger.success("  1. Run: python scripts/build_index.py")
        logger.success("  2. Start API: python src/main.py")
        logger.success("")
        
    except Exception as e:
        logger.error(f"✗ Data preparation failed: {e}")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()

