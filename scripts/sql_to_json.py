"""
SQL to JSON converter.
Parses SQL seed file and extracts documents into JSONL format.
"""

import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def parse_sql_inserts(sql_content: str) -> List[Dict[str, Any]]:
    """
    BUSINESS_RULE: Parse INSERT statements from SQL file.
    Extracts table name, columns, and values from INSERT INTO statements.
    
    Args:
        sql_content: SQL file content as string
    
    Returns:
        List of document dictionaries
    """
    documents = []
    
    # Pattern to match INSERT INTO statements
    # Example: INSERT INTO policies (id, title, body, category) VALUES (1, 'title', 'body', 'cat');
    insert_pattern = re.compile(
        r"INSERT\s+INTO\s+(\w+)\s*\((.*?)\)\s+VALUES\s*(.+?);",
        re.IGNORECASE | re.DOTALL
    )
    
    matches = insert_pattern.findall(sql_content)
    
    for table_name, columns_str, values_str in matches:
        # Parse column names
        columns = [col.strip().strip('`"\'') for col in columns_str.split(',')]
        
        # Parse values - handle multiple VALUE rows
        # Match individual value tuples: (value1, 'value2', value3)
        value_tuples = re.findall(r'\((.*?)\)(?:,|\s*$)', values_str, re.DOTALL)
        
        for value_tuple in value_tuples:
            values = []
            current_value = ""
            in_quotes = False
            quote_char = None
            
            for char in value_tuple:
                if char in ("'", '"') and (not in_quotes or char == quote_char):
                    if in_quotes:
                        in_quotes = False
                        quote_char = None
                    else:
                        in_quotes = True
                        quote_char = char
                elif char == ',' and not in_quotes:
                    values.append(current_value.strip().strip("'\""))
                    current_value = ""
                    continue
                current_value += char
            
            # Add last value
            if current_value.strip():
                values.append(current_value.strip().strip("'\""))
            
            # Create document object
            if len(columns) == len(values):
                doc = {"_table": table_name}
                for col, val in zip(columns, values):
                    doc[col] = val
                documents.append(doc)
            else:
                logger.warning(f"Column/value mismatch in {table_name}: {len(columns)} cols, {len(values)} vals")
    
    return documents


def normalize_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    BUSINESS_RULE: Normalize documents into standard format.
    Combines title/name and body/description into single text field.
    
    Args:
        documents: Raw documents from SQL
    
    Returns:
        Normalized documents with standard fields
    """
    normalized = []
    
    for doc in documents:
        # Extract title (title, name, or question)
        title = doc.get("title") or doc.get("name") or doc.get("question") or ""
        
        # Extract body (body, description, or answer)
        body = doc.get("body") or doc.get("description") or doc.get("answer") or ""
        
        # Combine title and body
        text = f"{title}\n\n{body}".strip()
        
        # Skip empty documents
        if not text:
            continue
        
        normalized.append({
            "source_id": str(doc.get("id", "")),
            "source_table": doc.get("_table", "unknown"),
            "title": title,
            "content": body,
            "text": text,
            "metadata": {
                "category": doc.get("category", ""),
                "price": doc.get("price", None)
            }
        })
    
    return normalized


def main():
    """
    Main function to convert SQL seed to JSONL format.
    """
    # Input/output paths
    sql_file = Path(settings.data.seed_file)
    output_file = Path(settings.data.corpus_output)
    
    if not sql_file.exists():
        logger.error(f"SQL seed file not found: {sql_file}")
        logger.info("Using sample_seed.sql for testing...")
        sql_file = Path("data/sample_seed.sql")
        
        if not sql_file.exists():
            logger.error("No seed file available. Please provide ecom_rag_seed_v2.sql")
            return
    
    # Read SQL file
    logger.info(f"Reading SQL file: {sql_file}")
    sql_content = sql_file.read_text(encoding="utf-8")
    
    # Parse documents
    logger.info("Parsing SQL INSERT statements...")
    documents = parse_sql_inserts(sql_content)
    logger.info(f"Found {len(documents)} raw documents")
    
    # Normalize documents
    logger.info("Normalizing documents...")
    normalized = normalize_documents(documents)
    logger.info(f"Normalized {len(normalized)} documents")
    
    # Write to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in normalized:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    logger.success(f"✓ Written {len(normalized)} documents to {output_file}")
    
    # Print sample
    if normalized:
        logger.info(f"\nSample document:")
        sample = normalized[0]
        logger.info(f"  Source: {sample['source_table']}:{sample['source_id']}")
        logger.info(f"  Title: {sample['title'][:100]}...")
        logger.info(f"  Text length: {len(sample['text'])} chars")


if __name__ == "__main__":
    main()

