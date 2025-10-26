"""
Extract Q&A pairs from FAQ and policies for fine-tuning.
Prepares instruction-format dataset for LoRA training.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def load_documents(corpus_file: Path) -> List[Dict[str, Any]]:
    """Load documents from corpus JSONL file."""
    documents = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
    return documents


def extract_qa_from_faq(documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extract Q&A pairs from FAQ documents.
    
    FAQ format:
    - title = question
    - content = answer
    """
    qa_pairs = []
    
    for doc in documents:
        if doc['source_table'] == 'faq':
            question = doc['title'].strip()
            answer = doc['content'].strip()
            
            if question and answer:
                qa_pairs.append({
                    "instruction": question,
                    "output": answer,
                    "source": f"faq:{doc['source_id']}"
                })
    
    return qa_pairs


def extract_qa_from_policies(documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Generate Q&A pairs from policy documents.
    Creates synthetic questions based on policy titles.
    """
    qa_pairs = []
    
    # Common question patterns for policies
    patterns = [
        "Каква е политиката за {topic}?",
        "Разкажи ми за {topic}.",
        "Какво трябва да знам за {topic}?",
        "Обясни политиката относно {topic}.",
    ]
    
    for doc in documents:
        if doc['source_table'] == 'policies':
            title = doc['title'].strip()
            content = doc['content'].strip()
            
            if not title or not content:
                continue
            
            # Extract topic from title (remove "Политика за/относно/за")
            topic = title.lower()
            for prefix in ['политика за ', 'политика относно ', 'политика: ']:
                if topic.startswith(prefix):
                    topic = topic[len(prefix):]
                    break
            
            # Generate questions using patterns
            for pattern in patterns[:2]:  # Use first 2 patterns per policy
                question = pattern.format(topic=topic)
                qa_pairs.append({
                    "instruction": question,
                    "output": content[:500],  # Limit answer length
                    "source": f"policy:{doc['source_id']}"
                })
    
    return qa_pairs


def format_for_training(qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Format Q&A pairs for instruction fine-tuning.
    
    Format: Alpaca-style instruction format
    {
        "instruction": "question",
        "input": "",
        "output": "answer"
    }
    """
    formatted = []
    
    for qa in qa_pairs:
        formatted.append({
            "instruction": qa["instruction"],
            "input": "",
            "output": qa["output"]
        })
    
    return formatted


def main():
    """Main function to extract Q&A pairs."""
    logger.info("=" * 80)
    logger.info("Extracting Q&A pairs for fine-tuning")
    logger.info("=" * 80)
    logger.info("")
    
    # Load corpus
    corpus_file = Path(settings.data.corpus_output)
    if not corpus_file.exists():
        logger.error(f"Corpus file not found: {corpus_file}")
        logger.info("Run scripts/prepare_data.py first")
        return
    
    logger.info(f"Loading corpus from: {corpus_file}")
    documents = load_documents(corpus_file)
    logger.info(f"Loaded {len(documents)} documents")
    
    # Extract Q&A from FAQ
    logger.info("\nExtracting Q&A from FAQ...")
    faq_qa = extract_qa_from_faq(documents)
    logger.info(f"Extracted {len(faq_qa)} Q&A pairs from FAQ")
    
    # Extract Q&A from policies
    logger.info("\nGenerating Q&A from policies...")
    policy_qa = extract_qa_from_policies(documents)
    logger.info(f"Generated {len(policy_qa)} Q&A pairs from policies")
    
    # Combine all Q&A pairs
    all_qa = faq_qa + policy_qa
    logger.info(f"\nTotal Q&A pairs: {len(all_qa)}")
    
    # Format for training
    training_data = format_for_training(all_qa)
    
    # Save to file
    output_file = Path("data/qa_training.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.success(f"\n✓ Saved {len(training_data)} training examples to: {output_file}")
    
    # Print sample
    if training_data:
        logger.info("\nSample Q&A pair:")
        sample = training_data[0]
        logger.info(f"  Q: {sample['instruction']}")
        logger.info(f"  A: {sample['output'][:100]}...")
    
    logger.info("")
    logger.info("Next step: Run scripts/finetune_lora.py to fine-tune model")


if __name__ == "__main__":
    main()

