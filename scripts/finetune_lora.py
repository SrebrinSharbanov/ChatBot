"""
LoRA Fine-tuning script for RAG chatbot.
Fine-tunes base model on Q&A pairs extracted from corpus.

Note: This is optional and demonstrates the fine-tuning process.
The chatbot works well without fine-tuning using RAG alone.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def load_qa_data(qa_file: Path) -> Dataset:
    """
    Load Q&A training data from JSONL file.
    
    Args:
        qa_file: Path to Q&A JSONL file
    
    Returns:
        HuggingFace Dataset
    """
    data = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    return Dataset.from_list(data)


def format_instruction(example: Dict) -> str:
    """
    Format example into instruction prompt.
    
    Uses Alpaca-style format:
    ### Instruction: {instruction}
    ### Input: {input}
    ### Response: {output}
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"""### Instruction: {instruction}
### Input: {input_text}
### Response: {output}"""
    else:
        prompt = f"""### Instruction: {instruction}
### Response: {output}"""
    
    return prompt


def tokenize_function(examples: Dict, tokenizer, max_length: int = 512):
    """
    Tokenize examples for training.
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
    
    Returns:
        Tokenized inputs
    """
    # Format prompts
    prompts = [format_instruction(ex) for ex in examples]
    
    # Tokenize
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Labels are the same as input_ids for causal LM
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized


def main():
    """
    Main fine-tuning function.
    """
    logger.info("=" * 80)
    logger.info("LoRA Fine-tuning for RAG Chatbot")
    logger.info("=" * 80)
    logger.info("")
    
    # Check if fine-tuning is enabled
    if not settings.finetuning.enabled:
        logger.warning("Fine-tuning is disabled in config. Set finetuning.enabled=true to proceed.")
        logger.info("Note: Fine-tuning is optional. RAG works well without it.")
        return
    
    # Load Q&A data
    qa_file = Path("data/qa_training.jsonl")
    if not qa_file.exists():
        logger.error(f"Q&A training file not found: {qa_file}")
        logger.info("Run scripts/extract_qa_pairs.py first")
        return
    
    logger.info(f"Loading Q&A data from: {qa_file}")
    dataset = load_qa_data(qa_file)
    logger.info(f"Loaded {len(dataset)} training examples")
    
    # Split dataset
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Load base model and tokenizer
    base_model = settings.finetuning.base_model
    logger.info(f"\nLoading base model: {base_model}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with 4-bit quantization (optional, saves memory)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            # load_in_4bit=True,  # Uncomment for 4-bit quantization
        )
        
        logger.success("✓ Model and tokenizer loaded")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("\nTip: Make sure you have enough GPU memory or enable 4-bit quantization")
        return
    
    # Prepare model for LoRA training
    logger.info("\nPreparing model for LoRA training...")
    # model = prepare_model_for_kbit_training(model)  # Uncomment if using quantization
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=settings.finetuning.lora_r,
        lora_alpha=settings.finetuning.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Adjust for your model
        lora_dropout=settings.finetuning.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize datasets
    logger.info("\nTokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # Training arguments
    output_dir = Path(settings.finetuning.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=settings.finetuning.num_epochs,
        per_device_train_batch_size=settings.finetuning.batch_size,
        per_device_eval_batch_size=settings.finetuning.batch_size,
        gradient_accumulation_steps=settings.finetuning.gradient_accumulation_steps,
        learning_rate=settings.finetuning.learning_rate,
        warmup_steps=settings.finetuning.warmup_steps,
        logging_steps=settings.finetuning.logging_steps,
        save_steps=settings.finetuning.save_steps,
        evaluation_strategy="steps",
        eval_steps=settings.finetuning.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb/tensorboard
        fp16=torch.cuda.is_available(),
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting LoRA training...")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        trainer.train()
        logger.success("\n✓ Training completed!")
        
        # Save final model
        final_output = output_dir / "final"
        model.save_pretrained(final_output)
        tokenizer.save_pretrained(final_output)
        logger.success(f"✓ Model saved to: {final_output}")
        
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run scripts/posttrain.py to convert and deploy to Ollama")
        logger.info("  2. Update config to use fine-tuned model")
        logger.info("")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception(e)
        return


if __name__ == "__main__":
    main()

