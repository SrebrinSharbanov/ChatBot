"""
LoRA Fine-tuning for Bulgarian Language
Demonstrates LoRA adaptation for better Bulgarian understanding.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from loguru import logger  # pyright: ignore[reportMissingImports]

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


class BulgarianLoRATrainer:
    """
    LoRA fine-tuning for Bulgarian language understanding.
    
    BUSINESS_RULE: Demonstrate LoRA adaptation on a small dataset of Bulgarian
    Q&A pairs to improve model performance on domain-specific Bulgarian queries.
    """
    
    def __init__(self):
        """Initialize LoRA trainer."""
        self.model_name = "microsoft/DialoGPT-small"  # Smaller model for demo
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def prepare_data(self) -> List[Dict[str, str]]:
        """
        Prepare Bulgarian training data.
        
        Returns:
            List of instruction-output pairs in Bulgarian
        """
        logger.info("Preparing Bulgarian training data...")
        
        # Bulgarian Q&A pairs for e-commerce domain
        training_data = [
            {
                "instruction": "Каква е политиката за връщане на стоки?",
                "output": "Можете да върнете продукта в рамките на 30 дни от датата на покупка. Продуктът трябва да бъде в оригиналното му състояние и опаковка."
            },
            {
                "instruction": "Как да използвам промо код?",
                "output": "Въведете промо кода в полето 'Промо код' при финализиране на поръчката. Кодът ще се приложи автоматично към общата сума."
            },
            {
                "instruction": "Какви са опциите за доставка?",
                "output": "Предлагаме стандартна доставка (3-5 работни дни), експресна доставка (1-2 работни дни) и доставка в същия ден за София."
            },
            {
                "instruction": "Как да проследя поръчката си?",
                "output": "Можете да проследите поръчката си в секцията 'Моите поръчки' или използвайки номера за проследяване, който получихте по имейл."
            },
            {
                "instruction": "Каква е гаранцията за продуктите?",
                "output": "Всички продукти имат гаранция от производителя. Електрониката има 24-месечна гаранция, а останалите продукти - 12 месеца."
            },
            {
                "instruction": "Как да се свържа с клиентския сервиз?",
                "output": "Можете да се свържете с нас на телефон 0700 123 456, по имейл support@example.com или чрез чат системата на сайта."
            },
            {
                "instruction": "Какви са начините за плащане?",
                "output": "Приемаме плащане с карти (Visa, Mastercard), банков превод, наложен платеж и дигитални портфейли (PayPal, Apple Pay)."
            },
            {
                "instruction": "Как да отменя поръчка?",
                "output": "Можете да отмените поръчка в рамките на 2 часа от направената поръчка чрез 'Моите поръчки' или се свържете с клиентския сервиз."
            },
            {
                "instruction": "Има ли безплатна доставка?",
                "output": "Да, предлагаме безплатна доставка за поръчки над 50 лева в цялата страна и над 30 лева за София."
            },
            {
                "instruction": "Как да сменя адреса за доставка?",
                "output": "Можете да смените адреса за доставка в 'Моите поръчки' ако поръчката все още не е изпратена, или се свържете с клиентския сервиз."
            }
        ]
        
        logger.info(f"Prepared {len(training_data)} Bulgarian training examples")
        return training_data
    
    def format_instruction(self, example: Dict[str, str]) -> str:
        """Format instruction-output pair for training."""
        return f"### Въпрос:\n{example['instruction']}\n\n### Отговор:\n{example['output']}"
    
    def tokenize_function(self, examples):
        """Tokenize training examples."""
        texts = [self.format_instruction(ex) for ex in examples]
        
        # Tokenize with padding and truncation
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def setup_model(self):
        """Setup model and tokenizer for LoRA training."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Setup LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # Rank
            lora_alpha=16,  # Scaling parameter
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention modules
            bias="none"
        )
        
        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("Model setup completed")
    
    def train(self, output_dir: str = "models/lora-bulgarian"):
        """
        Train LoRA model on Bulgarian data.
        
        Args:
            output_dir: Directory to save the trained model
        """
        logger.info("Starting LoRA training...")
        
        # Prepare data
        training_data = self.prepare_data()
        
        # Setup model
        self.setup_model()
        
        # Create dataset
        dataset = Dataset.from_list(training_data)
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,  # Small number for demo
            per_device_train_batch_size=1,  # Small batch size
            gradient_accumulation_steps=4,
            warmup_steps=10,
            learning_rate=2e-4,
            logging_steps=1,
            save_steps=50,
            evaluation_strategy="no",
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,  # Disable wandb
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed. Model saved to {output_dir}")
        
        # Test the model
        self.test_model()
    
    def test_model(self):
        """Test the trained model with a sample query."""
        logger.info("Testing trained model...")
        
        test_query = "Каква е политиката за връщане?"
        
        # Format input
        input_text = f"### Въпрос:\n{test_query}\n\n### Отговор:\n"
        
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Test query: {test_query}")
        logger.info(f"Model response: {response}")
        
        return response


def main():
    """Main training function."""
    logger.info("=" * 80)
    logger.info("Bulgarian LoRA Training")
    logger.info("=" * 80)
    
    try:
        trainer = BulgarianLoRATrainer()
        trainer.train()
        
        logger.info("=" * 80)
        logger.info("✓ LoRA training completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()
