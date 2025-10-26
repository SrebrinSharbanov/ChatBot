"""
Post-training deployment script.
Fuses LoRA adapter with base model, converts to GGUF, and registers in Ollama.

Note: This is optional and requires:
- mlx_lm for model fusion (or similar tools)
- llama.cpp for GGUF conversion
- Ollama CLI for model registration
"""

import os
import subprocess
import shutil
import sys
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def run_command(cmd: list, cwd: Path = None) -> bool:
    """
    Run shell command and return success status.
    
    Args:
        cmd: Command as list of strings
        cwd: Working directory
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(e.stderr)
        return False


def check_ollama_available() -> bool:
    """Check if Ollama CLI is available."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def fuse_lora_adapter(
    base_model: str,
    adapter_path: Path,
    output_path: Path
) -> bool:
    """
    BUSINESS_RULE: Fuse LoRA adapter with base model.
    
    Note: This is a placeholder. Actual implementation depends on your tools:
    - Use mlx_lm.fuse for MLX models
    - Use PEFT merge_and_unload for transformers
    - Use custom fusion scripts
    
    Args:
        base_model: Base model identifier
        adapter_path: Path to LoRA adapter
        output_path: Output path for fused model
    
    Returns:
        True if successful
    """
    logger.info("Fusing LoRA adapter with base model...")
    logger.info(f"  Base: {base_model}")
    logger.info(f"  Adapter: {adapter_path}")
    logger.info(f"  Output: {output_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Option 1: Using transformers + PEFT
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        logger.info("Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base, str(adapter_path))
        
        logger.info("Merging adapter...")
        model = model.merge_and_unload()
        
        logger.info(f"Saving fused model to {output_path}...")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        logger.success("✓ Model fusion completed")
        return True
        
    except Exception as e:
        logger.error(f"Model fusion failed: {e}")
        logger.info("Tip: Make sure transformers and peft are installed")
        return False


def convert_to_gguf(
    model_path: Path,
    output_path: Path,
    quantization: str = "Q4_K_M"
) -> bool:
    """
    BUSINESS_RULE: Convert model to GGUF format for Ollama.
    
    Requires llama.cpp conversion scripts.
    
    Args:
        model_path: Path to HuggingFace model
        output_path: Output GGUF file path
        quantization: Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)
    
    Returns:
        True if successful
    """
    logger.info("Converting model to GGUF format...")
    logger.info(f"  Input: {model_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Quantization: {quantization}")
    
    # Check if llama.cpp tools are available
    # This is a placeholder - actual paths depend on your setup
    llama_cpp_dir = Path.home() / "llama.cpp"
    
    if not llama_cpp_dir.exists():
        logger.warning("llama.cpp not found. Skipping GGUF conversion.")
        logger.info("Install from: https://github.com/ggerganov/llama.cpp")
        return False
    
    # Convert HF to GGUF
    convert_script = llama_cpp_dir / "convert.py"
    if convert_script.exists():
        success = run_command([
            "python",
            str(convert_script),
            str(model_path),
            "--outfile", str(output_path),
            "--outtype", quantization
        ])
        
        if success:
            logger.success("✓ GGUF conversion completed")
            return True
    
    logger.error("GGUF conversion failed")
    return False


def create_modelfile(
    model_path: Path,
    modelfile_path: Path,
    model_name: str
) -> bool:
    """
    BUSINESS_RULE: Create Ollama Modelfile.
    
    Args:
        model_path: Path to GGUF model
        modelfile_path: Output Modelfile path
        model_name: Model name for Ollama
    
    Returns:
        True if successful
    """
    logger.info(f"Creating Modelfile for: {model_name}")
    
    # Create Modelfile content
    modelfile_content = f"""FROM {model_path}

# Model parameters
PARAMETER temperature 0.0
PARAMETER top_p 0.9
PARAMETER top_k 40

# System prompt for RAG
SYSTEM Ти си асистент който отговаря на въпроси на български език. Използвай само предоставената информация от контекста. Не измисляй информация.

# Template (optional)
TEMPLATE \"\"\"{{{{ .System }}}}
Контекст: {{{{ .Context }}}}
Въпрос: {{{{ .Prompt }}}}
Отговор:\"\"\"
"""
    
    modelfile_path.write_text(modelfile_content, encoding='utf-8')
    logger.success(f"✓ Modelfile created: {modelfile_path}")
    return True


def register_in_ollama(
    modelfile_path: Path,
    model_name: str
) -> bool:
    """
    BUSINESS_RULE: Register model in Ollama.
    
    Args:
        modelfile_path: Path to Modelfile
        model_name: Model name
    
    Returns:
        True if successful
    """
    logger.info(f"Registering model in Ollama: {model_name}")
    
    if not check_ollama_available():
        logger.error("Ollama CLI not found. Install from: https://ollama.com")
        return False
    
    # Create model
    success = run_command([
        "ollama",
        "create",
        model_name,
        "-f",
        str(modelfile_path)
    ])
    
    if success:
        logger.success(f"✓ Model registered: {model_name}")
        logger.info(f"  Test with: ollama run {model_name}")
        return True
    
    return False


def main():
    """
    Main post-training deployment pipeline.
    """
    logger.info("=" * 80)
    logger.info("Post-Training Deployment Pipeline")
    logger.info("=" * 80)
    logger.info("")
    
    # Check if fine-tuning output exists
    adapter_path = Path(settings.finetuning.output_dir) / "final"
    if not adapter_path.exists():
        logger.error(f"LoRA adapter not found: {adapter_path}")
        logger.info("Run scripts/finetune_lora.py first")
        logger.info("")
        logger.info("Note: Post-training is optional. You can use the base model directly.")
        return
    
    # Setup paths
    temp_dir = Path(settings.posttrain.get("temp_dir", "/tmp/lora_deploy"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    fused_model_path = temp_dir / "fused_model"
    gguf_path = temp_dir / "model.gguf"
    modelfile_path = temp_dir / "Modelfile"
    
    base_model = settings.finetuning.base_model
    ollama_model_name = settings.posttrain.get("ollama_model_name", "qwen25-lora-rag")
    
    try:
        # Step 1: Fuse LoRA adapter
        if settings.posttrain.get("fuse_adapter", True):
            logger.info("\n[Step 1/4] Fusing LoRA adapter...")
            if not fuse_lora_adapter(base_model, adapter_path, fused_model_path):
                logger.error("✗ Fusion failed. Aborting.")
                return
        else:
            logger.info("\n[Step 1/4] Skipping fusion (disabled in config)")
            fused_model_path = adapter_path
        
        # Step 2: Convert to GGUF
        if settings.posttrain.get("convert_to_gguf", True):
            logger.info("\n[Step 2/4] Converting to GGUF...")
            if not convert_to_gguf(fused_model_path, gguf_path):
                logger.warning("✗ GGUF conversion failed. Trying direct registration...")
                gguf_path = fused_model_path  # Fallback
        else:
            logger.info("\n[Step 2/4] Skipping GGUF conversion")
            gguf_path = fused_model_path
        
        # Step 3: Create Modelfile
        logger.info("\n[Step 3/4] Creating Modelfile...")
        if not create_modelfile(gguf_path, modelfile_path, ollama_model_name):
            logger.error("✗ Modelfile creation failed")
            return
        
        # Step 4: Register in Ollama
        logger.info("\n[Step 4/4] Registering in Ollama...")
        if not register_in_ollama(modelfile_path, ollama_model_name):
            logger.error("✗ Ollama registration failed")
            return
        
        # Success
        logger.info("")
        logger.success("=" * 80)
        logger.success("✓ Post-training deployment completed!")
        logger.success("=" * 80)
        logger.success("")
        logger.success(f"Fine-tuned model deployed as: {ollama_model_name}")
        logger.success("")
        logger.success("Next steps:")
        logger.success(f"  1. Test: ollama run {ollama_model_name}")
        logger.success(f"  2. Update config.yaml: llm.model_name = '{ollama_model_name}'")
        logger.success("  3. Restart API: python src/main.py")
        logger.success("")
        
    except Exception as e:
        logger.error(f"✗ Deployment failed: {e}")
        logger.exception(e)
    finally:
        # Cleanup (optional)
        # shutil.rmtree(temp_dir, ignore_errors=True)
        pass


if __name__ == "__main__":
    main()

