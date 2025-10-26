#!/usr/bin/env python3
"""
Ollama Warmup Script
Pre-loads the model to avoid cold start delays.
"""

import requests
import time
import sys
from loguru import logger

def warmup_ollama(ollama_host: str = "http://localhost:11434", model: str = "qwen2.5:1.5b"):
    """
    Warm up Ollama by sending a simple request to pre-load the model.
    
    Args:
        ollama_host: Ollama API host
        model: Model name to warm up
    """
    logger.info(f"🔥 Warming up Ollama model: {model}")
    
    url = f"{ollama_host}/api/generate"
    payload = {
        "model": model,
        "prompt": "Hello",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "max_tokens": 10
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        elapsed = time.time() - start_time
        logger.success(f"✅ Ollama warmup completed in {elapsed:.2f}s")
        
        # Verify model is loaded
        models_url = f"{ollama_host}/api/tags"
        models_response = requests.get(models_url, timeout=10)
        if models_response.status_code == 200:
            models = models_response.json()
            logger.info(f"📋 Available models: {[m['name'] for m in models.get('models', [])]}")
        
        return True
        
    except requests.exceptions.Timeout:
        logger.error("⏰ Ollama warmup timed out")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("🔌 Cannot connect to Ollama")
        return False
    except Exception as e:
        logger.error(f"❌ Ollama warmup failed: {e}")
        return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Warm up Ollama model")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Model name")
    parser.add_argument("--wait", type=int, default=10, help="Wait time before warmup")
    
    args = parser.parse_args()
    
    logger.info(f"⏳ Waiting {args.wait}s for Ollama to start...")
    time.sleep(args.wait)
    
    success = warmup_ollama(args.host, args.model)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
