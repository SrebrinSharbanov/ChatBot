#!/usr/bin/env python3
"""
GPU Test Script for Mini RAG Chatbot
Tests GPU availability and performance for Ollama and embeddings.
"""

import subprocess
import time
import requests
import json
from loguru import logger

def test_gpu_availability():
    """Test if GPU is available in Docker container"""
    try:
        # Test NVIDIA runtime
        result = subprocess.run(['docker', 'run', '--rm', '--gpus', 'all', 'nvidia/cuda:11.0-base', 'nvidia-smi'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("✓ GPU available in Docker")
            return True
        else:
            logger.warning("✗ GPU not available in Docker")
            return False
    except Exception as e:
        logger.warning(f"✗ GPU test failed: {e}")
        return False

def test_ollama_gpu():
    """Test Ollama GPU performance"""
    try:
        # Test Ollama API
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info(f"✓ Ollama running with {len(models)} models")
            
            # Test model performance
            if models:
                model_name = models[0]['name']
                logger.info(f"Testing model: {model_name}")
                
                # Simple generation test
                test_prompt = {
                    "model": model_name,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {
                        "num_predict": 10,
                        "temperature": 0.1
                    }
                }
                
                start_time = time.time()
                response = requests.post("http://localhost:11434/api/generate", 
                                       json=test_prompt, timeout=30)
                end_time = time.time()
                
                if response.status_code == 200:
                    generation_time = end_time - start_time
                    logger.info(f"✓ Model generation: {generation_time:.2f}s")
                    return True
                else:
                    logger.error(f"✗ Model generation failed: {response.status_code}")
                    return False
            else:
                logger.warning("✗ No models found in Ollama")
                return False
        else:
            logger.error(f"✗ Ollama API failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"✗ Ollama test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting GPU performance tests...")
    
    # Test 1: GPU availability
    gpu_available = test_gpu_availability()
    
    # Test 2: Ollama performance
    ollama_working = test_ollama_gpu()
    
    # Summary
    logger.info("=" * 50)
    logger.info("📊 TEST RESULTS:")
    logger.info(f"GPU Available: {'✓' if gpu_available else '✗'}")
    logger.info(f"Ollama Working: {'✓' if ollama_working else '✗'}")
    
    if gpu_available and ollama_working:
        logger.success("🎉 All systems ready for GPU acceleration!")
    elif ollama_working:
        logger.info("💻 CPU-only mode working (GPU not available)")
    else:
        logger.error("❌ System not ready - check Ollama installation")

if __name__ == "__main__":
    main()
