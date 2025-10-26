# 🚀 GPU Setup за Ollama

## **📋 Изисквания**

### **1. NVIDIA GPU Driver**
```bash
# Провери дали имаш NVIDIA GPU
nvidia-smi
```

### **2. Docker с GPU поддръжка**
```bash
# Инсталирай NVIDIA Container Toolkit
# Windows: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker-desktop

# Или използвай Docker Desktop с WSL2 + NVIDIA drivers
```

### **3. Проверка на GPU в Docker**
```bash
# Тест за GPU поддръжка
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## **🔧 Конфигурация**

### **1. Ollama с GPU**
```yaml
# docker-compose.yml
ollama:
  image: ollama/ollama:latest
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  environment:
    OLLAMA_FLASH_ATTENTION: true  # GPU acceleration
```

### **2. Проверка на GPU в Ollama**
```bash
# Влез в Ollama контейнера
docker exec -it mini-rag-ollama bash

# Провери GPU
nvidia-smi

# Провери Ollama GPU поддръжка
ollama list
```

## **⚡ Performance Очаквания**

### **🚀 С GPU:**
- **Model loading:** ~5-10s (vs 30-60s CPU)
- **Inference speed:** ~3-5x по-бързо
- **Memory usage:** По-ефективно
- **Concurrent requests:** По-добро

### **📊 Сравнение:**
| Метрика | CPU | GPU |
|---------|-----|-----|
| Model load | 30-60s | 5-10s |
| First token | 2-5s | 0.5-1s |
| Tokens/sec | 5-10 | 20-50 |
| Memory | 2-4GB | 1-2GB |

## **🛠 Troubleshooting**

### **❌ GPU не се вижда:**
```bash
# Провери NVIDIA drivers
nvidia-smi

# Провери Docker GPU поддръжка
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### **❌ Windows Docker Desktop:**
```bash
# Windows Docker Desktop не поддържа GPU директно
# За GPU поддръжка на Windows:
# 1. Използвай WSL2 с NVIDIA drivers
# 2. Или инсталирай Ollama директно на Windows
# 3. Или използвай CPU оптимизации (по-долу)
```

### **❌ Ollama не използва GPU:**
```bash
# Провери Ollama logs
docker logs mini-rag-ollama

# Провери GPU в контейнера
docker exec -it mini-rag-ollama nvidia-smi
```

### **❌ CUDA грешки:**
```bash
# Провери CUDA версия
nvidia-smi

# Провери Ollama CUDA поддръжка
docker exec -it mini-rag-ollama ollama --version
```

## **🎯 Оптимизация**

### **1. Model Selection:**
- **Qwen2.5:1.5b** - добър баланс за GPU
- **Phi-3.5-mini** - по-бърз на GPU
- **Llama3.2:3b** - по-добро качество

### **2. GPU Settings:**
```yaml
environment:
  OLLAMA_FLASH_ATTENTION: true
  OLLAMA_NUM_PARALLEL: 1
  OLLAMA_MAX_LOADED_MODELS: 1
  OLLAMA_KEEP_ALIVE: 5m
```

### **3. CPU Settings (Windows Docker Desktop):**
```yaml
environment:
  OLLAMA_FLASH_ATTENTION: false  # Disabled for Windows
  OLLAMA_NUM_THREADS: 8  # Increased for CPU
  OLLAMA_NUM_PARALLEL: 1
  OLLAMA_MAX_LOADED_MODELS: 1
  OLLAMA_KEEP_ALIVE: 5m
```

### **3. Memory Management:**
```bash
# Мониторинг на GPU памет
nvidia-smi -l 1

# Освобождаване на памет
docker restart mini-rag-ollama
```

## **✅ Проверка**

### **1. Стартиране с GPU:**
```bash
cd docker
docker-compose up -d ollama
```

### **2. Проверка на GPU:**
```bash
# Провери GPU използване
nvidia-smi

# Провери Ollama logs
docker logs mini-rag-ollama
```

### **3. Тест на производителност:**
```bash
# Тест заявка
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"q":"Тест на GPU производителност", "k":3}'
```

## **🎉 Резултат**

С GPU трябва да видиш:
- **По-бързо зареждане** на модела
- **По-бързи отговори** (3-5x)
- **По-ниско CPU използване**
- **По-добра производителност** при много заявки

**Perfect! 🚀** Сега Ollama ще използва GPU за значително по-бърза производителност!
