# Mini RAG Chatbot - Кратко Ръководство

## 🚀 Бързо Стартиране

```bash
# 1. Клониране
git clone https://github.com/SrebrinSharbanov/ChatBot.git
cd ChatBot

# 2. Стартиране с Docker
cd docker
docker-compose up -d

# 3. Подготовка на данните
docker exec -it mini-rag-chatbot python scripts/prepare_data.py
docker exec -it mini-rag-chatbot python scripts/build_index.py

# 4. Готово! Отвори http://localhost:8000/api/frontend

# 5. Тестване (опционално)
# Разширени юнит тестове (150+ тест случая)
docker exec -it mini-rag-chatbot python unit_test_chatbot.py

# Тестване на конкретна категория
docker exec -it mini-rag-chatbot python unit_test_chatbot.py --category "продукти"
```

## ✨ Основни Функции

### 🧠 **Интелигентно Разпознаване на Намерения**
- Автоматично разпознава какво иска потребителят
- Поддържа 40+ разговорни форми на български
- **Пример:** "Плаща ли се?" → payment intent

### 🛍️ **Smart Product Filtering с Hybrid Retrieval**
- **Keyword Search** (бързо) → **Vector Search** (fallback)
- Хвата продукти дори с английски категории
- LLM генерира естествени отговори

**Примери:**
```
Въпрос: "Предлагате ли телефони?"
Отговор: "Да, предлагаме Samsung Galaxy S23 (1899 лв), iPhone 13 Mini (999 лв)..."

Въпрос: "Имате ли лаптопи?"  
Отговор: "Имаме Dell XPS 13, Lenovo ThinkPad, ASUS VivoBook..."
```

### 📚 **RAG (Retrieval-Augmented Generation)**
- Отговаря само от предоставените данни
- Показва източници и confidence score
- При нисък score: "Това не е в моята компетенция"

### 🇧🇬 **Български Post-Processing**
- Автоматично изглаждане на граматика
- Естествени отговори на български език

## 🧪 Тестване

### **📊 Разширени Юнит Тестове**
- **150+ тест случая** в 12 категории
- **Автоматично изчисляване** на статистики
- **Гъвкаво тестване** по категории

```bash
# Пълно тестване
python unit_test_chatbot.py

# Конкретна категория
python unit_test_chatbot.py --category "продукти"

# Списък категории
python unit_test_chatbot.py --list-categories
```

## 🎯 Примери на Работа

### **FAQ Въпроси:**
```
Q: "Каква е политиката за връщане?"
A: "Можете да върнете продукта в рамките на 30 дни от датата на покупка..."
Score: 92/100
```

### **Продуктови Заявки:**
```
Q: "Предлагате ли смартфони?"
A: "Да, в момента предлагаме няколко отличен модела смартфони:
   Samsung Galaxy S23 е флагмански модел (1899 лв)
   iPhone 13 Mini е компактен и мощен (999 лв)
   Всички са налични с 24 месеца гаранция."
Score: 95/100
```

### **Низък Confidence:**
```
Q: "Какво е квантовото изчисление?"
A: "Това не е в моята компетенция."
Score: 15/100
```

## 🛠 Технологии

- **LLM:** Qwen2.5:1.5b (via Ollama)
- **Embeddings:** BAAI/bge-m3 (1024-dim, multilingual)
- **Database:** PostgreSQL + pgvector
- **Language Processing:** Bulgarian lemmatization (simplemma)
- **Framework:** FastAPI + Python 3.11
- **Containerization:** Docker & Docker Compose

## 📊 Performance

| Операция | Време | Точност |
|----------|-------|---------|
| Keyword Search | 50-100ms | 95% |
| Hybrid Search | 150-250ms | 85% |
| Vector Fallback | 200-300ms | 75% |

## 🔧 API Endpoints

- `GET /api/health` - Health check
- `POST /api/query` - Стандартно търсене
- `GET /api/frontend` - Web интерфейс
- `GET /docs` - Swagger документация

## 📋 Изисквания

- Docker & Docker Compose
- 4GB+ RAM
- 10GB+ свободно място

## 🎨 Frontend

**Достъпен на:** `http://localhost:8000/api/frontend`

- Интерактивен чат интерфейс
- FAQ секция с случайни въпроси
- Настройки за търсене
- Responsive дизайн

---

## 🔗 GitHub Repository

**Репозиторий**: [https://github.com/SrebrinSharbanov/ChatBot](https://github.com/SrebrinSharbanov/ChatBot)

**🎯 Готово за production!** Просто стартирай Docker и започни да задаваш въпроси! 🚀
