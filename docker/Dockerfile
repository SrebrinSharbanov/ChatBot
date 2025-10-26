# Multi-stage build за минимален final image
FROM python:3.11-slim as builder

# Задай working directory
WORKDIR /app

# Инсталирай system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирай requirements
COPY requirements.txt .

# Инсталирай Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Задай working directory
WORKDIR /app

# Копирай installed packages от builder
COPY --from=builder /root/.local /root/.local

# Инсталирай само runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Добави local bin към PATH
ENV PATH=/root/.local/bin:$PATH

# Копирай application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Създай директории за cache (ще бъдат mounted от D:\)
RUN mkdir -p /app/models /app/embeddings /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (може да се override в docker-compose.yml)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

