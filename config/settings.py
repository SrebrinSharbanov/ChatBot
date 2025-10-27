"""
Configuration settings loader for Mini RAG Chatbot.
Loads configuration from YAML and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "rag_chatbot"
    user: str = "raguser"
    password: str = "ragpassword"
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_url(self) -> str:
        """Get async database URL"""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""
    model_name: str = "BAAI/bge-m3"  # Multilingual model with Bulgarian support
    dimension: int = 1024  # BGE-M3 has 1024 dimensions
    batch_size: int = 32
    device: str = "cpu"  # CPU mode until GPU driver is updated
    cache_dir: str = "D:/mini-rag-chatbot/models/embeddings"


class LLMConfig(BaseModel):
    """LLM configuration"""
    provider: str = "ollama"
    model_name: str = "qwen2.5:1.5b"
    
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.3  # Slightly higher for more natural responses
    max_tokens: int = 150     # Reduced for faster generation
    timeout: int = 120         # Reduced timeout for faster failure detection


class RAGConfig(BaseModel):
    """RAG configuration"""
    top_k: int = 5  # Increased for better retrieval coverage
    score_threshold: int = 80  # Keep original threshold
    max_context_length: int = 1500  # Reduced for faster processing
    min_similarity: float = 0.1  # Lowered for better recall with BGE-M3
    max_similarity: float = 0.8


class DataConfig(BaseModel):
    """Data preparation configuration"""
    seed_file: str = "data/ecom_rag_seed_v2.sql"
    corpus_output: str = "data/corpus.jsonl"
    segments_output: str = "data/corpus_segments.jsonl"
    max_segment_length: int = 1000
    overlap: int = 100


class FinetuningConfig(BaseModel):
    """Fine-tuning configuration"""
    enabled: bool = False
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "out/lora-qwen25"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 0.0001
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100


class APIConfig(BaseModel):
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    cors_origins: list = ["http://localhost:3000", "http://localhost:8000"]


class PosttrainConfig(BaseModel):
    """Post-training configuration"""
    fuse_adapter: bool = True
    convert_to_gguf: bool = True
    ollama_model_name: str = "qwen25-lora-rag"
    temp_dir: str = "/tmp/lora_deploy"


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    rotation: str = "500 MB"
    retention: str = "10 days"
    log_file: str = "D:/mini-rag-chatbot/logs/app.log"


class PathsConfig(BaseModel):
    """Paths configuration"""
    data_dir: str = "D:/mini-rag-chatbot/data"
    models_dir: str = "D:/mini-rag-chatbot/models"
    logs_dir: str = "D:/mini-rag-chatbot/logs"
    output_dir: str = "D:/mini-rag-chatbot/out"


class Settings(BaseSettings):
    """
    Main application settings.
    Loads from config.yaml and overrides with environment variables.
    """
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    finetuning: FinetuningConfig = Field(default_factory=FinetuningConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    posttrain: PosttrainConfig = Field(default_factory=PosttrainConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    BUSINESS_RULE: Load configuration from YAML file and environment variables.
    Environment variables take precedence over YAML configuration.
    
    Args:
        config_path: Path to config YAML file (default: config/config.yaml)
    
    Returns:
        Settings instance with loaded configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    # Load YAML config
    config_dict: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
            # Replace environment variables in YAML
            for key, value in os.environ.items():
                yaml_content = yaml_content.replace(f"${{{key}}}", value)
                yaml_content = yaml_content.replace(f"${{{key}:.*?}}", value)
            config_dict = yaml.safe_load(yaml_content) or {}

    # Override with environment variables
    env_overrides = {
        'database': {
            'host': os.getenv('DATABASE_HOST'),
            'port': int(os.getenv('DATABASE_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD'),
        },
        'embedding': {
            'model_name': os.getenv('EMBEDDING_MODEL'),
            'cache_dir': os.getenv('EMBEDDINGS_CACHE_DIR'),
        },
        'llm': {
            'ollama_host': os.getenv('OLLAMA_HOST'),
            'model_name': os.getenv('OLLAMA_MODEL'),
        },
        'rag': {
            'score_threshold': int(os.getenv('SCORE_THRESHOLD', 80)),
            'top_k': int(os.getenv('TOP_K_RETRIEVAL', 5)),
        },
        'api': {
            'host': os.getenv('APP_HOST', '0.0.0.0'),
            'port': int(os.getenv('APP_PORT', 8000)),
        }
    }

    # Merge configs (env overrides YAML)
    def deep_merge(base: dict, override: dict) -> dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if value is not None:
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
        return result

    merged_config = deep_merge(config_dict, env_overrides)
    
    return Settings(**merged_config)


# Global settings instance
settings = load_config()

