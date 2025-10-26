"""
Embedding model wrapper for text-to-vector conversion.
Uses sentence-transformers with multilingual E5-small model optimized for Bulgarian.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
import torch

from config.settings import settings


class EmbeddingModel:
    """
    BUSINESS_RULE: Wrapper for embedding model.
    Converts text to vector representations for semantic search.
    Uses intfloat/multilingual-e5-small optimized for Bulgarian language.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        cache_dir: str = None
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name (default from config)
            device: Device to run model on ('cpu', 'cuda', or 'mps')
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name or settings.embedding.model_name
        self.device = device or settings.embedding.device
        self.cache_dir = cache_dir or settings.embedding.cache_dir
        self.dimension = settings.embedding.dimension
        
        # Auto-detect device if not specified
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        logger.info(f"Loading embedding model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        # PERFORMANCE_CRITICAL: Load model with caching
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
            logger.success(f"✓ Embedding model loaded successfully")
            logger.info(f"  Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"✗ Failed to load embedding model: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        BUSINESS_RULE: Encode texts into embeddings.
        Converts text to vector representation for similarity search.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding (default from config)
            show_progress: Show progress bar for batch encoding
            normalize: L2-normalize embeddings for cosine similarity
        
        Returns:
            Numpy array of shape (n_texts, embedding_dim) or (embedding_dim,) for single text
        """
        # Convert single string to list
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        batch_size = batch_size or settings.embedding.batch_size
        
        try:
            # PERFORMANCE_CRITICAL: Batch encode with normalization
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            # Return single embedding if single input
            if single_input:
                return embeddings[0]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        BUSINESS_RULE: Encode query with model-specific instruction.
        For BGE models – без префикс; за E5 – "query: ".
        
        Args:
            query: User query text
        
        Returns:
            Query embedding vector
        """
        name = (self.model_name or "").lower()
        if "e5" in name:
            q = f"query: {query}"
        else:
            q = query  # BGE/BGE-M3
        return self.encode(q, normalize=True)
    
    def encode_documents(
        self,
        documents: List[str],
        batch_size: int = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        BUSINESS_RULE: Encode documents with model-specific instruction.
        For BGE – без префикс; за E5 – "passage: ".
        
        Args:
            documents: List of document texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            Document embeddings array of shape (n_docs, embedding_dim)
        """
        name = (self.model_name or "").lower()
        if "e5" in name:
            docs = [f"passage: {doc}" for doc in documents]
        else:
            docs = documents  # BGE/BGE-M3
        return self.encode(
            docs,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=True
        )
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        BUSINESS_RULE: Calculate cosine similarity between embeddings.
        Assumes embeddings are already L2-normalized.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score in range [-1, 1]
        """
        return float(np.dot(embedding1, embedding2))
    
    def __repr__(self):
        return f"EmbeddingModel(model={self.model_name}, device={self.device}, dim={self.dimension})"


# Global embedding model instance (lazy loaded)
_embedding_model: EmbeddingModel = None


def get_embedding_model() -> EmbeddingModel:
    """
    BUSINESS_RULE: Get global embedding model instance (singleton pattern).
    Lazily loads model on first access.
    
    Returns:
        Shared embedding model instance
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model

