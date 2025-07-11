"""
Embedding Generator for MQL5 RAG Engine
Task C12: Create Dockerfile for Lambda Container
Module: RAG Engine - Business Logic Layer

Handles text-to-vector conversion using SentenceTransformers:
- Model loading and caching for Lambda environment
- Query embedding with optimization for real-time inference
- Batch processing for offline operations
- Performance monitoring and error handling
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding operations."""
    model_name: str = "all-MiniLM-L6-v2"
    cache_dir: str = "/tmp/model_cache"
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    device: str = "cpu"  # Lambda runs on CPU
    batch_size: int = 32
    model_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}


@dataclass
class EmbeddingResult:
    """Result of embedding operation with metadata."""
    embeddings: np.ndarray
    processing_time: float
    input_count: int
    model_info: Dict[str, Any]
    truncated_inputs: int = 0


class EmbeddingGenerator:
    """
    Generates embeddings using SentenceTransformers.
    Optimized for Lambda environment with caching and performance monitoring.
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model: Optional[SentenceTransformer] = None
        self._model_info = {}
        self._load_stats = {}
        
    def _load_model(self) -> SentenceTransformer:
        """
        Load SentenceTransformer model with caching.
        
        Returns:
            Loaded SentenceTransformer model
            
        Raises:
            RuntimeError: If model loading fails
        """
        if self.model is not None:
            return self.model
            
        start_time = time.time()
        
        try:
            logger.info(f"Loading embedding model: {self.config.model_name}")
            
            # Create cache directory
            cache_path = Path(self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Check if model exists in cache
            model_cache_path = cache_path / self.config.model_name
            
            if model_cache_path.exists():
                logger.info(f"Loading model from cache: {model_cache_path}")
                self.model = SentenceTransformer(
                    str(model_cache_path),
                    device=self.config.device
                )
            else:
                logger.info(f"Downloading and caching model: {self.config.model_name}")
                self.model = SentenceTransformer(
                    self.config.model_name,
                    cache_folder=str(cache_path),
                    device=self.config.device,
                    **self.config.model_kwargs
                )
                
                # Save to cache for future use
                self.model.save(str(model_cache_path))
                logger.info(f"Model cached to: {model_cache_path}")
            
            # Configure model settings
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = min(
                    self.config.max_seq_length, 
                    self.model.max_seq_length
                )
            
            # Store model information
            self._model_info = {
                'model_name': self.config.model_name,
                'embedding_dimension': self.model.get_sentence_embedding_dimension(),
                'max_seq_length': getattr(self.model, 'max_seq_length', self.config.max_seq_length),
                'device': str(self.model.device),
                'tokenizer_vocab_size': getattr(self.model.tokenizer, 'vocab_size', 'unknown') if hasattr(self.model, 'tokenizer') else 'unknown'
            }
            
            load_time = time.time() - start_time
            self._load_stats = {
                'load_time': load_time,
                'cache_hit': model_cache_path.exists(),
                'model_size_mb': self._estimate_model_size()
            }
            
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            logger.info(f"Model info: {self._model_info}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.config.model_name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in MB."""
        if self.model is None:
            return 0.0
            
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            # Rough estimate: 4 bytes per parameter (float32)
            size_mb = (param_count * 4) / (1024 * 1024)
            return round(size_mb, 2)
        except Exception:
            return 0.0
    
    def embed_query(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single query.
        Optimized for real-time inference.
        
        Args:
            text: Input text to embed
            
        Returns:
            EmbeddingResult with embedding and metadata
            
        Raises:
            ValueError: If text is empty
            RuntimeError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
            
        return self.embed_texts([text])
    
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            EmbeddingResult with embeddings and metadata
            
        Raises:
            ValueError: If texts list is empty
            RuntimeError: If embedding generation fails
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
            
        start_time = time.time()
        
        try:
            model = self._load_model()
            
            # Preprocess texts
            processed_texts, truncated_count = self._preprocess_texts(texts)
            
            logger.info(f"Generating embeddings for {len(processed_texts)} texts")
            
            # Generate embeddings
            embeddings = model.encode(
                processed_texts,
                batch_size=min(self.config.batch_size, len(processed_texts)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings
            )
            
            # Ensure correct data type
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
                
            embeddings = embeddings.astype(np.float32)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Generated embeddings: shape={embeddings.shape}, time={processing_time*1000:.1f}ms")
            
            return EmbeddingResult(
                embeddings=embeddings,
                processing_time=processing_time,
                input_count=len(texts),
                model_info=self._model_info.copy(),
                truncated_inputs=truncated_count
            )
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    def _preprocess_texts(self, texts: List[str]) -> tuple[List[str], int]:
        """
        Preprocess texts for embedding generation.
        
        Args:
            texts: Raw input texts
            
        Returns:
            Tuple of (processed_texts, truncated_count)
        """
        processed_texts = []
        truncated_count = 0
        max_length = self.config.max_seq_length
        
        for text in texts:
            # Clean and strip text
            cleaned_text = text.strip()
            
            if not cleaned_text:
                logger.warning("Empty text found, replacing with placeholder")
                cleaned_text = "[EMPTY]"
            
            # Check length and truncate if necessary
            # Rough approximation: 1 token ≈ 4 characters for most languages
            approx_tokens = len(cleaned_text) / 4
            
            if approx_tokens > max_length:
                # Truncate to approximate token limit
                truncated_length = int(max_length * 4 * 0.9)  # 90% safety margin
                cleaned_text = cleaned_text[:truncated_length]
                truncated_count += 1
                logger.warning(f"Text truncated to {truncated_length} characters")
            
            processed_texts.append(cleaned_text)
        
        return processed_texts, truncated_count
    
    def embed_query_optimized(self, text: str) -> np.ndarray:
        """
        Fast embedding generation for single queries (Lambda optimized).
        Returns only the embedding vector for minimal latency.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
            
        try:
            model = self._load_model()
            
            # Minimal preprocessing
            processed_text = text.strip()
            if len(processed_text) > self.config.max_seq_length * 4:
                processed_text = processed_text[:self.config.max_seq_length * 4]
            
            # Fast embedding generation
            embedding = model.encode(
                [processed_text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings
            )[0]
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Optimized embedding generation failed: {e}")
            raise RuntimeError(f"Fast embedding failed: {e}")
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the current model."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        if not self._model_info:
            self._load_model()  # Ensure model is loaded
            
        return {
            **self._model_info,
            'load_stats': self._load_stats,
            'config': {
                'model_name': self.config.model_name,
                'cache_dir': self.config.cache_dir,
                'max_seq_length': self.config.max_seq_length,
                'normalize_embeddings': self.config.normalize_embeddings,
                'device': self.config.device,
                'batch_size': self.config.batch_size
            }
        }
    
    def warm_up(self) -> Dict[str, Any]:
        """
        Warm up the model by running a test embedding.
        Useful for Lambda container initialization.
        
        Returns:
            Warm-up statistics
        """
        logger.info("Warming up embedding model...")
        
        start_time = time.time()
        
        try:
            # Test embedding with sample MQL5 text
            test_text = "ArrayResize function in MQL5 Expert Advisor development"
            result = self.embed_query(test_text)
            
            warmup_time = time.time() - start_time
            
            warmup_stats = {
                'warmup_time': warmup_time,
                'test_embedding_shape': result.embeddings.shape,
                'model_ready': True,
                'embedding_time': result.processing_time
            }
            
            logger.info(f"Model warm-up completed in {warmup_time:.2f}s")
            return warmup_stats
            
        except Exception as e:
            logger.error(f"Model warm-up failed: {e}")
            return {
                'warmup_time': time.time() - start_time,
                'model_ready': False,
                'error': str(e)
            }
    
    def benchmark_performance(self, sample_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Benchmark embedding performance with sample texts.
        
        Args:
            sample_texts: Optional custom texts for benchmarking
            
        Returns:
            Performance benchmark results
        """
        if sample_texts is None:
            sample_texts = [
                "OnTick function implementation in MQL5",
                "ArrayResize dynamic memory allocation",
                "Expert Advisor trading strategy development",
                "MetaTrader 5 custom indicator programming",
                "MQL5 order management system design"
            ]
        
        logger.info(f"Benchmarking performance with {len(sample_texts)} samples")
        
        start_time = time.time()
        
        try:
            # Single text embedding benchmark
            single_times = []
            for text in sample_texts:
                single_start = time.time()
                self.embed_query_optimized(text)
                single_times.append(time.time() - single_start)
            
            # Batch embedding benchmark
            batch_start = time.time()
            batch_result = self.embed_texts(sample_texts)
            batch_time = time.time() - batch_start
            
            total_time = time.time() - start_time
            
            return {
                'total_benchmark_time': total_time,
                'single_embedding': {
                    'avg_time': np.mean(single_times),
                    'min_time': np.min(single_times),
                    'max_time': np.max(single_times),
                    'p95_time': np.percentile(single_times, 95)
                },
                'batch_embedding': {
                    'total_time': batch_time,
                    'per_text_time': batch_time / len(sample_texts),
                    'speedup_factor': np.mean(single_times) / (batch_time / len(sample_texts))
                },
                'model_info': self.get_model_info()
            }
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            return {'error': str(e)}


# Factory function for easy initialization
def create_embedding_generator(
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: str = "/tmp/model_cache",
    max_seq_length: int = 512
) -> EmbeddingGenerator:
    """
    Factory function to create an EmbeddingGenerator with common configurations.
    
    Args:
        model_name: SentenceTransformer model name
        cache_dir: Local cache directory for models
        max_seq_length: Maximum sequence length for inputs
        
    Returns:
        Configured EmbeddingGenerator instance
    """
    config = EmbeddingConfig(
        model_name=model_name,
        cache_dir=cache_dir,
        max_seq_length=max_seq_length,
        normalize_embeddings=True,
        device="cpu",
        batch_size=32
    )
    
    return EmbeddingGenerator(config)


# Utility functions
def validate_embedding_model(model_name: str) -> bool:
    """
    Validate if a SentenceTransformer model is available.
    
    Args:
        model_name: Model name to validate
        
    Returns:
        True if model is valid and accessible
    """
    try:
        # Try to load model info without downloading
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return True
    except Exception as e:
        logger.warning(f"Model validation failed for {model_name}: {e}")
        return False