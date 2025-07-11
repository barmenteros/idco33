"""
RAG Engine Module for MQL5 Prompt Enrichment Middleware
Task C12: Create Dockerfile for Lambda Container
Module: RAG Engine

This module provides the core RAG functionality:
- Embedding generation using SentenceTransformers
- Vector similarity search using FAISS
- Storage operations with AWS S3 and DynamoDB

Components:
- embeddings: Handles text-to-vector conversion
- vector_search: Manages FAISS index operations
- storage: AWS service integrations (S3, DynamoDB)
"""

import logging

# Configure module-level logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "MQL5 RAG Team"
__description__ = "RAG Engine for MQL5 Documentation Retrieval"

# Module constants
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 500

# Export main classes and functions for easy imports
try:
    from .embeddings import EmbeddingGenerator
    from .vector_search import VectorSearchEngine
    from .storage import StorageManager
    
    # Define what gets imported with "from rag_engine import *"
    __all__ = [
        'EmbeddingGenerator',
        'VectorSearchEngine', 
        'StorageManager',
        'DEFAULT_EMBEDDING_MODEL',
        'DEFAULT_TOP_K',
        'DEFAULT_CHUNK_SIZE'
    ]
    
    logger.info(f"RAG Engine v{__version__} initialized successfully")
    
except ImportError as e:
    logger.warning(f"Some RAG Engine components not available: {e}")
    # Graceful degradation - define partial __all__
    __all__ = [
        'DEFAULT_EMBEDDING_MODEL',
        'DEFAULT_TOP_K', 
        'DEFAULT_CHUNK_SIZE'
    ]


def get_version() -> str:
    """Return the current version of the RAG Engine module."""
    return __version__


def get_module_info() -> dict:
    """Return comprehensive module information."""
    return {
        'name': 'rag_engine',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'components': __all__,
        'default_model': DEFAULT_EMBEDDING_MODEL,
        'default_top_k': DEFAULT_TOP_K,
        'default_chunk_size': DEFAULT_CHUNK_SIZE
    }