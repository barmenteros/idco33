"""
Vector Search Engine for MQL5 RAG Engine
Task C12: Create Dockerfile for Lambda Container
Module: RAG Engine - Search Logic Layer

Handles vector similarity search using FAISS:
- FAISS index loading and management
- Fast similarity search with configurable parameters
- Search result ranking and filtering
- Performance optimization for Lambda environment
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import faiss

# Import from our modules
from .embeddings import EmbeddingGenerator
from .storage import StorageManager

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for vector search operations."""
    top_k: int = 5
    search_threshold: float = 0.0  # Minimum similarity score
    max_results: int = 20
    index_type: str = "auto"  # auto, flat, hnsw, ivf
    search_params: Dict[str, Any] = None
    rerank_results: bool = True
    
    def __post_init__(self):
        if self.search_params is None:
            self.search_params = {}


@dataclass
class SearchResult:
    """Individual search result with metadata."""
    doc_id: str
    similarity_score: float
    vector_index: int
    rank: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResponse:
    """Complete search response with results and metadata."""
    query: str
    results: List[SearchResult]
    search_time: float
    total_results: int
    index_info: Dict[str, Any]
    config_used: Dict[str, Any]


class VectorSearchEngine:
    """
    FAISS-based vector search engine optimized for Lambda environment.
    Handles index loading, caching, and fast similarity search.
    """
    
    def __init__(self, config: SearchConfig, storage_manager: StorageManager, embedding_generator: EmbeddingGenerator):
        self.config = config
        self.storage = storage_manager
        self.embedder = embedding_generator
        
        # FAISS components
        self.index: Optional[faiss.Index] = None
        self.doc_id_mapping: Optional[Dict[str, str]] = None
        self.reverse_mapping: Optional[Dict[str, int]] = None
        
        # Cache information
        self._index_info = {}
        self._load_stats = {}
        self._search_stats = {
            'total_searches': 0,
            'avg_search_time': 0.0,
            'cache_hits': 0
        }
    
    def _load_faiss_index(self, force_refresh: bool = False) -> None:
        """
        Load FAISS index from storage with caching.
        
        Args:
            force_refresh: Force reload even if index is cached
            
        Raises:
            RuntimeError: If index loading fails
        """
        if self.index is not None and not force_refresh:
            logger.info("Using cached FAISS index")
            return
        
        start_time = time.time()
        
        try:
            logger.info("Loading FAISS index from storage")
            
            # Download index files
            index_file, mapping_file = self.storage.download_faiss_index(force_refresh)
            
            # Load FAISS index
            logger.info(f"Reading FAISS index from: {index_file}")
            self.index = faiss.read_index(index_file)
            
            # Load document ID mapping
            logger.info(f"Loading doc ID mapping from: {mapping_file}")
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            self.doc_id_mapping = mapping_data.get('index_to_docid', {})
            
            # Create reverse mapping for efficiency
            self.reverse_mapping = {doc_id: int(idx) for idx, doc_id in self.doc_id_mapping.items()}
            
            # Validate loaded data
            self._validate_index()
            
            # Store index information
            self._index_info = {
                'total_vectors': self.index.ntotal,
                'vector_dimension': self.index.d,
                'index_type': type(self.index).__name__,
                'is_trained': getattr(self.index, 'is_trained', True),
                'mapping_size': len(self.doc_id_mapping),
                'metric_type': self._get_metric_type()
            }
            
            load_time = time.time() - start_time
            self._load_stats = {
                'load_time': load_time,
                'index_file_path': index_file,
                'mapping_file_path': mapping_file,
                'force_refresh': force_refresh
            }
            
            logger.info(f"FAISS index loaded successfully in {load_time:.2f}s")
            logger.info(f"Index info: {self._index_info}")
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise RuntimeError(f"FAISS index loading failed: {e}")
    
    def _validate_index(self) -> None:
        """Validate loaded FAISS index and mappings."""
        if self.index is None:
            raise ValueError("FAISS index is None")
        
        if self.index.ntotal == 0:
            raise ValueError("FAISS index is empty")
        
        if not self.doc_id_mapping:
            raise ValueError("Document ID mapping is empty")
        
        # Check dimension consistency with embedder
        try:
            expected_dim = self.embedder.get_embedding_dimension()
            if self.index.d != expected_dim:
                logger.warning(f"Dimension mismatch: index={self.index.d}, embedder={expected_dim}")
        except Exception:
            logger.warning("Could not validate embedding dimension consistency")
        
        # Validate mapping consistency
        if len(self.doc_id_mapping) != self.index.ntotal:
            logger.warning(f"Mapping size mismatch: mapping={len(self.doc_id_mapping)}, index={self.index.ntotal}")
        
        logger.info("FAISS index validation completed")
    
    def _get_metric_type(self) -> str:
        """Determine the metric type used by the FAISS index."""
        index_type = type(self.index).__name__
        
        if 'IP' in index_type or 'InnerProduct' in index_type:
            return 'inner_product'
        elif 'L2' in index_type:
            return 'l2'
        else:
            return 'unknown'
    
    def search_similar(self, query: str, top_k: Optional[int] = None) -> SearchResponse:
        """
        Search for similar documents using query text.
        
        Args:
            query: Query text to search for
            top_k: Number of results to return (overrides config)
            
        Returns:
            SearchResponse with results and metadata
            
        Raises:
            ValueError: If query is empty
            RuntimeError: If search fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_vector = self.embedder.embed_query_optimized(query.strip())
            
            # Perform vector search
            search_results = self.search_by_vector(query_vector, top_k)
            
            search_time = time.time() - start_time
            
            # Update search statistics
            self._update_search_stats(search_time)
            
            response = SearchResponse(
                query=query,
                results=search_results,
                search_time=search_time,
                total_results=len(search_results),
                index_info=self._index_info.copy(),
                config_used={
                    'top_k': top_k or self.config.top_k,
                    'search_threshold': self.config.search_threshold,
                    'rerank_results': self.config.rerank_results
                }
            )
            
            logger.info(f"Search completed: query='{query[:50]}...', results={len(search_results)}, time={search_time*1000:.1f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Search failed for query '{query[:50]}...': {e}")
            raise RuntimeError(f"Vector search failed: {e}")
    
    def search_by_vector(self, query_vector: np.ndarray, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Search for similar documents using a query vector.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
            
        Raises:
            RuntimeError: If search fails
        """
        if top_k is None:
            top_k = self.config.top_k
        
        top_k = min(top_k, self.config.max_results)
        
        start_time = time.time()
        
        try:
            # Load index if not already loaded
            self._load_faiss_index()
            
            # Validate query vector
            if query_vector.shape[0] != self.index.d:
                raise ValueError(f"Query vector dimension {query_vector.shape[0]} doesn't match index dimension {self.index.d}")
            
            # Reshape vector for FAISS (expects 2D array)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # Ensure correct data type
            query_vector = query_vector.astype(np.float32)
            
            # Perform FAISS search
            logger.debug(f"Performing FAISS search: top_k={top_k}")
            
            scores, indices = self.index.search(query_vector, top_k)
            
            # Process results
            results = self._process_search_results(scores[0], indices[0])
            
            search_time = time.time() - start_time
            logger.debug(f"FAISS search completed in {search_time*1000:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise RuntimeError(f"FAISS search error: {e}")
    
    def _process_search_results(self, scores: np.ndarray, indices: np.ndarray) -> List[SearchResult]:
        """
        Process raw FAISS search results into SearchResult objects.
        
        Args:
            scores: Similarity scores from FAISS
            indices: Vector indices from FAISS
            
        Returns:
            List of processed SearchResult objects
        """
        results = []
        
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            # Skip invalid indices
            if idx < 0 or idx >= len(self.doc_id_mapping):
                logger.warning(f"Invalid index {idx} in search results")
                continue
            
            # Apply threshold filter
            if score < self.config.search_threshold:
                logger.debug(f"Result {rank} below threshold: {score} < {self.config.search_threshold}")
                continue
            
            # Get document ID
            doc_id = self.doc_id_mapping.get(str(idx))
            if not doc_id:
                logger.warning(f"No doc_id found for index {idx}")
                continue
            
            # Create search result
            result = SearchResult(
                doc_id=doc_id,
                similarity_score=float(score),
                vector_index=int(idx),
                rank=rank + 1,
                metadata={
                    'raw_score': float(score),
                    'vector_index': int(idx),
                    'search_rank': rank + 1
                }
            )
            
            results.append(result)
        
        # Re-rank results if enabled
        if self.config.rerank_results:
            results = self._rerank_results(results)
        
        logger.debug(f"Processed {len(results)} search results")
        return results
    
    def _rerank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Re-rank search results using additional criteria.
        
        Args:
            results: Initial search results
            
        Returns:
            Re-ranked search results
        """
        # For now, just sort by similarity score (descending)
        # Future: Could add more sophisticated ranking factors
        reranked = sorted(results, key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks after re-ranking
        for i, result in enumerate(reranked):
            result.rank = i + 1
            result.metadata['reranked'] = True
        
        return reranked
    
    def _update_search_stats(self, search_time: float) -> None:
        """Update internal search statistics."""
        self._search_stats['total_searches'] += 1
        
        # Update running average
        total = self._search_stats['total_searches']
        current_avg = self._search_stats['avg_search_time']
        self._search_stats['avg_search_time'] = ((current_avg * (total - 1)) + search_time) / total
    
    def get_doc_by_id(self, doc_id: str) -> Optional[SearchResult]:
        """
        Get a document by its ID (if it exists in the index).
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            SearchResult object or None if not found
        """
        if not self.reverse_mapping:
            self._load_faiss_index()
        
        vector_idx = self.reverse_mapping.get(doc_id)
        if vector_idx is None:
            return None
        
        return SearchResult(
            doc_id=doc_id,
            similarity_score=1.0,  # Perfect match
            vector_index=vector_idx,
            rank=1,
            metadata={'direct_lookup': True}
        )
    
    def get_random_docs(self, count: int = 5) -> List[SearchResult]:
        """
        Get random documents from the index for testing/sampling.
        
        Args:
            count: Number of random documents to return
            
        Returns:
            List of random SearchResult objects
        """
        self._load_faiss_index()
        
        if self.index.ntotal == 0:
            return []
        
        # Generate random indices
        max_count = min(count, self.index.ntotal)
        random_indices = np.random.choice(self.index.ntotal, size=max_count, replace=False)
        
        results = []
        for i, idx in enumerate(random_indices):
            doc_id = self.doc_id_mapping.get(str(idx))
            if doc_id:
                result = SearchResult(
                    doc_id=doc_id,
                    similarity_score=0.0,  # No meaningful score for random selection
                    vector_index=int(idx),
                    rank=i + 1,
                    metadata={'random_selection': True}
                )
                results.append(result)
        
        return results
    
    def benchmark_search(self, test_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Benchmark search performance with test queries.
        
        Args:
            test_queries: Optional custom test queries
            
        Returns:
            Performance benchmark results
        """
        if test_queries is None:
            test_queries = [
                "ArrayResize function MQL5",
                "OnTick event handler implementation",
                "Expert Advisor trading strategy",
                "MetaTrader 5 custom indicator",
                "MQL5 order management system"
            ]
        
        logger.info(f"Benchmarking search with {len(test_queries)} queries")
        
        start_time = time.time()
        search_times = []
        result_counts = []
        
        try:
            for query in test_queries:
                query_start = time.time()
                response = self.search_similar(query, top_k=self.config.top_k)
                query_time = time.time() - query_start
                
                search_times.append(query_time)
                result_counts.append(len(response.results))
            
            total_time = time.time() - start_time
            
            return {
                'total_benchmark_time': total_time,
                'query_count': len(test_queries),
                'search_performance': {
                    'avg_time': np.mean(search_times),
                    'min_time': np.min(search_times),
                    'max_time': np.max(search_times),
                    'p95_time': np.percentile(search_times, 95),
                    'p99_time': np.percentile(search_times, 99)
                },
                'result_statistics': {
                    'avg_results': np.mean(result_counts),
                    'min_results': np.min(result_counts),
                    'max_results': np.max(result_counts)
                },
                'index_info': self._index_info,
                'search_stats': self._search_stats
            }
            
        except Exception as e:
            logger.error(f"Search benchmark failed: {e}")
            return {'error': str(e)}
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get comprehensive index information."""
        if not self._index_info:
            self._load_faiss_index()
        
        return {
            'index_info': self._index_info,
            'load_stats': self._load_stats,
            'search_stats': self._search_stats,
            'config': {
                'top_k': self.config.top_k,
                'search_threshold': self.config.search_threshold,
                'max_results': self.config.max_results,
                'rerank_results': self.config.rerank_results
            }
        }
    
    def warm_up(self) -> Dict[str, Any]:
        """
        Warm up the search engine by loading index and running test search.
        
        Returns:
            Warm-up statistics
        """
        logger.info("Warming up vector search engine...")
        
        start_time = time.time()
        
        try:
            # Load index
            self._load_faiss_index()
            
            # Run test search
            test_query = "MQL5 ArrayResize function implementation"
            response = self.search_similar(test_query, top_k=3)
            
            warmup_time = time.time() - start_time
            
            warmup_stats = {
                'warmup_time': warmup_time,
                'index_loaded': True,
                'test_search_time': response.search_time,
                'test_results_count': len(response.results),
                'index_ready': True
            }
            
            logger.info(f"Search engine warm-up completed in {warmup_time:.2f}s")
            return warmup_stats
            
        except Exception as e:
            logger.error(f"Search engine warm-up failed: {e}")
            return {
                'warmup_time': time.time() - start_time,
                'index_ready': False,
                'error': str(e)
            }


# Factory function for easy initialization
def create_vector_search_engine(
    storage_manager: StorageManager,
    embedding_generator: EmbeddingGenerator,
    top_k: int = 5,
    search_threshold: float = 0.0
) -> VectorSearchEngine:
    """
    Factory function to create a VectorSearchEngine with common configurations.
    
    Args:
        storage_manager: Configured StorageManager instance
        embedding_generator: Configured EmbeddingGenerator instance
        top_k: Default number of results to return
        search_threshold: Minimum similarity score threshold
        
    Returns:
        Configured VectorSearchEngine instance
    """
    config = SearchConfig(
        top_k=top_k,
        search_threshold=search_threshold,
        max_results=20,
        rerank_results=True
    )
    
    return VectorSearchEngine(config, storage_manager, embedding_generator)