#!/usr/bin/env python3
"""
MQL5 FAISS Index Builder - Task B9
Build FAISS index locally from embeddings for fast similarity search

This module constructs a FAISS index from Task B8 embeddings
for real-time similarity search in the RAG pipeline.
"""

import os
import json
import pickle
import logging
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import faiss


@dataclass
class IndexConfig:
    """Configuration for FAISS index construction"""
    index_type: str
    metric_type: str
    dimension: int
    total_vectors: int
    build_parameters: Dict[str, Any]
    search_parameters: Dict[str, Any]


@dataclass
class SearchResult:
    """Result from FAISS similarity search"""
    doc_id: str
    score: float
    vector_index: int
    distance: float


class MQL5FAISSBuilder:
    """Build FAISS index for MQL5 embeddings with comprehensive validation"""
    
    def __init__(self, 
                 embeddings_dir: str = "./mql5_test/embeddings",
                 output_dir: str = "./mql5_test/faiss_index",
                 index_type: str = "IndexFlatIP"):
        
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.index_type = index_type
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embeddings = None
        self.docid_mappings = None
        self.index = None
        self.index_config = None
        
        # Build statistics
        self.stats = {
            'build_time': 0,
            'index_size_mb': 0,
            'vectors_indexed': 0,
            'dimension': 0,
            'search_performance': {},
            'memory_usage': {}
        }
    
    def load_embeddings(self) -> bool:
        """Load embeddings matrix and DocID mappings from Task B8"""
        try:
            # Load embeddings matrix
            embeddings_path = self.embeddings_dir / "embeddings_matrix.npy"
            if not embeddings_path.exists():
                self.logger.error(f"Embeddings matrix not found: {embeddings_path}")
                return False
            
            self.logger.info(f"Loading embeddings from: {embeddings_path}")
            self.embeddings = np.load(embeddings_path)
            
            # Validate embeddings
            if len(self.embeddings.shape) != 2:
                self.logger.error(f"Invalid embeddings shape: {self.embeddings.shape}")
                return False
            
            vectors, dimensions = self.embeddings.shape
            self.logger.info(f"Loaded embeddings: {vectors} vectors × {dimensions} dimensions")
            
            # Load DocID mappings
            mappings_path = self.embeddings_dir / "docid_mappings.json"
            if not mappings_path.exists():
                self.logger.error(f"DocID mappings not found: {mappings_path}")
                return False
            
            with open(mappings_path, 'r') as f:
                self.docid_mappings = json.load(f)
            
            # Validate mappings
            expected_vectors = self.docid_mappings.get('total_embeddings', 0)
            if expected_vectors != vectors:
                self.logger.warning(f"Mapping mismatch: expected {expected_vectors}, got {vectors}")
            
            self.logger.info(f"Loaded DocID mappings for {len(self.docid_mappings['index_to_docid'])} vectors")
            
            # Update statistics
            self.stats['vectors_indexed'] = vectors
            self.stats['dimension'] = dimensions
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            return False
    
    def create_index_config(self) -> IndexConfig:
        """Create configuration for FAISS index based on data characteristics"""
        vectors, dimensions = self.embeddings.shape
        
        # Determine optimal index type based on dataset size
        if vectors <= 1000:
            # Small dataset: use exact search
            recommended_type = "IndexFlatIP"
            build_params = {}
            search_params = {"k": 5}
        elif vectors <= 10000:
            # Medium dataset: use HNSW for good balance
            recommended_type = "IndexHNSWFlat"
            build_params = {"M": 16, "efConstruction": 200}
            search_params = {"k": 5, "efSearch": 64}
        else:
            # Large dataset: use IVF for memory efficiency
            recommended_type = "IndexIVFFlat"
            ncentroids = min(int(np.sqrt(vectors)), 256)
            build_params = {"ncentroids": ncentroids, "nprobe": min(ncentroids // 4, 32)}
            search_params = {"k": 5, "nprobe": build_params["nprobe"]}
        
        # Override with user preference if specified
        if self.index_type:
            recommended_type = self.index_type
        
        # Determine metric type (IP for normalized embeddings, L2 for unnormalized)
        metric_type = "IP"  # Inner Product for normalized embeddings from SentenceTransformers
        
        config = IndexConfig(
            index_type=recommended_type,
            metric_type=metric_type,
            dimension=dimensions,
            total_vectors=vectors,
            build_parameters=build_params,
            search_parameters=search_params
        )
        
        self.logger.info(f"Index configuration:")
        self.logger.info(f"  - Type: {config.index_type}")
        self.logger.info(f"  - Metric: {config.metric_type}")
        self.logger.info(f"  - Vectors: {config.total_vectors}")
        self.logger.info(f"  - Dimensions: {config.dimension}")
        self.logger.info(f"  - Build params: {config.build_parameters}")
        
        return config
    
    def build_index(self, config: IndexConfig) -> bool:
        """Build FAISS index according to configuration"""
        try:
            self.logger.info(f"Building FAISS index: {config.index_type}")
            start_time = time.time()
            
            # Create index based on type
            if config.index_type == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(config.dimension)
                
            elif config.index_type == "IndexFlatL2":
                self.index = faiss.IndexFlatL2(config.dimension)
                
            elif config.index_type == "IndexHNSWFlat":
                self.index = faiss.IndexHNSWFlat(config.dimension, config.build_parameters.get("M", 16))
                # Set construction parameters
                self.index.hnsw.efConstruction = config.build_parameters.get("efConstruction", 200)
                
            elif config.index_type == "IndexIVFFlat":
                # Create quantizer
                quantizer = faiss.IndexFlatIP(config.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, config.dimension, 
                                              config.build_parameters.get("ncentroids", 100))
                # Train the index
                self.logger.info("Training IVF index...")
                self.index.train(self.embeddings.astype(np.float32))
                
            else:
                self.logger.error(f"Unsupported index type: {config.index_type}")
                return False
            
            # Add vectors to index
            self.logger.info(f"Adding {config.total_vectors} vectors to index...")
            embeddings_f32 = self.embeddings.astype(np.float32)
            
            # Use standard add method (IndexFlat types don't support add_with_ids)
            if config.index_type.startswith("IndexFlat"):
                self.index.add(embeddings_f32)
            elif hasattr(self.index, 'add_with_ids'):
                # Use IDs for other index types that support it
                ids = np.arange(config.total_vectors, dtype=np.int64)
                self.index.add_with_ids(embeddings_f32, ids)
            else:
                self.index.add(embeddings_f32)
            
            build_time = time.time() - start_time
            self.stats['build_time'] = build_time
            
            # Verify index
            if self.index.ntotal != config.total_vectors:
                self.logger.error(f"Index size mismatch: expected {config.total_vectors}, got {self.index.ntotal}")
                return False
            
            self.logger.info(f"Index built successfully:")
            self.logger.info(f"  - Build time: {build_time:.2f}s")
            self.logger.info(f"  - Vectors indexed: {self.index.ntotal}")
            self.logger.info(f"  - Index trained: {self.index.is_trained}")
            
            # Store configuration
            self.index_config = config
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build index: {e}")
            return False
    
    def test_search_performance(self, num_queries: int = 100) -> Dict[str, float]:
        """Test search performance with sample queries"""
        if not self.index or not self.embeddings.any():
            return {}
        
        try:
            self.logger.info(f"Testing search performance with {num_queries} queries...")
            
            # Select random vectors as queries
            query_indices = np.random.choice(len(self.embeddings), 
                                           min(num_queries, len(self.embeddings)), 
                                           replace=False)
            query_vectors = self.embeddings[query_indices].astype(np.float32)
            
            # Perform searches with timing
            k_values = [1, 3, 5, 10]
            performance_results = {}
            
            for k in k_values:
                search_times = []
                
                for i in range(min(20, len(query_vectors))):  # Test with subset for timing
                    query = query_vectors[i:i+1]  # Single query
                    
                    start_time = time.time()
                    distances, indices = self.index.search(query, k)
                    search_time = time.time() - start_time
                    
                    search_times.append(search_time * 1000)  # Convert to milliseconds
                
                performance_results[f'top_{k}'] = {
                    'avg_time_ms': np.mean(search_times),
                    'min_time_ms': np.min(search_times),
                    'max_time_ms': np.max(search_times),
                    'p95_time_ms': np.percentile(search_times, 95)
                }
            
            # Overall performance summary
            avg_search_time = np.mean([perf['avg_time_ms'] for perf in performance_results.values()])
            
            self.logger.info(f"Search performance results:")
            for k, perf in performance_results.items():
                self.logger.info(f"  - {k}: {perf['avg_time_ms']:.3f}ms avg, {perf['p95_time_ms']:.3f}ms p95")
            
            self.stats['search_performance'] = performance_results
            
            return performance_results
            
        except Exception as e:
            self.logger.error(f"Search performance test failed: {e}")
            return {}
    
    def validate_search_quality(self, num_tests: int = 50) -> Dict[str, Any]:
        """Validate search quality with known similar documents"""
        if not self.index or not self.embeddings.any():
            return {}
        
        try:
            self.logger.info(f"Validating search quality with {num_tests} tests...")
            
            validation_results = {
                'self_similarity_tests': [],
                'neighbor_consistency_tests': [],
                'quality_metrics': {}
            }
            
            # Test 1: Self-similarity (each vector should be most similar to itself)
            test_indices = np.random.choice(len(self.embeddings), min(num_tests, len(self.embeddings)), replace=False)
            
            self_similarity_scores = []
            for idx in test_indices[:20]:  # Test subset
                query = self.embeddings[idx:idx+1].astype(np.float32)
                distances, indices = self.index.search(query, k=1)
                
                # Check if the top result is the query itself
                is_correct = indices[0][0] == idx
                similarity_score = distances[0][0] if len(distances[0]) > 0 else 0
                
                self_similarity_scores.append(similarity_score)
                validation_results['self_similarity_tests'].append({
                    'query_index': int(idx),
                    'top_result_index': int(indices[0][0]) if len(indices[0]) > 0 else -1,
                    'is_correct': is_correct,
                    'similarity_score': float(similarity_score)
                })
            
            # Test 2: Neighbor consistency (similar docs should have similar neighbors)
            neighbor_consistency_scores = []
            for idx in test_indices[:10]:  # Smaller subset
                query = self.embeddings[idx:idx+1].astype(np.float32)
                distances, indices = self.index.search(query, k=5)
                
                if len(indices[0]) >= 2:
                    # Get neighbors of the top result
                    neighbor_query = self.embeddings[indices[0][1]:indices[0][1]+1].astype(np.float32)
                    neighbor_distances, neighbor_indices = self.index.search(neighbor_query, k=5)
                    
                    # Check overlap between neighborhoods
                    original_neighbors = set(indices[0])
                    neighbor_neighbors = set(neighbor_indices[0])
                    overlap = len(original_neighbors.intersection(neighbor_neighbors))
                    consistency_score = overlap / len(original_neighbors.union(neighbor_neighbors))
                    
                    neighbor_consistency_scores.append(consistency_score)
                    validation_results['neighbor_consistency_tests'].append({
                        'query_index': int(idx),
                        'consistency_score': consistency_score,
                        'overlap_count': overlap
                    })
            
            # Calculate quality metrics
            validation_results['quality_metrics'] = {
                'self_similarity_accuracy': np.mean([test['is_correct'] for test in validation_results['self_similarity_tests']]),
                'avg_self_similarity_score': np.mean(self_similarity_scores) if self_similarity_scores else 0,
                'avg_neighbor_consistency': np.mean(neighbor_consistency_scores) if neighbor_consistency_scores else 0,
                'total_tests_performed': len(test_indices)
            }
            
            # Quality assessment
            accuracy = validation_results['quality_metrics']['self_similarity_accuracy']
            consistency = validation_results['quality_metrics']['avg_neighbor_consistency']
            
            if accuracy >= 0.95 and consistency >= 0.3:
                quality_rating = "EXCELLENT"
            elif accuracy >= 0.9 and consistency >= 0.2:
                quality_rating = "GOOD"
            elif accuracy >= 0.8:
                quality_rating = "FAIR"
            else:
                quality_rating = "POOR"
            
            validation_results['quality_metrics']['overall_rating'] = quality_rating
            
            self.logger.info(f"Search quality validation:")
            self.logger.info(f"  - Self-similarity accuracy: {accuracy:.1%}")
            self.logger.info(f"  - Neighbor consistency: {consistency:.3f}")
            self.logger.info(f"  - Overall rating: {quality_rating}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Search quality validation failed: {e}")
            return {}
    
    def sample_search_demo(self, query_texts: List[str] = None) -> List[Dict]:
        """Demonstrate search with sample MQL5 queries"""
        if not query_texts:
            # Default MQL5-related queries for testing
            query_texts = [
                "OnTick function implementation",
                "ArrayResize dynamic memory",
                "Expert Advisor initialization",
                "OrderSend trading function",
                "double precision variables"
            ]
        
        if not self.index:
            self.logger.error("Index not built")
            return []
        
        try:
            # Note: In real implementation, we'd use the same SentenceTransformer
            # For demo, we'll use random vectors that represent these queries
            demo_results = []
            
            for i, query_text in enumerate(query_texts):
                # Simulate query embedding (in practice, use same model as Task B8)
                # For demo, select a random vector to represent the query
                random_idx = np.random.randint(0, len(self.embeddings))
                query_vector = self.embeddings[random_idx:random_idx+1].astype(np.float32)
                
                # Search
                distances, indices = self.index.search(query_vector, k=3)
                
                # Convert to results with DocIDs
                search_results = []
                for j, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx >= 0:  # Valid result
                        doc_id = self.docid_mappings['index_to_docid'].get(str(idx), f"unknown_{idx}")
                        search_results.append({
                            'rank': j + 1,
                            'doc_id': doc_id,
                            'score': float(dist),
                            'vector_index': int(idx)
                        })
                
                demo_results.append({
                    'query_text': query_text,
                    'results': search_results
                })
            
            self.logger.info("Sample search demonstration:")
            for demo in demo_results:
                self.logger.info(f"  Query: '{demo['query_text']}'")
                for result in demo['results']:
                    self.logger.info(f"    {result['rank']}. {result['doc_id']} (score: {result['score']:.3f})")
            
            return demo_results
            
        except Exception as e:
            self.logger.error(f"Sample search demo failed: {e}")
            return []
    
    def save_index(self) -> Dict[str, str]:
        """Save FAISS index and metadata files"""
        if not self.index or not self.index_config:
            self.logger.error("No index to save")
            return {}
        
        try:
            outputs = {}
            
            # Save FAISS index
            index_path = self.output_dir / "index.faiss"
            faiss.write_index(self.index, str(index_path))
            outputs['faiss_index'] = str(index_path)
            self.logger.info(f"FAISS index saved: {index_path}")
            
            # Calculate index file size
            index_size_mb = index_path.stat().st_size / (1024 * 1024)
            self.stats['index_size_mb'] = index_size_mb
            
            # Save index metadata and DocID mappings
            metadata = {
                'index_config': {
                    'index_type': self.index_config.index_type,
                    'metric_type': self.index_config.metric_type,
                    'dimension': self.index_config.dimension,
                    'total_vectors': self.index_config.total_vectors,
                    'build_parameters': self.index_config.build_parameters,
                    'search_parameters': self.index_config.search_parameters
                },
                'docid_mappings': self.docid_mappings,
                'build_statistics': self.stats,
                'created_at': datetime.now().isoformat(),
                'faiss_version': faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'
            }
            
            # Save as pickle (for Lambda loading)
            metadata_pkl_path = self.output_dir / "index.pkl"
            with open(metadata_pkl_path, 'wb') as f:
                pickle.dump(metadata, f)
            outputs['metadata_pickle'] = str(metadata_pkl_path)
            self.logger.info(f"Index metadata (pickle) saved: {metadata_pkl_path}")
            
            # Save as JSON (for human readability)
            metadata_json_path = self.output_dir / "index_metadata.json"
            # Convert numpy types for JSON serialization
            json_metadata = json.loads(json.dumps(metadata, default=str))
            with open(metadata_json_path, 'w') as f:
                json.dump(json_metadata, f, indent=2)
            outputs['metadata_json'] = str(metadata_json_path)
            self.logger.info(f"Index metadata (JSON) saved: {metadata_json_path}")
            
            # Summary report
            summary_path = self.output_dir / "build_summary.json"
            summary = {
                'index_file_size_mb': index_size_mb,
                'total_vectors': self.index.ntotal,
                'index_dimension': self.index.d,
                'index_type': self.index_config.index_type,
                'build_time_seconds': self.stats['build_time'],
                'vectors_per_second': self.stats['vectors_indexed'] / self.stats['build_time'] if self.stats['build_time'] > 0 else 0,
                'memory_footprint_estimate_mb': (self.stats['vectors_indexed'] * self.stats['dimension'] * 4) / (1024 * 1024),
                'ready_for_lambda': True,
                'ready_for_s3_upload': True
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            outputs['build_summary'] = str(summary_path)
            self.logger.info(f"Build summary saved: {summary_path}")
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            return {}
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete FAISS index building pipeline"""
        pipeline_results = {
            'success': False,
            'embeddings_loaded': False,
            'index_built': False,
            'index_tested': False,
            'outputs': {},
            'performance': {},
            'validation': {},
            'stats': {}
        }
        
        try:
            # Step 1: Load embeddings
            self.logger.info("=== Step 1: Loading Embeddings ===")
            if not self.load_embeddings():
                return pipeline_results
            pipeline_results['embeddings_loaded'] = True
            
            # Step 2: Create index configuration
            self.logger.info("=== Step 2: Creating Index Configuration ===")
            config = self.create_index_config()
            
            # Step 3: Build index
            self.logger.info("=== Step 3: Building FAISS Index ===")
            if not self.build_index(config):
                return pipeline_results
            pipeline_results['index_built'] = True
            
            # Step 4: Test performance
            self.logger.info("=== Step 4: Testing Search Performance ===")
            performance = self.test_search_performance()
            pipeline_results['performance'] = performance
            
            # Step 5: Validate quality
            self.logger.info("=== Step 5: Validating Search Quality ===")
            validation = self.validate_search_quality()
            pipeline_results['validation'] = validation
            
            # Step 6: Sample search demo
            self.logger.info("=== Step 6: Sample Search Demonstration ===")
            demo_results = self.sample_search_demo()
            
            # Step 7: Save index
            self.logger.info("=== Step 7: Saving FAISS Index ===")
            outputs = self.save_index()
            pipeline_results['outputs'] = outputs
            
            pipeline_results['index_tested'] = bool(performance and validation)
            pipeline_results['stats'] = self.stats
            pipeline_results['success'] = True
            
            self.logger.info("=== Pipeline Complete ===")
            self.logger.info(f"Index built successfully: {self.stats['vectors_indexed']} vectors")
            self.logger.info(f"Index size: {self.stats['index_size_mb']:.2f} MB")
            self.logger.info(f"Search quality: {validation.get('quality_metrics', {}).get('overall_rating', 'Unknown')}")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return pipeline_results


def main():
    """Main function for FAISS index building"""
    
    print("=== MQL5 FAISS Index Builder (Task B9) ===")
    
    # Initialize builder
    builder = MQL5FAISSBuilder(
        embeddings_dir="./mql5_test/embeddings",
        output_dir="./mql5_test/faiss_index",
        index_type="IndexFlatIP"  # Start with exact search for quality
    )
    
    # Run complete pipeline
    results = builder.run_complete_pipeline()
    
    # Display results
    print("\n=== FAISS Index Building Results ===")
    if results['success']:
        print(f"✅ Successfully built FAISS index")
        print(f"📊 Vectors indexed: {builder.stats.get('vectors_indexed', 'N/A')}")
        print(f"📏 Index dimension: {builder.stats.get('dimension', 'N/A')}")
        print(f"⚡ Build time: {builder.stats.get('build_time', 0):.2f}s")
        print(f"💾 Index size: {builder.stats.get('index_size_mb', 0):.2f} MB")
        
        if 'performance' in results and results['performance']:
            avg_search = np.mean([perf['avg_time_ms'] for perf in results['performance'].values()])
            print(f"🔍 Search speed: {avg_search:.3f}ms average")
        
        if 'validation' in results and results['validation']:
            quality = results['validation'].get('quality_metrics', {}).get('overall_rating', 'Unknown')
            print(f"🎯 Search quality: {quality}")
        
        if 'outputs' in results:
            print(f"💾 Index saved to: {results['outputs'].get('faiss_index', 'N/A')}")
    else:
        print("❌ FAISS index building failed. Check logs for details.")
    
    return results


if __name__ == "__main__":
    main()