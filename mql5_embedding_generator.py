#!/usr/bin/env python3
"""
MQL5 Embedding Generator - Task B8
Generate embeddings offline for MQL5 text chunks using SentenceTransformers

This module takes chunks from Task B7 and creates vector embeddings
for FAISS index construction and similarity search.
"""

import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
from datetime import datetime

from sentence_transformers import SentenceTransformer
import torch


@dataclass
class EmbeddingResult:
    """Structure for embedding computation results"""
    doc_id: str
    embedding: np.ndarray
    text_preview: str
    vector_index: int
    metadata: Dict[str, any]


class MQL5EmbeddingGenerator:
    """Generate embeddings for MQL5 text chunks using SentenceTransformers"""
    
    def __init__(self, 
                 chunks_file: str = "./mql5_test/chunks/mql5_chunks.json",
                 output_dir: str = "./mql5_test/embeddings",
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 32):
        
        self.chunks_file = Path(chunks_file)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model (will be loaded in load_model method)
        self.model = None
        self.model_info = {}
        
        # Processing statistics
        self.stats = {
            'total_chunks': 0,
            'processed_chunks': 0,
            'failed_chunks': 0,
            'embedding_dimension': 0,
            'processing_time': 0,
            'embeddings_per_second': 0
        }
    
    def load_model(self) -> bool:
        """Load SentenceTransformer model with validation"""
        try:
            self.logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            
            # Check if CUDA is available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"Using device: {device}")
            
            # Load model
            self.model = SentenceTransformer(self.model_name, device=device)
            
            # Get model information
            self.model_info = {
                'model_name': self.model_name,
                'max_seq_length': self.model.max_seq_length,
                'embedding_dimension': self.model.get_sentence_embedding_dimension(),
                'device': device,
                'pytorch_version': torch.__version__,
                'loaded_at': datetime.now().isoformat()
            }
            
            self.stats['embedding_dimension'] = self.model_info['embedding_dimension']
            
            self.logger.info(f"Model loaded successfully:")
            self.logger.info(f"  - Embedding dimension: {self.model_info['embedding_dimension']}")
            self.logger.info(f"  - Max sequence length: {self.model_info['max_seq_length']}")
            self.logger.info(f"  - Device: {device}")
            
            # Test embedding generation
            test_text = "This is a test sentence for MQL5 validation."
            test_embedding = self.model.encode([test_text])
            self.logger.info(f"  - Test embedding shape: {test_embedding.shape}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def load_chunks(self) -> List[Dict]:
        """Load text chunks from Task B7 output"""
        if not self.chunks_file.exists():
            self.logger.error(f"Chunks file not found: {self.chunks_file}")
            return []
        
        try:
            self.logger.info(f"Loading chunks from: {self.chunks_file}")
            
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            self.stats['total_chunks'] = len(chunks)
            
            self.logger.info(f"Loaded {len(chunks)} chunks")
            
            # Log sample chunk info
            if chunks:
                sample = chunks[0]
                self.logger.info(f"Sample chunk:")
                self.logger.info(f"  - DocID: {sample.get('doc_id', 'N/A')}")
                self.logger.info(f"  - Token count: {sample.get('token_count', 'N/A')}")
                self.logger.info(f"  - Text preview: {sample.get('text_content', '')[:100]}...")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to load chunks: {e}")
            return []
    
    def validate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Validate and filter chunks for embedding generation"""
        valid_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Check required fields
            if not chunk.get('doc_id'):
                self.logger.warning(f"Chunk {i} missing doc_id, skipping")
                continue
                
            if not chunk.get('text_content'):
                self.logger.warning(f"Chunk {chunk.get('doc_id')} missing text_content, skipping")
                continue
            
            # Check text length
            text = chunk['text_content'].strip()
            if len(text) < 10:
                self.logger.warning(f"Chunk {chunk.get('doc_id')} too short ({len(text)} chars), skipping")
                continue
            
            # Check for reasonable token count
            token_count = chunk.get('token_count', 0)
            if token_count > self.model_info.get('max_seq_length', 512):
                self.logger.warning(f"Chunk {chunk.get('doc_id')} exceeds max length ({token_count} tokens), truncating")
                # Truncate text roughly
                words = text.split()
                max_words = int(self.model_info.get('max_seq_length', 512) * 0.75)  # Conservative estimate
                if len(words) > max_words:
                    text = ' '.join(words[:max_words])
                    chunk['text_content'] = text
            
            valid_chunks.append(chunk)
        
        self.logger.info(f"Validated chunks: {len(valid_chunks)} valid out of {len(chunks)} total")
        return valid_chunks
    
    def generate_embeddings_batch(self, chunks_batch: List[Dict]) -> List[EmbeddingResult]:
        """Generate embeddings for a batch of chunks"""
        try:
            # Extract texts for batch processing
            texts = [chunk['text_content'] for chunk in chunks_batch]
            doc_ids = [chunk['doc_id'] for chunk in chunks_batch]
            
            # Generate embeddings in batch
            start_time = time.time()
            embeddings = self.model.encode(
                texts,
                batch_size=min(len(texts), self.batch_size),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity search
            )
            processing_time = time.time() - start_time
            
            # Create results
            results = []
            for i, (chunk, doc_id, embedding) in enumerate(zip(chunks_batch, doc_ids, embeddings)):
                
                # Create text preview (first 100 chars)
                text_preview = chunk['text_content'][:100] + "..." if len(chunk['text_content']) > 100 else chunk['text_content']
                
                result = EmbeddingResult(
                    doc_id=doc_id,
                    embedding=embedding,
                    text_preview=text_preview,
                    vector_index=len(results),  # Will be updated globally later
                    metadata={
                        'token_count': chunk.get('token_count', 0),
                        'section_hint': chunk.get('section_hint', 'unknown'),
                        'source_file': chunk.get('source_file', 'unknown'),
                        'chunk_index': chunk.get('chunk_index', i),
                        'embedding_norm': float(np.linalg.norm(embedding)),
                        'processing_time': processing_time / len(chunks_batch)
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings for batch: {e}")
            return []
    
    def process_all_chunks(self, chunks: List[Dict]) -> List[EmbeddingResult]:
        """Process all chunks with batch processing and progress tracking"""
        if not chunks:
            return []
        
        all_results = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"Processing {len(chunks)} chunks in {total_batches} batches of {self.batch_size}")
        
        start_time = time.time()
        
        for batch_idx in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_idx + self.batch_size, len(chunks))
            batch = chunks[batch_idx:batch_end]
            batch_num = (batch_idx // self.batch_size) + 1
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            # Generate embeddings for batch
            batch_results = self.generate_embeddings_batch(batch)
            
            if batch_results:
                # Update global vector indices
                for result in batch_results:
                    result.vector_index = len(all_results)
                    all_results.append(result)
                
                self.stats['processed_chunks'] += len(batch_results)
                
                # Log progress
                progress = (batch_num / total_batches) * 100
                elapsed = time.time() - start_time
                avg_time_per_batch = elapsed / batch_num
                eta = avg_time_per_batch * (total_batches - batch_num)
                
                self.logger.info(f"Progress: {progress:.1f}% | ETA: {eta:.1f}s | Processed: {len(all_results)}")
            else:
                self.stats['failed_chunks'] += len(batch)
                self.logger.warning(f"Failed to process batch {batch_num}")
        
        total_time = time.time() - start_time
        self.stats['processing_time'] = total_time
        self.stats['embeddings_per_second'] = len(all_results) / total_time if total_time > 0 else 0
        
        self.logger.info(f"Embedding generation complete:")
        self.logger.info(f"  - Total processed: {len(all_results)}")
        self.logger.info(f"  - Processing time: {total_time:.2f}s")
        self.logger.info(f"  - Speed: {self.stats['embeddings_per_second']:.2f} embeddings/sec")
        
        return all_results
    
    def save_embeddings(self, results: List[EmbeddingResult]) -> Dict[str, str]:
        """Save embeddings and metadata in multiple formats"""
        if not results:
            self.logger.warning("No embeddings to save")
            return {}
        
        outputs = {}
        
        # Extract embeddings matrix and docid mapping
        embeddings_matrix = np.array([result.embedding for result in results])
        docid_to_index = {result.doc_id: result.vector_index for result in results}
        index_to_docid = {result.vector_index: result.doc_id for result in results}
        
        # 1. Save embeddings matrix (for FAISS)
        embeddings_path = self.output_dir / 'embeddings_matrix.npy'
        np.save(embeddings_path, embeddings_matrix)
        outputs['embeddings_matrix'] = str(embeddings_path)
        self.logger.info(f"Embeddings matrix saved: {embeddings_path}")
        
        # 2. Save DocID mappings (for DynamoDB/FAISS coordination)
        docid_mapping_path = self.output_dir / 'docid_mappings.json'
        mappings = {
            'docid_to_index': docid_to_index,
            'index_to_docid': index_to_docid,
            'total_embeddings': len(results)
        }
        with open(docid_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=2)
        outputs['docid_mappings'] = str(docid_mapping_path)
        self.logger.info(f"DocID mappings saved: {docid_mapping_path}")
        
        # 3. Save complete results with metadata (for analysis)
        results_path = self.output_dir / 'embedding_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        outputs['complete_results'] = str(results_path)
        self.logger.info(f"Complete results saved: {results_path}")
        
        # 4. Save model information and statistics
        metadata_path = self.output_dir / 'embedding_metadata.json'
        metadata = {
            'model_info': self.model_info,
            'processing_stats': self.stats,
            'embedding_shape': embeddings_matrix.shape,
            'generated_at': datetime.now().isoformat(),
            'sample_embeddings': {
                'first_docid': results[0].doc_id,
                'first_embedding_preview': results[0].embedding[:5].tolist(),
                'embedding_norm_range': {
                    'min': float(np.min([np.linalg.norm(r.embedding) for r in results])),
                    'max': float(np.max([np.linalg.norm(r.embedding) for r in results])),
                    'mean': float(np.mean([np.linalg.norm(r.embedding) for r in results]))
                }
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        outputs['metadata'] = str(metadata_path)
        self.logger.info(f"Metadata saved: {metadata_path}")
        
        # 5. Save text previews for validation
        previews_path = self.output_dir / 'text_previews.json'
        previews = {
            result.doc_id: {
                'text_preview': result.text_preview,
                'vector_index': result.vector_index,
                'section_hint': result.metadata.get('section_hint', 'unknown')
            }
            for result in results[:100]  # Save first 100 for validation
        }
        
        with open(previews_path, 'w', encoding='utf-8') as f:
            json.dump(previews, f, indent=2, ensure_ascii=False)
        outputs['text_previews'] = str(previews_path)
        self.logger.info(f"Text previews saved: {previews_path}")
        
        return outputs
    
    def validate_embeddings(self, results: List[EmbeddingResult]) -> Dict[str, any]:
        """Validate embedding quality with similarity tests"""
        if len(results) < 2:
            return {'validation': 'insufficient_data'}
        
        self.logger.info("Running embedding quality validation...")
        
        validation_results = {}
        
        # Test 1: Embedding consistency
        embeddings = np.array([result.embedding for result in results])
        
        # Check dimensions
        expected_dim = self.model_info['embedding_dimension']
        actual_dim = embeddings.shape[1]
        validation_results['dimension_check'] = {
            'expected': expected_dim,
            'actual': actual_dim,
            'passed': expected_dim == actual_dim
        }
        
        # Test 2: Embedding norms (should be ~1 if normalized)
        norms = np.linalg.norm(embeddings, axis=1)
        validation_results['normalization_check'] = {
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'is_normalized': float(np.mean(norms)) > 0.95  # Should be close to 1
        }
        
        # Test 3: Sample similarity test
        # Find chunks with similar section hints
        section_groups = {}
        for result in results[:100]:  # Test with first 100
            section = result.metadata.get('section_hint', 'unknown')
            if section not in section_groups:
                section_groups[section] = []
            section_groups[section].append(result)
        
        similarity_tests = []
        for section, section_results in section_groups.items():
            if len(section_results) >= 2:
                # Calculate similarity within section
                emb1 = section_results[0].embedding
                emb2 = section_results[1].embedding
                similarity = np.dot(emb1, emb2)  # Cosine similarity (normalized vectors)
                
                similarity_tests.append({
                    'section': section,
                    'similarity': float(similarity),
                    'docid1': section_results[0].doc_id,
                    'docid2': section_results[1].doc_id
                })
        
        if similarity_tests:
            avg_similarity = np.mean([test['similarity'] for test in similarity_tests])
            validation_results['similarity_tests'] = {
                'average_within_section_similarity': float(avg_similarity),
                'tests_performed': len(similarity_tests),
                'sample_tests': similarity_tests[:3]  # First 3 tests
            }
        
        # Test 4: Diversity check (embeddings shouldn't be too similar)
        sample_embeddings = embeddings[:min(50, len(embeddings))]
        similarity_matrix = np.dot(sample_embeddings, sample_embeddings.T)
        off_diagonal = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        validation_results['diversity_check'] = {
            'mean_pairwise_similarity': float(np.mean(off_diagonal)),
            'std_pairwise_similarity': float(np.std(off_diagonal)),
            'too_similar_pairs': int(np.sum(off_diagonal > 0.95))  # Very high similarity
        }
        
        # Overall quality score
        quality_score = 0
        if validation_results['dimension_check']['passed']:
            quality_score += 25
        if validation_results['normalization_check']['is_normalized']:
            quality_score += 25
        if validation_results.get('similarity_tests', {}).get('average_within_section_similarity', 0) > 0.3:
            quality_score += 25
        if validation_results['diversity_check']['mean_pairwise_similarity'] < 0.8:
            quality_score += 25
        
        validation_results['overall_quality_score'] = quality_score
        validation_results['quality_assessment'] = (
            'EXCELLENT' if quality_score >= 90 else
            'GOOD' if quality_score >= 70 else
            'FAIR' if quality_score >= 50 else
            'POOR'
        )
        
        self.logger.info(f"Validation complete - Quality: {validation_results['quality_assessment']} ({quality_score}/100)")
        
        return validation_results
    
    def run_complete_pipeline(self) -> Dict[str, any]:
        """Run the complete embedding generation pipeline"""
        pipeline_results = {
            'success': False,
            'model_loaded': False,
            'chunks_loaded': 0,
            'embeddings_generated': 0,
            'outputs': {},
            'validation': {},
            'stats': {}
        }
        
        try:
            # Step 1: Load model
            self.logger.info("=== Step 1: Loading SentenceTransformer Model ===")
            if not self.load_model():
                return pipeline_results
            pipeline_results['model_loaded'] = True
            
            # Step 2: Load chunks
            self.logger.info("=== Step 2: Loading Text Chunks ===")
            chunks = self.load_chunks()
            if not chunks:
                return pipeline_results
            pipeline_results['chunks_loaded'] = len(chunks)
            
            # Step 3: Validate chunks
            self.logger.info("=== Step 3: Validating Chunks ===")
            valid_chunks = self.validate_chunks(chunks)
            if not valid_chunks:
                self.logger.error("No valid chunks found")
                return pipeline_results
            
            # Step 4: Generate embeddings
            self.logger.info("=== Step 4: Generating Embeddings ===")
            results = self.process_all_chunks(valid_chunks)
            if not results:
                return pipeline_results
            pipeline_results['embeddings_generated'] = len(results)
            
            # Step 5: Save embeddings
            self.logger.info("=== Step 5: Saving Embeddings ===")
            outputs = self.save_embeddings(results)
            pipeline_results['outputs'] = outputs
            
            # Step 6: Validate embeddings
            self.logger.info("=== Step 6: Validating Embeddings ===")
            validation = self.validate_embeddings(results)
            pipeline_results['validation'] = validation
            
            # Final statistics
            pipeline_results['stats'] = self.stats
            pipeline_results['success'] = True
            
            self.logger.info("=== Pipeline Complete ===")
            self.logger.info(f"Successfully generated {len(results)} embeddings")
            self.logger.info(f"Quality assessment: {validation.get('quality_assessment', 'Unknown')}")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return pipeline_results


def main():
    """Main function for embedding generation"""
    
    print("=== MQL5 Embedding Generator (Task B8) ===")
    
    # Initialize generator
    generator = MQL5EmbeddingGenerator(
        chunks_file="./mql5_test/chunks/mql5_chunks.json",
        output_dir="./mql5_test/embeddings",
        model_name="all-MiniLM-L6-v2",  # Lightweight, fast, good quality
        batch_size=32
    )
    
    # Run complete pipeline
    results = generator.run_complete_pipeline()
    
    # Display results
    print("\n=== Embedding Generation Results ===")
    if results['success']:
        print(f"✅ Successfully generated {results['embeddings_generated']} embeddings")
        print(f"📊 Model: {generator.model_name}")
        print(f"📏 Embedding dimension: {generator.stats.get('embedding_dimension', 'N/A')}")
        print(f"⚡ Processing speed: {generator.stats.get('embeddings_per_second', 0):.1f} embeddings/sec")
        print(f"🎯 Quality: {results['validation'].get('quality_assessment', 'Unknown')}")
        
        if 'outputs' in results:
            print(f"💾 Embeddings saved to: {results['outputs'].get('embeddings_matrix', 'N/A')}")
            print(f"🗂️ DocID mappings: {results['outputs'].get('docid_mappings', 'N/A')}")
    else:
        print("❌ Embedding generation failed. Check logs for details.")
    
    return results


if __name__ == "__main__":
    main()