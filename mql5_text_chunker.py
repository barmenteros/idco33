#!/usr/bin/env python3
"""
MQL5 Text Chunker - Task B7
Chunk extracted MQL5 text into ~500-token snippets for embedding and FAISS indexing

This module takes clean text from Task B6 and creates consistent chunks
with unique DocIDs for the RAG pipeline.
"""

import os
import re
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import tiktoken


@dataclass
class TextChunk:
    """Data structure for a text chunk with metadata"""
    doc_id: str
    text_content: str
    token_count: int
    char_start: int
    char_end: int
    chunk_index: int
    source_file: str
    section_hint: str
    metadata: Dict[str, any]


class MQL5TextChunker:
    """Chunks MQL5 text into ~500-token snippets for RAG pipeline"""
    
    def __init__(self, 
                 input_dir: str = "./mql5_test/extracted",
                 output_dir: str = "./mql5_test/chunks",
                 target_tokens: int = 500,
                 overlap_tokens: int = 50):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizer (using cl100k_base encoding - compatible with most models)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.logger.info("Initialized tiktoken tokenizer (cl100k_base)")
        except Exception as e:
            self.logger.warning(f"Failed to initialize tiktoken: {e}")
            self.logger.info("Falling back to simple word-based tokenization")
            self.tokenizer = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chunk statistics
        self.stats = {
            'total_chunks': 0,
            'total_tokens': 0,
            'avg_tokens_per_chunk': 0,
            'token_distribution': {},
            'section_distribution': {}
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback method"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate tokens as words * 1.3 (rough conversion)
            words = len(text.split())
            return int(words * 1.3)
    
    def detect_section_boundaries(self, text: str) -> List[Tuple[int, str]]:
        """Detect natural section boundaries in MQL5 documentation"""
        boundaries = []
        
        # Patterns that typically indicate section boundaries
        section_patterns = [
            # Chapter/section headers
            (r'^(\d+\.?\d*\.?\d*)\s+[A-Z][^\n]*$', 'chapter'),
            (r'^[A-Z][A-Z\s]{10,}$', 'section_header'),
            
            # Function definitions and signatures
            (r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*$', 'function_signature'),
            (r'^\s*(int|double|string|bool|void|datetime)\s+[a-zA-Z_]', 'function_def'),
            
            # Code block boundaries
            (r'^\s*//.*Example.*$', 'example_start'),
            (r'^\s*#include\s+', 'include_block'),
            (r'^\s*#property\s+', 'property_block'),
            
            # List items and bullet points
            (r'^\s*[-•]\s+', 'list_item'),
            (r'^\s*\d+\.\s+', 'numbered_list'),
            
            # Special MQL5 sections
            (r'^\s*(Note|Warning|Important):', 'special_note'),
            (r'^\s*Parameters?\s*:?\s*$', 'parameters_section'),
            (r'^\s*Returns?\s*:?\s*$', 'returns_section'),
            (r'^\s*See also\s*:?\s*$', 'see_also_section'),
        ]
        
        lines = text.split('\n')
        char_pos = 0
        
        for i, line in enumerate(lines):
            for pattern, section_type in section_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE | re.MULTILINE):
                    boundaries.append((char_pos, section_type))
                    break
            char_pos += len(line) + 1  # +1 for newline
        
        return boundaries
    
    def smart_chunk_split(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks respecting natural boundaries"""
        if self.count_tokens(text) <= max_tokens:
            return [text]
        
        # Try to split on paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed token limit
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            potential_tokens = self.count_tokens(potential_chunk)
            
            if potential_tokens <= max_tokens:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If single paragraph is too long, split it further
                if self.count_tokens(paragraph) > max_tokens:
                    # Split on sentences
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        temp_potential = temp_chunk + " " + sentence if temp_chunk else sentence
                        if self.count_tokens(temp_potential) <= max_tokens:
                            temp_chunk = temp_potential
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_overlapping_chunks(self, text: str) -> List[TextChunk]:
        """Create overlapping chunks with smart boundary detection"""
        chunks = []
        
        # First, get natural boundaries
        boundaries = self.detect_section_boundaries(text)
        
        # Initial smart chunking
        initial_chunks = self.smart_chunk_split(text, self.target_tokens - self.overlap_tokens)
        
        char_offset = 0
        for i, chunk_text in enumerate(initial_chunks):
            
            # Determine section type from boundaries
            section_hint = self._determine_section_type(char_offset, boundaries)
            
            # Create overlap with previous chunk (except first)
            if i > 0 and self.overlap_tokens > 0:
                # Get overlap from previous chunk
                prev_chunk_text = initial_chunks[i-1]
                overlap_tokens = min(self.overlap_tokens, self.count_tokens(prev_chunk_text))
                
                if self.tokenizer:
                    # Use tokenizer for precise overlap
                    prev_tokens = self.tokenizer.encode(prev_chunk_text)
                    overlap_text = self.tokenizer.decode(prev_tokens[-overlap_tokens:])
                    chunk_text = overlap_text + " " + chunk_text
                else:
                    # Fallback: word-based overlap
                    prev_words = prev_chunk_text.split()
                    overlap_words = int(overlap_tokens / 1.3)  # Convert back to words
                    if overlap_words > 0:
                        overlap_text = " ".join(prev_words[-overlap_words:])
                        chunk_text = overlap_text + " " + chunk_text
            
            # Generate unique DocID
            doc_id = self._generate_doc_id(chunk_text, i)
            
            # Calculate character positions
            char_start = char_offset
            char_end = char_offset + len(chunk_text)
            
            # Count actual tokens
            actual_tokens = self.count_tokens(chunk_text)
            
            # Create chunk object
            chunk = TextChunk(
                doc_id=doc_id,
                text_content=chunk_text.strip(),
                token_count=actual_tokens,
                char_start=char_start,
                char_end=char_end,
                chunk_index=i,
                source_file=self.current_source_file,
                section_hint=section_hint,
                metadata={
                    'target_tokens': self.target_tokens,
                    'overlap_tokens': self.overlap_tokens,
                    'original_length': len(chunk_text),
                    'processing_timestamp': str(logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None)))
                }
            )
            
            chunks.append(chunk)
            
            # Update character offset (without overlap for next chunk positioning)
            char_offset += len(initial_chunks[i])
        
        return chunks
    
    def _determine_section_type(self, char_offset: int, boundaries: List[Tuple[int, str]]) -> str:
        """Determine section type based on nearby boundaries"""
        closest_boundary = None
        min_distance = float('inf')
        
        for boundary_pos, section_type in boundaries:
            distance = abs(char_offset - boundary_pos)
            if distance < min_distance:
                min_distance = distance
                closest_boundary = section_type
        
        return closest_boundary or 'general'
    
    def _generate_doc_id(self, text: str, index: int) -> str:
        """Generate unique DocID for chunk"""
        # Create hash from content for uniqueness
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        
        # Format: mql5_prog_XXX_HASH
        return f"mql5_prog_{index:03d}_{content_hash}"
    
    def process_text_file(self, file_path: str) -> List[TextChunk]:
        """Process a single text file into chunks"""
        file_path = Path(file_path)
        self.current_source_file = file_path.name
        
        if not file_path.exists():
            self.logger.error(f"Text file not found: {file_path}")
            return []
        
        try:
            self.logger.info(f"Processing text file: {file_path.name}")
            
            # Read text content
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Log input statistics
            total_chars = len(text)
            total_tokens = self.count_tokens(text)
            estimated_chunks = max(1, total_tokens // self.target_tokens)
            
            self.logger.info(f"Input text: {total_chars:,} characters, {total_tokens:,} tokens")
            self.logger.info(f"Estimated chunks: {estimated_chunks}")
            
            # Create chunks
            chunks = self.create_overlapping_chunks(text)
            
            # Log chunk statistics
            self.logger.info(f"Created {len(chunks)} chunks")
            
            if chunks:
                token_counts = [chunk.token_count for chunk in chunks]
                avg_tokens = sum(token_counts) / len(token_counts)
                min_tokens = min(token_counts)
                max_tokens = max(token_counts)
                
                self.logger.info(f"Token distribution: avg={avg_tokens:.1f}, min={min_tokens}, max={max_tokens}")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to process text file {file_path}: {e}")
            return []
    
    def save_chunks(self, chunks: List[TextChunk], format: str = 'json') -> Dict[str, str]:
        """Save chunks in specified format"""
        if not chunks:
            self.logger.warning("No chunks to save")
            return {}
        
        outputs = {}
        
        if format == 'json':
            # Save as structured JSON
            chunks_data = [asdict(chunk) for chunk in chunks]
            json_path = self.output_dir / 'mql5_chunks.json'
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            outputs['json'] = str(json_path)
            self.logger.info(f"Chunks saved to JSON: {json_path}")
        
        if format == 'individual' or format == 'both':
            # Save individual chunk files
            chunks_dir = self.output_dir / 'individual'
            chunks_dir.mkdir(exist_ok=True)
            
            for chunk in chunks:
                chunk_file = chunks_dir / f"{chunk.doc_id}.txt"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(chunk.text_content)
            
            outputs['individual'] = str(chunks_dir)
            self.logger.info(f"Individual chunks saved to: {chunks_dir}")
        
        # Save chunks summary for DynamoDB preparation
        summary_path = self.output_dir / 'chunks_summary.json'
        summary = {
            'total_chunks': len(chunks),
            'chunks_metadata': [
                {
                    'doc_id': chunk.doc_id,
                    'token_count': chunk.token_count,
                    'section_hint': chunk.section_hint,
                    'source_file': chunk.source_file
                }
                for chunk in chunks
            ],
            'statistics': self._calculate_statistics(chunks)
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        outputs['summary'] = str(summary_path)
        self.logger.info(f"Chunks summary saved to: {summary_path}")
        
        return outputs
    
    def _calculate_statistics(self, chunks: List[TextChunk]) -> Dict[str, any]:
        """Calculate detailed statistics for chunks"""
        if not chunks:
            return {}
        
        token_counts = [chunk.token_count for chunk in chunks]
        section_types = [chunk.section_hint for chunk in chunks]
        
        # Token distribution
        token_stats = {
            'total_tokens': sum(token_counts),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'target_tokens': self.target_tokens,
            'chunks_within_target': sum(1 for t in token_counts if abs(t - self.target_tokens) <= 50)
        }
        
        # Section distribution
        section_stats = {}
        for section in set(section_types):
            section_stats[section] = section_types.count(section)
        
        return {
            'token_statistics': token_stats,
            'section_distribution': section_stats,
            'total_chunks': len(chunks),
            'quality_metrics': {
                'target_compliance_rate': token_stats['chunks_within_target'] / len(chunks),
                'avg_deviation_from_target': sum(abs(t - self.target_tokens) for t in token_counts) / len(chunks)
            }
        }
    
    def process_all_files(self) -> Dict[str, any]:
        """Process all text files in input directory"""
        if not self.input_dir.exists():
            self.logger.error(f"Input directory not found: {self.input_dir}")
            return {}
        
        text_files = list(self.input_dir.glob("*.txt"))
        if not text_files:
            self.logger.error(f"No text files found in: {self.input_dir}")
            return {}
        
        all_chunks = []
        results = {'processed_files': [], 'failed_files': [], 'total_chunks': 0}
        
        for text_file in text_files:
            self.logger.info(f"Processing: {text_file.name}")
            
            chunks = self.process_text_file(text_file)
            if chunks:
                all_chunks.extend(chunks)
                results['processed_files'].append(str(text_file))
                self.logger.info(f"Successfully processed {text_file.name}: {len(chunks)} chunks")
            else:
                results['failed_files'].append(str(text_file))
                self.logger.warning(f"Failed to process {text_file.name}")
        
        results['total_chunks'] = len(all_chunks)
        
        if all_chunks:
            # Save all chunks
            outputs = self.save_chunks(all_chunks, format='json')
            results['outputs'] = outputs
            
            # Final statistics
            stats = self._calculate_statistics(all_chunks)
            results['statistics'] = stats
            
            self.logger.info(f"=== Processing Complete ===")
            self.logger.info(f"Total files processed: {len(results['processed_files'])}")
            self.logger.info(f"Total chunks created: {results['total_chunks']}")
            self.logger.info(f"Average tokens per chunk: {stats['token_statistics']['avg_tokens']:.1f}")
            self.logger.info(f"Target compliance rate: {stats['quality_metrics']['target_compliance_rate']:.1%}")
        
        return results


def main():
    """Main function for text chunking"""
    
    print("=== MQL5 Text Chunker (Task B7) ===")
    
    # Initialize chunker
    chunker = MQL5TextChunker(
        input_dir="./mql5_test/extracted",
        output_dir="./mql5_test/chunks",
        target_tokens=500,
        overlap_tokens=50
    )
    
    # Process all text files
    results = chunker.process_all_files()
    
    # Display results
    print("\n=== Chunking Results ===")
    if results.get('total_chunks', 0) > 0:
        print(f"✅ Successfully created {results['total_chunks']} chunks")
        print(f"📁 Processed files: {len(results.get('processed_files', []))}")
        
        if 'statistics' in results:
            stats = results['statistics']
            print(f"📊 Average tokens/chunk: {stats['token_statistics']['avg_tokens']:.1f}")
            print(f"🎯 Target compliance: {stats['quality_metrics']['target_compliance_rate']:.1%}")
            print(f"📈 Token range: {stats['token_statistics']['min_tokens']}-{stats['token_statistics']['max_tokens']}")
        
        if 'outputs' in results:
            print(f"💾 Chunks saved to: {results['outputs'].get('json', 'N/A')}")
    else:
        print("❌ No chunks created. Check input files and logs.")
    
    return results


if __name__ == "__main__":
    main()