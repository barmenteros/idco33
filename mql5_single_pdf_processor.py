#!/usr/bin/env python3
"""
MQL5 Single PDF Processor - Test Version
Task B6: Process single PDF file for testing and validation

This simplified version processes just 'mql5 programming for traders.pdf'
to validate the text extraction pipeline before scaling up.
"""

import os
import re
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict

from pdfminer.high_level import extract_text as extract_pdf_text
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


class MQL5SinglePDFProcessor:
    """Simplified processor for testing with a single MQL5 PDF"""
    
    def __init__(self, output_dir: str = "./mql5_test"):
        self.output_dir = Path(output_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.raw_dir = self.output_dir / "raw"
        self.extracted_dir = self.output_dir / "extracted"
        self.analysis_dir = self.output_dir / "analysis"
        
        for directory in [self.raw_dir, self.extracted_dir, self.analysis_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized processor with output directory: {self.output_dir}")
    
    def copy_pdf_for_processing(self, pdf_path: str) -> Optional[str]:
        """Copy the target PDF to our processing directory"""
        source_path = Path(pdf_path)
        
        if not source_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        if not source_path.suffix.lower() == '.pdf':
            self.logger.error(f"File is not a PDF: {pdf_path}")
            return None
        
        try:
            # Generate hash for the file
            with open(source_path, 'rb') as f:
                content = f.read()
            file_hash = hashlib.sha256(content).hexdigest()[:12]
            
            # Create destination filename
            dest_filename = f"programming_guide_{source_path.stem}_{file_hash}.pdf"
            dest_path = self.raw_dir / dest_filename
            
            # Copy file
            with open(dest_path, 'wb') as f:
                f.write(content)
            
            self.logger.info(f"Copied PDF: {source_path.name} -> {dest_path.name}")
            self.logger.info(f"File size: {len(content):,} bytes")
            
            return str(dest_path)
            
        except Exception as e:
            self.logger.error(f"Failed to copy PDF: {e}")
            return None
    
    def extract_text_detailed(self, pdf_path: str) -> Optional[str]:
        """Extract text with detailed analysis and validation"""
        try:
            self.logger.info(f"Starting text extraction from: {Path(pdf_path).name}")
            
            # Method 1: Simple extraction
            self.logger.info("Using pdfminer high-level extraction...")
            text = extract_pdf_text(pdf_path)
            
            if not text or len(text.strip()) < 100:
                self.logger.warning("High-level extraction failed or insufficient content")
                return None
            
            # Log extraction statistics
            char_count = len(text)
            word_count = len(text.split())
            line_count = len(text.splitlines())
            
            self.logger.info(f"Extraction statistics:")
            self.logger.info(f"  - Characters: {char_count:,}")
            self.logger.info(f"  - Words: {word_count:,}")
            self.logger.info(f"  - Lines: {line_count:,}")
            
            # Save raw extracted text for analysis
            raw_text_path = self.analysis_dir / "raw_extracted_text.txt"
            with open(raw_text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            self.logger.info(f"Raw text saved to: {raw_text_path}")
            
            return text
            
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return None
    
    def clean_text_detailed(self, text: str) -> str:
        """Clean text with detailed logging of each step"""
        if not text:
            return ""
        
        self.logger.info("Starting text cleaning process...")
        
        # Step 1: Initial statistics
        original_length = len(text)
        self.logger.info(f"Original text length: {original_length:,} characters")
        
        # Step 2: Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        after_whitespace = len(text)
        self.logger.info(f"After whitespace cleanup: {after_whitespace:,} characters")
        
        # Step 3: Remove common PDF artifacts
        pdf_artifacts = [
            r'^\s*\d+\s*$',  # Page numbers on separate lines
            r'^\s*Page \d+\s*$',  # "Page X" headers
            r'^\s*Chapter \d+\s*$',  # "Chapter X" headers
            r'^\s*Table of Contents\s*$',  # TOC headers
            r'^\s*Index\s*$',  # Index headers
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        removed_artifacts = 0
        
        for line in lines:
            is_artifact = False
            for pattern in pdf_artifacts:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    is_artifact = True
                    removed_artifacts += 1
                    break
            
            if not is_artifact and line.strip():
                cleaned_lines.append(line.strip())
        
        text = '\n'.join(cleaned_lines)
        self.logger.info(f"Removed {removed_artifacts} PDF artifacts")
        
        # Step 4: Clean up special characters (preserve code examples)
        # Keep programming-related characters
        text = re.sub(r'[^\w\s\(\)\[\]{};.,!?:=+\-*/\'"<>&|#@$%\n]', ' ', text)
        after_special_chars = len(text)
        self.logger.info(f"After special character cleanup: {after_special_chars:,} characters")
        
        # Step 5: Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        final_length = len(text)
        self.logger.info(f"Final cleaned text length: {final_length:,} characters")
        self.logger.info(f"Reduction: {((original_length - final_length) / original_length * 100):.1f}%")
        
        return text
    
    def analyze_content_quality(self, text: str) -> Dict[str, any]:
        """Analyze the quality and characteristics of extracted content"""
        analysis = {}
        
        # Basic statistics
        analysis['char_count'] = len(text)
        analysis['word_count'] = len(text.split())
        analysis['line_count'] = len(text.splitlines())
        
        # Look for MQL5-specific content
        mql5_keywords = [
            'MQL5', 'Expert Advisor', 'MetaTrader', 'OnTick', 'OnInit', 'OnDeinit',
            'ArrayResize', 'Print', 'Comment', 'Alert', 'OrderSend', 'PositionOpen',
            'double', 'int', 'string', 'datetime', 'bool', '#include', '#property'
        ]
        
        found_keywords = []
        for keyword in mql5_keywords:
            if keyword.lower() in text.lower():
                found_keywords.append(keyword)
        
        analysis['mql5_keywords_found'] = found_keywords
        analysis['mql5_keyword_count'] = len(found_keywords)
        
        # Look for code examples
        code_patterns = [
            r'int\s+\w+\s*\(',  # Function definitions
            r'void\s+On\w+\s*\(',  # Event handlers
            r'#\w+',  # Preprocessor directives
            r'\w+\s*\[\s*\]',  # Arrays
            r'if\s*\(',  # Conditional statements
            r'for\s*\(',  # Loops
        ]
        
        code_examples = 0
        for pattern in code_patterns:
            code_examples += len(re.findall(pattern, text, re.IGNORECASE))
        
        analysis['code_examples_found'] = code_examples
        
        # Estimate content density
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            analysis['avg_word_length'] = round(avg_word_length, 2)
        else:
            analysis['avg_word_length'] = 0
        
        return analysis
    
    def save_analysis_report(self, analysis: Dict, original_pdf: str):
        """Save detailed analysis report"""
        report_path = self.analysis_dir / "content_analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== MQL5 PDF Content Analysis Report ===\n\n")
            f.write(f"Source PDF: {Path(original_pdf).name}\n")
            f.write(f"Processing Date: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}\n\n")
            
            f.write("Content Statistics:\n")
            f.write(f"  - Total Characters: {analysis['char_count']:,}\n")
            f.write(f"  - Total Words: {analysis['word_count']:,}\n")
            f.write(f"  - Total Lines: {analysis['line_count']:,}\n")
            f.write(f"  - Average Word Length: {analysis['avg_word_length']}\n\n")
            
            f.write("MQL5-Specific Content:\n")
            f.write(f"  - MQL5 Keywords Found: {analysis['mql5_keyword_count']}\n")
            f.write(f"  - Keywords: {', '.join(analysis['mql5_keywords_found'])}\n")
            f.write(f"  - Code Examples: {analysis['code_examples_found']}\n\n")
            
            # Quality assessment
            quality_score = min(100, (
                (analysis['mql5_keyword_count'] * 5) +
                (analysis['code_examples_found'] * 2) +
                (min(analysis['word_count'] / 1000, 50))
            ))
            
            f.write(f"Quality Assessment:\n")
            f.write(f"  - Content Quality Score: {quality_score:.1f}/100\n")
            
            if quality_score >= 70:
                f.write("  - Assessment: EXCELLENT - High-quality MQL5 content suitable for RAG\n")
            elif quality_score >= 50:
                f.write("  - Assessment: GOOD - Suitable MQL5 content with minor gaps\n")
            elif quality_score >= 30:
                f.write("  - Assessment: FAIR - Some MQL5 content, may need supplementation\n")
            else:
                f.write("  - Assessment: POOR - Limited MQL5 content, consider additional sources\n")
        
        self.logger.info(f"Analysis report saved to: {report_path}")
        return quality_score
    
    def process_single_pdf(self, pdf_path: str) -> Dict[str, any]:
        """Complete processing pipeline for a single PDF"""
        results = {
            'success': False,
            'copied_pdf': None,
            'extracted_text_file': None,
            'analysis': None,
            'quality_score': 0
        }
        
        try:
            # Step 1: Copy PDF to processing directory
            self.logger.info("=== Step 1: Copying PDF ===")
            copied_pdf = self.copy_pdf_for_processing(pdf_path)
            if not copied_pdf:
                return results
            results['copied_pdf'] = copied_pdf
            
            # Step 2: Extract text
            self.logger.info("=== Step 2: Extracting Text ===")
            raw_text = self.extract_text_detailed(copied_pdf)
            if not raw_text:
                return results
            
            # Step 3: Clean text
            self.logger.info("=== Step 3: Cleaning Text ===")
            cleaned_text = self.clean_text_detailed(raw_text)
            
            # Step 4: Save cleaned text
            output_filename = f"{Path(copied_pdf).stem}_cleaned.txt"
            output_path = self.extracted_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            self.logger.info(f"Cleaned text saved to: {output_path}")
            results['extracted_text_file'] = str(output_path)
            
            # Step 5: Analyze content quality
            self.logger.info("=== Step 4: Analyzing Content Quality ===")
            analysis = self.analyze_content_quality(cleaned_text)
            results['analysis'] = analysis
            
            # Step 6: Generate report
            quality_score = self.save_analysis_report(analysis, pdf_path)
            results['quality_score'] = quality_score
            
            results['success'] = True
            self.logger.info("=== Processing Complete ===")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return results


def main():
    """Main function for single PDF processing"""
    
    # Configuration
    PDF_PATH = "C:/Users/barme/OneDrive/2.PROFESSIONAL/MetaTrader/MetaTrader Documentation/mql5/mql5 programming for traders.pdf"  # Update this path as needed
    
    print("=== MQL5 Single PDF Processor (Test Version) ===")
    print(f"Target PDF: {PDF_PATH}")
    
    # Initialize processor
    processor = MQL5SinglePDFProcessor(output_dir="./mql5_test")
    
    # Process the PDF
    results = processor.process_single_pdf(PDF_PATH)
    
    # Display results
    print("\n=== Processing Results ===")
    if results['success']:
        print("✅ Processing completed successfully!")
        print(f"📁 Copied PDF: {Path(results['copied_pdf']).name}")
        print(f"📄 Extracted text: {Path(results['extracted_text_file']).name}")
        print(f"📊 Quality Score: {results['quality_score']:.1f}/100")
        print(f"📈 Word Count: {results['analysis']['word_count']:,}")
        print(f"🔧 MQL5 Keywords: {results['analysis']['mql5_keyword_count']}")
        print(f"💻 Code Examples: {results['analysis']['code_examples_found']}")
        print(f"\n📋 Check the analysis report in: ./mql5_test/analysis/")
    else:
        print("❌ Processing failed. Check the logs for details.")
    
    return results


if __name__ == "__main__":
    main()