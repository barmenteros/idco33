#!/usr/bin/env python3
"""
MQL5 Documentation Downloader & Extractor
Task B6: Download and extract MQL5 documentation for RAG pipeline

This module downloads official MQL5 documentation and extracts clean text
for subsequent chunking and embedding operations.
"""

import os
import re
import time
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse, unquote
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as extract_pdf_text
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


@dataclass
class DocumentSource:
    """Configuration for MQL5 documentation sources"""
    url: str
    doc_type: str  # 'html' or 'pdf'
    section: str   # e.g., 'functions', 'constants', 'basics'
    priority: int  # 1=high, 2=medium, 3=low


class MQL5DocDownloader:
    """Downloads MQL5 documentation from official sources"""
    
    def __init__(self, output_dir: str = "./mql5_docs", max_retries: int = 3):
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.raw_dir = self.output_dir / "raw"
        self.extracted_dir = self.output_dir / "extracted"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        
        # MQL5 documentation sources
        self.sources = self._get_documentation_sources()
    
    def _get_documentation_sources(self) -> List[DocumentSource]:
        """Define MQL5 documentation sources to download"""
        return [
            # Core MQL5 Reference
            DocumentSource(
                url="https://www.mql5.com/en/docs",
                doc_type="html",
                section="main_reference",
                priority=1
            ),
            # Functions documentation
            DocumentSource(
                url="https://www.mql5.com/en/docs/basis",
                doc_type="html", 
                section="basics",
                priority=1
            ),
            DocumentSource(
                url="https://www.mql5.com/en/docs/constants",
                doc_type="html",
                section="constants", 
                priority=1
            ),
            DocumentSource(
                url="https://www.mql5.com/en/docs/array",
                doc_type="html",
                section="arrays",
                priority=1
            ),
            DocumentSource(
                url="https://www.mql5.com/en/docs/common",
                doc_type="html",
                section="common_functions",
                priority=1
            ),
            DocumentSource(
                url="https://www.mql5.com/en/docs/dateandtime",
                doc_type="html",
                section="datetime_functions",
                priority=2
            ),
            DocumentSource(
                url="https://www.mql5.com/en/docs/math",
                doc_type="html",
                section="math_functions", 
                priority=2
            ),
            DocumentSource(
                url="https://www.mql5.com/en/docs/strings",
                doc_type="html",
                section="string_functions",
                priority=2
            ),
            DocumentSource(
                url="https://www.mql5.com/en/docs/convert",
                doc_type="html",
                section="conversion_functions",
                priority=2
            ),
            DocumentSource(
                url="https://www.mql5.com/en/docs/marketinformation",
                doc_type="html", 
                section="market_info",
                priority=1
            ),
            DocumentSource(
                url="https://www.mql5.com/en/docs/series",
                doc_type="html",
                section="timeseries",
                priority=1
            ),
            DocumentSource(
                url="https://www.mql5.com/en/docs/indicators",
                doc_type="html",
                section="indicators",
                priority=1
            ),
            DocumentSource(
                url="https://www.mql5.com/en/docs/trade",
                doc_type="html",
                section="trading_functions",
                priority=1
            )
        ]
    
    def _generate_file_hash(self, content: bytes) -> str:
        """Generate SHA256 hash for content deduplication"""
        return hashlib.sha256(content).hexdigest()[:12]
    
    def _sanitize_filename(self, url: str, section: str) -> str:
        """Create safe filename from URL and section"""
        parsed = urlparse(url)
        path_parts = [part for part in parsed.path.split('/') if part]
        if path_parts:
            name = path_parts[-1]
        else:
            name = parsed.netloc.replace('.', '_')
        
        # Clean and combine with section
        clean_name = re.sub(r'[^\w\-_.]', '_', name)
        return f"{section}_{clean_name}"
    
    def download_document(self, source: DocumentSource) -> Optional[Tuple[str, bytes]]:
        """Download a single document with retries"""
        filename = self._sanitize_filename(source.url, source.section)
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Downloading {source.url} (attempt {attempt + 1})")
                
                response = self.session.get(source.url, timeout=30)
                response.raise_for_status()
                
                content = response.content
                file_hash = self._generate_file_hash(content)
                
                # Save raw content
                if source.doc_type == 'pdf':
                    file_path = self.raw_dir / f"{filename}_{file_hash}.pdf"
                else:
                    file_path = self.raw_dir / f"{filename}_{file_hash}.html"
                
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                self.logger.info(f"Downloaded: {file_path}")
                return str(file_path), content
                
            except Exception as e:
                self.logger.warning(f"Download failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        self.logger.error(f"Failed to download {source.url} after {self.max_retries} attempts")
        return None
    
    def discover_additional_pages(self, base_url: str, content: bytes) -> List[str]:
        """Discover additional documentation pages from navigation links"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            links = []
            
            # Look for navigation menus, documentation links
            nav_selectors = [
                'nav a[href*="/docs/"]',
                '.navigation a[href*="/docs/"]', 
                '.menu a[href*="/docs/"]',
                'a[href*="/docs/"][title]',
                '.doc-nav a',
                '.sidebar a[href*="/docs/"]'
            ]
            
            for selector in nav_selectors:
                elements = soup.select(selector)
                for element in elements:
                    href = element.get('href')
                    if href and '/docs/' in href:
                        full_url = urljoin(base_url, href)
                        if full_url not in links and full_url != base_url:
                            links.append(full_url)
            
            # Deduplicate and limit
            unique_links = list(set(links))[:20]  # Limit to prevent explosion
            self.logger.info(f"Discovered {len(unique_links)} additional pages from {base_url}")
            return unique_links
            
        except Exception as e:
            self.logger.warning(f"Failed to discover additional pages: {e}")
            return []
    
def process_existing_pdfs(self, pdf_directory: str) -> Dict[str, List[str]]:
    """Process existing PDF files from a specified directory"""
    pdf_dir = Path(pdf_directory)
    results = {'success': [], 'failed': []}
    
    if not pdf_dir.exists():
        self.logger.error(f"PDF directory not found: {pdf_dir}")
        return results
    
    # Copy PDFs to raw directory for processing
    for pdf_file in pdf_dir.glob("*.pdf"):
        try:
            # Create descriptive filename based on original
            section = self._identify_section_from_filename(pdf_file.name)
            file_hash = self._generate_file_hash(pdf_file.read_bytes())
            new_filename = f"{section}_{pdf_file.stem}_{file_hash}.pdf"
            
            dest_path = self.raw_dir / new_filename
            
            # Copy file to processing directory
            import shutil
            shutil.copy2(pdf_file, dest_path)
            
            self.logger.info(f"Added existing PDF: {pdf_file.name} -> {dest_path.name}")
            results['success'].append(str(dest_path))
            
        except Exception as e:
            self.logger.error(f"Failed to process existing PDF {pdf_file}: {e}")
            results['failed'].append(str(pdf_file))
    
    return results

def _identify_section_from_filename(self, filename: str) -> str:
    """Identify document section from filename"""
    filename_lower = filename.lower()
    
    section_mapping = {
        'reference': 'reference',
        'programming': 'programming_guide', 
        'terminal': 'terminal_guide',
        'basis': 'basics',
        'function': 'functions',
        'constant': 'constants',
        'array': 'arrays',
        'trade': 'trading',
        'indicator': 'indicators',
        'math': 'math_functions',
        'string': 'string_functions',
        'datetime': 'datetime_functions'
    }
    
    for keyword, section in section_mapping.items():
        if keyword in filename_lower:
            return section
    
    return 'general'
        """Download all configured documentation sources"""
        results = {'success': [], 'failed': []}
        discovered_urls = set()
        
        # Download primary sources
        for source in sorted(self.sources, key=lambda x: x.priority):
            result = self.download_document(source)
            if result:
                file_path, content = result
                results['success'].append(file_path)
                
                # Discover additional pages for HTML sources
                if source.doc_type == 'html':
                    additional_urls = self.discover_additional_pages(source.url, content)
                    discovered_urls.update(additional_urls)
            else:
                results['failed'].append(source.url)
        
        # Download discovered pages (limited to prevent explosion)
        for url in list(discovered_urls)[:10]:  # Limit discovered downloads
            try:
                # Create temporary source for discovered URL
                section = 'discovered'
                temp_source = DocumentSource(url, 'html', section, 3)
                result = self.download_document(temp_source)
                if result:
                    results['success'].append(result[0])
                else:
                    results['failed'].append(url)
            except Exception as e:
                self.logger.warning(f"Failed to download discovered URL {url}: {e}")
                results['failed'].append(url)
        
        self.logger.info(f"Download complete: {len(results['success'])} success, {len(results['failed'])} failed")
        return results


class MQL5TextExtractor:
    """Extracts and cleans text from downloaded MQL5 documentation"""
    
    def __init__(self, input_dir: str = "./mql5_docs"):
        self.input_dir = Path(input_dir)
        self.raw_dir = self.input_dir / "raw"
        self.extracted_dir = self.input_dir / "extracted"
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = extract_pdf_text(pdf_path)
            return self._clean_text(text)
        except Exception as e:
            self.logger.error(f"Failed to extract PDF {pdf_path}: {e}")
            return ""
    
    def extract_html_text(self, html_path: str) -> str:
        """Extract text from HTML file"""
        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Focus on main content areas
            main_content = None
            content_selectors = [
                'main', '.content', '.documentation', '.doc-content', 
                '.main-content', '#content', '.article', '.post-content'
            ]
            
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    main_content = element
                    break
            
            if not main_content:
                main_content = soup.body or soup
            
            # Extract text
            text = main_content.get_text(separator=' ', strip=True)
            return self._clean_text(text)
            
        except Exception as e:
            self.logger.error(f"Failed to extract HTML {html_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common navigation/UI text
        patterns_to_remove = [
            r'(?i)cookie policy',
            r'(?i)privacy policy', 
            r'(?i)terms of service',
            r'(?i)sign in',
            r'(?i)register',
            r'(?i)search\s*:',
            r'(?i)menu',
            r'(?i)navigation',
            r'(?i)breadcrumb',
            r'(?i)skip to content',
            r'(?i)back to top'
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text)
        
        # Clean up special characters while preserving code examples
        text = re.sub(r'[^\w\s\(\)\[\]{};.,!?:=+\-*/\'"<>&|#@$%]', ' ', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Final whitespace cleanup
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def extract_all(self) -> Dict[str, List[str]]:
        """Extract text from all downloaded documents"""
        results = {'success': [], 'failed': []}
        
        if not self.raw_dir.exists():
            self.logger.error(f"Raw directory not found: {self.raw_dir}")
            return results
        
        for file_path in self.raw_dir.iterdir():
            if not file_path.is_file():
                continue
                
            try:
                self.logger.info(f"Extracting text from: {file_path.name}")
                
                if file_path.suffix.lower() == '.pdf':
                    text = self.extract_pdf_text(str(file_path))
                elif file_path.suffix.lower() in ['.html', '.htm']:
                    text = self.extract_html_text(str(file_path))
                else:
                    self.logger.warning(f"Unsupported file type: {file_path}")
                    continue
                
                if text and len(text.strip()) > 100:  # Minimum content threshold
                    # Save extracted text
                    output_file = self.extracted_dir / f"{file_path.stem}.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    results['success'].append(str(output_file))
                    self.logger.info(f"Extracted {len(text)} characters to: {output_file.name}")
                else:
                    self.logger.warning(f"Insufficient content extracted from: {file_path.name}")
                    results['failed'].append(str(file_path))
                    
            except Exception as e:
                self.logger.error(f"Failed to extract {file_path}: {e}")
                results['failed'].append(str(file_path))
        
        self.logger.info(f"Extraction complete: {len(results['success'])} success, {len(results['failed'])} failed")
        return results


def main():
    """Main function to run the complete download and extraction pipeline"""
    
    # Initialize components
    downloader = MQL5DocDownloader(output_dir="./mql5_docs")
    extractor = MQL5TextExtractor(input_dir="./mql5_docs")
    
    print("=== MQL5 Documentation Download & Extraction Pipeline ===")
    
    # Step 1: Download documentation
    print("\n1. Downloading MQL5 documentation...")
    download_results = downloader.download_all()
    
    print(f"Downloaded: {len(download_results['success'])} files")
    if download_results['failed']:
        print(f"Failed downloads: {len(download_results['failed'])}")
    
    # Step 2: Extract text
    print("\n2. Extracting text from downloaded documents...")
    extraction_results = extractor.extract_all()
    
    print(f"Extracted: {len(extraction_results['success'])} text files")
    if extraction_results['failed']:
        print(f"Failed extractions: {len(extraction_results['failed'])}")
    
    # Summary
    print(f"\n=== Pipeline Complete ===")
    print(f"Total extracted documents: {len(extraction_results['success'])}")
    print(f"Text files available in: {extractor.extracted_dir}")
    
    return extraction_results


def main_with_existing_pdfs(pdf_directory: str):
    """Main function for processing existing PDF collection"""
    
    # Initialize components
    downloader = MQL5DocDownloader(output_dir="./mql5_docs")
    extractor = MQL5TextExtractor(input_dir="./mql5_docs")
    
    print("=== MQL5 Documentation Processing (Existing PDFs) ===")
    
    # Step 1: Process existing PDFs
    print(f"\n1. Processing existing PDFs from: {pdf_directory}")
    pdf_results = downloader.process_existing_pdfs(pdf_directory)
    
    print(f"Processed: {len(pdf_results['success'])} PDF files")
    if pdf_results['failed']:
        print(f"Failed to process: {len(pdf_results['failed'])} files")
    
    # Step 2: Extract text
    print("\n2. Extracting text from PDF documents...")
    extraction_results = extractor.extract_all()
    
    print(f"Extracted: {len(extraction_results['success'])} text files")
    if extraction_results['failed']:
        print(f"Failed extractions: {len(extraction_results['failed'])}")
    
    # Summary
    print(f"\n=== Processing Complete ===")
    print(f"Total extracted documents: {len(extraction_results['success'])}")
    print(f"Text files available in: {extractor.extracted_dir}")
    
    return extraction_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process existing PDFs from specified directory
        pdf_directory = sys.argv[1]
        main_with_existing_pdfs(pdf_directory)
    else:
        # Default: download and process
        main()