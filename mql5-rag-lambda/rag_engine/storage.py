"""
Storage Manager for MQL5 RAG Engine
Task C12: Create Dockerfile for Lambda Container
Module: RAG Engine - Data Access Layer

Handles all AWS storage operations:
- S3: FAISS index files and model artifacts
- DynamoDB: Document snippets and metadata
- Error handling and retry logic
- Performance optimization for Lambda environment
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError, BotoCoreError

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for storage operations."""
    s3_bucket: str
    dynamodb_table: str
    aws_region: str = 'us-east-1'
    cache_dir: str = '/tmp'
    download_timeout: int = 30
    max_retries: int = 3


@dataclass
class SnippetData:
    """Structure for document snippet data."""
    doc_id: str
    snippet_text: str
    source: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None


class StorageManager:
    """
    Manages AWS storage operations for the RAG engine.
    Optimized for Lambda environment with caching and error handling.
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.s3_client = None
        self.dynamodb_client = None
        self._initialized = False
        
    def _init_clients(self) -> None:
        """Initialize AWS clients with error handling."""
        if self._initialized:
            return
            
        try:
            self.s3_client = boto3.client('s3', region_name=self.config.aws_region)
            self.dynamodb_client = boto3.client('dynamodb', region_name=self.config.aws_region)
            self._initialized = True
            logger.info("AWS clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise RuntimeError(f"AWS client initialization failed: {e}")
    
    def download_faiss_index(self, force_refresh: bool = False) -> Tuple[str, str]:
        """
        Download FAISS index files from S3 to local cache.
        
        Args:
            force_refresh: Force re-download even if cached files exist
            
        Returns:
            Tuple of (index_file_path, mapping_file_path)
            
        Raises:
            RuntimeError: If download fails
        """
        self._init_clients()
        
        # Define local file paths
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(exist_ok=True)
        
        index_file = cache_dir / "index.faiss"
        mapping_file = cache_dir / "docid_mappings.json"
        
        # Check if files exist and force_refresh is False
        if not force_refresh and index_file.exists() and mapping_file.exists():
            logger.info("Using cached FAISS index files")
            return str(index_file), str(mapping_file)
        
        start_time = time.time()
        
        try:
            # Download FAISS index
            logger.info(f"Downloading FAISS index from s3://{self.config.s3_bucket}/index.faiss")
            self._download_file_with_retry("index.faiss", str(index_file))
            
            # Download doc ID mapping
            logger.info(f"Downloading doc mappings from s3://{self.config.s3_bucket}/docid_mappings.json")
            self._download_file_with_retry("docid_mappings.json", str(mapping_file))
            
            download_time = time.time() - start_time
            
            # Validate downloaded files
            self._validate_faiss_files(str(index_file), str(mapping_file))
            
            logger.info(f"FAISS index files downloaded successfully in {download_time:.2f}s")
            return str(index_file), str(mapping_file)
            
        except Exception as e:
            logger.error(f"Failed to download FAISS index: {e}")
            # Cleanup partial downloads
            for file_path in [index_file, mapping_file]:
                if file_path.exists():
                    file_path.unlink()
            raise RuntimeError(f"FAISS index download failed: {e}")
    
    def _download_file_with_retry(self, s3_key: str, local_path: str) -> None:
        """Download file with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                self.s3_client.download_file(
                    self.config.s3_bucket, 
                    s3_key, 
                    local_path
                )
                return
                
            except ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                logger.warning(f"Download attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _validate_faiss_files(self, index_file: str, mapping_file: str) -> None:
        """Validate downloaded FAISS files."""
        # Check file sizes
        index_size = Path(index_file).stat().st_size
        mapping_size = Path(mapping_file).stat().st_size
        
        if index_size == 0:
            raise ValueError("Downloaded FAISS index file is empty")
        if mapping_size == 0:
            raise ValueError("Downloaded mapping file is empty")
            
        # Validate mapping file JSON structure
        try:
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            if 'index_to_docid' not in mapping_data:
                raise ValueError("Mapping file missing 'index_to_docid' key")
                
            logger.info(f"Validated files: index={index_size:,} bytes, mapping={len(mapping_data['index_to_docid'])} entries")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in mapping file: {e}")
    
    def fetch_snippets_batch(self, doc_ids: List[str]) -> List[SnippetData]:
        """
        Fetch document snippets from DynamoDB in batch.
        
        Args:
            doc_ids: List of document IDs to retrieve
            
        Returns:
            List of SnippetData objects
            
        Raises:
            RuntimeError: If batch fetch fails
        """
        self._init_clients()
        
        if not doc_ids:
            return []
        
        start_time = time.time()
        
        try:
            # Prepare batch get request
            request_items = {
                self.config.dynamodb_table: {
                    'Keys': [{'DocID': {'S': doc_id}} for doc_id in doc_ids]
                }
            }
            
            logger.info(f"Fetching {len(doc_ids)} snippets from DynamoDB")
            
            response = self.dynamodb_client.batch_get_item(RequestItems=request_items)
            
            # Process response
            snippets = []
            items = response.get('Responses', {}).get(self.config.dynamodb_table, [])
            
            for item in items:
                try:
                    snippet = self._parse_dynamodb_item(item)
                    snippets.append(snippet)
                except Exception as e:
                    doc_id = item.get('DocID', {}).get('S', 'unknown')
                    logger.warning(f"Failed to parse snippet {doc_id}: {e}")
                    continue
            
            # Handle unprocessed items (rare in batch_get_item)
            unprocessed = response.get('UnprocessedKeys', {})
            if unprocessed:
                logger.warning(f"Unprocessed items: {len(unprocessed)}")
            
            fetch_time = time.time() - start_time
            logger.info(f"Retrieved {len(snippets)} snippets in {fetch_time*1000:.1f}ms")
            
            return snippets
            
        except Exception as e:
            logger.error(f"Failed to fetch snippets from DynamoDB: {e}")
            raise RuntimeError(f"DynamoDB batch fetch failed: {e}")
    
    def _parse_dynamodb_item(self, item: Dict[str, Any]) -> SnippetData:
        """Parse DynamoDB item into SnippetData object."""
        try:
            # Extract basic fields
            doc_id = item['DocID']['S']
            snippet_text = item.get('snippet_text', {}).get('S', '')
            source = item.get('source', {}).get('S', 'unknown')
            
            # Parse metadata (stored as DynamoDB Map)
            metadata = {}
            metadata_map = item.get('metadata', {}).get('M', {})
            
            for key, value in metadata_map.items():
                if 'S' in value:
                    metadata[key] = value['S']
                elif 'N' in value:
                    try:
                        metadata[key] = float(value['N'])
                    except ValueError:
                        metadata[key] = value['N']  # Keep as string if conversion fails
                elif 'BOOL' in value:
                    metadata[key] = value['BOOL']
                else:
                    metadata[key] = str(value)  # Fallback to string
            
            return SnippetData(
                doc_id=doc_id,
                snippet_text=snippet_text,
                source=source,
                metadata=metadata
            )
            
        except KeyError as e:
            raise ValueError(f"Missing required field in DynamoDB item: {e}")
    
    def get_snippet_by_id(self, doc_id: str) -> Optional[SnippetData]:
        """
        Fetch a single snippet by document ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            SnippetData object or None if not found
        """
        snippets = self.fetch_snippets_batch([doc_id])
        return snippets[0] if snippets else None
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on storage connections.
        
        Returns:
            Dict with health status of each service
        """
        self._init_clients()
        
        health = {
            's3_accessible': False,
            'dynamodb_accessible': False,
            'faiss_index_available': False
        }
        
        # Check S3 access
        try:
            self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
            health['s3_accessible'] = True
        except Exception as e:
            logger.warning(f"S3 health check failed: {e}")
        
        # Check DynamoDB access
        try:
            self.dynamodb_client.describe_table(TableName=self.config.dynamodb_table)
            health['dynamodb_accessible'] = True
        except Exception as e:
            logger.warning(f"DynamoDB health check failed: {e}")
        
        # Check FAISS index availability
        try:
            self.s3_client.head_object(Bucket=self.config.s3_bucket, Key="index.faiss")
            health['faiss_index_available'] = True
        except Exception as e:
            logger.warning(f"FAISS index check failed: {e}")
        
        return health
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage configuration and status information."""
        return {
            'config': {
                's3_bucket': self.config.s3_bucket,
                'dynamodb_table': self.config.dynamodb_table,
                'aws_region': self.config.aws_region,
                'cache_dir': self.config.cache_dir
            },
            'initialized': self._initialized,
            'health': self.health_check() if self._initialized else {}
        }


# Factory function for easy initialization
def create_storage_manager(
    s3_bucket: str,
    dynamodb_table: str,
    aws_region: str = 'us-east-1',
    cache_dir: str = '/tmp'
) -> StorageManager:
    """
    Factory function to create a StorageManager with default configuration.
    
    Args:
        s3_bucket: S3 bucket name for FAISS index
        dynamodb_table: DynamoDB table name for snippets
        aws_region: AWS region
        cache_dir: Local cache directory
        
    Returns:
        Configured StorageManager instance
    """
    config = StorageConfig(
        s3_bucket=s3_bucket,
        dynamodb_table=dynamodb_table,
        aws_region=aws_region,
        cache_dir=cache_dir
    )
    
    return StorageManager(config)