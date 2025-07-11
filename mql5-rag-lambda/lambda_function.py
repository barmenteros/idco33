"""
MQL5 RAG Engine Lambda Handler - Working Version
Task C12: Create Dockerfile for Lambda Container
"""

import json
import time
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import traceback

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import heavy dependencies after logging setup
try:
    import boto3
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logger.error(f"Failed to import required dependencies: {e}")
    raise

# Global variables for container reuse
embedding_model: Optional[SentenceTransformer] = None
faiss_index: Optional[faiss.Index] = None
doc_id_mapping: Optional[Dict[str, str]] = None
dynamodb_client = None
s3_client = None

# Configuration from environment
CONFIG = {
    'embedding_model': os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
    's3_bucket': os.environ.get('FAISS_INDEX_BUCKET', 'mql5-rag-faiss-index-20250106-minimal'),
    'dynamodb_table': os.environ.get('SNIPPETS_TABLE', 'mql5-doc-snippets'),
    'top_k': int(os.environ.get('TOP_K_RESULTS', '5')),
    'cache_dir': '/tmp/model_cache',
    'max_query_length': int(os.environ.get('MAX_QUERY_LENGTH', '512'))
}

def init_aws_clients() -> None:
    """Initialize AWS service clients with error handling."""
    global dynamodb_client, s3_client
    
    try:
        if not dynamodb_client:
            dynamodb_client = boto3.client('dynamodb')
            logger.info("DynamoDB client initialized")
            
        if not s3_client:
            s3_client = boto3.client('s3')
            logger.info("S3 client initialized")
            
    except Exception as e:
        logger.error(f"Failed to initialize AWS clients: {e}")
        raise


def load_embedding_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.
    Now optimized for container initialization - will reuse preloaded model.
    """
    global embedding_model
    
    if embedding_model is not None:
        logger.debug("Using preloaded embedding model")
        return embedding_model
    
    start_time = time.time()
    model_name = CONFIG['embedding_model']
    
    try:
        logger.info(f"Loading embedding model: {model_name}")
        
        # Use Lambda's /tmp directory for caching
        cache_path = f"{CONFIG['cache_dir']}/{model_name}"
        os.makedirs(CONFIG['cache_dir'], exist_ok=True)
        
        # Load model (SentenceTransformers handles caching automatically)
        embedding_model = SentenceTransformer(model_name, cache_folder=CONFIG['cache_dir'])
        
        load_time = time.time() - start_time
        logger.info(f"Embedding model loaded in {load_time:.2f}s")
        
        return embedding_model
        
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise


def load_faiss_index() -> Tuple[faiss.Index, Dict[str, str]]:
    """
    Load FAISS index and doc ID mapping from S3.
    Now optimized for container initialization - will reuse preloaded index.
    """
    global faiss_index, doc_id_mapping
    
    if faiss_index is not None and doc_id_mapping is not None:
        logger.debug("Using preloaded FAISS index and mappings")
        return faiss_index, doc_id_mapping
    
    start_time = time.time()
    bucket = CONFIG['s3_bucket']
    
    try:
        logger.info(f"Loading FAISS index from S3 bucket: {bucket}")
        
        # List available files for debugging
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, MaxKeys=20)
            available_files = [obj['Key'] for obj in response.get('Contents', [])]
            logger.info(f"Available files in S3: {available_files}")
        except Exception as list_error:
            logger.warning(f"Could not list S3 files: {list_error}")
        
        # Download files to /tmp
        index_file = "/tmp/index.faiss"
        
        # Download FAISS index
        logger.info("Downloading index.faiss")
        s3_client.download_file(bucket, "index.faiss", index_file)
        
        # Load FAISS index
        faiss_index = faiss.read_index(index_file)
        logger.info(f"FAISS index loaded: {faiss_index.ntotal} vectors, {faiss_index.d} dimensions")
        
        # Download and load mapping from metadata/index_metadata.json
        mapping_file = "/tmp/metadata.json"
        logger.info("Downloading metadata/index_metadata.json")
        s3_client.download_file(bucket, "metadata/index_metadata.json", mapping_file)
        
        with open(mapping_file, 'r') as f:
            metadata = json.load(f)
        
        # Extract docid mapping
        if 'docid_mappings' in metadata and 'index_to_docid' in metadata['docid_mappings']:
            doc_id_mapping = metadata['docid_mappings']['index_to_docid']
            logger.info(f"Doc ID mapping loaded: {len(doc_id_mapping)} entries")
        else:
            raise ValueError("No valid doc ID mapping found in metadata")
        
        load_time = time.time() - start_time
        logger.info(f"FAISS index and mapping loaded in {load_time:.2f}s")
        
        return faiss_index, doc_id_mapping
        
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        raise


def embed_query(query_text: str) -> np.ndarray:
    """Convert query text to embedding vector."""
    if len(query_text) > CONFIG['max_query_length']:
        query_text = query_text[:CONFIG['max_query_length']]
    
    start_time = time.time()
    
    try:
        model = load_embedding_model()
        embedding = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        embed_time = time.time() - start_time
        logger.info(f"Query embedded in {embed_time*1000:.1f}ms")
        
        return embedding.astype('float32')
        
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise


def search_similar_docs(query_embedding: np.ndarray) -> Tuple[List[str], List[float]]:
    """Perform vector similarity search using FAISS."""
    start_time = time.time()
    top_k = CONFIG['top_k']
    
    try:
        index, doc_mapping = load_faiss_index()
        
        # Reshape for FAISS
        query_vector = query_embedding.reshape(1, -1)
        
        # Search
        scores, doc_indices = index.search(query_vector, top_k)
        
        # Convert to lists and get doc IDs - FIXED ORDER
        scores_list = scores[0].tolist()
        doc_ids = [doc_mapping.get(str(idx), f"unknown_{idx}") for idx in doc_indices[0]]
        
        search_time = time.time() - start_time
        logger.info(f"Vector search completed in {search_time*1000:.1f}ms")
        
        return doc_ids, scores_list  # Return doc_ids first, scores second
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise


def fetch_snippets_from_dynamodb(doc_ids: List[str]) -> List[Dict[str, Any]]:
    """Retrieve snippet text and metadata from DynamoDB."""
    start_time = time.time()
    table_name = CONFIG['dynamodb_table']
    
    try:
        # Prepare batch get request
        request_items = {
            table_name: {
                'Keys': [{'DocID': {'S': doc_id}} for doc_id in doc_ids]
            }
        }
        
        logger.info(f"Fetching {len(doc_ids)} snippets from DynamoDB")
        response = dynamodb_client.batch_get_item(RequestItems=request_items)
        
        snippets = []
        items = response.get('Responses', {}).get(table_name, [])
        
        for item in items:
            try:
                snippet = {
                    'doc_id': item['DocID']['S'],
                    'snippet_text': item.get('snippet_text', {}).get('S', ''),
                    'source': item.get('source', {}).get('S', 'unknown'),
                    'metadata': {}
                }
                snippets.append(snippet)
                
            except Exception as e:
                logger.warning(f"Failed to parse snippet: {e}")
                continue
        
        fetch_time = time.time() - start_time
        logger.info(f"Retrieved {len(snippets)} snippets in {fetch_time*1000:.1f}ms")
        
        return snippets
        
    except Exception as e:
        logger.error(f"DynamoDB fetch failed: {e}")
        raise


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda handler for MQL5 RAG processing."""
    start_time = time.time()
    request_id = context.aws_request_id if context else "local"
    
    logger.info(f"Processing request {request_id}")
    
    try:
        # Initialize AWS clients
        init_aws_clients()
        
        # Parse event body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event
        
        query_text = body.get('prompt', '').strip()
        
        if not query_text:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'success': False,
                    'error': 'Missing or empty prompt in request',
                    'request_id': request_id
                })
            }
        
        logger.info(f"Processing query: {query_text[:100]}...")
        
        # Step 1: Embed the query
        query_embedding = embed_query(query_text)
        
        # Step 2: Search for similar documents
        doc_ids, scores = search_similar_docs(query_embedding)
        
        # Step 3: Fetch snippet text from DynamoDB
        snippets = fetch_snippets_from_dynamodb(doc_ids)
        
        # Step 4: Combine results with scores
        results = []
        for i, snippet in enumerate(snippets):
            if i < len(scores):
                snippet['similarity_score'] = float(scores[i])
            results.append(snippet)
        
        # Sort by similarity score
        results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        total_time = time.time() - start_time
        
        response_body = {
            'success': True,
            'query': query_text,
            'results': results,
            'metadata': {
                'processing_time_ms': round(total_time * 1000, 2),
                'retrieved_count': len(results),
                'request_id': request_id,
                'timestamp': int(time.time())
            }
        }
        
        logger.info(f"Request {request_id} completed in {total_time*1000:.1f}ms")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_body)
        }
        
    except Exception as e:
        error_time = time.time() - start_time
        error_trace = traceback.format_exc()
        
        logger.error(f"Request {request_id} failed after {error_time*1000:.1f}ms: {e}")
        logger.error(f"Error trace: {error_trace}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': str(e),
                'request_id': request_id,
                'processing_time_ms': round(error_time * 1000, 2)
            })
        }


# Container initialization
def init_container():
    """
    Initialize container resources during startup.
    This function runs during Lambda container initialization to preload
    heavy resources (FAISS index and embedding model) for optimal performance.
    """
    global embedding_model, faiss_index, doc_id_mapping
    
    start_time = time.time()
    
    try:
        logger.info("Initializing Lambda container with preloaded resources...")
        
        # Initialize AWS clients first
        init_aws_clients()
        
        # Preload embedding model during initialization
        logger.info("Preloading embedding model during container init...")
        try:
            embedding_model = load_embedding_model()
            logger.info(f"✅ Embedding model preloaded successfully: {type(embedding_model)}")
        except Exception as e:
            logger.error(f"❌ Failed to preload embedding model: {e}")
            # Don't raise - allow container to start, will load on first request
            embedding_model = None
        
        # Preload FAISS index and mappings during initialization
        logger.info("Preloading FAISS index during container init...")
        try:
            faiss_index, doc_id_mapping = load_faiss_index()
            logger.info(f"✅ FAISS index preloaded successfully: {faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"❌ Failed to preload FAISS index: {e}")
            # Don't raise - allow container to start, will load on first request
            faiss_index = None
            doc_id_mapping = None
        
        init_time = time.time() - start_time
        
        # Log initialization summary
        resources_loaded = []
        if embedding_model is not None:
            resources_loaded.append("embedding_model")
        if faiss_index is not None:
            resources_loaded.append("faiss_index")
        if doc_id_mapping is not None:
            resources_loaded.append("doc_mappings")
        
        if resources_loaded:
            logger.info(f"🚀 Lambda container ready with preloaded resources: {resources_loaded}")
            logger.info(f"⚡ Container initialization completed in {init_time:.2f}s")
            logger.info("🎯 First request should be ~100ms (no cold start overhead)")
        else:
            logger.warning("⚠️ Container initialized but no resources preloaded - will load on first request")
            
    except Exception as e:
        logger.error(f"❌ Container initialization failed: {e}")
        # Log but don't crash - allow Lambda to start in degraded mode


# Initialize on module import
init_container()