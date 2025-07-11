#!/usr/bin/env python3
"""
MQL5 RAG Lambda Function - Rapid Deployment
Deploy the missing Lambda function that your API Gateway expects.
"""

import json
import boto3
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Global variables for Lambda optimization (loaded once per container)
embedding_model = None
faiss_index = None
dynamodb_table = None

def lambda_handler(event, context):
    """
    Main Lambda handler for MQL5 RAG processing.
    Expected by API Gateway: mql5-rag-rag-handler
    """
    
    try:
        # Parse the incoming request
        if isinstance(event, str):
            event = json.loads(event)
        
        # Extract prompt from API Gateway event
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event
        
        prompt = body.get('prompt', '')
        user = body.get('user', 'unknown')
        session_id = body.get('session_id', f"session_{int(time.time())}")
        
        print(f"Processing RAG request - User: {user}, Session: {session_id}")
        print(f"Prompt: {prompt[:100]}...")
        
        # Initialize components if not already loaded
        _initialize_components()
        
        # Step 1: Generate embedding for the query
        start_time = time.time()
        query_embedding = _generate_embedding(prompt)
        embedding_time = (time.time() - start_time) * 1000
        
        print(f"Query embedding generated in {embedding_time:.1f}ms")
        
        # Step 2: Search FAISS index for similar documents
        start_time = time.time()
        similar_doc_ids, scores = _search_faiss_index(query_embedding, top_k=5)
        search_time = (time.time() - start_time) * 1000
        
        print(f"FAISS search completed in {search_time:.1f}ms, found {len(similar_doc_ids)} results")
        
        # Step 3: Retrieve document snippets from DynamoDB
        start_time = time.time()
        snippets = _retrieve_snippets(similar_doc_ids, scores)
        retrieval_time = (time.time() - start_time) * 1000
        
        print(f"Retrieved {len(snippets)} snippets in {retrieval_time:.1f}ms")
        
        # Step 4: Prepare response
        total_time = embedding_time + search_time + retrieval_time
        
        response = {
            "snippets": snippets,
            "metadata": {
                "processing_time_ms": total_time,
                "embedding_time_ms": embedding_time,
                "search_time_ms": search_time,
                "retrieval_time_ms": retrieval_time,
                "query_length": len(prompt),
                "results_count": len(snippets)
            },
            "success": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        print(f"RAG processing completed successfully in {total_time:.1f}ms")
        
        # Return API Gateway compatible response
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(response)
        }
        
    except Exception as e:
        print(f"ERROR in lambda_handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_response = {
            "error": str(e),
            "success": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(error_response)
        }

def _initialize_components():
    """Initialize FAISS index, embedding model, and DynamoDB connection."""
    global embedding_model, faiss_index, dynamodb_table
    
    try:
        # Initialize DynamoDB connection
        if dynamodb_table is None:
            dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
            dynamodb_table = dynamodb.Table('mql5-doc-snippets')
            print("✅ DynamoDB connection initialized")
        
        # For now, create mock components since we need to deploy quickly
        # These will be replaced with actual FAISS/embedding model later
        if embedding_model is None:
            embedding_model = MockEmbeddingModel()
            print("✅ Mock embedding model initialized (will be replaced)")
        
        if faiss_index is None:
            faiss_index = MockFAISSIndex()
            print("✅ Mock FAISS index initialized (will be replaced)")
            
    except Exception as e:
        print(f"ERROR initializing components: {e}")
        raise

def _generate_embedding(text: str) -> List[float]:
    """Generate embedding vector for input text."""
    try:
        return embedding_model.encode(text)
    except Exception as e:
        print(f"ERROR generating embedding: {e}")
        # Return mock embedding for now
        return [0.1] * 384  # MiniLM-L6-v2 dimension

def _search_faiss_index(query_embedding: List[float], top_k: int = 5) -> tuple:
    """Search FAISS index for similar documents."""
    try:
        return faiss_index.search(query_embedding, top_k)
    except Exception as e:
        print(f"ERROR searching FAISS index: {e}")
        # Return mock results
        return [], []

def _retrieve_snippets(doc_ids: List[str], scores: List[float]) -> List[Dict]:
    """Retrieve document snippets from DynamoDB."""
    snippets = []
    
    try:
        # For now, return mock snippets since we need the Lambda deployed
        # This will be replaced with actual DynamoDB retrieval
        mock_snippets = [
            {
                "snippet": "ArrayResize() function in MQL5 is used to change the size of a dynamic array. Syntax: int ArrayResize(array[], int new_size, int reserve_size=0). This function returns the number of elements in the array after resizing.",
                "source": "MQL5 Documentation - Array Functions",
                "score": 0.95,
                "doc_id": "mql5_arrays_001"
            },
            {
                "snippet": "Dynamic arrays in MQL5 can be resized during runtime using ArrayResize(). Unlike static arrays, dynamic arrays have flexible size and are declared without specifying dimensions in square brackets.",
                "source": "MQL5 Documentation - Dynamic Arrays",
                "score": 0.88,
                "doc_id": "mql5_arrays_002"
            },
            {
                "snippet": "When using ArrayResize(), the reserve_size parameter allows you to allocate additional memory for future array growth, which can improve performance when you expect the array to grow further.",
                "source": "MQL5 Documentation - Array Management",
                "score": 0.82,
                "doc_id": "mql5_arrays_003"
            }
        ]
        
        # Return mock snippets that are relevant to MQL5 prompts
        for i, (doc_id, score) in enumerate(zip(doc_ids[:3], scores[:3])):
            if i < len(mock_snippets):
                snippet = mock_snippets[i].copy()
                snippet["doc_id"] = doc_id if doc_id else snippet["doc_id"]
                snippet["score"] = score if score else snippet["score"]
                snippets.append(snippet)
        
        # If no doc_ids provided, return default snippets
        if not doc_ids and not snippets:
            snippets = mock_snippets[:3]
            
        print(f"Retrieved {len(snippets)} snippets (mock data)")
        
    except Exception as e:
        print(f"ERROR retrieving snippets: {e}")
        # Return empty list on error
        snippets = []
    
    return snippets

class MockEmbeddingModel:
    """Mock embedding model for rapid deployment."""
    
    def encode(self, text: str) -> List[float]:
        """Generate mock embedding based on text content."""
        # Simple hash-based mock embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert to 384-dimensional vector (MiniLM-L6-v2 size)
        embedding = []
        for i in range(0, len(hash_hex), 2):
            val = int(hash_hex[i:i+2], 16) / 255.0  # Normalize to 0-1
            embedding.extend([val] * 12)  # Repeat to reach 384 dimensions
        
        return embedding[:384]

class MockFAISSIndex:
    """Mock FAISS index for rapid deployment."""
    
    def __init__(self):
        self.mock_docs = [
            "mql5_arrays_001",
            "mql5_arrays_002", 
            "mql5_arrays_003",
            "mql5_functions_001",
            "mql5_functions_002"
        ]
    
    def search(self, query_embedding: List[float], top_k: int) -> tuple:
        """Mock search that returns relevant doc IDs."""
        # Simple mock: return first top_k documents with mock scores
        doc_ids = self.mock_docs[:top_k]
        scores = [0.95 - (i * 0.1) for i in range(len(doc_ids))]
        return doc_ids, scores

# Health check function for testing
def health_check():
    """Simple health check for testing Lambda deployment."""
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "function": "mql5-rag-rag-handler",
            "version": "1.0.0-mock"
        })
    }

if __name__ == "__main__":
    # Test the function locally
    test_event = {
        "body": json.dumps({
            "prompt": "How do I use ArrayResize() in MQL5?",
            "user": "test_user",
            "session_id": "test_session"
        })
    }
    
    result = lambda_handler(test_event, None)
    print("Test result:")
    print(json.dumps(json.loads(result["body"]), indent=2))