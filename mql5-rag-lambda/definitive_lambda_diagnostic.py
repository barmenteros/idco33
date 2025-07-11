#!/usr/bin/env python3
"""
Definitive Lambda Diagnostic
This will show us exactly what's happening and where the code is failing
"""

import sys
import os
import json

# Add parent directory to path to import our lambda function
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
print(f"Looking for lambda_function.py in: {parent_dir}")

# Also try current directory
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir) 
print(f"Also checking: {current_dir}")

# List what's actually in the parent directory
try:
    files_in_parent = os.listdir(parent_dir)
    print(f"Files in parent dir: {[f for f in files_in_parent if f.endswith('.py')]}")
except:
    pass

def test_lambda_step_by_step():
    """Test each step of the Lambda function individually"""
    
    print("=== DEFINITIVE LAMBDA DIAGNOSTIC ===")
    
    # Test 1: Import the lambda function
    print("1. Testing Lambda function import...")
    try:
        import lambda_function
        print("✅ Lambda function imported successfully")
        
        # Check if our updates are actually in the code
        with open('lambda_function.py', 'r') as f:
            code_content = f.read()
        
        if 'Available files in S3:' in code_content:
            print("✅ Updated debugging code is present")
        else:
            print("❌ Updated debugging code is NOT present - using old version")
            return
            
    except Exception as e:
        print(f"❌ Failed to import lambda function: {e}")
        return
    
    # Test 2: Test individual components
    print("\n2. Testing individual components...")
    
    # Test AWS clients initialization
    try:
        lambda_function.init_aws_clients()
        print("✅ AWS clients initialized")
    except Exception as e:
        print(f"❌ AWS clients failed: {e}")
        return
    
    # Test embedding model loading
    try:
        print("3. Testing embedding model loading...")
        model = lambda_function.load_embedding_model()
        print(f"✅ Embedding model loaded: {type(model)}")
    except Exception as e:
        print(f"❌ Embedding model failed: {e}")
        return
    
    # Test embedding generation
    try:
        print("4. Testing query embedding...")
        query_embedding = lambda_function.embed_query("Test MQL5 query")
        print(f"✅ Query embedded: shape {query_embedding.shape}")
    except Exception as e:
        print(f"❌ Query embedding failed: {e}")
        return
    
    # Test S3 access step by step
    print("5. Testing S3 access step by step...")
    
    try:
        import boto3
        s3_client = boto3.client('s3')
        bucket = 'mql5-rag-faiss-index-20250106-minimal'  # Use the correct bucket name directly
        
        print(f"   Bucket: {bucket}")
        
        # Test bucket access
        s3_client.head_bucket(Bucket=bucket)
        print("   ✅ Bucket accessible")
        
        # List files
        response = s3_client.list_objects_v2(Bucket=bucket)
        files = [obj['Key'] for obj in response.get('Contents', [])]
        print(f"   ✅ Files found: {files}")
        
        # Test each file individually
        for file_key in files:
            try:
                s3_client.head_object(Bucket=bucket, Key=file_key)
                print(f"   ✅ {file_key}: HeadObject works")
            except Exception as e:
                print(f"   ❌ {file_key}: HeadObject failed - {e}")
        
        # Test downloads
        for file_key in ['index.faiss', 'metadata/index_metadata.json']:
            try:
                local_file = f"/tmp/test_{file_key.replace('/', '_')}"
                s3_client.download_file(bucket, file_key, local_file)
                print(f"   ✅ {file_key}: Download works")
                
                # Check file content
                if os.path.exists(local_file):
                    size = os.path.getsize(local_file)
                    print(f"      Downloaded {size} bytes")
                    
                    if file_key.endswith('.json'):
                        with open(local_file, 'r') as f:
                            data = json.load(f)
                        print(f"      JSON keys: {list(data.keys())}")
                
            except Exception as e:
                print(f"   ❌ {file_key}: Download failed - {e}")
        
    except Exception as e:
        print(f"❌ S3 testing failed: {e}")
        return
    
    # Test the load_faiss_index function directly
    print("6. Testing load_faiss_index function directly...")
    try:
        faiss_index, doc_mapping = lambda_function.load_faiss_index()
        print(f"✅ FAISS index loaded: {faiss_index.ntotal} vectors")
        print(f"✅ Doc mapping loaded: {len(doc_mapping)} entries")
        
        # Test a sample mapping
        sample_keys = list(doc_mapping.keys())[:3]
        for key in sample_keys:
            print(f"   Sample mapping: {key} -> {doc_mapping[key]}")
            
    except Exception as e:
        print(f"❌ load_faiss_index failed: {e}")
        import traceback
        print(f"   Full traceback: {traceback.format_exc()}")
        return
    
    # Test vector search
    print("7. Testing vector search...")
    try:
        query_vector = lambda_function.embed_query("ArrayResize MQL5")
        doc_ids, scores = lambda_function.search_similar_docs(query_vector)  # Fixed order
        print(f"✅ Vector search works: found {len(doc_ids)} results")
        print(f"   Sample results: {list(zip(doc_ids[:3], scores[:3]))}")
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return
    
    # Test DynamoDB access
    print("8. Testing DynamoDB access...")
    try:
        # Use the doc_ids from search results
        sample_doc_ids = doc_ids[:3]  # Take first 3 doc IDs
        snippets = lambda_function.fetch_snippets_from_dynamodb(sample_doc_ids)
        print(f"✅ DynamoDB access works: retrieved {len(snippets)} snippets")
        
        for snippet in snippets[:2]:
            print(f"   Sample snippet: {snippet['doc_id']} - {snippet['snippet_text'][:50]}...")
            
    except Exception as e:
        print(f"❌ DynamoDB access failed: {e}")
        return
    
    # Test full lambda handler
    print("9. Testing full Lambda handler...")
    try:
        class MockContext:
            aws_request_id = "diagnostic-test-123"
        
        event = {"prompt": "How to use ArrayResize in MQL5?"}
        context = MockContext()
        
        response = lambda_function.lambda_handler(event, context)
        print(f"✅ Full Lambda handler works!")
        
        body = json.loads(response['body'])
        if body['success']:
            print(f"   Retrieved {body['metadata']['retrieved_count']} results")
            print(f"   Processing time: {body['metadata']['processing_time_ms']}ms")
        else:
            print(f"   ❌ Lambda returned error: {body['error']}")
            
    except Exception as e:
        print(f"❌ Full Lambda handler failed: {e}")
        import traceback
        print(f"   Full traceback: {traceback.format_exc()}")
        return
    
    print("\n🎉 ALL TESTS PASSED! Lambda function is working correctly.")


if __name__ == "__main__":
    test_lambda_step_by_step()