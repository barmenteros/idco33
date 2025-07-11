"""Local testing for Lambda function"""
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lambda_function import lambda_handler

class MockContext:
    aws_request_id = "test-request-123"

def test_lambda_locally():
    """Test the Lambda function locally"""
    
    # Test event
    event = {
        "prompt": "How to use ArrayResize in MQL5?"
    }
    
    context = MockContext()
    
    # Call the handler
    try:
        response = lambda_handler(event, context)
        print("Response:", json.dumps(response, indent=2))
        
        # Parse response body
        body = json.loads(response['body'])
        if body['success']:
            print(f"✅ Success: Retrieved {body['metadata']['retrieved_count']} results")
            print(f"⏱️ Processing time: {body['metadata']['processing_time_ms']}ms")
        else:
            print(f"❌ Error: {body['error']}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_lambda_locally()