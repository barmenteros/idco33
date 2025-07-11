#!/usr/bin/env python3
"""
Direct Lambda Test Tool
Bypass PromptProxy and test AWS Lambda directly to see the actual error
"""

import boto3
import json
import time
from datetime import datetime, timedelta

def test_lambda_directly():
    """Test the Lambda function directly via AWS SDK."""
    
    print("🧪 Direct Lambda Test Tool")
    print("=" * 40)
    
    function_name = "mql5-rag-rag-handler"
    region = "us-east-1"
    
    try:
        # Initialize Lambda client
        lambda_client = boto3.client('lambda', region_name=region)
        logs_client = boto3.client('logs', region_name=region)
        
        print(f"📋 Testing function: {function_name}")
        print(f"📋 Region: {region}")
        print()
        
        # Step 1: Verify function exists
        print("🔍 Step 1: Verifying Lambda function exists...")
        try:
            func_config = lambda_client.get_function(FunctionName=function_name)
            
            config = func_config['Configuration']
            print(f"   ✅ Function found: {config['FunctionName']}")
            print(f"   📊 Runtime: {config['Runtime']}")
            print(f"   📊 Memory: {config['MemorySize']} MB")
            print(f"   📊 Timeout: {config['Timeout']} seconds")
            print(f"   📊 Last Modified: {config['LastModified']}")
            print(f"   📊 State: {config['State']}")
            
            if config['State'] != 'Active':
                print(f"   ⚠️ Function state is not Active: {config['State']}")
                
        except Exception as e:
            print(f"   ❌ Function not found: {e}")
            return
        
        # Step 2: Test Lambda function with direct invocation
        print("\n🚀 Step 2: Direct Lambda invocation...")
        
        test_payload = {
            "body": json.dumps({
                "prompt": "How do I use ArrayResize() function in MQL5?",
                "user": "direct_test",
                "session_id": "direct_test_session"
            })
        }
        
        print(f"   📤 Payload: {json.dumps(test_payload, indent=2)}")
        print()
        
        try:
            start_time = time.time()
            
            response = lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(test_payload)
            )
            
            invoke_time = (time.time() - start_time) * 1000
            
            print(f"   📊 Invocation time: {invoke_time:.1f}ms")
            print(f"   📊 Status Code: {response['StatusCode']}")
            
            # Read the response payload
            response_payload = response['Payload'].read().decode('utf-8')
            
            if response['StatusCode'] == 200:
                print("   ✅ Lambda invocation: SUCCESS")
                
                try:
                    result = json.loads(response_payload)
                    print(f"   📊 Response structure: {json.dumps(result, indent=2)}")
                    
                    # Check if it's an API Gateway response format
                    if 'statusCode' in result and 'body' in result:
                        print(f"   📊 API Gateway Status: {result['statusCode']}")
                        
                        if result['statusCode'] == 200:
                            body = json.loads(result['body']) if isinstance(result['body'], str) else result['body']
                            
                            if 'snippets' in body:
                                snippets = body['snippets']
                                print(f"   ✅ Snippets returned: {len(snippets)}")
                                
                                if snippets:
                                    print("   📋 First snippet preview:")
                                    first_snippet = snippets[0]
                                    print(f"      Text: {first_snippet.get('snippet', '')[:100]}...")
                                    print(f"      Source: {first_snippet.get('source', 'N/A')}")
                                    print(f"      Score: {first_snippet.get('score', 'N/A')}")
                                    
                                    print("   🎉 LAMBDA IS WORKING CORRECTLY!")
                                else:
                                    print("   ❌ No snippets in response - check Lambda logic")
                            else:
                                print("   ❌ No 'snippets' field in response")
                                print(f"   📋 Available fields: {list(body.keys())}")
                        else:
                            print(f"   ❌ API Gateway error: {result['statusCode']}")
                            if 'body' in result:
                                error_body = json.loads(result['body']) if isinstance(result['body'], str) else result['body']
                                print(f"   📋 Error details: {error_body}")
                    else:
                        print("   ⚠️ Unexpected response format (not API Gateway)")
                        
                except json.JSONDecodeError as e:
                    print(f"   ❌ Response is not valid JSON: {e}")
                    print(f"   📋 Raw response: {response_payload[:500]}")
                    
            else:
                print(f"   ❌ Lambda invocation failed: {response['StatusCode']}")
                print(f"   📋 Error response: {response_payload}")
                
        except Exception as e:
            print(f"   ❌ Lambda invocation error: {e}")
            return
        
        # Step 3: Check recent CloudWatch logs
        print("\n📋 Step 3: Checking recent CloudWatch logs...")
        
        log_group_name = f"/aws/lambda/{function_name}"
        
        try:
            # Get the most recent log stream
            streams_response = logs_client.describe_log_streams(
                logGroupName=log_group_name,
                orderBy='LastEventTime',
                descending=True,
                limit=1
            )
            
            if streams_response['logStreams']:
                latest_stream = streams_response['logStreams'][0]
                stream_name = latest_stream['logStreamName']
                
                print(f"   📋 Latest log stream: {stream_name}")
                
                # Get recent events from this stream
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(minutes=5)  # Last 5 minutes
                
                events_response = logs_client.get_log_events(
                    logGroupName=log_group_name,
                    logStreamName=stream_name,
                    startTime=int(start_time.timestamp() * 1000),
                    endTime=int(end_time.timestamp() * 1000)
                )
                
                events = events_response['events']
                
                if events:
                    print(f"   📊 Found {len(events)} recent log events:")
                    print("   " + "-" * 50)
                    
                    for event in events[-10:]:  # Show last 10 events
                        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
                        message = event['message'].strip()
                        
                        # Highlight errors and important messages
                        if 'ERROR' in message or 'Exception' in message:
                            prefix = "   ❌"
                        elif 'Processing RAG request' in message:
                            prefix = "   🔍"
                        elif 'snippets' in message.lower():
                            prefix = "   📊"
                        else:
                            prefix = "   📄"
                        
                        print(f"{prefix} {timestamp.strftime('%H:%M:%S')} | {message}")
                else:
                    print("   ⚠️ No recent log events found")
            else:
                print("   ⚠️ No log streams found")
                
        except Exception as e:
            print(f"   ❌ Error checking logs: {e}")
    
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check AWS credentials: aws sts get-caller-identity")
        print("   2. Verify Lambda function exists: aws lambda list-functions --region us-east-1")
        print("   3. Check IAM permissions for Lambda operations")

def main():
    """Main function."""
    print("🧪 Direct Lambda Test Tool")
    print("This will test the Lambda function directly via AWS SDK")
    print("to identify the exact issue causing 'no_snippets' errors")
    print()
    
    try:
        test_lambda_directly()
        
        print("\n" + "=" * 60)
        print("🎯 NEXT STEPS:")
        print("1. If Lambda works here but fails via PromptProxy → API Gateway issue")
        print("2. If Lambda fails here → Fix Lambda code")
        print("3. If 'no snippets' → Lambda logic issue (normal for mock deployment)")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")

if __name__ == "__main__":
    main()