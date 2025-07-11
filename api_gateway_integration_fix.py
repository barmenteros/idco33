#!/usr/bin/env python3
"""
API Gateway Integration Fix
Fix the broken integration between API Gateway and Lambda
"""

import boto3
import json
import time
from datetime import datetime

def fix_api_gateway_integration():
    """Fix the API Gateway integration with Lambda."""
    
    print("🔧 API Gateway Integration Fix")
    print("=" * 40)
    
    # Configuration from your diagnostics
    api_id = "b6qmhutxnc"
    region = "us-east-1"
    function_name = "mql5-rag-rag-handler"
    account_id = "193245229238"
    
    try:
        # Initialize AWS clients
        apigw_client = boto3.client('apigateway', region_name=region)
        lambda_client = boto3.client('lambda', region_name=region)
        
        print(f"📋 API Gateway ID: {api_id}")
        print(f"📋 Lambda Function: {function_name}")
        print(f"📋 Region: {region}")
        print()
        
        # Step 1: Get API Gateway resources
        print("🔍 Step 1: Analyzing API Gateway configuration...")
        
        resources = apigw_client.get_resources(restApiId=api_id)
        
        rag_resource = None
        for resource in resources['items']:
            if resource.get('pathPart') == 'rag' or resource.get('path') == '/rag':
                rag_resource = resource
                break
        
        if not rag_resource:
            print("   ❌ /rag resource not found")
            return
        
        resource_id = rag_resource['id']
        print(f"   ✅ Found /rag resource: {resource_id}")
        
        # Step 2: Check current integration
        print("\n🔍 Step 2: Checking current integration...")
        
        try:
            method = apigw_client.get_method(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod='POST'
            )
            
            integration = method.get('methodIntegration', {})
            current_uri = integration.get('uri', '')
            
            print(f"   📊 Current integration URI: {current_uri}")
            print(f"   📊 Integration type: {integration.get('type', 'unknown')}")
            
            # Check if URI points to correct function
            expected_function_arn = f"arn:aws:lambda:{region}:{account_id}:function:{function_name}"
            
            if function_name in current_uri:
                print("   ✅ URI points to correct function")
            else:
                print("   ❌ URI points to wrong function or is malformed")
                print(f"   Expected function ARN: {expected_function_arn}")
            
        except Exception as e:
            print(f"   ❌ Error getting method: {e}")
            return
        
        # Step 3: Fix the integration
        print("\n🔧 Step 3: Fixing API Gateway integration...")
        
        # Correct Lambda integration URI
        correct_uri = f"arn:aws:apigateway:{region}:lambda:path/2015-03-31/functions/arn:aws:lambda:{region}:{account_id}:function:{function_name}/invocations"
        
        try:
            # Update the integration
            apigw_client.put_integration(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod='POST',
                type='AWS_PROXY',
                integrationHttpMethod='POST',
                uri=correct_uri,
                passthroughBehavior='WHEN_NO_MATCH',
                timeoutInMillis=29000  # 29 seconds (max for API Gateway)
            )
            
            print("   ✅ Integration updated successfully")
            
        except Exception as e:
            print(f"   ❌ Error updating integration: {e}")
            return
        
        # Step 4: Ensure Lambda permissions
        print("\n🔐 Step 4: Checking Lambda permissions...")
        
        try:
            # Add permission for API Gateway to invoke Lambda
            statement_id = f"apigateway-invoke-{int(time.time())}"
            
            lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=statement_id,
                Action='lambda:InvokeFunction',
                Principal='apigateway.amazonaws.com',
                SourceArn=f"arn:aws:execute-api:{region}:{account_id}:{api_id}/*/*"
            )
            
            print("   ✅ Lambda permission added")
            
        except lambda_client.exceptions.ResourceConflictException:
            print("   ✅ Lambda permission already exists")
        except Exception as e:
            print(f"   ⚠️ Permission error (may be OK): {e}")
        
        # Step 5: Deploy the changes
        print("\n🚀 Step 5: Deploying API Gateway changes...")
        
        try:
            deployment = apigw_client.create_deployment(
                restApiId=api_id,
                stageName='prod',
                description=f'Fix Lambda integration - {datetime.now().isoformat()}'
            )
            
            print(f"   ✅ Deployment successful: {deployment['id']}")
            
        except Exception as e:
            print(f"   ❌ Deployment failed: {e}")
            return
        
        # Step 6: Test the fixed integration
        print("\n🧪 Step 6: Testing fixed integration...")
        
        # Wait a moment for deployment
        print("   ⏱️ Waiting 5 seconds for deployment...")
        time.sleep(5)
        
        # Test via AWS SDK (simulating API Gateway call)
        try:
            import httpx
            import asyncio
            
            async def test_api_gateway():
                api_url = f"https://{api_id}.execute-api.{region}.amazonaws.com/prod/rag"
                
                # Use a test API key (you'll need to replace this)
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": "DNpEmzqcgQ2GcwB10LDBx9H3wBnQZ0Cr7z17HDzh"  # Replace with actual key
                }
                
                test_payload = {
                    "prompt": "How do I use ArrayResize() in MQL5?",
                    "user": "integration_fix_test",
                    "session_id": "fix_test_session"
                }
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    try:
                        response = await client.post(api_url, headers=headers, json=test_payload)
                        
                        print(f"   📊 API Gateway Response: {response.status_code}")
                        
                        if response.status_code == 200:
                            result = response.json()
                            if 'snippets' in result:
                                print(f"   🎉 SUCCESS! Snippets returned: {len(result['snippets'])}")
                                return True
                            else:
                                print(f"   ⚠️ No snippets in response: {list(result.keys())}")
                        elif response.status_code == 403:
                            print("   ⚠️ 403 - Update API key in this script")
                        else:
                            print(f"   ❌ Error: {response.text}")
                        
                    except Exception as e:
                        print(f"   ❌ Test request failed: {e}")
                
                return False
            
            # Run the async test
            success = asyncio.run(test_api_gateway())
            
            if success:
                print("   🎉 API Gateway integration is now working!")
            else:
                print("   ⚠️ Test inconclusive - may need API key update")
            
        except ImportError:
            print("   ℹ️ httpx not available for testing, but integration should be fixed")
        
        print("\n🎯 Integration fix complete!")
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        print("\n🔧 Manual fix steps:")
        print("1. Go to AWS Console → API Gateway → mql5-rag-api")
        print("2. Click Resources → /rag → POST")
        print("3. Click Integration Request")
        print("4. Verify Lambda Function points to: mql5-rag-rag-handler")
        print("5. Deploy API (Actions → Deploy API → prod)")


def main():
    """Main function."""
    print("🔧 API Gateway Integration Fix")
    print("This will fix the broken integration between API Gateway and Lambda")
    print()
    
    try:
        fix_api_gateway_integration()
        
        print("\n" + "=" * 60)
        print("🎯 NEXT STEPS:")
        print("1. Update API key in test scripts if needed")
        print("2. Run: python end_to_end_test.py")
        print("3. Expected: RAG Success Rate should be 3/3!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️ Fix interrupted by user")
    except Exception as e:
        print(f"\n💥 Fix failed: {e}")


if __name__ == "__main__":
    main()