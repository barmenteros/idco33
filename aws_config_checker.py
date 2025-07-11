#!/usr/bin/env python3
"""
AWS Configuration Verification Tool
Directly tests AWS API Gateway and Lambda without going through PromptProxy
"""

import asyncio
import json
import time
import traceback
import httpx
from typing import Dict, Any

class AWSConfigChecker:
    """Direct AWS configuration and connectivity tester."""
    
    def __init__(self):
        # These should match your actual AWS configuration
        self.api_gateway_url = "https://b6qmhutxnc.execute-api.us-east-1.amazonaws.com/prod/rag"
        self.api_key = "DNpEmzqcgQ2GcwB10LDBx9H3wBnQZ0Cr7z17HDzh"  # UPDATE THIS!
        
    async def run_aws_verification(self):
        """Run comprehensive AWS configuration verification."""
        print("☁️ AWS Configuration Verification Tool")
        print("=" * 60)
        print(f"API Gateway URL: {self.api_gateway_url}")
        print(f"API Key: {'*' * 20}...{self.api_key[-4:] if len(self.api_key) > 24 else '[SET YOUR KEY]'}")
        print()
        
        if self.api_key == "REPLACE_WITH_YOUR_ACTUAL_API_KEY":
            print("❌ CRITICAL: Update the API key in this script before running!")
            print("   Edit line 15 with your real AWS API Gateway key")
            return
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test 1: Basic AWS API Gateway connectivity
            await self._test_api_gateway_connectivity(client)
            
            # Test 2: Direct Lambda invocation
            await self._test_direct_lambda_call(client)
            
            # Test 3: Authentication verification
            await self._test_authentication(client)
            
            # Test 4: AWS service health
            await self._test_aws_service_health(client)
    
    async def _test_api_gateway_connectivity(self, client: httpx.AsyncClient):
        """Test basic API Gateway connectivity."""
        print("🔗 Test 1: API Gateway Connectivity")
        print("-" * 40)
        
        try:
            # Test without payload first (should get method not allowed or similar)
            start_time = time.time()
            response = await client.get(self.api_gateway_url)
            response_time = (time.time() - start_time) * 1000
            
            print(f"📊 GET Response: {response.status_code} ({response_time:.1f}ms)")
            print(f"📊 Response Text: {response.text[:200]}")
            
            if response.status_code == 403:
                print("❌ 403 Forbidden - Likely API key authentication issue")
            elif response.status_code == 404:
                print("❌ 404 Not Found - API Gateway URL might be incorrect")
            elif response.status_code == 405:
                print("✅ 405 Method Not Allowed - Expected (endpoint exists, needs POST)")
            else:
                print(f"ℹ️ Unexpected status code: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Connection failed: {type(e).__name__} - {e}")
            print("   This indicates network/DNS issues or incorrect URL")
    
    async def _test_direct_lambda_call(self, client: httpx.AsyncClient):
        """Test direct Lambda invocation through API Gateway."""
        print("\n🚀 Test 2: Direct Lambda Invocation")
        print("-" * 40)
        
        test_payload = {
            "prompt": "How do I use ArrayResize() in MQL5 for dynamic arrays?",
            "user": "aws_direct_test",
            "session_id": "direct_test_session"
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        try:
            start_time = time.time()
            response = await client.post(
                self.api_gateway_url, 
                json=test_payload,
                headers=headers
            )
            response_time = (time.time() - start_time) * 1000
            
            print(f"📊 POST Response: {response.status_code} ({response_time:.1f}ms)")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    snippets = result.get("snippets", [])
                    print(f"✅ Lambda executed successfully!")
                    print(f"📊 Snippets returned: {len(snippets)}")
                    
                    if len(snippets) > 0:
                        print(f"📊 First snippet preview: {snippets[0].get('snippet', '')[:100]}...")
                        print("✅ AWS integration is working correctly!")
                    else:
                        print("⚠️ Lambda executed but returned no snippets")
                        print("   This suggests DynamoDB/FAISS index issues")
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ Response is not valid JSON: {e}")
                    print(f"📊 Response text: {response.text[:500]}")
                    
            elif response.status_code == 403:
                print("❌ 403 Forbidden - API key authentication failed")
                print("   Check if your API key is correct and active")
                
            elif response.status_code == 500:
                print("❌ 500 Internal Server Error - Lambda execution failed")
                try:
                    error_data = response.json()
                    print(f"📊 Error details: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"📊 Error response: {response.text}")
                print("   Check CloudWatch logs for Lambda error details")
                
            elif response.status_code == 502 or response.status_code == 503:
                print(f"❌ {response.status_code} - API Gateway/Lambda service issue")
                print("   This might be a temporary AWS service problem")
                
            else:
                print(f"❌ Unexpected status: {response.status_code}")
                print(f"📊 Response: {response.text[:500]}")
                
        except Exception as e:
            print(f"❌ Direct Lambda call failed: {type(e).__name__} - {e}")
            traceback.print_exc()
    
    async def _test_authentication(self, client: httpx.AsyncClient):
        """Test API key authentication specifically."""
        print("\n🔐 Test 3: Authentication Verification")
        print("-" * 40)
        
        # Test with correct API key
        correct_headers = {"x-api-key": self.api_key}
        
        # Test with incorrect API key  
        wrong_headers = {"x-api-key": "wrong-key-123"}
        
        # Test with no API key
        no_headers = {}
        
        test_cases = [
            ("Correct API Key", correct_headers),
            ("Wrong API Key", wrong_headers),
            ("No API Key", no_headers)
        ]
        
        test_payload = {"prompt": "test", "user": "auth_test"}
        
        for test_name, headers in test_cases:
            try:
                response = await client.post(
                    self.api_gateway_url,
                    json=test_payload,
                    headers=headers
                )
                
                print(f"📊 {test_name}: {response.status_code}")
                
                if test_name == "Correct API Key" and response.status_code != 200:
                    print(f"   ❌ Expected 200, got {response.status_code}")
                elif test_name != "Correct API Key" and response.status_code == 403:
                    print(f"   ✅ Correctly rejected invalid auth")
                    
            except Exception as e:
                print(f"   ❌ {test_name} test failed: {e}")
    
    async def _test_aws_service_health(self, client: httpx.AsyncClient):
        """Test general AWS service health."""
        print("\n💚 Test 4: AWS Service Health Check")
        print("-" * 40)
        
        # Test API Gateway health by hitting different HTTP methods
        methods_to_test = ["GET", "POST", "PUT", "DELETE"]
        
        for method in methods_to_test:
            try:
                if method == "GET":
                    response = await client.get(self.api_gateway_url)
                elif method == "POST":
                    response = await client.post(self.api_gateway_url, json={})
                elif method == "PUT":
                    response = await client.put(self.api_gateway_url, json={})
                elif method == "DELETE":
                    response = await client.delete(self.api_gateway_url)
                
                print(f"📊 {method}: {response.status_code}")
                
                # Any response (even error) indicates the service is reachable
                if response.status_code in [200, 400, 403, 404, 405, 500]:
                    print(f"   ✅ API Gateway is responding")
                else:
                    print(f"   ⚠️ Unusual response: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ {method} failed: {type(e).__name__}")
        
        print(f"\n📋 Summary:")
        print(f"   API Gateway URL: {self.api_gateway_url}")
        print(f"   Region: us-east-1 (from URL)")
        print(f"   Stage: prod (from URL)")
        print(f"   Resource: /rag (from URL)")


async def main():
    """Run AWS configuration verification."""
    checker = AWSConfigChecker()
    
    print("⚠️ IMPORTANT: Before running this test:")
    print("   1. Update the API key on line 15 of this script")
    print("   2. Verify the API Gateway URL is correct")
    print("   3. Ensure you have internet connectivity")
    print()
    
    try:
        await checker.run_aws_verification()
        
        print("\n" + "=" * 60)
        print("🎯 NEXT STEPS BASED ON RESULTS:")
        print("1. If direct AWS calls work → Problem is in PromptProxy circuit breaker")
        print("2. If AWS calls fail → Fix AWS configuration first")
        print("3. If 403 errors → Check API key")
        print("4. If 500 errors → Check CloudWatch logs")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())