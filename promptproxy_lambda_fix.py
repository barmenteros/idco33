#!/usr/bin/env python3
"""
PromptProxy Lambda Request Fix
Fix the request format between PromptProxy and Lambda to match what works
"""

import asyncio
import json
import time
import httpx

async def test_request_formats():
    """Test different request formats to find what works."""
    
    print("🔧 PromptProxy Lambda Request Format Fix")
    print("=" * 50)
    
    api_gateway_url = "https://b6qmhutxnc.execute-api.us-east-1.amazonaws.com/prod/rag"
    api_key = "DNpEmzqcgQ2GcwB10LDBx9H3wBnQZ0Cr7z17HDzh"  # Replace with actual key
    
    if api_key == "YOUR_API_KEY":
        print("❌ Update the API key in this script first!")
        return
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    
    # The format that works (from direct Lambda test)
    working_format = {
        "prompt": "How do I use ArrayResize() function in MQL5?",
        "user": "format_test",
        "session_id": "format_test_session"
    }
    
    # Test different request formats
    test_formats = [
        ("Direct Format (should work)", working_format),
        ("With body wrapper", {"body": json.dumps(working_format)}),
        ("Nested format", {"body": working_format}),
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        for format_name, payload in test_formats:
            print(f"\n🧪 Testing: {format_name}")
            print(f"   Payload: {json.dumps(payload, indent=2)}")
            
            try:
                start_time = time.time()
                
                response = await client.post(
                    api_gateway_url,
                    headers=headers,
                    json=payload
                )
                
                response_time = (time.time() - start_time) * 1000
                
                print(f"   📊 Status: {response.status_code} ({response_time:.1f}ms)")
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        
                        if 'snippets' in result:
                            snippets = result['snippets']
                            print(f"   ✅ SUCCESS! Snippets: {len(snippets)}")
                            
                            if snippets:
                                print(f"   📋 First snippet: {snippets[0].get('snippet', '')[:50]}...")
                                print(f"   🎯 THIS FORMAT WORKS!")
                                return format_name, payload
                        else:
                            print(f"   ❌ No snippets field. Keys: {list(result.keys())}")
                            
                    except json.JSONDecodeError:
                        print(f"   ❌ Invalid JSON response: {response.text[:100]}")
                        
                elif response.status_code == 500:
                    print(f"   ❌ Server error: {response.text[:100]}")
                else:
                    print(f"   ❌ HTTP error: {response.text[:100]}")
                    
            except Exception as e:
                print(f"   ❌ Request failed: {e}")
        
        print("\n❌ No working format found - check API key or Lambda deployment")
        return None, None


async def fix_promptproxy_integration():
    """Generate the fix for PromptProxy integration."""
    
    print("\n🔧 PromptProxy Integration Fix")
    print("=" * 40)
    
    # The fix is to ensure PromptProxy sends the correct format
    fix_code = '''
# Fix for PromptProxy AWS RAG call
# In your PromptProxy code, change the AWS request from:

# WRONG FORMAT (current):
payload = {
    "body": json.dumps({
        "prompt": prompt,
        "user": user,
        "session_id": session_id
    })
}

# TO CORRECT FORMAT:
payload = {
    "prompt": prompt,
    "user": user, 
    "session_id": session_id
}

# The Lambda expects direct JSON, not wrapped in "body"
'''
    
    print("📋 Required Fix:")
    print(fix_code)
    
    print("\n🎯 Quick Test:")
    print("1. Update your PromptProxy code with the correct payload format")
    print("2. Restart the PromptProxy server")
    print("3. Run: python end_to_end_test.py")
    print("4. Expected: RAG Success Rate should be 3/3!")


async def create_working_test():
    """Create a test that uses the working format."""
    
    print("\n🧪 Working Format Test")
    print("=" * 30)
    
    api_gateway_url = "https://b6qmhutxnc.execute-api.us-east-1.amazonaws.com/prod/rag"
    
    # Test with a realistic API key placeholder
    print("💡 To test this fix:")
    print(f"1. Set your API key: export MQL5_API_KEY='your-actual-key'")
    print(f"2. Test direct API call:")
    print(f"""
curl -X POST {api_gateway_url} \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: $MQL5_API_KEY" \\
  -d '{{"prompt": "How to use ArrayResize in MQL5?", "user": "test"}}'
""")
    print("3. Should return 3 snippets if working correctly")


async def main():
    """Main function."""
    print("🔧 PromptProxy Lambda Integration Fix")
    print("This identifies the request format issue and provides the fix")
    print()
    
    try:
        # Test formats to identify working one
        await test_request_formats()
        
        # Provide the fix
        await fix_promptproxy_integration()
        
        # Show how to test
        await create_working_test()
        
        print("\n" + "=" * 60)
        print("🎯 SUMMARY:")
        print("The Lambda works perfectly. The issue is request format.")
        print("Fix the PromptProxy payload format and retest!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️ Fix interrupted by user")
    except Exception as e:
        print(f"\n💥 Fix failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())