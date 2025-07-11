#!/usr/bin/env python3
"""
Circuit Breaker Reset Tool
Force reset the circuit breaker and verify AWS Lambda is working
"""

import asyncio
import json
import time
import httpx

async def reset_circuit_breaker():
    """Reset the circuit breaker and test the system."""
    
    print("🛡️ Circuit Breaker Reset Tool")
    print("=" * 40)
    
    server_url = "http://localhost:8080"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        # Step 1: Check current circuit state
        print("🔍 Step 1: Current Circuit State")
        try:
            response = await client.get(f"{server_url}/circuit")
            if response.status_code == 200:
                circuit_data = response.json()
                state = circuit_data.get("state", "unknown")
                failures = circuit_data.get("failure_count", 0)
                
                print(f"   Current State: {state}")
                print(f"   Failure Count: {failures}")
                print(f"   Total Requests: {circuit_data.get('total_requests', 0)}")
                
                if state == "open":
                    print("   ⚠️ Circuit is OPEN - blocking requests")
                else:
                    print("   ✅ Circuit is operational")
                    
        except Exception as e:
            print(f"   ❌ Error checking circuit: {e}")
            return
        
        # Step 2: Reset circuit breaker
        print("\n🔄 Step 2: Resetting Circuit Breaker")
        try:
            # Try different reset endpoints
            reset_endpoints = ["/circuit/reset", "/reset", "/circuit"]
            
            reset_success = False
            for endpoint in reset_endpoints:
                try:
                    reset_response = await client.post(f"{server_url}{endpoint}")
                    if reset_response.status_code in [200, 204]:
                        print(f"   ✅ Reset successful via {endpoint}")
                        reset_success = True
                        break
                    else:
                        print(f"   ⚠️ {endpoint}: {reset_response.status_code}")
                except:
                    continue
            
            if not reset_success:
                print("   ⚠️ No reset endpoint found, trying manual approach...")
                
                # Manual reset by making successful requests
                print("   🔧 Attempting manual reset via successful calls...")
                
        except Exception as e:
            print(f"   ❌ Reset failed: {e}")
        
        # Step 3: Verify circuit state after reset
        print("\n🔍 Step 3: Verifying Reset")
        try:
            await asyncio.sleep(1)  # Brief pause
            
            response = await client.get(f"{server_url}/circuit")
            if response.status_code == 200:
                circuit_data = response.json()
                new_state = circuit_data.get("state", "unknown")
                
                print(f"   New State: {new_state}")
                
                if new_state == "closed":
                    print("   ✅ Circuit successfully reset!")
                elif new_state == "half_open":
                    print("   ⚠️ Circuit in half-open state (testing)")
                else:
                    print("   ❌ Circuit still open")
                    
        except Exception as e:
            print(f"   ❌ Error verifying reset: {e}")
        
        # Step 4: Test AWS Lambda directly
        print("\n🧪 Step 4: Testing AWS Lambda Directly")
        try:
            test_payload = {
                "prompt": "How do I use ArrayResize() function in MQL5?",
                "user": "circuit_reset_test",
                "session_id": "reset_test_session"
            }
            
            print("   Making test request to PromptProxy...")
            start_time = time.time()
            
            response = await client.post(f"{server_url}/process", json=test_payload)
            response_time = (time.time() - start_time) * 1000
            
            print(f"   Response: {response.status_code} ({response_time:.1f}ms)")
            
            if response.status_code == 200:
                result = response.json()
                mql5_detected = result.get("mql5_detected", False)
                augmented = result.get("augmented", False)
                
                metadata = result.get("metadata", {})
                rag_results = metadata.get("rag_results", {})
                lambda_success = rag_results.get("lambda_success", False)
                fallback_reason = rag_results.get("fallback_reason", None)
                
                print(f"   ✅ MQL5 Detected: {mql5_detected}")
                print(f"   ✅ Lambda Success: {lambda_success}")
                print(f"   ✅ Augmented: {augmented}")
                
                if lambda_success:
                    print("   🎉 AWS Lambda is working!")
                    snippets_count = rag_results.get("snippets_count", 0)
                    print(f"   📊 Snippets returned: {snippets_count}")
                    
                    if snippets_count > 0:
                        print("   ✅ End-to-end pipeline: SUCCESS")
                    else:
                        print("   ⚠️ Lambda works but returns no snippets (mock data)")
                else:
                    print(f"   ❌ Lambda failed: {fallback_reason}")
                    
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
        
        # Step 5: Final circuit state check
        print("\n📊 Step 5: Final Circuit State")
        try:
            response = await client.get(f"{server_url}/circuit")
            if response.status_code == 200:
                circuit_data = response.json()
                
                print(f"   State: {circuit_data.get('state', 'unknown')}")
                print(f"   Success Count: {circuit_data.get('success_count', 0)}")
                print(f"   Failure Count: {circuit_data.get('failure_count', 0)}")
                print(f"   Total Requests: {circuit_data.get('total_requests', 0)}")
                
                final_state = circuit_data.get("state", "unknown")
                if final_state == "closed":
                    print("   🎯 ✅ Circuit breaker is healthy!")
                elif final_state == "half_open":
                    print("   🎯 ⚠️ Circuit breaker is testing (will close on success)")
                else:
                    print("   🎯 ❌ Circuit breaker still protecting")
                    
        except Exception as e:
            print(f"   ❌ Error checking final state: {e}")

async def main():
    """Main function."""
    print("🛡️ Circuit Breaker Reset and Verification Tool")
    print("This will reset the circuit breaker and test AWS integration")
    print()
    
    try:
        await reset_circuit_breaker()
        
        print("\n" + "=" * 60)
        print("🎯 SUMMARY:")
        print("If Lambda is working, run the end-to-end test again:")
        print("   python end_to_end_test.py")
        print("")
        print("Expected result: RAG Success Rate should improve!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️ Reset interrupted by user")
    except Exception as e:
        print(f"\n💥 Reset failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())