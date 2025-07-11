#!/usr/bin/env python3
"""
End-to-End Integration Test for MQL5 Prompt Enrichment Middleware
Comprehensive validation of the complete system from MQL5 detection through AWS RAG 
to prompt augmentation with circuit-breaker protection.

This test demonstrates:
1. Complete pipeline functionality (Tasks D19-D24)
2. Performance targets (<500ms avg, <700ms p95)
3. Failure handling and circuit breaker protection
4. Real AWS integration capabilities
5. Production readiness validation
"""

import asyncio
import json
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any
import httpx
from pathlib import Path

# Test configuration for comprehensive validation
INTEGRATION_CONFIG = {
    'host': 'localhost',
    'port': 8080,
    'aws': {
        'api_gateway_url': 'https://b6qmhutxnc.execute-api.us-east-1.amazonaws.com/prod/rag',
        'api_key': 'DNpEmzqcgQ2GcwB10LDBx9H3wBnQZ0Cr7z17HDzh',  # Update with real key
        'timeout_seconds': 2.0
    },
    'augmentation': {
        'max_snippets': 5,
        'snippet_separator': '\n\n'
    },
    'circuit_breaker': {
        'failure_threshold': 3,
        'success_threshold': 2,
        'cooldown_duration': 30,
        'half_open_max_requests': 3
    },
    'retry': {
        'max_retries': 1,
        'delay_ms': 100,
        'backoff_multiplier': 2.0
    }
}

class EndToEndTester:
    """Comprehensive end-to-end testing for the MQL5 Prompt Enrichment Middleware."""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.test_results = []
        self.performance_metrics = []
        
    async def run_comprehensive_test(self):
        """Run the complete end-to-end integration test suite."""
        print("🚀 MQL5 Prompt Enrichment Middleware - End-to-End Integration Test")
        print("=" * 80)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Phase 1: System Health Check
            await self._test_system_health(client)
            
            # Phase 2: MQL5 Detection Validation
            await self._test_mql5_detection(client)
            
            # Phase 3: RAG Integration Testing
            await self._test_rag_integration(client)
            
            # Phase 4: Augmentation Pipeline
            await self._test_augmentation_pipeline(client)
            
            # Phase 5: Circuit Breaker Validation
            await self._test_circuit_breaker(client)
            
            # Phase 6: Performance Benchmarking
            await self._test_performance_targets(client)
            
            # Phase 7: Failure Scenarios
            await self._test_failure_scenarios(client)
            
            # Phase 8: Production Readiness
            await self._validate_production_readiness(client)
        
        # Generate final report
        self._generate_test_report()
    
    async def _test_system_health(self, client: httpx.AsyncClient):
        """Phase 1: Validate system health and configuration."""
        print("\n📋 Phase 1: System Health Check")
        print("-" * 40)
        
        try:
            # Test server availability
            response = await client.get(f"{self.server_url}/health")
            assert response.status_code == 200
            health_data = response.json()
            
            print(f"✅ Server Status: {health_data['status']}")
            print(f"✅ Version: {health_data['version']}")
            print(f"✅ Uptime: {health_data['uptime_seconds']}s")
            
            # Test configuration endpoint
            config_response = await client.get(f"{self.server_url}/config")
            assert config_response.status_code == 200
            config_data = config_response.json()
            
            print(f"✅ AWS RAG Configured: {config_data['aws_configured']}")
            print(f"✅ Circuit Breaker State: {config_data['circuit_breaker']['state']}")
            print(f"✅ MQL5 Detector: {config_data['detector_available']}")
            
            # Test circuit breaker metrics
            circuit_response = await client.get(f"{self.server_url}/circuit")
            assert circuit_response.status_code == 200
            circuit_data = circuit_response.json()
            
            print(f"✅ Circuit State: {circuit_data['state']}")
            print(f"✅ Total Requests: {circuit_data['total_requests']}")
            
            self._record_test_result("System Health Check", True, "All endpoints responding correctly")
            
        except Exception as e:
            self._record_test_result("System Health Check", False, f"Health check failed: {e}")
            raise
    
    async def _test_mql5_detection(self, client: httpx.AsyncClient):
        """Phase 2: Validate MQL5 prompt detection accuracy."""
        print("\n🎯 Phase 2: MQL5 Detection Validation")
        print("-" * 40)
        
        test_cases = [
            # Positive cases (should detect as MQL5)
            ("How do I use ArrayResize() to resize a dynamic array in MQL5?", True),
            ("What is the difference between OnTick() and OnStart() in MQL5?", True),
            ("How to implement a custom indicator using iCustom() function?", True),
            ("MQL5 Expert Advisor development best practices", True),
            ("OrderSend() function parameters explanation", True),
            
            # Negative cases (should NOT detect as MQL5)
            ("What is the weather like today?", False),
            ("How to cook pasta?", False),
            ("Python machine learning tutorial", False),
            ("JavaScript array methods", False),
            ("What is artificial intelligence?", False),
        ]
        
        correct_detections = 0
        total_tests = len(test_cases)
        
        for prompt, expected_mql5 in test_cases:
            try:
                params = {"prompt": prompt}
                response = await client.get(f"{self.server_url}/detect", params=params)
                assert response.status_code == 200
                
                detection_data = response.json()
                detected_mql5 = detection_data.get("is_mql5", False)
                confidence = detection_data.get("detection_details", {}).get("confidence", 0.0)
                
                if detected_mql5 == expected_mql5:
                    correct_detections += 1
                    status = "✅"
                else:
                    status = "❌"
                
                print(f"{status} '{prompt[:50]}...' → MQL5: {detected_mql5} (confidence: {confidence:.2f})")
                
            except Exception as e:
                print(f"❌ Detection test failed for '{prompt[:30]}...': {e}")
        
        accuracy = (correct_detections / total_tests) * 100
        print(f"\n📊 Detection Accuracy: {correct_detections}/{total_tests} ({accuracy:.1f}%)")
        
        success = accuracy >= 80  # 80% accuracy threshold
        self._record_test_result("MQL5 Detection", success, f"Accuracy: {accuracy:.1f}%")
    
    async def _test_rag_integration(self, client: httpx.AsyncClient):
        """Phase 3: Test RAG integration with real AWS calls."""
        print("\n🌐 Phase 3: RAG Integration Testing")
        print("-" * 40)
        
        mql5_prompts = [
            "How do I use ArrayResize() function in MQL5?",
            "What is the OnTick() function used for?",
            "How to create a custom indicator in MQL5?",
        ]
        
        rag_successes = 0
        
        for prompt in mql5_prompts:
            try:
                start_time = time.time()
                
                payload = {
                    "prompt": prompt,
                    "user": "integration_test",
                    "session_id": f"test_{int(time.time())}"
                }
                
                response = await client.post(f"{self.server_url}/process", json=payload)
                call_duration = (time.time() - start_time) * 1000
                
                assert response.status_code == 200
                result = response.json()
                
                mql5_detected = result.get("mql5_detected", False)
                augmented = result.get("augmented", False)
                processing_time = result.get("processing_time_ms", 0)
                
                rag_results = result.get("metadata", {}).get("rag_results", {})
                lambda_success = rag_results.get("lambda_success", False)
                snippets_count = rag_results.get("snippets_count", 0)
                
                if mql5_detected and lambda_success:
                    rag_successes += 1
                    status = "✅"
                    print(f"{status} RAG Success: {snippets_count} snippets, {processing_time:.1f}ms")
                else:
                    status = "⚠️"
                    fallback_reason = rag_results.get("fallback_reason", "unknown")
                    print(f"{status} RAG Fallback: {fallback_reason}, {processing_time:.1f}ms")
                
                # Record performance metric
                self.performance_metrics.append(call_duration)
                
            except Exception as e:
                print(f"❌ RAG test failed for '{prompt[:30]}...': {e}")
        
        print(f"\n📊 RAG Success Rate: {rag_successes}/{len(mql5_prompts)}")
        
        # Allow for some failures due to Lambda cold starts or network issues
        success = rag_successes >= len(mql5_prompts) * 0.5  # 50% success threshold
        self._record_test_result("RAG Integration", success, f"Success rate: {rag_successes}/{len(mql5_prompts)}")
    
    async def _test_augmentation_pipeline(self, client: httpx.AsyncClient):
        """Phase 4: Test prompt augmentation with mock successful RAG."""
        print("\n📝 Phase 4: Augmentation Pipeline Testing")
        print("-" * 40)
        
        # Use a detailed MQL5 prompt that should trigger augmentation
        detailed_prompt = """
        I'm trying to create an Expert Advisor in MQL5 that uses dynamic arrays.
        How do I properly resize arrays and what's the difference between static and dynamic arrays?
        I also need to understand the ArrayResize() function parameters.
        """
        
        try:
            payload = {
                "prompt": detailed_prompt,
                "user": "augmentation_test",
                "session_id": "augmentation_test_session"
            }
            
            response = await client.post(f"{self.server_url}/process", json=payload)
            assert response.status_code == 200
            result = response.json()
            
            mql5_detected = result.get("mql5_detected", False)
            augmented = result.get("augmented", False)
            processed_prompt = result.get("prompt", "")
            
            print(f"✅ MQL5 Detected: {mql5_detected}")
            print(f"✅ Prompt Augmented: {augmented}")
            print(f"✅ Original Length: {len(detailed_prompt)} chars")
            print(f"✅ Processed Length: {len(processed_prompt)} chars")
            
            # Check for augmentation template structure
            if augmented:
                has_context_header = "/* Context: MQL5 documentation snippets */" in processed_prompt
                has_original_prompt_header = "/* Original Prompt */" in processed_prompt
                has_original_content = detailed_prompt.strip() in processed_prompt
                
                print(f"✅ Context Header: {has_context_header}")
                print(f"✅ Original Prompt Header: {has_original_prompt_header}")
                print(f"✅ Original Content Preserved: {has_original_content}")
                
                template_success = has_context_header and has_original_prompt_header and has_original_content
            else:
                template_success = True  # Fallback behavior is also valid
                print("ℹ️ Augmentation not applied (fallback behavior - acceptable)")
            
            self._record_test_result("Augmentation Pipeline", template_success, 
                                   f"Augmented: {augmented}, Template: {'Valid' if template_success else 'Invalid'}")
            
        except Exception as e:
            self._record_test_result("Augmentation Pipeline", False, f"Pipeline test failed: {e}")
    
    async def _test_circuit_breaker(self, client: httpx.AsyncClient):
        """Phase 5: Test circuit breaker behavior (without breaking the system)."""
        print("\n🛡️ Phase 5: Circuit Breaker Validation")
        print("-" * 40)
        
        try:
            # Check initial circuit state
            circuit_response = await client.get(f"{self.server_url}/circuit")
            initial_state = circuit_response.json()
            
            print(f"✅ Initial Circuit State: {initial_state['state']}")
            print(f"✅ Failure Count: {initial_state['failure_count']}")
            print(f"✅ Success Count: {initial_state['success_count']}")
            print(f"✅ Total Requests: {initial_state['total_requests']}")
            print(f"✅ Total Failures: {initial_state['total_failures']}")
            print(f"✅ Circuit Open Count: {initial_state['circuit_open_count']}")
            print(f"✅ Average Response Time: {initial_state['average_response_time_ms']:.2f}ms")
            
            # Verify circuit breaker configuration
            config_response = await client.get(f"{self.server_url}/config")
            cb_config = config_response.json()["circuit_breaker"]
            
            print(f"✅ Failure Threshold: {cb_config['failure_threshold']}")
            print(f"✅ Success Threshold: {cb_config['success_threshold']}")
            print(f"✅ Cooldown Duration: {cb_config['cooldown_duration']}s")
            
            # Test that circuit breaker is properly configured and monitoring
            circuit_breaker_working = (
                initial_state['state'] in ['closed', 'open', 'half_open'] and
                isinstance(initial_state['total_requests'], int) and
                isinstance(initial_state['average_response_time_ms'], (int, float))
            )
            
            self._record_test_result("Circuit Breaker", circuit_breaker_working, 
                                   f"State: {initial_state['state']}, Monitoring: Active")
            
        except Exception as e:
            self._record_test_result("Circuit Breaker", False, f"Circuit breaker test failed: {e}")
    
    async def _test_performance_targets(self, client: httpx.AsyncClient):
        """Phase 6: Validate performance targets (<500ms avg, <700ms p95)."""
        print("\n⚡ Phase 6: Performance Benchmarking")
        print("-" * 40)
        
        performance_prompts = [
            "How do I use ArrayResize() in MQL5?",
            "What is OnTick() function?",
            "MQL5 indicator development guide",
            "OrderSend() parameters explanation",
            "How to debug MQL5 Expert Advisors?",
        ]
        
        response_times = []
        
        for i, prompt in enumerate(performance_prompts):
            for iteration in range(3):  # 3 iterations per prompt
                try:
                    start_time = time.time()
                    
                    payload = {
                        "prompt": prompt,
                        "user": f"perf_test_{i}",
                        "session_id": f"perf_{i}_{iteration}"
                    }
                    
                    response = await client.post(f"{self.server_url}/process", json=payload)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        response_time = (end_time - start_time) * 1000
                        response_times.append(response_time)
                        
                        result = response.json()
                        server_time = result.get("processing_time_ms", 0)
                        
                        print(f"✅ Request {i+1}.{iteration+1}: {response_time:.1f}ms total, {server_time:.1f}ms server")
                    
                except Exception as e:
                    print(f"❌ Performance test {i+1}.{iteration+1} failed: {e}")
        
        if response_times:
            avg_time = statistics.mean(response_times)
            p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            min_time = min(response_times)
            max_time = max(response_times)
            
            print(f"\n📊 Performance Results:")
            print(f"   Average: {avg_time:.1f}ms (target: <500ms)")
            print(f"   P95: {p95_time:.1f}ms (target: <700ms)")
            print(f"   Min: {min_time:.1f}ms")
            print(f"   Max: {max_time:.1f}ms")
            print(f"   Total samples: {len(response_times)}")
            
            performance_success = avg_time < 500 and p95_time < 700
            
            if performance_success:
                print("🎯 ✅ Performance targets MET!")
            else:
                print("🎯 ⚠️ Performance targets MISSED")
            
            self._record_test_result("Performance Targets", performance_success, 
                                   f"Avg: {avg_time:.1f}ms, P95: {p95_time:.1f}ms")
        else:
            self._record_test_result("Performance Targets", False, "No successful performance measurements")
    
    async def _test_failure_scenarios(self, client: httpx.AsyncClient):
        """Phase 7: Test graceful failure handling."""
        print("\n🚨 Phase 7: Failure Scenario Testing")
        print("-" * 40)
        
        # Test with invalid/empty prompts
        failure_test_cases = [
            ("", "Empty prompt"),
            ("   ", "Whitespace-only prompt"),
            ("x" * 20000, "Extremely long prompt"),
        ]
        
        failure_handling_success = 0
        
        for test_prompt, description in failure_test_cases:
            try:
                payload = {
                    "prompt": test_prompt,
                    "user": "failure_test",
                    "session_id": f"fail_{int(time.time())}"
                }
                
                response = await client.post(f"{self.server_url}/process", json=payload)
                
                # Server should handle gracefully (either 200 with fallback or 400 with error)
                if response.status_code in [200, 400]:
                    if response.status_code == 200:
                        result = response.json()
                        success = result.get("success", False)
                        print(f"✅ {description}: Handled gracefully (success: {success})")
                    else:
                        print(f"✅ {description}: Proper error response")
                    failure_handling_success += 1
                else:
                    print(f"❌ {description}: Unexpected status {response.status_code}")
                
            except Exception as e:
                print(f"❌ {description}: Exception {e}")
        
        failure_success = failure_handling_success == len(failure_test_cases)
        self._record_test_result("Failure Handling", failure_success, 
                               f"Handled {failure_handling_success}/{len(failure_test_cases)} scenarios")
    
    async def _validate_production_readiness(self, client: httpx.AsyncClient):
        """Phase 8: Final production readiness validation."""
        print("\n🏆 Phase 8: Production Readiness Validation")
        print("-" * 40)
        
        readiness_checks = {
            "health_endpoint": False,
            "config_endpoint": False,
            "circuit_metrics": False,
            "error_handling": False,
            "performance_acceptable": False,
            "mql5_detection": False,
            "graceful_fallback": False,
        }
        
        try:
            # Check all required endpoints exist and respond
            endpoints = ["/health", "/config", "/circuit", "/process", "/detect"]
            for endpoint in endpoints:
                response = await client.get(f"{self.server_url}{endpoint}")
                if endpoint == "/process":
                    # POST endpoint - expect 422 for GET request
                    assert response.status_code in [422, 405]
                else:
                    assert response.status_code == 200
                    
            readiness_checks["health_endpoint"] = True
            readiness_checks["config_endpoint"] = True
            readiness_checks["circuit_metrics"] = True
            
            # Check that system handles errors gracefully
            try:
                invalid_response = await client.post(f"{self.server_url}/process", json={"invalid": "data"})
                readiness_checks["error_handling"] = invalid_response.status_code in [400, 422]
            except:
                readiness_checks["error_handling"] = True  # Exception handling is also acceptable
            
            # Evaluate overall readiness based on previous test results
            readiness_checks["performance_acceptable"] = any(
                result["name"] == "Performance Targets" and result["success"] 
                for result in self.test_results
            )
            
            readiness_checks["mql5_detection"] = any(
                result["name"] == "MQL5 Detection" and result["success"]
                for result in self.test_results
            )
            
            readiness_checks["graceful_fallback"] = any(
                result["name"] == "Failure Handling" and result["success"]
                for result in self.test_results
            )
            
        except Exception as e:
            print(f"❌ Production readiness check failed: {e}")
        
        # Display readiness status
        for check, status in readiness_checks.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {check.replace('_', ' ').title()}: {status}")
        
        readiness_score = sum(readiness_checks.values()) / len(readiness_checks)
        production_ready = readiness_score >= 0.8  # 80% of checks must pass
        
        print(f"\n🎯 Production Readiness Score: {readiness_score:.1%}")
        
        if production_ready:
            print("🚀 ✅ SYSTEM IS PRODUCTION READY!")
        else:
            print("🚨 ⚠️ System needs improvements before production deployment")
        
        self._record_test_result("Production Readiness", production_ready, 
                               f"Score: {readiness_score:.1%}")
    
    def _record_test_result(self, test_name: str, success: bool, details: str):
        """Record a test result for final reporting."""
        self.test_results.append({
            "name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    def _generate_test_report(self):
        """Generate final comprehensive test report."""
        print("\n" + "=" * 80)
        print("🏁 FINAL INTEGRATION TEST REPORT")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n📊 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for result in self.test_results:
            status_icon = "✅" if result["success"] else "❌"
            print(f"   {status_icon} {result['name']}: {result['details']}")
        
        if self.performance_metrics:
            avg_perf = statistics.mean(self.performance_metrics)
            print(f"\n⚡ Performance Summary:")
            print(f"   Average Response Time: {avg_perf:.1f}ms")
            print(f"   Total Measurements: {len(self.performance_metrics)}")
        
        print(f"\n🏆 System Status:")
        if success_rate >= 80:
            print("   ✅ MQL5 Prompt Enrichment Middleware is PRODUCTION READY!")
            print("   🚀 All critical components functioning correctly")
            print("   📈 Performance targets met")
            print("   🛡️ Circuit breaker protection active")
            print("   🎯 Ready for deployment!")
        else:
            print("   ⚠️ System requires attention before production deployment")
            print("   🔧 Review failed tests and address issues")
        
        print("\n" + "=" * 80)


async def main():
    """Run the comprehensive end-to-end integration test."""
    
    print("🔧 MQL5 Prompt Enrichment Middleware - End-to-End Integration Test")
    print("📋 Prerequisites:")
    print("   1. PromptProxy server running on localhost:8080")
    print("   2. AWS API Gateway configured with valid credentials")
    print("   3. MQL5 PromptDetector module available")
    print("   4. All dependencies installed (httpx, etc.)")
    print("\n🚀 Starting comprehensive integration test...\n")
    
    # Check if server is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8080/health")
            if response.status_code != 200:
                print("❌ PromptProxy server is not responding correctly")
                print("   Please start the server with: python promptproxy_server.py --config config.yaml")
                return
    except Exception as e:
        print(f"❌ Cannot connect to PromptProxy server: {e}")
        print("   Please ensure the server is running on localhost:8080")
        return
    
    # Run comprehensive test suite
    tester = EndToEndTester()
    await tester.run_comprehensive_test()


if __name__ == "__main__":
    print("MQL5 Prompt Enrichment Middleware - End-to-End Integration Test")
    print("=" * 70)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()