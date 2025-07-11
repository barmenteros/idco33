#!/usr/bin/env python3
"""
MQL5 RAG System Diagnostic Tool
Comprehensive diagnosis of system failures to identify root causes without speculation.

This tool systematically tests each component and provides detailed failure analysis.
"""

import asyncio
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
import httpx
import sys
from pathlib import Path

class MQL5DiagnosticTool:
    """Systematic diagnostic tool for MQL5 RAG system failures."""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.diagnostic_results = {}
        self.failure_details = []
        
    async def run_full_diagnosis(self):
        """Run comprehensive system diagnosis."""
        print("🔍 MQL5 RAG System Diagnostic Tool")
        print("=" * 60)
        print(f"Target Server: {self.server_url}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Phase 1: Basic connectivity
            await self._diagnose_connectivity(client)
            
            # Phase 2: Server endpoints analysis
            await self._diagnose_endpoints(client)
            
            # Phase 3: AWS integration deep dive
            await self._diagnose_aws_integration(client)
            
            # Phase 4: Circuit breaker state analysis
            await self._diagnose_circuit_breaker(client)
            
            # Phase 5: RAG failure root cause
            await self._diagnose_rag_failures(client)
            
            # Phase 6: Performance anomaly analysis
            await self._diagnose_performance_anomalies(client)
        
        # Generate diagnostic report
        self._generate_diagnostic_report()
    
    async def _diagnose_connectivity(self, client: httpx.AsyncClient):
        """Phase 1: Basic server connectivity diagnosis."""
        print("🔌 Phase 1: Connectivity Diagnosis")
        print("-" * 40)
        
        connectivity_tests = [
            ("TCP Connection", f"{self.server_url}/"),
            ("Health Endpoint", f"{self.server_url}/health"),
            ("Response Time", f"{self.server_url}/health"),
        ]
        
        connectivity_results = {}
        
        for test_name, url in connectivity_tests:
            try:
                start_time = time.time()
                response = await client.get(url)
                response_time = (time.time() - start_time) * 1000
                
                connectivity_results[test_name] = {
                    "success": True,
                    "status_code": response.status_code,
                    "response_time_ms": response_time,
                    "content_length": len(response.content) if response.content else 0
                }
                
                print(f"✅ {test_name}: {response.status_code} ({response_time:.1f}ms)")
                
                if test_name == "Health Endpoint" and response.status_code == 200:
                    try:
                        health_data = response.json()
                        print(f"   📊 Server Status: {health_data.get('status', 'unknown')}")
                        print(f"   📊 Version: {health_data.get('version', 'unknown')}")
                        print(f"   📊 Uptime: {health_data.get('uptime_seconds', 0):.1f}s")
                    except Exception as e:
                        print(f"   ⚠️ Health data parsing failed: {e}")
                
            except Exception as e:
                connectivity_results[test_name] = {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                print(f"❌ {test_name}: {type(e).__name__} - {e}")
                self.failure_details.append(f"Connectivity failure: {test_name} - {e}")
        
        self.diagnostic_results["connectivity"] = connectivity_results
    
    async def _diagnose_endpoints(self, client: httpx.AsyncClient):
        """Phase 2: Detailed endpoint analysis."""
        print("\n🔗 Phase 2: Endpoint Analysis")
        print("-" * 40)
        
        endpoints = [
            ("GET", "/health", "Health check endpoint"),
            ("GET", "/config", "Configuration endpoint"),
            ("GET", "/circuit", "Circuit breaker metrics"),
            ("GET", "/detect", "MQL5 detection endpoint", {"prompt": "test MQL5 prompt"}),
            ("POST", "/process", "Main processing endpoint", {"prompt": "test", "user": "diagnostic"}),
        ]
        
        endpoint_results = {}
        
        for method, path, description, *params in endpoints:
            test_params = params[0] if params else None
            endpoint_key = f"{method} {path}"
            
            try:
                url = f"{self.server_url}{path}"
                
                if method == "GET":
                    if test_params:
                        response = await client.get(url, params=test_params)
                    else:
                        response = await client.get(url)
                elif method == "POST":
                    response = await client.post(url, json=test_params)
                
                endpoint_results[endpoint_key] = {
                    "success": True,
                    "status_code": response.status_code,
                    "description": description,
                    "response_size": len(response.content) if response.content else 0
                }
                
                print(f"✅ {endpoint_key}: {response.status_code} - {description}")
                
                # Parse response for additional insights
                if response.status_code == 200 and response.content:
                    try:
                        data = response.json()
                        
                        if path == "/config":
                            aws_configured = data.get("aws_configured", False)
                            detector_available = data.get("detector_available", False)
                            print(f"   📊 AWS Configured: {aws_configured}")
                            print(f"   📊 Detector Available: {detector_available}")
                            
                            if not aws_configured:
                                self.failure_details.append("AWS not configured properly in /config endpoint")
                            if not detector_available:
                                self.failure_details.append("MQL5 detector not available in /config endpoint")
                        
                        elif path == "/circuit":
                            state = data.get("state", "unknown")
                            failure_count = data.get("failure_count", 0)
                            total_failures = data.get("total_failures", 0)
                            print(f"   📊 Circuit State: {state}")
                            print(f"   📊 Failure Count: {failure_count}")
                            print(f"   📊 Total Failures: {total_failures}")
                            
                            if state == "open":
                                self.failure_details.append(f"Circuit breaker is OPEN - {failure_count} consecutive failures")
                        
                        elif path == "/detect":
                            is_mql5 = data.get("is_mql5", False)
                            confidence = data.get("detection_details", {}).get("confidence", 0)
                            print(f"   📊 MQL5 Detected: {is_mql5} (confidence: {confidence})")
                        
                        elif path == "/process":
                            mql5_detected = data.get("mql5_detected", False)
                            augmented = data.get("augmented", False)
                            success = data.get("success", False)
                            print(f"   📊 Process Success: {success}")
                            print(f"   📊 MQL5 Detected: {mql5_detected}")
                            print(f"   📊 Augmented: {augmented}")
                            
                            # Check for metadata about failures
                            metadata = data.get("metadata", {})
                            rag_results = metadata.get("rag_results", {})
                            if "fallback_reason" in rag_results:
                                fallback_reason = rag_results["fallback_reason"]
                                print(f"   ⚠️ Fallback Reason: {fallback_reason}")
                                self.failure_details.append(f"RAG fallback in /process: {fallback_reason}")
                    
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️ JSON parsing failed: {e}")
                        endpoint_results[endpoint_key]["json_error"] = str(e)
                
                elif response.status_code != 200:
                    print(f"   ⚠️ Non-200 status code: {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"   ⚠️ Error details: {error_data}")
                    except:
                        print(f"   ⚠️ Response text: {response.text[:200]}")
                
            except Exception as e:
                endpoint_results[endpoint_key] = {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "description": description
                }
                print(f"❌ {endpoint_key}: {type(e).__name__} - {e}")
                self.failure_details.append(f"Endpoint failure: {endpoint_key} - {e}")
        
        self.diagnostic_results["endpoints"] = endpoint_results
    
    async def _diagnose_aws_integration(self, client: httpx.AsyncClient):
        """Phase 3: Deep dive into AWS integration issues."""
        print("\n☁️ Phase 3: AWS Integration Analysis")
        print("-" * 40)
        
        # Test raw AWS RAG call simulation
        test_payload = {
            "prompt": "How do I use ArrayResize() function in MQL5?",
            "user": "aws_diagnostic_test",
            "session_id": "aws_diag_session"
        }
        
        try:
            start_time = time.time()
            response = await client.post(f"{self.server_url}/process", json=test_payload)
            aws_call_time = (time.time() - start_time) * 1000
            
            print(f"📊 AWS RAG Test Call: {response.status_code} ({aws_call_time:.1f}ms)")
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract detailed AWS integration info
                metadata = result.get("metadata", {})
                rag_results = metadata.get("rag_results", {})
                
                print(f"📊 MQL5 Detected: {result.get('mql5_detected', False)}")
                print(f"📊 Augmented: {result.get('augmented', False)}")
                print(f"📊 Processing Time: {result.get('processing_time_ms', 0):.1f}ms")
                
                # AWS-specific diagnostics
                aws_diagnostics = {
                    "lambda_called": rag_results.get("lambda_called", False),
                    "lambda_success": rag_results.get("lambda_success", False),
                    "api_gateway_status": rag_results.get("api_gateway_status", None),
                    "snippets_count": rag_results.get("snippets_count", 0),
                    "fallback_reason": rag_results.get("fallback_reason", None),
                    "aws_error": rag_results.get("aws_error", None),
                    "timeout_occurred": rag_results.get("timeout_occurred", False)
                }
                
                for key, value in aws_diagnostics.items():
                    if value is not None:
                        print(f"📊 {key.replace('_', ' ').title()}: {value}")
                
                # Identify specific AWS failure patterns
                if not aws_diagnostics["lambda_called"]:
                    self.failure_details.append("AWS Lambda was never called - check API Gateway URL/credentials")
                elif aws_diagnostics["lambda_called"] and not aws_diagnostics["lambda_success"]:
                    if aws_diagnostics["timeout_occurred"]:
                        self.failure_details.append("AWS Lambda timeout - check Lambda performance/cold starts")
                    elif aws_diagnostics["aws_error"]:
                        self.failure_details.append(f"AWS error: {aws_diagnostics['aws_error']}")
                    else:
                        self.failure_details.append("AWS Lambda call failed for unknown reason")
                elif aws_diagnostics["lambda_success"] and aws_diagnostics["snippets_count"] == 0:
                    self.failure_details.append("AWS Lambda succeeded but returned no snippets - check DynamoDB/FAISS index")
                
                self.diagnostic_results["aws_integration"] = {
                    "call_successful": response.status_code == 200,
                    "response_time_ms": aws_call_time,
                    "diagnostics": aws_diagnostics
                }
            
        except Exception as e:
            print(f"❌ AWS Integration Test Failed: {type(e).__name__} - {e}")
            self.failure_details.append(f"AWS integration test failure: {e}")
            self.diagnostic_results["aws_integration"] = {
                "call_successful": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def _diagnose_circuit_breaker(self, client: httpx.AsyncClient):
        """Phase 4: Circuit breaker state analysis."""
        print("\n🛡️ Phase 4: Circuit Breaker Analysis")
        print("-" * 40)
        
        try:
            response = await client.get(f"{self.server_url}/circuit")
            
            if response.status_code == 200:
                circuit_data = response.json()
                
                state = circuit_data.get("state", "unknown")
                failure_count = circuit_data.get("failure_count", 0)
                success_count = circuit_data.get("success_count", 0)
                total_requests = circuit_data.get("total_requests", 0)
                total_failures = circuit_data.get("total_failures", 0)
                circuit_open_count = circuit_data.get("circuit_open_count", 0)
                avg_response_time = circuit_data.get("average_response_time_ms", 0)
                
                print(f"📊 Circuit State: {state}")
                print(f"📊 Current Failure Streak: {failure_count}")
                print(f"📊 Current Success Streak: {success_count}")
                print(f"📊 Total Requests: {total_requests}")
                print(f"📊 Total Failures: {total_failures}")
                print(f"📊 Times Circuit Opened: {circuit_open_count}")
                print(f"📊 Average Response Time: {avg_response_time:.2f}ms")
                
                # Calculate failure rate
                if total_requests > 0:
                    failure_rate = (total_failures / total_requests) * 100
                    print(f"📊 Overall Failure Rate: {failure_rate:.1f}%")
                else:
                    failure_rate = 0
                
                # Circuit breaker diagnostics
                circuit_diagnostics = {
                    "state": state,
                    "is_protecting": state == "open",
                    "high_failure_rate": failure_rate > 50,
                    "consecutive_failures": failure_count,
                    "has_opened": circuit_open_count > 0
                }
                
                # Identify circuit breaker issues
                if state == "open":
                    self.failure_details.append(f"Circuit breaker is OPEN - protecting against {failure_count} consecutive failures")
                elif failure_rate > 50:
                    self.failure_details.append(f"High failure rate detected: {failure_rate:.1f}% - circuit may open soon")
                
                self.diagnostic_results["circuit_breaker"] = {
                    "data_retrieved": True,
                    "diagnostics": circuit_diagnostics,
                    "raw_data": circuit_data
                }
                
            else:
                print(f"❌ Circuit endpoint returned {response.status_code}")
                self.failure_details.append(f"Circuit breaker endpoint failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Circuit Breaker Analysis Failed: {type(e).__name__} - {e}")
            self.failure_details.append(f"Circuit breaker analysis failure: {e}")
    
    async def _diagnose_rag_failures(self, client: httpx.AsyncClient):
        """Phase 5: Deep dive into RAG failure root causes."""
        print("\n🔍 Phase 5: RAG Failure Root Cause Analysis")
        print("-" * 40)
        
        # Test multiple MQL5 prompts to identify patterns
        test_prompts = [
            "How do I use ArrayResize() in MQL5?",
            "What is OnTick() function?",
            "MQL5 OrderSend() parameters",
            "Expert Advisor development in MQL5",
            "How to create custom indicators in MQL5?"
        ]
        
        rag_failure_patterns = {
            "always_fails": True,
            "timeout_issues": 0,
            "no_snippets": 0,
            "aws_errors": 0,
            "circuit_protection": 0,
            "unknown_failures": 0
        }
        
        for i, prompt in enumerate(test_prompts):
            try:
                payload = {
                    "prompt": prompt,
                    "user": f"rag_diagnostic_{i}",
                    "session_id": f"rag_diag_{i}"
                }
                
                response = await client.post(f"{self.server_url}/process", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    metadata = result.get("metadata", {})
                    rag_results = metadata.get("rag_results", {})
                    
                    lambda_success = rag_results.get("lambda_success", False)
                    fallback_reason = rag_results.get("fallback_reason", None)
                    
                    if lambda_success:
                        rag_failure_patterns["always_fails"] = False
                        print(f"✅ Prompt {i+1}: RAG SUCCESS")
                    else:
                        print(f"❌ Prompt {i+1}: RAG FAILED - {fallback_reason}")
                        
                        # Categorize failure type
                        if fallback_reason == "timeout":
                            rag_failure_patterns["timeout_issues"] += 1
                        elif fallback_reason == "no_snippets":
                            rag_failure_patterns["no_snippets"] += 1
                        elif "aws" in fallback_reason.lower() or "lambda" in fallback_reason.lower():
                            rag_failure_patterns["aws_errors"] += 1
                        elif "circuit" in fallback_reason.lower():
                            rag_failure_patterns["circuit_protection"] += 1
                        else:
                            rag_failure_patterns["unknown_failures"] += 1
                
            except Exception as e:
                print(f"❌ Prompt {i+1}: Exception - {e}")
        
        # Analyze failure patterns
        print(f"\n📊 RAG Failure Pattern Analysis:")
        print(f"   Always Fails: {rag_failure_patterns['always_fails']}")
        print(f"   Timeout Issues: {rag_failure_patterns['timeout_issues']}")
        print(f"   No Snippets Returned: {rag_failure_patterns['no_snippets']}")
        print(f"   AWS Errors: {rag_failure_patterns['aws_errors']}")
        print(f"   Circuit Protection: {rag_failure_patterns['circuit_protection']}")
        print(f"   Unknown Failures: {rag_failure_patterns['unknown_failures']}")
        
        # Identify primary failure cause
        if rag_failure_patterns["always_fails"]:
            if rag_failure_patterns["circuit_protection"] > 0:
                self.failure_details.append("PRIMARY ISSUE: Circuit breaker is protecting against AWS failures")
            elif rag_failure_patterns["timeout_issues"] > 0:
                self.failure_details.append("PRIMARY ISSUE: AWS Lambda timeouts - check Lambda performance")
            elif rag_failure_patterns["no_snippets"] > 0:
                self.failure_details.append("PRIMARY ISSUE: Lambda succeeds but returns no snippets - check DynamoDB/FAISS")
            elif rag_failure_patterns["aws_errors"] > 0:
                self.failure_details.append("PRIMARY ISSUE: AWS integration errors - check credentials/permissions")
        
        self.diagnostic_results["rag_failures"] = rag_failure_patterns
    
    async def _diagnose_performance_anomalies(self, client: httpx.AsyncClient):
        """Phase 6: Performance anomaly analysis."""
        print("\n⚡ Phase 6: Performance Anomaly Analysis")
        print("-" * 40)
        
        # The test shows very fast responses (0.0ms) which is suspicious
        performance_issues = []
        
        # Test response time distribution
        response_times = []
        
        for i in range(5):
            try:
                start_time = time.time()
                response = await client.get(f"{self.server_url}/health")
                response_time = (time.time() - start_time) * 1000
                response_times.append(response_time)
                
            except Exception as e:
                print(f"❌ Performance test {i+1} failed: {e}")
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            print(f"📊 Health Endpoint Response Times:")
            print(f"   Average: {avg_time:.2f}ms")
            print(f"   Min: {min_time:.2f}ms")
            print(f"   Max: {max_time:.2f}ms")
            print(f"   All times: {[f'{t:.1f}ms' for t in response_times]}")
            
            # Check for anomalies
            if min_time < 1.0:
                performance_issues.append("Suspiciously fast responses (<1ms) - possible measurement error")
            
            if max_time - min_time > 100:
                performance_issues.append("High response time variance - possible performance instability")
        
        # Test processing endpoint performance specifically
        try:
            start_time = time.time()
            response = await client.post(f"{self.server_url}/process", json={
                "prompt": "Test performance prompt for MQL5",
                "user": "perf_test"
            })
            process_time = (time.time() - start_time) * 1000
            
            print(f"📊 Process Endpoint Response Time: {process_time:.2f}ms")
            
            if response.status_code == 200:
                result = response.json()
                server_reported_time = result.get("processing_time_ms", 0)
                print(f"📊 Server Reported Processing Time: {server_reported_time:.2f}ms")
                
                # Check for time discrepancies
                time_diff = abs(process_time - server_reported_time)
                if time_diff > 100:
                    performance_issues.append(f"Time measurement discrepancy: {time_diff:.1f}ms difference")
        
        except Exception as e:
            performance_issues.append(f"Process endpoint performance test failed: {e}")
        
        if performance_issues:
            print("\n⚠️ Performance Issues Detected:")
            for issue in performance_issues:
                print(f"   • {issue}")
                self.failure_details.append(f"Performance issue: {issue}")
        
        self.diagnostic_results["performance"] = {
            "health_response_times": response_times,
            "issues": performance_issues
        }
    
    def _generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report with actionable recommendations."""
        print("\n" + "=" * 80)
        print("🏥 COMPREHENSIVE DIAGNOSTIC REPORT")
        print("=" * 80)
        
        # Summary of findings
        total_issues = len(self.failure_details)
        print(f"\n📋 SUMMARY:")
        print(f"   Total Issues Identified: {total_issues}")
        
        if total_issues == 0:
            print("   ✅ No significant issues detected")
            return
        
        # Categorize issues by severity
        critical_issues = []
        warning_issues = []
        
        for issue in self.failure_details:
            if any(keyword in issue.lower() for keyword in ["circuit", "aws", "timeout", "primary issue"]):
                critical_issues.append(issue)
            else:
                warning_issues.append(issue)
        
        print(f"   🚨 Critical Issues: {len(critical_issues)}")
        print(f"   ⚠️ Warnings: {len(warning_issues)}")
        
        # Display critical issues
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES (Fix These First):")
            for i, issue in enumerate(critical_issues, 1):
                print(f"   {i}. {issue}")
        
        # Display warnings
        if warning_issues:
            print(f"\n⚠️ WARNINGS:")
            for i, issue in enumerate(warning_issues, 1):
                print(f"   {i}. {issue}")
        
        # Generate specific recommendations
        print(f"\n🔧 RECOMMENDED ACTIONS:")
        
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Show diagnostic data summary
        print(f"\n📊 DIAGNOSTIC DATA SUMMARY:")
        for phase, data in self.diagnostic_results.items():
            if isinstance(data, dict):
                success_count = sum(1 for v in data.values() if isinstance(v, dict) and v.get("success", False))
                total_count = len([v for v in data.values() if isinstance(v, dict) and "success" in v])
                if total_count > 0:
                    print(f"   {phase.replace('_', ' ').title()}: {success_count}/{total_count} tests passed")
                else:
                    print(f"   {phase.replace('_', ' ').title()}: Data collected")
        
        print("\n" + "=" * 80)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate specific actionable recommendations based on diagnostic findings."""
        recommendations = []
        
        # Check for circuit breaker issues
        circuit_data = self.diagnostic_results.get("circuit_breaker", {})
        if circuit_data.get("diagnostics", {}).get("is_protecting", False):
            recommendations.append("URGENT: Reset circuit breaker by fixing underlying AWS issues and waiting for cooldown period")
        
        # Check for AWS integration issues
        aws_data = self.diagnostic_results.get("aws_integration", {})
        if not aws_data.get("call_successful", True):
            recommendations.append("Check AWS API Gateway URL and API key configuration")
            recommendations.append("Verify AWS Lambda function is deployed and accessible")
            recommendations.append("Test AWS connectivity manually: curl -X POST [API_GATEWAY_URL] -H 'x-api-key: [KEY]'")
        
        # Check for specific failure patterns
        rag_data = self.diagnostic_results.get("rag_failures", {})
        if rag_data.get("no_snippets", 0) > 0:
            recommendations.append("Check DynamoDB table for documentation snippets")
            recommendations.append("Verify FAISS index is properly loaded in Lambda")
            recommendations.append("Run document ingestion pipeline to populate vector store")
        
        if rag_data.get("timeout_issues", 0) > 0:
            recommendations.append("Optimize Lambda performance - check memory allocation")
            recommendations.append("Implement Lambda warming to reduce cold starts")
            recommendations.append("Check CloudWatch logs for Lambda timeout details")
        
        # Check connectivity issues
        connectivity_data = self.diagnostic_results.get("connectivity", {})
        failed_connections = [k for k, v in connectivity_data.items() if not v.get("success", False)]
        if failed_connections:
            recommendations.append("Fix basic connectivity issues before testing RAG functionality")
        
        # Default recommendations if no specific issues identified
        if not recommendations:
            recommendations.append("Review server logs for detailed error messages")
            recommendations.append("Check AWS CloudWatch logs for Lambda execution details")
            recommendations.append("Verify all environment variables and configuration files")
        
        return recommendations


async def main():
    """Run the diagnostic tool."""
    print("🏥 MQL5 RAG System Diagnostic Tool")
    print("This tool will systematically identify the root cause of system failures.")
    print()
    
    # Check if server is reachable
    server_url = "http://localhost:8080"
    
    diagnostic_tool = MQL5DiagnosticTool(server_url)
    
    try:
        await diagnostic_tool.run_full_diagnosis()
    except KeyboardInterrupt:
        print("\n⏹️ Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n💥 Diagnostic tool failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())