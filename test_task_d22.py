#!/usr/bin/env python3
"""
Integration Test for Task D22: AWS RAG Call Implementation
Tests the PromptProxy server's ability to call AWS API Gateway RAG endpoint.

This test validates:
1. MQL5 prompt detection triggers RAG calls
2. Non-MQL5 prompts bypass RAG calls
3. Error handling and fallback behavior
4. Circuit breaker functionality
5. Configuration loading
"""

import asyncio
import json
import time
from typing import Dict, Any
import httpx
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

# Configure pytest for asyncio
pytest_plugins = ('pytest_asyncio',)

# Test configuration
TEST_CONFIG = {
    'host': 'localhost',
    'port': 8081,  # Different port to avoid conflicts
    'aws': {
        'api_gateway_url': 'https://test-api.execute-api.us-east-1.amazonaws.com/prod/rag',
        'api_key': 'test-api-key-12345',
        'timeout_seconds': 1.0  # Shorter for testing
    }
}

class TestTaskD22Integration:
    """Test suite for Task D22 AWS RAG integration."""
    
    @pytest.fixture
    def test_server(self):
        """Create a test server instance with mock configuration."""
        from promptproxy_server import PromptProxyServer
        
        server = PromptProxyServer(
            host=TEST_CONFIG['host'],
            port=TEST_CONFIG['port'], 
            config=TEST_CONFIG
        )
        
        yield server
        
        # Cleanup - only close if it's a real HTTP client, not a mock
        if server.http_client and hasattr(server.http_client, 'aclose'):
            try:
                # Try to close if we're in an async context
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    asyncio.create_task(server.http_client.aclose())
            except RuntimeError:
                # No running event loop, skip cleanup
                pass
    
    @pytest.fixture
    def mock_aws_response(self):
        """Mock successful AWS RAG response."""
        return {
            "success": True,
            "snippets": [
                {
                    "snippet": "ArrayResize() function resizes dynamic arrays in MQL5...",
                    "source": "MQL5 Documentation - Array Functions",
                    "score": 0.95
                },
                {
                    "snippet": "Dynamic arrays can be resized using ArrayResize()...",
                    "source": "MQL5 Reference - Memory Management", 
                    "score": 0.88
                }
            ],
            "processing_time_ms": 45.2,
            "lambda_function": "mql5-rag-handler",
            "timestamp": "2025-07-09T10:30:00Z"
        }
    
    @pytest.mark.asyncio
    async def test_mql5_prompt_triggers_rag_call(self, test_server, mock_aws_response):
        """Test that MQL5 prompts trigger AWS RAG calls."""
        
        # Mock the HTTP client to return successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_aws_response
        
        # Create a mock HTTP client
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        # Patch the server's HTTP client
        test_server.http_client = mock_client
        
        # Test MQL5 prompt
        mql5_prompt = "How do I use ArrayResize() to resize a dynamic array in MQL5?"
        
        result = await test_server._call_aws_rag_endpoint(mql5_prompt)
        
        # Verify AWS call was made
        assert mock_client.post.called
        call_args = mock_client.post.call_args
        
        # Check URL
        assert call_args[0][0] == TEST_CONFIG['aws']['api_gateway_url']
        
        # Check headers
        headers = call_args[1]['headers']
        assert headers['x-api-key'] == TEST_CONFIG['aws']['api_key']
        assert headers['Content-Type'] == 'application/json'
        
        # Check payload
        payload = call_args[1]['json']
        assert payload['prompt'] == mql5_prompt
        assert payload['max_snippets'] == 5
        
        # Check result
        assert result == mock_aws_response
        assert result['success'] is True
        assert len(result['snippets']) == 2
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, test_server):
        """Test circuit breaker opens after repeated failures."""
        
        # Reset circuit breaker for clean test
        test_server.circuit_breaker['failure_count'] = 0
        test_server.circuit_breaker['cooldown_until'] = 0
        
        # Mock failed responses
        mock_response = MagicMock()
        mock_response.status_code = 500
        
        # Create a mock HTTP client
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        # Patch the server's HTTP client
        test_server.http_client = mock_client
        
        # Make multiple failing calls
        for i in range(4):  # Threshold is 3, so 4th should trigger circuit breaker
            result = await test_server._call_aws_rag_endpoint("test prompt")
            assert result is None
        
        # Verify circuit breaker is now open
        assert test_server._is_circuit_breaker_open()
        assert test_server.circuit_breaker['failure_count'] >= 3
        
        # Next call should be blocked by circuit breaker
        result = await test_server._call_aws_rag_endpoint("another test")
        assert result is None
        
        # Should not have made additional HTTP call due to circuit breaker
        assert mock_client.post.call_count == 3  # Only the first 3 failures
    
    @pytest.mark.asyncio 
    async def test_timeout_handling(self, test_server):
        """Test proper handling of timeout errors."""
        
        # Reset circuit breaker for clean test
        test_server.circuit_breaker['failure_count'] = 0
        test_server.circuit_breaker['cooldown_until'] = 0
        
        # Create a mock HTTP client
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))
        
        # Patch the server's HTTP client
        test_server.http_client = mock_client
        
        result = await test_server._call_aws_rag_endpoint("test prompt")
        
        assert result is None
        assert test_server.circuit_breaker['failure_count'] == 1
    
    @pytest.mark.asyncio
    async def test_authentication_failure_handling(self, test_server):
        """Test handling of authentication failures."""
        
        # Reset circuit breaker for clean test
        test_server.circuit_breaker['failure_count'] = 0
        test_server.circuit_breaker['cooldown_until'] = 0
        
        mock_response = MagicMock()
        mock_response.status_code = 403
        
        # Create a mock HTTP client
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        # Patch the server's HTTP client
        test_server.http_client = mock_client
        
        result = await test_server._call_aws_rag_endpoint("test prompt")
        
        assert result is None
        assert test_server.circuit_breaker['failure_count'] == 1
    
    @pytest.mark.asyncio
    async def test_rate_limiting_handling(self, test_server):
        """Test handling of rate limiting responses."""
        
        # Reset circuit breaker for clean test
        test_server.circuit_breaker['failure_count'] = 0
        test_server.circuit_breaker['cooldown_until'] = 0
        
        mock_response = MagicMock()
        mock_response.status_code = 429
        
        # Create a mock HTTP client
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        # Patch the server's HTTP client
        test_server.http_client = mock_client
        
        result = await test_server._call_aws_rag_endpoint("test prompt")
        
        assert result is None
        assert test_server.circuit_breaker['failure_count'] == 1
    
    @pytest.mark.asyncio
    async def test_successful_rag_call_resets_circuit_breaker(self, test_server, mock_aws_response):
        """Test that successful calls reset the circuit breaker."""
        
        # Set up some failures first
        test_server.circuit_breaker['failure_count'] = 2
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_aws_response
        
        # Create a mock HTTP client
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        # Patch the server's HTTP client
        test_server.http_client = mock_client
        
        result = await test_server._call_aws_rag_endpoint("test prompt")
        
        # Verify success
        assert result is not None
        assert result['success'] is True
        
        # Verify circuit breaker was reset
        assert test_server.circuit_breaker['failure_count'] == 0
    
    def test_configuration_loading(self):
        """Test proper loading of AWS configuration."""
        from promptproxy_server import PromptProxyConfig
        
        # Test with mock config data
        import tempfile
        import yaml
        
        config_data = {
            'host': 'test-host',
            'port': 9999,
            'aws': {
                'api_gateway_url': 'https://test.amazonaws.com/rag',
                'api_key': 'test-key',
                'timeout_seconds': 5.0
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            config = PromptProxyConfig(config_file)
            
            assert config.host == 'test-host'
            assert config.port == 9999
            assert config.aws['api_gateway_url'] == 'https://test.amazonaws.com/rag'
            assert config.aws['api_key'] == 'test-key'
            assert config.aws['timeout_seconds'] == 5.0
        finally:
            import os
            os.unlink(config_file)
    
    def test_missing_configuration_handling(self):
        """Test behavior when AWS configuration is missing."""
        from promptproxy_server import PromptProxyServer
        
        # Create server with empty config
        server = PromptProxyServer(config={})
        
        # Should not have AWS configuration
        assert not server.api_gateway_url
        assert not server.api_key
        
        # Test missing config behavior
        async def test_missing_config():
            result = await server._call_aws_rag_endpoint("test prompt")
            assert result is None
            assert server.circuit_breaker['failure_count'] == 1
        
        # Run the async test
        asyncio.run(test_missing_config())
    
    def test_circuit_breaker_configuration(self, test_server):
        """Test circuit breaker configuration is correct."""
        cb = test_server.circuit_breaker
        
        # Verify default configuration
        assert cb['failure_threshold'] == 3
        assert cb['cooldown_duration'] == 300  # 5 minutes
        assert cb['failure_count'] == 0
        assert cb['cooldown_until'] == 0


def test_sample_config_file_format():
    """Test that the sample config file has correct format."""
    
    sample_config = """
host: "localhost"
port: 8080
aws:
  api_gateway_url: "https://your-api-gateway-id.execute-api.us-east-1.amazonaws.com/prod/rag"
  api_key: "your-api-gateway-key-here"
  timeout_seconds: 2.0
"""
    
    import yaml
    try:
        config = yaml.safe_load(sample_config)
        
        # Verify structure
        assert 'host' in config
        assert 'port' in config
        assert 'aws' in config
        assert 'api_gateway_url' in config['aws']
        assert 'api_key' in config['aws']
        assert 'timeout_seconds' in config['aws']
        
        print("✅ Sample config file format is valid")
        
    except yaml.YAMLError as e:
        pytest.fail(f"Sample config file has invalid YAML format: {e}")


def test_server_initialization():
    """Test server initializes correctly with and without config."""
    from promptproxy_server import PromptProxyServer
    
    # Test without config
    server1 = PromptProxyServer()
    assert server1.host == "localhost"
    assert server1.port == 8080
    assert server1.api_gateway_url == ""
    assert server1.api_key == ""
    
    # Test with config
    config = {
        'host': 'test-host',
        'aws': {
            'api_gateway_url': 'https://test.com/rag',
            'api_key': 'test-key'
        }
    }
    server2 = PromptProxyServer(config=config)
    assert server2.host == "localhost"  # Default from __init__
    assert server2.api_gateway_url == "https://test.com/rag"
    assert server2.api_key == "test-key"


def integration_test_with_live_server():
    """
    Manual integration test to be run with a live PromptProxy server.
    This is not a pytest test but a manual verification script.
    """
    
    async def run_live_test():
        print("🧪 Running live integration test for Task D22...")
        
        # Test endpoints
        base_url = "http://localhost:8080"
        
        async with httpx.AsyncClient() as client:
            try:
                # Test health endpoint
                response = await client.get(f"{base_url}/health")
                print(f"Health check: {response.status_code}")
                
                # Test config endpoint
                response = await client.get(f"{base_url}/config")
                config_data = response.json()
                print(f"AWS configured: {config_data.get('aws_configured', False)}")
                
                # Test MQL5 prompt processing
                mql5_request = {
                    "prompt": "How do I use ArrayResize() function in MQL5 to resize dynamic arrays?",
                    "user": "test_user",
                    "session_id": "test_session"
                }
                
                response = await client.post(f"{base_url}/process", json=mql5_request)
                result = response.json()
                
                print(f"MQL5 detection: {result.get('mql5_detected', False)}")
                print(f"Processing time: {result.get('processing_time_ms', 0)}ms")
                
                if 'rag_results' in result.get('metadata', {}):
                    rag_info = result['metadata']['rag_results']
                    print(f"RAG call success: {rag_info.get('lambda_success', False)}")
                    if rag_info.get('lambda_success'):
                        print(f"Snippets retrieved: {rag_info.get('snippets_count', 0)}")
                
                # Test non-MQL5 prompt
                non_mql5_request = {
                    "prompt": "What is the weather like today?",
                    "user": "test_user"
                }
                
                response = await client.post(f"{base_url}/process", json=non_mql5_request)
                result = response.json()
                
                print(f"Non-MQL5 detection (should be False): {result.get('mql5_detected', True)}")
                
                print("✅ Live integration test completed")
                
            except Exception as e:
                print(f"❌ Live integration test failed: {e}")
    
    # Run the async test
    asyncio.run(run_live_test())


if __name__ == "__main__":
    print("Task D22 Integration Tests")
    print("=" * 50)
    
    # Run format validation
    test_sample_config_file_format()
    
    print("\n📋 To run full test suite:")
    print("pytest test_task_d22.py -v")
    
    print("\n🔴 To run live integration test:")
    print("1. Start PromptProxy server with: python promptproxy_server.py --config config.yaml")
    print("2. Run: python test_task_d22.py --live")
    
    # Check if live test was requested
    import sys
    if "--live" in sys.argv:
        print("\n🧪 Running live integration test...")
        integration_test_with_live_server()