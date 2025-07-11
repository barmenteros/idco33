#!/usr/bin/env python3
"""
Quick Start Script for End-to-End Integration Testing
Starts the PromptProxy server with production configuration for testing.
"""

import subprocess
import sys
import time
import yaml
from pathlib import Path

# Create production-ready configuration
PRODUCTION_CONFIG = {
    'host': 'localhost',
    'port': 8080,
    'log_level': 'INFO',
    'cors_origins': ['*'],
    'max_prompt_length': 10000,
    
    'aws': {
        'api_gateway_url': 'https://b6qmhutxnc.execute-api.us-east-1.amazonaws.com/prod/rag',
        'api_key': 'DNpEmzqcgQ2GcwB10LDBx9H3wBnQZ0Cr7z17HDzh',  # Replace with real API key
        'timeout_seconds': 2.0
    },
    
    'augmentation': {
        'max_snippets': 5,
        'snippet_separator': '\n\n'
    },
    
    'circuit_breaker': {
        'failure_threshold': 3,
        'success_threshold': 2,
        'cooldown_duration': 300,
        'half_open_max_requests': 3
    },
    
    'retry': {
        'max_retries': 1,
        'delay_ms': 100,
        'backoff_multiplier': 2.0
    }
}

def create_config_file():
    """Create the configuration file for testing."""
    config_path = Path("integration_test_config.yaml")
    
    with open(config_path, 'w') as f:
        yaml.dump(PRODUCTION_CONFIG, f, default_flow_style=False)
    
    print(f"✅ Configuration file created: {config_path}")
    return config_path

def start_server(config_path):
    """Start the PromptProxy server."""
    print("🚀 Starting PromptProxy server...")
    print("   Press Ctrl+C to stop the server")
    print("   In another terminal, run: python end_to_end_test.py")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "promptproxy_server.py", 
            "--config", str(config_path)
        ])
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped")

if __name__ == "__main__":
    print("MQL5 PromptProxy - Integration Test Server Startup")
    print("=" * 50)
    
    # Check if promptproxy_server.py exists
    if not Path("promptproxy_server.py").exists():
        print("❌ promptproxy_server.py not found in current directory")
        print("   Please run this script from the directory containing promptproxy_server.py")
        sys.exit(1)
    
    print("⚠️ IMPORTANT: Update the API key in the configuration!")
    print("   Edit 'YOUR_ACTUAL_API_KEY_HERE' with your real AWS API Gateway key")
    print()
    
    config_path = create_config_file()
    start_server(config_path)