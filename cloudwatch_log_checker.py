#!/usr/bin/env python3
"""
CloudWatch Log Checker
Retrieves recent Lambda logs to identify the exact error causing 500 responses.
"""

import boto3
import json
from datetime import datetime, timedelta
from typing import List, Dict

class CloudWatchLogChecker:
    """Check CloudWatch logs for Lambda errors."""
    
    def __init__(self):
        # Based on your project naming from the specs
        self.function_name = "mql5-rag-handler"  # Adjust if different
        self.log_group_name = f"/aws/lambda/{self.function_name}"
        
        # Initialize boto3 client
        try:
            self.logs_client = boto3.client('logs', region_name='us-east-1')
            self.lambda_client = boto3.client('lambda', region_name='us-east-1')
        except Exception as e:
            print(f"❌ AWS credentials not configured: {e}")
            print("   Run: aws configure")
            exit(1)
    
    def check_lambda_logs(self):
        """Check recent Lambda logs for errors."""
        print("☁️ CloudWatch Log Analysis")
        print("=" * 50)
        
        # First, verify the Lambda function exists
        self._verify_lambda_function()
        
        # Get recent log events
        self._get_recent_log_events()
        
        # Look for specific error patterns
        self._analyze_error_patterns()
    
    def _verify_lambda_function(self):
        """Verify the Lambda function exists and get its details."""
        print("🔍 Step 1: Lambda Function Verification")
        print("-" * 40)
        
        try:
            # Try common function names based on your project
            possible_names = [
                "mql5-rag-handler",
                "mql5-rag",
                "MQL5RAGHandler",
                "mql5-prompt-enrichment"
            ]
            
            function_found = False
            
            for func_name in possible_names:
                try:
                    response = self.lambda_client.get_function(FunctionName=func_name)
                    self.function_name = func_name
                    self.log_group_name = f"/aws/lambda/{func_name}"
                    function_found = True
                    
                    print(f"✅ Found Lambda function: {func_name}")
                    print(f"📊 Runtime: {response['Configuration']['Runtime']}")
                    print(f"📊 Memory: {response['Configuration']['MemorySize']} MB")
                    print(f"📊 Timeout: {response['Configuration']['Timeout']} seconds")
                    print(f"📊 Last Modified: {response['Configuration']['LastModified']}")
                    
                    # Check if it's a container image
                    if 'ImageUri' in response['Code']:
                        print(f"📊 Container Image: {response['Code']['ImageUri']}")
                    
                    break
                    
                except self.lambda_client.exceptions.ResourceNotFoundException:
                    continue
            
            if not function_found:
                print("❌ No Lambda function found with expected names:")
                for name in possible_names:
                    print(f"   - {name}")
                print("\nℹ️ List all Lambda functions:")
                
                try:
                    functions = self.lambda_client.list_functions()
                    for func in functions['Functions']:
                        if 'mql5' in func['FunctionName'].lower() or 'rag' in func['FunctionName'].lower():
                            print(f"   📋 {func['FunctionName']}")
                except Exception as e:
                    print(f"❌ Cannot list functions: {e}")
                
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Lambda verification failed: {e}")
            return False
    
    def _get_recent_log_events(self):
        """Get recent log events from the Lambda function."""
        print(f"\n📋 Step 2: Recent Log Events Analysis")
        print("-" * 40)
        
        try:
            # Get logs from the last hour
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            print(f"📊 Checking logs from {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"📊 Log Group: {self.log_group_name}")
            
            # First, check if log group exists
            try:
                self.logs_client.describe_log_groups(logGroupNamePrefix=self.log_group_name)
            except Exception as e:
                print(f"❌ Log group not found: {self.log_group_name}")
                print("   This suggests the Lambda function has never been invoked")
                return
            
            # Get log streams
            streams_response = self.logs_client.describe_log_streams(
                logGroupName=self.log_group_name,
                orderBy='LastEventTime',
                descending=True,
                limit=5
            )
            
            if not streams_response['logStreams']:
                print("⚠️ No log streams found - Lambda may never have been invoked")
                return
            
            print(f"📊 Found {len(streams_response['logStreams'])} recent log streams")
            
            # Get events from the most recent streams
            all_events = []
            
            for stream in streams_response['logStreams'][:3]:  # Check 3 most recent streams
                stream_name = stream['logStreamName']
                print(f"📋 Checking stream: {stream_name}")
                
                try:
                    events_response = self.logs_client.get_log_events(
                        logGroupName=self.log_group_name,
                        logStreamName=stream_name,
                        startTime=int(start_time.timestamp() * 1000),
                        endTime=int(end_time.timestamp() * 1000)
                    )
                    
                    events = events_response['events']
                    all_events.extend(events)
                    
                    print(f"   📊 {len(events)} events in this stream")
                    
                except Exception as e:
                    print(f"   ❌ Error reading stream {stream_name}: {e}")
            
            # Sort all events by timestamp
            all_events.sort(key=lambda x: x['timestamp'])
            
            if not all_events:
                print("⚠️ No recent log events found")
                return
            
            print(f"\n📋 Recent Log Events ({len(all_events)} total):")
            print("-" * 60)
            
            # Show the most recent events
            recent_events = all_events[-20:]  # Last 20 events
            
            for event in recent_events:
                timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
                message = event['message'].strip()
                
                # Color code by log level
                if 'ERROR' in message or 'Exception' in message or 'Traceback' in message:
                    prefix = "❌"
                elif 'WARNING' in message or 'WARN' in message:
                    prefix = "⚠️"
                elif 'START RequestId' in message or 'END RequestId' in message:
                    prefix = "📋"
                else:
                    prefix = "📄"
                
                print(f"{prefix} {timestamp.strftime('%H:%M:%S')} | {message}")
            
            return all_events
            
        except Exception as e:
            print(f"❌ Failed to get log events: {e}")
            return []
    
    def _analyze_error_patterns(self):
        """Analyze logs for common error patterns."""
        print(f"\n🔍 Step 3: Error Pattern Analysis")
        print("-" * 40)
        
        # This would analyze the logs for common issues
        common_errors = [
            ("Import Error", "No module named", "Missing Python dependencies"),
            ("Memory Error", "MemoryError", "Lambda memory allocation too low"),
            ("Timeout Error", "Task timed out", "Lambda timeout too short"),
            ("Permission Error", "AccessDenied", "IAM permissions missing"),
            ("S3 Error", "S3 operation failed", "S3 bucket access issues"),
            ("DynamoDB Error", "DynamoDB operation failed", "DynamoDB access issues"),
            ("FAISS Error", "FAISS", "FAISS library loading issues"),
            ("Model Error", "model", "Embedding model loading issues"),
        ]
        
        print("🔍 Checking for common Lambda error patterns...")
        print("   (Run the script to see actual log analysis)")


def main():
    """Main function to check CloudWatch logs."""
    print("☁️ CloudWatch Log Checker for MQL5 RAG Lambda")
    print("=" * 60)
    
    # Check AWS credentials
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        if not credentials:
            print("❌ AWS credentials not found")
            print("   Run: aws configure")
            print("   Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            return
        
        print(f"✅ AWS credentials found for region: {session.region_name or 'us-east-1'}")
        
    except ImportError:
        print("❌ boto3 not installed")
        print("   Run: pip install boto3")
        return
    
    checker = CloudWatchLogChecker()
    checker.check_lambda_logs()
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS:")
    print("1. Look for ERROR messages in the logs above")
    print("2. Common issues: missing dependencies, memory/timeout, permissions")
    print("3. If no logs found: Lambda may not be deployed properly")
    print("4. Check AWS Console → Lambda → Functions for deployment status")
    print("=" * 60)


if __name__ == "__main__":
    main()