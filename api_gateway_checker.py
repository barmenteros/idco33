#!/usr/bin/env python3
"""
API Gateway Configuration Checker
Check what Lambda function the API Gateway is trying to invoke
"""

import boto3
import json
from pprint import pprint

def check_api_gateway_config():
    """Check API Gateway configuration to find the missing Lambda."""
    
    print("🔍 API Gateway Configuration Analysis")
    print("=" * 50)
    
    try:
        # Initialize API Gateway client
        apigw_client = boto3.client('apigateway', region_name='us-east-1')
        
        # Get all REST APIs
        apis = apigw_client.get_rest_apis()
        
        print(f"📊 Found {len(apis['items'])} API Gateway REST APIs")
        
        # Look for MQL5 related APIs
        mql5_apis = []
        for api in apis['items']:
            api_name = api.get('name', '').lower()
            api_desc = api.get('description', '').lower()
            
            if 'mql5' in api_name or 'rag' in api_name or 'mql5' in api_desc:
                mql5_apis.append(api)
            
            print(f"📋 API: {api['name']} (ID: {api['id']})")
            print(f"   Created: {api.get('createdDate', 'unknown')}")
            print(f"   Description: {api.get('description', 'none')}")
        
        if not mql5_apis:
            print("\n⚠️ No MQL5/RAG related APIs found by name")
            print("   Checking all APIs for /rag resource...")
            
            # Check each API for /rag resource
            for api in apis['items']:
                try:
                    resources = apigw_client.get_resources(restApiId=api['id'])
                    
                    for resource in resources['items']:
                        if '/rag' in resource.get('path', ''):
                            print(f"✅ Found /rag resource in API: {api['name']}")
                            mql5_apis.append(api)
                            break
                            
                except Exception as e:
                    print(f"   ❌ Error checking API {api['name']}: {e}")
        
        # Analyze the MQL5 API configuration
        for api in mql5_apis:
            print(f"\n🔍 Analyzing API: {api['name']} ({api['id']})")
            
            try:
                # Get resources
                resources = apigw_client.get_resources(restApiId=api['id'])
                
                for resource in resources['items']:
                    path = resource.get('path', '')
                    if '/rag' in path or resource.get('resourceMethods'):
                        print(f"📋 Resource: {path}")
                        
                        # Check methods
                        methods = resource.get('resourceMethods', {})
                        for method, method_data in methods.items():
                            print(f"   Method: {method}")
                            
                            # Get method details including integration
                            try:
                                method_details = apigw_client.get_method(
                                    restApiId=api['id'],
                                    resourceId=resource['id'],
                                    httpMethod=method
                                )
                                
                                integration = method_details.get('methodIntegration', {})
                                integration_type = integration.get('type', 'unknown')
                                
                                print(f"   Integration Type: {integration_type}")
                                
                                if integration_type == 'AWS_PROXY':
                                    uri = integration.get('uri', '')
                                    print(f"   Integration URI: {uri}")
                                    
                                    # Extract Lambda function name from URI
                                    if 'lambda' in uri:
                                        # URI format: arn:aws:apigateway:region:lambda:path/2015-03-31/functions/arn:aws:lambda:region:account:function:function-name/invocations
                                        parts = uri.split('/')
                                        for i, part in enumerate(parts):
                                            if part == 'function:' and i + 1 < len(parts):
                                                lambda_name = parts[i + 1]
                                                print(f"   🎯 Target Lambda Function: {lambda_name}")
                                                
                                                # Check if this Lambda exists
                                                lambda_client = boto3.client('lambda', region_name='us-east-1')
                                                try:
                                                    lambda_client.get_function(FunctionName=lambda_name)
                                                    print(f"   ✅ Lambda function exists")
                                                except lambda_client.exceptions.ResourceNotFoundException:
                                                    print(f"   ❌ Lambda function DOES NOT EXIST - This is the problem!")
                                                except Exception as e:
                                                    print(f"   ⚠️ Error checking Lambda: {e}")
                                                break
                                
                            except Exception as e:
                                print(f"   ❌ Error getting method details: {e}")
                
            except Exception as e:
                print(f"❌ Error analyzing API {api['name']}: {e}")
        
        if not mql5_apis:
            print("\n❌ No APIs found with /rag endpoint")
            print("   This suggests the API Gateway was not deployed correctly either")
    
    except Exception as e:
        print(f"❌ Failed to check API Gateway: {e}")

def main():
    """Main function."""
    print("🔍 API Gateway Configuration Checker")
    print("This will show what Lambda function your API Gateway is trying to call")
    print()
    
    try:
        check_api_gateway_config()
        
        print("\n" + "=" * 60)
        print("🎯 SUMMARY:")
        print("If you see 'Lambda function DOES NOT EXIST' above,")
        print("that confirms the Lambda was never deployed.")
        print("")
        print("🚀 NEXT STEPS:")
        print("1. Deploy the MQL5 RAG Lambda function")
        print("2. Or update API Gateway to point to correct Lambda")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()