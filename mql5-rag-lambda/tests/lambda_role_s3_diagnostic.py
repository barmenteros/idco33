#!/usr/bin/env python3
"""
Lambda Role S3 Access Diagnostic Script
Tests S3 access using the exact same Lambda execution role and permissions
that the Lambda function would use.
"""

import boto3
import json
import os
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError

def test_lambda_role_s3_access():
    """Test S3 access using Lambda execution role"""
    
    # Configuration from Lambda environment
    bucket_name = "mql5-rag-faiss-index-20250106-minimal"
    role_arn = "arn:aws:iam::193245229238:role/mql5-rag-lambda-execution-role"
    region = "us-east-1"
    
    print("=== Lambda Role S3 Access Diagnostic ===")
    print(f"Target Bucket: {bucket_name}")
    print(f"Lambda Role: {role_arn}")
    print(f"Region: {region}")
    print(f"Test Time: {datetime.now()}\n")
    
    # Test 1: Check current credentials
    print("1. Testing current AWS credentials...")
    try:
        sts_client = boto3.client('sts', region_name=region)
        identity = sts_client.get_caller_identity()
        print(f"✅ Current identity: {identity.get('Arn', 'unknown')}")
        print(f"   Account: {identity.get('Account', 'unknown')}")
        print(f"   User ID: {identity.get('UserId', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to get caller identity: {e}")
        return
    
    # Test 2: Assume Lambda execution role
    print(f"\n2. Attempting to assume Lambda execution role...")
    try:
        # Assume the Lambda execution role
        assumed_role = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName='lambda-s3-diagnostic-test'
        )
        
        credentials = assumed_role['Credentials']
        print("✅ Successfully assumed Lambda execution role")
        
        # Create S3 client with assumed role credentials
        s3_client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        print(f"❌ Failed to assume Lambda role: {error_code} - {error_message}")
        
        if error_code == 'AccessDenied':
            print("   💡 Check if your current credentials have sts:AssumeRole permission")
        elif error_code == 'InvalidUserID.NotFound':
            print("   💡 The Lambda execution role may not exist or be accessible")
        
        # Fallback: use current credentials
        print("\n   🔄 Falling back to current credentials for S3 testing...")
        s3_client = boto3.client('s3', region_name=region)
        
    except Exception as e:
        print(f"❌ Unexpected error assuming role: {e}")
        return
    
    # Test 3: Verify S3 bucket access
    print(f"\n3. Testing S3 bucket access...")
    try:
        # Test bucket existence and access
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✅ Bucket '{bucket_name}' exists and is accessible")
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        print(f"❌ Bucket access failed: {error_code} - {error_message}")
        
        if error_code == 'NoSuchBucket':
            print("   💡 The bucket does not exist")
        elif error_code == 'AccessDenied':
            print("   💡 No permission to access the bucket")
        elif error_code == '403':
            print("   💡 Forbidden - check bucket permissions and region")
        
        return
    
    # Test 4: List bucket contents
    print(f"\n4. Listing bucket contents...")
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=20)
        
        if 'Contents' not in response:
            print("📁 Bucket is empty")
            return
        
        objects = response['Contents']
        print(f"📁 Found {len(objects)} objects:")
        
        for obj in objects:
            key = obj['Key']
            size = obj['Size']
            modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"   📄 {key} ({size:,} bytes) - {modified}")
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        print(f"❌ Failed to list bucket contents: {error_code} - {error_message}")
        return
    
    # Test 5: Test specific file access (HeadObject operations)
    print(f"\n5. Testing specific file access (HeadObject operations)...")
    
    test_files = [
        "index.faiss",
        "index.pkl", 
        "metadata/index_metadata.json"
    ]
    
    for file_key in test_files:
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=file_key)
            size = response.get('ContentLength', 0)
            modified = response.get('LastModified', 'unknown')
            content_type = response.get('ContentType', 'unknown')
            print(f"   ✅ {file_key}: {size:,} bytes, {content_type}, modified {modified}")
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                print(f"   ❌ {file_key}: File not found (NoSuchKey)")
            elif error_code == 'AccessDenied':
                print(f"   ❌ {file_key}: Access denied")
            else:
                print(f"   ❌ {file_key}: {error_code} - {e}")
    
    # Test 6: Test download operations
    print(f"\n6. Testing download operations...")
    
    for file_key in test_files:
        try:
            local_file = f"/tmp/test_{file_key.replace('/', '_')}"
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            
            s3_client.download_file(bucket_name, file_key, local_file)
            
            # Check downloaded file
            if os.path.exists(local_file):
                file_size = os.path.getsize(local_file)
                print(f"   ✅ {file_key}: Downloaded {file_size:,} bytes to {local_file}")
                
                # Clean up
                os.remove(local_file)
            else:
                print(f"   ❌ {file_key}: Download succeeded but file not found locally")
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                print(f"   ❌ {file_key}: File not found for download")
            elif error_code == 'AccessDenied':
                print(f"   ❌ {file_key}: Access denied for download")
            else:
                print(f"   ❌ {file_key}: Download failed - {error_code}")
        except Exception as e:
            print(f"   ❌ {file_key}: Unexpected download error - {e}")
    
    # Test 7: IAM permissions analysis
    print(f"\n7. Analyzing IAM permissions...")
    try:
        iam_client = boto3.client('iam', region_name=region)
        
        # Get role details
        role_name = role_arn.split('/')[-1]
        role_details = iam_client.get_role(RoleName=role_name)
        
        print(f"   📋 Role created: {role_details['Role']['CreateDate']}")
        print(f"   📋 Role path: {role_details['Role']['Path']}")
        
        # List attached policies
        attached_policies = iam_client.list_attached_role_policies(RoleName=role_name)
        print(f"   📋 Attached policies: {len(attached_policies['AttachedPolicies'])}")
        
        for policy in attached_policies['AttachedPolicies']:
            print(f"      - {policy['PolicyName']} ({policy['PolicyArn']})")
        
        # List inline policies
        inline_policies = iam_client.list_role_policies(RoleName=role_name)
        print(f"   📋 Inline policies: {len(inline_policies['PolicyNames'])}")
        
        for policy_name in inline_policies['PolicyNames']:
            print(f"      - {policy_name}")
            
            # Get policy document for S3-related policies
            if 's3' in policy_name.lower():
                try:
                    policy_doc = iam_client.get_role_policy(RoleName=role_name, PolicyName=policy_name)
                    policy_statements = policy_doc['PolicyDocument'].get('Statement', [])
                    
                    for stmt in policy_statements:
                        if isinstance(stmt.get('Action'), list):
                            s3_actions = [action for action in stmt['Action'] if 's3:' in action]
                        else:
                            s3_actions = [stmt.get('Action')] if 's3:' in str(stmt.get('Action', '')) else []
                        
                        if s3_actions:
                            print(f"         S3 Actions: {s3_actions}")
                            print(f"         Resources: {stmt.get('Resource', 'Not specified')}")
                
                except Exception as e:
                    print(f"         ❌ Could not read policy details: {e}")
        
    except Exception as e:
        print(f"   ❌ IAM analysis failed: {e}")
    
    print(f"\n=== Diagnostic Complete ===")
    print("If files exist but downloads fail, check:")
    print("1. Lambda execution role has s3:GetObject permission")
    print("2. Lambda execution role has s3:GetObjectVersion permission")
    print("3. Bucket policy allows access from the Lambda role")
    print("4. Files are in the correct region")


if __name__ == "__main__":
    test_lambda_role_s3_access()
