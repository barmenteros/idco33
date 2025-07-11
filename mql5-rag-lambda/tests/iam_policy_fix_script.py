#!/usr/bin/env python3
"""
Fix IAM S3 Policy for Lambda Execution Role
Updates the S3AccessPolicy to point to the correct bucket name
"""

import boto3
import json

def fix_lambda_s3_policy():
    """Fix the S3 bucket name in Lambda execution role policy"""
    
    role_name = "mql5-rag-lambda-execution-role"
    policy_name = "S3AccessPolicy"
    correct_bucket_name = "mql5-rag-faiss-index-20250106-minimal"
    
    print("=== Fixing Lambda S3 Access Policy ===")
    print(f"Role: {role_name}")
    print(f"Policy: {policy_name}")
    print(f"Correct Bucket: {correct_bucket_name}\n")
    
    try:
        iam_client = boto3.client('iam')
        
        # Get current policy
        print("1. Reading current policy...")
        current_policy = iam_client.get_role_policy(
            RoleName=role_name,
            PolicyName=policy_name
        )
        
        policy_doc = current_policy['PolicyDocument']
        print(f"✅ Current policy retrieved")
        
        # Update the policy document
        print("2. Updating policy document...")
        
        updated = False
        for statement in policy_doc.get('Statement', []):
            resources = statement.get('Resource', [])
            
            if isinstance(resources, str):
                resources = [resources]
            
            # Update bucket resources
            new_resources = []
            for resource in resources:
                if 'mql5-rag-faiss-index-20250106' in resource and '-minimal' not in resource:
                    # Replace old bucket name with correct one
                    new_resource = resource.replace(
                        'mql5-rag-faiss-index-20250106',
                        correct_bucket_name
                    )
                    new_resources.append(new_resource)
                    print(f"   🔄 Updated: {resource} → {new_resource}")
                    updated = True
                else:
                    new_resources.append(resource)
            
            # Update the statement
            if len(new_resources) == 1:
                statement['Resource'] = new_resources[0]
            else:
                statement['Resource'] = new_resources
        
        if not updated:
            print("   ℹ️ No updates needed - policy already correct")
            return
        
        # Put the updated policy
        print("3. Updating IAM policy...")
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_doc)
        )
        
        print("✅ Policy updated successfully!")
        
        # Verify the update
        print("4. Verifying update...")
        updated_policy = iam_client.get_role_policy(
            RoleName=role_name,
            PolicyName=policy_name
        )
        
        print("   Updated policy resources:")
        for statement in updated_policy['PolicyDocument'].get('Statement', []):
            resources = statement.get('Resource', [])
            if isinstance(resources, str):
                resources = [resources]
            
            for resource in resources:
                if 's3:::' in resource:
                    print(f"   ✅ {resource}")
        
        print(f"\n🎉 Lambda execution role now has access to {correct_bucket_name}")
        print("You can now test the Lambda function again!")
        
    except Exception as e:
        print(f"❌ Failed to update policy: {e}")


if __name__ == "__main__":
    fix_lambda_s3_policy()
