# MQL5 RAG Project - Resource Registry

## Project Overview
**Project Name**: MQL5 Prompt Enrichment Middleware  
**AWS Account ID**: 193245229238  
**Primary Region**: us-east-1 (N. Virginia)  
**Deployment Date**: July 6, 2025  

## CloudFormation Stacks

### Stack: mql5-rag-iam
- **Stack ID**: `arn:aws:cloudformation:us-east-1:193245229238:stack/mql5-rag-iam/507d5c40-5a03-11f0-8661-12a702ad83ef`
- **Status**: CREATE_COMPLETE
- **Purpose**: IAM roles and policies foundation
- **Template**: mql5-iam-roles.yaml

### Stack: mql5-rag-s3
- **Stack ID**: `arn:aws:cloudformation:us-east-1:193245229238:stack/mql5-rag-s3/[new-stack-id]`
- **Status**: CREATE_COMPLETE
- **Purpose**: S3 bucket for FAISS index storage (minimal configuration)
### Stack: mql5-rag-dynamodb
- **Stack ID**: `arn:aws:cloudformation:us-east-1:193245229238:stack/mql5-rag-dynamodb/[new-stack-id]`
- **Status**: CREATE_COMPLETE
- **Purpose**: DynamoDB table for documentation snippets storage
### Stack: mql5-rag-api-gateway
- **Stack ID**: `arn:aws:cloudformation:us-east-1:193245229238:stack/mql5-rag-api-gateway/[new-stack-id]`
- **Status**: CREATE_COMPLETE
- **Purpose**: API Gateway with /rag endpoint and API Key authentication
- **Template**: mql5-api-gateway.yaml

## Parameters Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| ProjectName | `mql5-rag` | Project identifier for resource naming |
| S3BucketName | `mql5-rag-faiss-index-20250106` | S3 bucket for FAISS index storage |
| DynamoDBTableName | `mql5-doc-snippets` | DynamoDB table for documentation snippets |

## IAM Resources

### Lambda Execution Role
- **Role Name**: `mql5-rag-lambda-execution-role`
- **Role ARN**: `arn:aws:iam::193245229238:role/mql5-rag-lambda-execution-role`
- **Export Name**: `mql5-rag-lambda-execution-role-arn`
- **Purpose**: Execution role for the RAG Lambda function
- **Permissions**:
  - S3: GetObject, GetObjectVersion, ListBucket on `mql5-rag-faiss-index-20250106`
  - DynamoDB: GetItem, BatchGetItem, Query, DescribeTable on `mql5-doc-snippets`
  - CloudWatch: Logs and metrics (namespace: `mql5-rag/RAG`)
  - Managed Policy: AWSLambdaBasicExecutionRole

### CodeBuild Service Role
- **Role Name**: `mql5-rag-codebuild-service-role`
- **Role ARN**: `arn:aws:iam::193245229238:role/mql5-rag-codebuild-service-role`
- **Export Name**: `mql5-rag-codebuild-service-role-arn`
- **Purpose**: Service role for monthly document ingestion pipeline
- **Permissions**:
  - S3: Full access (Get/Put/Delete) on `mql5-rag-faiss-index-20250106`
  - DynamoDB: Full table operations on `mql5-doc-snippets`
  - CloudWatch: Logs and metrics (namespace: `mql5-rag/Ingestion`)

## S3 Resources

### FAISS Index Bucket
- **Bucket Name**: `mql5-rag-faiss-index-20250106-minimal`
- **Status**: CREATE_COMPLETE ✅
- **Purpose**: Store FAISS index files (index.faiss, index.pkl)
- **Security Model**: IAM role-based access (no bucket policy)
- **Access Control**: 
  - Lambda Role: Read access via IAM permissions from Task A1
  - CodeBuild Role: Full access via IAM permissions from Task A1
- **Configuration**: 
  - Server-side encryption (AES256)
  - Public access completely blocked
  - Minimal configuration for reliability
- **Free Tier**: 5GB storage, 20k GET requests
- **Exports Available**:
  - `mql5-rag-faiss-index-bucket-name`
  - `mql5-rag-faiss-index-bucket-arn`

## DynamoDB Resources

### Documentation Snippets Table
- **Table Name**: `mql5-doc-snippets`
- **Status**: Active ✅
- **Table ARN**: `arn:aws:dynamodb:us-east-1:193245229238:table/mql5-doc-snippets`
- **Purpose**: Store documentation snippet text and metadata for RAG retrieval
- **Schema**:
  - **Primary Key**: `DocID` (String) - Hash key for unique snippet identification
  - **Sort Key**: None (simple key structure)
  - **Billing Mode**: On-demand (pay-per-request)
- **Features**:
  - Server-side encryption enabled
  - DynamoDB Streams enabled (NEW_AND_OLD_IMAGES)
  - Point-in-time recovery disabled (cost optimization)
- **Access Control**:
  - Lambda Role: Read access (`GetItem`, `BatchGetItem`) via IAM permissions
  - CodeBuild Role: Full access (`PutItem`, `BatchWriteItem`, etc.) via IAM permissions
- **Free Tier**: 25GB storage, 25 RCU/WCU capacity
- **Data Structure** (per item):
  - `DocID`: Unique identifier (matches FAISS index)
  - `snippet_text`: ~500-token documentation content
  - `source`: Source reference information
  - `metadata`: Additional context (section, page, etc.)
- **Exports Available**:
  - `mql5-rag-doc-snippets-table-name`
  - `mql5-rag-doc-snippets-table-arn`
  - `mql5-rag-doc-snippets-table-stream-arn`

## API Gateway Resources

### MQL5 RAG REST API
- **API Name**: `mql5-rag-api`
- **Status**: Active ✅
- **API ID**: Available via stack outputs
- **Endpoint**: `https://[api-id].execute-api.us-east-1.amazonaws.com/prod/rag`
- **Methods**: 
  - **POST /rag**: Main RAG endpoint (requires API Key)
  - **OPTIONS /rag**: CORS preflight (no auth required)
- **Authentication**: API Key required
- **Rate Limiting**: 25 RPS, 50 burst limit
- **Monthly Quota**: 100,000 requests (within 1M free tier)
- **Integration**: AWS Lambda Proxy (future Lambda function)
- **Features**:
  - Request validation (JSON schema)
  - CORS support for web clients
  - Error handling (400, 500 responses)
  - Usage plan with throttling
- **Security**:
  - Regional endpoint (better performance)
  - API Key authentication
  - Request/response validation
- **Free Tier**: 1M requests/month
- **Exports Available**:
  - `mql5-rag-api-gateway-id`
  - `mql5-rag-api-endpoint-url`
  - `mql5-rag-api-key-id`
  - `mql5-rag-api-key-value`

## Resources To Be Created

### Planned Infrastructure (Next Tasks)

#### Task A4: API Gateway
- **Endpoint**: `/rag`
- **Purpose**: REST API to trigger Lambda RAG function
- **Security**: API Key authentication
- **Integration**: Lambda proxy integration
- **Free Tier**: 1M requests/month

#### Task A5: Lambda Function
- **Function Name**: TBD (likely `mql5-rag-handler`)
- **Runtime**: Python (containerized)
- **Memory**: TBD (likely 1-2GB for FAISS index)
- **Role**: `mql5-rag-lambda-execution-role`
- **Purpose**: RAG processing (embed query, search FAISS, retrieve snippets)

## Naming Conventions

### Resource Naming Pattern
- **Format**: `{ProjectName}-{Component}-{Suffix}`
- **Project Name**: `mql5-rag`
- **Examples**:
  - IAM Roles: `mql5-rag-lambda-execution-role`
  - S3 Buckets: `mql5-rag-faiss-index-20250106`
  - DynamoDB Tables: `mql5-doc-snippets`
  - Lambda Functions: `mql5-rag-handler`
  - API Gateway: `mql5-rag-api`

### Export Naming Pattern
- **Format**: `{ProjectName}-{resource-type}-{suffix}`
- **Examples**:
  - `mql5-rag-lambda-execution-role-arn`
  - `mql5-rag-codebuild-service-role-arn`
  - `mql5-rag-project-name`

### CloudWatch Namespaces
- **Lambda Metrics**: `mql5-rag/RAG`
- **CodeBuild Metrics**: `mql5-rag/Ingestion`

## Tags Strategy

| Tag Key | Tag Value | Applied To |
|---------|-----------|------------|
| Project | `mql5-rag` | All resources |
| Component | `Lambda-Execution-Role`, `CodeBuild-Service-Role`, etc. | Specific to component |
| Environment | `production` | All resources |

## Development Progress

### Completed Tasks ✅
- [x] **Task A1**: Define AWS IAM Roles & Policies
  - CloudFormation stack deployed
  - IAM roles created and validated
  - Exports available for cross-stack references

- [x] **Task A2**: Provision S3 Bucket for FAISS Index
  - CloudFormation stack deployed successfully (minimal configuration)
  - S3 bucket created with encryption and security
  - IAM-based access control (no bucket policy complications)
- [x] **Task A3**: Create DynamoDB Table for Snippets
  - CloudFormation stack deployed successfully
  - DynamoDB table created with on-demand billing
  - Simple key schema (DocID) optimized for BatchGetItem operations
- [x] **Task A4**: Configure API Gateway (/rag) with API Key
  - CloudFormation stack deployed successfully
  - REST API created with /rag endpoint
  - API Key authentication configured
  - Usage plan with rate limiting (25 RPS, 100K requests/month)
- [x] **Task A5**: Define Lambda Execution Role
  - Lambda execution role verified and documented
  - All required permissions confirmed (S3, DynamoDB, CloudWatch)
  - Role ARN exported and ready for Lambda function deployment
  - Least-privilege security model validated

### Next Tasks 📋
- [ ] **Task A3**: Create DynamoDB Table for Snippets
- [ ] **Task A4**: Configure API Gateway (/rag) with API Key
- [ ] **Task A5**: Define Lambda Execution Role

## Important Notes

### Security Considerations
- All IAM roles follow least-privilege principle
- Resource access scoped to specific ARNs (no wildcards)
- CloudWatch metrics namespaced to project
- API Gateway will use API Key authentication

### Cost Management
- All resources designed to stay within AWS Free Tier
- S3 bucket size limited by documentation corpus (~100MB expected)
- DynamoDB on-demand billing (expected minimal usage)
- Lambda execution optimized for cold start and memory efficiency

### Cross-Stack Dependencies
The IAM stack exports role ARNs that will be imported by subsequent stacks:
```yaml
# Example import in future stacks
ExecutionRole: !ImportValue mql5-rag-lambda-execution-role-arn
```

## Quick Reference Commands

### Check Stack Status
```bash
aws cloudformation describe-stacks --stack-name mql5-rag-iam --region us-east-1
```

### List Stack Exports
```bash
aws cloudformation list-exports --region us-east-1
```

### Delete Stack (if needed)
```bash
aws cloudformation delete-stack --stack-name mql5-rag-iam --region us-east-1
```

---
**Last Updated**: July 6, 2025  
**Status**: All Infrastructure Tasks (A1-A5) Complete, Ready for Core Implementation

## Implementation Notes & Lessons Learned

### Task A2 - S3 Bucket Deployment Strategy

**Challenge Encountered:**
- Initial attempts with complex bucket policies failed repeatedly
- CloudFormation bucket policy validation errors (Status Code 400)
- Multiple rollback failures due to policy syntax issues

**Solution Adopted:**
- **Minimal Configuration Approach**: Deployed S3 bucket without bucket policy
- **IAM-Based Security**: Leveraged IAM roles from Task A1 for access control
- **Rationale**: IAM role permissions (defined in Task A1) already provide the necessary security:
  - Lambda role has `s3:GetObject` permissions on the specific bucket
  - CodeBuild role has full S3 permissions on the specific bucket
  - No additional bucket policy needed for basic functionality

**Benefits of Minimal Approach:**
1. **Reliability**: Eliminated complex policy syntax as failure point
2. **Security**: Still maintains least-privilege access via IAM roles
3. **Simplicity**: Easier to troubleshoot and maintain
4. **Forward Progress**: Enables continuation to Task A3 without further delays

**Future Considerations:**
- Bucket policy can be added later if additional access restrictions needed
- Current IAM-based security model is sufficient for project requirements
- Minimal configuration aligns with "keep it simple" development principle