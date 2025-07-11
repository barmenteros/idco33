#!/bin/bash
# MQL5 RAG Engine Lambda Container Deployment Script
# Task C12: Create Dockerfile for Lambda Container

set -e

# Configuration
PROJECT_NAME="mql5-rag"
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${PROJECT_NAME}-rag-engine"
IMAGE_TAG="latest"
LAMBDA_FUNCTION_NAME="${PROJECT_NAME}-handler"

echo "=== MQL5 RAG Engine Lambda Deployment ==="
echo "Project: ${PROJECT_NAME}"
echo "Region: ${REGION}"
echo "Account: ${ACCOUNT_ID}"
echo "ECR Repo: ${ECR_REPO}"

# Step 1: Create ECR repository if it doesn't exist
echo "Creating ECR repository..."
aws ecr describe-repositories --repository-names ${ECR_REPO} --region ${REGION} 2>/dev/null || \
aws ecr create-repository --repository-name ${ECR_REPO} --region ${REGION}

# Step 2: Get ECR login token
echo "Logging into ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Step 3: Build Docker image
echo "Building Docker image..."
docker build -t ${ECR_REPO}:${IMAGE_TAG} .

# Step 4: Tag image for ECR
echo "Tagging image for ECR..."
docker tag ${ECR_REPO}:${IMAGE_TAG} ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}

# Step 5: Push image to ECR
echo "Pushing image to ECR..."
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}

# Step 6: Create or update Lambda function
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
EXECUTION_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${PROJECT_NAME}-lambda-execution-role"

echo "Checking if Lambda function exists..."
if aws lambda get-function --function-name ${LAMBDA_FUNCTION_NAME} --region ${REGION} 2>/dev/null; then
    echo "Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name ${LAMBDA_FUNCTION_NAME} \
        --image-uri ${IMAGE_URI} \
        --region ${REGION}
    
    aws lambda update-function-configuration \
        --function-name ${LAMBDA_FUNCTION_NAME} \
        --timeout 30 \
        --memory-size 1024 \
        --environment Variables="{
            FAISS_INDEX_BUCKET=mql5-rag-faiss-index-20250106-minimal,
            SNIPPETS_TABLE=mql5-doc-snippets
        }" \
        --region ${REGION}
else
    echo "Creating new Lambda function..."
    aws lambda create-function \
        --function-name ${LAMBDA_FUNCTION_NAME} \
        --package-type Image \
        --code ImageUri=${IMAGE_URI} \
        --role ${EXECUTION_ROLE_ARN} \
        --timeout 30 \
        --memory-size 1024 \
        --environment Variables="{
            FAISS_INDEX_BUCKET=mql5-rag-faiss-index-20250106-minimal,
            SNIPPETS_TABLE=mql5-doc-snippets
        }" \
        --region ${REGION}
fi

echo "=== Deployment Complete ==="
echo "Lambda Function: ${LAMBDA_FUNCTION_NAME}"
echo "Image URI: ${IMAGE_URI}"
echo ""
echo "Test the function with:"
echo "aws lambda invoke --function-name ${LAMBDA_FUNCTION_NAME} --payload '{\"prompt\":\"ArrayResize MQL5\"}' response.json"