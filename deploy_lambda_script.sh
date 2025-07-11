#!/bin/bash
# Deploy MQL5 RAG Lambda Function
# This creates the exact function your API Gateway expects: mql5-rag-rag-handler

set -e

echo "🚀 Deploying MQL5 RAG Lambda Function"
echo "======================================"

# Configuration
FUNCTION_NAME="mql5-rag-rag-handler"
REGION="us-east-1"
ACCOUNT_ID="193245229238"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/mql5-rag-lambda-execution-role"
PYTHON_FILE="lambda_deployment_package.py"

echo "📋 Configuration:"
echo "   Function Name: $FUNCTION_NAME"
echo "   Region: $REGION"
echo "   Role ARN: $ROLE_ARN"
echo ""

# Step 1: Create deployment package
echo "📦 Step 1: Creating deployment package..."

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "   Working directory: $TEMP_DIR"

# Copy Lambda function code
cp "$PYTHON_FILE" "$TEMP_DIR/lambda_function.py"

# Create simple requirements.txt for basic dependencies
cat > "$TEMP_DIR/requirements.txt" << EOF
boto3>=1.26.0
botocore>=1.29.0
EOF

# Install dependencies (if any)
if [ -f "$TEMP_DIR/requirements.txt" ]; then
    echo "   Installing dependencies..."
    pip install -r "$TEMP_DIR/requirements.txt" -t "$TEMP_DIR/" --quiet
fi

# Create ZIP package
cd "$TEMP_DIR"
ZIP_FILE="lambda-deployment.zip"
zip -r "$ZIP_FILE" . -x "*.pyc" "__pycache__/*" "*.zip" > /dev/null
ZIP_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
echo "   ✅ Created deployment package: $ZIP_SIZE"

# Step 2: Check if function exists
echo ""
echo "🔍 Step 2: Checking if function exists..."

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" > /dev/null 2>&1; then
    echo "   ⚠️ Function exists, updating code..."
    
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file "fileb://$ZIP_FILE" \
        --region "$REGION" > /dev/null
    
    echo "   ✅ Function code updated"
else
    echo "   📝 Function doesn't exist, creating new..."
    
    # Step 3: Create Lambda function
    echo ""
    echo "🔧 Step 3: Creating Lambda function..."
    
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime "python3.9" \
        --role "$ROLE_ARN" \
        --handler "lambda_function.lambda_handler" \
        --zip-file "fileb://$ZIP_FILE" \
        --timeout 30 \
        --memory-size 1024 \
        --region "$REGION" \
        --description "MQL5 RAG middleware Lambda function for prompt enrichment" \
        --environment Variables='{
            "S3_BUCKET_NAME":"mql5-rag-faiss-index-20250106-minimal",
            "DYNAMODB_TABLE_NAME":"mql5-doc-snippets",
            "LOG_LEVEL":"INFO"
        }' > /dev/null
    
    echo "   ✅ Function created successfully"
fi

# Step 4: Configure function settings
echo ""
echo "⚙️ Step 4: Configuring function settings..."

aws lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --timeout 30 \
    --memory-size 1024 \
    --region "$REGION" > /dev/null

echo "   ✅ Function configuration updated"

# Step 5: Test the function
echo ""
echo "🧪 Step 5: Testing Lambda function..."

TEST_PAYLOAD='{"body": "{\"prompt\": \"How do I use ArrayResize() in MQL5?\", \"user\": \"deployment_test\"}"}'

INVOKE_RESULT=$(aws lambda invoke \
    --function-name "$FUNCTION_NAME" \
    --payload "$TEST_PAYLOAD" \
    --region "$REGION" \
    response.json)

STATUS_CODE=$(echo "$INVOKE_RESULT" | grep -o '"StatusCode": [0-9]*' | grep -o '[0-9]*')

if [ "$STATUS_CODE" = "200" ]; then
    echo "   ✅ Lambda function test: SUCCESS"
    
    # Show response preview
    if [ -f "response.json" ]; then
        RESPONSE_PREVIEW=$(cat response.json | head -c 200)
        echo "   📋 Response preview: $RESPONSE_PREVIEW..."
    fi
else
    echo "   ❌ Lambda function test: FAILED (Status: $STATUS_CODE)"
    if [ -f "response.json" ]; then
        echo "   Error details:"
        cat response.json
    fi
fi

# Step 6: Verify API Gateway integration
echo ""
echo "🔗 Step 6: Verifying API Gateway integration..."

API_GATEWAY_URL="https://b6qmhutxnc.execute-api.us-east-1.amazonaws.com/prod/rag"
echo "   Testing endpoint: $API_GATEWAY_URL"

# Test with curl (if available)
if command -v curl > /dev/null; then
    echo "   Making test request..."
    
    CURL_RESPONSE=$(curl -s -w "%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -H "x-api-key: $MQL5_API_KEY" \
        -d '{"prompt": "Test MQL5 ArrayResize function", "user": "integration_test"}' \
        "$API_GATEWAY_URL" 2>/dev/null || echo "000")
    
    HTTP_CODE="${CURL_RESPONSE: -3}"
    RESPONSE_BODY="${CURL_RESPONSE%???}"
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo "   ✅ API Gateway integration: SUCCESS"
        echo "   📋 End-to-end test: PASSED"
    elif [ "$HTTP_CODE" = "403" ]; then
        echo "   ⚠️ API Gateway returned 403 - Check API key"
        echo "   Set MQL5_API_KEY environment variable for full test"
    else
        echo "   ⚠️ API Gateway returned: $HTTP_CODE"
        echo "   Response: $RESPONSE_BODY"
    fi
else
    echo "   ℹ️ curl not available, skipping API Gateway test"
fi

# Cleanup
cd - > /dev/null
rm -rf "$TEMP_DIR"
rm -f response.json

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "======================"
echo ""
echo "✅ Lambda Function: $FUNCTION_NAME"
echo "✅ Runtime: Python 3.9"
echo "✅ Memory: 1024 MB"
echo "✅ Timeout: 30 seconds"
echo "✅ Handler: lambda_function.lambda_handler"
echo ""
echo "🔗 Integration:"
echo "   API Gateway: https://b6qmhutxnc.execute-api.us-east-1.amazonaws.com/prod/rag"
echo "   Target Lambda: arn:aws:lambda:us-east-1:193245229238:function:mql5-rag-rag-handler"
echo ""
echo "🎯 NEXT STEPS:"
echo "   1. Test your PromptProxy again - it should now work!"
echo "   2. Run: python end_to_end_test.py"
echo "   3. Later: Replace mock components with real FAISS/embeddings"
echo ""
echo "ℹ️ NOTE: This deployment uses mock data for rapid deployment."
echo "   Real FAISS index and embedding models will be added in Tasks C14-C17."