# MQL5 RAG Engine Lambda Container
# Task C12: Create Dockerfile for Lambda Container
# Module: RAG Engine

FROM public.ecr.aws/lambda/python:3.10

# Set environment variables for Python
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}
ENV PYTHONUNBUFFERED=1

# Install system dependencies for FAISS
RUN yum update -y && \
    yum install -y gcc gcc-c++ make && \
    yum clean all

# Copy requirements first for better caching
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Lambda function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY rag_engine/ ${LAMBDA_TASK_ROOT}/rag_engine/

# Create cache directory for models and indexes
RUN mkdir -p ${LAMBDA_TASK_ROOT}/cache

# Set the Lambda handler
CMD ["lambda_function.lambda_handler"]