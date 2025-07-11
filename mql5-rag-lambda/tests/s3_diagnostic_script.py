#!/usr/bin/env python3
"""
S3 Bucket Diagnostic Script
Checks exactly what files exist in the MQL5 RAG S3 bucket
and attempts to download and analyze them
"""

import boto3
import json
import os
from pathlib import Path

def diagnose_s3_bucket():
    """Diagnose S3 bucket contents and file structures"""
    
    bucket_name = "mql5-rag-faiss-index-20250106-minimal"
    
    print(f"=== S3 Bucket Diagnostic: {bucket_name} ===")
    
    try:
        s3_client = boto3.client('s3')
        
        # List all objects in bucket
        print("\n1. Listing ALL objects in bucket:")
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' not in response:
            print("❌ Bucket is empty or inaccessible")
            return
        
        files = []
        for obj in response['Contents']:
            key = obj['Key']
            size = obj['Size']
            modified = obj['LastModified']
            files.append((key, size, modified))
            print(f"   📄 {key} ({size:,} bytes) - {modified}")
        
        print(f"\nTotal files found: {len(files)}")
        
        # Download and analyze each file
        print("\n2. Analyzing file contents:")
        
        download_dir = Path("./s3_analysis")
        download_dir.mkdir(exist_ok=True)
        
        for file_key, file_size, _ in files:
            print(f"\n--- Analyzing: {file_key} ---")
            
            local_path = download_dir / file_key.replace('/', '_')
            
            try:
                # Download file
                s3_client.download_file(bucket_name, file_key, str(local_path))
                print(f"✅ Downloaded to: {local_path}")
                
                # Analyze file content
                if file_key.endswith('.json'):
                    analyze_json_file(local_path, file_key)
                elif file_key.endswith('.faiss'):
                    analyze_faiss_file(local_path, file_key)
                elif file_key.endswith('.pkl'):
                    analyze_pickle_file(local_path, file_key)
                else:
                    print(f"   📋 File type: {file_key.split('.')[-1] if '.' in file_key else 'unknown'}")
                    
            except Exception as e:
                print(f"❌ Failed to download {file_key}: {e}")
        
        print(f"\n3. Files downloaded to: {download_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ S3 diagnostic failed: {e}")


def analyze_json_file(file_path: Path, original_key: str):
    """Analyze JSON file structure"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"   📋 JSON file structure:")
        
        if isinstance(data, dict):
            print(f"   📊 Top-level keys: {list(data.keys())}")
            
            # Look for mapping-like structures
            for key, value in data.items():
                if isinstance(value, dict) and len(value) > 0:
                    sample_key = next(iter(value.keys()))
                    sample_value = value[sample_key]
                    print(f"   🔍 '{key}': dict with {len(value)} entries (sample: {sample_key} -> {str(sample_value)[:50]})")
                elif isinstance(value, list):
                    print(f"   🔍 '{key}': list with {len(value)} items")
                else:
                    print(f"   🔍 '{key}': {type(value).__name__} = {str(value)[:50]}")
        
        elif isinstance(data, list):
            print(f"   📊 JSON list with {len(data)} items")
            if data:
                print(f"   🔍 First item type: {type(data[0]).__name__}")
        
        # Check for doc ID mapping patterns
        potential_mappings = []
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and any(k.isdigit() for k in value.keys()):
                    potential_mappings.append(key)
        
        if potential_mappings:
            print(f"   🎯 Potential doc ID mappings found: {potential_mappings}")
        
    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON: {e}")
    except Exception as e:
        print(f"   ❌ JSON analysis failed: {e}")


def analyze_faiss_file(file_path: Path, original_key: str):
    """Analyze FAISS index file"""
    try:
        import faiss
        
        index = faiss.read_index(str(file_path))
        print(f"   📋 FAISS index loaded successfully")
        print(f"   📊 Vectors: {index.ntotal}")
        print(f"   📊 Dimensions: {index.d}")
        print(f"   📊 Index type: {type(index).__name__}")
        print(f"   📊 Trained: {getattr(index, 'is_trained', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ FAISS analysis failed: {e}")


def analyze_pickle_file(file_path: Path, original_key: str):
    """Analyze pickle file"""
    try:
        import pickle
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"   📋 Pickle file loaded successfully")
        print(f"   📊 Data type: {type(data).__name__}")
        
        if isinstance(data, dict):
            print(f"   📊 Keys: {list(data.keys())}")
            
            # Look for mapping structures
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"   🔍 '{key}': dict with {len(value)} entries")
                    if len(value) > 0:
                        sample_key = next(iter(value.keys()))
                        sample_value = value[sample_key]
                        print(f"       Sample: {sample_key} -> {str(sample_value)[:50]}")
                else:
                    print(f"   🔍 '{key}': {type(value).__name__}")
        
    except Exception as e:
        print(f"   ❌ Pickle analysis failed: {e}")


if __name__ == "__main__":
    diagnose_s3_bucket()
