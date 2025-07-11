#!/usr/bin/env python3
"""
MQL5 Data Uploader - Task B10
Upload FAISS index to S3 and populate DynamoDB with text snippets

This module completes the data layer by uploading processed MQL5 data
to AWS services for the RAG pipeline.
"""

import os
import json
import boto3
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from botocore.exceptions import ClientError, BotoCoreError
import hashlib


@dataclass
class UploadResult:
    """Result of upload operation"""
    success: bool
    operation: str
    details: Dict[str, Any]
    error_message: Optional[str] = None


class MQL5DataUploader:
    """Upload FAISS index to S3 and text snippets to DynamoDB"""
    
    def __init__(self, 
                 faiss_dir: str = "./mql5_test/faiss_index",
                 chunks_file: str = "./mql5_test/chunks/mql5_chunks.json",
                 s3_bucket: str = "mql5-rag-faiss-index-20250106-minimal",
                 dynamodb_table: str = "mql5-doc-snippets",
                 aws_region: str = "us-east-1"):
        
        self.faiss_dir = Path(faiss_dir)
        self.chunks_file = Path(chunks_file)
        self.s3_bucket = s3_bucket
        self.dynamodb_table = dynamodb_table
        self.aws_region = aws_region
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize AWS clients
        try:
            self.s3_client = boto3.client('s3', region_name=aws_region)
            self.dynamodb_client = boto3.client('dynamodb', region_name=aws_region)
            self.dynamodb_resource = boto3.resource('dynamodb', region_name=aws_region)
            self.logger.info(f"Initialized AWS clients for region: {aws_region}")
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS clients: {e}")
            raise
        
        # Upload statistics
        self.stats = {
            's3_uploads': {
                'files_uploaded': 0,
                'total_size_mb': 0,
                'upload_time': 0,
                'failed_uploads': []
            },
            'dynamodb_inserts': {
                'items_inserted': 0,
                'batch_operations': 0,
                'insert_time': 0,
                'failed_items': []
            },
            'validation': {
                's3_verified': False,
                'dynamodb_verified': False,
                'data_integrity_checked': False
            }
        }
    
    def verify_prerequisites(self) -> bool:
        """Verify all required files and AWS resources exist"""
        try:
            # Check local files
            required_files = ['index.faiss', 'index.pkl', 'index_metadata.json']
            missing_files = []
            
            for file_name in required_files:
                file_path = self.faiss_dir / file_name
                if not file_path.exists():
                    missing_files.append(str(file_path))
            
            if missing_files:
                self.logger.error(f"Missing FAISS files: {missing_files}")
                return False
            
            # Check chunks file
            if not self.chunks_file.exists():
                self.logger.error(f"Chunks file not found: {self.chunks_file}")
                return False
            
            # Verify S3 bucket access
            try:
                self.s3_client.head_bucket(Bucket=self.s3_bucket)
                self.logger.info(f"S3 bucket accessible: {self.s3_bucket}")
            except ClientError as e:
                self.logger.error(f"S3 bucket not accessible: {e}")
                return False
            
            # Verify DynamoDB table access
            try:
                table = self.dynamodb_resource.Table(self.dynamodb_table)
                table.load()
                self.logger.info(f"DynamoDB table accessible: {self.dynamodb_table} (status: {table.table_status})")
            except ClientError as e:
                self.logger.error(f"DynamoDB table not accessible: {e}")
                return False
            
            self.logger.info("All prerequisites verified successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Prerequisites verification failed: {e}")
            return False
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for integrity verification"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def upload_file_to_s3(self, local_path: Path, s3_key: str) -> UploadResult:
        """Upload a single file to S3 with verification"""
        try:
            self.logger.info(f"Uploading {local_path.name} to S3: s3://{self.s3_bucket}/{s3_key}")
            
            # Calculate file size and hash
            file_size = local_path.stat().st_size
            file_hash = self.calculate_file_hash(local_path)
            
            start_time = time.time()
            
            # Upload file with metadata
            extra_args = {
                'Metadata': {
                    'original-filename': local_path.name,
                    'upload-timestamp': datetime.now().isoformat(),
                    'file-hash-sha256': file_hash,
                    'project': 'mql5-rag',
                    'component': 'faiss-index'
                }
            }
            
            self.s3_client.upload_file(
                str(local_path),
                self.s3_bucket,
                s3_key,
                ExtraArgs=extra_args
            )
            
            upload_time = time.time() - start_time
            
            # Verify upload
            try:
                response = self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
                uploaded_size = response['ContentLength']
                
                if uploaded_size != file_size:
                    raise ValueError(f"Size mismatch: local={file_size}, s3={uploaded_size}")
                
                self.logger.info(f"Upload successful: {local_path.name} ({file_size:,} bytes, {upload_time:.2f}s)")
                
                return UploadResult(
                    success=True,
                    operation="s3_upload",
                    details={
                        'local_path': str(local_path),
                        's3_key': s3_key,
                        'file_size': file_size,
                        'upload_time': upload_time,
                        'file_hash': file_hash,
                        's3_url': f"s3://{self.s3_bucket}/{s3_key}"
                    }
                )
                
            except ClientError as e:
                raise ValueError(f"Upload verification failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to upload {local_path.name}: {e}")
            return UploadResult(
                success=False,
                operation="s3_upload",
                details={'local_path': str(local_path), 's3_key': s3_key},
                error_message=str(e)
            )
    
    def upload_faiss_to_s3(self) -> List[UploadResult]:
        """Upload all FAISS index files to S3"""
        self.logger.info("=== Uploading FAISS Index to S3 ===")
        
        files_to_upload = [
            ('index.faiss', 'index.faiss'),
            ('index.pkl', 'index.pkl'),
            ('index_metadata.json', 'metadata/index_metadata.json')
        ]
        
        results = []
        total_size = 0
        total_time = 0
        
        for local_filename, s3_key in files_to_upload:
            local_path = self.faiss_dir / local_filename
            
            if local_path.exists():
                result = self.upload_file_to_s3(local_path, s3_key)
                results.append(result)
                
                if result.success:
                    total_size += result.details['file_size']
                    total_time += result.details['upload_time']
                    self.stats['s3_uploads']['files_uploaded'] += 1
                else:
                    self.stats['s3_uploads']['failed_uploads'].append(local_filename)
            else:
                self.logger.warning(f"File not found: {local_path}")
                results.append(UploadResult(
                    success=False,
                    operation="s3_upload",
                    details={'local_path': str(local_path)},
                    error_message="File not found"
                ))
        
        self.stats['s3_uploads']['total_size_mb'] = total_size / (1024 * 1024)
        self.stats['s3_uploads']['upload_time'] = total_time
        
        successful_uploads = sum(1 for r in results if r.success)
        self.logger.info(f"S3 upload complete: {successful_uploads}/{len(files_to_upload)} files successful")
        self.logger.info(f"Total uploaded: {self.stats['s3_uploads']['total_size_mb']:.2f} MB in {total_time:.2f}s")
        
        return results
    
    def load_chunks_data(self) -> List[Dict]:
        """Load text chunks data for DynamoDB insertion"""
        try:
            self.logger.info(f"Loading chunks data from: {self.chunks_file}")
            
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            self.logger.info(f"Loaded {len(chunks)} chunks for DynamoDB insertion")
            
            # Validate chunk structure
            if chunks:
                sample = chunks[0]
                required_fields = ['doc_id', 'text_content']
                missing_fields = [field for field in required_fields if field not in sample]
                
                if missing_fields:
                    raise ValueError(f"Chunks missing required fields: {missing_fields}")
                
                self.logger.info("Chunk data validation successful")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to load chunks data: {e}")
            return []
    
    def prepare_dynamodb_item(self, chunk: Dict) -> Dict:
        """Prepare chunk data for DynamoDB insertion"""
        # Base item structure
        item = {
            'DocID': {'S': chunk['doc_id']},
            'snippet_text': {'S': chunk['text_content']},
            'source': {'S': chunk.get('source_file', 'unknown')},
        }
        
        # Add metadata
        metadata = {
            'token_count': {'N': str(chunk.get('token_count', 0))},
            'section_hint': {'S': chunk.get('section_hint', 'unknown')},
            'chunk_index': {'N': str(chunk.get('chunk_index', 0))},
            'char_start': {'N': str(chunk.get('char_start', 0))},
            'char_end': {'N': str(chunk.get('char_end', 0))},
            'processed_at': {'S': datetime.now().isoformat()}
        }
        
        # Add optional metadata if present
        if 'metadata' in chunk and chunk['metadata']:
            for key, value in chunk['metadata'].items():
                if isinstance(value, (str, int, float)):
                    if isinstance(value, str):
                        metadata[key] = {'S': str(value)}
                    else:
                        metadata[key] = {'N': str(value)}
        
        item['metadata'] = {'M': metadata}
        
        return item
    
    def batch_insert_to_dynamodb(self, chunks: List[Dict], batch_size: int = 25) -> List[UploadResult]:
        """Insert chunks to DynamoDB using batch operations"""
        self.logger.info(f"=== Inserting {len(chunks)} items to DynamoDB ===")
        
        results = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        inserted_count = 0
        failed_count = 0
        
        start_time = time.time()
        
        for batch_num in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_num:batch_num + batch_size]
            batch_index = (batch_num // batch_size) + 1
            
            self.logger.info(f"Processing batch {batch_index}/{total_batches} ({len(batch_chunks)} items)")
            
            try:
                # Prepare batch write request
                request_items = []
                for chunk in batch_chunks:
                    item = self.prepare_dynamodb_item(chunk)
                    request_items.append({
                        'PutRequest': {
                            'Item': item
                        }
                    })
                
                # Execute batch write
                response = self.dynamodb_client.batch_write_item(
                    RequestItems={
                        self.dynamodb_table: request_items
                    }
                )
                
                # Handle unprocessed items
                unprocessed = response.get('UnprocessedItems', {})
                if unprocessed:
                    self.logger.warning(f"Batch {batch_index}: {len(unprocessed)} unprocessed items")
                    # Could implement retry logic here
                
                batch_inserted = len(batch_chunks) - len(unprocessed.get(self.dynamodb_table, []))
                inserted_count += batch_inserted
                
                self.logger.info(f"Batch {batch_index} complete: {batch_inserted}/{len(batch_chunks)} items inserted")
                
                results.append(UploadResult(
                    success=True,
                    operation="dynamodb_batch_insert",
                    details={
                        'batch_number': batch_index,
                        'items_requested': len(batch_chunks),
                        'items_inserted': batch_inserted,
                        'unprocessed_count': len(unprocessed.get(self.dynamodb_table, []))
                    }
                ))
                
                self.stats['dynamodb_inserts']['batch_operations'] += 1
                
                # Small delay between batches to avoid throttling
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Batch {batch_index} failed: {e}")
                failed_count += len(batch_chunks)
                
                results.append(UploadResult(
                    success=False,
                    operation="dynamodb_batch_insert",
                    details={'batch_number': batch_index, 'items_requested': len(batch_chunks)},
                    error_message=str(e)
                ))
        
        total_time = time.time() - start_time
        
        self.stats['dynamodb_inserts']['items_inserted'] = inserted_count
        self.stats['dynamodb_inserts']['insert_time'] = total_time
        
        self.logger.info(f"DynamoDB insertion complete:")
        self.logger.info(f"  - Inserted: {inserted_count}/{len(chunks)} items")
        self.logger.info(f"  - Failed: {failed_count} items")
        self.logger.info(f"  - Time: {total_time:.2f}s")
        self.logger.info(f"  - Rate: {inserted_count/total_time:.1f} items/sec")
        
        return results
    
    def verify_s3_uploads(self) -> bool:
        """Verify all FAISS files are correctly uploaded to S3"""
        try:
            self.logger.info("Verifying S3 uploads...")
            
            required_objects = ['index.faiss', 'index.pkl']
            verified_objects = []
            
            for s3_key in required_objects:
                try:
                    response = self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
                    size = response['ContentLength']
                    last_modified = response['LastModified']
                    
                    verified_objects.append({
                        'key': s3_key,
                        'size': size,
                        'last_modified': last_modified.isoformat()
                    })
                    
                    self.logger.info(f"✅ {s3_key}: {size:,} bytes")
                    
                except ClientError as e:
                    self.logger.error(f"❌ {s3_key}: {e}")
                    return False
            
            self.stats['validation']['s3_verified'] = True
            self.logger.info(f"S3 verification complete: {len(verified_objects)}/{len(required_objects)} objects verified")
            return True
            
        except Exception as e:
            self.logger.error(f"S3 verification failed: {e}")
            return False
    
    def verify_dynamodb_data(self, expected_count: int) -> bool:
        """Verify DynamoDB data insertion"""
        try:
            self.logger.info("Verifying DynamoDB data...")
            
            table = self.dynamodb_resource.Table(self.dynamodb_table)
            
            # Get item count (note: this might be approximate for large tables)
            response = table.scan(Select='COUNT')
            actual_count = response['Count']
            
            # Scan for more accurate count if needed (for smaller datasets)
            while 'LastEvaluatedKey' in response:
                response = table.scan(
                    Select='COUNT',
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                actual_count += response['Count']
            
            self.logger.info(f"DynamoDB item count: {actual_count} (expected: {expected_count})")
            
            # Sample a few items to verify structure
            sample_response = table.scan(Limit=3)
            sample_items = sample_response.get('Items', [])
            
            if sample_items:
                sample_item = sample_items[0]
                required_fields = ['DocID', 'snippet_text', 'source', 'metadata']
                
                for field in required_fields:
                    if field not in sample_item:
                        self.logger.error(f"Missing required field in DynamoDB: {field}")
                        return False
                
                self.logger.info(f"Sample item validation successful:")
                self.logger.info(f"  - DocID: {sample_item['DocID']}")
                self.logger.info(f"  - Text length: {len(sample_item['snippet_text'])} chars")
                self.logger.info(f"  - Source: {sample_item.get('source', 'N/A')}")
            
            verification_success = (actual_count == expected_count and sample_items)
            self.stats['validation']['dynamodb_verified'] = verification_success
            
            if verification_success:
                self.logger.info("✅ DynamoDB verification successful")
            else:
                self.logger.error("❌ DynamoDB verification failed")
            
            return verification_success
            
        except Exception as e:
            self.logger.error(f"DynamoDB verification failed: {e}")
            return False
    
    def verify_data_integrity(self) -> bool:
        """Verify data integrity between FAISS index and DynamoDB"""
        try:
            self.logger.info("Verifying data integrity...")
            
            # Load DocID mappings from FAISS metadata
            metadata_path = self.faiss_dir / 'index_metadata.json'
            if not metadata_path.exists():
                self.logger.error("FAISS metadata file not found")
                return False
            
            with open(metadata_path, 'r') as f:
                faiss_metadata = json.load(f)
            
            docid_mappings = faiss_metadata.get('docid_mappings', {})
            index_to_docid = docid_mappings.get('index_to_docid', {})
            
            if not index_to_docid:
                self.logger.error("No DocID mappings found in FAISS metadata")
                return False
            
            # Sample verification: check a few DocIDs exist in DynamoDB
            table = self.dynamodb_resource.Table(self.dynamodb_table)
            sample_docids = list(index_to_docid.values())[:10]  # Test first 10
            
            found_count = 0
            for docid in sample_docids:
                try:
                    response = table.get_item(Key={'DocID': docid})
                    if 'Item' in response:
                        found_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to check DocID {docid}: {e}")
            
            integrity_score = found_count / len(sample_docids)
            self.logger.info(f"Data integrity check: {found_count}/{len(sample_docids)} DocIDs found ({integrity_score:.1%})")
            
            integrity_success = integrity_score >= 0.95  # 95% threshold
            self.stats['validation']['data_integrity_checked'] = integrity_success
            
            if integrity_success:
                self.logger.info("✅ Data integrity verification successful")
            else:
                self.logger.error("❌ Data integrity verification failed")
            
            return integrity_success
            
        except Exception as e:
            self.logger.error(f"Data integrity verification failed: {e}")
            return False
    
    def run_complete_upload(self) -> Dict[str, Any]:
        """Run the complete upload pipeline"""
        pipeline_results = {
            'success': False,
            'prerequisites_verified': False,
            's3_upload_results': [],
            'dynamodb_upload_results': [],
            'verification_results': {},
            'stats': {},
            'summary': {}
        }
        
        try:
            # Step 1: Verify prerequisites
            self.logger.info("=== Step 1: Verifying Prerequisites ===")
            if not self.verify_prerequisites():
                return pipeline_results
            pipeline_results['prerequisites_verified'] = True
            
            # Step 2: Upload FAISS index to S3
            self.logger.info("=== Step 2: Uploading FAISS Index to S3 ===")
            s3_results = self.upload_faiss_to_s3()
            pipeline_results['s3_upload_results'] = s3_results
            
            # Step 3: Load and upload chunks to DynamoDB
            self.logger.info("=== Step 3: Loading Chunks Data ===")
            chunks = self.load_chunks_data()
            if not chunks:
                self.logger.error("No chunks data loaded")
                return pipeline_results
            
            self.logger.info("=== Step 4: Uploading to DynamoDB ===")
            dynamodb_results = self.batch_insert_to_dynamodb(chunks)
            pipeline_results['dynamodb_upload_results'] = dynamodb_results
            
            # Step 5: Verification
            self.logger.info("=== Step 5: Verification ===")
            verification = {
                's3_verified': self.verify_s3_uploads(),
                'dynamodb_verified': self.verify_dynamodb_data(len(chunks)),
                'data_integrity_verified': self.verify_data_integrity()
            }
            pipeline_results['verification_results'] = verification
            
            # Final statistics and summary
            pipeline_results['stats'] = self.stats
            
            all_verifications_passed = all(verification.values())
            s3_success = all(r.success for r in s3_results)
            dynamodb_success = self.stats['dynamodb_inserts']['items_inserted'] > 0
            
            pipeline_results['success'] = all_verifications_passed and s3_success and dynamodb_success
            
            # Summary
            pipeline_results['summary'] = {
                's3_files_uploaded': self.stats['s3_uploads']['files_uploaded'],
                's3_total_size_mb': self.stats['s3_uploads']['total_size_mb'],
                'dynamodb_items_inserted': self.stats['dynamodb_inserts']['items_inserted'],
                'total_upload_time': (self.stats['s3_uploads']['upload_time'] + 
                                    self.stats['dynamodb_inserts']['insert_time']),
                'all_verifications_passed': all_verifications_passed,
                'lambda_ready': all_verifications_passed and s3_success and dynamodb_success
            }
            
            if pipeline_results['success']:
                self.logger.info("=== Upload Pipeline Complete ===")
                self.logger.info(f"✅ S3 uploads: {self.stats['s3_uploads']['files_uploaded']} files")
                self.logger.info(f"✅ DynamoDB inserts: {self.stats['dynamodb_inserts']['items_inserted']} items")
                self.logger.info(f"✅ All verifications passed")
                self.logger.info(f"🚀 System ready for Lambda deployment")
            else:
                self.logger.error("❌ Upload pipeline completed with errors")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Upload pipeline failed: {e}")
            return pipeline_results


def main():
    """Main function for data upload"""
    
    print("=== MQL5 Data Uploader (Task B10) ===")
    
    # Initialize uploader
    uploader = MQL5DataUploader(
        faiss_dir="./mql5_test/faiss_index",
        chunks_file="./mql5_test/chunks/mql5_chunks.json",
        s3_bucket="mql5-rag-faiss-index-20250106-minimal",
        dynamodb_table="mql5-doc-snippets",
        aws_region="us-east-1"
    )
    
    # Run complete upload pipeline
    results = uploader.run_complete_upload()
    
    # Display results
    print("\n=== Upload Results ===")
    if results['success']:
        summary = results['summary']
        print(f"✅ Upload completed successfully!")
        print(f"📁 S3 files uploaded: {summary['s3_files_uploaded']} ({summary['s3_total_size_mb']:.2f} MB)")
        print(f"📄 DynamoDB items: {summary['dynamodb_items_inserted']}")
        print(f"⏱️ Total time: {summary['total_upload_time']:.2f}s")
        print(f"🔍 Verifications: {'✅ All passed' if summary['all_verifications_passed'] else '❌ Some failed'}")
        print(f"🚀 Lambda ready: {'Yes' if summary['lambda_ready'] else 'No'}")
    else:
        print("❌ Upload failed. Check logs for details.")
        if results.get('stats'):
            print(f"📊 Partial results:")
            print(f"   - S3 files: {results['stats']['s3_uploads']['files_uploaded']}")
            print(f"   - DynamoDB items: {results['stats']['dynamodb_inserts']['items_inserted']}")
    
    return results


if __name__ == "__main__":
    main()