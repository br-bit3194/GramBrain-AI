"""S3 client for file storage operations with retry logic and validation."""

import asyncio
import hashlib
import mimetypes
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, BinaryIO
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)


@dataclass
class FileValidationConfig:
    """Configuration for file validation."""
    max_file_size_mb: int = 50
    allowed_content_types: set = None
    
    def __post_init__(self):
        """Set default allowed content types if not provided."""
        if self.allowed_content_types is None:
            self.allowed_content_types = {
                'image/jpeg',
                'image/png',
                'image/gif',
                'image/webp',
                'application/pdf',
                'text/plain',
                'text/csv',
                'application/json',
            }


class S3Client:
    """S3 client for file operations with bucket organization and validation.
    
    Bucket structure:
        grambrain-assets-{env}/
        ├── user-uploads/
        │   ├── {user_id}/
        │   │   ├── profile-images/
        │   │   └── farm-images/
        ├── product-images/
        │   └── {product_id}/
        ├── satellite-data/
        │   └── {farm_id}/
        │       └── {date}/
        └── knowledge-documents/
            └── {topic}/
    """
    
    def __init__(
        self,
        bucket_name: str,
        region_name: str = 'us-east-1',
        endpoint_url: Optional[str] = None,
        validation_config: Optional[FileValidationConfig] = None
    ):
        """Initialize S3 client.
        
        Args:
            bucket_name: S3 bucket name (e.g., 'gram-brain-bucket')
            region_name: AWS region name
            endpoint_url: Custom endpoint URL (for LocalStack)
            validation_config: File validation configuration
        """
        # Create boto3 client
        client_kwargs = {'region_name': region_name}
        if endpoint_url:
            client_kwargs['endpoint_url'] = endpoint_url
        
        self.s3_client = boto3.client('s3', **client_kwargs)
        self.bucket_name = bucket_name
        self.validation_config = validation_config or FileValidationConfig()
    
    def _get_file_path(
        self,
        file_type: str,
        identifier: str,
        filename: str,
        date: Optional[datetime] = None
    ) -> str:
        """Generate organized file path based on type and date.
        
        Args:
            file_type: Type of file (user-uploads, product-images, satellite-data, knowledge-documents)
            identifier: User ID, product ID, farm ID, or topic
            filename: Original filename
            date: Optional date for organization (used for satellite-data)
            
        Returns:
            Organized S3 key path
        """
        # Sanitize filename
        safe_filename = filename.replace(' ', '_').replace('..', '')
        
        if file_type == 'user-uploads':
            # user-uploads/{user_id}/profile-images/ or farm-images/
            subtype = 'profile-images' if 'profile' in filename.lower() else 'farm-images'
            return f"user-uploads/{identifier}/{subtype}/{safe_filename}"
        
        elif file_type == 'product-images':
            # product-images/{product_id}/
            return f"product-images/{identifier}/{safe_filename}"
        
        elif file_type == 'satellite-data':
            # satellite-data/{farm_id}/{date}/
            date_str = (date or datetime.now()).strftime('%Y-%m-%d')
            return f"satellite-data/{identifier}/{date_str}/{safe_filename}"
        
        elif file_type == 'knowledge-documents':
            # knowledge-documents/{topic}/
            return f"knowledge-documents/{identifier}/{safe_filename}"
        
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
    def _validate_file(
        self,
        file_content: bytes,
        content_type: str,
        filename: str
    ) -> None:
        """Validate file before upload.
        
        Args:
            file_content: File content as bytes
            content_type: MIME type of the file
            filename: Original filename
            
        Raises:
            ValueError: If validation fails
        """
        # Check file size
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > self.validation_config.max_file_size_mb:
            raise ValueError(
                f"File size {file_size_mb:.2f}MB exceeds maximum "
                f"{self.validation_config.max_file_size_mb}MB"
            )
        
        # Check content type
        if content_type not in self.validation_config.allowed_content_types:
            raise ValueError(
                f"Content type '{content_type}' not allowed. "
                f"Allowed types: {self.validation_config.allowed_content_types}"
            )
        
        # Basic malware check: reject files with suspicious patterns
        # In production, integrate with AWS GuardDuty or third-party scanner
        suspicious_patterns = [b'<script', b'<?php', b'eval(']
        content_lower = file_content[:1024].lower()  # Check first 1KB
        for pattern in suspicious_patterns:
            if pattern in content_lower:
                logger.warning(f"Suspicious pattern detected in file: {filename}")
                raise ValueError("File contains suspicious content")
    
    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        file_type: str,
        identifier: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        date: Optional[datetime] = None
    ) -> str:
        """Upload file to S3 with validation and organization.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            file_type: Type of file (user-uploads, product-images, etc.)
            identifier: User ID, product ID, farm ID, or topic
            content_type: MIME type (auto-detected if not provided)
            metadata: Optional metadata to attach to the file
            date: Optional date for organization
            
        Returns:
            S3 key of uploaded file
            
        Raises:
            ValueError: If validation fails
            ClientError: If upload fails
        """
        # Auto-detect content type if not provided
        if content_type is None:
            content_type, _ = mimetypes.guess_type(filename)
            if content_type is None:
                content_type = 'application/octet-stream'
        
        # Validate file
        self._validate_file(file_content, content_type, filename)
        
        # Generate organized path
        s3_key = self._get_file_path(file_type, identifier, filename, date)
        
        # Prepare upload parameters
        upload_params = {
            'Bucket': self.bucket_name,
            'Key': s3_key,
            'Body': file_content,
            'ContentType': content_type,
        }
        
        # Add metadata if provided
        if metadata:
            upload_params['Metadata'] = metadata
        
        # Upload file asynchronously
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.put_object(**upload_params)
            )
            logger.info(f"Successfully uploaded file to s3://{self.bucket_name}/{s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Failed to upload file to S3: {e}")
            raise
    
    async def generate_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600,
        http_method: str = 'get_object'
    ) -> str:
        """Generate presigned URL for secure file access.
        
        Args:
            s3_key: S3 key of the file
            expiration: URL expiration time in seconds (default: 1 hour)
            http_method: HTTP method ('get_object' or 'put_object')
            
        Returns:
            Presigned URL
            
        Raises:
            ClientError: If URL generation fails
        """
        loop = asyncio.get_event_loop()
        try:
            url = await loop.run_in_executor(
                None,
                lambda: self.s3_client.generate_presigned_url(
                    http_method,
                    Params={'Bucket': self.bucket_name, 'Key': s3_key},
                    ExpiresIn=expiration
                )
            )
            logger.info(f"Generated presigned URL for s3://{self.bucket_name}/{s3_key}")
            return url
            
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise
    
    async def download_file(self, s3_key: str) -> bytes:
        """Download file from S3.
        
        Args:
            s3_key: S3 key of the file
            
        Returns:
            File content as bytes
            
        Raises:
            ClientError: If download fails
        """
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
            )
            content = response['Body'].read()
            logger.info(f"Successfully downloaded file from s3://{self.bucket_name}/{s3_key}")
            return content
            
        except ClientError as e:
            logger.error(f"Failed to download file from S3: {e}")
            raise
    
    async def delete_file(self, s3_key: str) -> None:
        """Delete file from S3.
        
        Args:
            s3_key: S3 key of the file
            
        Raises:
            ClientError: If deletion fails
        """
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
            )
            logger.info(f"Successfully deleted file from s3://{self.bucket_name}/{s3_key}")
            
        except ClientError as e:
            logger.error(f"Failed to delete file from S3: {e}")
            raise
    
    async def list_files(
        self,
        prefix: str,
        max_keys: int = 1000
    ) -> list:
        """List files in S3 with given prefix.
        
        Args:
            prefix: S3 key prefix to filter files
            max_keys: Maximum number of keys to return
            
        Returns:
            List of S3 keys
            
        Raises:
            ClientError: If listing fails
        """
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix,
                    MaxKeys=max_keys
                )
            )
            
            keys = [obj['Key'] for obj in response.get('Contents', [])]
            logger.info(f"Listed {len(keys)} files with prefix '{prefix}'")
            return keys
            
        except ClientError as e:
            logger.error(f"Failed to list files from S3: {e}")
            raise
    
    async def configure_lifecycle_policy(self) -> None:
        """Configure S3 lifecycle policies for cost optimization.
        
        Lifecycle rules:
        - Move satellite-data older than 90 days to Glacier
        - Delete temporary uploads older than 7 days
        - Move knowledge-documents older than 180 days to Glacier
        """
        lifecycle_config = {
            'Rules': [
                {
                    'Id': 'archive-satellite-data',
                    'Status': 'Enabled',
                    'Prefix': 'satellite-data/',
                    'Transitions': [
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                },
                {
                    'Id': 'delete-temp-uploads',
                    'Status': 'Enabled',
                    'Prefix': 'temp/',
                    'Expiration': {
                        'Days': 7
                    }
                },
                {
                    'Id': 'archive-knowledge-docs',
                    'Status': 'Enabled',
                    'Prefix': 'knowledge-documents/',
                    'Transitions': [
                        {
                            'Days': 180,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                }
            ]
        }
        
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.put_bucket_lifecycle_configuration(
                    Bucket=self.bucket_name,
                    LifecycleConfiguration=lifecycle_config
                )
            )
            logger.info(f"Successfully configured lifecycle policies for bucket {self.bucket_name}")
            
        except ClientError as e:
            logger.error(f"Failed to configure lifecycle policies: {e}")
            raise
