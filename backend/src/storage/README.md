# Storage Module

This module provides S3 file storage integration for the GramBrain AI platform.

## Features

- **Organized Bucket Structure**: Files are automatically organized by type and date
- **File Validation**: Validates file size, content type, and scans for suspicious content
- **Presigned URLs**: Generates secure, time-limited URLs for file access
- **Lifecycle Policies**: Automatically archives old files to reduce costs
- **Async Operations**: All operations are async for better performance

## Bucket Organization

```
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
```

## Usage

### Initialize Client

```python
from src.storage import S3Client
from src.config import config

s3_client = S3Client(
    bucket_name=config.s3.bucket_name,
    region_name=config.aws.region
)
```

### Upload File

```python
# Upload user profile image
s3_key = await s3_client.upload_file(
    file_content=image_bytes,
    filename='profile.jpg',
    file_type='user-uploads',
    identifier=user_id,
    content_type='image/jpeg'
)

# Upload product image
s3_key = await s3_client.upload_file(
    file_content=image_bytes,
    filename='product.jpg',
    file_type='product-images',
    identifier=product_id
)

# Upload satellite data
s3_key = await s3_client.upload_file(
    file_content=data_bytes,
    filename='ndvi.tif',
    file_type='satellite-data',
    identifier=farm_id,
    date=datetime(2026, 1, 15)
)
```

### Generate Presigned URL

```python
# Generate URL for downloading (valid for 1 hour)
url = await s3_client.generate_presigned_url(
    s3_key='product-images/prod-123/image.jpg',
    expiration=3600
)

# Generate URL for uploading
url = await s3_client.generate_presigned_url(
    s3_key='user-uploads/user-456/farm-images/field.jpg',
    expiration=300,  # 5 minutes
    http_method='put_object'
)
```

### Download File

```python
content = await s3_client.download_file(
    s3_key='knowledge-documents/rice/guide.pdf'
)
```

### List Files

```python
# List all product images for a product
keys = await s3_client.list_files(
    prefix='product-images/prod-123/'
)

# List satellite data for a farm
keys = await s3_client.list_files(
    prefix='satellite-data/farm-789/2026-01-15/'
)
```

### Configure Lifecycle Policies

```python
# Configure lifecycle policies for cost optimization
await s3_client.configure_lifecycle_policy()
```

## File Validation

The S3 client validates files before upload:

1. **Size Validation**: Files must be under the configured maximum size (default: 50MB)
2. **Content Type Validation**: Only allowed MIME types are accepted
3. **Malware Scanning**: Basic pattern matching for suspicious content

### Allowed Content Types

By default, the following content types are allowed:
- `image/jpeg`
- `image/png`
- `image/gif`
- `image/webp`
- `application/pdf`
- `text/plain`
- `text/csv`
- `application/json`

You can customize allowed types:

```python
from src.storage.s3_client import FileValidationConfig

validation_config = FileValidationConfig(
    max_file_size_mb=100,
    allowed_content_types={'image/jpeg', 'image/png', 'application/pdf'}
)

s3_client = S3Client(
    bucket_name='my-bucket',
    validation_config=validation_config
)
```

## Lifecycle Policies

The client configures the following lifecycle policies:

1. **Satellite Data**: Moved to Glacier after 90 days
2. **Temporary Uploads**: Deleted after 7 days
3. **Knowledge Documents**: Moved to Glacier after 180 days

## Error Handling

All operations raise `ClientError` from boto3 on failure. Handle errors appropriately:

```python
from botocore.exceptions import ClientError

try:
    s3_key = await s3_client.upload_file(...)
except ValueError as e:
    # Validation error
    logger.error(f"File validation failed: {e}")
except ClientError as e:
    # AWS error
    logger.error(f"S3 operation failed: {e}")
```

## Configuration

Set the following environment variables:

```bash
S3_BUCKET_NAME=gram-brain-bucket
S3_MAX_FILE_SIZE_MB=50
S3_PRESIGNED_URL_EXPIRATION=3600
```

## Testing

See `tests/test_s3_properties.py` for property-based tests that validate:
- File organization by type and date
- Presigned URL generation with expiration
- File validation rules
