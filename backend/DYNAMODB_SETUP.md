# DynamoDB Setup Guide

This guide explains how to configure and connect to DynamoDB in different environments.

## Understanding DynamoDB Connection

Unlike traditional databases, **DynamoDB doesn't have a single URL**. Instead, it uses:
- **AWS Region**: Determines which AWS data center to connect to
- **AWS Credentials**: Your access key and secret key for authentication
- **Endpoint URL** (optional): Only needed for local development

## Configuration Options

### 1. Production (AWS DynamoDB)

For production, you connect to AWS DynamoDB service:

```bash
# .env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_actual_access_key
AWS_SECRET_ACCESS_KEY=your_actual_secret_key

DYNAMODB_ENV=production
DYNAMODB_ENDPOINT_URL=  # Leave empty for AWS
```

**How it works:**
- boto3 automatically connects to: `https://dynamodb.us-east-1.amazonaws.com`
- The URL is constructed from the region
- No explicit URL needed

### 2. Local Development (DynamoDB Local)

For local testing without AWS costs:

```bash
# .env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=dummy  # Any value works locally
AWS_SECRET_ACCESS_KEY=dummy

DYNAMODB_ENV=dev
DYNAMODB_ENDPOINT_URL=http://localhost:8000
```

**Setup DynamoDB Local:**

```bash
# Using Docker
docker run -p 8000:8000 amazon/dynamodb-local

# Or download JAR
java -Djava.library.path=./DynamoDBLocal_lib -jar DynamoDBLocal.jar -sharedDb
```

### 3. LocalStack (Full AWS Emulation)

For testing multiple AWS services locally:

```bash
# .env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test

DYNAMODB_ENV=dev
DYNAMODB_ENDPOINT_URL=http://localhost:4566
```

**Setup LocalStack:**

```bash
# Using Docker Compose
docker-compose up localstack
```

### 4. Staging Environment

```bash
# .env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=staging_access_key
AWS_SECRET_ACCESS_KEY=staging_secret_key

DYNAMODB_ENV=staging
DYNAMODB_ENDPOINT_URL=  # Empty for AWS
```

## Environment Variables Reference

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `AWS_REGION` | AWS region for DynamoDB | `us-east-1` | Yes |
| `AWS_ACCESS_KEY_ID` | AWS access key | `AKIAIOSFODNN7EXAMPLE` | Yes |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` | Yes |
| `DYNAMODB_ENV` | Environment name (affects table names) | `dev`, `staging`, `production` | Yes |
| `DYNAMODB_ENDPOINT_URL` | Custom endpoint (local only) | `http://localhost:8000` | No |
| `DYNAMODB_RETRY_MAX_ATTEMPTS` | Max retry attempts | `3` | No |
| `DYNAMODB_RETRY_BASE_DELAY_MS` | Initial retry delay | `100` | No |
| `DYNAMODB_RETRY_MAX_DELAY_MS` | Max retry delay | `5000` | No |

## Table Naming Convention

Tables are named with environment suffix:
- Dev: `grambrain-users-dev`, `grambrain-farms-dev`
- Staging: `grambrain-users-staging`, `grambrain-farms-staging`
- Production: `grambrain-users-production`, `grambrain-farms-production`

## Usage Examples

### Basic Usage

```python
from data.client_factory import create_dynamodb_client, create_user_repository

# Automatically uses environment variables
client = create_dynamodb_client()
user_repo = create_user_repository()

# Create a user
user = User(user_id='123', phone_number='+919876543210', name='Ramesh')
await user_repo.create_user(user)
```

### Explicit Configuration

```python
# Production
client = create_dynamodb_client(
    region_name='us-east-1',
    endpoint_url=None  # Uses AWS DynamoDB
)

# Local development
client = create_dynamodb_client(
    region_name='us-east-1',
    endpoint_url='http://localhost:8000'
)

# LocalStack
client = create_dynamodb_client(
    region_name='us-east-1',
    endpoint_url='http://localhost:4566'
)
```

### Using Singleton Pattern

```python
from data.client_factory import get_user_repository

# Get cached repository instance
user_repo = get_user_repository()
```

## Initialize Tables

### For AWS DynamoDB

```bash
# Set environment
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export DYNAMODB_ENV=dev

# Create tables
python backend/src/data/table_definitions.py dev us-east-1
```

### For DynamoDB Local

```bash
# Start DynamoDB Local
docker run -d -p 8000:8000 amazon/dynamodb-local

# Set environment
export AWS_REGION=us-east-1
export DYNAMODB_ENDPOINT_URL=http://localhost:8000
export DYNAMODB_ENV=dev

# Create tables
python backend/src/data/table_definitions.py dev us-east-1
```

## AWS Regions

Common AWS regions for DynamoDB:

| Region Code | Location |
|-------------|----------|
| `us-east-1` | US East (N. Virginia) |
| `us-west-2` | US West (Oregon) |
| `ap-south-1` | Asia Pacific (Mumbai) |
| `ap-southeast-1` | Asia Pacific (Singapore) |
| `eu-west-1` | Europe (Ireland) |

**For India-based deployment, use:** `ap-south-1` (Mumbai)

## Getting AWS Credentials

### Option 1: IAM User (Recommended for Development)

1. Go to AWS Console → IAM → Users
2. Create new user with programmatic access
3. Attach policy: `AmazonDynamoDBFullAccess`
4. Save Access Key ID and Secret Access Key

### Option 2: IAM Role (Recommended for Production)

1. Create IAM role with DynamoDB permissions
2. Attach role to EC2/ECS/Lambda
3. No credentials needed in code (automatic)

### Option 3: AWS CLI Profile

```bash
# Configure AWS CLI
aws configure

# Use profile in code
export AWS_PROFILE=grambrain
```

## Security Best Practices

1. **Never commit credentials** to git
2. **Use IAM roles** in production (no hardcoded keys)
3. **Rotate credentials** regularly
4. **Use least privilege** - only grant needed permissions
5. **Enable encryption** at rest and in transit
6. **Use AWS Secrets Manager** for production credentials

## Troubleshooting

### Connection Issues

```python
# Test connection
import boto3

dynamodb = boto3.resource('dynamodb', 
    region_name='us-east-1',
    endpoint_url='http://localhost:8000'  # or None for AWS
)

# List tables
tables = list(dynamodb.tables.all())
print(f"Tables: {[t.name for t in tables]}")
```

### Common Errors

**Error: "Unable to locate credentials"**
- Solution: Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

**Error: "Could not connect to the endpoint URL"**
- Solution: Check `DYNAMODB_ENDPOINT_URL` is correct
- For AWS: Remove or leave empty
- For local: Ensure DynamoDB Local is running

**Error: "ResourceNotFoundException"**
- Solution: Tables don't exist, run table creation script

**Error: "ProvisionedThroughputExceededException"**
- Solution: Increase table capacity or use on-demand billing

## Docker Compose Example

```yaml
version: '3.8'

services:
  dynamodb-local:
    image: amazon/dynamodb-local
    ports:
      - "8000:8000"
    command: "-jar DynamoDBLocal.jar -sharedDb -dbPath /data"
    volumes:
      - dynamodb-data:/data
  
  backend:
    build: ./backend
    environment:
      - AWS_REGION=us-east-1
      - DYNAMODB_ENDPOINT_URL=http://dynamodb-local:8000
      - DYNAMODB_ENV=dev
    depends_on:
      - dynamodb-local

volumes:
  dynamodb-data:
```

## Cost Considerations

### AWS DynamoDB Pricing

- **On-Demand**: Pay per request (~$1.25 per million writes)
- **Provisioned**: Pay for capacity (~$0.47 per WCU/month)
- **Storage**: $0.25 per GB/month
- **Free Tier**: 25 GB storage, 25 WCU, 25 RCU

### Recommendations

- **Development**: Use DynamoDB Local (free)
- **Staging**: Use on-demand mode
- **Production**: Use provisioned with auto-scaling

## Next Steps

1. Set up your `.env` file based on your environment
2. Initialize DynamoDB tables
3. Test connection with the provided examples
4. Review the [Data Access Layer README](src/data/README.md)
