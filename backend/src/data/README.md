# DynamoDB Data Access Layer

This module provides a production-ready DynamoDB integration with the repository pattern, exponential backoff retry logic, and comprehensive error handling.

## Components

### 1. DynamoDBClient (`dynamodb_client.py`)
- Manages DynamoDB connections with retry logic
- Implements exponential backoff with jitter for retryable errors
- Configurable retry behavior via `RetryConfig`
- Handles transient errors: `ProvisionedThroughputExceededException`, `ThrottlingException`, etc.

### 2. Base Repository (`repository.py`)
- Abstract base class implementing the repository pattern
- Generic CRUD operations with retry logic
- Batch operations (BatchWriteItem, BatchGetItem) with automatic chunking
- Query support with pagination
- Projection expressions for efficient data retrieval

### 3. Concrete Repositories (`repositories.py`)
- **UserRepository**: User management with phone number GSI
- **FarmRepository**: Farm management with owner GSI
- **RecommendationRepository**: Recommendation history with query GSI
- **ProductRepository**: Marketplace products with farmer and type-score GSIs
- **KnowledgeRepository**: RAG knowledge chunks with topic GSI

### 4. Table Definitions (`table_definitions.py`)
- DynamoDB table schemas for all entities
- Global Secondary Index (GSI) definitions
- Table creation and verification utilities
- Environment-specific table naming

## Usage

### Initialize DynamoDB Client

```python
from data import DynamoDBClient, RetryConfig

# Create client with custom retry config
retry_config = RetryConfig(
    max_attempts=3,
    base_delay_ms=100,
    max_delay_ms=5000,
    exponential_base=2.0,
    jitter=True
)

client = DynamoDBClient(
    region_name='us-east-1',
    retry_config=retry_config
)
```

### Use Repositories

```python
from data import UserRepository, User, UserRole
from datetime import datetime

# Create repository
user_repo = UserRepository(client, env='dev')

# Create a user
user = User(
    user_id='user123',
    phone_number='+919876543210',
    name='Ramesh Kumar',
    language_preference='hi',
    role=UserRole.FARMER
)

await user_repo.create_user(user)

# Get user by ID
user = await user_repo.get_user('user123')

# Get user by phone (using GSI)
user = await user_repo.get_user_by_phone('+919876543210')

# Update user
await user_repo.update_user('user123', {
    'language_preference': 'ta'
})
```

### Query with Pagination

```python
from data import FarmRepository

farm_repo = FarmRepository(client, env='dev')

# List user's farms with pagination
result = await farm_repo.list_user_farms(
    owner_id='user123',
    limit=10
)

farms = result['items']
next_token = result.get('last_evaluated_key')

# Get next page
if next_token:
    result = await farm_repo.list_user_farms(
        owner_id='user123',
        limit=10,
        exclusive_start_key=next_token
    )
```

### Batch Operations

```python
# Batch write
users = [user1, user2, user3, ...]
await user_repo.batch_write(users)

# Batch get
keys = [
    {'user_id': 'user1'},
    {'user_id': 'user2'},
    {'user_id': 'user3'}
]
users = await user_repo.batch_get(keys)
```

## Table Schemas

### Users Table
- **Partition Key**: `user_id` (String)
- **GSI**: `phone-index` on `phone_number`
- **Attributes**: name, language_preference, role, created_at, last_active, metadata

### Farms Table
- **Partition Key**: `farm_id` (String)
- **Sort Key**: `owner_id` (String)
- **GSI**: `owner-index` on `owner_id` + `created_at`
- **Attributes**: location, area_hectares, soil_type, irrigation_type, crops, metadata

### Recommendations Table
- **Partition Key**: `user_id` (String)
- **Sort Key**: `timestamp` (String - ISO8601)
- **GSI**: `query-index` on `query_id`
- **Attributes**: recommendation_id, farm_id, recommendation_text, reasoning_chain, confidence, agent_contributions, language

### Products Table
- **Partition Key**: `product_id` (String)
- **GSI 1**: `farmer-index` on `farmer_id` + `created_at`
- **GSI 2**: `type-score-index` on `product_type` + `pure_product_score`
- **Attributes**: farm_id, name, quantity_kg, price_per_kg, harvest_date, images, status, metadata

### Knowledge Table
- **Partition Key**: `chunk_id` (String)
- **GSI**: `topic-index` on `topic` + `created_at`
- **Attributes**: content, source, crop_type, region, embedding_indexed

## Initialize Tables

```python
from data.table_definitions import initialize_tables

# Create all tables for an environment
results = initialize_tables(region_name='us-east-1', env='dev')

for table_name, created in results.items():
    status = "Created" if created else "Already exists"
    print(f"{table_name}: {status}")
```

Or use the CLI:

```bash
python backend/src/data/table_definitions.py dev us-east-1
```

## Testing

Property-based tests validate correctness properties:

1. **Property 1: DynamoDB write key consistency** - All writes include correct partition/sort keys
2. **Property 2: DynamoDB retry with exponential backoff** - Retryable errors are retried with proper backoff

Run tests:

```bash
pytest tests/test_dynamodb_properties.py -v
```

## Error Handling

The client automatically retries these errors:
- `ProvisionedThroughputExceededException`
- `ThrottlingException`
- `RequestLimitExceeded`
- `InternalServerError`
- `ServiceUnavailable`

Non-retryable errors fail immediately:
- `ResourceNotFoundException`
- `ValidationException`
- `ConditionalCheckFailedException`
- etc.

## Configuration

### Retry Configuration

```python
RetryConfig(
    max_attempts=3,        # Maximum retry attempts
    base_delay_ms=100,     # Initial delay in milliseconds
    max_delay_ms=5000,     # Maximum delay cap
    exponential_base=2.0,  # Exponential backoff base
    jitter=True            # Add randomness to prevent thundering herd
)
```

### Environment Variables

```bash
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
DYNAMODB_ENV=dev  # dev, staging, production
```

## Best Practices

1. **Use GSIs for queries**: Don't scan tables, use Global Secondary Indexes
2. **Batch operations**: Use batch_write/batch_get for multiple items
3. **Pagination**: Always implement pagination for list operations
4. **Projection expressions**: Fetch only required attributes
5. **Consistent reads**: Use only when necessary (costs 2x)
6. **Error handling**: Let the client handle retries automatically
7. **Connection pooling**: Reuse DynamoDBClient instances

## Requirements

- boto3 >= 1.28.85
- botocore >= 1.31.85
- Python >= 3.8

## License

Part of the GramBrain AI platform.
