# Data Layer Implementation

This directory contains the complete data access layer for the GramBrain AI production system.

## Overview

The data layer implements the repository pattern with DynamoDB as the backing store, providing:
- Type-safe data models
- Pydantic schema validation
- Automatic retry with exponential backoff
- Pagination support
- Batch operations

## Components

### 1. Data Models (`models.py`)
Dataclass-based domain models representing core entities:
- `User`: User accounts with roles and preferences
- `Farm`: Farm information with location and crops
- `Recommendation`: AI-generated recommendations
- `Product`: Marketplace products
- `CropCycle`, `SoilHealthData`, `WeatherData`: Supporting models

### 2. Pydantic Schemas (`schemas.py`)
Request/response validation schemas:
- `UserSchema`, `UserCreateSchema`, `UserUpdateSchema`
- `FarmSchema`, `FarmCreateSchema`, `FarmUpdateSchema`
- `RecommendationSchema`, `RecommendationCreateSchema`
- `ProductSchema`, `ProductCreateSchema`, `ProductUpdateSchema`
- `KnowledgeChunkSchema`, `KnowledgeChunkCreateSchema`
- `PaginationParams`, `PaginatedResponse`

**Validation Features:**
- Phone number format validation (digits only, min 10 chars)
- Name validation (non-empty, trimmed)
- Date range validation (harvest dates not in future)
- Numeric bounds (area, price, quantity > 0)
- Enum validation for roles, irrigation types, product categories
- Pagination limit validation (1-1000)

### 3. DynamoDB Client (`dynamodb_client.py`)
Low-level DynamoDB client with:
- **Exponential backoff retry**: Configurable max attempts, base delay, max delay
- **Jitter**: Randomization to prevent thundering herd
- **Retryable error detection**: Handles throttling, provisioning errors
- **Async execution**: Non-blocking operations

**Retry Configuration:**
```python
RetryConfig(
    max_attempts=3,
    base_delay_ms=100,
    max_delay_ms=5000,
    exponential_base=2.0,
    jitter=True
)
```

### 4. Base Repository (`repository.py`)
Generic repository pattern implementation:
- `get_item()`: Single item retrieval
- `put_item()`: Create/update item
- `update_item()`: Partial updates with expression builder
- `delete_item()`: Item deletion
- `query()`: Query with pagination, GSI support, filters
- `batch_write()`: Batch writes with automatic chunking (25 items)
- `batch_get()`: Batch reads with automatic chunking (100 items)

### 5. Concrete Repositories (`repositories.py`)

#### UserRepository
- `create_user()`: Create user with validation
- `get_user()`: Get by user_id
- `get_user_by_phone()`: Query by phone number (GSI)
- `update_user()`: Update user attributes

#### FarmRepository
- `create_farm()`: Create farm
- `get_farm()`: Get by farm_id
- `list_user_farms()`: List farms by owner (GSI, paginated)
- `update_farm()`: Update farm attributes

#### RecommendationRepository
- `create_recommendation()`: Create recommendation
- `get_recommendation()`: Get by user_id + timestamp
- `list_user_recommendations()`: List by user (paginated)
- `get_recommendation_by_query_id()`: Query by query_id (GSI)

#### ProductRepository
- `create_product()`: Create product
- `get_product()`: Get by product_id
- `list_farmer_products()`: List by farmer (GSI, paginated)
- `list_products_by_type()`: List by type and score (GSI, paginated)
- `update_product()`: Update product attributes

#### KnowledgeRepository
- `create_chunk()`: Create knowledge chunk
- `get_chunk()`: Get by chunk_id
- `list_chunks_by_topic()`: List by topic (GSI, paginated)

### 6. Table Definitions (`table_definitions.py`)
DynamoDB table schemas with:
- Primary keys (partition + optional sort key)
- Global Secondary Indexes (GSIs)
- Pay-per-request billing mode
- Resource tags

**Tables:**
- `grambrain-users-{env}`: Users with phone-index GSI
- `grambrain-farms-{env}`: Farms with owner-index GSI
- `grambrain-recommendations-{env}`: Recommendations with query-index GSI
- `grambrain-products-{env}`: Products with farmer-index and type-score-index GSIs
- `grambrain-knowledge-{env}`: Knowledge chunks with topic-index GSI

### 7. Client Factory (`client_factory.py`)
Factory for creating DynamoDB clients with environment-specific configuration.

## Property-Based Tests

### Property 1: DynamoDB Write Key Consistency
**Validates: Requirements 1.2**

For any data write operation, partition and sort keys must match table schema.

**Tests:**
- User writes include `user_id` partition key
- Farm writes include `farm_id` partition key and `owner_id` sort key
- Keys are non-empty strings

### Property 2: DynamoDB Retry with Exponential Backoff
**Validates: Requirements 1.4**

For any retryable error, system retries with exponential backoff up to 3 attempts.

**Tests:**
- Delay calculation follows exponential backoff formula
- Retryable errors (throttling, provisioning) trigger retry
- Non-retryable errors fail immediately
- Max attempts respected
- Jitter adds randomness to delays

### Property 43: DynamoDB Pagination
**Validates: Requirements 14.2**

For any list operation, system implements pagination with limit and LastEvaluatedKey.

**Tests:**
- Results respect limit parameter
- LastEvaluatedKey present when more items exist
- LastEvaluatedKey absent when all items fit in one page
- Can retrieve all items across multiple pages
- ExclusiveStartKey correctly passed to queries
- Limit validation (1-1000)
- Empty results handled correctly

## Usage Examples

### Creating a User with Validation
```python
from data.client_factory import create_dynamodb_client
from data.repositories import UserRepository
from data.models import User, UserRole
from datetime import datetime

# Create client and repository
client = create_dynamodb_client(env='dev')
user_repo = UserRepository(client, env='dev')

# Create user (validates automatically)
user = User(
    user_id='user_123',
    phone_number='9876543210',
    name='Ramesh Kumar',
    language_preference='hi',
    role=UserRole.FARMER,
    created_at=datetime.now(),
    last_active=datetime.now()
)

created_user = await user_repo.create_user(user)
```

### Paginated Query
```python
from data.repositories import FarmRepository

farm_repo = FarmRepository(client, env='dev')

# First page
result = await farm_repo.list_user_farms(
    owner_id='user_123',
    limit=10
)

farms = result['items']
next_key = result.get('last_evaluated_key')

# Next page
if next_key:
    result = await farm_repo.list_user_farms(
        owner_id='user_123',
        limit=10,
        exclusive_start_key=next_key
    )
```

### Batch Operations
```python
from data.repositories import ProductRepository

product_repo = ProductRepository(client, env='dev')

# Batch write
products = [product1, product2, product3, ...]
await product_repo.batch_write(products)

# Batch get
keys = [{'product_id': 'p1'}, {'product_id': 'p2'}]
products = await product_repo.batch_get(keys)
```

## Testing

Run all property-based tests:
```bash
python -m pytest tests/test_dynamodb_properties.py -v
```

Run validation tests:
```bash
python -m pytest tests/test_repository_validation.py -v
```

## Requirements Validated

- ✅ **1.1**: DynamoDB tables for all entities
- ✅ **1.2**: Proper partition and sort keys
- ✅ **1.4**: Exponential backoff retry (max 3 attempts)
- ✅ **1.5**: Batch operations (BatchWriteItem, BatchGetItem)
- ✅ **14.2**: Pagination with limit and LastEvaluatedKey
- ✅ **16.1**: Pydantic schema validation
- ✅ **16.2**: Field-level validation errors
- ✅ **16.5**: Business rule validation (date ranges, numeric bounds)

## Next Steps

1. Implement AWS Bedrock LLM client (Task 4)
2. Implement S3 file storage integration (Task 6)
3. Implement OpenSearch vector database (Task 7)
4. Implement ElastiCache Redis integration (Task 8)
