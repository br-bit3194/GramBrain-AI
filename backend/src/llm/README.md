# AWS Bedrock LLM Client

## Overview

Production-ready AWS Bedrock LLM client with comprehensive features for fault tolerance, cost monitoring, and response validation.

## Features

### 1. Circuit Breaker Pattern
- Automatic fault detection and recovery
- Three states: CLOSED, OPEN, HALF_OPEN
- Configurable failure threshold (default: 5 failures)
- Automatic reset after timeout (default: 60 seconds)

### 2. Model Fallback
- Automatic fallback to alternative models on failure
- Default fallback chain: Claude 3 Sonnet → Claude 3 Haiku → Titan Text
- Customizable fallback model list

### 3. Token Usage Tracking
- Tracks input and output tokens for all API calls
- Calculates cost estimates based on model pricing
- Maintains usage history for analytics
- Provides aggregate statistics by model

### 4. Response Validation
- Validates responses against Pydantic schemas
- Ensures data integrity before processing
- Optional validation for performance-critical paths

### 5. Retry Logic
- Exponential backoff retry for transient errors
- Configurable max retries (default: 3)
- Automatic retry on throttling and service errors

## Usage

### Basic Usage

```python
from backend.src.llm.bedrock_client import BedrockClient

# Initialize client
client = BedrockClient(
    region="us-east-1",
    default_model=BedrockClient.CLAUDE_3_SONNET,
    enable_circuit_breaker=True,
    track_usage=True
)

# Invoke model
response = await client.invoke(
    prompt="What is the best time to plant wheat?",
    temperature=0.7,
    max_tokens=500
)

print(response.text)
print(f"Tokens used: {response.token_usage.total_tokens}")
print(f"Cost: ${response.token_usage.cost_estimate:.4f}")
```

### With Fallback

```python
# Automatic fallback to alternative models
response = await client.invoke_with_fallback(
    prompt="Analyze soil health for wheat farming",
    temperature=0.7,
    max_tokens=1000
)
```

### Token Usage Statistics

```python
# Get usage statistics
stats = client.get_usage_stats()
print(f"Total calls: {stats['total_calls']}")
print(f"Total cost: ${stats['total_cost']:.2f}")
print(f"By model: {stats['by_model']}")
```

## Configuration

### Environment Variables

```bash
AWS_REGION=us-east-1
DEFAULT_LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000
```

### Model IDs

- Claude 3 Sonnet: `anthropic.claude-3-sonnet-20240229-v1:0`
- Claude 3 Haiku: `anthropic.claude-3-haiku-20240307-v1:0`
- Titan Text: `amazon.titan-text-express-v1`
- Llama 2 70B: `meta.llama2-70b-chat-v1`

## Cost Estimates

Approximate costs per 1000 tokens (USD):

| Model | Input | Output |
|-------|-------|--------|
| Claude 3 Sonnet | $0.003 | $0.015 |
| Claude 3 Haiku | $0.00025 | $0.00125 |
| Titan Text | $0.0008 | $0.0016 |
| Llama 2 70B | $0.00195 | $0.00256 |

## Testing

Property-based tests validate:
1. Fallback behavior on model failures
2. Token usage tracking accuracy
3. Response validation correctness

Run tests:
```bash
pytest tests/test_bedrock_properties.py -v
```

## Architecture

```
BedrockClient
├── CircuitBreaker (fault tolerance)
├── Token Usage Tracking (cost monitoring)
├── Response Validation (data integrity)
└── Model Routing
    ├── Claude (Anthropic)
    ├── Titan (Amazon)
    └── Llama (Meta)
```

## Requirements Validated

- **Requirement 2.1**: IAM role configuration for Bedrock access
- **Requirement 2.2**: Fallback to alternative models on failure
- **Requirement 2.3**: Token usage tracking and cost monitoring
- **Requirement 2.5**: Response validation against schemas

## Property Tests

- **Property 3**: Bedrock fallback on failure
- **Property 4**: Token usage tracking
- **Property 6**: Bedrock response validation
