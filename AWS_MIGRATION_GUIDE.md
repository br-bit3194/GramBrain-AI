# AWS Migration Guide - GramBrain

## Overview

This guide covers the migration from Google ADK stack to AWS stack for GramBrain.

### Stack Comparison

| Component | Google ADK Stack | AWS Stack |
|-----------|-----------------|-----------|
| **LLM** | Google Gemini 2.0 Flash | AWS Bedrock (Claude 3.5 Sonnet) |
| **Database** | SQLite | AWS DynamoDB |
| **Agent Framework** | Google ADK | Strands Framework (Custom) |
| **Image Analysis** | Gemini Vision | Claude Vision (Bedrock) |
| **Session Management** | InMemorySessionService | DynamoDB Sessions |

---

## Architecture

### New AWS Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│                      (main_aws.py)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              FarmBot Service AWS                             │
│         (farmbot_service_aws.py)                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Strands Orchestrator                              │
│         (Multi-Agent Coordinator)                           │
└─────┬──────┬──────┬──────┬─────────────────────────────────┘
      │      │      │      │
      ▼      ▼      ▼      ▼
   ┌────┐ ┌────┐ ┌────┐ ┌────┐
   │ W  │ │ M  │ │ C  │ │ G  │  Specialized Agents
   │ e  │ │ a  │ │ r  │ │ o  │  (Weather, Market,
   │ a  │ │ r  │ │ o  │ │ v  │   Crop Health, Schemes)
   │ t  │ │ k  │ │ p  │ │ t  │
   │ h  │ │ e  │ │    │ │    │
   │ e  │ │ t  │ │ H  │ │ S  │
   │ r  │ │    │ │ e  │ │ c  │
   │    │ │    │ │ a  │ │ h  │
   │    │ │    │ │ l  │ │ e  │
   │    │ │    │ │ t  │ │ m  │
   │    │ │    │ │ h  │ │ e  │
   └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘
     │      │      │      │
     ▼      ▼      ▼      ▼
┌─────────────────────────────────────────────────────────────┐
│                    AWS Bedrock                               │
│              (Claude 3.5 Sonnet)                            │
│         - Text Generation                                    │
│         - Vision Analysis                                    │
│         - Tool Calling                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    AWS DynamoDB                              │
│         - Market Prices Table                               │
│         - Sessions Table                                     │
│         - Analytics Table                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### 1. AWS Account Setup

1. Create an AWS account at https://aws.amazon.com
2. Set up IAM user with appropriate permissions:
   - Bedrock access
   - DynamoDB access
3. Generate access keys (Access Key ID and Secret Access Key)

### 2. Enable AWS Bedrock

1. Go to AWS Bedrock console
2. Request access to Claude models:
   - Claude 3.5 Sonnet
   - Claude 3 Haiku (optional, for faster responses)
3. Wait for approval (usually instant for most regions)

### 3. Required Permissions

Create an IAM policy with these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:CreateTable",
        "dynamodb:DescribeTable",
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:UpdateItem",
        "dynamodb:Query",
        "dynamodb:Scan",
        "dynamodb:BatchWriteItem"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/farmbot_*"
    }
  ]
}
```

---

## Installation

### 1. Install AWS Dependencies

```bash
# Install AWS-specific requirements
pip install -r requirements-aws.txt

# Or install individually
pip install boto3 botocore
```

### 2. Configure Environment Variables

Update your `.env` file:

```bash
# AWS Credentials
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# AWS Bedrock Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_TEMPERATURE=0.7
BEDROCK_MAX_TOKENS=4096

# DynamoDB Tables
DYNAMODB_MARKET_PRICES_TABLE=farmbot_market_prices
DYNAMODB_SESSIONS_TABLE=farmbot_sessions
DYNAMODB_ANALYTICS_TABLE=farmbot_analytics

# External APIs (same as before)
WEATHER_API_KEY=your_openweathermap_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key_optional
MANDI_API_KEY=your_data_gov_api_key_optional
```

### 3. Initialize DynamoDB Tables

The tables will be created automatically on first run, or you can create them manually:

```python
from backend.app.aws_integration.database.dynamodb_client import dynamodb_client

# Create all tables
dynamodb_client.create_tables()
```

---

## Running the Application

### Development Mode

```bash
# Run with AWS stack
python -m uvicorn backend.app.main_aws:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# Run with gunicorn
gunicorn backend.app.main_aws:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## Key Differences

### 1. Agent Framework

**Google ADK:**
```python
from google.adk.agents import Agent

agent = Agent(
    name="weather_agent",
    model="gemini-2.0-flash",
    tools=[tool1, tool2],
    sub_agents=[child_agent]
)
```

**Strands Framework:**
```python
from aws_integration.strands import BaseAgent

class WeatherAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="weather_specialist",
            description="Weather expert",
            instruction="System prompt",
            tools=[tool1, tool2]
        )
    
    async def process(self, message, context):
        # Custom processing logic
        return result
```

### 2. LLM Invocation

**Google ADK:**
```python
# Handled automatically by ADK
response = await runner.run_async(message)
```

**AWS Bedrock:**
```python
from aws_integration.bedrock.bedrock_client import bedrock_client

response = bedrock_client.invoke_model(
    prompt="Your prompt",
    system_prompt="System instructions",
    temperature=0.7
)
```

### 3. Database Operations

**SQLite (Old):**
```python
from sqlalchemy.orm import Session

with get_db_session() as db:
    prices = db.query(MarketPrice).filter(...).all()
```

**DynamoDB (New):**
```python
from aws_integration.database.dynamodb_client import dynamodb_client

prices = dynamodb_client.get_market_prices_by_commodity("Onion", days=7)
```

### 4. Session Management

**Google ADK:**
```python
session_service = InMemorySessionService()
session = await session_service.get_session(app_name, user_id, session_id)
```

**DynamoDB:**
```python
session = dynamodb_client.get_session(session_id)
dynamodb_client.update_session_state(session_id, new_state)
```

---

## DynamoDB Table Structure

### 1. Market Prices Table

```
Table Name: farmbot_market_prices
Partition Key: pk (String) - Format: COMMODITY#{commodity}
Sort Key: sk (String) - Format: DATE#{date}#MARKET#{market_id}

Attributes:
- state, district, market, commodity
- variety, grade, arrival_date
- min_price, max_price, modal_price
- price_change, percentage_change, trend
- created_at, updated_at, is_active
```

### 2. Sessions Table

```
Table Name: farmbot_sessions
Partition Key: pk (String) - Format: SESSION#{session_id}
Sort Key: sk (String) - Always "METADATA"

Attributes:
- user_id, session_id
- state (JSON object with session data)
- created_at, updated_at
- ttl (Time to Live - auto-delete after 7 days)
```

### 3. Analytics Table

```
Table Name: farmbot_analytics
Partition Key: pk (String) - Format: ANALYTICS#{commodity}
Sort Key: sk (String) - Format: DATE#{date}

Attributes:
- commodity, analysis_date
- avg_price, highest_price, lowest_price
- price_volatility, total_markets
- weekly_trend, monthly_trend
- predicted_price_7d, predicted_price_14d
- price_history (JSON), recommendations (JSON)
```

---

## Testing

### 1. Test Bedrock Connection

```python
from backend.app.aws_integration.bedrock.bedrock_client import bedrock_client

response = bedrock_client.invoke_model(
    prompt="Hello, how are you?",
    temperature=0.7
)
print(response)
```

### 2. Test DynamoDB Connection

```python
from backend.app.aws_integration.database.dynamodb_client import dynamodb_client

# Create tables
dynamodb_client.create_tables()

# Test session creation
success = dynamodb_client.create_session(
    session_id="test_session",
    user_id="test_user",
    initial_state={"test": True}
)
print(f"Session created: {success}")
```

### 3. Test Agent Processing

```python
from backend.app.aws_integration.farmbot_service_aws import farmbot_service_aws

# Initialize service
await farmbot_service_aws.initialize()

# Test query
response = await farmbot_service_aws.process_message(
    message="आज मौसम कैसा है?",
    session_id="test_session",
    user_context={"user_location": "Delhi"}
)
print(response.response)
```

---

## Cost Optimization

### AWS Bedrock Pricing

- **Claude 3.5 Sonnet**: ~$3 per 1M input tokens, ~$15 per 1M output tokens
- **Claude 3 Haiku**: ~$0.25 per 1M input tokens, ~$1.25 per 1M output tokens

**Tips:**
- Use Haiku for simple queries
- Cache system prompts
- Limit max_tokens appropriately

### DynamoDB Pricing

- **On-Demand Mode**: Pay per request
  - Write: $1.25 per million requests
  - Read: $0.25 per million requests
- **Provisioned Mode**: Pay for capacity (cheaper for consistent traffic)

**Tips:**
- Use on-demand for development
- Switch to provisioned for production
- Enable TTL for automatic cleanup
- Use batch operations when possible

---

## Monitoring

### CloudWatch Metrics

Monitor these key metrics:

1. **Bedrock:**
   - Invocation count
   - Latency
   - Error rate

2. **DynamoDB:**
   - Read/Write capacity units
   - Throttled requests
   - Table size

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
```

---

## Troubleshooting

### Common Issues

1. **Bedrock Access Denied**
   - Ensure model access is enabled in Bedrock console
   - Check IAM permissions
   - Verify region supports Claude models

2. **DynamoDB Table Not Found**
   - Run `dynamodb_client.create_tables()`
   - Check table names in environment variables
   - Verify IAM permissions

3. **High Latency**
   - Use Claude Haiku for faster responses
   - Implement caching
   - Optimize prompts

4. **Cost Overruns**
   - Monitor CloudWatch metrics
   - Set up billing alerts
   - Implement rate limiting

---

## Migration Checklist

- [ ] AWS account created and configured
- [ ] IAM user with appropriate permissions
- [ ] Bedrock access enabled for Claude models
- [ ] Environment variables configured
- [ ] AWS dependencies installed
- [ ] DynamoDB tables created
- [ ] Bedrock connection tested
- [ ] DynamoDB connection tested
- [ ] Agent processing tested
- [ ] WebSocket functionality tested
- [ ] CloudWatch monitoring set up
- [ ] Billing alerts configured

---

## Support

For issues or questions:
- Check AWS Bedrock documentation: https://docs.aws.amazon.com/bedrock/
- Check DynamoDB documentation: https://docs.aws.amazon.com/dynamodb/
- Review CloudWatch logs for errors
- Contact AWS support for service-specific issues

---

## Next Steps

1. **Deploy to AWS:**
   - Use AWS Elastic Beanstalk
   - Or AWS ECS/Fargate
   - Or AWS Lambda with API Gateway

2. **Add Features:**
   - Implement caching with ElastiCache
   - Add S3 for image storage
   - Use SQS for async processing
   - Implement CloudFront CDN

3. **Optimize:**
   - Fine-tune prompts
   - Implement response caching
   - Use DynamoDB DAX for faster reads
   - Enable DynamoDB auto-scaling
