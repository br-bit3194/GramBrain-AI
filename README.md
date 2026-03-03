# GramBrain - AWS Stack Version 🌾

<div align="center">

[![AWS](https://img.shields.io/badge/AWS-Bedrock-orange?style=for-the-badge&logo=amazon-aws)](https://aws.amazon.com/bedrock/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![DynamoDB](https://img.shields.io/badge/DynamoDB-NoSQL-blue?style=for-the-badge&logo=amazon-dynamodb)](https://aws.amazon.com/dynamodb/)

</div>

AI-powered agricultural assistant for Indian farmers, now powered by AWS Bedrock and DynamoDB with Strands multi-agent framework.

---

## 🚀 What's New in AWS Version

### AWS Stack Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **🤖 LLM** | AWS Bedrock (Amazon Nova Lite) | Advanced AI conversations & vision |
| **💾 Database** | AWS DynamoDB | Scalable NoSQL storage |
| **🎯 Agent Framework** | Strands (Custom) | Multi-agent orchestration |
| **🖼️ Vision** | Claude Vision (Bedrock) | Crop disease diagnosis |
| **📊 Sessions** | DynamoDB | Distributed session management |

### Key Improvements

✅ **Scalability**: DynamoDB auto-scales with demand  
✅ **Performance**: Amazon Nova lite for better responses  
✅ **Reliability**: AWS managed services with 99.99% uptime  
✅ **Cost-Effective**: Pay-per-use pricing model  
✅ **Global**: Deploy in multiple AWS regions  

---

## 📋 Prerequisites

### 1. AWS Account

- Create AWS account at https://aws.amazon.com
- Set up IAM user with Bedrock and DynamoDB permissions
- Generate access keys

### 2. Enable AWS Bedrock

1. Go to AWS Bedrock console
2. Request access to Claude models
3. Wait for approval (usually instant)

### 3. Required Permissions

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
        "dynamodb:*"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/farmbot_*"
    }
  ]
}
```

---

## 🛠️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/codeprofile/Project-Kisan.git
cd Project-Kisan
```

### 2. Install Dependencies

```bash
# Install AWS-specific requirements
pip install -r requirements-aws.txt
```

### 3. Configure Environment

Create `.env` file:

```bash
# AWS Credentials
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# AWS Bedrock
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_TEMPERATURE=0.7
BEDROCK_MAX_TOKENS=4096

# DynamoDB Tables
DYNAMODB_MARKET_PRICES_TABLE=farmbot_market_prices
DYNAMODB_SESSIONS_TABLE=farmbot_sessions
DYNAMODB_ANALYTICS_TABLE=farmbot_analytics

# External APIs
WEATHER_API_KEY=your_openweathermap_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key_optional
MANDI_API_KEY=your_data_gov_api_key_optional
```

### 4. Initialize Database

```bash
# Tables will be created automatically on first run
# Or create manually:
python -c "from backend.app.aws_integration.database.dynamodb_client import dynamodb_client; dynamodb_client.create_tables()"
```

---

## 🚀 Running the Application

### Development Mode

```bash
python -m uvicorn backend.app.main_aws:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
gunicorn backend.app.main_aws:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Access Application

Open browser: `http://localhost:8000`

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│           (main_aws.py)                 │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      FarmBot Service AWS                │
│    (farmbot_service_aws.py)             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Strands Orchestrator               │
│    (Multi-Agent Coordinator)            │
└─┬────┬────┬────┬──────────────────────┘
  │    │    │    │
  ▼    ▼    ▼    ▼
┌───┐┌───┐┌───┐┌───┐
│ W ││ M ││ C ││ G │  Specialized Agents
│ e ││ a ││ r ││ o │
│ a ││ r ││ o ││ v │
│ t ││ k ││ p ││ t │
│ h ││ e ││   ││   │
│ e ││ t ││ H ││ S │
│ r ││   ││ e ││ c │
│   ││   ││ a ││ h │
│   ││   ││ l ││ e │
│   ││   ││ t ││ m │
│   ││   ││ h ││ e │
└───┘└───┘└───┘└───┘
  │    │    │    │
  └────┴────┴────┴──────────┐
                             ▼
              ┌──────────────────────────┐
              │     AWS Bedrock          │
              │  (Claude 3.5 Sonnet)     │
              └──────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │     AWS DynamoDB         │
              │  (3 Tables)              │
              └──────────────────────────┘
```

---

## ✨ Features

### 1. 🌱 Crop Health Diagnosis
- Upload crop images
- AI-powered disease detection using Claude Vision
- Treatment recommendations
- Locally available medicines

### 2. 🌤️ Weather Forecasting
- Real-time weather data
- 7-day forecasts
- Irrigation recommendations
- Farming operation timing

### 3. 📊 Market Intelligence
- Live mandi prices from DynamoDB
- Price trend analysis
- Selling recommendations
- Best market identification

### 4. 🏛️ Government Schemes
- AI-powered scheme search
- Eligibility checking
- Application guidance
- Document requirements

---

## 📊 DynamoDB Tables

### Market Prices Table
```
PK: COMMODITY#{commodity}
SK: DATE#{date}#MARKET#{market_id}
Attributes: state, district, market, prices, trends
```

### Sessions Table
```
PK: SESSION#{session_id}
SK: METADATA
Attributes: user_id, state, created_at, ttl
```

### Analytics Table
```
PK: ANALYTICS#{commodity}
SK: DATE#{date}
Attributes: statistics, trends, predictions
```

---

## 🧪 Testing

### Test Bedrock Connection

```python
from backend.app.aws_integration.bedrock.bedrock_client import bedrock_client

response = bedrock_client.invoke_model(
    prompt="Hello, how are you?",
    temperature=0.7
)
print(response)
```

### Test DynamoDB

```python
from backend.app.aws_integration.database.dynamodb_client import dynamodb_client

# Create tables
dynamodb_client.create_tables()

# Test session
success = dynamodb_client.create_session(
    session_id="test",
    user_id="user1",
    initial_state={"test": True}
)
```

---

## 💰 Cost Estimation

### AWS Bedrock (Claude 3.5 Sonnet)
- Input: ~$3 per 1M tokens
- Output: ~$15 per 1M tokens
- **Estimated**: $10-50/month for moderate usage

### DynamoDB (On-Demand)
- Write: $1.25 per 1M requests
- Read: $0.25 per 1M requests
- **Estimated**: $5-20/month for moderate usage

### Total Estimated Cost
**$15-70/month** for moderate usage (1000-5000 queries/day)

---

## 🔧 Configuration

### Bedrock Models

Available models:
- `anthropic.claude-3-5-sonnet-20241022-v2:0` (Recommended)
- `anthropic.claude-3-haiku-20240307-v1:0` (Faster, cheaper)
- `anthropic.claude-3-opus-20240229-v1:0` (Most capable)

### DynamoDB Settings

```python
# On-Demand (default)
BillingMode='PAY_PER_REQUEST'

# Provisioned (for consistent traffic)
BillingMode='PROVISIONED'
ReadCapacityUnits=5
WriteCapacityUnits=5
```

---

## 📈 Monitoring

### CloudWatch Metrics

Monitor:
- Bedrock invocation count
- DynamoDB read/write units
- API latency
- Error rates

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## 🚀 Deployment

### AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.8 project-kisan

# Create environment
eb create project-kisan-env

# Deploy
eb deploy
```

### AWS ECS/Fargate

```bash
# Build Docker image
docker build -t project-kisan-aws .

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag project-kisan-aws:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/project-kisan-aws:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/project-kisan-aws:latest
```

---

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- AWS Bedrock team for Claude models
- AWS DynamoDB team for scalable database
- OpenWeatherMap for weather data
- Data.gov.in for market data

---

<div align="center">

**Built with ❤️ for Indian Farmers**

[⭐ Star this repository](https://github.com/codeprofile/Project-Kisan) to support AI-powered agriculture!

</div>
