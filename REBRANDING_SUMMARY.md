# GramBrain Rebranding Summary

## Project Renamed: FarmBot → GramBrain

### Overview
The project has been successfully rebranded from "FarmBot" / "GramBrain" to **GramBrain**.

---

## Changes Made

### 1. **File Renames**
- `farmbot_service_aws.py` → `grambrain_service_aws.py`

### 2. **Class Renames**
- `FarmBotServiceAWS` → `GramBrainServiceAWS`
- `farmbot_service_aws` (instance) → `grambrain_service_aws`

### 3. **Database Table Names**
**Old:**
- `farmbot_market_prices`
- `farmbot_sessions`
- `farmbot_analytics`

**New:**
- `grambrain_market_prices`
- `grambrain_sessions`
- `grambrain_analytics`

### 4. **Application Configuration**
**Old:**
```bash
APP_NAME=project-kisan
DATABASE_URL=sqlite:///./farmbot_market.db
```

**New:**
```bash
APP_NAME=grambrain
DATABASE_URL=sqlite:///./grambrain_market.db
```

### 5. **Code References Updated**

#### Main Application (`main_aws.py`)
- Service initialization messages
- Page titles
- Status endpoints
- Import statements

#### Service Layer (`grambrain_service_aws.py`)
- Class name and docstrings
- Logger messages
- Service status responses
- Error messages

#### Orchestrator (`strands/orchestrator.py`)
- System prompts
- Agent instructions
- Response synthesis

#### Database Client (`dynamodb_client.py`)
- Class docstrings
- Comments

#### Environment Configuration (`.env.example`)
- All table names
- Application name
- Database URLs
- Header comments

---

## Brand Identity

### GramBrain
**Meaning:** 
- **Gram** (ग्राम) = Village in Hindi
- **Brain** = Intelligence/AI

**Tagline:** "Village Intelligence for Modern Farming"

**Description:** AI-powered agricultural assistant bringing intelligent farming solutions to rural India.

---

## Updated Features

### Core Capabilities
1. 🌱 **Crop Health Diagnosis** - AI vision-based disease detection
2. 🌤️ **Weather Intelligence** - Real-time forecasts and farming advice
3. 📊 **Market Analytics** - Live mandi prices and selling strategies
4. 🏛️ **Scheme Navigator** - Government schemes and subsidies

### Technology Stack
- **LLM:** AWS Bedrock (Claude 3.5 Sonnet)
- **Database:** AWS DynamoDB
- **Framework:** Strands Multi-Agent System
- **Vision:** Claude Vision for crop analysis
- **Backend:** FastAPI + Python

---

## Migration Checklist

### For Existing Deployments

- [ ] Update environment variables in `.env`
- [ ] Update DynamoDB table names
- [ ] Recreate tables with new names (or migrate data)
- [ ] Update CloudWatch dashboards
- [ ] Update IAM policies (table name patterns)
- [ ] Update documentation
- [ ] Update frontend branding
- [ ] Update API documentation
- [ ] Test all endpoints
- [ ] Update monitoring alerts

### Environment Variables to Update

```bash
# Old
DYNAMODB_MARKET_PRICES_TABLE=farmbot_market_prices
DYNAMODB_SESSIONS_TABLE=farmbot_sessions
DYNAMODB_ANALYTICS_TABLE=farmbot_analytics
APP_NAME=project-kisan

# New
DYNAMODB_MARKET_PRICES_TABLE=grambrain_market_prices
DYNAMODB_SESSIONS_TABLE=grambrain_sessions
DYNAMODB_ANALYTICS_TABLE=grambrain_analytics
APP_NAME=grambrain
```

---

## IAM Policy Update

Update your IAM policies to reflect new table names:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:*"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/grambrain_*"
    }
  ]
}
```

---

## Data Migration (If Needed)

### Option 1: Fresh Start
Simply create new tables with new names. Old data will remain in old tables.

### Option 2: Migrate Data
```python
# Example migration script
from aws_integration.database.dynamodb_client import dynamodb_client

# Scan old table
old_table = dynamodb.Table('farmbot_sessions')
items = old_table.scan()['Items']

# Write to new table
new_table = dynamodb.Table('grambrain_sessions')
for item in items:
    new_table.put_item(Item=item)
```

---

## Testing

### Quick Test Commands

```bash
# Test service initialization
python -c "from backend.app.aws_integration.grambrain_service_aws import grambrain_service_aws; import asyncio; asyncio.run(grambrain_service_aws.initialize())"

# Test API
curl http://localhost:8000/health

# Test status endpoint
curl http://localhost:8000/api/service-status
```

---

## Documentation Updates

### Files Updated
- ✅ `.env.example` - All references updated
- ✅ `main_aws.py` - Service imports and usage
- ✅ `grambrain_service_aws.py` - Class and methods
- ✅ `orchestrator.py` - System prompts
- ✅ `dynamodb_client.py` - Comments

### Files to Update (Frontend/Docs)
- [ ] `README.md` - Main documentation
- [ ] `README_AWS.md` - AWS-specific docs
- [ ] `AWS_MIGRATION_GUIDE.md` - Migration guide
- [ ] Frontend templates (if any)
- [ ] API documentation
- [ ] User guides

---

## Backward Compatibility

### Legacy Support
The old Google ADK implementation remains unchanged:
- `google_adk_integration/` folder intact
- Can still use old FarmBot service if needed
- Gradual migration possible

### Running Old Version
```bash
# Use old Google ADK version
python -m uvicorn backend.app.main:app --reload
```

### Running New Version
```bash
# Use new AWS GramBrain version
python -m uvicorn backend.app.main_aws:app --reload
```

---

## Next Steps

1. **Update Frontend Branding**
   - Logo and colors
   - Application name
   - Page titles

2. **Update Documentation**
   - README files
   - API docs
   - User guides

3. **Deploy to Production**
   - Update environment variables
   - Create new DynamoDB tables
   - Update IAM policies
   - Test thoroughly

4. **Marketing Materials**
   - Update website
   - Update app store listings
   - Update social media

---

## Support

For questions or issues with the rebranding:
- Check updated documentation
- Review environment variables
- Verify table names in AWS Console
- Test with health check endpoint

---

## Version History

- **v3.0.0-aws** - GramBrain AWS Stack (Current)
- **v2.1.0** - FarmBot with ElevenLabs Voice
- **v2.0.0** - FarmBot with Google ADK
- **v1.0.0** - Initial GramBrain

---

**Rebranding Date:** March 3, 2026  
**Status:** ✅ Complete  
**Impact:** Low (backward compatible)
