# GramBrain AI - Project Structure

```
grambrain-ai/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── orchestrator.py          # Master orchestrator agent
│   │   ├── agent_base.py            # Base agent class
│   │   └── agent_registry.py        # Agent registration and discovery
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── weather_agent.py
│   │   ├── soil_agent.py
│   │   ├── crop_advisory_agent.py
│   │   ├── pest_agent.py
│   │   ├── irrigation_agent.py
│   │   ├── yield_agent.py
│   │   ├── market_agent.py
│   │   ├── sustainability_agent.py
│   │   ├── marketplace_agent.py
│   │   ├── farmer_interaction_agent.py
│   │   └── village_agent.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── bedrock_client.py        # AWS Bedrock integration
│   │   ├── prompt_templates.py      # Prompt engineering
│   │   └── llm_utils.py             # LLM utilities
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vector_db.py             # Vector database interface
│   │   ├── embeddings.py            # Embedding generation
│   │   ├── retrieval.py             # RAG retrieval logic
│   │   └── knowledge_base.py        # Knowledge management
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.py                # Data models
│   │   ├── database.py              # Database layer
│   │   └── migrations.py            # DB migrations
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py                # API endpoints
│   │   └── middleware.py            # Auth, validation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── config.py
│       └── errors.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── config/
│   ├── settings.py
│   └── aws_config.py
│
├── requirements.txt
├── .env.example
└── README.md
```
