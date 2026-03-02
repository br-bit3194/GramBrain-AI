# GramBrain AI - Project Structure Guide

## Overview
This project follows a clean backend/frontend separation with all Python code consolidated in the `/backend` directory.

## Directory Structure

```
GramBrain-AI/
│
├── backend/                    # Python Backend (FastAPI)
│   ├── src/
│   │   ├── agents/            # AI Agents (11 specialized agents)
│   │   │   ├── weather_agent.py
│   │   │   ├── soil_agent.py
│   │   │   ├── crop_advisory_agent.py
│   │   │   ├── pest_agent.py
│   │   │   ├── irrigation_agent.py
│   │   │   ├── yield_agent.py
│   │   │   ├── market_agent.py
│   │   │   ├── sustainability_agent.py
│   │   │   ├── marketplace_agent.py
│   │   │   ├── farmer_interaction_agent.py
│   │   │   └── village_agent.py
│   │   │
│   │   ├── api/               # REST API Layer
│   │   │   ├── __init__.py
│   │   │   └── routes.py      # All API endpoints
│   │   │
│   │   ├── core/              # Core Framework
│   │   │   ├── agent_base.py  # Base agent classes
│   │   │   ├── agent_registry.py
│   │   │   └── orchestrator.py
│   │   │
│   │   ├── data/              # Data Models
│   │   │   └── models.py      # Pydantic models
│   │   │
│   │   ├── llm/               # LLM Integration
│   │   │   └── bedrock_client.py
│   │   │
│   │   ├── rag/               # RAG Pipeline
│   │   │   ├── vector_db.py
│   │   │   ├── embeddings.py
│   │   │   └── retrieval.py
│   │   │
│   │   └── system.py          # Main System Orchestrator
│   │
│   ├── Dockerfile
│   ├── main.py                # Backend entry point
│   └── requirements.txt
│
├── frontend/                   # Next.js Frontend
│   ├── src/
│   │   ├── app/               # Next.js 13+ app directory
│   │   ├── components/        # React components
│   │   ├── services/          # API clients
│   │   ├── store/             # Zustand state management
│   │   └── types/             # TypeScript types
│   │
│   ├── package.json
│   └── tsconfig.json
│
├── tests/                      # All Tests
│   ├── test_agents.py
│   ├── test_api.py
│   ├── test_data_models.py
│   ├── test_orchestrator.py
│   └── test_rag.py
│
├── docs/                       # Documentation
│   ├── requirements.md        # Product requirements
│   ├── design.md              # Technical design
│   ├── SYSTEM_ARCHITECTURE.md
│   └── CONSOLIDATION_SUMMARY.md
│
├── main.py                     # Root entry point
├── docker-compose.yml          # Docker orchestration
├── requirements.txt            # Python dependencies
└── README.md
```

## Key Files

### Backend Entry Points
- `main.py` (root) - Runs backend server from project root
- `backend/main.py` - Runs backend server from backend directory (Docker)
- `backend/src/api/routes.py` - All API endpoints

### System Core
- `backend/src/system.py` - Main GramBrain system orchestrator
- `backend/src/core/orchestrator.py` - Multi-agent coordinator
- `backend/src/core/agent_registry.py` - Agent management

### API Endpoints
All endpoints in `backend/src/api/routes.py`:
- `/health` - Health check
- `/api/users` - User management
- `/api/farms` - Farm management
- `/api/query` - Query processing (main AI endpoint)
- `/api/products` - Marketplace
- `/api/knowledge` - RAG knowledge base

## Running the Project

### Development Mode
```bash
# Backend only
python main.py

# Frontend only
cd frontend
npm run dev

# Both with Docker
docker-compose up
```

### Testing
```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_agents.py

# With coverage
pytest tests/ --cov=backend.src --cov-report=html
```

## Import Patterns

### In Backend Code
```python
# Importing agents
from backend.src.agents import WeatherAgent, SoilAgent

# Importing core
from backend.src.core import OrchestratorAgent, Query

# Importing data models
from backend.src.data.models import User, Farm

# Importing system
from backend.src.system import GramBrainSystem
```

### In Tests
```python
# Same pattern as backend
from backend.src.agents import WeatherAgent
from backend.src.core import Query, UserContext
```

## Adding New Features

### New Agent
1. Create file in `backend/src/agents/new_agent.py`
2. Inherit from `Agent` base class
3. Implement `analyze()` method
4. Register in `backend/src/system.py`
5. Add tests in `tests/test_agents.py`

### New API Endpoint
1. Add route in `backend/src/api/routes.py`
2. Create Pydantic request model if needed
3. Use `Depends(get_system)` for system access
4. Add tests in `tests/test_api.py`

### New Data Model
1. Add model in `backend/src/data/models.py`
2. Use Pydantic BaseModel
3. Add `to_dict()` method
4. Add tests in `tests/test_data_models.py`

## Configuration

### Environment Variables
Create `.env` file:
```
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
LOG_LEVEL=INFO
```

### Docker
- Backend: Port 8000
- Frontend: Port 3000
- Redis: Port 6379

## Notes

- All Python code is in `/backend`
- No duplicate `/src` directory at root
- Tests import from `backend.src`
- Frontend calls backend API at `http://localhost:8000`
- CORS is configured for frontend origins
