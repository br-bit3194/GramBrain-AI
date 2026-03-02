# GramBrain AI - Reorganization Guide

## New Project Structure

The project has been reorganized to separate backend and frontend into distinct folders:

```
grambrain-ai/
├── backend/                    # Backend (Python/FastAPI)
│   ├── src/
│   │   ├── core/              # Agent framework
│   │   │   ├── __init__.py
│   │   │   ├── agent_base.py
│   │   │   ├── agent_registry.py
│   │   │   └── orchestrator.py
│   │   │
│   │   ├── agents/            # 11 specialized agents
│   │   │   ├── __init__.py
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
│   │   ├── llm/               # LLM integration
│   │   │   ├── __init__.py
│   │   │   └── bedrock_client.py
│   │   │
│   │   ├── rag/               # RAG pipeline
│   │   │   ├── __init__.py
│   │   │   ├── vector_db.py
│   │   │   ├── embeddings.py
│   │   │   └── retrieval.py
│   │   │
│   │   ├── data/              # Data models
│   │   │   ├── __init__.py
│   │   │   └── models.py
│   │   │
│   │   ├── api/               # REST API
│   │   │   ├── __init__.py
│   │   │   └── routes.py
│   │   │
│   │   ├── __init__.py
│   │   └── system.py          # Main system
│   │
│   ├── tests/                 # Test suite
│   │   ├── __init__.py
│   │   ├── test_agents.py
│   │   ├── test_orchestrator.py
│   │   ├── test_data_models.py
│   │   ├── test_rag.py
│   │   └── test_api.py
│   │
│   ├── main.py                # API server entry point
│   ├── requirements.txt        # Python dependencies
│   ├── pytest.ini              # Pytest configuration
│   ├── .env.example            # Environment template
│   └── README.md               # Backend README
│
├── frontend/                   # Frontend (React/Next.js - To be created)
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/             # Next.js pages
│   │   ├── hooks/             # Custom hooks
│   │   ├── services/          # API services
│   │   ├── styles/            # CSS/styling
│   │   └── utils/             # Utilities
│   │
│   ├── public/                # Static assets
│   ├── package.json           # NPM dependencies
│   ├── next.config.js         # Next.js config
│   ├── tsconfig.json          # TypeScript config
│   └── README.md              # Frontend README
│
├── docs/                      # Documentation
│   ├── API.md                 # API reference
│   ├── TESTING.md             # Testing guide
│   ├── QUICKSTART.md          # Quick start
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── COMPLETION_REPORT.md
│   ├── design.md              # System design
│   └── requirements.md        # Requirements
│
├── docker-compose.yml         # Docker compose for full stack
├── .gitignore                 # Git ignore
├── README.md                  # Main README
├── INDEX.md                   # Complete index
└── BUILD_SUMMARY.txt          # Build summary
```

## Migration Steps

### Step 1: Copy Backend Files

All Python source files should be moved to `backend/src/`:

```bash
# Core framework
cp src/core/* backend/src/core/

# Agents
cp src/agents/* backend/src/agents/

# LLM integration
cp src/llm/* backend/src/llm/

# RAG pipeline
cp src/rag/* backend/src/rag/

# Data models
cp src/data/* backend/src/data/

# API
cp src/api/* backend/src/api/

# Main system
cp src/system.py backend/src/

# Tests
cp tests/* backend/tests/

# Configuration files
cp main.py backend/
cp requirements.txt backend/
cp pytest.ini backend/
cp .env.example backend/
```

### Step 2: Create Frontend Structure

```bash
mkdir -p frontend/src/{components,pages,hooks,services,styles,utils}
mkdir -p frontend/public
```

### Step 3: Update Import Paths

In backend files, update imports from:
```python
from src.core import ...
```

To:
```python
from src.core import ...
# (No change needed - relative imports work)
```

### Step 4: Create Documentation Folder

```bash
mkdir -p docs
cp API.md docs/
cp TESTING.md docs/
cp QUICKSTART.md docs/
cp IMPLEMENTATION_SUMMARY.md docs/
cp COMPLETION_REPORT.md docs/
cp design.md docs/
cp requirements.md docs/
```

### Step 5: Update Root Files

Keep at root level:
- README.md (main project README)
- INDEX.md (complete index)
- BUILD_SUMMARY.txt (build summary)
- docker-compose.yml (full stack orchestration)
- .gitignore (git configuration)

## Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

API available at: `http://localhost:8000`

## Frontend Setup (To be created)

```bash
cd frontend
npm install
npm run dev
```

Frontend available at: `http://localhost:3000`

## Running Full Stack

### Option 1: Manual

Terminal 1:
```bash
cd backend
python main.py
```

Terminal 2:
```bash
cd frontend
npm run dev
```

### Option 2: Docker Compose

```bash
docker-compose up
```

## File Organization Benefits

✅ **Clear Separation of Concerns**
- Backend: Python/FastAPI/AI logic
- Frontend: React/Next.js/UI

✅ **Independent Development**
- Backend team can work independently
- Frontend team can work independently
- Different tech stacks

✅ **Easier Deployment**
- Backend can be deployed to AWS Lambda/ECS
- Frontend can be deployed to Vercel/CloudFront
- Different scaling strategies

✅ **Better Testing**
- Backend tests in `backend/tests/`
- Frontend tests in `frontend/tests/` (to be created)

✅ **Cleaner Repository**
- Documentation in `docs/`
- Each folder has its own README
- Clear entry points (main.py, package.json)

## Environment Variables

### Backend (.env)
```
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
DEFAULT_LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
```

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_APP_NAME=GramBrain AI
```

## Dependencies

### Backend
- Python 3.9+
- FastAPI
- boto3
- pytest

### Frontend (To be created)
- Node.js 16+
- React 18+
- Next.js 13+
- TypeScript

## Documentation Structure

```
docs/
├── API.md                    # REST API reference
├── TESTING.md                # Testing guide
├── QUICKSTART.md             # Quick start guide
├── IMPLEMENTATION_SUMMARY.md # Implementation details
├── COMPLETION_REPORT.md      # Completion status
├── design.md                 # System design
└── requirements.md           # Original requirements
```

## Next Steps

1. **Backend**: Already complete and ready for deployment
2. **Frontend**: Create React/Next.js application
3. **Integration**: Connect frontend to backend API
4. **Deployment**: Deploy backend and frontend separately
5. **Documentation**: Update docs with frontend information

## Quick Reference

### Backend Commands
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py              # Start API
pytest tests/ -v            # Run tests
pytest tests/ --cov=src     # Run with coverage
```

### Frontend Commands (To be created)
```bash
cd frontend
npm install
npm run dev                 # Start dev server
npm run build               # Build for production
npm test                    # Run tests
```

### Root Commands
```bash
docker-compose up           # Start full stack
docker-compose down         # Stop full stack
```

## File Locations

| Component | Location |
|-----------|----------|
| Backend API | `backend/src/api/routes.py` |
| Agents | `backend/src/agents/` |
| Data Models | `backend/src/data/models.py` |
| Tests | `backend/tests/` |
| API Docs | `docs/API.md` |
| Testing Guide | `docs/TESTING.md` |
| Frontend (TBD) | `frontend/src/` |

## Support

- Backend Issues: See `backend/README.md`
- Frontend Issues: See `frontend/README.md` (to be created)
- General Issues: See `README.md`
- API Reference: See `docs/API.md`
- Testing: See `docs/TESTING.md`

---

**Status:** Backend reorganization complete. Frontend structure ready for development.
