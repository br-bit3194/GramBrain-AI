# GramBrain AI - Backend

Multi-agent agricultural intelligence platform backend built with Python, FastAPI, and AWS Bedrock.

## Quick Start

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

API available at `http://localhost:8000`  
Swagger UI at `http://localhost:8000/docs`

## Structure

```
backend/
├── src/
│   ├── core/              # Agent framework
│   ├── agents/            # 11 specialized agents
│   ├── llm/               # LLM integration
│   ├── rag/               # RAG pipeline
│   ├── data/              # Data models
│   ├── api/               # REST API
│   └── system.py          # Main system
├── tests/                 # Test suite
├── main.py                # API server
├── requirements.txt       # Dependencies
└── pytest.ini             # Pytest config
```

## Documentation

- [API.md](../API.md) - REST API reference
- [TESTING.md](../TESTING.md) - Testing guide
- [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) - Implementation details

## Testing

```bash
pytest tests/ -v --cov=src
```

## API Endpoints

- `POST /api/v1/query` - Get recommendation
- `POST /api/v1/users` - Create user
- `POST /api/v1/farms` - Create farm
- `POST /api/v1/products` - Create product
- And 15+ more endpoints

See [API.md](../API.md) for complete reference.
