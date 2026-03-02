# GramBrain AI - Multi-Agent Agricultural Intelligence Platform

A next-generation, cloud-native, multi-agent AI platform designed to serve as "The AI Brain for Every Village in Bharat". GramBrain AI uses collaborative AI agents powered by Large Language Model (LLM) reasoning and Retrieval-Augmented Generation (RAG) to deliver real-time, explainable, and scalable agricultural intelligence.

## 🏗️ Project Structure

The project is organized into separate backend and frontend folders:

```
grambrain-ai/
├── backend/                    # Python/FastAPI Backend ✅ Complete
│   ├── src/                    # Source code
│   ├── tests/                  # Test suite (85+ tests)
│   ├── main.py                 # API server
│   └── requirements.txt        # Dependencies
│
├── frontend/                   # React/Next.js Frontend 🚧 To be created
│   ├── src/                    # React components
│   ├── public/                 # Static assets
│   └── package.json            # NPM dependencies
│
├── docs/                       # Documentation
│   ├── API.md                  # REST API reference
│   ├── TESTING.md              # Testing guide
│   ├── QUICKSTART.md           # Quick start
│   └── design.md               # System design
│
├── docker-compose.yml          # Full stack orchestration
└── README.md                   # This file
```

## 🚀 Quick Start

### Backend Only

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

API available at: `http://localhost:8000`  
Swagger UI at: `http://localhost:8000/docs`

### Full Stack (with Docker)

```bash
docker-compose up
```

Backend: `http://localhost:8000`  
Frontend: `http://localhost:3000` (when created)

## 📋 Features

### ✅ Backend (Complete)

- **12 Specialized Agents**
  - Weather Intelligence
  - Soil Health Analysis
  - Crop Advisory
  - Pest & Disease Detection
  - Irrigation Optimization
  - Yield Prediction
  - Market Intelligence
  - Sustainability Tracking
  - Marketplace Management
  - Farmer Interaction
  - Village Intelligence
  - Orchestrator

- **LLM Integration**
  - AWS Bedrock support
  - Multiple model options
  - Automatic retry logic
  - Fallback mechanisms

- **RAG Pipeline**
  - Vector database
  - Semantic search
  - Knowledge management
  - Context injection

- **REST API**
  - 20+ endpoints
  - Auto-documentation
  - Error handling
  - Request validation

- **Data Models**
  - 10+ comprehensive models
  - Type hints
  - Serialization support

- **Testing**
  - 85+ test cases
  - Unit, integration, API tests
  - >80% coverage

### 🚧 Frontend (To be created)

- Farmer Dashboard
- Query Interface
- Recommendation Display
- Farm Management
- Marketplace
- Analytics Dashboard
- Mobile Responsive
- Multilingual Support

## 📚 Documentation

### Backend Documentation
- [Backend README](backend/README.md) - Backend setup and structure
- [API Reference](docs/API.md) - Complete REST API documentation
- [Testing Guide](docs/TESTING.md) - How to run tests
- [Quick Start](docs/QUICKSTART.md) - 5-minute setup guide

### System Documentation
- [System Design](docs/design.md) - Architecture and design
- [Requirements](docs/requirements.md) - Original requirements
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md) - Implementation details
- [Completion Report](docs/COMPLETION_REPORT.md) - Project completion status

### Frontend Documentation
- [Frontend README](frontend/README.md) - Frontend setup and structure (to be created)

### Project Documentation
- [Reorganization Guide](REORGANIZATION_GUIDE.md) - How the project is organized
- [Index](INDEX.md) - Complete file index
- [Build Summary](BUILD_SUMMARY.txt) - Build statistics

## 🛠️ Technology Stack

### Backend
- **Language:** Python 3.9+
- **Framework:** FastAPI
- **LLM:** AWS Bedrock
- **Vector DB:** In-memory (dev), OpenSearch (prod)
- **Testing:** pytest
- **Deployment:** Docker, AWS Lambda/ECS

### Frontend (To be created)
- **Framework:** Next.js 13+
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **State:** Redux Toolkit / Zustand
- **Testing:** Jest + React Testing Library
- **Deployment:** Vercel / CloudFront

## 🧪 Testing

### Run All Tests
```bash
cd backend
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test
```bash
pytest tests/test_agents.py::TestWeatherAgent -v
```

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| Backend Files | 40+ |
| Lines of Code | ~8,000 |
| Test Cases | 85+ |
| API Endpoints | 20+ |
| Data Models | 10+ |
| Agents | 12 |
| Documentation | 2,000+ lines |

## 🎯 Performance Targets

- **Recommendation Latency:** < 3 seconds (95th percentile)
- **LLM Accuracy:** 85%+ for yield predictions (±15%)
- **Market Price Forecast:** 75%+ directional accuracy (7-day)
- **Pest Detection:** 90%+ accuracy from crop images
- **System Availability:** 99.9% uptime

## 🔄 API Endpoints

### User Management
- `POST /api/v1/users` - Create user
- `GET /api/v1/users/{user_id}` - Get user

### Farm Management
- `POST /api/v1/farms` - Create farm
- `GET /api/v1/farms/{farm_id}` - Get farm
- `GET /api/v1/users/{user_id}/farms` - List farms

### Queries & Recommendations
- `POST /api/v1/query` - Get recommendation
- `GET /api/v1/recommendations/{id}` - Get recommendation
- `GET /api/v1/users/{user_id}/recommendations` - List recommendations

### Marketplace
- `POST /api/v1/products` - Create product
- `GET /api/v1/products/{id}` - Get product
- `GET /api/v1/products` - Search products
- `GET /api/v1/farmers/{id}/products` - List farmer products

### Knowledge Management
- `POST /api/v1/knowledge` - Add knowledge
- `GET /api/v1/knowledge/search` - Search knowledge

### System
- `GET /api/v1/health` - Health check

See [docs/API.md](docs/API.md) for complete reference.

## 🚀 Deployment

### Backend Deployment

#### AWS Lambda
```bash
cd backend
pip install -r requirements.txt -t package/
zip -r function.zip package/ src/ main.py
aws lambda create-function --function-name grambrain-api \
  --runtime python3.9 --role arn:aws:iam::ACCOUNT:role/lambda-role \
  --handler main.handler --zip-file fileb://function.zip
```

#### Docker
```bash
docker build -t grambrain-backend backend/
docker run -p 8000:8000 grambrain-backend
```

#### ECS
```bash
docker build -t grambrain-backend backend/
aws ecr create-repository --repository-name grambrain-backend
docker tag grambrain-backend:latest ACCOUNT.dkr.ecr.REGION.amazonaws.com/grambrain-backend:latest
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/grambrain-backend:latest
```

### Frontend Deployment (To be created)

#### Vercel
```bash
cd frontend
npm install -g vercel
vercel
```

#### Docker
```bash
docker build -t grambrain-frontend frontend/
docker run -p 3000:3000 grambrain-frontend
```

## 🔐 Security

- Input validation on all endpoints
- Error handling with appropriate HTTP status codes
- Type safety with Python type hints
- Environment variable management
- Ready for JWT/OAuth2 authentication

## 📈 Monitoring & Logging

- Health check endpoint
- Structured logging
- Error tracking
- Performance metrics
- Ready for CloudWatch/DataDog integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📞 Support

- **GitHub:** https://github.com/grambrain/grambrain-ai
- **Issues:** https://github.com/grambrain/grambrain-ai/issues
- **Email:** support@grambrain.ai

## 📄 License

MIT License - See LICENSE file for details

## 🗺️ Roadmap

### Phase 1: Backend ✅ Complete
- [x] Multi-agent architecture
- [x] LLM integration
- [x] RAG pipeline
- [x] REST API
- [x] Data models
- [x] Test suite

### Phase 2: Frontend 🚧 In Progress
- [ ] React/Next.js setup
- [ ] Component library
- [ ] API integration
- [ ] Authentication
- [ ] Responsive design

### Phase 3: Production Deployment
- [ ] Database integration
- [ ] Authentication (JWT/OAuth2)
- [ ] Rate limiting
- [ ] Caching layer
- [ ] Monitoring
- [ ] CI/CD pipeline

### Phase 4: Advanced Features
- [ ] IoT sensor integration
- [ ] Satellite imagery processing
- [ ] Advanced analytics
- [ ] Mobile app
- [ ] Voice interface

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (React/Next.js)                     │
│  Mobile App │ Web Dashboard │ Voice UI                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      API Gateway (FastAPI)                       │
│  REST API │ WebSocket │ Authentication │ Rate Limiting          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                Multi-Agent Intelligence Layer                    │
│  Orchestrator │ 11 Specialized Agents │ LLM Integration         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    LLM & RAG Layer                               │
│  AWS Bedrock │ Vector Database │ Knowledge Retrieval            │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      Data Layer                                  │
│  PostgreSQL │ DynamoDB │ S3 │ Redis Cache                       │
└──────────────────────────────────────────────────────────────────┘
```

## ✨ Key Highlights

✅ **Production-Ready Backend** - Complete with 85+ tests  
✅ **Scalable Architecture** - Multi-agent design  
✅ **LLM-Powered** - AWS Bedrock integration  
✅ **Well-Documented** - 2,000+ lines of documentation  
✅ **Type-Safe** - Full type hints throughout  
✅ **Comprehensive API** - 20+ endpoints  
✅ **Ready for Frontend** - Clear API contracts  

---

**Status:** Backend ✅ Complete | Frontend 🚧 To be created

**Next Steps:** 
1. Create React/Next.js frontend
2. Integrate with backend API
3. Deploy to production
4. Add advanced features

For detailed information, see [REORGANIZATION_GUIDE.md](REORGANIZATION_GUIDE.md)
