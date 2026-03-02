# GramBrain AI - Start Here 🚀

Welcome to GramBrain AI! This document will guide you through the project and help you get started.

## 📋 What is GramBrain AI?

GramBrain AI is a comprehensive agricultural advisory system powered by AI agents. It helps farmers make better decisions about their farms through intelligent recommendations.

**Key Components:**
- 🤖 12 specialized AI agents
- 🌾 Farm management system
- 🛒 Product marketplace
- 💬 AI query interface
- 📊 User dashboard

## 🎯 Quick Navigation

### For First-Time Users
1. **Start here**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick reference guide
2. **Then read**: [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - How to run the application
3. **Finally**: [FRONTEND_SETUP.md](frontend/FRONTEND_SETUP.md) - Frontend development guide

### For Developers
1. **Backend**: [backend/README.md](backend/README.md) - Backend setup
2. **Frontend**: [frontend/README.md](frontend/README.md) - Frontend setup
3. **Integration**: [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - How to integrate

### For Project Managers
1. **Status**: [PROJECT_STATUS.md](PROJECT_STATUS.md) - Complete project status
2. **Summary**: [FINAL_COMPLETION_REPORT.md](FINAL_COMPLETION_REPORT.md) - Completion report
3. **Frontend**: [FRONTEND_COMPLETION_SUMMARY.md](FRONTEND_COMPLETION_SUMMARY.md) - Frontend summary

## 🚀 Getting Started (5 minutes)

### Option 1: Using Docker (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd grambrain-ai

# Start everything
docker-compose up

# Open in browser
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 2: Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn src.api.routes:app --reload
```

**Frontend (in another terminal):**
```bash
cd frontend
npm install
npm run dev
```

## 📚 Documentation Guide

### Essential Documents
| Document | Purpose | Audience |
|----------|---------|----------|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Quick reference for common tasks | Everyone |
| [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) | How to integrate frontend and backend | Developers |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Complete project status | Project Managers |
| [FINAL_COMPLETION_REPORT.md](FINAL_COMPLETION_REPORT.md) | Project completion report | Stakeholders |

### Setup Guides
| Document | Purpose | Audience |
|----------|---------|----------|
| [FRONTEND_SETUP.md](frontend/FRONTEND_SETUP.md) | Frontend development guide | Frontend Developers |
| [backend/README.md](backend/README.md) | Backend setup guide | Backend Developers |
| [frontend/README.md](frontend/README.md) | Frontend quick start | Frontend Developers |

### Project Documentation
| Document | Purpose | Audience |
|----------|---------|----------|
| [STRUCTURE_REORGANIZATION_SUMMARY.md](STRUCTURE_REORGANIZATION_SUMMARY.md) | Project structure overview | Everyone |
| [FRONTEND_COMPLETION_SUMMARY.md](FRONTEND_COMPLETION_SUMMARY.md) | Frontend implementation summary | Developers |
| [REORGANIZATION_GUIDE.md](REORGANIZATION_GUIDE.md) | Project reorganization details | Developers |

## 🏗️ Project Structure

```
grambrain-ai/
├── backend/                          # Python/FastAPI backend
│   ├── src/
│   │   ├── agents/                   # 12 AI agents
│   │   ├── api/routes.py             # 20+ API endpoints
│   │   ├── core/                     # Agent framework
│   │   ├── data/models.py            # Data models
│   │   ├── llm/                      # LLM integration
│   │   └── rag/                      # RAG pipeline
│   ├── tests/                        # 85+ test cases
│   └── README.md
│
├── frontend/                         # Next.js/React frontend
│   ├── src/
│   │   ├── app/                      # 8 pages
│   │   ├── components/               # 7 components
│   │   ├── hooks/                    # 3 custom hooks
│   │   ├── services/api.ts           # API client
│   │   ├── store/appStore.ts         # State management
│   │   └── types/index.ts            # Type definitions
│   └── README.md
│
├── docker-compose.yml                # Docker orchestration
├── INTEGRATION_GUIDE.md              # Integration guide
├── QUICK_REFERENCE.md                # Quick reference
├── PROJECT_STATUS.md                 # Project status
└── START_HERE.md                     # This file
```

## 🎯 What Can You Do?

### As a User
1. Register for an account
2. Create and manage farms
3. Ask AI agents for recommendations
4. Browse the marketplace
5. View your profile

### As a Developer
1. Extend the AI agents
2. Add new API endpoints
3. Create new frontend pages
4. Integrate new services
5. Write tests

### As a Project Manager
1. Track project status
2. Monitor deployment
3. Review documentation
4. Plan next phases

## 🔑 Key Features

### Backend
- ✅ 12 specialized AI agents
- ✅ 20+ REST API endpoints
- ✅ AWS Bedrock LLM integration
- ✅ RAG pipeline for knowledge retrieval
- ✅ 85+ test cases
- ✅ Production-ready code

### Frontend
- ✅ 8 complete pages
- ✅ 7 reusable components
- ✅ 3 custom hooks
- ✅ Complete API integration
- ✅ Responsive design
- ✅ Global state management

## 🧪 Testing

### Run Backend Tests
```bash
cd backend
pytest tests/ -v
```

### Run Frontend Tests
```bash
cd frontend
npm test
```

## 📊 Project Status

**Overall Status**: ✅ COMPLETE

- ✅ Backend: 100% complete
- ✅ Frontend: 100% complete
- ✅ API Integration: 100% complete
- ✅ Documentation: 100% complete
- ✅ Docker Support: 100% complete

**Ready for**: Integration Testing & Deployment

## 🚢 Deployment

### Development
```bash
docker-compose up
```

### Production
```bash
docker-compose -f docker-compose.yml up -d
```

## 🆘 Need Help?

### Common Issues

**Backend won't start:**
```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall dependencies
pip install -r requirements.txt
```

**Frontend won't start:**
```bash
# Check Node version
node --version  # Should be 18+

# Clear cache and reinstall
rm -rf node_modules .next
npm install
```

**API connection errors:**
```bash
# Check backend is running
curl http://localhost:8000/health

# Check frontend API URL
# Verify NEXT_PUBLIC_API_URL in .env.local
```

### Documentation
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick reference
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Integration help
- [FRONTEND_SETUP.md](frontend/FRONTEND_SETUP.md) - Frontend help
- [backend/README.md](backend/README.md) - Backend help

## 📞 Support

For issues or questions:
1. Check the relevant README file
2. Review the QUICK_REFERENCE.md
3. Check the INTEGRATION_GUIDE.md
4. Review error messages in logs

## 🎓 Learning Path

### Beginner
1. Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. Run the application with Docker
3. Explore the UI
4. Read [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)

### Intermediate
1. Read [FRONTEND_SETUP.md](frontend/FRONTEND_SETUP.md)
2. Read [backend/README.md](backend/README.md)
3. Explore the code
4. Run tests

### Advanced
1. Understand the architecture
2. Extend the AI agents
3. Add new features
4. Deploy to production

## 🎯 Next Steps

1. **Get Started**: Run `docker-compose up`
2. **Explore**: Visit http://localhost:3000
3. **Read**: Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
4. **Develop**: Follow [FRONTEND_SETUP.md](frontend/FRONTEND_SETUP.md)
5. **Deploy**: Follow [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)

## 📝 Important Files

### Configuration
- `backend/.env` - Backend configuration
- `frontend/.env.local` - Frontend configuration
- `docker-compose.yml` - Docker configuration

### Documentation
- `QUICK_REFERENCE.md` - Quick reference
- `INTEGRATION_GUIDE.md` - Integration guide
- `PROJECT_STATUS.md` - Project status
- `FINAL_COMPLETION_REPORT.md` - Completion report

### Code
- `backend/src/` - Backend source code
- `frontend/src/` - Frontend source code
- `backend/tests/` - Backend tests
- `frontend/tests/` - Frontend tests

## 🎉 You're Ready!

Everything is set up and ready to go. Choose your path:

- **I want to run the app**: Go to [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- **I want to develop**: Go to [FRONTEND_SETUP.md](frontend/FRONTEND_SETUP.md)
- **I want quick reference**: Go to [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **I want project status**: Go to [PROJECT_STATUS.md](PROJECT_STATUS.md)

---

**Welcome to GramBrain AI! 🌾🤖**

**Last Updated**: February 28, 2026  
**Version**: 1.0.0
