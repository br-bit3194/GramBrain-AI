# How to Run GramBrain AI - Complete Guide

## 📋 Table of Contents
1. [Quick Start (5 min)](#quick-start)
2. [Detailed Setup](#detailed-setup)
3. [Testing the Application](#testing)
4. [Troubleshooting](#troubleshooting)
5. [Docker Alternative](#docker)

---

## Quick Start

### Option 1: Using Docker (Easiest)
```bash
docker-compose up
```
Then open:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs

### Option 2: Manual Setup (5 minutes)

**Terminal 1 - Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:3000

---

## Detailed Setup

### Prerequisites
```bash
# Check versions
node --version      # Should be 18+
npm --version       # Should be 8+
python --version    # Should be 3.9+
```

### Backend Setup

**Step 1: Navigate to backend**
```bash
cd backend
```

**Step 2: Create virtual environment**
```bash
# macOS/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Setup environment**
```bash
cp .env.example .env
# Edit .env with your settings (optional for basic testing)
```

**Step 5: Start backend**
```bash
python -m uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**Verify it's working:**
```bash
# In another terminal
curl http://localhost:8000/health
```

### Frontend Setup

**Step 1: Navigate to frontend**
```bash
cd frontend
```

**Step 2: Install dependencies**
```bash
npm install
```

**Step 3: Setup environment**
```bash
cp .env.example .env.local
# Edit .env.local if needed
# Default: NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

**Step 4: Start frontend**
```bash
npm run dev
```

**Expected output:**
```
> next dev
  ▲ Next.js 13.x.x
  - Local:        http://localhost:3000
```

**Verify it's working:**
- Open http://localhost:3000 in browser
- Should see landing page

---

## Testing the Application

### 1. Test Home Page
```
1. Open http://localhost:3000
2. See landing page with features
3. See "Get Started" button
```

### 2. Test User Registration
```
1. Click "Get Started"
2. Fill form:
   - Name: Test Farmer
   - Phone: +91 98765 43210
   - Language: English
   - Role: Farmer
3. Click "Create Account"
4. Should redirect to dashboard
```

### 3. Test Farm Creation
```
1. Click "Manage Farms"
2. Click "Add Farm"
3. Fill form:
   - Latitude: 28.7041
   - Longitude: 77.1025
   - Area: 5.5
   - Soil Type: Loamy
   - Irrigation: Drip
4. Click "Create Farm"
5. Farm should appear in list
```

### 4. Test Query Interface
```
1. Click "Ask Now"
2. Enter: "How should I irrigate my wheat farm?"
3. Select crop: Wheat
4. Select stage: Vegetative
5. Click "Get Recommendation"
6. Should see recommendation
```

### 5. Test Marketplace
```
1. Click "Browse Products"
2. Should see product list
3. Try search and filter
```

### 6. Test Profile
```
1. Click user name in header
2. Should see profile info
3. Click "Logout"
4. Should redirect to home
```

### 7. Test Login
```
1. Go to /login
2. Enter phone: +91 98765 43210
3. Click "Login"
4. Should redirect to dashboard
```

---

## API Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### API Documentation
```
Open: http://localhost:8000/docs
```

### Test Create User
```bash
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "+91 98765 43210",
    "name": "Test Farmer",
    "language_preference": "en",
    "role": "farmer"
  }'
```

### Test Create Farm
```bash
curl -X POST http://localhost:8000/api/farms \
  -H "Content-Type: application/json" \
  -d '{
    "owner_id": "test-user",
    "latitude": 28.7041,
    "longitude": 77.1025,
    "area_hectares": 5.5,
    "soil_type": "loamy",
    "irrigation_type": "drip"
  }'
```

---

## Running Tests

### Backend Tests
```bash
cd backend
pytest tests/ -v
```

Expected: 85+ tests passing

### Frontend Tests
```bash
cd frontend
npm test
```

---

## Troubleshooting

### Backend Won't Start

**Error: "Port 8000 already in use"**
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Error: "Module not found"**
```bash
pip install -r requirements.txt
```

**Error: "Python not found"**
```bash
# Make sure Python 3.9+ is installed
python --version
```

### Frontend Won't Start

**Error: "Port 3000 already in use"**
```bash
# macOS/Linux
lsof -ti:3000 | xargs kill -9

# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

**Error: "Cannot find module"**
```bash
rm -rf node_modules .next
npm install
```

**Error: "Node not found"**
```bash
# Make sure Node 18+ is installed
node --version
```

### API Connection Issues

**Error: "Cannot reach API"**
1. Check backend is running: `curl http://localhost:8000/health`
2. Check NEXT_PUBLIC_API_URL in frontend/.env.local
3. Check browser console for CORS errors

**Error: "CORS error"**
- Backend CORS is configured
- Ensure frontend URL is allowed
- Check browser console for details

---

## Docker Alternative

### Using Docker Compose

**Start everything:**
```bash
docker-compose up
```

**Stop everything:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs -f
```

**Access:**
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## File Structure

```
grambrain-ai/
├── backend/
│   ├── src/
│   │   ├── agents/          # AI agents
│   │   ├── api/routes.py    # API endpoints
│   │   ├── core/            # Framework
│   │   ├── data/models.py   # Data models
│   │   ├── llm/             # LLM integration
│   │   └── rag/             # RAG pipeline
│   ├── tests/               # Test suite
│   ├── requirements.txt     # Dependencies
│   └── README.md
│
├── frontend/
│   ├── src/
│   │   ├── app/             # Pages
│   │   ├── components/      # Components
│   │   ├── hooks/           # Custom hooks
│   │   ├── services/        # API client
│   │   ├── store/           # State management
│   │   └── types/           # Type definitions
│   ├── package.json         # Dependencies
│   └── README.md
│
├── docker-compose.yml       # Docker setup
├── RUN_END_TO_END.md        # Detailed guide
├── QUICK_START_CHECKLIST.md # Checklist
└── HOW_TO_RUN.md           # This file
```

---

## Environment Variables

### Backend (.env)
```
DATABASE_URL=postgresql://user:password@localhost/grambrain
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
```

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

---

## Common Commands

### Backend
```bash
# Start development server
python -m uvicorn src.api.routes:app --reload

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_agents.py -v

# Check code quality
pylint src/
```

### Frontend
```bash
# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run tests
npm test

# Run linter
npm run lint
```

---

## Verification Checklist

- [ ] Backend running on http://localhost:8000
- [ ] Frontend running on http://localhost:3000
- [ ] Health check passes
- [ ] User registration works
- [ ] Farm creation works
- [ ] Query submission works
- [ ] Marketplace works
- [ ] Profile works
- [ ] Login/logout works
- [ ] API documentation accessible
- [ ] Tests passing
- [ ] No console errors

---

## Next Steps

1. **Explore the code**
   - Backend: `backend/src/`
   - Frontend: `frontend/src/`

2. **Make changes**
   - Backend auto-reloads
   - Frontend hot-reloads

3. **Test changes**
   - Run tests
   - Check browser console

4. **Deploy**
   - Follow INTEGRATION_GUIDE.md

---

## Support

For issues:
1. Check Troubleshooting section
2. Review error messages
3. Check logs
4. Refer to documentation files

---

## Documentation Files

- **RUN_END_TO_END.md** - Detailed step-by-step guide
- **QUICK_START_CHECKLIST.md** - Quick checklist
- **INTEGRATION_GUIDE.md** - Integration instructions
- **QUICK_REFERENCE.md** - Quick reference
- **FRONTEND_SETUP.md** - Frontend development guide
- **PROJECT_STATUS.md** - Project status
- **START_HERE.md** - Entry point

---

**You're ready to run GramBrain AI! 🚀**

Choose your path:
- **Quick**: Use Docker (`docker-compose up`)
- **Manual**: Follow Quick Start section
- **Detailed**: Follow Detailed Setup section

Happy coding! 🌾🤖
