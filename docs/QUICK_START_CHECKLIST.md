# GramBrain AI - Quick Start Checklist

Use this checklist to run the application end-to-end.

## 🚀 Quick Start (5 minutes)

### Prerequisites
- [ ] Node.js 18+ installed (`node --version`)
- [ ] Python 3.9+ installed (`python --version`)
- [ ] npm installed (`npm --version`)

### Backend Setup (Terminal 1)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)
pip install -r requirements.txt
cp .env.example .env
# Edit .env if needed
python -m uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

**Checklist:**
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] .env file created
- [ ] Backend running on http://localhost:8000
- [ ] Health check passes: `curl http://localhost:8000/health`

### Frontend Setup (Terminal 2)
```bash
cd frontend
npm install
cp .env.example .env.local
# Edit .env.local: NEXT_PUBLIC_API_URL=http://localhost:8000/api
npm run dev
```

**Checklist:**
- [ ] Dependencies installed
- [ ] .env.local created
- [ ] Frontend running on http://localhost:3000
- [ ] No console errors in browser

---

## 🧪 Testing Workflow

### 1. Home Page
- [ ] Open http://localhost:3000
- [ ] See landing page with features
- [ ] See "Get Started" and "Login" buttons

### 2. User Registration
- [ ] Click "Get Started"
- [ ] Fill registration form:
  - [ ] Name: Test Farmer
  - [ ] Phone: +91 98765 43210
  - [ ] Language: English
  - [ ] Role: Farmer
- [ ] Click "Create Account"
- [ ] Redirected to dashboard

### 3. Dashboard
- [ ] See welcome message with user name
- [ ] See quick stats (farm area, soil type, irrigation)
- [ ] See quick action buttons

### 4. Farm Management
- [ ] Click "Manage Farms"
- [ ] Click "Add Farm"
- [ ] Fill farm form:
  - [ ] Latitude: 28.7041
  - [ ] Longitude: 77.1025
  - [ ] Area: 5.5
  - [ ] Soil Type: Loamy
  - [ ] Irrigation: Drip
- [ ] Click "Create Farm"
- [ ] Farm appears in list

### 5. Query Interface
- [ ] Click "Ask Now" or go to /query
- [ ] Enter query: "How should I irrigate my wheat farm?"
- [ ] Select crop type: Wheat
- [ ] Select growth stage: Vegetative
- [ ] Click "Get Recommendation"
- [ ] See loading state
- [ ] See recommendation response

### 6. Marketplace
- [ ] Click "Browse Products" or go to /marketplace
- [ ] See product list (if products exist)
- [ ] Test search: type "tomato"
- [ ] Test filter: select "Vegetables"
- [ ] See filtered results

### 7. User Profile
- [ ] Click user name in header or go to /profile
- [ ] See profile information:
  - [ ] Full name
  - [ ] Phone number
  - [ ] Language preference
  - [ ] Role
  - [ ] Member since date
- [ ] Click "Logout"
- [ ] Redirected to home page

### 8. Login
- [ ] Go to /login
- [ ] Enter phone: +91 98765 43210
- [ ] Click "Login"
- [ ] Redirected to dashboard
- [ ] User data restored

---

## 🔍 API Testing

### Health Check
```bash
curl http://localhost:8000/health
```
- [ ] Returns 200 status
- [ ] Shows agent list

### API Documentation
- [ ] Open http://localhost:8000/docs
- [ ] See Swagger UI
- [ ] Explore endpoints
- [ ] Try "Try it out" on any endpoint

### Test User Creation
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
- [ ] Returns 200 status
- [ ] Returns user object

### Test Farm Creation
```bash
curl -X POST http://localhost:8000/api/farms \
  -H "Content-Type: application/json" \
  -d '{
    "owner_id": "user-id",
    "latitude": 28.7041,
    "longitude": 77.1025,
    "area_hectares": 5.5,
    "soil_type": "loamy",
    "irrigation_type": "drip"
  }'
```
- [ ] Returns 200 status
- [ ] Returns farm object

---

## 🧪 Run Tests

### Backend Tests
```bash
cd backend
pytest tests/ -v
```
- [ ] All tests pass
- [ ] 85+ tests passing
- [ ] No failures

### Frontend Tests
```bash
cd frontend
npm test
```
- [ ] Tests run without errors
- [ ] Can press 'q' to quit

---

## 🔧 Troubleshooting

### Backend Issues
- [ ] Port 8000 in use? Kill process: `lsof -ti:8000 | xargs kill -9`
- [ ] Module not found? Reinstall: `pip install -r requirements.txt`
- [ ] Database error? Check .env DATABASE_URL

### Frontend Issues
- [ ] Port 3000 in use? Kill process: `lsof -ti:3000 | xargs kill -9`
- [ ] Module not found? Reinstall: `rm -rf node_modules && npm install`
- [ ] API error? Check NEXT_PUBLIC_API_URL in .env.local

### API Connection Issues
- [ ] CORS error? Check backend CORS config
- [ ] Cannot reach API? Ensure backend is running
- [ ] Check browser console for errors

---

## 📊 Verification Checklist

### Backend
- [ ] Running on http://localhost:8000
- [ ] Health endpoint responds
- [ ] API documentation accessible at /docs
- [ ] No errors in terminal

### Frontend
- [ ] Running on http://localhost:3000
- [ ] Landing page loads
- [ ] No errors in browser console
- [ ] Can navigate between pages

### Integration
- [ ] Frontend can reach backend API
- [ ] User registration works
- [ ] Farm creation works
- [ ] Query submission works
- [ ] All pages load without errors

### Tests
- [ ] Backend tests pass (85+)
- [ ] Frontend tests run
- [ ] No test failures

---

## 🎯 Success Criteria

All items checked = ✅ Application is running successfully!

- [ ] Backend running
- [ ] Frontend running
- [ ] User registration works
- [ ] Farm management works
- [ ] Query interface works
- [ ] Marketplace works
- [ ] Profile works
- [ ] Login/logout works
- [ ] API documentation accessible
- [ ] Tests passing
- [ ] No console errors
- [ ] API calls visible in Network tab

---

## 📚 Documentation

For more details, see:
- **RUN_END_TO_END.md** - Detailed step-by-step guide
- **INTEGRATION_GUIDE.md** - Integration instructions
- **QUICK_REFERENCE.md** - Quick reference
- **FRONTEND_SETUP.md** - Frontend development guide

---

## 🚀 You're Ready!

Once all items are checked, your GramBrain AI application is running end-to-end!

**Next Steps:**
1. Explore the code
2. Make changes
3. Test changes
4. Deploy to production

---

**Happy coding! 🌾🤖**
