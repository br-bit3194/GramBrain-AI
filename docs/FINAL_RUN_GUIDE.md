# GramBrain AI - Final Running Guide

## ✅ Everything is Fixed and Ready!

All issues have been resolved. Follow this guide to run the application end-to-end.

---

## 🚀 Step 1: Start Backend (Terminal 1)

```bash
cd backend
source venv/bin/activate
python -m uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**Verify Backend:**
```bash
# In another terminal
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "success",
  "data": {
    "status": "healthy",
    "timestamp": "...",
    "agents": [...]
  }
}
```

---

## 🚀 Step 2: Start Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

**Expected Output:**
```
> next dev
  ▲ Next.js 13.x.x
  - Local:        http://localhost:3000
```

**Verify Frontend:**
- Open http://localhost:3000 in browser
- Should see landing page with "AI Brain for Every Farm"

---

## 🧪 Step 3: Test the Application

### 3.1 Test Home Page
- ✅ Open http://localhost:3000
- ✅ See landing page
- ✅ See "Get Started" button

### 3.2 Test User Registration
- ✅ Click "Get Started"
- ✅ Fill form:
  - Name: Test Farmer
  - Phone: +91 98765 43210
  - Language: English
  - Role: Farmer
- ✅ Click "Create Account"
- ✅ Should redirect to dashboard

### 3.3 Test Dashboard
- ✅ See welcome message
- ✅ See quick stats
- ✅ See action buttons

### 3.4 Test Farm Creation
- ✅ Click "Manage Farms"
- ✅ Click "Add Farm"
- ✅ Fill form:
  - Latitude: 28.7041
  - Longitude: 77.1025
  - Area: 5.5
  - Soil Type: Loamy
  - Irrigation: Drip
- ✅ Click "Create Farm"
- ✅ Farm appears in list

### 3.5 Test Query Interface
- ✅ Click "Ask Now"
- ✅ Enter: "How should I irrigate my wheat farm?"
- ✅ Select crop: Wheat
- ✅ Select stage: Vegetative
- ✅ Click "Get Recommendation"
- ✅ See recommendation response

### 3.6 Test Marketplace
- ✅ Click "Browse Products"
- ✅ See product list
- ✅ Test search and filter

### 3.7 Test Profile
- ✅ Click user name in header
- ✅ See profile info
- ✅ Click "Logout"
- ✅ Redirected to home

### 3.8 Test Login
- ✅ Go to /login
- ✅ Enter phone: +91 98765 43210
- ✅ Click "Login"
- ✅ Redirected to dashboard

---

## 🔗 Important URLs

### Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Pages
- Home: http://localhost:3000/
- Dashboard: http://localhost:3000/dashboard
- Farms: http://localhost:3000/farms
- Query: http://localhost:3000/query
- Marketplace: http://localhost:3000/marketplace
- Login: http://localhost:3000/login
- Register: http://localhost:3000/register
- Profile: http://localhost:3000/profile

---

## ✅ Verification Checklist

- [ ] Backend running on http://localhost:8000
- [ ] Frontend running on http://localhost:3000
- [ ] Health check passes
- [ ] User registration works
- [ ] Farm creation works
- [ ] Query submission works
- [ ] Marketplace works
- [ ] Profile works
- [ ] Login/logout works
- [ ] No console errors
- [ ] API calls visible in Network tab

---

## 🐛 Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
lsof -ti:8000 | xargs kill -9
```

**Module not found:**
```bash
cd backend
pip install -r requirements.txt
```

**Import error:**
- Ensure you're in the backend directory
- Ensure venv is activated
- Ensure all files are in place

### Frontend Issues

**Port 3000 already in use:**
```bash
lsof -ti:3000 | xargs kill -9
```

**Module not found:**
```bash
cd frontend
rm -rf node_modules .next
npm install
```

**Component export error:**
- Clear cache: `rm -rf .next`
- Restart: `npm run dev`

**API connection error:**
- Check backend is running
- Check NEXT_PUBLIC_API_URL in .env.local
- Check browser console for CORS errors

---

## 📊 What's Running

### Backend
- ✅ FastAPI server on port 8000
- ✅ 12 AI agents ready
- ✅ 20+ API endpoints
- ✅ Mock LLM and RAG
- ✅ Auto-reload enabled

### Frontend
- ✅ Next.js server on port 3000
- ✅ 8 pages
- ✅ 7 components
- ✅ 3 custom hooks
- ✅ Hot reload enabled

---

## 🎯 Success Criteria

All of these should work:
- ✅ Backend starts without errors
- ✅ Frontend starts without errors
- ✅ Can register user
- ✅ Can create farm
- ✅ Can submit query
- ✅ Can browse marketplace
- ✅ Can view profile
- ✅ Can logout and login
- ✅ API documentation accessible
- ✅ No console errors

**If all above work → Application is running successfully! 🎉**

---

## 📚 Documentation

For more details, see:
- **COMPLETE_SETUP_READY.md** - Setup overview
- **HOW_TO_RUN.md** - Detailed guide
- **RUN_END_TO_END.md** - Step-by-step
- **QUICK_REFERENCE.md** - Quick lookup
- **INTEGRATION_GUIDE.md** - Integration details

---

## 🚀 You're Ready!

Everything is set up and ready to go!

**Start the application:**
1. Terminal 1: `cd backend && source venv/bin/activate && python -m uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000`
2. Terminal 2: `cd frontend && npm run dev`
3. Open: http://localhost:3000

**Happy coding! 🌾🤖**
