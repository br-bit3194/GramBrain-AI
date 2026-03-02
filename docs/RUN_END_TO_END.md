# GramBrain AI - End-to-End Run Guide

This guide will walk you through running the entire application from start to finish.

## Prerequisites Check

Before starting, ensure you have:
- Node.js 18+ (`node --version`)
- npm or yarn (`npm --version`)
- Python 3.9+ (`python --version`)
- Git (`git --version`)

## Step 1: Setup Backend (Terminal 1)

### 1.1 Navigate to Backend
```bash
cd backend
```

### 1.2 Create Virtual Environment
```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 1.3 Install Dependencies
```bash
pip install -r requirements.txt
```

### 1.4 Setup Environment Variables
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your configuration
# Required variables:
# - DATABASE_URL (if using database)
# - AWS_REGION
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - LLM_MODEL
```

### 1.5 Start Backend Server
```bash
python -m uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**Verify Backend is Running:**
```bash
# In another terminal, test the health endpoint
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "success",
  "data": {
    "status": "healthy",
    "agents": ["crop_advisory", "farmer_interaction", ...]
  }
}
```

---

## Step 2: Setup Frontend (Terminal 2)

### 2.1 Navigate to Frontend
```bash
cd frontend
```

### 2.2 Install Dependencies
```bash
npm install
```

### 2.3 Setup Environment Variables
```bash
# Copy example env file
cp .env.example .env.local

# Edit .env.local
# Set: NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

### 2.4 Start Frontend Development Server
```bash
npm run dev
```

**Expected Output:**
```
> next dev
  ▲ Next.js 13.x.x
  - Local:        http://localhost:3000
  - Environments: .env.local
```

---

## Step 3: Access the Application

### 3.1 Open in Browser
- **Frontend**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs
- **Backend ReDoc**: http://localhost:8000/redoc

### 3.2 Verify Frontend Loads
You should see the GramBrain AI landing page with:
- Hero section with "AI Brain for Every Farm"
- Features section
- Call-to-action buttons

---

## Step 4: Test User Registration

### 4.1 Click "Get Started" or "Register"
- Navigate to http://localhost:3000/register

### 4.2 Fill Registration Form
```
Full Name: Test Farmer
Phone Number: +91 98765 43210
Language: English
Role: Farmer
```

### 4.3 Click "Create Account"
- Should redirect to dashboard
- User data stored in Zustand store

### 4.4 Verify in Browser Console
```javascript
// Open browser DevTools (F12)
// Go to Console tab
// Type: localStorage
// Should see user data
```

---

## Step 5: Test Farm Management

### 5.1 Navigate to Farms Page
- Click "Manage Farms" on dashboard or go to http://localhost:3000/farms

### 5.2 Create a Farm
- Click "Add Farm" button
- Fill in form:
  ```
  Latitude: 28.7041
  Longitude: 77.1025
  Area: 5.5 hectares
  Soil Type: Loamy
  Irrigation: Drip
  ```
- Click "Create Farm"

### 5.3 Verify Farm Created
- Farm should appear in the list
- Should show all entered information

### 5.4 Test API Call
```bash
# In terminal, test the API
curl -X GET http://localhost:8000/api/users/test-user-id/farms
```

---

## Step 6: Test Query Interface

### 6.1 Navigate to Query Page
- Click "Ask Now" on dashboard or go to http://localhost:3000/query

### 6.2 Submit a Query
- Enter query: "How should I irrigate my wheat farm?"
- Select crop type: Wheat
- Select growth stage: Vegetative
- Click "Get Recommendation"

### 6.3 View Response
- Should see loading state
- Then display recommendation from AI agents

### 6.4 Test API Call
```bash
# Test query endpoint
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user",
    "query_text": "How should I irrigate my wheat farm?",
    "crop_type": "wheat",
    "growth_stage": "vegetative",
    "language": "en"
  }'
```

---

## Step 7: Test Marketplace

### 7.1 Navigate to Marketplace
- Click "Browse Products" on dashboard or go to http://localhost:3000/marketplace

### 7.2 Browse Products
- Should see product list (if products exist in backend)
- Can search by product name
- Can filter by product type

### 7.3 Test Search
- Type in search box: "tomato"
- Should filter products

### 7.4 Test Filter
- Select filter: "Vegetables"
- Should show only vegetable products

---

## Step 8: Test User Profile

### 8.1 Navigate to Profile
- Click user name in header or go to http://localhost:3000/profile

### 8.2 View Profile Information
- Should display:
  - Full name
  - Phone number
  - Language preference
  - Role
  - Member since date

### 8.3 Test Logout
- Click "Logout" button
- Should redirect to home page
- User data cleared from store

---

## Step 9: Test Login

### 9.1 Navigate to Login
- Go to http://localhost:3000/login

### 9.2 Enter Credentials
- Phone Number: +91 98765 43210 (from registration)
- Click "Login"

### 9.3 Verify Login
- Should redirect to dashboard
- User data restored

---

## Step 10: Test API Documentation

### 10.1 Open API Docs
- Go to http://localhost:8000/docs

### 10.2 Explore Endpoints
- Click on any endpoint to expand
- See request/response schemas
- Try "Try it out" button to test endpoints

### 10.3 Test Health Endpoint
- Find `/health` endpoint
- Click "Try it out"
- Click "Execute"
- Should see 200 response with agent list

---

## Step 11: Run Tests

### 11.1 Backend Tests
```bash
# In backend terminal (or new terminal)
cd backend
pytest tests/ -v
```

Expected output:
```
tests/test_agents.py::test_crop_advisory_agent PASSED
tests/test_api.py::test_create_user PASSED
...
85+ tests passing
```

### 11.2 Frontend Tests
```bash
# In frontend terminal (or new terminal)
cd frontend
npm test
```

---

## Step 12: Check Logs

### 12.1 Backend Logs
- Check terminal 1 for backend logs
- Should see API requests logged
- Look for any errors

### 12.2 Frontend Logs
- Open browser DevTools (F12)
- Check Console tab for errors
- Check Network tab for API calls

### 12.3 Check API Calls
- Open DevTools Network tab
- Perform actions (register, create farm, etc.)
- Should see API calls to http://localhost:8000/api/*

---

## Troubleshooting

### Backend Won't Start

**Error: "Port 8000 already in use"**
```bash
# Kill process on port 8000
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Error: "Module not found"**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**Error: "Database connection failed"**
- Check DATABASE_URL in .env
- Ensure database is running
- Or comment out database code if not needed

### Frontend Won't Start

**Error: "Port 3000 already in use"**
```bash
# Kill process on port 3000
# macOS/Linux
lsof -ti:3000 | xargs kill -9

# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

**Error: "Cannot find module"**
```bash
# Clear cache and reinstall
rm -rf node_modules .next
npm install
```

**Error: "API connection failed"**
- Check NEXT_PUBLIC_API_URL in .env.local
- Ensure backend is running on http://localhost:8000
- Check browser console for CORS errors

### API Connection Issues

**Error: "CORS error"**
- Backend CORS is configured in src/api/routes.py
- Ensure frontend URL is allowed
- Check browser console for specific error

**Error: "Cannot reach API"**
```bash
# Test backend is running
curl http://localhost:8000/health

# Check firewall
# Ensure ports 8000 and 3000 are open
```

---

## Complete Test Workflow

Here's a complete workflow to test everything:

1. **Register** → Create account
2. **Create Farm** → Add farm information
3. **Submit Query** → Get AI recommendation
4. **Browse Marketplace** → View products
5. **View Profile** → Check user info
6. **Logout** → Clear session
7. **Login** → Restore session
8. **Check API Docs** → Explore endpoints

---

## Performance Checks

### Frontend Performance
```bash
# In frontend terminal
npm run build
```

Expected:
- Build completes without errors
- Bundle size reasonable (~200KB gzipped)

### Backend Performance
```bash
# Test response time
time curl http://localhost:8000/health
```

Expected:
- Response time < 100ms

---

## Docker Alternative (Optional)

If you prefer Docker:

```bash
# From project root
docker-compose up

# This starts:
# - Backend on http://localhost:8000
# - Frontend on http://localhost:3000
```

---

## Summary Checklist

- ✅ Backend running on http://localhost:8000
- ✅ Frontend running on http://localhost:3000
- ✅ User registration working
- ✅ Farm creation working
- ✅ Query submission working
- ✅ Marketplace browsing working
- ✅ Profile viewing working
- ✅ Login/logout working
- ✅ API documentation accessible
- ✅ Tests passing
- ✅ No console errors
- ✅ API calls visible in Network tab

---

## Next Steps

Once everything is running:

1. **Explore the Code**
   - Backend: `backend/src/`
   - Frontend: `frontend/src/`

2. **Make Changes**
   - Backend auto-reloads with `--reload`
   - Frontend hot-reloads automatically

3. **Test Changes**
   - Run tests: `pytest tests/` or `npm test`
   - Check browser console for errors

4. **Deploy**
   - Follow INTEGRATION_GUIDE.md for production setup

---

## Support

If you encounter issues:

1. Check the Troubleshooting section above
2. Review INTEGRATION_GUIDE.md
3. Check QUICK_REFERENCE.md
4. Review error messages in logs

---

**You're all set! The application is now running end-to-end.** 🚀

For questions or issues, refer to the documentation files in the project root.
