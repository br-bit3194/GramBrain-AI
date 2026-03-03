# How to Test Frontend-Backend Connection

## Prerequisites

1. **Backend Requirements:**
   - Python 3.8+ installed
   - Virtual environment activated
   - Dependencies installed: `pip install -r backend/requirements.txt`
   - AWS credentials configured in `.env`

2. **Frontend Requirements:**
   - Node.js 18+ installed
   - Dependencies installed: `npm install` (in frontend directory)

3. **Test Script Requirements:**
   - Node.js installed (for running test-connection.js)
   - axios package: `npm install axios` (in root directory)

---

## Quick Start (Windows)

### Option 1: Automated Start (Recommended)

1. **Double-click `start-servers.bat`**
   - This will open two command windows
   - Backend will start on http://localhost:8000
   - Frontend will start on http://localhost:3000

2. **Wait for both servers to start** (about 10-30 seconds)

3. **Run the connection test:**
   ```bash
   node test-connection.js
   ```

### Option 2: Manual Start

1. **Start Backend (Terminal 1):**
   ```bash
   cd backend
   python main.py
   ```
   Wait for: `Uvicorn running on http://0.0.0.0:8000`

2. **Start Frontend (Terminal 2):**
   ```bash
   cd frontend
   npm run dev
   ```
   Wait for: `Ready on http://localhost:3000`

3. **Run Connection Test (Terminal 3):**
   ```bash
   node test-connection.js
   ```

---

## What the Test Does

The `test-connection.js` script runs 12 comprehensive tests:

1. ✓ **Backend Server Running** - Verifies backend is accessible
2. ✓ **CORS Configuration** - Tests cross-origin requests
3. ✓ **User Registration** - Creates a new user account
4. ✓ **User Login** - Tests authentication flow
5. ✓ **Protected Route with Token** - Verifies JWT token works
6. ✓ **Create Farm** - Tests farm creation endpoint
7. ✓ **Process AI Query** - Tests multi-agent query processing
8. ✓ **Search Knowledge** - Tests RAG knowledge search
9. ✓ **Error Handling (401)** - Tests unauthorized access
10. ✓ **Error Handling (422)** - Tests validation errors
11. ✓ **List User Farms** - Tests farm listing
12. ✓ **Create Product** - Tests marketplace product creation

---

## Expected Output

```
╔════════════════════════════════════════════════════════════╗
║     GramBrain AI - Frontend-Backend Connection Test       ║
╚════════════════════════════════════════════════════════════╝

Starting connection tests...

Test 1: Backend Server Running...
ℹ   Backend is running on http://localhost:8000
ℹ   Status: healthy
ℹ   Agents: 11 registered
✓ Test 1: Backend Server Running - PASSED

Test 2: CORS Configuration...
ℹ   CORS is properly configured for localhost:3000
✓ Test 2: CORS Configuration - PASSED

[... more tests ...]

╔════════════════════════════════════════════════════════════╗
║                      TEST SUMMARY                          ║
╚════════════════════════════════════════════════════════════╝

Total Tests: 12
Passed: 12

🎉 All tests passed! Frontend-Backend connection is working perfectly!
```

---

## Troubleshooting

### Backend Not Starting

**Error:** `ModuleNotFoundError: No module named 'fastapi'`
```bash
cd backend
pip install -r requirements.txt
```

**Error:** `AWS credentials not found`
- Check `.env` file has AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
- Or set environment variables

### Frontend Not Starting

**Error:** `Cannot find module 'next'`
```bash
cd frontend
npm install
```

**Error:** `Port 3000 already in use`
```bash
# Kill the process using port 3000
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

### Connection Test Fails

**Error:** `ECONNREFUSED`
- Make sure backend is running on port 8000
- Check `http://localhost:8000/health` in browser

**Error:** `CORS error`
- Verify backend CORS settings in `backend/src/api/routes.py`
- Should include `http://localhost:3000`

**Error:** `401 Unauthorized on protected routes`
- This is expected for Test 9
- If other tests fail with 401, check token generation

---

## Manual Testing in Browser

1. **Open Frontend:** http://localhost:3000

2. **Register a User:**
   - Click "Get Started Free"
   - Fill in phone number, name, password
   - Submit

3. **Login:**
   - Use the credentials you just created
   - Should redirect to dashboard

4. **Create a Farm:**
   - Navigate to farm creation
   - Fill in farm details
   - Submit

5. **Ask a Question:**
   - Go to query/chat interface
   - Type: "Should I irrigate my wheat field today?"
   - Submit and wait for AI response

---

## API Testing with cURL

### Health Check
```bash
curl http://localhost:8000/health
```

### Register User
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d "{\"phone_number\":\"+919876543210\",\"name\":\"Test User\",\"password\":\"test123\"}"
```

### Login
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"phone_number\":\"+919876543210\",\"password\":\"test123\"}"
```

### Protected Route (replace TOKEN)
```bash
curl http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

---

## Verifying Specific Features

### 1. Authentication Flow
```bash
# Register
node -e "
const axios = require('axios');
axios.post('http://localhost:8000/api/auth/register', {
  phone_number: '+919999999999',
  name: 'Test',
  password: 'test123'
}).then(r => console.log('Token:', r.data.data.access_token))
"
```

### 2. AI Query Processing
```bash
# Process query (replace TOKEN and USER_ID)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TOKEN" \
  -d "{\"user_id\":\"USER_ID\",\"query_text\":\"What crops should I plant?\"}"
```

### 3. Knowledge Search
```bash
# Search knowledge (replace TOKEN)
curl "http://localhost:8000/api/knowledge/search?query=wheat&top_k=5" \
  -H "Authorization: Bearer TOKEN"
```

---

## Performance Benchmarks

Expected response times (local development):

- Health check: < 50ms
- User registration: < 200ms
- User login: < 200ms
- Create farm: < 300ms
- AI query processing: 1-5 seconds (depends on agents)
- Knowledge search: < 500ms

---

## Next Steps After Successful Test

1. ✅ Connection verified
2. ✅ Authentication working
3. ✅ AI agents responding
4. ✅ Error handling functional

**Now you can:**
- Build frontend UI components
- Implement user workflows
- Add more AI agent capabilities
- Deploy to production

---

## Production Deployment Checklist

Before deploying to production:

- [ ] Update CORS origins to production domain
- [ ] Change `NEXT_PUBLIC_API_URL` to production URL
- [ ] Enable HTTPS
- [ ] Set up proper secret management
- [ ] Configure production database
- [ ] Enable monitoring and logging
- [ ] Set up CI/CD pipeline
- [ ] Configure rate limiting
- [ ] Enable request validation
- [ ] Set up backup and recovery

---

## Support

If tests fail or you encounter issues:

1. Check the detailed analysis: `FRONTEND_BACKEND_CONNECTION_ANALYSIS.md`
2. Review backend logs in terminal
3. Check browser console for frontend errors
4. Verify environment variables in `.env` files
5. Ensure all dependencies are installed

---

## Summary

The frontend and backend are properly connected with:
- ✅ JWT authentication
- ✅ CORS configured
- ✅ All API endpoints mapped
- ✅ Error handling implemented
- ✅ Multi-agent AI system integrated
- ✅ Type-safe data models

Run `node test-connection.js` to verify everything is working!
