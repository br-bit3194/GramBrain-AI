# GramBrain AI - Complete Testing Guide

## Overview
This guide walks you through testing the entire GramBrain AI system end-to-end, from backend API to frontend UI.

## Prerequisites
- Python 3.13+ installed
- Node.js 18+ installed
- Backend dependencies installed: `cd backend && pip install -r requirements.txt`
- Frontend dependencies installed: `cd frontend && npm install`

---

## Part 1: Backend Testing

### Step 1: Start the Backend Server

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Step 2: Test Health Check

Open your browser or use curl:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "success",
  "data": {
    "status": "healthy",
    "timestamp": "2026-03-02T...",
    "agents": [
      "crop_advisory",
      "farmer_interaction",
      "irrigation",
      "market",
      "marketplace",
      "pest_management",
      "soil",
      "sustainability",
      "village",
      "weather",
      "yield"
    ]
  }
}
```

### Step 3: Test User Creation (Valid Request)

```bash
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "+91 98765 43210",
    "name": "Rajesh Kumar",
    "language_preference": "hi",
    "role": "farmer"
  }'
```

Expected response:
```json
{
  "status": "success",
  "data": {
    "user": {
      "user_id": "...",
      "phone_number": "+91 98765 43210",
      "name": "Rajesh Kumar",
      "language_preference": "hi",
      "role": "farmer",
      "created_at": "...",
      "last_active": "..."
    }
  }
}
```

### Step 4: Test User Creation (Invalid Request - Missing Required Field)

```bash
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Rajesh Kumar"
  }'
```

Expected response (validation error):
```json
{
  "status": "error",
  "detail": "Validation error",
  "errors": [
    {
      "field": "phone_number",
      "message": "Field required"
    }
  ]
}
```

### Step 5: Test Farm Creation

```bash
curl -X POST http://localhost:8000/api/farms \
  -H "Content-Type: application/json" \
  -d '{
    "owner_id": "user-123",
    "latitude": 28.7041,
    "longitude": 77.1025,
    "area_hectares": 5.5,
    "soil_type": "loamy",
    "irrigation_type": "drip"
  }'
```

Expected response:
```json
{
  "status": "success",
  "data": {
    "farm": {
      "farm_id": "...",
      "owner_id": "user-123",
      "location": {"lat": 28.7041, "lon": 77.1025},
      "area_hectares": 5.5,
      "soil_type": "loamy",
      "irrigation_type": "drip",
      "crops": [],
      "created_at": "...",
      "updated_at": "..."
    }
  }
}
```

### Step 6: Test Query Processing

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "query_text": "How should I irrigate my wheat crop?",
    "farm_id": "farm-123",
    "crop_type": "wheat",
    "growth_stage": "vegetative",
    "soil_type": "loamy",
    "language": "en"
  }'
```

Expected response:
```json
{
  "status": "success",
  "data": {
    "recommendation": {
      "recommendation_id": "...",
      "query_id": "...",
      "user_id": "user-123",
      "farm_id": "farm-123",
      "timestamp": "...",
      "recommendation_text": "Based on your query about wheat, here are my recommendations...",
      "reasoning_chain": [...],
      "confidence": 0.85,
      "agent_contributions": [...],
      "language": "en"
    }
  }
}
```

### Step 7: Test Product Creation

```bash
curl -X POST http://localhost:8000/api/products \
  -H "Content-Type: application/json" \
  -d '{
    "farmer_id": "user-123",
    "farm_id": "farm-123",
    "product_type": "vegetables",
    "name": "Organic Tomatoes",
    "quantity_kg": 100,
    "price_per_kg": 50,
    "harvest_date": "2026-03-02"
  }'
```

Expected response:
```json
{
  "status": "success",
  "data": {
    "product": {
      "product_id": "...",
      "farmer_id": "user-123",
      "farm_id": "farm-123",
      "product_type": "vegetables",
      "name": "Organic Tomatoes",
      "quantity_kg": 100,
      "price_per_kg": 50,
      "harvest_date": "2026-03-02",
      "images": [],
      "pure_product_score": 0.95,
      "status": "available",
      "created_at": "..."
    }
  }
}
```

### Step 8: Test Knowledge Addition

```bash
curl -X POST http://localhost:8000/api/knowledge \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_id": "chunk-001",
    "content": "Wheat requires 450-650mm of water during its growing season",
    "source": "Agricultural Research Institute",
    "topic": "irrigation",
    "crop_type": "wheat",
    "region": "North India"
  }'
```

Expected response:
```json
{
  "status": "success",
  "data": {
    "message": "Knowledge added successfully"
  }
}
```

---

## Part 2: Frontend Testing

### Step 1: Start the Frontend Server

In a new terminal:

```bash
cd frontend
npm run dev
```

Expected output:
```
> next dev
  ▲ Next.js 13.x.x
  - Local:        http://localhost:3000
```

### Step 2: Test Home Page

Navigate to: `http://localhost:3000`

Expected:
- Page loads without errors
- Header displays "GramBrain AI"
- Hero section visible with features
- No console errors

### Step 3: Test User Registration

1. Click "Register" in the header
2. Navigate to: `http://localhost:3000/register`
3. Fill in the form:
   - Phone: +91 98765 43210
   - Name: Rajesh Kumar
   - Language: English
   - Role: Farmer
4. Click "Register"

Expected:
- Form submits successfully
- No "Objects are not valid as a React child" error
- Success message appears
- User is created in backend

### Step 4: Test Login

1. Click "Login" in the header
2. Navigate to: `http://localhost:3000/login`
3. Enter a user ID
4. Click "Login"

Expected:
- Login succeeds
- Redirected to dashboard

### Step 5: Test Dashboard

Navigate to: `http://localhost:3000/dashboard`

Expected:
- Dashboard page loads
- Shows user information
- No errors in console

### Step 6: Test Farms Page

Navigate to: `http://localhost:3000/farms`

Expected:
- Farms page loads
- Can view farm list
- Can create new farm

### Step 7: Test Query Interface

Navigate to: `http://localhost:3000/query`

Expected:
- Query form loads
- Can submit a query
- Receives recommendation response

### Step 8: Test Marketplace

Navigate to: `http://localhost:3000/marketplace`

Expected:
- Marketplace page loads
- Can view products
- Can search/filter products

---

## Part 3: Integration Testing

### Test 1: Complete User Registration Flow

1. Start backend: `cd backend && python -m uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000`
2. Start frontend: `cd frontend && npm run dev`
3. Navigate to `http://localhost:3000/register`
4. Fill in registration form
5. Submit

Expected:
- No CORS errors
- No validation errors
- User created successfully
- Success message displayed

### Test 2: Complete Query Flow

1. Register a user
2. Create a farm
3. Navigate to query page
4. Submit a query
5. Receive recommendation

Expected:
- All steps complete without errors
- Proper JSON responses from backend
- Frontend displays recommendation

### Test 3: Error Handling

1. Try to create user with missing phone number
2. Try to create farm with invalid coordinates
3. Try to submit query with missing required fields

Expected:
- Backend returns proper error JSON
- Frontend displays error message
- No validation error objects in UI

---

## Troubleshooting

### Backend Issues

**Error: "Address already in use"**
```bash
# Kill the process using port 8000
lsof -ti:8000 | xargs kill -9
```

**Error: "No module named uvicorn"**
```bash
cd backend
pip install -r requirements.txt
```

**Error: "Validation error" responses**
- Check that all required fields are provided
- Verify field types match the schema

### Frontend Issues

**Error: "Objects are not valid as a React child"**
- This should be fixed now with proper error handling
- Clear browser cache: Ctrl+Shift+Delete (or Cmd+Shift+Delete on Mac)
- Restart frontend: `npm run dev`

**Error: "CORS policy blocked"**
- Ensure backend is running on port 8000
- Verify `frontend/.env.local` has `NEXT_PUBLIC_API_URL=http://localhost:8000/api`
- Restart both backend and frontend

**Error: "Cannot find module"**
```bash
cd frontend
npm install
npm run dev
```

---

## Success Criteria

✅ Backend health check returns 200 OK
✅ User creation works with valid data
✅ Validation errors return proper JSON (not objects)
✅ CORS allows requests from frontend
✅ Frontend pages load without errors
✅ User registration completes end-to-end
✅ Query processing returns recommendations
✅ All API endpoints return proper JSON responses

---

## Next Steps

Once all tests pass:
1. Deploy backend to production server
2. Deploy frontend to production server
3. Update environment variables for production
4. Set up monitoring and logging
5. Create user documentation

