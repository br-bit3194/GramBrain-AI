# API Client Configuration

## Changes Made

### 1. Base URL Configuration
- **Updated**: API client now uses `NEXT_PUBLIC_API_URL` environment variable with fallback to `http://localhost:8000/api`
- **File**: `frontend/src/services/api.ts`
- **Environment Files**: Updated `.env`, `.env.example`, and `.env.local` to use correct base URL

### 2. Axios Instance Configuration
The axios instance is now configured with:
- **Base URL**: Includes `/api` prefix (from environment variable)
- **Headers**: `Content-Type: application/json`
- **Timeout**: 30 seconds (30000ms)

### 3. Request Interceptor
Added request interceptor that:
- Logs all outgoing requests (method, URL, params, data)
- Helps with debugging API calls
- Returns the config to continue the request

### 4. Response Interceptor
Added response interceptor that:
- Logs successful responses (status, data)
- Logs error responses with details
- Handles network errors separately
- Returns appropriate error for error handling

### 5. Authentication Endpoints
Added new authentication methods:
- `register(data)` - POST `/auth/register`
- `login(data)` - POST `/auth/login`
- `getCurrentUser()` - GET `/auth/me`

## API Endpoint Structure

All endpoints are relative to the base URL (`http://localhost:8000/api`):

### Authentication
- POST `/auth/register` - Register new user
- POST `/auth/login` - Login user
- GET `/auth/me` - Get current user info

### Users
- POST `/users` - Create user
- GET `/users/{user_id}` - Get user by ID

### Farms
- POST `/farms` - Create farm
- GET `/farms/{farm_id}` - Get farm by ID
- GET `/users/{user_id}/farms` - List user's farms

### Query/Recommendations
- POST `/query` - Process query and get recommendation
- GET `/recommendations/{recommendation_id}` - Get recommendation by ID
- GET `/users/{user_id}/recommendations` - List user's recommendations

### Products/Marketplace
- POST `/products` - Create product listing
- GET `/products/{product_id}` - Get product by ID
- GET `/products` - Search products (with filters)
- GET `/farmers/{farmer_id}/products` - List farmer's products

### Knowledge/RAG
- POST `/knowledge` - Add knowledge chunk
- GET `/knowledge/search` - Search knowledge base

### Health
- GET `/health` - Health check

## Environment Variables

### `.env` and `.env.local`
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

### Production
For production, update to:
```
NEXT_PUBLIC_API_URL=https://your-domain.com/api
```

## Verification

### Manual Testing
1. Start the backend server: `cd backend && python main.py`
2. Start the frontend: `cd frontend && npm run dev`
3. Open browser console and check for API request logs
4. Make any API call and verify:
   - Request URL includes `/api` prefix
   - Request headers include `Content-Type: application/json`
   - Logs appear in console

### Automated Testing
Run the test suite:
```bash
cd frontend
npm test
```

## Console Logging

With the interceptors in place, you'll see logs like:

**Request:**
```
[API Request] POST /auth/login {
  params: undefined,
  data: { phone_number: "+91...", password: "..." }
}
```

**Success Response:**
```
[API Response] POST /auth/login {
  status: 200,
  data: { status: "success", data: { ... } }
}
```

**Error Response:**
```
[API Error Response] POST /auth/login {
  status: 401,
  data: { status: "error", detail: "Invalid credentials" }
}
```

## Next Steps

The API client is now properly configured. Next tasks:
1. Implement authentication header injection (Task 2)
2. Implement centralized error handling (Task 3)
3. Update all components to use the API client
