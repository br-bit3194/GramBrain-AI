# Quick Authentication & Rate Limiting Setup

## ✅ What's Been Implemented

### 1. Authentication System
- **JWT-based authentication** with bcrypt password hashing (cost factor 12)
- **Access tokens** (24-hour expiration)
- **Refresh tokens** (7-day expiration)
- **Password hashing** with bcrypt salt

### 2. Role-Based Access Control (RBAC)
- **5 Roles**: farmer, village_leader, policymaker, consumer, admin
- **Permission system** with granular access control
- **Middleware** for protecting routes

### 3. Redis Caching & Rate Limiting
- **Redis client** for caching and rate limiting
- **Rate limits per role**:
  - Farmer: 100 req/min
  - Village Leader: 200 req/min
  - Policymaker: 150 req/min
  - Consumer: 100 req/min
  - Admin: 1000 req/min
  - Anonymous: 20 req/min
- **429 responses** with Retry-After headers

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install bcrypt pyjwt redis
```

### 2. Environment Variables
Add to `.env`:
```env
JWT_SECRET_KEY=your-super-secret-key-change-this
REDIS_ENABLED=false  # Set to true when Redis is available
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. API Endpoints

#### Register
```bash
POST /api/auth/register
{
  "phone_number": "+91 98765 43210",
  "name": "Test Farmer",
  "password": "securepassword123",
  "role": "farmer"
}
```

#### Login
```bash
POST /api/auth/login
{
  "phone_number": "+91 98765 43210",
  "password": "securepassword123"
}
```

#### Get Current User
```bash
GET /api/auth/me
Headers: Authorization: Bearer <access_token>
```

### 4. Protected Routes Example

To protect a route:
```python
from ..auth import get_current_user, require_permission, Permission

@app.get("/api/protected")
async def protected_route(current_user: dict = Depends(get_current_user)):
    return {"user": current_user}

@app.get("/api/admin-only")
async def admin_route(current_user: dict = Depends(require_role("admin"))):
    return {"message": "Admin access"}
```

## 📁 Files Created

```
backend/src/
├── auth/
│   ├── __init__.py
│   ├── auth_service.py      # JWT & password hashing
│   ├── rbac.py               # Role-based access control
│   └── middleware.py         # FastAPI auth middleware
└── cache/
    ├── __init__.py
    ├── redis_client.py       # Redis cache client
    └── rate_limiter.py       # Rate limiting middleware
```

## 🔐 Security Features

- ✅ Bcrypt password hashing (cost factor 12)
- ✅ JWT tokens with expiration
- ✅ Role-based permissions
- ✅ Rate limiting per user/role
- ✅ 403 Forbidden for unauthorized access
- ✅ 429 Too Many Requests for rate limits

## 🎯 Next Steps for Frontend

1. **Create login/register pages**
2. **Store JWT token** in localStorage/sessionStorage
3. **Add Authorization header** to all API calls
4. **Handle 401/403 errors** (redirect to login)
5. **Show rate limit info** in UI

## Frontend Integration Example

```javascript
// Login
const response = await fetch('http://localhost:8000/api/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    phone_number: '+91 98765 43210',
    password: 'password123'
  })
});
const data = await response.json();
localStorage.setItem('access_token', data.data.access_token);

// Protected API call
const queryResponse = await fetch('http://localhost:8000/api/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${localStorage.getItem('access_token')}`
  },
  body: JSON.stringify({ query_text: 'What should I plant?' })
});
```

## ⚡ Performance Notes

- Redis is **optional** - system works without it (in-memory fallback)
- Rate limiting is **graceful** - doesn't break if Redis is down
- Tokens are **stateless** - no database lookup needed

## 🔧 Testing

```bash
# Test authentication
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"phone_number":"+91 98765 43210","name":"Test","password":"test123","role":"farmer"}'

# Test protected endpoint
curl http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer <your_token>"
```

---

**Status**: ✅ Ready for integration
**Time to implement**: ~15 minutes
**Next**: Focus on frontend integration and UI polish
