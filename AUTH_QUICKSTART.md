# Authentication Quick Start Guide

Get authentication working between frontend and backend in 3 steps.

## Prerequisites

- Python 3.8+ installed
- Node.js 16+ installed
- AWS credentials configured in `.env`
- Backend running on port 8000
- Frontend running on port 3000

## Step 1: Install Dependencies

### Backend
```bash
cd backend
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

## Step 2: Setup DynamoDB Tables

Run the setup script to create the necessary tables:

```bash
python setup-auth-db.py
```

This will create:
- `grambrain-users-dev` - User accounts with passwords
- `grambrain-farms-dev` - Farm data
- `grambrain-recommendations-dev` - Query history
- `grambrain-products-dev` - Marketplace products
- `grambrain-knowledge-dev` - Knowledge base

## Step 3: Start Servers

### Start Backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Start Frontend (in another terminal)
```bash
cd frontend
npm run dev
```

## Step 4: Test Authentication

### Option 1: Use the Demo Page
1. Open http://localhost:3000/auth-demo
2. Click "Register" and create an account
3. You'll be automatically logged in
4. Your session persists across page reloads

### Option 2: Use the Test Script
```bash
node test-auth-flow.js
```

## How It Works

### Registration Flow
1. User enters phone, name, password, and role
2. Frontend sends POST to `/api/auth/register`
3. Backend checks if phone number exists in DynamoDB
4. If new, password is hashed with bcrypt
5. User is saved to DynamoDB with hashed password
6. JWT tokens are generated and returned
7. Frontend stores tokens in Zustand + localStorage

### Login Flow
1. User enters phone and password
2. Frontend sends POST to `/api/auth/login`
3. Backend looks up user by phone in DynamoDB
4. Password is verified against stored hash
5. If valid, JWT tokens are generated
6. Frontend stores tokens and user data

### Protected Requests
1. Frontend automatically adds `Authorization: Bearer <token>` header
2. Backend middleware validates JWT token
3. User info is extracted and passed to route handler
4. If token invalid/expired, 401 is returned
5. Frontend automatically logs out on 401

## Environment Variables

### Backend (.env)
```env
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# DynamoDB
DYNAMODB_ENV=dev

# Authentication
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production-min-32-chars
JWT_ACCESS_TOKEN_EXPIRE_HOURS=24
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
```

### Frontend (frontend/.env)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## Common Issues

### "Phone number already registered"
- The phone number is already in DynamoDB
- Try a different phone number or login instead

### "Invalid phone number or password"
- Check that you're using the correct credentials
- Password is case-sensitive

### "401 Unauthorized"
- Token may be expired (24h default)
- Try logging in again
- Check that JWT_SECRET_KEY is the same in .env

### "Cannot connect to backend"
- Verify backend is running on port 8000
- Check NEXT_PUBLIC_API_URL in frontend/.env
- Check CORS settings in backend/src/api/routes.py

### "DynamoDB table not found"
- Run `python setup-auth-db.py` to create tables
- Check AWS credentials in .env
- Verify DYNAMODB_ENV matches table names

## API Endpoints

### POST /api/auth/register
Register a new user
```json
{
  "phone_number": "+91 98765 43210",
  "name": "John Doe",
  "password": "securepassword",
  "role": "farmer"
}
```

### POST /api/auth/login
Login with credentials
```json
{
  "phone_number": "+91 98765 43210",
  "password": "securepassword"
}
```

### GET /api/auth/me
Get current user (requires auth token)
```
Authorization: Bearer <access_token>
```

## Using Authentication in Your Code

### Frontend - Check if logged in
```typescript
import { useAuth } from '@/hooks/useAuth'

function MyComponent() {
  const { isAuthenticated, user } = useAuth()
  
  if (!isAuthenticated) {
    return <div>Please login</div>
  }
  
  return <div>Welcome {user?.name}!</div>
}
```

### Frontend - Login/Register
```typescript
import { useAuth } from '@/hooks/useAuth'

function LoginForm() {
  const { login, register } = useAuth()
  
  const handleLogin = async () => {
    const result = await login({
      phone_number: '+91 98765 43210',
      password: 'mypassword'
    })
    
    if (result.success) {
      console.log('Logged in!')
    }
  }
  
  return <button onClick={handleLogin}>Login</button>
}
```

### Frontend - Protected Route
```typescript
import { ProtectedRoute } from '@/components/ProtectedRoute'

export default function FarmerDashboard() {
  return (
    <ProtectedRoute requiredRole="farmer">
      <div>Farmer Dashboard Content</div>
    </ProtectedRoute>
  )
}
```

### Backend - Protect a Route
```python
from fastapi import Depends
from backend.src.auth import get_current_user

@app.get("/api/my-farms")
async def get_my_farms(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    # ... fetch farms for user
```

### Backend - Require Specific Role
```python
from fastapi import Depends
from backend.src.auth import require_role

@app.post("/api/admin/users")
async def create_user(current_user: dict = Depends(require_role("admin"))):
    # Only admins can access this
    pass
```

## Security Notes

1. **Never commit JWT_SECRET_KEY** - Use a strong random key in production
2. **Use HTTPS in production** - Tokens should never be sent over HTTP
3. **Passwords are hashed** - bcrypt with cost factor 12
4. **Tokens expire** - Access tokens expire after 24h by default
5. **Phone numbers are unique** - One account per phone number

## Next Steps

- Implement password reset flow
- Add phone number verification via SMS
- Implement refresh token rotation
- Add rate limiting on auth endpoints
- Set up session management dashboard

## Support

For more details, see:
- `AUTH_INTEGRATION_GUIDE.md` - Complete integration guide
- `AUTH_CONNECTION_SUMMARY.md` - Architecture overview
- `test-auth-flow.js` - Test script with examples
