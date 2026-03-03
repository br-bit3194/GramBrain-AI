# Authentication Integration Guide

Complete guide for the authentication system connecting frontend and backend.

## Overview

The authentication system uses JWT (JSON Web Tokens) for secure user authentication between the Next.js frontend and FastAPI backend.

## Architecture

### Backend (FastAPI)
- **JWT Token Generation**: Uses HS256 algorithm with bcrypt for password hashing
- **Token Types**: Access tokens (24h) and refresh tokens (7 days)
- **Middleware**: Bearer token authentication with role-based access control (RBAC)
- **Endpoints**: `/api/auth/register`, `/api/auth/login`, `/api/auth/me`

### Frontend (Next.js + Zustand)
- **State Management**: Zustand store for user, tokens, and auth state
- **Persistence**: LocalStorage for session persistence across page reloads
- **API Client**: Axios with automatic token injection and 401 handling
- **Hooks**: `useAuth` hook for authentication operations

## Setup

### 1. Backend Configuration

Add to `.env`:
```env
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production-min-32-chars
JWT_ACCESS_TOKEN_EXPIRE_HOURS=24
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
```

### 2. Frontend Configuration

Already configured in `frontend/.env`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## Usage

### Frontend - Using the Auth Hook

```typescript
import { useAuth } from '@/hooks/useAuth'

function MyComponent() {
  const { 
    user, 
    isAuthenticated, 
    login, 
    register, 
    logout,
    isLoading,
    error 
  } = useAuth()

  // Register
  const handleRegister = async () => {
    const result = await register({
      phone_number: '+91 98765 43210',
      name: 'John Doe',
      password: 'securepassword',
      role: 'farmer'
    })
    
    if (result.success) {
      console.log('Registered:', result.user)
    }
  }

  // Login
  const handleLogin = async () => {
    const result = await login({
      phone_number: '+91 98765 43210',
      password: 'securepassword'
    })
    
    if (result.success) {
      console.log('Logged in:', result.user)
    }
  }

  // Logout
  const handleLogout = () => {
    logout()
  }

  return (
    <div>
      {isAuthenticated ? (
        <div>
          <p>Welcome, {user?.name}!</p>
          <button onClick={handleLogout}>Logout</button>
        </div>
      ) : (
        <div>
          <button onClick={handleLogin}>Login</button>
          <button onClick={handleRegister}>Register</button>
        </div>
      )}
    </div>
  )
}
```

### Using the AuthForm Component

```typescript
import { AuthForm } from '@/components/AuthForm'

function LoginPage() {
  return (
    <div>
      <h1>Welcome to GramBrain AI</h1>
      <AuthForm />
    </div>
  )
}
```

### Wrapping Your App with AuthProvider

```typescript
// app/layout.tsx
import { AuthProvider } from '@/components/AuthProvider'

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  )
}
```

## API Endpoints

### POST /api/auth/register
Register a new user.

**Request:**
```json
{
  "phone_number": "+91 98765 43210",
  "name": "John Doe",
  "password": "securepassword",
  "language_preference": "en",
  "role": "farmer"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "user": {
      "user_id": "uuid",
      "phone_number": "+91 98765 43210",
      "name": "John Doe",
      "role": "farmer"
    },
    "access_token": "eyJ...",
    "refresh_token": "eyJ...",
    "token_type": "bearer"
  }
}
```

### POST /api/auth/login
Login with credentials.

**Request:**
```json
{
  "phone_number": "+91 98765 43210",
  "password": "securepassword"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "user": {
      "user_id": "uuid",
      "phone_number": "+91 98765 43210",
      "role": "farmer"
    },
    "access_token": "eyJ...",
    "refresh_token": "eyJ...",
    "token_type": "bearer"
  }
}
```

### GET /api/auth/me
Get current authenticated user.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "user": {
      "user_id": "uuid",
      "phone_number": "+91 98765 43210",
      "role": "farmer"
    }
  }
}
```

## Security Features

1. **Password Hashing**: bcrypt with cost factor 12
2. **JWT Tokens**: HS256 algorithm with expiration
3. **CORS Protection**: Configured for localhost:3000
4. **Bearer Authentication**: Standard HTTP Authorization header
5. **Role-Based Access Control**: Permissions based on user roles
6. **Automatic Token Injection**: API client adds token to all requests
7. **401 Handling**: Automatic logout on unauthorized responses

## Testing

Run the authentication test script:

```bash
# Make sure backend is running on port 8000
node test-auth-flow.js
```

This will test:
- User registration
- User login
- Token validation
- Protected endpoints
- Error handling
- CORS configuration

## Token Flow

1. **Registration/Login**: User provides credentials
2. **Token Generation**: Backend creates access + refresh tokens
3. **Token Storage**: Frontend stores in Zustand + localStorage
4. **Token Injection**: API client adds token to all requests
5. **Token Validation**: Backend middleware validates on protected routes
6. **Token Refresh**: (TODO) Use refresh token to get new access token
7. **Logout**: Clear tokens from store and localStorage

## Role-Based Access Control (RBAC)

### Roles
- `farmer`: Farm owners, can manage their farms and products
- `village_leader`: Community leaders, can view all farms in area
- `policymaker`: Government officials, analytics access
- `consumer`: Product buyers
- `admin`: Full system access

### Permissions
See `backend/src/auth/rbac.py` for complete permission matrix.

### Using RBAC in Routes

```python
from fastapi import Depends
from backend.src.auth import require_permission, require_role, Permission

@app.get("/api/farms")
async def list_farms(
    current_user: dict = Depends(require_permission(Permission.VIEW_ALL_FARMS))
):
    # Only users with VIEW_ALL_FARMS permission can access
    pass

@app.post("/api/products")
async def create_product(
    current_user: dict = Depends(require_role("farmer", "admin"))
):
    # Only farmers and admins can create products
    pass
```

## Next Steps

1. **Database Integration**: Store users and hashed passwords in DynamoDB
2. **Token Refresh**: Implement refresh token endpoint
3. **Password Reset**: Add forgot password flow
4. **Email/SMS Verification**: Verify phone numbers
5. **Social Login**: Add OAuth providers
6. **Session Management**: Track active sessions
7. **Rate Limiting**: Prevent brute force attacks

## Troubleshooting

### Token not being sent
- Check that `AuthProvider` wraps your app
- Verify token is in Zustand store: `useAppStore.getState().accessToken`
- Check browser console for API request logs

### 401 Unauthorized errors
- Verify JWT_SECRET_KEY matches between requests
- Check token expiration
- Ensure Authorization header format: `Bearer <token>`

### CORS errors
- Verify frontend URL in backend CORS middleware
- Check that API_URL in frontend .env is correct
- Ensure backend is running on correct port

## Files Modified/Created

### Backend
- `.env` - Added JWT configuration
- `backend/src/auth/auth_service.py` - JWT and password handling
- `backend/src/auth/middleware.py` - Authentication middleware
- `backend/src/auth/rbac.py` - Role-based access control
- `backend/src/api/routes.py` - Auth endpoints

### Frontend
- `frontend/src/services/api.ts` - Token injection and 401 handling
- `frontend/src/hooks/useAuth.ts` - Authentication hook
- `frontend/src/components/AuthProvider.tsx` - Session restoration
- `frontend/src/components/AuthForm.tsx` - Login/register UI
- `frontend/src/store/appStore.ts` - Auth state management

### Testing
- `test-auth-flow.js` - End-to-end auth testing
