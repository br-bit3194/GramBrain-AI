# Quick Reference Card - POC Submission

## 🚀 Start Commands

```bash
# Backend
cd backend && python main.py

# Frontend
cd frontend && npm run dev
```

## 🔐 Test Credentials

```json
{
  "phone_number": "+91 98765 43210",
  "password": "test123",
  "role": "farmer"
}
```

## 📡 Key API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/auth/register` | POST | Register user |
| `/api/auth/login` | POST | Login user |
| `/api/auth/me` | GET | Get current user |
| `/api/query` | POST | AI query processing |
| `/api/farms` | POST/GET | Farm management |
| `/api/products` | POST/GET | Marketplace |
| `/api/knowledge` | POST | Add knowledge |
| `/api/knowledge/search` | GET | Search knowledge |

## 🎨 Frontend Tasks (Priority Order)

### 1. Auth (30 min) ⚡
- [ ] Login page
- [ ] Register page
- [ ] Auth context
- [ ] Protected routes

### 2. Pages (2 hours) 🎯
- [ ] Home - Hero + features
- [ ] Dashboard - Stats + quick actions
- [ ] Query - Chat interface
- [ ] Marketplace - Product grid
- [ ] Farms - Management UI

### 3. Polish (1 hour) ✨
- [ ] Loading states
- [ ] Error handling
- [ ] Animations
- [ ] Mobile responsive
- [ ] Professional design

## 💻 Copy-Paste Code

### Auth Context
```typescript
// src/context/AuthContext.tsx
import { createContext, useContext, useState } from 'react';

const AuthContext = createContext<any>(null);

export function AuthProvider({ children }: any) {
  const [token, setToken] = useState(localStorage.getItem('access_token'));
  
  const login = async (phone: string, password: string) => {
    const res = await fetch('http://localhost:8000/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phone_number: phone, password }),
    });
    const data = await res.json();
    if (data.status === 'success') {
      setToken(data.data.access_token);
      localStorage.setItem('access_token', data.data.access_token);
    }
  };
  
  const logout = () => {
    setToken(null);
    localStorage.removeItem('access_token');
  };
  
  return (
    <AuthContext.Provider value={{ token, login, logout, isAuthenticated: !!token }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
```

### API Helper
```typescript
// src/lib/api.ts
const API_BASE = 'http://localhost:8000';

export async function apiCall(endpoint: string, options: any = {}) {
  const token = localStorage.getItem('access_token');
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` }),
      ...options.headers,
    },
  });
  
  if (response.status === 401) {
    localStorage.removeItem('access_token');
    window.location.href = '/login';
  }
  
  return response.json();
}
```

### Login Page
```typescript
// src/pages/Login.tsx
import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const [phone, setPhone] = useState('');
  const [password, setPassword] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();
  
  const handleSubmit = async (e: any) => {
    e.preventDefault();
    await login(phone, password);
    navigate('/dashboard');
  };
  
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full p-8 bg-white rounded-lg shadow">
        <h2 className="text-3xl font-bold text-center mb-6">Login</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="tel"
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
            placeholder="Phone Number"
            className="w-full px-4 py-2 border rounded"
            required
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            className="w-full px-4 py-2 border rounded"
            required
          />
          <button
            type="submit"
            className="w-full py-2 bg-green-600 text-white rounded hover:bg-green-700"
          >
            Login
          </button>
        </form>
      </div>
    </div>
  );
}
```

## 🎨 Design System

### Colors
```css
--primary: #10b981 (green)
--secondary: #059669 (dark green)
--accent: #fbbf24 (amber)
--background: #f9fafb (light gray)
--text: #111827 (dark gray)
```

### Components
- Loading: `<div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600" />`
- Button: `className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700"`
- Card: `className="bg-white p-6 rounded-lg shadow"`

## ✅ Checklist

### Backend ✅
- [x] Authentication
- [x] Rate limiting
- [x] OpenSearch
- [x] All APIs
- [x] Tests passing

### Frontend 🎯
- [ ] Auth UI
- [ ] Page polish
- [ ] Loading states
- [ ] Error handling
- [ ] Mobile responsive

## 📊 Success Metrics

- ✅ Users can register/login
- ✅ All pages look professional
- ✅ AI queries work
- ✅ No broken functionality
- ✅ Mobile responsive
- ✅ Ready for demo

## ⏰ Time Budget

- Auth UI: 30 min
- Page improvements: 2 hours
- Polish: 1 hour
- Testing: 30 min
- **Total**: 4 hours

**Current time**: Check clock
**Deadline**: 11:59 PM
**Status**: On track! 🚀

## 🆘 Quick Fixes

### CORS Error
Already configured in backend for `localhost:3000`

### 401 Unauthorized
Check token in localStorage, redirect to login

### 429 Rate Limit
Show message: "Too many requests, try again in X seconds"

### Loading Forever
Add timeout, show error after 30 seconds

## 📞 Support

- Backend docs: `BACKEND_COMPLETE_SUMMARY.md`
- Frontend guide: `FRONTEND_INTEGRATION_GUIDE.md`
- Auth setup: `QUICK_SETUP_AUTH.md`
- OpenSearch: `OPENSEARCH_QUICK_SETUP.md`

---

**Backend**: ✅ Complete
**Frontend**: 🎨 In Progress
**Deadline**: Tonight 11:59 PM
**You got this!** 💪
