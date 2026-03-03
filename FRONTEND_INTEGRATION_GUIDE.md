# Frontend Integration Guide - Quick Reference

## 🚀 Backend is Ready!

**Backend URL**: `http://localhost:8000`
**API Docs**: `http://localhost:8000/docs` (Swagger UI)

## 📡 Available API Endpoints

### Authentication
```javascript
// Register
POST /api/auth/register
Body: { phone_number, name, password, role }

// Login
POST /api/auth/login
Body: { phone_number, password }

// Get current user
GET /api/auth/me
Headers: { Authorization: "Bearer <token>" }
```

### Farms
```javascript
// Create farm
POST /api/farms
Body: { owner_id, latitude, longitude, area_hectares, soil_type }

// Get farm
GET /api/farms/{farm_id}

// List user farms
GET /api/users/{user_id}/farms
```

### Query/Recommendations
```javascript
// Process query (main AI feature)
POST /api/query
Body: {
  user_id,
  query_text,
  farm_id,
  latitude,
  longitude,
  crop_type,
  language
}

// Get recommendation
GET /api/recommendations/{recommendation_id}

// List user recommendations
GET /api/users/{user_id}/recommendations
```

### Products/Marketplace
```javascript
// Create product
POST /api/products
Body: { farmer_id, farm_id, product_type, name, quantity_kg, price_per_kg }

// Get product
GET /api/products/{product_id}

// Search products
GET /api/products?product_type=vegetables&min_score=0.8

// List farmer products
GET /api/farmers/{farmer_id}/products
```

## 🔐 Authentication Flow

### 1. Login Page
```tsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function LoginPage() {
  const [phone, setPhone] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const res = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone_number: phone, password }),
      });
      
      const data = await res.json();
      
      if (data.status === 'success') {
        localStorage.setItem('access_token', data.data.access_token);
        localStorage.setItem('user', JSON.stringify(data.data.user));
        navigate('/dashboard');
      } else {
        alert('Login failed');
      }
    } catch (error) {
      alert('Error: ' + error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full space-y-8 p-8 bg-white rounded-lg shadow">
        <h2 className="text-3xl font-bold text-center">Login to GramBrain</h2>
        <form onSubmit={handleLogin} className="space-y-6">
          <div>
            <label className="block text-sm font-medium">Phone Number</label>
            <input
              type="tel"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
              placeholder="+91 98765 43210"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
              required
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 px-4 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>
        <p className="text-center text-sm">
          Don't have an account?{' '}
          <a href="/register" className="text-green-600 hover:underline">
            Register
          </a>
        </p>
      </div>
    </div>
  );
}
```

### 2. API Helper with Auth
```typescript
// lib/api.ts
const API_BASE = 'http://localhost:8000';

export async function apiCall(endpoint: string, options: RequestInit = {}) {
  const token = localStorage.getItem('access_token');
  
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(token && { Authorization: `Bearer ${token}` }),
    ...options.headers,
  };

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (response.status === 401) {
    localStorage.removeItem('access_token');
    window.location.href = '/login';
    throw new Error('Unauthorized');
  }

  return response.json();
}

// Usage examples
export const api = {
  // Auth
  login: (phone: string, password: string) =>
    apiCall('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ phone_number: phone, password }),
    }),

  register: (data: any) =>
    apiCall('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Query
  processQuery: (queryData: any) =>
    apiCall('/api/query', {
      method: 'POST',
      body: JSON.stringify(queryData),
    }),

  // Farms
  getUserFarms: (userId: string) =>
    apiCall(`/api/users/${userId}/farms`),

  createFarm: (farmData: any) =>
    apiCall('/api/farms', {
      method: 'POST',
      body: JSON.stringify(farmData),
    }),

  // Products
  getProducts: (filters?: any) => {
    const params = new URLSearchParams(filters);
    return apiCall(`/api/products?${params}`);
  },

  createProduct: (productData: any) =>
    apiCall('/api/products', {
      method: 'POST',
      body: JSON.stringify(productData),
    }),
};
```

### 3. Protected Route Component
```tsx
// components/ProtectedRoute.tsx
import { Navigate } from 'react-router-dom';

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const token = localStorage.getItem('access_token');
  
  if (!token) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
}

// Usage in App.tsx
import { ProtectedRoute } from './components/ProtectedRoute';

<Route path="/dashboard" element={
  <ProtectedRoute>
    <Dashboard />
  </ProtectedRoute>
} />
```

## 🎨 UI Components to Add

### Loading Spinner
```tsx
export function LoadingSpinner() {
  return (
    <div className="flex justify-center items-center">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600"></div>
    </div>
  );
}
```

### Error Message
```tsx
export function ErrorMessage({ message }: { message: string }) {
  return (
    <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded">
      {message}
    </div>
  );
}
```

### Success Toast
```tsx
export function SuccessToast({ message }: { message: string }) {
  return (
    <div className="fixed top-4 right-4 bg-green-50 border border-green-200 text-green-800 px-6 py-4 rounded-lg shadow-lg">
      ✓ {message}
    </div>
  );
}
```

## 📱 Example: Query Page with Auth
```tsx
import { useState } from 'react';
import { api } from '../lib/api';
import { LoadingSpinner } from '../components/LoadingSpinner';

export default function QueryPage() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const user = JSON.parse(localStorage.getItem('user') || '{}');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const data = await api.processQuery({
        user_id: user.user_id,
        query_text: query,
        language: 'en',
      });
      
      if (data.status === 'success') {
        setResult(data.data.recommendation);
      }
    } catch (error) {
      alert('Error processing query');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Ask GramBrain AI</h1>
      
      <form onSubmit={handleSubmit} className="mb-8">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full p-4 border border-gray-300 rounded-lg"
          rows={4}
          placeholder="Ask anything about farming..."
          required
        />
        <button
          type="submit"
          disabled={loading}
          className="mt-4 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
        >
          {loading ? 'Processing...' : 'Get Recommendation'}
        </button>
      </form>

      {loading && <LoadingSpinner />}

      {result && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-bold mb-4">Recommendation</h2>
          <p className="text-gray-700 whitespace-pre-wrap">
            {result.recommendation_text}
          </p>
          <div className="mt-4 text-sm text-gray-500">
            Confidence: {(result.confidence * 100).toFixed(0)}%
          </div>
        </div>
      )}
    </div>
  );
}
```

## ✅ Quick Checklist

- [ ] Install dependencies: `npm install`
- [ ] Create login page
- [ ] Create register page
- [ ] Add API helper with auth
- [ ] Add protected route wrapper
- [ ] Update existing pages to use auth
- [ ] Add loading states
- [ ] Add error handling
- [ ] Test all flows
- [ ] Polish UI

## 🎯 Priority Pages to Polish

1. **Home** - Hero, features, CTA
2. **Dashboard** - Overview, stats, quick actions
3. **Query** - Chat interface, recommendations
4. **Marketplace** - Product grid, filters
5. **Farms** - Farm cards, management

---

**Backend Status**: ✅ Ready
**Your Task**: Integrate auth + polish UI
**Time Available**: ~5 hours
**Let's make it amazing!** 🚀
