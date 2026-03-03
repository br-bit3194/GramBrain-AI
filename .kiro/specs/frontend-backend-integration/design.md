# Design Document: Frontend-Backend Integration

## Overview

This design document outlines the integration architecture for connecting the GramBrain AI Next.js frontend with the FastAPI backend. The integration focuses on establishing reliable API communication, implementing authentication flows, managing application state, and ensuring type safety across the frontend-backend boundary.

The design follows a layered architecture pattern with clear separation between:
- Presentation layer (React components)
- State management layer (Zustand store)
- Service layer (API client)
- Backend API layer (FastAPI endpoints)

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js)                       │
├─────────────────────────────────────────────────────────────┤
│  Pages/Routes                                                │
│  ├─ Home, Login, Register, Dashboard                        │
│  ├─ Query, Farms, Marketplace, Profile                      │
│  └─ Protected Route Wrapper                                 │
├─────────────────────────────────────────────────────────────┤
│  Components                                                  │
│  ├─ Layout (Header, Footer)                                 │
│  ├─ Forms (Login, Register, Farm, Product)                  │
│  ├─ Cards (Farm, Product, Recommendation)                   │
│  └─ UI (Loading, Error, Success)                            │
├─────────────────────────────────────────────────────────────┤
│  State Management (Zustand)                                 │
│  ├─ User State                                              │
│  ├─ Farm State                                              │
│  └─ UI State (loading, errors)                              │
├─────────────────────────────────────────────────────────────┤
│  Service Layer                                              │
│  ├─ API Client (axios)                                      │
│  ├─ Auth Interceptors                                       │
│  └─ Error Handlers                                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/REST
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
├─────────────────────────────────────────────────────────────┤
│  API Routes                                                  │
│  ├─ /api/auth/* (register, login, me)                       │
│  ├─ /api/users/* (CRUD operations)                          │
│  ├─ /api/farms/* (CRUD operations)                          │
│  ├─ /api/query (AI recommendations)                         │
│  ├─ /api/products/* (marketplace)                           │
│  └─ /api/knowledge/* (RAG operations)                       │
├─────────────────────────────────────────────────────────────┤
│  Middleware                                                  │
│  ├─ CORS                                                     │
│  ├─ Auth (JWT verification)                                 │
│  └─ Error Handling                                          │
├─────────────────────────────────────────────────────────────┤
│  Business Logic                                             │
│  ├─ GramBrain System                                        │
│  ├─ Agent Orchestrator                                      │
│  └─ Data Repositories                                       │
└─────────────────────────────────────────────────────────────┘
```

### Authentication Flow

```
User → Login Form → API Client → POST /api/auth/login → Backend
                                                           │
                                                           ├─ Verify Credentials
                                                           ├─ Generate JWT Tokens
                                                           └─ Return Tokens + User
                                                           
Backend → API Client → Store Tokens → Update Store → Redirect to Dashboard
```

### API Request Flow

```
Component → Hook → API Client → Add Auth Header → HTTP Request → Backend
                                                                    │
                                                                    ├─ Verify Token
                                                                    ├─ Process Request
                                                                    └─ Return Response
                                                                    
Backend → API Client → Parse Response → Update Store → Re-render Component
```

## Components and Interfaces

### API Client Interface

The API client serves as the single point of communication with the backend. It handles:
- Base URL configuration
- Request/response interceptors
- Authentication header injection
- Error handling and transformation
- Type-safe request/response handling

```typescript
interface ApiClient {
  // Auth
  register(data: RegisterRequest): Promise<AuthResponse>
  login(data: LoginRequest): Promise<AuthResponse>
  getCurrentUser(): Promise<UserResponse>
  
  // Users
  createUser(data: CreateUserRequest): Promise<UserResponse>
  getUser(userId: string): Promise<UserResponse>
  
  // Farms
  createFarm(data: CreateFarmRequest): Promise<FarmResponse>
  getFarm(farmId: string): Promise<FarmResponse>
  listUserFarms(userId: string): Promise<FarmsResponse>
  
  // Query
  processQuery(data: QueryRequest): Promise<RecommendationResponse>
  getRecommendation(id: string): Promise<RecommendationResponse>
  listUserRecommendations(userId: string): Promise<RecommendationsResponse>
  
  // Products
  createProduct(data: CreateProductRequest): Promise<ProductResponse>
  getProduct(productId: string): Promise<ProductResponse>
  searchProducts(filters: ProductFilters): Promise<ProductsResponse>
  listFarmerProducts(farmerId: string): Promise<ProductsResponse>
  
  // Knowledge
  addKnowledge(data: KnowledgeRequest): Promise<MessageResponse>
  searchKnowledge(query: string, topK: number): Promise<KnowledgeResponse>
  
  // Health
  healthCheck(): Promise<HealthResponse>
}
```

### State Management Interface

The Zustand store manages global application state:

```typescript
interface AppStore {
  // User state
  user: User | null
  accessToken: string | null
  refreshToken: string | null
  
  // Farm state
  selectedFarm: Farm | null
  userFarms: Farm[]
  
  // UI state
  isLoading: boolean
  error: string | null
  
  // Actions
  setUser(user: User | null): void
  setTokens(access: string, refresh: string): void
  setSelectedFarm(farm: Farm | null): void
  setUserFarms(farms: Farm[]): void
  setLoading(loading: boolean): void
  setError(error: string | null): void
  clearStore(): void
  logout(): void
}
```

### Custom Hooks Interface

React hooks provide reusable logic for common operations:

```typescript
interface UseAuth {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  login(phone: string, password: string): Promise<void>
  register(data: RegisterData): Promise<void>
  logout(): void
}

interface UseFarm {
  farms: Farm[]
  selectedFarm: Farm | null
  isLoading: boolean
  error: string | null
  loadFarms(): Promise<void>
  selectFarm(farmId: string): void
  createFarm(data: CreateFarmData): Promise<Farm>
}

interface UseQuery {
  isLoading: boolean
  error: string | null
  recommendation: Recommendation | null
  submitQuery(query: string, context: QueryContext): Promise<void>
  clearRecommendation(): void
}
```

## Data Models

### Frontend Type Definitions

```typescript
// Auth types
interface RegisterRequest {
  phone_number: string
  name: string
  password: string
  language_preference?: string
  role?: string
}

interface LoginRequest {
  phone_number: string
  password: string
}

interface AuthResponse {
  status: 'success' | 'error'
  data: {
    user: User
    access_token: string
    refresh_token: string
    token_type: string
  }
}

// User types
interface User {
  user_id: string
  phone_number: string
  name: string
  language_preference: string
  role: string
  created_at: string
  last_active: string
}

// Farm types
interface Farm {
  farm_id: string
  owner_id: string
  location: {
    lat: number
    lon: number
  }
  area_hectares: number
  soil_type: string
  irrigation_type: string
  crops: Crop[]
  created_at: string
  updated_at: string
}

interface CreateFarmRequest {
  owner_id: string
  latitude: number
  longitude: number
  area_hectares: number
  soil_type: string
  irrigation_type?: string
}

// Query types
interface QueryRequest {
  user_id: string
  query_text: string
  farm_id?: string
  latitude?: number
  longitude?: number
  farm_size_hectares?: number
  crop_type?: string
  growth_stage?: string
  soil_type?: string
  language?: string
}

interface Recommendation {
  recommendation_id: string
  query_id: string
  user_id: string
  farm_id?: string
  timestamp: string
  recommendation_text: string
  reasoning_chain: string[]
  confidence: number
  agent_contributions: AgentContribution[]
  language: string
}

// Product types
interface Product {
  product_id: string
  farmer_id: string
  farm_id: string
  product_type: string
  name: string
  quantity_kg: number
  price_per_kg: number
  harvest_date: string
  images: string[]
  pure_product_score: number
  status: string
  created_at: string
}

// Generic API response wrapper
interface ApiResponse<T> {
  status: 'success' | 'error'
  data?: T
  detail?: string
  errors?: ValidationError[]
}

interface ValidationError {
  field: string
  message: string
}
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: API URL Construction

*For any* API endpoint method in the API client, the constructed URL should include the `/api` prefix after the base URL.

**Validates: Requirements 1.1**

### Property 2: Request Header Inclusion

*For any* authenticated API request, the request headers should include both `Content-Type: application/json` and `Authorization: Bearer <token>` when a token is available.

**Validates: Requirements 1.5**

### Property 3: Response Handling Completeness

*For any* API response (success or error), the API client should handle it without throwing unhandled exceptions and should return a properly structured response object.

**Validates: Requirements 1.3**

### Property 4: Error Message Transformation

*For any* network error or API error response, the system should transform it into a user-friendly error message that does not expose technical details.

**Validates: Requirements 1.4, 6.2**

### Property 5: Authentication Token Storage

*For any* successful authentication (login or register), the system should store both access_token and refresh_token in browser storage.

**Validates: Requirements 2.3**

### Property 6: Protected Route Redirection

*For any* protected route access attempt without valid authentication tokens, the system should redirect to the login page.

**Validates: Requirements 2.4, 8.4**

### Property 7: Farm Association

*For any* farm creation request, the API payload should include the owner_id field matching the authenticated user's ID.

**Validates: Requirements 3.1**

### Property 8: Query Payload Structure

*For any* query submission, the request payload should include at minimum the user_id and query_text fields, with optional farm context fields.

**Validates: Requirements 4.1**

### Property 9: Loading State Management

*For any* async API operation, the loading state should be set to true when the operation starts and set to false when it completes (success or error).

**Validates: Requirements 4.2, 6.1**

### Property 10: Product Filter Propagation

*For any* product search with filters, all non-null filter values should be included as query parameters in the API request.

**Validates: Requirements 5.2**

### Property 11: Logout Cleanup

*For any* logout operation, all authentication-related data (tokens, user object) should be removed from browser storage and application state.

**Validates: Requirements 8.3**

### Property 12: Session Restoration

*For any* application initialization, if valid tokens exist in storage, the system should restore the user session by loading the user object into state.

**Validates: Requirements 8.2**

## Error Handling

### Error Categories

1. **Network Errors**: Connection failures, timeouts, DNS errors
   - Display: "Unable to connect to server. Please check your internet connection."
   - Action: Provide retry button

2. **Authentication Errors**: Invalid credentials, expired tokens, unauthorized access
   - Display: "Authentication failed. Please login again."
   - Action: Redirect to login page, clear stored tokens

3. **Validation Errors**: Invalid form data, missing required fields
   - Display: Field-specific error messages below inputs
   - Action: Highlight invalid fields, prevent submission

4. **Server Errors**: 500 errors, backend failures
   - Display: "Something went wrong. Please try again later."
   - Action: Log error details, provide retry option

5. **Not Found Errors**: Resource doesn't exist
   - Display: "The requested resource was not found."
   - Action: Redirect to appropriate page

### Error Handling Strategy

```typescript
// Centralized error handler
function handleApiError(error: any): UserFriendlyError {
  if (error.response) {
    // Server responded with error status
    const status = error.response.status
    const data = error.response.data
    
    if (status === 401) {
      return {
        type: 'auth',
        message: 'Please login to continue',
        action: 'redirect_login'
      }
    }
    
    if (status === 422 && data.errors) {
      return {
        type: 'validation',
        message: 'Please check your input',
        fields: data.errors
      }
    }
    
    if (status >= 500) {
      return {
        type: 'server',
        message: 'Server error. Please try again later',
        action: 'retry'
      }
    }
    
    return {
      type: 'api',
      message: data.detail || 'An error occurred',
      action: 'retry'
    }
  }
  
  if (error.request) {
    // Request made but no response
    return {
      type: 'network',
      message: 'Unable to connect. Check your internet connection',
      action: 'retry'
    }
  }
  
  // Something else happened
  return {
    type: 'unknown',
    message: 'An unexpected error occurred',
    action: 'retry'
  }
}
```

### Retry Logic

For transient errors (network, server errors), implement exponential backoff:
- First retry: immediate
- Second retry: 1 second delay
- Third retry: 2 seconds delay
- Max retries: 3

## Testing Strategy

### Unit Testing

Unit tests will verify individual components and functions in isolation:

1. **API Client Tests**
   - Test URL construction for each endpoint
   - Test header injection (auth, content-type)
   - Test request payload formatting
   - Test response parsing
   - Test error transformation

2. **Hook Tests**
   - Test useAuth hook login/logout flows
   - Test useFarm hook CRUD operations
   - Test useQuery hook submission and state management
   - Test state updates and side effects

3. **Component Tests**
   - Test form validation and submission
   - Test loading state rendering
   - Test error message display
   - Test success notification display
   - Test conditional rendering based on auth state

4. **Store Tests**
   - Test state mutations
   - Test action creators
   - Test state selectors
   - Test store persistence

### Property-Based Testing

Property-based tests will verify universal properties across many inputs using the `fast-check` library for JavaScript/TypeScript. Each test will run a minimum of 100 iterations.

1. **Property Test: API URL Construction**
   - Generate random endpoint paths
   - Verify all constructed URLs include `/api` prefix
   - **Validates: Property 1**

2. **Property Test: Request Headers**
   - Generate random API requests with/without auth
   - Verify headers always include Content-Type
   - Verify Authorization header present when token exists
   - **Validates: Property 2**

3. **Property Test: Error Transformation**
   - Generate random error responses
   - Verify all errors are transformed to user-friendly messages
   - Verify no technical details are exposed
   - **Validates: Property 4**

4. **Property Test: Token Storage**
   - Generate random auth responses
   - Verify tokens are always stored after successful auth
   - **Validates: Property 5**

5. **Property Test: Loading State**
   - Generate random async operations
   - Verify loading state is true during operation
   - Verify loading state is false after completion
   - **Validates: Property 9**

6. **Property Test: Query Payload**
   - Generate random query contexts
   - Verify all payloads include required fields
   - Verify optional fields are included when provided
   - **Validates: Property 8**

7. **Property Test: Filter Propagation**
   - Generate random filter combinations
   - Verify all non-null filters appear in query params
   - **Validates: Property 10**

8. **Property Test: Logout Cleanup**
   - Generate random user sessions
   - Verify logout removes all auth data
   - **Validates: Property 11**

### Integration Testing

Integration tests will verify end-to-end flows:

1. **Authentication Flow**
   - Register → Login → Access Protected Route → Logout
   - Verify tokens are stored and used correctly
   - Verify protected routes are accessible after auth

2. **Farm Management Flow**
   - Create Farm → List Farms → Select Farm → Update Farm
   - Verify farm data persists correctly
   - Verify farm context is maintained

3. **Query Flow**
   - Submit Query → View Recommendation → View History
   - Verify query is processed correctly
   - Verify recommendations are displayed

4. **Marketplace Flow**
   - Browse Products → Apply Filters → View Product → Create Listing
   - Verify products are fetched and displayed
   - Verify filters work correctly

### Testing Tools

- **Unit Tests**: Jest + React Testing Library
- **Property-Based Tests**: fast-check
- **Integration Tests**: Playwright or Cypress
- **API Mocking**: MSW (Mock Service Worker)
- **Coverage Target**: 80% code coverage

## Implementation Phases

### Phase 1: Core API Client (Priority: High)

1. Fix API client base URL to include `/api` prefix
2. Implement request/response interceptors
3. Add authentication header injection
4. Implement centralized error handling
5. Update all endpoint methods to match backend routes

### Phase 2: Authentication (Priority: High)

1. Implement login page with form validation
2. Implement register page with form validation
3. Create useAuth hook for auth operations
4. Implement token storage and retrieval
5. Create ProtectedRoute wrapper component
6. Add logout functionality

### Phase 3: State Management (Priority: High)

1. Enhance Zustand store with auth state
2. Add farm state management
3. Add UI state (loading, errors)
4. Implement store persistence
5. Add state selectors

### Phase 4: Farm Management (Priority: Medium)

1. Update farms page to use API client
2. Implement farm creation form
3. Implement farm selection logic
4. Add farm context to query submissions
5. Display farm details

### Phase 5: Query Interface (Priority: High)

1. Update query page to use API client
2. Add loading states during query processing
3. Display recommendations with formatting
4. Show confidence scores and reasoning
5. Implement query history view

### Phase 6: Marketplace (Priority: Medium)

1. Update marketplace page to use API client
2. Implement product filtering
3. Add product creation form
4. Display Pure Product Score
5. Show farmer details

### Phase 7: Error Handling & UX (Priority: High)

1. Implement loading spinners
2. Create error message components
3. Add success notifications
4. Implement retry logic
5. Add form validation feedback

### Phase 8: Testing (Priority: High)

1. Write unit tests for API client
2. Write property-based tests
3. Write component tests
4. Write integration tests
5. Achieve 80% coverage

## Security Considerations

1. **Token Storage**: Store tokens in httpOnly cookies when possible, or use secure localStorage with XSS protection
2. **Token Expiration**: Implement token refresh logic before expiration
3. **HTTPS**: Ensure all API calls use HTTPS in production
4. **Input Validation**: Validate all user inputs on frontend before sending to backend
5. **Error Messages**: Never expose sensitive information in error messages
6. **CORS**: Backend already configured with appropriate CORS settings
7. **Rate Limiting**: Respect backend rate limits, implement client-side throttling

## Performance Considerations

1. **Request Caching**: Cache GET requests for farms, products, and recommendations
2. **Debouncing**: Debounce search inputs to reduce API calls
3. **Lazy Loading**: Lazy load components and routes
4. **Code Splitting**: Split bundles by route
5. **Image Optimization**: Use Next.js Image component for product images
6. **Pagination**: Implement pagination for lists (products, recommendations)
7. **Optimistic Updates**: Update UI optimistically for better perceived performance

## Deployment Considerations

1. **Environment Variables**: 
   - `NEXT_PUBLIC_API_URL`: Backend API base URL
   - Different values for dev, staging, production

2. **Build Process**:
   - Run TypeScript type checking
   - Run linting
   - Run tests
   - Build production bundle

3. **Monitoring**:
   - Log API errors to monitoring service
   - Track API response times
   - Monitor authentication failures

4. **Rollback Strategy**:
   - Keep previous working version
   - Feature flags for gradual rollout
   - Quick rollback capability
