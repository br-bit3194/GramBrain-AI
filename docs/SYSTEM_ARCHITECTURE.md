# GramBrain AI - System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Port 3000)                      │
│                    Next.js + React + TypeScript                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Pages:                    Components:           Services:       │
│  ├─ Home                   ├─ Header             ├─ API Client   │
│  ├─ Dashboard              ├─ Footer             └─ Zustand      │
│  ├─ Farms                  ├─ Layout                Store        │
│  ├─ Query                  ├─ FarmCard                           │
│  ├─ Marketplace            ├─ ProductCard                        │
│  ├─ Login                  ├─ QueryForm                          │
│  ├─ Register               └─ FarmForm                           │
│  └─ Profile                                                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓ HTTP/REST
                         CORS Enabled
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Backend (Port 8000)                       │
│                    FastAPI + Python + Pydantic                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  API Endpoints:                                                  │
│  ├─ /health                    (Health Check)                    │
│  ├─ /api/users                 (User Management)                 │
│  ├─ /api/farms                 (Farm Management)                 │
│  ├─ /api/query                 (Query Processing)                │
│  ├─ /api/products              (Marketplace)                     │
│  └─ /api/knowledge             (RAG System)                      │
│                                                                   │
│  Request Validation:                                             │
│  ├─ CreateUserRequest                                            │
│  ├─ CreateFarmRequest                                            │
│  ├─ ProcessQueryRequest                                          │
│  ├─ CreateProductRequest                                         │
│  └─ AddKnowledgeRequest                                          │
│                                                                   │
│  Error Handling:                                                 │
│  ├─ ValidationErrorHandler     (422 - Validation Error)          │
│  ├─ HTTPExceptionHandler       (4xx/5xx - HTTP Errors)           │
│  └─ GeneralExceptionHandler    (500 - Server Errors)             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Agent System (Backend)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Specialized Agents:                                             │
│  ├─ Crop Advisory Agent        (Crop recommendations)            │
│  ├─ Farmer Interaction Agent   (User communication)              │
│  ├─ Irrigation Agent           (Water management)                │
│  ├─ Market Agent               (Market analysis)                 │
│  ├─ Marketplace Agent          (Product trading)                 │
│  ├─ Pest Management Agent      (Pest control)                    │
│  ├─ Soil Agent                 (Soil analysis)                   │
│  ├─ Sustainability Agent       (Eco-friendly practices)          │
│  ├─ Village Agent              (Community support)               │
│  ├─ Weather Agent              (Weather forecasting)             │
│  ├─ Yield Agent                (Yield prediction)                │
│  └─ Village Agent              (Village-level insights)          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Data & Knowledge Layer                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Data Models:                  RAG System:                       │
│  ├─ User                       ├─ Vector DB                      │
│  ├─ Farm                       ├─ Embeddings                     │
│  ├─ CropCycle                  ├─ Retrieval                      │
│  ├─ Recommendation             └─ Knowledge Base                 │
│  ├─ Product                                                      │
│  └─ Knowledge                  LLM Integration:                  │
│                                ├─ AWS Bedrock                    │
│                                └─ Prompt Engineering             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Request/Response Flow

### User Registration Flow

```
Frontend                          Backend
   │                                │
   ├─ User fills form               │
   │                                │
   ├─ POST /api/users ─────────────→│
   │  {                             │
   │    phone_number: "...",        │
   │    name: "...",                │
   │    language_preference: "...", │
   │    role: "..."                 │
   │  }                             │
   │                                │
   │                    ┌─ Validate with CreateUserRequest
   │                    ├─ Generate user_id
   │                    ├─ Create user object
   │                    └─ Return response
   │                                │
   │←─────────────────────────────── ←─ 200 OK
   │  {                             │
   │    status: "success",          │
   │    data: {                     │
   │      user: { ... }             │
   │    }                           │
   │  }                             │
   │                                │
   ├─ Display success message       │
   │                                │
```

### Error Handling Flow

```
Frontend                          Backend
   │                                │
   ├─ POST /api/users ─────────────→│
   │  {                             │
   │    name: "..."                 │
   │    (missing phone_number)      │
   │  }                             │
   │                                │
   │                    ┌─ Validate with CreateUserRequest
   │                    ├─ Validation fails
   │                    ├─ RequestValidationError caught
   │                    └─ Convert to JSON response
   │                                │
   │←─────────────────────────────── ←─ 422 Unprocessable Entity
   │  {                             │
   │    status: "error",            │
   │    detail: "Validation error", │
   │    errors: [                   │
   │      {                         │
   │        field: "phone_number",  │
   │        message: "Field required"
   │      }                         │
   │    ]                           │
   │  }                             │
   │                                │
   ├─ Display error message         │
   │                                │
```

### Query Processing Flow

```
Frontend                          Backend                    Agents
   │                                │                          │
   ├─ User submits query            │                          │
   │                                │                          │
   ├─ POST /api/query ─────────────→│                          │
   │  {                             │                          │
   │    user_id: "...",             │                          │
   │    query_text: "...",          │                          │
   │    crop_type: "...",           │                          │
   │    ...                         │                          │
   │  }                             │                          │
   │                                │                          │
   │                    ┌─ Validate with ProcessQueryRequest
   │                    ├─ Route to appropriate agents
   │                    │                                      │
   │                    │  ├─ Crop Advisory Agent ────────────→│
   │                    │  ├─ Weather Agent ─────────────────→│
   │                    │  ├─ Soil Agent ────────────────────→│
   │                    │  └─ Irrigation Agent ──────────────→│
   │                    │                                      │
   │                    │←─ Collect responses ────────────────│
   │                    │                                      │
   │                    ├─ Synthesize recommendation
   │                    ├─ Generate reasoning chain
   │                    └─ Return response
   │                                │
   │←─────────────────────────────── ←─ 200 OK
   │  {                             │
   │    status: "success",          │
   │    data: {                     │
   │      recommendation: {         │
   │        recommendation_id: "...",
   │        recommendation_text: "...",
   │        reasoning_chain: [...],
   │        confidence: 0.85,       │
   │        agent_contributions: [...]
   │      }                         │
   │    }                           │
   │  }                             │
   │                                │
   ├─ Display recommendation        │
   │                                │
```

## API Endpoint Structure

```
GET  /health
     └─ Health check

POST /api/users
GET  /api/users/{user_id}
     └─ User management

POST /api/farms
GET  /api/farms/{farm_id}
GET  /api/users/{user_id}/farms
     └─ Farm management

POST /api/query
GET  /api/recommendations/{recommendation_id}
GET  /api/users/{user_id}/recommendations
     └─ Query & recommendations

POST /api/products
GET  /api/products/{product_id}
GET  /api/products (search)
GET  /api/farmers/{farmer_id}/products
     └─ Marketplace

POST /api/knowledge
GET  /api/knowledge/search
     └─ Knowledge base
```

## Data Model Relationships

```
User
├─ user_id (PK)
├─ phone_number
├─ name
├─ language_preference
├─ role
└─ created_at

Farm
├─ farm_id (PK)
├─ owner_id (FK → User)
├─ location (lat, lon)
├─ area_hectares
├─ soil_type
├─ irrigation_type
└─ crops

CropCycle
├─ cycle_id (PK)
├─ farm_id (FK → Farm)
├─ crop_type
├─ variety
├─ planting_date
├─ expected_harvest_date
└─ growth_stage

Recommendation
├─ recommendation_id (PK)
├─ query_id
├─ user_id (FK → User)
├─ farm_id (FK → Farm)
├─ recommendation_text
├─ reasoning_chain
├─ confidence
└─ agent_contributions

Product
├─ product_id (PK)
├─ farmer_id (FK → User)
├─ farm_id (FK → Farm)
├─ product_type
├─ name
├─ quantity_kg
├─ price_per_kg
├─ harvest_date
└─ status

Knowledge
├─ chunk_id (PK)
├─ content
├─ source
├─ topic
├─ crop_type
└─ region
```

## Technology Stack

### Frontend
- **Framework:** Next.js 13+
- **Language:** TypeScript
- **UI Library:** React 18+
- **Styling:** Tailwind CSS
- **State Management:** Zustand
- **HTTP Client:** Axios
- **Icons:** React Icons

### Backend
- **Framework:** FastAPI
- **Language:** Python 3.13+
- **Validation:** Pydantic
- **CORS:** FastAPI CORSMiddleware
- **LLM:** AWS Bedrock
- **Vector DB:** (Configurable)
- **Testing:** Pytest

### Infrastructure
- **Containerization:** Docker
- **Orchestration:** Docker Compose
- **Deployment:** (Configurable)

## Error Handling Strategy

```
Request
   ↓
Validation (Pydantic)
   ├─ Valid → Process
   └─ Invalid → ValidationErrorHandler
                  ↓
                422 Unprocessable Entity
                {
                  status: "error",
                  detail: "Validation error",
                  errors: [...]
                }

Processing
   ├─ Success → Return data
   └─ Error → ExceptionHandler
              ↓
              500 Internal Server Error
              {
                status: "error",
                detail: "Error message"
              }
```

## Security Considerations

1. **CORS:** Configured to allow only frontend origin
2. **Validation:** All inputs validated with Pydantic
3. **Error Messages:** Generic error messages in production
4. **Type Safety:** TypeScript on frontend, Python type hints on backend
5. **Environment Variables:** Sensitive data in .env files

## Performance Considerations

1. **Async/Await:** All endpoints are async
2. **Request Validation:** Happens before processing
3. **Error Handling:** Efficient error responses
4. **Caching:** (Can be added)
5. **Database Indexing:** (Can be optimized)

## Scalability Considerations

1. **Microservices:** Agents can be separated into microservices
2. **Load Balancing:** Can be added with reverse proxy
3. **Database:** Can be scaled with replication
4. **Caching:** Redis can be added for caching
5. **Message Queue:** Can be added for async processing

