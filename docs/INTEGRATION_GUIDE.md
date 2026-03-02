# Frontend-Backend Integration Guide

## Overview
This guide explains how to integrate the frontend and backend applications and run them together.

## Prerequisites

### Backend Requirements
- Python 3.9+
- FastAPI
- AWS Bedrock access (for LLM)
- PostgreSQL or compatible database

### Frontend Requirements
- Node.js 18+
- npm or yarn

## Setup Instructions

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration:
# - AWS credentials
# - Database URL
# - API port (default: 8000)

# Run migrations (if applicable)
# python -m alembic upgrade head

# Start backend server
python -m uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local:
NEXT_PUBLIC_API_URL=http://localhost:8000/api

# Start development server
npm run dev
```

Frontend will be available at: `http://localhost:3000`

## Running Together

### Option 1: Separate Terminals

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python -m uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Option 2: Docker Compose

```bash
# From project root
docker-compose up
```

This will start:
- Backend API on `http://localhost:8000`
- Frontend on `http://localhost:3000`

## API Integration Points

### 1. User Authentication

**Frontend Flow:**
1. User registers at `/register`
2. Frontend calls `POST /api/users` with user data
3. Backend creates user and returns user object
4. Frontend stores user in Zustand store
5. User redirected to dashboard

**Backend Endpoint:**
```
POST /api/users
{
  "phone_number": "string",
  "name": "string",
  "language_preference": "string",
  "role": "farmer|village_leader|policymaker|consumer"
}
```

### 2. Farm Management

**Frontend Flow:**
1. User navigates to `/farms`
2. Frontend calls `GET /api/users/{userId}/farms`
3. Displays list of farms
4. User can create new farm via form
5. Frontend calls `POST /api/farms` with farm data

**Backend Endpoints:**
```
POST /api/farms
GET /api/farms/{farmId}
GET /api/users/{userId}/farms
```

### 3. Query Processing

**Frontend Flow:**
1. User navigates to `/query`
2. User submits query via QueryForm
3. Frontend calls `POST /api/query` with query data
4. Backend processes through agent orchestrator
5. Returns recommendation
6. Frontend displays recommendation

**Backend Endpoint:**
```
POST /api/query
{
  "user_id": "string",
  "query_text": "string",
  "farm_id": "string (optional)",
  "crop_type": "string (optional)",
  "growth_stage": "string (optional)",
  "language": "string"
}
```

### 4. Marketplace

**Frontend Flow:**
1. User navigates to `/marketplace`
2. Frontend calls `GET /api/products` with optional filters
3. Displays product list
4. User can search and filter products

**Backend Endpoints:**
```
GET /api/products?product_type=vegetables&limit=20
POST /api/products
GET /api/products/{productId}
```

## Environment Configuration

### Backend (.env)
```
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_TITLE=GramBrain AI API

# Database
DATABASE_URL=postgresql://user:password@localhost/grambrain

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# LLM Configuration
LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
```

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## Testing the Integration

### 1. Test User Registration
```bash
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "+91 98765 43210",
    "name": "Test Farmer",
    "language_preference": "en",
    "role": "farmer"
  }'
```

### 2. Test Farm Creation
```bash
curl -X POST http://localhost:8000/api/farms \
  -H "Content-Type: application/json" \
  -d '{
    "owner_id": "user_id_from_above",
    "latitude": 28.7041,
    "longitude": 77.1025,
    "area_hectares": 5.5,
    "soil_type": "loamy",
    "irrigation_type": "drip"
  }'
```

### 3. Test Query Processing
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_id",
    "query_text": "How should I irrigate my wheat farm?",
    "farm_id": "farm_id",
    "crop_type": "wheat",
    "language": "en"
  }'
```

## Troubleshooting

### CORS Issues
If frontend can't reach backend, check:
1. Backend CORS configuration in `src/api/routes.py`
2. Ensure `NEXT_PUBLIC_API_URL` is correct in frontend `.env.local`
3. Backend is running on correct port

### API Connection Errors
```
Error: Failed to fetch from http://localhost:8000/api
```

Solutions:
1. Verify backend is running: `curl http://localhost:8000/health`
2. Check firewall settings
3. Verify API URL in frontend environment

### Database Connection Errors
```
Error: Could not connect to database
```

Solutions:
1. Ensure PostgreSQL is running
2. Check DATABASE_URL in backend `.env`
3. Verify database credentials

### Module Not Found Errors
```
Error: Cannot find module 'react'
```

Solutions:
1. Run `npm install` in frontend directory
2. Clear node_modules: `rm -rf node_modules && npm install`
3. Check Node.js version: `node --version` (should be 18+)

## Development Workflow

### Making Changes

**Backend Changes:**
1. Edit files in `backend/src/`
2. Backend auto-reloads with `--reload` flag
3. Test with API documentation at `http://localhost:8000/docs`

**Frontend Changes:**
1. Edit files in `frontend/src/`
2. Frontend auto-reloads with hot module replacement
3. Check browser console for errors

### Adding New Features

**Example: Add new API endpoint**

1. **Backend:**
   - Add route in `backend/src/api/routes.py`
   - Add business logic in appropriate agent/service
   - Test with curl or Postman

2. **Frontend:**
   - Add API method in `frontend/src/services/api.ts`
   - Create component/page using the API
   - Add TypeScript types in `frontend/src/types/index.ts`

## Production Deployment

### Build Frontend
```bash
cd frontend
npm run build
npm start
```

### Build Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn src.api.routes:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build images
docker-compose build

# Run containers
docker-compose up -d

# View logs
docker-compose logs -f
```

## Performance Optimization

### Frontend
- Enable caching in Next.js
- Optimize images
- Code splitting
- Lazy loading

### Backend
- Database query optimization
- Caching with Redis
- Rate limiting
- Load balancing

## Monitoring

### Backend Logs
```bash
# View logs
docker-compose logs backend

# Follow logs
docker-compose logs -f backend
```

### Frontend Logs
```bash
# Browser console
# Check Network tab for API calls
# Check Application tab for local storage
```

## Support

For issues or questions:
1. Check API documentation: `http://localhost:8000/docs`
2. Review error messages in browser console
3. Check backend logs for API errors
4. Refer to README files in backend and frontend directories
