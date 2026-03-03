# Frontend-Backend Integration Spec

## Overview

This spec defines the integration between the GramBrain AI Next.js frontend and FastAPI backend. The goal is to connect the existing UI with the backend API to create a fully functional agricultural intelligence platform.

## Current Status

✅ **Requirements**: Complete
✅ **Design**: Complete  
✅ **Tasks**: Complete

## Key Integration Points

### 1. API Client
- Fix base URL to include `/api` prefix
- Add authentication header injection
- Implement centralized error handling
- Update all endpoint methods

### 2. Authentication
- Login and registration flows
- Token storage and management
- Protected route guards
- Session restoration

### 3. State Management
- Enhanced Zustand store with auth state
- Farm context management
- Loading and error states

### 4. Core Features
- Farm management (CRUD operations)
- Query interface with AI recommendations
- Marketplace with product listings
- User profile and history

## Quick Start

To begin implementation:

1. Open `tasks.md` in this directory
2. Click "Start task" next to task 1
3. Follow the implementation plan sequentially
4. Optional tasks (marked with *) can be skipped for MVP

## Testing Strategy

- **Unit Tests**: Jest + React Testing Library
- **Property Tests**: fast-check (optional for MVP)
- **Integration Tests**: Playwright (optional for MVP)
- **Coverage Target**: 80% (optional for MVP)

## Key Files to Modify

### Frontend
- `frontend/src/services/api.ts` - API client
- `frontend/src/store/appStore.ts` - State management
- `frontend/src/hooks/useAuth.ts` - Auth hook
- `frontend/src/hooks/useFarm.ts` - Farm hook
- `frontend/src/hooks/useQuery.ts` - Query hook
- `frontend/src/app/login/page.tsx` - Login page
- `frontend/src/app/register/page.tsx` - Register page
- `frontend/.env.local` - Environment config

### Backend (Reference Only)
- `backend/src/api/routes.py` - API endpoints
- `backend/main.py` - Server entry point

## Environment Variables

```bash
# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## API Endpoints Reference

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user

### Farms
- `POST /api/farms` - Create farm
- `GET /api/farms/{farm_id}` - Get farm
- `GET /api/users/{user_id}/farms` - List user farms

### Query
- `POST /api/query` - Process query and get recommendation
- `GET /api/recommendations/{id}` - Get recommendation
- `GET /api/users/{user_id}/recommendations` - List recommendations

### Products
- `POST /api/products` - Create product
- `GET /api/products` - Search products
- `GET /api/products/{id}` - Get product
- `GET /api/farmers/{farmer_id}/products` - List farmer products

## Success Criteria

The integration is complete when:

1. ✅ Users can register and login
2. ✅ Authentication tokens are stored and used
3. ✅ Protected routes redirect unauthenticated users
4. ✅ Users can create and manage farms
5. ✅ Users can submit queries and receive recommendations
6. ✅ Users can browse and list products
7. ✅ All API calls use correct endpoints
8. ✅ Loading and error states are handled gracefully
9. ✅ Navigation maintains auth and farm context

## Next Steps

1. Start with task 1: Fix API Client Base Configuration
2. Work through tasks sequentially
3. Test each feature as you implement it
4. Skip optional tasks for faster MVP
5. Come back to optional tasks for production readiness

## Questions?

If you encounter issues or need clarification:
- Review the design document for detailed architecture
- Check the requirements document for acceptance criteria
- Refer to backend routes.py for exact API contracts
- Ask for help if stuck!

---

**Ready to start?** Open `tasks.md` and click "Start task" on task 1! 🚀
