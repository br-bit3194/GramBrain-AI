# GramBrain AI - Quick Reference Guide

## 🚀 Quick Start

### Start Everything (Docker)
```bash
docker-compose up
```
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

### Start Backend Only
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn src.api.routes:app --reload
```

### Start Frontend Only
```bash
cd frontend
npm install
npm run dev
```

## 📁 Project Structure

```
grambrain-ai/
├── backend/              # Python/FastAPI backend
├── frontend/             # Next.js/React frontend
├── docker-compose.yml    # Docker orchestration
├── INTEGRATION_GUIDE.md  # How to integrate
├── FRONTEND_SETUP.md     # Frontend guide
└── PROJECT_STATUS.md     # Project status
```

## 🔑 Key Files

### Backend
- `backend/src/api/routes.py` - API endpoints
- `backend/src/agents/` - AI agents
- `backend/src/core/orchestrator.py` - Agent orchestrator
- `backend/tests/` - Test suite

### Frontend
- `frontend/src/app/` - Pages
- `frontend/src/components/` - Components
- `frontend/src/hooks/` - Custom hooks
- `frontend/src/services/api.ts` - API client
- `frontend/src/store/appStore.ts` - State management

## 🌐 API Endpoints

### Users
```
POST   /api/users              # Create user
GET    /api/users/{userId}     # Get user
```

### Farms
```
POST   /api/farms              # Create farm
GET    /api/farms/{farmId}     # Get farm
GET    /api/users/{userId}/farms  # List farms
```

### Queries
```
POST   /api/query              # Process query
GET    /api/recommendations/{id}   # Get recommendation
GET    /api/users/{userId}/recommendations  # List recommendations
```

### Products
```
POST   /api/products           # Create product
GET    /api/products           # Search products
GET    /api/products/{id}      # Get product
```

### Knowledge
```
POST   /api/knowledge          # Add knowledge
GET    /api/knowledge/search   # Search knowledge
```

## 🎨 Frontend Pages

| Page | Route | Purpose |
|------|-------|---------|
| Home | `/` | Landing page |
| Dashboard | `/dashboard` | User dashboard |
| Farms | `/farms` | Farm management |
| Query | `/query` | AI query interface |
| Marketplace | `/marketplace` | Product marketplace |
| Login | `/login` | User login |
| Register | `/register` | User registration |
| Profile | `/profile` | User profile |

## 🧩 Frontend Components

### Layout
- `Header` - Navigation
- `Footer` - Footer
- `Layout` - Main wrapper

### Cards
- `FarmCard` - Farm display
- `ProductCard` - Product display

### Forms
- `QueryForm` - Query submission
- `FarmForm` - Farm creation

## 🪝 Custom Hooks

### useAuth
```typescript
const { user, isAuthenticated, login, register, logout } = useAuth()
```

### useFarm
```typescript
const { farm, farms, loading, error, createFarm, getFarm, listUserFarms } = useFarm()
```

### useQuery
```typescript
const { recommendation, recommendations, loading, error, processQuery, getRecommendation, listUserRecommendations } = useQuery()
```

## 🔧 Environment Setup

### Backend (.env)
```
DATABASE_URL=postgresql://user:password@localhost/grambrain
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
```

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## 📦 Dependencies

### Backend
- FastAPI
- SQLAlchemy
- Pydantic
- boto3 (AWS)
- pytest

### Frontend
- Next.js
- React
- TypeScript
- Tailwind CSS
- Zustand
- Axios
- React Icons

## 🧪 Testing

### Backend
```bash
cd backend
pytest tests/ -v
```

### Frontend
```bash
cd frontend
npm test
```

## 🚢 Deployment

### Build Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn src.api.routes:app --host 0.0.0.0 --port 8000
```

### Build Frontend
```bash
cd frontend
npm run build
npm start
```

### Docker
```bash
docker-compose build
docker-compose up -d
```

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.9+

# Check dependencies
pip install -r requirements.txt

# Check database connection
# Verify DATABASE_URL in .env
```

### Frontend won't start
```bash
# Check Node version
node --version  # Should be 18+

# Clear cache
rm -rf node_modules .next
npm install

# Check API URL
# Verify NEXT_PUBLIC_API_URL in .env.local
```

### API connection errors
```bash
# Check backend is running
curl http://localhost:8000/health

# Check CORS settings
# Verify API URL in frontend

# Check firewall
# Ensure ports 8000 and 3000 are open
```

## 📚 Documentation

- **INTEGRATION_GUIDE.md** - How to integrate frontend and backend
- **FRONTEND_SETUP.md** - Frontend development guide
- **PROJECT_STATUS.md** - Complete project status
- **FRONTEND_COMPLETION_SUMMARY.md** - Frontend summary
- **README.md** - Project overview

## 🎯 Common Tasks

### Add a new API endpoint
1. Add route in `backend/src/api/routes.py`
2. Add method in `frontend/src/services/api.ts`
3. Add types in `frontend/src/types/index.ts`
4. Create component/page using the API

### Add a new page
1. Create file in `frontend/src/app/[page]/page.tsx`
2. Add navigation link in `Header.tsx`
3. Add route to navigation menu

### Add a new component
1. Create file in `frontend/src/components/`
2. Export from `frontend/src/components/index.ts`
3. Import and use in pages

### Add a new hook
1. Create file in `frontend/src/hooks/`
2. Export from `frontend/src/hooks/index.ts`
3. Use in components with `'use client'` directive

## 🔐 Security Checklist

- ✅ Input validation on backend
- ✅ CORS configured
- ✅ Environment variables for secrets
- ✅ Error handling
- ✅ Rate limiting ready
- ⏳ JWT authentication (planned)
- ⏳ HTTPS in production (planned)

## 📊 Performance Tips

### Backend
- Use database indexes
- Cache frequently accessed data
- Optimize agent queries
- Use connection pooling

### Frontend
- Lazy load components
- Optimize images
- Use code splitting
- Enable caching

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Test thoroughly
4. Submit pull request
5. Code review
6. Merge to main

## 📞 Support

For issues:
1. Check documentation
2. Review error messages
3. Check logs
4. Refer to troubleshooting section

## 🎓 Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev/)
- [TypeScript Documentation](https://www.typescriptlang.org/docs/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)

## 📝 Notes

- Backend auto-reloads with `--reload` flag
- Frontend hot-reloads automatically
- API documentation at `http://localhost:8000/docs`
- Check browser console for frontend errors
- Check terminal for backend errors

---

**Last Updated**: February 28, 2026  
**Version**: 1.0.0
