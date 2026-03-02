# GramBrain AI - Quick Commands Reference

## Start the System

### Backend
```bash
cd backend
source venv/bin/activate
python -m uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm run dev
```

### Both (in separate terminals)
```bash
# Terminal 1
cd backend && source venv/bin/activate && python -m uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000

# Terminal 2
cd frontend && npm run dev
```

---

## Test Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Create User (Valid)
```bash
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "+91 98765 43210",
    "name": "Rajesh Kumar",
    "language_preference": "en",
    "role": "farmer"
  }'
```

### Create User (Invalid - Missing Field)
```bash
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Rajesh Kumar"
  }'
```

### Create Farm
```bash
curl -X POST http://localhost:8000/api/farms \
  -H "Content-Type: application/json" \
  -d '{
    "owner_id": "user-123",
    "latitude": 28.7041,
    "longitude": 77.1025,
    "area_hectares": 5.5,
    "soil_type": "loamy",
    "irrigation_type": "drip"
  }'
```

### Process Query
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "query_text": "How should I irrigate my wheat crop?",
    "crop_type": "wheat",
    "growth_stage": "vegetative",
    "soil_type": "loamy",
    "language": "en"
  }'
```

### Create Product
```bash
curl -X POST http://localhost:8000/api/products \
  -H "Content-Type: application/json" \
  -d '{
    "farmer_id": "user-123",
    "farm_id": "farm-123",
    "product_type": "vegetables",
    "name": "Organic Tomatoes",
    "quantity_kg": 100,
    "price_per_kg": 50,
    "harvest_date": "2026-03-02"
  }'
```

### Add Knowledge
```bash
curl -X POST http://localhost:8000/api/knowledge \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_id": "chunk-001",
    "content": "Wheat requires 450-650mm of water during its growing season",
    "source": "Agricultural Research Institute",
    "topic": "irrigation",
    "crop_type": "wheat",
    "region": "North India"
  }'
```

---

## Browser URLs

| Page | URL |
|------|-----|
| Home | http://localhost:3000 |
| Dashboard | http://localhost:3000/dashboard |
| Farms | http://localhost:3000/farms |
| Query | http://localhost:3000/query |
| Marketplace | http://localhost:3000/marketplace |
| Login | http://localhost:3000/login |
| Register | http://localhost:3000/register |
| Profile | http://localhost:3000/profile |

---

## Troubleshooting Commands

### Kill Process on Port 8000
```bash
lsof -ti:8000 | xargs kill -9
```

### Kill Process on Port 3000
```bash
lsof -ti:3000 | xargs kill -9
```

### Clear Frontend Cache
```bash
cd frontend
rm -rf node_modules .next
npm install
npm run dev
```

### Check Backend Logs
```bash
# If running in background
tail -f backend.log

# If running in terminal, check terminal output
```

### Check Frontend Logs
```bash
# Check browser console: F12 or Cmd+Option+I
# Check terminal where npm run dev is running
```

---

## Installation Commands

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

---

## Testing Commands

### Run Backend Tests
```bash
cd backend
source venv/bin/activate
pytest tests/
```

### Run Frontend Build
```bash
cd frontend
npm run build
```

### Run Frontend Tests
```bash
cd frontend
npm test
```

---

## Docker Commands

### Build Images
```bash
docker build -t grambrain-backend:latest ./backend
docker build -t grambrain-frontend:latest ./frontend
```

### Run with Docker Compose
```bash
docker-compose up -d
```

### View Logs
```bash
docker logs -f grambrain-backend
docker logs -f grambrain-frontend
```

### Stop Services
```bash
docker-compose down
```

---

## Environment Variables

### Backend (.env)
```
DATABASE_URL=postgresql://user:password@localhost:5432/grambrain
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
API_HOST=0.0.0.0
API_PORT=8000
```

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

---

## File Locations

| Component | Location |
|-----------|----------|
| Backend API | `backend/src/api/routes.py` |
| Backend Agents | `backend/src/agents/` |
| Frontend Pages | `frontend/src/app/` |
| Frontend Components | `frontend/src/components/` |
| Frontend Services | `frontend/src/services/` |
| Frontend Types | `frontend/src/types/` |
| Frontend Store | `frontend/src/store/` |
| Backend Tests | `backend/tests/` |
| Docker Compose | `docker-compose.yml` |

---

## API Endpoints Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check |
| POST | `/api/users` | Create user |
| GET | `/api/users/{id}` | Get user |
| POST | `/api/farms` | Create farm |
| GET | `/api/farms/{id}` | Get farm |
| GET | `/api/users/{id}/farms` | List user farms |
| POST | `/api/query` | Process query |
| GET | `/api/recommendations/{id}` | Get recommendation |
| GET | `/api/users/{id}/recommendations` | List recommendations |
| POST | `/api/products` | Create product |
| GET | `/api/products/{id}` | Get product |
| GET | `/api/products` | Search products |
| GET | `/api/farmers/{id}/products` | List farmer products |
| POST | `/api/knowledge` | Add knowledge |
| GET | `/api/knowledge/search` | Search knowledge |

---

## Response Format

### Success Response
```json
{
  "status": "success",
  "data": {
    "key": "value"
  }
}
```

### Error Response
```json
{
  "status": "error",
  "detail": "Error message"
}
```

### Validation Error Response
```json
{
  "status": "error",
  "detail": "Validation error",
  "errors": [
    {
      "field": "field_name",
      "message": "Error message"
    }
  ]
}
```

---

## Documentation Files

| File | Purpose |
|------|---------|
| `RUN_NOW.md` | Quick start (5 min) |
| `TESTING_GUIDE.md` | Comprehensive testing |
| `FINAL_RUN_GUIDE.md` | Detailed instructions |
| `SYSTEM_ARCHITECTURE.md` | Architecture overview |
| `BACKEND_FIX_SUMMARY.md` | Technical details |
| `DEPLOYMENT_CHECKLIST.md` | Deployment guide |
| `FINAL_STATUS.md` | Project status |
| `QUICK_COMMANDS.md` | This file |

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Port 8000 in use | `lsof -ti:8000 \| xargs kill -9` |
| Port 3000 in use | `lsof -ti:3000 \| xargs kill -9` |
| Module not found | `pip install -r requirements.txt` |
| npm not found | `npm install` |
| CORS error | Check backend running, check .env.local |
| Validation error objects | Clear cache, restart frontend |
| Backend won't start | Check Python version, check dependencies |
| Frontend won't start | Check Node version, check dependencies |

---

## Quick Checklist

- [ ] Backend running on port 8000
- [ ] Frontend running on port 3000
- [ ] Health check responds
- [ ] User creation works
- [ ] No CORS errors
- [ ] No validation error objects
- [ ] Success messages display
- [ ] All pages load

---

## Next Steps

1. **Start:** `RUN_NOW.md`
2. **Test:** `TESTING_GUIDE.md`
3. **Deploy:** `DEPLOYMENT_CHECKLIST.md`

---

**Ready to run? Start with the commands above!** 🚀

