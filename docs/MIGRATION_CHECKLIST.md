# GramBrain AI - Migration Checklist

## Project Reorganization: Backend & Frontend Separation

### ✅ Completed

- [x] Created `backend/` folder structure
- [x] Created `frontend/` folder structure
- [x] Created `docs/` folder for documentation
- [x] Created `docker-compose.yml` for full stack
- [x] Created `backend/Dockerfile` for containerization
- [x] Created `frontend/Dockerfile` template
- [x] Created `REORGANIZATION_GUIDE.md` with migration steps
- [x] Created `README_NEW.md` with updated structure
- [x] Created backend README
- [x] Created frontend README template

### 📋 Manual Migration Steps Required

#### Step 1: Move Backend Files to `backend/src/`

```bash
# Core framework
cp src/core/agent_base.py backend/src/core/
cp src/core/agent_registry.py backend/src/core/
cp src/core/orchestrator.py backend/src/core/

# Agents (11 files)
cp src/agents/*.py backend/src/agents/

# LLM integration
cp src/llm/bedrock_client.py backend/src/llm/

# RAG pipeline
cp src/rag/vector_db.py backend/src/rag/
cp src/rag/embeddings.py backend/src/rag/
cp src/rag/retrieval.py backend/src/rag/

# Data models
cp src/data/models.py backend/src/data/

# API
cp src/api/routes.py backend/src/api/

# Main system
cp src/system.py backend/src/

# Tests
cp tests/*.py backend/tests/

# Configuration
cp main.py backend/
cp requirements.txt backend/
cp pytest.ini backend/
cp .env.example backend/
```

#### Step 2: Create Backend __init__.py Files

```bash
# Already created:
# backend/src/__init__.py
# backend/src/core/__init__.py

# Still needed:
touch backend/src/agents/__init__.py
touch backend/src/llm/__init__.py
touch backend/src/rag/__init__.py
touch backend/src/data/__init__.py
touch backend/src/api/__init__.py
touch backend/tests/__init__.py
```

#### Step 3: Move Documentation to `docs/`

```bash
mkdir -p docs
cp API.md docs/
cp TESTING.md docs/
cp QUICKSTART.md docs/
cp IMPLEMENTATION_SUMMARY.md docs/
cp COMPLETION_REPORT.md docs/
cp design.md docs/
cp requirements.md docs/
```

#### Step 4: Update Root Level Files

```bash
# Keep at root:
# - README_NEW.md (rename to README.md after review)
# - INDEX.md
# - BUILD_SUMMARY.txt
# - REORGANIZATION_GUIDE.md
# - MIGRATION_CHECKLIST.md (this file)
# - docker-compose.yml
# - .gitignore
# - LICENSE
```

#### Step 5: Create Frontend Structure

```bash
mkdir -p frontend/src/{components,pages,hooks,services,styles,utils,types}
mkdir -p frontend/public/{images,icons,fonts}
mkdir -p frontend/tests/{components,hooks,services,utils}

# Create placeholder files
touch frontend/src/components/.gitkeep
touch frontend/src/pages/.gitkeep
touch frontend/src/hooks/.gitkeep
touch frontend/src/services/.gitkeep
touch frontend/src/styles/.gitkeep
touch frontend/src/utils/.gitkeep
touch frontend/src/types/.gitkeep
```

#### Step 6: Create Frontend Configuration Files

```bash
# Frontend package.json template
cat > frontend/package.json << 'EOF'
{
  "name": "grambrain-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "test": "jest",
    "test:coverage": "jest --coverage"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "next": "^13.4.0",
    "axios": "^1.4.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "@types/react": "^18.2.0",
    "@types/node": "^20.0.0",
    "tailwindcss": "^3.3.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0",
    "jest": "^29.5.0",
    "@testing-library/react": "^14.0.0",
    "@testing-library/jest-dom": "^5.16.0"
  }
}
EOF
```

#### Step 7: Create Frontend Configuration Files

```bash
# next.config.js
cat > frontend/next.config.js << 'EOF'
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1',
  },
}

module.exports = nextConfig
EOF

# tsconfig.json
cat > frontend/tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "jsx": "preserve",
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "allowJs": true,
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "noEmit": true,
    "incremental": true,
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
EOF

# tailwind.config.js
cat > frontend/tailwind.config.js << 'EOF'
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx}',
    './src/components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
EOF
```

#### Step 8: Create .gitignore

```bash
cat > .gitignore << 'EOF'
# Backend
backend/venv/
backend/__pycache__/
backend/*.pyc
backend/.pytest_cache/
backend/.coverage
backend/htmlcov/
backend/.env
backend/.env.local

# Frontend
frontend/node_modules/
frontend/.next/
frontend/out/
frontend/build/
frontend/.env.local
frontend/.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Misc
.cache/
dist/
EOF
```

#### Step 9: Update Backend main.py

Ensure `backend/main.py` has correct imports:

```python
import uvicorn
import sys

def main():
    """Run the API server."""
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

if __name__ == "__main__":
    main()
```

#### Step 10: Verify Backend Structure

```bash
# Check backend structure
ls -la backend/src/
ls -la backend/src/core/
ls -la backend/src/agents/
ls -la backend/tests/
```

### 🧪 Testing After Migration

#### Backend Tests
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

#### Backend API
```bash
cd backend
python main.py
# Visit http://localhost:8000/docs
```

### 📁 Final Structure Verification

```
grambrain-ai/
├── backend/                    ✅
│   ├── src/
│   │   ├── core/              ✅
│   │   ├── agents/            ✅
│   │   ├── llm/               ✅
│   │   ├── rag/               ✅
│   │   ├── data/              ✅
│   │   ├── api/               ✅
│   │   └── system.py          ✅
│   ├── tests/                 ✅
│   ├── main.py                ✅
│   ├── requirements.txt        ✅
│   ├── pytest.ini              ✅
│   ├── Dockerfile              ✅
│   └── README.md               ✅
│
├── frontend/                   🚧
│   ├── src/
│   │   ├── components/        ✅
│   │   ├── pages/             ✅
│   │   ├── hooks/             ✅
│   │   ├── services/          ✅
│   │   ├── styles/            ✅
│   │   ├── utils/             ✅
│   │   └── types/             ✅
│   ├── public/                ✅
│   ├── package.json           ✅
│   ├── next.config.js         ✅
│   ├── tsconfig.json          ✅
│   ├── tailwind.config.js     ✅
│   ├── Dockerfile             ✅
│   └── README.md              ✅
│
├── docs/                       ✅
│   ├── API.md
│   ├── TESTING.md
│   ├── QUICKSTART.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── COMPLETION_REPORT.md
│   ├── design.md
│   └── requirements.md
│
├── docker-compose.yml          ✅
├── README_NEW.md               ✅
├── REORGANIZATION_GUIDE.md     ✅
├── MIGRATION_CHECKLIST.md      ✅
├── INDEX.md                    ✅
├── BUILD_SUMMARY.txt           ✅
├── .gitignore                  ✅
└── LICENSE                     ✅
```

### 🎯 Next Steps

1. **Execute Migration Steps 1-10** above
2. **Test Backend** - Run tests and verify API works
3. **Create Frontend** - Start with Next.js setup
4. **Integrate Frontend** - Connect to backend API
5. **Deploy** - Use docker-compose for full stack

### 📝 Notes

- Backend is complete and ready to use
- Frontend structure is prepared but needs implementation
- Docker setup is ready for containerization
- All documentation is in place
- Migration is non-breaking - old files can coexist during transition

### ✅ Verification Checklist

- [ ] Backend files copied to `backend/src/`
- [ ] All `__init__.py` files created
- [ ] Tests copied to `backend/tests/`
- [ ] Documentation moved to `docs/`
- [ ] Frontend structure created
- [ ] Frontend config files created
- [ ] `.gitignore` updated
- [ ] Backend tests pass
- [ ] Backend API runs successfully
- [ ] Docker compose works
- [ ] Old `src/` folder can be removed (after verification)

### 🚀 After Migration

```bash
# Start full stack
docker-compose up

# Or manually:
# Terminal 1
cd backend && python main.py

# Terminal 2 (when frontend is ready)
cd frontend && npm run dev
```

---

**Status:** Migration structure ready. Manual file copying required.

**Estimated Time:** 30 minutes for complete migration

**Support:** See REORGANIZATION_GUIDE.md for detailed instructions
