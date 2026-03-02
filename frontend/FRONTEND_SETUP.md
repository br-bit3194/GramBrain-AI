# Frontend Setup & Development Guide

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js app directory
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Home page
│   │   ├── dashboard/          # Dashboard page
│   │   ├── farms/              # Farms management
│   │   ├── query/              # Query interface
│   │   ├── marketplace/        # Product marketplace
│   │   ├── login/              # Login page
│   │   ├── register/           # Registration page
│   │   └── profile/            # User profile
│   ├── components/             # Reusable components
│   │   ├── layout/             # Layout components (Header, Footer)
│   │   ├── cards/              # Card components (FarmCard, ProductCard)
│   │   ├── forms/              # Form components (QueryForm, FarmForm)
│   │   └── QueryInterface.tsx  # Main query interface
│   ├── hooks/                  # Custom React hooks
│   │   ├── useAuth.ts          # Authentication hook
│   │   ├── useFarm.ts          # Farm management hook
│   │   └── useQuery.ts         # Query processing hook
│   ├── services/               # API services
│   │   └── api.ts              # API client
│   ├── store/                  # State management
│   │   └── appStore.ts         # Zustand store
│   ├── types/                  # TypeScript types
│   │   └── index.ts            # Type definitions
│   └── styles/                 # Global styles
│       └── globals.css         # Tailwind CSS
├── package.json                # Dependencies
├── tsconfig.json               # TypeScript config
├── next.config.js              # Next.js config
├── tailwind.config.js          # Tailwind CSS config
└── jest.config.js              # Jest testing config
```

## Installation & Setup

### Prerequisites
- Node.js 18+ and npm/yarn
- Backend API running (see backend README)

### Install Dependencies
```bash
cd frontend
npm install
```

### Environment Setup
```bash
cp .env.example .env.local
```

Update `.env.local` with your backend API URL:
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

### Development Server
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint
- `npm test` - Run Jest tests
- `npm run test:watch` - Run tests in watch mode

## Key Features Implemented

### Pages
- **Home** (`/`) - Landing page with features overview
- **Dashboard** (`/dashboard`) - User dashboard with quick stats
- **Farms** (`/farms`) - Farm management interface
- **Query** (`/query`) - AI query interface
- **Marketplace** (`/marketplace`) - Product marketplace
- **Login** (`/login`) - User authentication
- **Register** (`/register`) - User registration
- **Profile** (`/profile`) - User profile management

### Components

#### Layout Components
- `Header` - Navigation header with user menu
- `Footer` - Footer with links
- `Layout` - Main layout wrapper

#### Card Components
- `FarmCard` - Display farm information
- `ProductCard` - Display product listings

#### Form Components
- `QueryForm` - Submit queries to AI agents
- `FarmForm` - Create/edit farm information

#### Other Components
- `QueryInterface` - Main query interface component

### Custom Hooks

#### useAuth
```typescript
const { user, isAuthenticated, login, register, logout } = useAuth()
```

#### useFarm
```typescript
const { farm, farms, loading, error, createFarm, getFarm, listUserFarms } = useFarm()
```

#### useQuery
```typescript
const { recommendation, recommendations, loading, error, processQuery, getRecommendation, listUserRecommendations } = useQuery()
```

## State Management

Using Zustand for global state:

```typescript
import { useAppStore } from '@/store/appStore'

const { user, farm, setUser, setFarm, clearStore } = useAppStore()
```

## API Integration

The API client is configured in `src/services/api.ts`:

```typescript
import { apiClient } from '@/services/api'

// User endpoints
await apiClient.createUser(userData)
await apiClient.getUser(userId)

// Farm endpoints
await apiClient.createFarm(farmData)
await apiClient.getFarm(farmId)
await apiClient.listUserFarms(userId)

// Query endpoints
await apiClient.processQuery(queryData)
await apiClient.getRecommendation(recommendationId)
await apiClient.listUserRecommendations(userId)

// Product endpoints
await apiClient.createProduct(productData)
await apiClient.searchProducts(filters)
```

## Styling

The frontend uses Tailwind CSS with custom configuration:

- Primary color: `#10b981` (emerald)
- Custom utilities in `globals.css`:
  - `.container-custom` - Max-width container
  - `.card` - Card styling
  - `.btn-primary` - Primary button
  - `.btn-outline` - Outline button
  - `.input-field` - Input styling

## Type Definitions

All TypeScript types are defined in `src/types/index.ts`:

- `User` - User information
- `Farm` - Farm details
- `CropCycle` - Crop cycle information
- `Recommendation` - AI recommendations
- `Product` - Marketplace products
- `QueryRequest` - Query parameters
- `ApiResponse` - API response format

## Testing

Jest is configured for testing. Create test files in `frontend/tests/`:

```bash
npm test
```

## Deployment

### Build for Production
```bash
npm run build
npm start
```

### Docker
```bash
docker build -t grambrain-frontend .
docker run -p 3000:3000 grambrain-frontend
```

## Troubleshooting

### API Connection Issues
- Ensure backend is running on the configured URL
- Check `NEXT_PUBLIC_API_URL` in `.env.local`
- Verify CORS settings on backend

### Build Errors
- Clear `.next` folder: `rm -rf .next`
- Reinstall dependencies: `rm -rf node_modules && npm install`
- Check TypeScript errors: `npm run lint`

### State Management Issues
- Verify Zustand store is properly initialized
- Check that `useAppStore` is called in client components only
- Use `'use client'` directive in components using hooks

## Next Steps

1. Implement authentication flow with backend
2. Add form validation and error handling
3. Create comprehensive test suite
4. Add loading states and error boundaries
5. Implement real-time notifications
6. Add image upload for products
7. Implement cart functionality
8. Add payment integration
