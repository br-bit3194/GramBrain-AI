# GramBrain AI - Frontend

React/Next.js frontend for the GramBrain AI agricultural intelligence platform.

## Status

🚧 **To be created** - Frontend structure ready for development

## Planned Tech Stack

- **Framework:** Next.js 13+ (React 18)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **State Management:** Redux Toolkit or Zustand
- **API Client:** Axios or Fetch API
- **Testing:** Jest + React Testing Library
- **UI Components:** Shadcn/ui or Material-UI

## Planned Features

### User Interface
- [ ] Farmer Dashboard
- [ ] Query Interface (Chat-like)
- [ ] Recommendation Display
- [ ] Farm Management
- [ ] Marketplace Browse
- [ ] Product Listings
- [ ] Village Dashboard
- [ ] Analytics Dashboard

### Pages
- [ ] Home/Landing
- [ ] Login/Register
- [ ] Dashboard
- [ ] Query/Chat
- [ ] Recommendations
- [ ] Farms
- [ ] Marketplace
- [ ] Profile
- [ ] Settings

### Components
- [ ] Navigation Bar
- [ ] Query Input
- [ ] Recommendation Card
- [ ] Farm Card
- [ ] Product Card
- [ ] Charts/Analytics
- [ ] Forms
- [ ] Modals

## Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable React components
│   │   ├── common/          # Common components
│   │   ├── layout/          # Layout components
│   │   ├── forms/           # Form components
│   │   └── cards/           # Card components
│   │
│   ├── pages/               # Next.js pages
│   │   ├── index.tsx        # Home page
│   │   ├── dashboard.tsx    # Dashboard
│   │   ├── query.tsx        # Query interface
│   │   ├── farms.tsx        # Farm management
│   │   ├── marketplace.tsx  # Marketplace
│   │   └── profile.tsx      # User profile
│   │
│   ├── hooks/               # Custom React hooks
│   │   ├── useQuery.ts      # Query hook
│   │   ├── useFarm.ts       # Farm hook
│   │   └── useAuth.ts       # Auth hook
│   │
│   ├── services/            # API services
│   │   ├── api.ts           # API client
│   │   ├── queryService.ts  # Query API
│   │   ├── farmService.ts   # Farm API
│   │   └── productService.ts # Product API
│   │
│   ├── styles/              # Global styles
│   │   ├── globals.css
│   │   └── variables.css
│   │
│   ├── utils/               # Utility functions
│   │   ├── constants.ts
│   │   ├── helpers.ts
│   │   └── validators.ts
│   │
│   ├── types/               # TypeScript types
│   │   ├── index.ts
│   │   ├── api.ts
│   │   └── models.ts
│   │
│   └── app.tsx              # Root component
│
├── public/                  # Static assets
│   ├── images/
│   ├── icons/
│   └── fonts/
│
├── tests/                   # Test files
│   ├── components/
│   ├── hooks/
│   ├── services/
│   └── utils/
│
├── .env.example             # Environment template
├── .env.local               # Local environment (git ignored)
├── package.json             # NPM dependencies
├── next.config.js           # Next.js configuration
├── tsconfig.json            # TypeScript configuration
├── tailwind.config.js       # Tailwind CSS configuration
├── jest.config.js           # Jest configuration
└── README.md                # This file
```

## Getting Started

### Prerequisites
- Node.js 16+ 
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
npm start
```

### Testing

```bash
npm test
npm run test:coverage
```

## Environment Variables

Create `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_APP_NAME=GramBrain AI
NEXT_PUBLIC_APP_VERSION=0.1.0
```

## API Integration

The frontend connects to the backend API at `http://localhost:8000/api/v1`

### Key Endpoints

- `POST /query` - Get recommendation
- `POST /users` - Create user
- `POST /farms` - Create farm
- `POST /products` - Create product
- `GET /products` - Search products
- And 15+ more endpoints

See [docs/API.md](../docs/API.md) for complete reference.

## Component Examples

### Query Component
```tsx
import { useQuery } from '@/hooks/useQuery';

export function QueryInterface() {
  const { query, loading, error } = useQuery();
  
  return (
    <div>
      <input type="text" placeholder="Ask about your farm..." />
      <button onClick={() => query(text)}>Get Recommendation</button>
    </div>
  );
}
```

### Farm Card Component
```tsx
import { Farm } from '@/types';

interface FarmCardProps {
  farm: Farm;
}

export function FarmCard({ farm }: FarmCardProps) {
  return (
    <div className="card">
      <h3>{farm.name}</h3>
      <p>Size: {farm.area_hectares} hectares</p>
      <p>Soil: {farm.soil_type}</p>
    </div>
  );
}
```

## Styling

Using Tailwind CSS for styling:

```tsx
<div className="flex items-center justify-between p-4 bg-white rounded-lg shadow">
  <h2 className="text-2xl font-bold">Dashboard</h2>
  <button className="px-4 py-2 bg-blue-500 text-white rounded">
    New Farm
  </button>
</div>
```

## State Management

Recommended: Redux Toolkit or Zustand

```tsx
// Example with Zustand
import create from 'zustand';

interface AppStore {
  user: User | null;
  setUser: (user: User) => void;
}

export const useAppStore = create<AppStore>((set) => ({
  user: null,
  setUser: (user) => set({ user }),
}));
```

## Testing

Using Jest and React Testing Library:

```tsx
import { render, screen } from '@testing-library/react';
import { QueryInterface } from '@/components/QueryInterface';

describe('QueryInterface', () => {
  it('renders query input', () => {
    render(<QueryInterface />);
    expect(screen.getByPlaceholderText(/ask about/i)).toBeInTheDocument();
  });
});
```

## Deployment

### Vercel (Recommended)

```bash
npm install -g vercel
vercel
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Performance Optimization

- [ ] Image optimization with Next.js Image
- [ ] Code splitting and lazy loading
- [ ] Caching strategies
- [ ] API response caching
- [ ] Bundle size optimization

## Accessibility

- [ ] WCAG 2.1 AA compliance
- [ ] Keyboard navigation
- [ ] Screen reader support
- [ ] Color contrast
- [ ] ARIA labels

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Contributing

1. Create feature branch
2. Make changes
3. Run tests
4. Submit pull request

## Documentation

- [Backend API](../docs/API.md)
- [System Design](../docs/design.md)
- [Testing Guide](../docs/TESTING.md)

## Support

- Issues: GitHub Issues
- Email: support@grambrain.ai

## License

MIT License - See LICENSE file for details

---

**Status:** 🚧 To be created  
**Next Steps:** Set up Next.js project and create components
