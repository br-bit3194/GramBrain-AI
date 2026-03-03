// User types
export interface User {
  user_id: string
  phone_number: string
  name: string
  language_preference: string
  role: 'farmer' | 'village_leader' | 'policymaker' | 'consumer'
  created_at: string
  last_active: string
}

// Farm types
export interface Farm {
  farm_id: string
  owner_id: string
  location: {
    lat: number
    lon: number
  }
  area_hectares: number
  soil_type: string
  irrigation_type: 'drip' | 'flood' | 'sprinkler' | 'rainfed'
  crops: string[]
  created_at: string
  updated_at: string
}

// Crop types
export interface CropCycle {
  cycle_id: string
  farm_id: string
  crop_type: string
  variety: string
  planting_date: string
  expected_harvest_date: string
  actual_harvest_date?: string
  growth_stage: string
  area_hectares: number
  yield_predicted?: number
  yield_actual?: number
}

// Recommendation types
export interface Recommendation {
  recommendation_id: string
  query_id: string
  user_id: string
  farm_id?: string
  timestamp: string
  recommendation_text: string
  reasoning_chain: string[]
  confidence: number
  agent_contributions: string[]
  language: string
}

// Product types
export interface Product {
  product_id: string
  farmer_id: string
  farm_id: string
  product_type: 'vegetables' | 'grains' | 'pulses' | 'dairy' | 'honey' | 'spices'
  name: string
  quantity_kg: number
  price_per_kg: number
  harvest_date: string
  images: string[]
  pure_product_score: number
  status: 'available' | 'reserved' | 'sold'
  created_at: string
}

// API Response types
export interface ApiResponse<T> {
  status: 'success' | 'error'
  data?: T
  message?: string
  detail?: string
}

// Query types
export interface QueryRequest {
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

// Store types
export interface AppStore {
  user: User | null
  farm: Farm | null
  accessToken: string | null
  refreshToken: string | null
  setUser: (user: User | null) => void
  setFarm: (farm: Farm | null) => void
  setTokens: (accessToken: string, refreshToken: string) => void
  clearStore: () => void
  logout: () => void
}

// Export error types
export * from './errors'
