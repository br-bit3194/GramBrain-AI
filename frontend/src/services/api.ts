import axios, { AxiosInstance, AxiosError } from 'axios'
import { ApiResponse, QueryRequest, Recommendation, Farm, User, Product } from '@/types'
import { handleApiError, type UserFriendlyError } from '@/utils/errorHandler'

// Function to get token from store
// This is defined outside the class to avoid circular dependencies
let getAccessToken: (() => string | null) | null = null
let clearAuthStore: (() => void) | null = null

export function setTokenGetter(getter: () => string | null) {
  getAccessToken = getter
}

export function setClearAuthStore(clearer: () => void) {
  clearAuthStore = clearer
}

class ApiClient {
  private client: AxiosInstance

  constructor() {
    // Base URL should already include /api prefix from environment variable
    const baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'
    
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000, // 30 second timeout
    })

    // Request interceptor for authentication header injection
    this.client.interceptors.request.use(
      (config) => {
        // Try to get token from store
        const token = getAccessToken ? getAccessToken() : null
        
        // If token exists, add Authorization header
        if (token) {
          config.headers.Authorization = `Bearer ${token}`
        }
        
        // Log the request
        console.log(`[API Request] ${config.method?.toUpperCase()} ${config.url}`, {
          params: config.params,
          data: config.data,
          hasAuthHeader: !!config.headers.Authorization,
        })
        
        return config
      },
      (error) => {
        console.error('[API Request Error]', error)
        return Promise.reject(error)
      }
    )

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => {
        console.log(`[API Response] ${response.config.method?.toUpperCase()} ${response.config.url}`, {
          status: response.status,
          data: response.data,
        })
        return response
      },
      (error: AxiosError) => {
        // Log the error for debugging
        if (error.response) {
          console.error(`[API Error Response] ${error.config?.method?.toUpperCase()} ${error.config?.url}`, {
            status: error.response.status,
            data: error.response.data,
          })
          
          // Handle 401 Unauthorized - clear auth and redirect to login
          if (error.response.status === 401 && clearAuthStore) {
            console.log('[API] Unauthorized - clearing auth store')
            clearAuthStore()
          }
        } else if (error.request) {
          console.error('[API Network Error]', {
            message: error.message,
            url: error.config?.url,
          })
        } else {
          console.error('[API Error]', error.message)
        }
        
        // Transform error to user-friendly format and attach to error object
        (error as any).userFriendlyError = handleApiError(error)
        
        return Promise.reject(error)
      }
    )
  }

  // Auth endpoints
  async register(data: {
    phone_number: string
    name: string
    password: string
    language_preference?: string
    role?: string
  }): Promise<ApiResponse<{ user: User; access_token: string; refresh_token: string; token_type: string }>> {
    const response = await this.client.post('/auth/register', data)
    return response.data
  }

  async login(data: {
    phone_number: string
    password: string
  }): Promise<ApiResponse<{ user: User; access_token: string; refresh_token: string; token_type: string }>> {
    const response = await this.client.post('/auth/login', data)
    return response.data
  }

  async getCurrentUser(): Promise<ApiResponse<{ user: User }>> {
    const response = await this.client.get('/auth/me')
    return response.data
  }

  // User endpoints
  async createUser(data: {
    phone_number: string
    name: string
    language_preference?: string
    role?: string
  }): Promise<ApiResponse<{ user: User }>> {
    const response = await this.client.post('/users', data)
    return response.data
  }

  async getUser(userId: string): Promise<ApiResponse<{ user: User }>> {
    const response = await this.client.get(`/users/${userId}`)
    return response.data
  }

  // Farm endpoints
  async createFarm(data: {
    owner_id: string
    latitude: number
    longitude: number
    area_hectares: number
    soil_type: string
    irrigation_type?: string
  }): Promise<ApiResponse<{ farm: Farm }>> {
    const response = await this.client.post('/farms', data)
    return response.data
  }

  async getFarm(farmId: string): Promise<ApiResponse<{ farm: Farm }>> {
    const response = await this.client.get(`/farms/${farmId}`)
    return response.data
  }

  async listUserFarms(userId: string): Promise<ApiResponse<{ farms: Farm[] }>> {
    const response = await this.client.get(`/users/${userId}/farms`)
    return response.data
  }

  // Query endpoints
  async processQuery(data: QueryRequest): Promise<ApiResponse<{ recommendation: Recommendation }>> {
    const response = await this.client.post('/query', data)
    return response.data
  }

  async getRecommendation(recommendationId: string): Promise<ApiResponse<{ recommendation: Recommendation }>> {
    const response = await this.client.get(`/recommendations/${recommendationId}`)
    return response.data
  }

  async listUserRecommendations(userId: string, limit: number = 10): Promise<ApiResponse<{ recommendations: Recommendation[] }>> {
    const response = await this.client.get(`/users/${userId}/recommendations`, {
      params: { limit },
    })
    return response.data
  }

  // Product endpoints
  async createProduct(data: {
    farmer_id: string
    farm_id: string
    product_type: string
    name: string
    quantity_kg: number
    price_per_kg: number
    harvest_date: string
  }): Promise<ApiResponse<{ product: Product }>> {
    const response = await this.client.post('/products', data)
    return response.data
  }

  async getProduct(productId: string): Promise<ApiResponse<{ product: Product }>> {
    const response = await this.client.get(`/products/${productId}`)
    return response.data
  }

  async searchProducts(filters?: {
    product_type?: string
    min_score?: number
    max_price?: number
    limit?: number
  }): Promise<ApiResponse<{ products: Product[] }>> {
    const response = await this.client.get('/products', { params: filters })
    return response.data
  }

  async listFarmerProducts(farmerId: string): Promise<ApiResponse<{ products: Product[] }>> {
    const response = await this.client.get(`/farmers/${farmerId}/products`)
    return response.data
  }

  // Knowledge endpoints
  async addKnowledge(data: {
    chunk_id: string
    content: string
    source: string
    topic: string
    crop_type?: string
    region?: string
  }): Promise<ApiResponse<{ message: string; chunk_id: string }>> {
    const response = await this.client.post('/knowledge', data)
    return response.data
  }

  async searchKnowledge(
    query: string, 
    topK: number = 5,
    filters?: {
      crop_type?: string
      region?: string
    }
  ): Promise<ApiResponse<{ results: any[]; count: number }>> {
    const response = await this.client.get('/knowledge/search', {
      params: { 
        query, 
        top_k: topK,
        ...filters
      },
    })
    return response.data
  }

  async addBulkKnowledge(knowledgeItems: Array<{
    chunk_id: string
    content: string
    source: string
    topic: string
    crop_type?: string
    region?: string
  }>): Promise<ApiResponse<{ added: number; errors: any[]; total: number }>> {
    const response = await this.client.post('/knowledge/bulk', knowledgeItems)
    return response.data
  }

  // Health check
  async healthCheck(): Promise<ApiResponse<{ status: string; timestamp: string; agents: string[] }>> {
    const response = await this.client.get('/health')
    return response.data
  }
}

export const apiClient = new ApiClient()
