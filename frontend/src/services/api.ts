import axios, { AxiosInstance } from 'axios'
import { ApiResponse, QueryRequest, Recommendation, Farm, User, Product } from '@/types'

class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: process.env.NEXT_PUBLIC_API_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    })
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
  }): Promise<ApiResponse<{ message: string }>> {
    const response = await this.client.post('/knowledge', data)
    return response.data
  }

  async searchKnowledge(query: string, topK: number = 5): Promise<ApiResponse<{ results: any[] }>> {
    const response = await this.client.get('/knowledge/search', {
      params: { query, top_k: topK },
    })
    return response.data
  }

  // Health check
  async healthCheck(): Promise<ApiResponse<{ status: string; agents: string[] }>> {
    const response = await this.client.get('/health')
    return response.data
  }
}

export const apiClient = new ApiClient()
