import axios from 'axios'

// Mock axios before importing apiClient
jest.mock('axios')
const mockedAxios = axios as jest.Mocked<typeof axios>

// Create a mock axios instance
const mockAxiosInstance = {
  interceptors: {
    request: {
      use: jest.fn(),
    },
    response: {
      use: jest.fn(),
    },
  },
  get: jest.fn(),
  post: jest.fn(),
  put: jest.fn(),
  delete: jest.fn(),
}

// Setup the mock before importing
mockedAxios.create.mockReturnValue(mockAxiosInstance as any)

// Now import apiClient after mocks are set up
import { apiClient, setTokenGetter } from '../api'

describe('API Client Configuration', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should create axios instance with correct base URL including /api prefix', () => {
    // The axios.create should have been called during module initialization
    expect(mockedAxios.create).toHaveBeenCalled()
    
    const createCall = mockedAxios.create.mock.calls[0][0]
    expect(createCall?.baseURL).toBeDefined()
    
    // Should either use env variable or default to localhost:8000/api
    const baseURL = createCall?.baseURL || ''
    expect(baseURL).toMatch(/\/api$/)
  })

  it('should configure axios with proper defaults', () => {
    const createCall = mockedAxios.create.mock.calls[0][0]
    
    // Check Content-Type header
    expect(createCall?.headers).toBeDefined()
    expect(createCall?.headers?.['Content-Type']).toBe('application/json')
    
    // Check timeout is set
    expect(createCall?.timeout).toBeDefined()
    expect(createCall?.timeout).toBeGreaterThan(0)
  })

  it('should have request and response interceptors configured', () => {
    // Verify interceptors were registered
    expect(mockAxiosInstance.interceptors.request.use).toHaveBeenCalled()
    expect(mockAxiosInstance.interceptors.response.use).toHaveBeenCalled()
  })
})

describe('Authentication Header Injection', () => {
  let requestInterceptor: any

  beforeEach(() => {
    jest.clearAllMocks()
    
    // Get the request interceptor that was registered
    const requestInterceptorCall = mockAxiosInstance.interceptors.request.use.mock.calls[0]
    requestInterceptor = requestInterceptorCall ? requestInterceptorCall[0] : null
  })

  it('should add Authorization header when token is available', () => {
    // Setup token getter to return a token
    const testToken = 'test-access-token-123'
    setTokenGetter(() => testToken)

    // Create a mock config
    const mockConfig = {
      method: 'get',
      url: '/test',
      headers: {},
    }

    // Call the interceptor
    const result = requestInterceptor(mockConfig)

    // Verify Authorization header was added
    expect(result.headers.Authorization).toBe(`Bearer ${testToken}`)
  })

  it('should not add Authorization header when token is not available', () => {
    // Setup token getter to return null
    setTokenGetter(() => null)

    // Create a mock config
    const mockConfig = {
      method: 'get',
      url: '/test',
      headers: {},
    }

    // Call the interceptor
    const result = requestInterceptor(mockConfig)

    // Verify Authorization header was not added
    expect(result.headers.Authorization).toBeUndefined()
  })

  it('should handle case when token getter is not set', () => {
    // Reset token getter
    setTokenGetter(null as any)

    // Create a mock config
    const mockConfig = {
      method: 'get',
      url: '/test',
      headers: {},
    }

    // Call the interceptor - should not throw
    expect(() => requestInterceptor(mockConfig)).not.toThrow()
    
    const result = requestInterceptor(mockConfig)
    
    // Verify Authorization header was not added
    expect(result.headers.Authorization).toBeUndefined()
  })
})

