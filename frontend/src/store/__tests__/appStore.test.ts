import { renderHook, act } from '@testing-library/react'
import { useAppStore } from '../appStore'

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {}

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString()
    },
    removeItem: (key: string) => {
      delete store[key]
    },
    clear: () => {
      store = {}
    },
  }
})()

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
})

describe('AppStore - Auth State Management', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorageMock.clear()
    
    // Reset store state
    const { result } = renderHook(() => useAppStore())
    act(() => {
      result.current.clearStore()
    })
  })

  describe('Token Management', () => {
    it('should store access and refresh tokens', () => {
      const { result } = renderHook(() => useAppStore())

      act(() => {
        result.current.setTokens('access-token-123', 'refresh-token-456')
      })

      expect(result.current.accessToken).toBe('access-token-123')
      expect(result.current.refreshToken).toBe('refresh-token-456')
    })

    it('should persist tokens to localStorage', () => {
      const { result } = renderHook(() => useAppStore())

      act(() => {
        result.current.setTokens('access-token-123', 'refresh-token-456')
      })

      // Check that localStorage was updated
      const stored = localStorageMock.getItem('grambrain-storage')
      expect(stored).toBeTruthy()
      
      const parsed = JSON.parse(stored!)
      expect(parsed.state.accessToken).toBe('access-token-123')
      expect(parsed.state.refreshToken).toBe('refresh-token-456')
    })

    it('should restore tokens from localStorage on initialization', () => {
      // Pre-populate localStorage
      const storeData = {
        state: {
          user: null,
          farm: null,
          accessToken: 'stored-access-token',
          refreshToken: 'stored-refresh-token',
        },
        version: 0,
      }
      localStorageMock.setItem('grambrain-storage', JSON.stringify(storeData))

      // Create a new hook instance (simulating app restart)
      const { result } = renderHook(() => useAppStore())

      // Tokens should be restored
      expect(result.current.accessToken).toBe('stored-access-token')
      expect(result.current.refreshToken).toBe('stored-refresh-token')
    })
  })

  describe('Logout Action', () => {
    it('should clear all auth data from store', () => {
      const { result } = renderHook(() => useAppStore())

      // Set up some data
      act(() => {
        result.current.setUser({
          user_id: 'user-123',
          phone_number: '+1234567890',
          name: 'Test User',
          language_preference: 'en',
          role: 'farmer',
          created_at: '2024-01-01',
          last_active: '2024-01-01',
        })
        result.current.setTokens('access-token', 'refresh-token')
        result.current.setFarm({
          farm_id: 'farm-123',
          owner_id: 'user-123',
          location: { lat: 0, lon: 0 },
          area_hectares: 10,
          soil_type: 'loam',
          irrigation_type: 'drip',
          crops: ['wheat'],
          created_at: '2024-01-01',
          updated_at: '2024-01-01',
        })
      })

      // Verify data is set
      expect(result.current.user).not.toBeNull()
      expect(result.current.accessToken).not.toBeNull()
      expect(result.current.refreshToken).not.toBeNull()
      expect(result.current.farm).not.toBeNull()

      // Call logout
      act(() => {
        result.current.logout()
      })

      // Verify all data is cleared
      expect(result.current.user).toBeNull()
      expect(result.current.accessToken).toBeNull()
      expect(result.current.refreshToken).toBeNull()
      expect(result.current.farm).toBeNull()
    })

    it('should clear auth data from localStorage', () => {
      const { result } = renderHook(() => useAppStore())

      // Set up some data
      act(() => {
        result.current.setUser({
          user_id: 'user-123',
          phone_number: '+1234567890',
          name: 'Test User',
          language_preference: 'en',
          role: 'farmer',
          created_at: '2024-01-01',
          last_active: '2024-01-01',
        })
        result.current.setTokens('access-token', 'refresh-token')
      })

      // Verify data is in localStorage
      let stored = localStorageMock.getItem('grambrain-storage')
      expect(stored).toBeTruthy()
      let parsed = JSON.parse(stored!)
      expect(parsed.state.accessToken).toBe('access-token')

      // Call logout
      act(() => {
        result.current.logout()
      })

      // Verify localStorage is updated with null values
      stored = localStorageMock.getItem('grambrain-storage')
      expect(stored).toBeTruthy()
      parsed = JSON.parse(stored!)
      expect(parsed.state.accessToken).toBeNull()
      expect(parsed.state.refreshToken).toBeNull()
      expect(parsed.state.user).toBeNull()
    })
  })

  describe('User State Management', () => {
    it('should set and persist user data', () => {
      const { result } = renderHook(() => useAppStore())

      const testUser = {
        user_id: 'user-123',
        phone_number: '+1234567890',
        name: 'Test User',
        language_preference: 'en',
        role: 'farmer' as const,
        created_at: '2024-01-01',
        last_active: '2024-01-01',
      }

      act(() => {
        result.current.setUser(testUser)
      })

      expect(result.current.user).toEqual(testUser)

      // Check localStorage
      const stored = localStorageMock.getItem('grambrain-storage')
      expect(stored).toBeTruthy()
      const parsed = JSON.parse(stored!)
      expect(parsed.state.user).toEqual(testUser)
    })

    it('should clear user data when set to null', () => {
      const { result } = renderHook(() => useAppStore())

      // Set user first
      act(() => {
        result.current.setUser({
          user_id: 'user-123',
          phone_number: '+1234567890',
          name: 'Test User',
          language_preference: 'en',
          role: 'farmer',
          created_at: '2024-01-01',
          last_active: '2024-01-01',
        })
      })

      expect(result.current.user).not.toBeNull()

      // Clear user
      act(() => {
        result.current.setUser(null)
      })

      expect(result.current.user).toBeNull()
    })
  })

  describe('Farm State Management', () => {
    it('should set and persist farm data', () => {
      const { result } = renderHook(() => useAppStore())

      const testFarm = {
        farm_id: 'farm-123',
        owner_id: 'user-123',
        location: { lat: 12.34, lon: 56.78 },
        area_hectares: 10,
        soil_type: 'loam',
        irrigation_type: 'drip' as const,
        crops: ['wheat', 'rice'],
        created_at: '2024-01-01',
        updated_at: '2024-01-01',
      }

      act(() => {
        result.current.setFarm(testFarm)
      })

      expect(result.current.farm).toEqual(testFarm)

      // Check localStorage
      const stored = localStorageMock.getItem('grambrain-storage')
      expect(stored).toBeTruthy()
      const parsed = JSON.parse(stored!)
      expect(parsed.state.farm).toEqual(testFarm)
    })
  })

  describe('clearStore Action', () => {
    it('should clear all store data', () => {
      const { result } = renderHook(() => useAppStore())

      // Set up data
      act(() => {
        result.current.setUser({
          user_id: 'user-123',
          phone_number: '+1234567890',
          name: 'Test User',
          language_preference: 'en',
          role: 'farmer',
          created_at: '2024-01-01',
          last_active: '2024-01-01',
        })
        result.current.setTokens('access-token', 'refresh-token')
        result.current.setFarm({
          farm_id: 'farm-123',
          owner_id: 'user-123',
          location: { lat: 0, lon: 0 },
          area_hectares: 10,
          soil_type: 'loam',
          irrigation_type: 'drip',
          crops: ['wheat'],
          created_at: '2024-01-01',
          updated_at: '2024-01-01',
        })
      })

      // Clear store
      act(() => {
        result.current.clearStore()
      })

      // Verify all data is cleared
      expect(result.current.user).toBeNull()
      expect(result.current.accessToken).toBeNull()
      expect(result.current.refreshToken).toBeNull()
      expect(result.current.farm).toBeNull()
    })
  })
})
