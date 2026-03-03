import { useState, useEffect } from 'react'
import { useAppStore } from '@/store/appStore'
import { apiClient, setTokenGetter, setClearAuthStore } from '@/services/api'
import { User } from '@/types'

export function useAuth() {
  const { user, accessToken, refreshToken, setUser, setTokens, clearStore } = useAppStore()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Initialize token getter and clear function for API client
  useEffect(() => {
    setTokenGetter(() => accessToken)
    setClearAuthStore(() => clearStore)
  }, [accessToken, clearStore])

  const register = async (data: {
    phone_number: string
    name: string
    password: string
    language_preference?: string
    role?: string
  }) => {
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await apiClient.register(data)
      
      if (response.status === 'success' && response.data) {
        const { user, access_token, refresh_token } = response.data
        setUser(user)
        setTokens(access_token, refresh_token)
        
        // Store tokens in localStorage for persistence
        localStorage.setItem('accessToken', access_token)
        localStorage.setItem('refreshToken', refresh_token)
        localStorage.setItem('user', JSON.stringify(user))
        
        return { success: true, user }
      } else {
        throw new Error(response.detail || 'Registration failed')
      }
    } catch (err: any) {
      const errorMessage = err.userFriendlyError?.message || err.message || 'Registration failed'
      setError(errorMessage)
      return { success: false, error: errorMessage }
    } finally {
      setIsLoading(false)
    }
  }

  const login = async (data: {
    phone_number: string
    password: string
  }) => {
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await apiClient.login(data)
      
      if (response.status === 'success' && response.data) {
        const { user, access_token, refresh_token } = response.data
        setUser(user)
        setTokens(access_token, refresh_token)
        
        // Store tokens in localStorage for persistence
        localStorage.setItem('accessToken', access_token)
        localStorage.setItem('refreshToken', refresh_token)
        localStorage.setItem('user', JSON.stringify(user))
        
        return { success: true, user }
      } else {
        throw new Error(response.detail || 'Login failed')
      }
    } catch (err: any) {
      const errorMessage = err.userFriendlyError?.message || err.message || 'Login failed'
      setError(errorMessage)
      return { success: false, error: errorMessage }
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => {
    clearStore()
    localStorage.removeItem('accessToken')
    localStorage.removeItem('refreshToken')
    localStorage.removeItem('user')
  }

  const restoreSession = () => {
    const storedAccessToken = localStorage.getItem('accessToken')
    const storedRefreshToken = localStorage.getItem('refreshToken')
    const storedUser = localStorage.getItem('user')
    
    if (storedAccessToken && storedRefreshToken && storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser) as User
        setUser(parsedUser)
        setTokens(storedAccessToken, storedRefreshToken)
        return true
      } catch (err) {
        console.error('Failed to restore session:', err)
        logout()
        return false
      }
    }
    return false
  }

  const getCurrentUser = async () => {
    if (!accessToken) {
      return { success: false, error: 'Not authenticated' }
    }
    
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await apiClient.getCurrentUser()
      
      if (response.status === 'success' && response.data) {
        setUser(response.data.user)
        localStorage.setItem('user', JSON.stringify(response.data.user))
        return { success: true, user: response.data.user }
      } else {
        throw new Error(response.detail || 'Failed to get user')
      }
    } catch (err: any) {
      const errorMessage = err.userFriendlyError?.message || err.message || 'Failed to get user'
      setError(errorMessage)
      
      // If unauthorized, clear auth
      if (err.response?.status === 401) {
        logout()
      }
      
      return { success: false, error: errorMessage }
    } finally {
      setIsLoading(false)
    }
  }

  return {
    user,
    accessToken,
    refreshToken,
    isAuthenticated: !!accessToken && !!user,
    isLoading,
    error,
    register,
    login,
    logout,
    restoreSession,
    getCurrentUser,
  }
}
