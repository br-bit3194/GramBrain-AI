import { useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { useAppStore } from '@/store/appStore'
import { apiClient } from '@/services/api'
import { User } from '@/types'

export const useAuth = () => {
  const router = useRouter()
  const { user, setUser, clearStore } = useAppStore()

  const login = useCallback(
    async (phoneNumber: string) => {
      try {
        const response = await apiClient.getUser(phoneNumber)
        if (response.status === 'success' && response.data?.user) {
          setUser(response.data.user)
          router.push('/dashboard')
          return { success: true }
        }
        return { success: false, error: 'User not found' }
      } catch (error: any) {
        return {
          success: false,
          error: error.response?.data?.detail || 'Login failed',
        }
      }
    },
    [setUser, router]
  )

  const register = useCallback(
    async (data: {
      name: string
      phoneNumber: string
      language: string
      role: string
    }) => {
      try {
        const response = await apiClient.createUser({
          name: data.name,
          phone_number: data.phoneNumber,
          language_preference: data.language,
          role: data.role,
        })

        if (response.status === 'success' && response.data?.user) {
          setUser(response.data.user)
          router.push('/dashboard')
          return { success: true }
        }
        return { success: false, error: 'Registration failed' }
      } catch (error: any) {
        return {
          success: false,
          error: error.response?.data?.detail || 'Registration failed',
        }
      }
    },
    [setUser, router]
  )

  const logout = useCallback(() => {
    clearStore()
    router.push('/')
  }, [clearStore, router])

  return {
    user,
    isAuthenticated: !!user,
    login,
    register,
    logout,
  }
}
