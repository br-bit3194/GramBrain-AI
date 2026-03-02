import { useState, useCallback } from 'react'
import { useAppStore } from '@/store/appStore'
import { apiClient } from '@/services/api'
import { Farm } from '@/types'

export const useFarm = () => {
  const { user, farm, setFarm } = useAppStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [farms, setFarms] = useState<Farm[]>([])

  const createFarm = useCallback(
    async (data: {
      latitude: number
      longitude: number
      area_hectares: number
      soil_type: string
      irrigation_type?: string
    }) => {
      if (!user) {
        setError('User not authenticated')
        return { success: false }
      }

      setLoading(true)
      setError('')

      try {
        const response = await apiClient.createFarm({
          owner_id: user.user_id,
          ...data,
        })

        if (response.status === 'success' && response.data?.farm) {
          setFarm(response.data.farm)
          return { success: true, farm: response.data.farm }
        }
        setError('Failed to create farm')
        return { success: false }
      } catch (err: any) {
        const errorMsg = err.response?.data?.detail || 'Failed to create farm'
        setError(errorMsg)
        return { success: false, error: errorMsg }
      } finally {
        setLoading(false)
      }
    },
    [user, setFarm]
  )

  const getFarm = useCallback(
    async (farmId: string) => {
      setLoading(true)
      setError('')

      try {
        const response = await apiClient.getFarm(farmId)

        if (response.status === 'success' && response.data?.farm) {
          setFarm(response.data.farm)
          return { success: true, farm: response.data.farm }
        }
        setError('Failed to fetch farm')
        return { success: false }
      } catch (err: any) {
        const errorMsg = err.response?.data?.detail || 'Failed to fetch farm'
        setError(errorMsg)
        return { success: false, error: errorMsg }
      } finally {
        setLoading(false)
      }
    },
    [setFarm]
  )

  const listUserFarms = useCallback(
    async (userId: string) => {
      setLoading(true)
      setError('')

      try {
        const response = await apiClient.listUserFarms(userId)

        if (response.status === 'success' && response.data?.farms) {
          setFarms(response.data.farms)
          return { success: true, farms: response.data.farms }
        }
        setError('Failed to fetch farms')
        return { success: false }
      } catch (err: any) {
        const errorMsg = err.response?.data?.detail || 'Failed to fetch farms'
        setError(errorMsg)
        return { success: false, error: errorMsg }
      } finally {
        setLoading(false)
      }
    },
    []
  )

  return {
    farm,
    farms,
    loading,
    error,
    createFarm,
    getFarm,
    listUserFarms,
  }
}
