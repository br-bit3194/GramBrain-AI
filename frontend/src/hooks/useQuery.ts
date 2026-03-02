import { useState, useCallback } from 'react'
import { apiClient } from '@/services/api'
import { QueryRequest, Recommendation } from '@/types'

export const useQuery = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null)
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])

  const processQuery = useCallback(async (query: QueryRequest) => {
    setLoading(true)
    setError('')

    try {
      const response = await apiClient.processQuery(query)

      if (response.status === 'success' && response.data?.recommendation) {
        setRecommendation(response.data.recommendation)
        return { success: true, recommendation: response.data.recommendation }
      }
      setError('Failed to process query')
      return { success: false }
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to process query'
      setError(errorMsg)
      return { success: false, error: errorMsg }
    } finally {
      setLoading(false)
    }
  }, [])

  const getRecommendation = useCallback(async (recommendationId: string) => {
    setLoading(true)
    setError('')

    try {
      const response = await apiClient.getRecommendation(recommendationId)

      if (response.status === 'success' && response.data?.recommendation) {
        setRecommendation(response.data.recommendation)
        return { success: true, recommendation: response.data.recommendation }
      }
      setError('Failed to fetch recommendation')
      return { success: false }
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to fetch recommendation'
      setError(errorMsg)
      return { success: false, error: errorMsg }
    } finally {
      setLoading(false)
    }
  }, [])

  const listUserRecommendations = useCallback(async (userId: string, limit: number = 10) => {
    setLoading(true)
    setError('')

    try {
      const response = await apiClient.listUserRecommendations(userId, limit)

      if (response.status === 'success' && response.data?.recommendations) {
        setRecommendations(response.data.recommendations)
        return { success: true, recommendations: response.data.recommendations }
      }
      setError('Failed to fetch recommendations')
      return { success: false }
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || 'Failed to fetch recommendations'
      setError(errorMsg)
      return { success: false, error: errorMsg }
    } finally {
      setLoading(false)
    }
  }, [])

  return {
    recommendation,
    recommendations,
    loading,
    error,
    processQuery,
    getRecommendation,
    listUserRecommendations,
  }
}
