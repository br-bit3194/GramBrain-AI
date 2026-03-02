'use client'

import { useState } from 'react'
import { apiClient } from '@/services/api'
import { useAppStore } from '@/store/appStore'
import { Recommendation } from '@/types'
import { FiSend, FiLoader } from 'react-icons/fi'

export function QueryInterface() {
  const { user, farm } = useAppStore()
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || !user || !farm) return

    setLoading(true)
    setError(null)

    try {
      const response = await apiClient.processQuery({
        user_id: user.user_id,
        query_text: query,
        farm_id: farm.farm_id,
        latitude: farm.location.lat,
        longitude: farm.location.lon,
        farm_size_hectares: farm.area_hectares,
        crop_type: farm.crops[0],
        soil_type: farm.soil_type,
        language: user.language_preference,
      })

      if (response.status === 'success' && response.data) {
        setRecommendation(response.data.recommendation)
        setQuery('')
      } else {
        setError(response.detail || 'Failed to get recommendation')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  if (!user || !farm) {
    return (
      <div className="card text-center">
        <p className="text-gray-600">Please login and select a farm to ask questions.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Query Form */}
      <div className="card">
        <h2 className="text-2xl font-bold mb-4">Ask GramBrain AI</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Your Question
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., Should I irrigate my wheat field today?"
              className="input-field h-24 resize-none"
              disabled={loading}
            />
          </div>

          <div className="flex gap-4">
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <FiLoader className="animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <FiSend />
                  Get Recommendation
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-error">
              {error}
            </div>
          )}
        </form>
      </div>

      {/* Recommendation Display */}
      {recommendation && (
        <div className="card">
          <h3 className="text-xl font-bold mb-4">Recommendation</h3>

          <div className="space-y-4">
            {/* Main Recommendation */}
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-lg text-gray-800">{recommendation.recommendation_text}</p>
            </div>

            {/* Confidence */}
            <div className="flex items-center gap-4">
              <span className="text-sm font-medium text-gray-600">Confidence:</span>
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div
                  className="bg-primary h-2 rounded-full transition-all"
                  style={{ width: `${recommendation.confidence * 100}%` }}
                />
              </div>
              <span className="text-sm font-medium text-gray-600">
                {(recommendation.confidence * 100).toFixed(0)}%
              </span>
            </div>

            {/* Reasoning Chain */}
            {recommendation.reasoning_chain.length > 0 && (
              <div>
                <h4 className="font-semibold text-gray-700 mb-2">How we arrived at this:</h4>
                <ol className="space-y-2">
                  {recommendation.reasoning_chain.map((step, index) => (
                    <li key={index} className="flex gap-3">
                      <span className="flex-shrink-0 w-6 h-6 bg-primary text-white rounded-full flex items-center justify-center text-sm font-bold">
                        {index + 1}
                      </span>
                      <span className="text-gray-700">{step}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {/* Agent Contributions */}
            {recommendation.agent_contributions.length > 0 && (
              <div>
                <h4 className="font-semibold text-gray-700 mb-2">Agents involved:</h4>
                <div className="flex flex-wrap gap-2">
                  {recommendation.agent_contributions.map((agent, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm"
                    >
                      {agent}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
