'use client'

import { useState } from 'react'
import { QueryRequest } from '@/types'
import { FiMessageSquare, FiAlertCircle } from 'react-icons/fi'

interface QueryFormProps {
  userId: string
  farmId?: string
  onSubmit: (query: QueryRequest) => Promise<void>
  loading?: boolean
}

export const QueryForm: React.FC<QueryFormProps> = ({ userId, farmId, onSubmit, loading = false }) => {
  const [queryText, setQueryText] = useState('')
  const [cropType, setCropType] = useState('')
  const [growthStage, setGrowthStage] = useState('')
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    try {
      if (!queryText.trim()) {
        setError('Please enter a query')
        return
      }

      await onSubmit({
        user_id: userId,
        query_text: queryText,
        farm_id: farmId,
        crop_type: cropType || undefined,
        growth_stage: growthStage || undefined,
        language: 'en',
      })

      setQueryText('')
      setCropType('')
      setGrowthStage('')
    } catch (err: any) {
      setError(err.message || 'Failed to process query')
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
          <FiAlertCircle className="text-red-600 mt-0.5 flex-shrink-0" />
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      <div>
        <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
          <FiMessageSquare className="inline mr-2" />
          Your Question
        </label>
        <textarea
          id="query"
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          placeholder="Ask GramBrain AI anything about your farm..."
          rows={4}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none resize-none"
          disabled={loading}
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label htmlFor="crop" className="block text-sm font-medium text-gray-700 mb-2">
            Crop Type (Optional)
          </label>
          <select
            id="crop"
            value={cropType}
            onChange={(e) => setCropType(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
            disabled={loading}
          >
            <option value="">Select crop...</option>
            <option value="wheat">Wheat</option>
            <option value="rice">Rice</option>
            <option value="corn">Corn</option>
            <option value="cotton">Cotton</option>
            <option value="sugarcane">Sugarcane</option>
            <option value="vegetables">Vegetables</option>
          </select>
        </div>

        <div>
          <label htmlFor="stage" className="block text-sm font-medium text-gray-700 mb-2">
            Growth Stage (Optional)
          </label>
          <select
            id="stage"
            value={growthStage}
            onChange={(e) => setGrowthStage(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
            disabled={loading}
          >
            <option value="">Select stage...</option>
            <option value="germination">Germination</option>
            <option value="vegetative">Vegetative</option>
            <option value="flowering">Flowering</option>
            <option value="fruiting">Fruiting</option>
            <option value="maturity">Maturity</option>
          </select>
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-primary text-white py-2 rounded-lg font-medium hover:bg-primary-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Processing...' : 'Get Recommendation'}
      </button>
    </form>
  )
}
