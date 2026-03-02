'use client'

import { useState } from 'react'
import { FiMapPin, FiBox, FiDroplet, FiAlertCircle } from 'react-icons/fi'

interface FarmFormProps {
  onSubmit: (data: {
    latitude: number
    longitude: number
    area_hectares: number
    soil_type: string
    irrigation_type: string
  }) => Promise<void>
  loading?: boolean
}

export const FarmForm: React.FC<FarmFormProps> = ({ onSubmit, loading = false }) => {
  const [formData, setFormData] = useState({
    latitude: '',
    longitude: '',
    area_hectares: '',
    soil_type: 'loamy',
    irrigation_type: 'drip',
  })
  const [error, setError] = useState('')

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    try {
      if (!formData.latitude || !formData.longitude || !formData.area_hectares) {
        setError('All fields are required')
        return
      }

      await onSubmit({
        latitude: parseFloat(formData.latitude),
        longitude: parseFloat(formData.longitude),
        area_hectares: parseFloat(formData.area_hectares),
        soil_type: formData.soil_type,
        irrigation_type: formData.irrigation_type,
      })
    } catch (err: any) {
      setError(err.message || 'Failed to create farm')
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

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label htmlFor="latitude" className="block text-sm font-medium text-gray-700 mb-2">
            <FiMapPin className="inline mr-2" />
            Latitude
          </label>
          <input
            id="latitude"
            type="number"
            name="latitude"
            step="0.0001"
            value={formData.latitude}
            onChange={handleChange}
            placeholder="e.g., 28.7041"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
            disabled={loading}
          />
        </div>

        <div>
          <label htmlFor="longitude" className="block text-sm font-medium text-gray-700 mb-2">
            <FiMapPin className="inline mr-2" />
            Longitude
          </label>
          <input
            id="longitude"
            type="number"
            name="longitude"
            step="0.0001"
            value={formData.longitude}
            onChange={handleChange}
            placeholder="e.g., 77.1025"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
            disabled={loading}
          />
        </div>
      </div>

      <div>
        <label htmlFor="area" className="block text-sm font-medium text-gray-700 mb-2">
          <FiBox className="inline mr-2" />
          Area (Hectares)
        </label>
        <input
          id="area"
          type="number"
          name="area_hectares"
          step="0.1"
          value={formData.area_hectares}
          onChange={handleChange}
          placeholder="e.g., 5.5"
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
          disabled={loading}
        />
      </div>

      <div>
        <label htmlFor="soil" className="block text-sm font-medium text-gray-700 mb-2">
          Soil Type
        </label>
        <select
          id="soil"
          name="soil_type"
          value={formData.soil_type}
          onChange={handleChange}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
          disabled={loading}
        >
          <option value="loamy">Loamy</option>
          <option value="sandy">Sandy</option>
          <option value="clay">Clay</option>
          <option value="silty">Silty</option>
          <option value="peaty">Peaty</option>
        </select>
      </div>

      <div>
        <label htmlFor="irrigation" className="block text-sm font-medium text-gray-700 mb-2">
          <FiDroplet className="inline mr-2" />
          Irrigation Type
        </label>
        <select
          id="irrigation"
          name="irrigation_type"
          value={formData.irrigation_type}
          onChange={handleChange}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
          disabled={loading}
        >
          <option value="drip">Drip</option>
          <option value="flood">Flood</option>
          <option value="sprinkler">Sprinkler</option>
          <option value="rainfed">Rainfed</option>
        </select>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-primary text-white py-2 rounded-lg font-medium hover:bg-primary-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Creating Farm...' : 'Create Farm'}
      </button>
    </form>
  )
}
