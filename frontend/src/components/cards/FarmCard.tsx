import Link from 'next/link'
import { Farm } from '@/types'
import { FiMapPin, FiTrendingUp, FiDroplet } from 'react-icons/fi'

interface FarmCardProps {
  farm: Farm
  onSelect?: (farm: Farm) => void
}

export const FarmCard: React.FC<FarmCardProps> = ({ farm, onSelect }) => {
  return (
    <div className="card hover:shadow-lg transition-shadow cursor-pointer" onClick={() => onSelect?.(farm)}>
      <div className="flex items-start justify-between mb-4">
        <h3 className="text-lg font-bold">{farm.farm_id}</h3>
        <span className="px-3 py-1 bg-green-100 text-green-700 text-xs font-medium rounded-full">
          Active
        </span>
      </div>

      <div className="space-y-3 mb-4">
        <div className="flex items-center gap-2 text-gray-600">
          <FiMapPin className="text-primary" />
          <span className="text-sm">
            {farm.location.lat.toFixed(4)}, {farm.location.lon.toFixed(4)}
          </span>
        </div>

        <div className="flex items-center gap-2 text-gray-600">
          <FiTrendingUp className="text-primary" />
          <span className="text-sm">{farm.area_hectares} hectares</span>
        </div>

        <div className="flex items-center gap-2 text-gray-600">
          <FiDroplet className="text-primary" />
          <span className="text-sm capitalize">{farm.irrigation_type} irrigation</span>
        </div>
      </div>

      <div className="mb-4 p-3 bg-gray-50 rounded">
        <p className="text-xs text-gray-600 mb-1">Soil Type</p>
        <p className="font-semibold capitalize">{farm.soil_type}</p>
      </div>

      {farm.crops && farm.crops.length > 0 && (
        <div className="mb-4">
          <p className="text-xs text-gray-600 mb-2">Crops</p>
          <div className="flex flex-wrap gap-2">
            {farm.crops.map((crop) => (
              <span key={crop} className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                {crop}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="text-xs text-gray-500">
        Updated {new Date(farm.updated_at).toLocaleDateString()}
      </div>
    </div>
  )
}
