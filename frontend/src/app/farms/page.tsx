'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { useAppStore } from '@/store/appStore'
import { useFarm } from '@/hooks/useFarm'
import { FarmForm } from '@/components/forms/FarmForm'
import { FarmCard } from '@/components/cards/FarmCard'
import { FiPlus, FiAlertCircle } from 'react-icons/fi'

export default function FarmsPage() {
  const { user, farm: selectedFarm, setFarm } = useAppStore()
  const { farms, loading, error, createFarm, listUserFarms } = useFarm()
  const [showForm, setShowForm] = useState(false)

  useEffect(() => {
    if (user) {
      listUserFarms(user.user_id)
    }
  }, [user, listUserFarms])

  const handleSelectFarm = (farm: any) => {
    setFarm(farm)
  }

  if (!user) {
    return (
      <div className="container-custom py-12">
        <div className="card text-center">
          <h2 className="text-2xl font-bold mb-4">Please Login</h2>
          <p className="text-gray-600 mb-6">You need to login to access farms.</p>
          <Link href="/login" className="btn-primary inline-block">
            Go to Login
          </Link>
        </div>
      </div>
    )
  }

  const handleCreateFarm = async (data: any) => {
    const result = await createFarm(data)
    if (result.success) {
      setShowForm(false)
      await listUserFarms(user.user_id)
    }
  }

  return (
    <div className="container-custom py-12">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold mb-2">My Farms</h1>
          <p className="text-gray-600">Manage your agricultural properties</p>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          className="btn-primary flex items-center gap-2"
        >
          <FiPlus />
          Add Farm
        </button>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
          <FiAlertCircle className="text-red-600 mt-0.5 flex-shrink-0" />
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      {showForm && (
        <div className="card mb-8">
          <h2 className="text-2xl font-bold mb-6">Create New Farm</h2>
          <FarmForm onSubmit={handleCreateFarm} loading={loading} />
        </div>
      )}

      {selectedFarm && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-blue-700 text-sm">
            <strong>Selected Farm:</strong> {selectedFarm.farm_id} ({selectedFarm.area_hectares} hectares)
          </p>
        </div>
      )}

      {loading && !showForm ? (
        <div className="text-center py-12">
          <p className="text-gray-600">Loading farms...</p>
        </div>
      ) : farms.length === 0 ? (
        <div className="card text-center">
          <p className="text-gray-600 mb-4">No farms yet. Create your first farm to get started.</p>
          <button
            onClick={() => setShowForm(true)}
            className="btn-primary inline-flex items-center gap-2"
          >
            <FiPlus />
            Create First Farm
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {farms.map((farm) => (
            <div key={farm.farm_id} className="relative">
              <FarmCard farm={farm} onSelect={handleSelectFarm} />
              {selectedFarm?.farm_id === farm.farm_id && (
                <div className="absolute top-4 right-4 px-3 py-1 bg-blue-500 text-white text-xs font-medium rounded-full">
                  Selected
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
