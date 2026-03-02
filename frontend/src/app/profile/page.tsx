'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useAppStore } from '@/store/appStore'
import { FiUser, FiPhone, FiGlobe, FiLogOut, FiAlertCircle } from 'react-icons/fi'

export default function ProfilePage() {
  const router = useRouter()
  const { user, clearStore } = useAppStore()
  const [isEditing, setIsEditing] = useState(false)
  const [error, setError] = useState('')

  if (!user) {
    return (
      <div className="container-custom py-12">
        <div className="card text-center">
          <h2 className="text-2xl font-bold mb-4">Please Login</h2>
          <p className="text-gray-600 mb-6">You need to login to access your profile.</p>
          <Link href="/login" className="btn-primary inline-block">
            Go to Login
          </Link>
        </div>
      </div>
    )
  }

  const handleLogout = () => {
    clearStore()
    router.push('/')
  }

  return (
    <div className="container-custom py-12">
      <div className="max-w-2xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">My Profile</h1>
          <p className="text-gray-600">Manage your account information</p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
            <FiAlertCircle className="text-red-600 mt-0.5 flex-shrink-0" />
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        )}

        <div className="card mb-8">
          <div className="flex items-start justify-between mb-6">
            <h2 className="text-2xl font-bold">Account Information</h2>
            <button
              onClick={() => setIsEditing(!isEditing)}
              className="btn-outline text-sm"
            >
              {isEditing ? 'Cancel' : 'Edit'}
            </button>
          </div>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <FiUser className="inline mr-2" />
                Full Name
              </label>
              <p className="text-lg font-semibold">{user.name}</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <FiPhone className="inline mr-2" />
                Phone Number
              </label>
              <p className="text-lg font-semibold">{user.phone_number}</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <FiGlobe className="inline mr-2" />
                Language Preference
              </label>
              <p className="text-lg font-semibold capitalize">{user.language_preference}</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Role
              </label>
              <p className="text-lg font-semibold capitalize">{user.role.replace('_', ' ')}</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Member Since
              </label>
              <p className="text-lg font-semibold">
                {new Date(user.created_at).toLocaleDateString()}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <h2 className="text-2xl font-bold mb-6">Account Actions</h2>
          <div className="space-y-3">
            <button
              onClick={handleLogout}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors font-medium"
            >
              <FiLogOut />
              Logout
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
