'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/hooks/useAuth'
import { FiPhone, FiLock, FiAlertCircle } from 'react-icons/fi'

export default function LoginPage() {
  const router = useRouter()
  const { login } = useAuth()
  const [phoneNumber, setPhoneNumber] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      if (!phoneNumber.trim() || !password.trim()) {
        setError('Phone number and password are required')
        setLoading(false)
        return
      }

      const result = await login({
        phone_number: phoneNumber,
        password: password,
      })
      
      if (result.success) {
        router.push('/dashboard')
      } else {
        setError(result.error || 'Login failed. Please try again.')
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Login failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary to-green-600 flex items-center justify-center py-12 px-4">
      <div className="w-full max-w-md">
        <div className="bg-white rounded-lg shadow-xl p-8">
          <h1 className="text-3xl font-bold text-center mb-2">Welcome Back</h1>
          <p className="text-gray-600 text-center mb-8">Login to your GramBrain AI account</p>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
              <FiAlertCircle className="text-red-600 mt-0.5 flex-shrink-0" />
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="phone" className="block text-sm font-medium text-gray-700 mb-2">
                Phone Number
              </label>
              <div className="relative">
                <FiPhone className="absolute left-3 top-3 text-gray-400" />
                <input
                  id="phone"
                  type="tel"
                  value={phoneNumber}
                  onChange={(e) => setPhoneNumber(e.target.value)}
                  placeholder="+91 98765 43210"
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
                  disabled={loading}
                />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <div className="relative">
                <FiLock className="absolute left-3 top-3 text-gray-400" />
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
                  disabled={loading}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-primary text-white py-2 rounded-lg font-medium hover:bg-primary-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Logging in...' : 'Login'}
            </button>
          </form>

          <div className="mt-6 pt-6 border-t border-gray-200">
            <p className="text-center text-gray-600">
              Don't have an account?{' '}
              <Link href="/register" className="text-primary font-medium hover:underline">
                Register here
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
