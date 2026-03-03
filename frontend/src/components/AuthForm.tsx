'use client'

import { useState } from 'react'
import { useAuth } from '@/hooks/useAuth'

type AuthMode = 'login' | 'register'

export function AuthForm() {
  const [mode, setMode] = useState<AuthMode>('login')
  const [phoneNumber, setPhoneNumber] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [role, setRole] = useState<'farmer' | 'village_leader' | 'policymaker' | 'consumer'>('farmer')
  
  const { login, register, isLoading, error } = useAuth()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (mode === 'login') {
      const result = await login({ phone_number: phoneNumber, password })
      if (result.success) {
        console.log('Login successful:', result.user)
      }
    } else {
      const result = await register({
        phone_number: phoneNumber,
        password,
        name,
        role,
        language_preference: 'en',
      })
      if (result.success) {
        console.log('Registration successful:', result.user)
      }
    }
  }

  return (
    <div className="max-w-md mx-auto mt-8 p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-center">
        {mode === 'login' ? 'Login' : 'Register'}
      </h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        {mode === 'register' && (
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
              Name
            </label>
            <input
              id="name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
              placeholder="Enter your name"
            />
          </div>
        )}
        
        <div>
          <label htmlFor="phone" className="block text-sm font-medium text-gray-700 mb-1">
            Phone Number
          </label>
          <input
            id="phone"
            type="tel"
            value={phoneNumber}
            onChange={(e) => setPhoneNumber(e.target.value)}
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
            placeholder="+91 98765 43210"
          />
        </div>
        
        <div>
          <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
            Password
          </label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
            placeholder="Enter your password"
          />
        </div>
        
        {mode === 'register' && (
          <div>
            <label htmlFor="role" className="block text-sm font-medium text-gray-700 mb-1">
              Role
            </label>
            <select
              id="role"
              value={role}
              onChange={(e) => setRole(e.target.value as any)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
            >
              <option value="farmer">Farmer</option>
              <option value="village_leader">Village Leader</option>
              <option value="policymaker">Policymaker</option>
              <option value="consumer">Consumer</option>
            </select>
          </div>
        )}
        
        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-md">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}
        
        <button
          type="submit"
          disabled={isLoading}
          className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Processing...' : mode === 'login' ? 'Login' : 'Register'}
        </button>
      </form>
      
      <div className="mt-4 text-center">
        <button
          type="button"
          onClick={() => setMode(mode === 'login' ? 'register' : 'login')}
          className="text-sm text-green-600 hover:text-green-700"
        >
          {mode === 'login' ? "Don't have an account? Register" : 'Already have an account? Login'}
        </button>
      </div>
    </div>
  )
}
