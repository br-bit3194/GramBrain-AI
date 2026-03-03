'use client'

import { useEffect, useState } from 'react'
import { useAuth } from '@/hooks/useAuth'

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const { restoreSession } = useAuth()
  const [isInitialized, setIsInitialized] = useState(false)

  useEffect(() => {
    // Restore session from localStorage on mount
    restoreSession()
    setIsInitialized(true)
  }, [])

  // Optionally show a loading state while initializing
  if (!isInitialized) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  return <>{children}</>
}
