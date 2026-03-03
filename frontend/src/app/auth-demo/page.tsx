'use client'

import { useAuth } from '@/hooks/useAuth'
import { AuthForm } from '@/components/AuthForm'

export default function AuthDemoPage() {
  const { user, isAuthenticated, logout, getCurrentUser } = useAuth()

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8 text-gray-900">
          Authentication Demo
        </h1>

        {isAuthenticated ? (
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-semibold mb-4 text-green-600">
                ✅ Authenticated
              </h2>
              
              <div className="bg-gray-50 rounded-lg p-4 mb-4">
                <h3 className="font-semibold mb-2">User Information:</h3>
                <pre className="text-sm overflow-auto">
                  {JSON.stringify(user, null, 2)}
                </pre>
              </div>

              <div className="flex gap-4">
                <button
                  onClick={logout}
                  className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
                >
                  Logout
                </button>
                
                <button
                  onClick={getCurrentUser}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  Refresh User Data
                </button>
              </div>
            </div>

            <div className="border-t pt-6">
              <h3 className="font-semibold mb-3">Test Protected Endpoints:</h3>
              <div className="space-y-2">
                <p className="text-sm text-gray-600">
                  Now that you're authenticated, you can make requests to protected endpoints.
                  The API client will automatically include your access token.
                </p>
                <div className="bg-blue-50 border border-blue-200 rounded-md p-3">
                  <p className="text-sm font-mono">
                    Authorization: Bearer {user?.user_id?.substring(0, 20)}...
                  </p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div>
            <div className="mb-6 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <p className="text-sm text-yellow-800">
                ℹ️ You are not authenticated. Please login or register to continue.
              </p>
            </div>
            
            <AuthForm />
          </div>
        )}

        <div className="mt-8 bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-4">How It Works</h3>
          <div className="space-y-3 text-sm text-gray-700">
            <div className="flex items-start gap-2">
              <span className="font-semibold text-green-600">1.</span>
              <p>
                <strong>Register/Login:</strong> Enter your credentials to get JWT tokens
              </p>
            </div>
            <div className="flex items-start gap-2">
              <span className="font-semibold text-green-600">2.</span>
              <p>
                <strong>Token Storage:</strong> Tokens are stored in Zustand state and localStorage
              </p>
            </div>
            <div className="flex items-start gap-2">
              <span className="font-semibold text-green-600">3.</span>
              <p>
                <strong>Auto Injection:</strong> API client automatically adds token to all requests
              </p>
            </div>
            <div className="flex items-start gap-2">
              <span className="font-semibold text-green-600">4.</span>
              <p>
                <strong>Session Persistence:</strong> Your session persists across page reloads
              </p>
            </div>
            <div className="flex items-start gap-2">
              <span className="font-semibold text-green-600">5.</span>
              <p>
                <strong>Auto Logout:</strong> 401 responses automatically clear your session
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
