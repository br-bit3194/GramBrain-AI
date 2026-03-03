import { AxiosError } from 'axios'
import { UserFriendlyError, ApiErrorResponse, ValidationError } from '@/types/errors'

// Re-export types for convenience
export type { UserFriendlyError, ApiErrorResponse, ValidationError } from '@/types/errors'

/**
 * Transform API errors into user-friendly error messages
 * This function handles different types of errors and maps them to appropriate user messages
 * 
 * @param error - The error object from axios or other sources
 * @returns UserFriendlyError object with type, message, and action
 */
export function handleApiError(error: any): UserFriendlyError {
  // Handle axios errors
  if (error.isAxiosError || error.response) {
    const axiosError = error as AxiosError<ApiErrorResponse>
    
    // Server responded with an error status
    if (axiosError.response) {
      const status = axiosError.response.status
      const data = axiosError.response.data
      
      // Authentication errors (401)
      if (status === 401) {
        return {
          type: 'auth',
          message: 'Your session has expired. Please login to continue.',
          action: 'redirect_login',
          originalError: error,
        }
      }
      
      // Forbidden errors (403)
      if (status === 403) {
        return {
          type: 'auth',
          message: 'You do not have permission to perform this action.',
          action: 'none',
          originalError: error,
        }
      }
      
      // Not found errors (404)
      if (status === 404) {
        return {
          type: 'not_found',
          message: 'The requested resource was not found.',
          action: 'none',
          originalError: error,
        }
      }
      
      // Validation errors (422)
      if (status === 422) {
        const validationErrors: ValidationError[] = []
        
        // Extract validation errors from response
        if (data && typeof data === 'object') {
          if ('errors' in data && Array.isArray(data.errors)) {
            validationErrors.push(...data.errors)
          } else if ('detail' in data && Array.isArray(data.detail)) {
            // FastAPI validation errors format
            data.detail.forEach((err: any) => {
              const field = Array.isArray(err.loc) ? err.loc.join('.') : 'unknown'
              validationErrors.push({
                field,
                message: err.msg || 'Invalid value',
              })
            })
          }
        }
        
        return {
          type: 'validation',
          message: validationErrors.length > 0 
            ? 'Please check your input and try again.' 
            : 'The data you provided is invalid.',
          action: 'none',
          fields: validationErrors.length > 0 ? validationErrors : undefined,
          originalError: error,
        }
      }
      
      // Server errors (500+)
      if (status >= 500) {
        return {
          type: 'server',
          message: 'Our servers are experiencing issues. Please try again later.',
          action: 'retry',
          originalError: error,
        }
      }
      
      // Other API errors (400, etc.)
      const errorMessage = 
        (data && typeof data === 'object' && 'detail' in data && typeof data.detail === 'string') 
          ? data.detail 
          : (data && typeof data === 'object' && 'message' in data && typeof data.message === 'string')
          ? data.message
          : 'An error occurred while processing your request.'
      
      return {
        type: 'api',
        message: errorMessage,
        action: 'retry',
        originalError: error,
      }
    }
    
    // Request was made but no response received (network error)
    if (axiosError.request) {
      return {
        type: 'network',
        message: 'Unable to connect to the server. Please check your internet connection and try again.',
        action: 'retry',
        originalError: error,
      }
    }
  }
  
  // Request timeout
  if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
    return {
      type: 'network',
      message: 'The request took too long. Please check your connection and try again.',
      action: 'retry',
      originalError: error,
    }
  }
  
  // Generic error fallback
  return {
    type: 'unknown',
    message: 'An unexpected error occurred. Please try again.',
    action: 'retry',
    originalError: error,
  }
}

/**
 * Get a short error message suitable for toast notifications
 * @param error - UserFriendlyError object
 * @returns Short error message string
 */
export function getShortErrorMessage(error: UserFriendlyError): string {
  switch (error.type) {
    case 'auth':
      return 'Authentication required'
    case 'validation':
      return 'Invalid input'
    case 'network':
      return 'Connection error'
    case 'server':
      return 'Server error'
    case 'not_found':
      return 'Not found'
    case 'api':
      return 'Request failed'
    default:
      return 'Error occurred'
  }
}

/**
 * Check if an error should trigger a logout/redirect to login
 * @param error - UserFriendlyError object
 * @returns true if should redirect to login
 */
export function shouldRedirectToLogin(error: UserFriendlyError): boolean {
  return error.action === 'redirect_login'
}

/**
 * Check if an error is retryable
 * @param error - UserFriendlyError object
 * @returns true if the operation can be retried
 */
export function isRetryableError(error: UserFriendlyError): boolean {
  return error.action === 'retry'
}
