import { handleApiError, getShortErrorMessage, shouldRedirectToLogin, isRetryableError } from '../errorHandler'
import { AxiosError } from 'axios'

describe('errorHandler', () => {
  describe('handleApiError', () => {
    it('should handle 401 authentication errors', () => {
      const error = {
        isAxiosError: true,
        response: {
          status: 401,
          data: { detail: 'Unauthorized' },
        },
      } as any

      const result = handleApiError(error)

      expect(result.type).toBe('auth')
      expect(result.message).toContain('session has expired')
      expect(result.action).toBe('redirect_login')
    })

    it('should handle 403 forbidden errors', () => {
      const error = {
        isAxiosError: true,
        response: {
          status: 403,
          data: { detail: 'Forbidden' },
        },
      } as any

      const result = handleApiError(error)

      expect(result.type).toBe('auth')
      expect(result.message).toContain('do not have permission')
      expect(result.action).toBe('none')
    })

    it('should handle 404 not found errors', () => {
      const error = {
        isAxiosError: true,
        response: {
          status: 404,
          data: { detail: 'Not found' },
        },
      } as any

      const result = handleApiError(error)

      expect(result.type).toBe('not_found')
      expect(result.message).toContain('not found')
      expect(result.action).toBe('none')
    })

    it('should handle 422 validation errors with FastAPI format', () => {
      const error = {
        isAxiosError: true,
        response: {
          status: 422,
          data: {
            detail: [
              { loc: ['body', 'phone_number'], msg: 'field required' },
              { loc: ['body', 'password'], msg: 'ensure this value has at least 8 characters' },
            ],
          },
        },
      } as any

      const result = handleApiError(error)

      expect(result.type).toBe('validation')
      expect(result.message).toContain('check your input')
      expect(result.action).toBe('none')
      expect(result.fields).toHaveLength(2)
      expect(result.fields?.[0].field).toBe('body.phone_number')
      expect(result.fields?.[1].field).toBe('body.password')
    })

    it('should handle 422 validation errors with custom format', () => {
      const error = {
        isAxiosError: true,
        response: {
          status: 422,
          data: {
            errors: [
              { field: 'email', message: 'Invalid email format' },
            ],
          },
        },
      } as any

      const result = handleApiError(error)

      expect(result.type).toBe('validation')
      expect(result.fields).toHaveLength(1)
      expect(result.fields?.[0].field).toBe('email')
      expect(result.fields?.[0].message).toBe('Invalid email format')
    })

    it('should handle 500 server errors', () => {
      const error = {
        isAxiosError: true,
        response: {
          status: 500,
          data: { detail: 'Internal server error' },
        },
      } as any

      const result = handleApiError(error)

      expect(result.type).toBe('server')
      expect(result.message).toContain('servers are experiencing issues')
      expect(result.action).toBe('retry')
    })

    it('should handle 503 service unavailable errors', () => {
      const error = {
        isAxiosError: true,
        response: {
          status: 503,
          data: { detail: 'Service unavailable' },
        },
      } as any

      const result = handleApiError(error)

      expect(result.type).toBe('server')
      expect(result.action).toBe('retry')
    })

    it('should handle 400 bad request with custom message', () => {
      const error = {
        isAxiosError: true,
        response: {
          status: 400,
          data: { detail: 'Invalid farm location' },
        },
      } as any

      const result = handleApiError(error)

      expect(result.type).toBe('api')
      expect(result.message).toBe('Invalid farm location')
      expect(result.action).toBe('retry')
    })

    it('should handle network errors (no response)', () => {
      const error = {
        isAxiosError: true,
        request: {},
        message: 'Network Error',
      } as any

      const result = handleApiError(error)

      expect(result.type).toBe('network')
      expect(result.message).toContain('Unable to connect')
      expect(result.action).toBe('retry')
    })

    it('should handle timeout errors', () => {
      const error = {
        code: 'ECONNABORTED',
        message: 'timeout of 30000ms exceeded',
      } as any

      const result = handleApiError(error)

      expect(result.type).toBe('network')
      expect(result.message).toContain('took too long')
      expect(result.action).toBe('retry')
    })

    it('should handle unknown errors', () => {
      const error = new Error('Something went wrong')

      const result = handleApiError(error)

      expect(result.type).toBe('unknown')
      expect(result.message).toContain('unexpected error')
      expect(result.action).toBe('retry')
    })

    it('should preserve original error in all cases', () => {
      const error = {
        isAxiosError: true,
        response: {
          status: 500,
          data: {},
        },
      } as any

      const result = handleApiError(error)

      expect(result.originalError).toBe(error)
    })
  })

  describe('getShortErrorMessage', () => {
    it('should return short message for auth errors', () => {
      const error = {
        type: 'auth' as const,
        message: 'Long auth message',
        action: 'redirect_login' as const,
      }

      expect(getShortErrorMessage(error)).toBe('Authentication required')
    })

    it('should return short message for validation errors', () => {
      const error = {
        type: 'validation' as const,
        message: 'Long validation message',
        action: 'none' as const,
      }

      expect(getShortErrorMessage(error)).toBe('Invalid input')
    })

    it('should return short message for network errors', () => {
      const error = {
        type: 'network' as const,
        message: 'Long network message',
        action: 'retry' as const,
      }

      expect(getShortErrorMessage(error)).toBe('Connection error')
    })

    it('should return short message for server errors', () => {
      const error = {
        type: 'server' as const,
        message: 'Long server message',
        action: 'retry' as const,
      }

      expect(getShortErrorMessage(error)).toBe('Server error')
    })

    it('should return short message for not found errors', () => {
      const error = {
        type: 'not_found' as const,
        message: 'Long not found message',
        action: 'none' as const,
      }

      expect(getShortErrorMessage(error)).toBe('Not found')
    })

    it('should return short message for api errors', () => {
      const error = {
        type: 'api' as const,
        message: 'Long api message',
        action: 'retry' as const,
      }

      expect(getShortErrorMessage(error)).toBe('Request failed')
    })

    it('should return short message for unknown errors', () => {
      const error = {
        type: 'unknown' as const,
        message: 'Long unknown message',
        action: 'retry' as const,
      }

      expect(getShortErrorMessage(error)).toBe('Error occurred')
    })
  })

  describe('shouldRedirectToLogin', () => {
    it('should return true for redirect_login action', () => {
      const error = {
        type: 'auth' as const,
        message: 'Auth error',
        action: 'redirect_login' as const,
      }

      expect(shouldRedirectToLogin(error)).toBe(true)
    })

    it('should return false for other actions', () => {
      const error = {
        type: 'network' as const,
        message: 'Network error',
        action: 'retry' as const,
      }

      expect(shouldRedirectToLogin(error)).toBe(false)
    })
  })

  describe('isRetryableError', () => {
    it('should return true for retry action', () => {
      const error = {
        type: 'network' as const,
        message: 'Network error',
        action: 'retry' as const,
      }

      expect(isRetryableError(error)).toBe(true)
    })

    it('should return false for other actions', () => {
      const error = {
        type: 'validation' as const,
        message: 'Validation error',
        action: 'none' as const,
      }

      expect(isRetryableError(error)).toBe(false)
    })
  })
})
