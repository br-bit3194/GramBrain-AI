// Error types for the application

export type ErrorType = 'auth' | 'validation' | 'network' | 'server' | 'not_found' | 'api' | 'unknown'

export type ErrorAction = 'redirect_login' | 'retry' | 'none'

export interface ValidationError {
  field: string
  message: string
}

export interface UserFriendlyError {
  type: ErrorType
  message: string
  action: ErrorAction
  fields?: ValidationError[]
  originalError?: any
}

export interface ApiErrorResponse {
  status: 'error'
  detail?: string
  message?: string
  errors?: ValidationError[]
}
