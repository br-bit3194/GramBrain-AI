import { setTokenGetter } from '@/services/api'
import { useAppStore } from '@/store/appStore'

/**
 * Initialize the API client with the token getter from the store.
 * This should be called once when the app initializes.
 */
export function initializeApiClient() {
  setTokenGetter(() => {
    return useAppStore.getState().accessToken
  })
}
