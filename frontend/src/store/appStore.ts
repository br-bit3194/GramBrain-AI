import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { AppStore, User, Farm } from '@/types'

export const useAppStore = create<AppStore>()(
  persist(
    (set) => ({
      user: null,
      farm: null,
      accessToken: null,
      refreshToken: null,

      setUser: (user: User | null) => set({ user }),
      setFarm: (farm: Farm | null) => set({ farm }),
      setTokens: (accessToken: string, refreshToken: string) => 
        set({ accessToken, refreshToken }),

      clearStore: () => set({ 
        user: null, 
        farm: null, 
        accessToken: null, 
        refreshToken: null 
      }),

      logout: () => {
        // Clear all auth data from store
        set({ 
          user: null, 
          farm: null, 
          accessToken: null, 
          refreshToken: null 
        })
        // Clear from localStorage (handled by persist middleware)
      },
    }),
    {
      name: 'grambrain-storage', // unique name for localStorage key
      storage: createJSONStorage(() => localStorage),
      // Only persist auth-related data
      partialize: (state) => ({
        user: state.user,
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        farm: state.farm,
      }),
    }
  )
)
