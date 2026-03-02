import { create } from 'zustand'
import { AppStore, User, Farm } from '@/types'

export const useAppStore = create<AppStore>((set) => ({
  user: null,
  farm: null,

  setUser: (user: User | null) => set({ user }),
  setFarm: (farm: Farm | null) => set({ farm }),

  clearStore: () => set({ user: null, farm: null }),
}))
