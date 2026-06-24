import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { BenchmarkType, QueryMode } from '@/types'

interface SettingsState {
  apiKey: string
  mode: QueryMode
  benchmarkType: BenchmarkType
  sidebarOpen: boolean
  setMode: (mode: QueryMode) => void
  setBenchmarkType: (type: BenchmarkType) => void
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      apiKey: '',
      mode: 'kg_rag',
      benchmarkType: 'bioasq',
      sidebarOpen: true,
      setMode: (mode) => set({ mode }),
      setBenchmarkType: (benchmarkType) => set({ benchmarkType }),
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
    }),
    {
      name: 'chronomedkg-settings',
      partialize: (s) => ({ apiKey: s.apiKey, mode: s.mode, benchmarkType: s.benchmarkType }),
    },
  ),
)
