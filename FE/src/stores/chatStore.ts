import { create } from 'zustand'
import type { ChatMessage, QueryResponse } from '@/types'

interface ChatState {
  messages: ChatMessage[]
  isLoading: boolean
  addUserMessage: (query: string) => string
  addBotMessage: (id: string, response: QueryResponse) => void
  addErrorMessage: (id: string, errorMessage: string) => void
  setLoading: (v: boolean) => void
  clearMessages: () => void
}

let counter = 0
function nextId(): string {
  counter += 1
  return `msg-${Date.now()}-${counter}`
}

export const useChatStore = create<ChatState>()((set) => ({
  messages: [],
  isLoading: false,

  addUserMessage: (query) => {
    const id = nextId()
    set((s) => ({
      messages: [
        ...s.messages,
        { id, role: 'user', query, timestamp: Date.now() },
      ],
    }))
    return id
  },

  addBotMessage: (_id, response) => {
    const id = nextId()
    set((s) => ({
      messages: [
        ...s.messages,
        { id, role: 'assistant', response, timestamp: Date.now() },
      ],
    }))
  },

  addErrorMessage: (_id, errorMessage) => {
    const id = nextId()
    set((s) => ({
      messages: [
        ...s.messages,
        { id, role: 'error', errorMessage, timestamp: Date.now() },
      ],
    }))
  },

  setLoading: (isLoading) => set({ isLoading }),
  clearMessages: () => set({ messages: [] }),
}))
