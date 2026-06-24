import axios from 'axios'
import type {
  QueryRequest,
  QueryResponse,
  BatchRequest,
  BatchResponse,
  HealthResponse,
  StatsResponse,
} from '@/types'
import { useSettingsStore } from '@/stores/settingsStore'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
  timeout: 120_000,
  headers: { 'Content-Type': 'application/json' },
})

// Attach API key from store on every request
api.interceptors.request.use((config) => {
  const apiKey = useSettingsStore.getState().apiKey
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey
  }
  return config
})

export async function queryApi(req: QueryRequest): Promise<QueryResponse> {
  const { data } = await api.post<QueryResponse>('/query', req)
  return data
}

export async function batchApi(req: BatchRequest): Promise<BatchResponse> {
  const { data } = await api.post<BatchResponse>('/batch', req)
  return data
}

export async function healthApi(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health')
  return data
}

export async function statsApi(): Promise<StatsResponse> {
  const { data } = await api.get<StatsResponse>('/stats')
  return data
}
