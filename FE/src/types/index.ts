// ── API Request / Response Types (mirrors Python schemas/models.py) ──

export type BenchmarkType = 'bioasq' | 'medqa' | 'pubmedqa'
export type QueryMode = 'kg_rag'

export interface QueryRequest {
  query: string
  benchmark_type: BenchmarkType
  mode: QueryMode
  options: Record<string, unknown>
}

export interface QueryResponse {
  answer: unknown
  question_type: string
  sources: string[]
  kg_coverage: boolean
  matched_entities: string[]
  lang_detected: string
  latency_ms: number
  tokens_used: number
  error: string | null
}

export interface BatchItem {
  id: string
  query: string
  benchmark_type: BenchmarkType
  mode: QueryMode
  options: Record<string, unknown>
}

export interface BatchRequest {
  queries: BatchItem[]
}

export interface BatchResultItem {
  id: string
  result: QueryResponse | null
  error: string | null
}

export interface BatchResponse {
  results: BatchResultItem[]
  summary: {
    total: number
    success: number
    failed: number
    kg_hit_rate: number
  }
}

export interface HealthResponse {
  status: string
  neo4j: string
  faiss: string
  pipeline: string
}

export interface StatsResponse {
  requests_last_hour: number
  total_requests: number
  avg_latency_ms: number
  avg_tokens: number
  kg_hit_rate: number
}

// ── Internal UI Types ──

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'error'
  query?: string
  response?: QueryResponse
  errorMessage?: string
  timestamp: number
}
