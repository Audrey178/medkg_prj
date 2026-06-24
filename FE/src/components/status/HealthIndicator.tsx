import { useState, useEffect, useCallback } from 'react'
import {
  Activity,
  Database,
  Cpu,
  Search,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import { healthApi, statsApi } from '@/api/client'
import { cn, formatLatency, formatNumber, formatPercent } from '@/lib/utils'
import type { HealthResponse, StatsResponse } from '@/types'

// ── Health Indicator ──

export function HealthIndicator() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState(false)

  const fetchHealth = useCallback(async () => {
    try {
      const data = await healthApi()
      setHealth(data)
      setError(false)
    } catch {
      setError(true)
    }
  }, [])

  useEffect(() => {
    fetchHealth()
    const interval = setInterval(fetchHealth, 30_000)
    return () => clearInterval(interval)
  }, [fetchHealth])

  const isOk = health?.status === 'ok' && !error
  const isPartial =
    health && (health.neo4j !== 'connected' || health.faiss !== 'loaded')

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs font-medium text-text-secondary uppercase tracking-wider">
        <Activity className="h-3.5 w-3.5" />
        System Status
      </div>

      <div className="space-y-1.5">
        <StatusRow
          icon={<div className={cn(
            'h-2 w-2 rounded-full',
            error ? 'bg-error' : isOk && !isPartial ? 'bg-success animate-pulse-dot' : 'bg-warning',
          )} />}
          label="API"
          value={error ? 'Offline' : 'Online'}
          ok={!error}
        />
        {health && (
          <>
            <StatusRow
              icon={<Database className="h-3.5 w-3.5" />}
              label="Neo4j"
              value={health.neo4j}
              ok={health.neo4j === 'connected'}
            />
            <StatusRow
              icon={<Search className="h-3.5 w-3.5" />}
              label="FAISS"
              value={health.faiss}
              ok={health.faiss === 'loaded'}
            />
            <StatusRow
              icon={<Cpu className="h-3.5 w-3.5" />}
              label="Pipeline"
              value={health.pipeline}
              ok={health.pipeline === 'ready'}
            />
          </>
        )}
      </div>
    </div>
  )
}

function StatusRow({
  icon,
  label,
  value,
  ok,
}: {
  icon: React.ReactNode
  label: string
  value: string
  ok: boolean
}) {
  return (
    <div className="flex items-center justify-between gap-2 text-xs py-1">
      <span className="flex items-center gap-1.5 text-text-secondary">
        {icon}
        {label}
      </span>
      <span className={cn('font-mono', ok ? 'text-success' : 'text-error')}>
        {value}
      </span>
    </div>
  )
}

// ── Stats Panel ──

export function StatsPanel() {
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const fetchStats = useCallback(async () => {
    setLoading(true)
    try {
      const data = await statsApi()
      setStats(data)
    } catch {
      // stats endpoint requires valid API key
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (expanded && !stats && !loading) {
      fetchStats()
    }
  }, [expanded, stats, loading, fetchStats])

  return (
    <div className="space-y-2">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center justify-between w-full text-xs font-medium text-text-secondary uppercase tracking-wider hover:text-text-primary transition-colors"
      >
        <span className="flex items-center gap-2">
          <Activity className="h-3.5 w-3.5" />
          Statistics
        </span>
        {expanded ? (
          <ChevronUp className="h-3.5 w-3.5" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5" />
        )}
      </button>

      {expanded && (
        <div className="space-y-1.5 animate-fade-in">
          {loading && (
            <p className="text-xs text-text-muted">Loading stats…</p>
          )}
          {stats && (
            <>
              <StatRow label="Requests / hr" value={formatNumber(stats.requests_last_hour)} />
              <StatRow label="Total requests" value={formatNumber(stats.total_requests)} />
              <StatRow label="Avg latency" value={formatLatency(stats.avg_latency_ms)} />
              <StatRow label="Avg tokens" value={formatNumber(stats.avg_tokens)} />
              <StatRow label="KG hit rate" value={formatPercent(stats.kg_hit_rate)} />
            </>
          )}
          {!loading && !stats && (
            <p className="text-xs text-text-muted">Enter API key to view stats</p>
          )}
        </div>
      )}
    </div>
  )
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-xs py-0.5">
      <span className="text-text-secondary">{label}</span>
      <span className="font-mono text-text-primary">{value}</span>
    </div>
  )
}
