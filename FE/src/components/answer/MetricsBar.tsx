import { cn, formatLatency, formatNumber } from '@/lib/utils'
import type { QueryResponse } from '@/types'
import { Clock, Hash, Brain, Globe } from 'lucide-react'

interface MetricsBarProps {
  response: QueryResponse
}

export function MetricsBar({ response }: MetricsBarProps) {
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs text-text-secondary mt-3 pt-3 border-t border-border-subtle">
      <MetricChip
        icon={<Clock className="h-3 w-3" />}
        label={formatLatency(response.latency_ms)}
      />
      <MetricChip
        icon={<Hash className="h-3 w-3" />}
        label={`${formatNumber(response.tokens_used)} tokens`}
      />
      <MetricChip
        icon={<Brain className="h-3 w-3" />}
        label={response.kg_coverage ? 'KG Enhanced' : 'LLM Only'}
        highlight={response.kg_coverage}
      />
      <MetricChip
        icon={<Globe className="h-3 w-3" />}
        label={response.lang_detected.toUpperCase()}
      />
    </div>
  )
}

function MetricChip({
  icon,
  label,
  highlight = false,
}: {
  icon: React.ReactNode
  label: string
  highlight?: boolean
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded-full',
        highlight
          ? 'bg-primary-soft text-primary'
          : 'bg-surface-elevated text-text-secondary',
      )}
    >
      {icon}
      {label}
    </span>
  )
}
