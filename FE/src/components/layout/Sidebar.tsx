import { useSettingsStore } from '@/stores/settingsStore'
import { useChatStore } from '@/stores/chatStore'
import { HealthIndicator, StatsPanel } from '@/components/status/HealthIndicator'
import { cn } from '@/lib/utils'
import {
  Brain,
  Settings,
  Trash2,
  Eye,
  EyeOff,
  Zap,
  Server,
  Layers,
} from 'lucide-react'
import { useState } from 'react'
import type { BenchmarkType, QueryMode } from '@/types'

const MODES: { value: QueryMode; label: string; desc: string; icon: React.ReactNode }[] = [
  { value: 'kg_rag', label: 'KG + RAG', desc: 'Full pipeline', icon: <Brain className="h-3.5 w-3.5" /> },
]

const BENCHMARKS: { value: BenchmarkType; label: string }[] = [
  { value: 'bioasq', label: 'BioASQ' },
  { value: 'medqa', label: 'MedQA' },
  { value: 'pubmedqa', label: 'PubMedQA' },
]

export function Sidebar() {
  const apiKey = useSettingsStore((s) => s.apiKey)
  const setApiKey = useSettingsStore((s) => s.setApiKey)
  const mode = useSettingsStore((s) => s.mode)
  const setMode = useSettingsStore((s) => s.setMode)
  const benchmarkType = useSettingsStore((s) => s.benchmarkType)
  const setBenchmarkType = useSettingsStore((s) => s.setBenchmarkType)
  const clearMessages = useChatStore((s) => s.clearMessages)
  const messageCount = useChatStore((s) => s.messages.length)

  const [showKey, setShowKey] = useState(false)

  return (
    <aside className="flex flex-col h-full overflow-y-auto">
      {/* Logo */}
      <div className="px-4 py-5 border-b border-border-subtle">
        <div className="flex items-center gap-2.5">
          <div className="h-8 w-8 rounded-lg bg-primary-soft flex items-center justify-center">
            <Brain className="h-4.5 w-4.5 text-primary" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-text-primary leading-tight">
              ChronoMedKG
            </h1>
            <p className="text-[10px] text-text-muted">Biomedical QA System</p>
          </div>
        </div>
      </div>

      <div className="flex-1 px-4 py-4 space-y-6 overflow-y-auto">
        {/* API Key */}
        <div className="space-y-2">
          <label
            htmlFor="api-key-input"
            className="flex items-center gap-1.5 text-xs font-medium text-text-secondary uppercase tracking-wider"
          >
            <Settings className="h-3.5 w-3.5" />
            API Key
          </label>
          <div className="relative">
            <input
              id="api-key-input"
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter API key…"
              className={cn(
                'w-full rounded-lg border border-border bg-surface px-3 py-2 pr-9 text-sm',
                'text-text-primary placeholder:text-text-muted',
                'focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20',
                'transition-colors',
              )}
            />
            <button
              onClick={() => setShowKey((v) => !v)}
              aria-label={showKey ? 'Hide API key' : 'Show API key'}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary transition-colors"
            >
              {showKey ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
            </button>
          </div>
          {apiKey && (
            <p className="text-[10px] text-success flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-success inline-block" />
              Key configured
            </p>
          )}
        </div>

        {/* Mode Selector */}
        <div className="space-y-2">
          <label className="flex items-center gap-1.5 text-xs font-medium text-text-secondary uppercase tracking-wider">
            <Server className="h-3.5 w-3.5" />
            Inference Mode
          </label>
          <div className="space-y-1">
            {MODES.map((m) => (
              <button
                key={m.value}
                onClick={() => setMode(m.value)}
                className={cn(
                  'w-full flex items-center gap-2.5 rounded-lg px-3 py-2 text-left transition-all duration-150',
                  mode === m.value
                    ? 'bg-primary-soft border border-primary/30 text-primary'
                    : 'border border-transparent text-text-secondary hover:bg-surface-hover hover:text-text-primary',
                )}
              >
                {m.icon}
                <div>
                  <span className="text-xs font-medium block">{m.label}</span>
                  <span className="text-[10px] opacity-70">{m.desc}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Benchmark Type */}
        <div className="space-y-2">
          <label
            htmlFor="benchmark-select"
            className="flex items-center gap-1.5 text-xs font-medium text-text-secondary uppercase tracking-wider"
          >
            <Layers className="h-3.5 w-3.5" />
            Benchmark
          </label>
          <select
            id="benchmark-select"
            value={benchmarkType}
            onChange={(e) => setBenchmarkType(e.target.value as BenchmarkType)}
            className={cn(
              'w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm',
              'text-text-primary',
              'focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20',
              'transition-colors appearance-none',
            )}
          >
            {BENCHMARKS.map((b) => (
              <option key={b.value} value={b.value}>
                {b.label}
              </option>
            ))}
          </select>
        </div>

        {/* Divider */}
        <div className="border-t border-border-subtle" />

        {/* Health */}
        <HealthIndicator />

        {/* Stats */}
        <StatsPanel />
      </div>

      {/* Footer actions */}
      <div className="px-4 py-3 border-t border-border-subtle">
        <button
          onClick={clearMessages}
          disabled={messageCount === 0}
          className={cn(
            'w-full flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs',
            'transition-colors',
            messageCount > 0
              ? 'text-error/80 hover:bg-error/10 hover:text-error'
              : 'text-text-muted cursor-not-allowed',
          )}
        >
          <Trash2 className="h-3.5 w-3.5" />
          Clear conversation ({messageCount})
        </button>
      </div>
    </aside>
  )
}
