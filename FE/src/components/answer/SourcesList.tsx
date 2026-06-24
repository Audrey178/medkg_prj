import { useState } from 'react'
import { BookOpen, ChevronDown, ChevronUp } from 'lucide-react'
import { cn } from '@/lib/utils'

interface SourcesListProps {
  sources: string[]
}

export function SourcesList({ sources }: SourcesListProps) {
  const [expanded, setExpanded] = useState(false)

  if (sources.length === 0) return null

  const visibleSources = expanded ? sources : sources.slice(0, 3)

  return (
    <div className="mt-3">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1.5 mb-2 text-xs font-medium text-text-secondary hover:text-text-primary transition-colors"
      >
        <BookOpen className="h-3 w-3" />
        Sources ({sources.length})
        {sources.length > 3 && (
          expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />
        )}
      </button>
      <ul className="space-y-1">
        {visibleSources.map((source, i) => (
          <li
            key={i}
            className={cn(
              'text-xs text-text-secondary pl-4 relative',
              'before:content-[""] before:absolute before:left-1 before:top-1.5',
              'before:h-1 before:w-1 before:rounded-full before:bg-text-muted',
            )}
          >
            {source}
          </li>
        ))}
      </ul>
    </div>
  )
}
