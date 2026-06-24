import { cn } from '@/lib/utils'
import { Tag } from 'lucide-react'

interface EntityBadgesProps {
  entities: string[]
}

export function EntityBadges({ entities }: EntityBadgesProps) {
  if (entities.length === 0) return null

  return (
    <div className="mt-3">
      <div className="flex items-center gap-1.5 mb-2 text-xs font-medium text-text-secondary">
        <Tag className="h-3 w-3" />
        Matched Entities ({entities.length})
      </div>
      <div className="flex flex-wrap gap-1.5">
        {entities.map((entity) => (
          <button
            key={entity}
            onClick={() => navigator.clipboard.writeText(entity)}
            title="Click to copy"
            className={cn(
              'inline-flex items-center px-2 py-0.5 rounded-md text-xs',
              'bg-accent/10 text-accent border border-accent/20',
              'hover:bg-accent/20 hover:border-accent/30',
              'transition-colors cursor-pointer',
            )}
          >
            {entity}
          </button>
        ))}
      </div>
    </div>
  )
}
