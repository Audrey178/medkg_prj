import { cn } from '@/lib/utils'
import type { ChatMessage as ChatMessageType } from '@/types'
import { AnswerCard } from '@/components/answer/AnswerCard'
import { EntityBadges } from '@/components/answer/EntityBadges'
import { SourcesList } from '@/components/answer/SourcesList'
import { MetricsBar } from '@/components/answer/MetricsBar'
import { User, Bot, AlertTriangle } from 'lucide-react'

interface ChatMessageProps {
  message: ChatMessageType
}

export function ChatMessage({ message }: ChatMessageProps) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end animate-fade-in">
        <div className="flex items-start gap-2.5 max-w-[75%]">
          <div
            className={cn(
              'rounded-2xl rounded-tr-sm px-4 py-2.5',
              'bg-primary text-white',
            )}
          >
            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {message.query}
            </p>
          </div>
          <div className="flex-shrink-0 h-7 w-7 rounded-full bg-primary/20 flex items-center justify-center mt-0.5">
            <User className="h-3.5 w-3.5 text-primary" />
          </div>
        </div>
      </div>
    )
  }

  if (message.role === 'error') {
    return (
      <div className="flex justify-start animate-fade-in">
        <div className="flex items-start gap-2.5 max-w-[75%]">
          <div className="flex-shrink-0 h-7 w-7 rounded-full bg-error/20 flex items-center justify-center mt-0.5">
            <AlertTriangle className="h-3.5 w-3.5 text-error" />
          </div>
          <div className="rounded-2xl rounded-tl-sm px-4 py-2.5 bg-error/10 border border-error/20">
            <p className="text-sm text-error">{message.errorMessage}</p>
          </div>
        </div>
      </div>
    )
  }

  // Assistant message
  const response = message.response
  if (!response) return null

  return (
    <div className="flex justify-start animate-slide-up">
      <div className="flex items-start gap-2.5 max-w-[85%]">
        <div className="flex-shrink-0 h-7 w-7 rounded-full bg-accent/20 flex items-center justify-center mt-0.5">
          <Bot className="h-3.5 w-3.5 text-accent" />
        </div>
        <div
          className={cn(
            'rounded-2xl rounded-tl-sm px-4 py-3',
            'bg-surface-elevated border border-border-subtle',
          )}
        >
          <AnswerCard response={response} />
          <EntityBadges entities={response.matched_entities} />
          <SourcesList sources={response.sources} />
          <MetricsBar response={response} />
        </div>
      </div>
    </div>
  )
}
