import type { QueryResponse } from '@/types'
import { cn } from '@/lib/utils'
import { CheckCircle, XCircle, List, FileText, HelpCircle } from 'lucide-react'

interface AnswerCardProps {
  response: QueryResponse
}

export function AnswerCard({ response }: AnswerCardProps) {
  const { answer, question_type } = response

  if (response.error) {
    return (
      <div className="rounded-lg bg-error/10 border border-error/20 p-3">
        <p className="text-sm text-error">{response.error}</p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <QuestionTypeBadge type={question_type} />
      {renderAnswer(answer, question_type)}
    </div>
  )
}

function QuestionTypeBadge({ type }: { type: string }) {
  const config: Record<string, { icon: React.ReactNode; label: string; color: string }> = {
    yesno: { icon: <CheckCircle className="h-3 w-3" />, label: 'Yes/No', color: 'text-success bg-success/10' },
    factoid: { icon: <HelpCircle className="h-3 w-3" />, label: 'Factoid', color: 'text-primary bg-primary-soft' },
    list: { icon: <List className="h-3 w-3" />, label: 'List', color: 'text-accent bg-accent/10' },
    summary: { icon: <FileText className="h-3 w-3" />, label: 'Summary', color: 'text-warning bg-warning/10' },
  }

  const c = config[type] ?? { icon: <HelpCircle className="h-3 w-3" />, label: type, color: 'text-text-secondary bg-surface-elevated' }

  return (
    <span className={cn('inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium', c.color)}>
      {c.icon}
      {c.label}
    </span>
  )
}

function renderAnswer(answer: unknown, questionType: string): React.ReactNode {
  if (answer === null || answer === undefined) {
    return <p className="text-sm text-text-muted italic">No answer generated.</p>
  }

  // Handle structured JSON answers
  if (typeof answer === 'object' && answer !== null) {
    const obj = answer as Record<string, unknown>

    // Yes/No type
    if ('answer' in obj && 'explanation' in obj && questionType === 'yesno') {
      const isYes = String(obj.answer).toLowerCase().includes('yes')
      return (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            {isYes ? (
              <CheckCircle className="h-5 w-5 text-success" />
            ) : (
              <XCircle className="h-5 w-5 text-error" />
            )}
            <span className="text-lg font-semibold">{String(obj.answer)}</span>
          </div>
          <p className="text-sm text-text-secondary leading-relaxed">
            {String(obj.explanation)}
          </p>
        </div>
      )
    }

    // List type
    if ('answer' in obj && Array.isArray(obj.answer)) {
      return (
        <div className="space-y-2">
          <ol className="list-decimal list-inside space-y-1">
            {(obj.answer as unknown[]).map((item, i) => (
              <li key={i} className="text-sm text-text-primary">
                {String(item)}
              </li>
            ))}
          </ol>
          {'explanation' in obj && (
            <p className="text-sm text-text-secondary leading-relaxed mt-2">
              {String(obj.explanation)}
            </p>
          )}
        </div>
      )
    }

    // Summary type with key_points
    if ('answer' in obj && 'key_points' in obj) {
      return (
        <div className="space-y-2">
          <p className="text-sm text-text-primary leading-relaxed">
            {String(obj.answer)}
          </p>
          {Array.isArray(obj.key_points) && obj.key_points.length > 0 && (
            <div className="mt-2">
              <p className="text-xs font-medium text-text-secondary mb-1">Key Points:</p>
              <ul className="space-y-1">
                {(obj.key_points as unknown[]).map((point, i) => (
                  <li key={i} className="text-sm text-text-secondary flex items-start gap-1.5">
                    <span className="text-primary mt-1">•</span>
                    {String(point)}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )
    }

    // Generic object with answer field
    if ('answer' in obj) {
      return (
        <div className="space-y-2">
          <p className="text-sm text-text-primary leading-relaxed font-medium">
            {String(obj.answer)}
          </p>
          {'explanation' in obj && (
            <p className="text-sm text-text-secondary leading-relaxed">
              {String(obj.explanation)}
            </p>
          )}
        </div>
      )
    }

    // Fallback: render JSON
    return (
      <pre className="text-xs font-mono bg-surface-elevated rounded-lg p-3 overflow-x-auto text-text-secondary">
        {JSON.stringify(answer, null, 2)}
      </pre>
    )
  }

  // Simple string answer
  return (
    <p className="text-sm text-text-primary leading-relaxed">
      {String(answer)}
    </p>
  )
}
