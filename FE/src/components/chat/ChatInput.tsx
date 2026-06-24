import { useState, useRef, useEffect } from 'react'
import { Send, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ChatInputProps {
  onSend: (query: string) => void
  disabled: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [value])

  const handleSubmit = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    // Reset height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="relative">
      <div
        className={cn(
          'flex items-end gap-2 rounded-xl border transition-colors',
          'bg-surface-elevated border-border',
          'focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/20',
        )}
      >
        <textarea
          ref={textareaRef}
          id="chat-input"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a biomedical question…"
          disabled={disabled}
          rows={1}
          className={cn(
            'flex-1 resize-none bg-transparent px-4 py-3 text-sm',
            'text-text-primary placeholder:text-text-muted',
            'outline-none disabled:opacity-50',
            'min-h-[44px] max-h-[160px]',
          )}
        />
        <button
          onClick={handleSubmit}
          disabled={disabled || !value.trim()}
          aria-label="Send message"
          className={cn(
            'flex items-center justify-center h-9 w-9 rounded-lg m-1.5',
            'transition-all duration-150',
            value.trim() && !disabled
              ? 'bg-primary text-white hover:bg-primary-hover active:scale-95'
              : 'bg-transparent text-text-muted cursor-not-allowed',
          )}
        >
          {disabled ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </button>
      </div>
      <p className="text-[10px] text-text-muted mt-1.5 text-center">
        Press Enter to send · Shift+Enter for new line
      </p>
    </div>
  )
}
