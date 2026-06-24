import { useRef, useEffect, useCallback } from "react";
import { useChatStore } from "@/stores/chatStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { queryApi } from "@/api/client";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { ChatInput } from "@/components/chat/ChatInput";
import { Brain, Sparkles, Stethoscope, Pill, Dna } from "lucide-react";
import { cn } from "@/lib/utils";

const EXAMPLE_QUESTIONS = [
  {
    icon: <Stethoscope className="h-4 w-4" />,
    q: "Gen nào thường bị đột biến trong bệnh xơ nang (cystic fibrosis)?",
  },
  {
    icon: <Pill className="h-4 w-4" />,
    q: "Liệu aspirin có làm giảm nguy cơ ung thư đại trực tràng không?",
  },
  {
    icon: <Dna className="h-4 w-4" />,
    q: "Thuốc nào là lựa chọn điều trị đầu tay cho bệnh Parkinson?",
  },
  {
    icon: <Sparkles className="h-4 w-4" />,
    q: "Mô tả vai trò của microglia trong bệnh Alzheimer.",
  },
];

export function ChatPanel() {
  const messages = useChatStore((s) => s.messages);
  const isLoading = useChatStore((s) => s.isLoading);
  const addUserMessage = useChatStore((s) => s.addUserMessage);
  const addBotMessage = useChatStore((s) => s.addBotMessage);
  const addErrorMessage = useChatStore((s) => s.addErrorMessage);
  const setLoading = useChatStore((s) => s.setLoading);

  const mode = useSettingsStore((s) => s.mode);
  const benchmarkType = useSettingsStore((s) => s.benchmarkType);

  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSend = useCallback(
    async (query: string) => {
      const msgId = addUserMessage(query);
      setLoading(true);

      try {
        const response = await queryApi({
          query,
          mode,
          benchmark_type: benchmarkType,
          options: {},
        });
        addBotMessage(msgId, response);
      } catch (err: unknown) {
        let errorMsg = "An unexpected error occurred.";
        if (err && typeof err === "object" && "response" in err) {
          const axiosErr = err as {
            response?: { status?: number; data?: { detail?: string } };
          };
          const status = axiosErr.response?.status;
          const detail = axiosErr.response?.data?.detail;
          if (status === 401) {
            errorMsg = "Invalid API key. Please check your key in the sidebar.";
          } else if (status === 429) {
            errorMsg = "Rate limit exceeded. Please try again later.";
          } else if (status === 503) {
            errorMsg =
              detail ?? "Service unavailable. The pipeline may not be ready.";
          } else {
            errorMsg = detail ?? `Request failed (${status}).`;
          }
        } else if (err instanceof Error) {
          if (err.message.includes("Network Error")) {
            errorMsg = "Cannot reach the API server. Is it running?";
          } else {
            errorMsg = err.message;
          }
        }
        addErrorMessage(msgId, errorMsg);
      } finally {
        setLoading(false);
      }
    },
    [
      mode,
      benchmarkType,
      addUserMessage,
      addBotMessage,
      addErrorMessage,
      setLoading,
    ],
  );

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-6 space-y-4"
      >
        {isEmpty && <EmptyState onSelect={handleSend} />}

        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}

        {isLoading && <TypingIndicator />}
      </div>

      {/* Input area */}
      <div className="border-t border-border-subtle px-4 py-4 bg-surface/50 backdrop-blur-sm">
        <div className="max-w-3xl mx-auto">
          <ChatInput onSend={handleSend} disabled={isLoading} />
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onSelect }: { onSelect: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center py-12 animate-fade-in">
      <div className="h-16 w-16 rounded-2xl bg-primary-soft flex items-center justify-center mb-6">
        <Brain className="h-8 w-8 text-primary" />
      </div>
      <h2 className="text-xl font-semibold text-text-primary mb-2">
        ChronoMedKG
      </h2>
      <p className="text-sm text-text-secondary max-w-md mb-8">
        Biomedical Question Answering powered by Knowledge Graph
        Retrieval-Augmented Generation. Ask any medical or biomedical question
        below.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
        {EXAMPLE_QUESTIONS.map(({ icon, q }) => (
          <button
            key={q}
            onClick={() => onSelect(q)}
            className={cn(
              "flex items-center gap-2.5 text-left rounded-xl px-4 py-3",
              "border border-border-subtle bg-surface-elevated",
              "text-sm text-text-secondary",
              "hover:bg-surface-hover hover:border-border hover:text-text-primary",
              "transition-all duration-150",
              "disabled:opacity-50 disabled:cursor-not-allowed",
            )}
          >
            <span className="flex-shrink-0 text-primary">{icon}</span>
            <span className="line-clamp-2">{q}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex justify-start animate-fade-in">
      <div className="flex items-start gap-2.5">
        <div className="flex-shrink-0 h-7 w-7 rounded-full bg-accent/20 flex items-center justify-center mt-0.5">
          <Brain className="h-3.5 w-3.5 text-accent" />
        </div>
        <div className="rounded-2xl rounded-tl-sm px-4 py-3 bg-surface-elevated border border-border-subtle">
          <div className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-text-muted typing-dot" />
            <span className="h-1.5 w-1.5 rounded-full bg-text-muted typing-dot" />
            <span className="h-1.5 w-1.5 rounded-full bg-text-muted typing-dot" />
          </div>
        </div>
      </div>
    </div>
  );
}
