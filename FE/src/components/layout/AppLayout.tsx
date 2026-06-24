import { useSettingsStore } from '@/stores/settingsStore'
import { Sidebar } from '@/components/layout/Sidebar'
import { cn } from '@/lib/utils'
import { PanelLeftClose, PanelLeft } from 'lucide-react'

interface AppLayoutProps {
  children: React.ReactNode
}

export function AppLayout({ children }: AppLayoutProps) {
  const sidebarOpen = useSettingsStore((s) => s.sidebarOpen)
  const toggleSidebar = useSettingsStore((s) => s.toggleSidebar)

  return (
    <div className="flex w-full h-dvh overflow-hidden bg-bg">
      {/* Sidebar */}
      <div
        className={cn(
          'flex-shrink-0 border-r border-border-subtle bg-surface transition-all duration-300 ease-in-out',
          sidebarOpen ? 'w-72' : 'w-0',
          'overflow-hidden',
        )}
      >
        <div className="w-72 h-full">
          <Sidebar />
        </div>
      </div>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="flex items-center h-12 px-4 border-b border-border-subtle bg-surface/50 backdrop-blur-sm flex-shrink-0">
          <button
            onClick={toggleSidebar}
            aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
            className="h-8 w-8 flex items-center justify-center rounded-lg text-text-secondary hover:text-text-primary hover:bg-surface-hover transition-colors"
          >
            {sidebarOpen ? (
              <PanelLeftClose className="h-4 w-4" />
            ) : (
              <PanelLeft className="h-4 w-4" />
            )}
          </button>

          {!sidebarOpen && (
            <span className="ml-2 text-sm font-medium text-text-secondary animate-fade-in">
              ChronoMedKG
            </span>
          )}
        </header>

        {/* Content */}
        <main className="flex-1 overflow-hidden">{children}</main>
      </div>
    </div>
  )
}
