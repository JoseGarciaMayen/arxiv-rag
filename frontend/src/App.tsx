import { useState, useRef, useEffect } from 'react'
import { ChatWindow } from './components/ChatWindow'
import { useChat } from './hooks/useChat'

export default function App() {
  const { messages, isLoading, uploadedFiles, sendMessage, uploadPDF, deleteFile, clearFiles, regenerateMessage } = useChat()
  const [showFiles, setShowFiles] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const sidebarRef = useRef<HTMLDivElement>(null)
  const menuButtonRef = useRef<HTMLButtonElement>(null)
  const [ingestingCount, setIngestingCount] = useState(0)
  const [isClearing, setIsClearing] = useState(false)

  // Dark mode state
  const [isDarkMode, setIsDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
      return document.documentElement.classList.contains('dark') || 
             window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  })

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [isDarkMode])

  useEffect(() => {
    if (ingestingCount > 0) {
      setShowFiles(true)
    }
  }, [ingestingCount])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        showFiles &&
        sidebarRef.current &&
        !sidebarRef.current.contains(event.target as Node) &&
        menuButtonRef.current &&
        !menuButtonRef.current.contains(event.target as Node)
      ) {
        setShowFiles(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [showFiles])

  const handleFile = async (file: File) => {
    if (!file.name.endsWith('.pdf')) return
    setIngestingCount(prev => prev + 1)
    try {
      await uploadPDF(file)
    } finally {
      setIngestingCount(prev => prev - 1)
    }
  }

  const handleClearAll = async () => {
    setIsClearing(true)
    await clearFiles()
    setIsClearing(false)
  }

  return (
    <div className="flex h-screen bg-[var(--color-bg-main)] text-[var(--color-text-primary)] font-sans selection:bg-[var(--color-accent-blue)]/20 overflow-hidden">
      
      {/* Sidebar Drawer */}
      <div 
        ref={sidebarRef}
        className={`fixed inset-y-0 left-0 z-40 w-72 bg-[var(--color-bg-card)] transform transition-transform duration-300 ease-in-out flex flex-col border-r border-[var(--color-border)] ${showFiles ? 'translate-x-0' : '-translate-x-full'}`}
      >
        <div className="flex items-center justify-between p-4 border-b border-[var(--color-border)]">
          <span className="text-sm font-medium text-[var(--color-text-secondary)]">Ingested Documents</span>
          <button onClick={() => setShowFiles(false)} className="p-2 hover:bg-gray-200 dark:hover:bg-white/5 rounded-full text-gray-500 transition-colors">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4">
          <div className="flex justify-between items-center mb-4">
            <span className="text-xs text-[var(--color-text-secondary)] uppercase tracking-wider">{uploadedFiles?.length ?? 0} Docs</span>
            {(uploadedFiles?.length ?? 0) > 0 && ingestingCount === 0 && (
              <button 
                onClick={handleClearAll} 
                disabled={isClearing}
                className="text-xs text-red-500 hover:text-red-600 transition-colors disabled:opacity-50"
              >
                {isClearing ? 'Clearing...' : 'Clear All'}
              </button>
            )}
          </div>
          
          <div className="flex flex-col gap-2">
            {ingestingCount > 0 && (
              <div className="flex items-center gap-3 bg-[var(--color-bg-main)] border border-[var(--color-border)] rounded-xl p-3 shadow-sm opacity-70 animate-pulse">
                <div className="w-8 h-8 rounded-lg bg-[var(--color-bg-card)] flex items-center justify-center shrink-0">
                  <svg className="animate-spin h-4 w-4 text-[var(--color-accent-blue)]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                </div>
                <div className="flex flex-col min-w-0 flex-1">
                  <div className="h-4 bg-[var(--color-bg-card)] rounded w-3/4 mb-1.5"></div>
                  <div className="h-3 bg-[var(--color-bg-card)] rounded w-1/2"></div>
                </div>
              </div>
            )}
            {ingestingCount === 0 && (uploadedFiles?.length ?? 0) === 0 && (
              <div className="text-sm text-[var(--color-text-secondary)] text-center py-8">
                No documents yet. Use the + button in the chat to upload one.
              </div>
            )}
            {uploadedFiles?.map(file => (
              <div key={file.name} className="flex items-center gap-3 bg-[var(--color-bg-main)] border border-[var(--color-border)] rounded-xl p-3 shadow-sm group">
                <div className="w-8 h-8 rounded-lg bg-[var(--color-bg-card)] flex items-center justify-center shrink-0">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-[var(--color-accent-blue)]">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                  </svg>
                </div>
                <div className="flex flex-col min-w-0 flex-1">
                  <span className="text-sm text-[var(--color-text-primary)] truncate font-medium">{file.name}</span>
                  <span className="text-xs text-[var(--color-text-secondary)]">{file.chunks} chunks</span>
                </div>
                <button 
                  onClick={() => deleteFile(file.name)}
                  className="p-1.5 hover:bg-red-500/10 text-gray-400 hover:text-red-500 rounded-md opacity-0 group-hover:opacity-100 transition-all shrink-0 focus:opacity-100"
                  title="Delete document"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 6h18" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    <line x1="10" y1="11" x2="10" y2="17" />
                    <line x1="14" y1="11" x2="14" y2="17" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Backdrop for sidebar on mobile */}
      {showFiles && (
        <div className="fixed inset-0 bg-black/20 z-30 lg:hidden" onClick={() => setShowFiles(false)} />
      )}

      {/* Main Content Area */}
      <div className="flex flex-col flex-1 h-full min-w-0 relative">
        {/* Top bar */}
        <header className="flex items-center justify-between px-4 py-3 h-14 shrink-0 bg-[var(--color-bg-main)]">
          <div className="flex items-center gap-4">
            <button 
              ref={menuButtonRef}
              onClick={() => setShowFiles(!showFiles)}
              className="p-2 hover:bg-gray-200 dark:hover:bg-white/5 rounded-full transition-colors text-[var(--color-text-secondary)]"
              title="Main menu"
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            </button>
            <div className="text-[22px] text-[var(--color-text-primary)] font-normal tracking-tight flex items-baseline gap-4">
              ArXiv Papers RAG
              <span className="text-[13px] text-[var(--color-text-secondary)] font-medium bg-[var(--color-bg-card)] px-2.5 py-0.5 rounded-full border border-[var(--color-border)] hidden sm:inline-block">
                {uploadedFiles.length === 1 ? '1 document loaded' : `${uploadedFiles.length} documents loaded`}
              </span>
            </div>
          </div>
          
          <div className="flex items-center">
            <button 
              onClick={() => setIsDarkMode(!isDarkMode)} 
              className="p-2 hover:bg-gray-200 dark:hover:bg-white/5 rounded-full transition-colors text-[var(--color-text-secondary)]"
              title={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
            >
              {isDarkMode ? (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="5" />
                  <line x1="12" y1="1" x2="12" y2="3" />
                  <line x1="12" y1="21" x2="12" y2="23" />
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                  <line x1="1" y1="12" x2="3" y2="12" />
                  <line x1="21" y1="12" x2="23" y2="12" />
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                </svg>
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                </svg>
              )}
            </button>
          </div>
        </header>

        {/* Chat Window Container */}
        <div className="flex-1 overflow-hidden relative">
          <ChatWindow
            messages={messages}
            isLoading={isLoading}
            onSend={sendMessage}
            ingestingCount={ingestingCount}
            onFileSelect={() => fileInputRef.current?.click()}
            onRegenerate={regenerateMessage}
          />
        </div>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          style={{ display: 'none' }}
          onChange={e => Array.from(e.target.files || []).forEach(handleFile)}
        />
      </div>
    </div>
  )
}