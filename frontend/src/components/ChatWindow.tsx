import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '../types'

interface Props {
    messages: Message[]
    isLoading: boolean
    ingestingCount: number
    onSend: (question: string) => void
    onFileSelect: () => void
    onRegenerate?: (index: number) => void
}

export function ChatWindow({ messages, isLoading, ingestingCount, onSend, onFileSelect, onRegenerate }: Props) {
    const [input, setInput] = useState('')
    const scrollContainerRef = useRef<HTMLDivElement>(null)
    const textareaRef = useRef<HTMLTextAreaElement>(null)
    const [copiedId, setCopiedId] = useState<string | null>(null)

    const handleCopy = async (id: string, text: string) => {
        try {
            if (navigator?.clipboard?.writeText) {
                await navigator.clipboard.writeText(text);
            } else {
                // Fallback para entornos sin HTTPS (localhost en red local)
                const textArea = document.createElement("textarea");
                textArea.value = text;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand("copy");
                textArea.remove();
            }
            setCopiedId(id);
            setTimeout(() => setCopiedId(null), 2000);
        } catch (err) {
            console.error('Failed to copy', err);
        }
    }

    const isUploading = ingestingCount > 0
    const processingText = ingestingCount === 1 ? "Processing 1 document..." : `Processing ${ingestingCount} documents...`
    const placeholderText = isUploading
        ? (ingestingCount === 1 ? "Ingesting 1 document... Please wait." : `Ingesting ${ingestingCount} documents... Please wait.`)
        : "Ask a question..."

    useEffect(() => {
        const el = scrollContainerRef.current
        if (el) el.scrollTop = el.scrollHeight
    }, [messages, isLoading, isUploading])

    const handleSend = () => {
        if (!input.trim() || isLoading || isUploading) return
        onSend(input.trim())
        setInput('')
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto'
        }
    }

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSend()
        }
    }

    const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setInput(e.target.value)
        e.target.style.height = 'auto'
        e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`
    }

    return (
        <div className="flex flex-col h-full relative">
            {/* Messages Area */}
            <div ref={scrollContainerRef} className="flex-1 overflow-y-auto pb-56 px-4">
                <div className="max-w-[850px] mx-auto min-h-full flex flex-col">
                    {messages.length === 0 ? (
                        <div className="flex flex-col flex-1 mt-[20vh] items-center text-center">
                            <h1 className="text-[44px] md:text-[56px] font-medium leading-tight tracking-tight mb-2 text-[var(--color-text-primary)]">
                                Hello,
                            </h1>
                            <h2 className="text-[28px] md:text-[36px] font-medium text-[var(--color-text-secondary)] leading-tight tracking-tight">
                                How can I help you with your papers today?
                            </h2>
                            {isUploading && (
                                <div className="mt-12 flex items-center gap-3 bg-[var(--color-bg-card)] px-5 py-2.5 rounded-full border border-[var(--color-border)] shadow-sm">
                                    <svg className="animate-spin h-5 w-5 text-[var(--color-accent-blue)]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    <span className="text-[16px] text-[var(--color-text-secondary)] font-medium">{processingText}</span>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="py-8 flex flex-col gap-8 w-full">
                            {messages.map((msg, index) => (
                                <div key={msg.id} className={`flex gap-6 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'} w-full`}>
                                    {/* Bubble */}
                                    <div className={`max-w-[85%] text-[17px] leading-relaxed font-medium ${msg.role === 'user'
                                        ? 'bg-[var(--color-bg-card)] px-6 py-4 rounded-[28px] text-[var(--color-text-primary)]'
                                        : 'text-[var(--color-text-primary)] pt-1.5'
                                        }`}>
                                        {msg.role === 'assistant' ? (
                                            <div className="w-full">
                                                <ReactMarkdown
                                                    remarkPlugins={[remarkGfm]}
                                                    components={{
                                                        code({ className, children }: { className?: string; children?: React.ReactNode }) {
                                                            const isBlock = className?.includes('language-')
                                                            return isBlock ? (
                                                                <pre className="bg-[#1a1a1a] border border-white/10 rounded-xl p-4 overflow-x-auto my-3">
                                                                    <code className="text-emerald-300 text-xs font-mono">{children}</code>
                                                                </pre>
                                                            ) : (
                                                                <code className="bg-white/8 text-emerald-300 px-1.5 py-0.5 rounded text-xs font-mono">{children}</code>
                                                            )
                                                        },
                                                        p: ({ children }: { children?: React.ReactNode }) => <p className="mb-3 last:mb-0">{children}</p>,
                                                        ul: ({ children }: { children?: React.ReactNode }) => <ul className="list-disc list-outside ml-4 mb-3 space-y-1">{children}</ul>,
                                                        ol: ({ children }: { children?: React.ReactNode }) => <ol className="list-decimal list-outside ml-4 mb-3 space-y-1">{children}</ol>,
                                                        li: ({ children }: { children?: React.ReactNode }) => <li className="text-gray-300">{children}</li>,
                                                        strong: ({ children }: { children?: React.ReactNode }) => <strong className="font-semibold text-white">{children}</strong>,
                                                        h1: ({ children }: { children?: React.ReactNode }) => <h1 className="text-base font-semibold text-white mb-2 mt-4">{children}</h1>,
                                                        h2: ({ children }: { children?: React.ReactNode }) => <h2 className="text-sm font-semibold text-white mb-2 mt-3">{children}</h2>,
                                                        h3: ({ children }: { children?: React.ReactNode }) => <h3 className="text-sm font-medium text-gray-200 mb-1 mt-3">{children}</h3>,
                                                        blockquote: ({ children }: { children?: React.ReactNode }) => (
                                                            <blockquote className="border-l-2 border-emerald-500/50 pl-3 my-3 text-gray-400 italic">{children}</blockquote>
                                                        ),
                                                    }}
                                                >
                                                    {msg.content}
                                                </ReactMarkdown>

                                                {/* Action buttons under assistant message */}
                                                <div className="flex items-center gap-1 mt-3 -ml-2">
                                                    <button
                                                        onClick={() => handleCopy(msg.id, msg.content)}
                                                        className="p-2 hover:bg-gray-200 dark:hover:bg-white/5 rounded-full transition-colors text-gray-400 hover:text-[var(--color-text-secondary)]"
                                                        title="Copy"
                                                    >
                                                        {copiedId === msg.id ? (
                                                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-green-500">
                                                                <polyline points="20 6 9 17 4 12" />
                                                            </svg>
                                                        ) : (
                                                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                                                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                                                            </svg>
                                                        )}
                                                    </button>
                                                    <button
                                                        onClick={() => onRegenerate && onRegenerate(index)}
                                                        disabled={isLoading || isUploading}
                                                        className="p-2 hover:bg-gray-200 dark:hover:bg-white/5 rounded-full transition-colors text-gray-400 hover:text-[var(--color-text-secondary)] disabled:opacity-50 disabled:cursor-not-allowed"
                                                        title="Regenerate"
                                                    >
                                                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                            <path d="M21 2v6h-6" />
                                                            <path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
                                                            <path d="M3 22v-6h6" />
                                                            <path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
                                                        </svg>
                                                    </button>
                                                </div>
                                            </div>
                                        ) : (
                                            <p className="whitespace-pre-wrap">{msg.content}</p>
                                        )}
                                    </div>
                                </div>
                            ))}

                            {isLoading && (
                                <div className="flex gap-6 w-full">
                                    <div className="flex items-center gap-1.5 pt-3 pl-2">
                                        <span className="w-2.5 h-2.5 bg-gray-300 dark:bg-gray-600 rounded-full animate-bounce [animation-delay:0ms]" />
                                        <span className="w-2.5 h-2.5 bg-gray-300 dark:bg-gray-600 rounded-full animate-bounce [animation-delay:150ms]" />
                                        <span className="w-2.5 h-2.5 bg-gray-300 dark:bg-gray-600 rounded-full animate-bounce [animation-delay:300ms]" />
                                    </div>
                                </div>
                            )}

                            {isUploading && messages.length > 0 && (
                                <div className="flex justify-center mt-4">
                                    <div className="flex items-center gap-3 bg-[var(--color-bg-card)] px-5 py-2.5 rounded-full border border-[var(--color-border)] shadow-sm">
                                        <svg className="animate-spin h-5 w-5 text-[var(--color-accent-blue)]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        <span className="text-[16px] text-[var(--color-text-secondary)] font-medium">{processingText}</span>
                                    </div>
                                </div>
                            )}

                        </div>
                    )}
                </div>
            </div>

            {/* Input area - fixed at bottom */}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-[var(--color-bg-main)] via-[var(--color-bg-main)] to-transparent pt-6 pb-8 px-4 pointer-events-none">
                <div className="max-w-[850px] mx-auto pointer-events-auto">
                    <div className={`relative bg-[var(--color-bg-card)] rounded-full flex items-center px-2 py-1 border border-transparent shadow-sm ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}`}>

                        {/* Attach button */}
                        <button
                            onClick={onFileSelect}
                            disabled={isUploading}
                            className={`p-3 hover:bg-gray-200 dark:hover:bg-white/5 rounded-full transition-colors text-gray-500 disabled:opacity-50 shrink-0 ml-1 ${isUploading ? 'cursor-not-allowed' : ''}`}
                            title="Attach PDF"
                        >
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <line x1="12" y1="5" x2="12" y2="19" />
                                <line x1="5" y1="12" x2="19" y2="12" />
                            </svg>
                        </button>

                        {/* Textarea */}
                        <textarea
                            ref={textareaRef}
                            value={input}
                            onChange={handleInput}
                            onKeyDown={handleKeyDown}
                            placeholder={placeholderText}
                            disabled={isUploading}
                            rows={1}
                            className={`flex-1 bg-transparent text-[var(--color-text-primary)] text-[16px] px-3 py-3 outline-none resize-none placeholder-[var(--color-text-secondary)] max-h-[150px] font-sans overflow-hidden ${isUploading ? 'cursor-not-allowed' : ''}`}
                            style={{ minHeight: '48px', paddingTop: '12px' }}
                        />

                        {/* Right side actions */}
                        <div className="flex items-center shrink-0 mr-2 gap-1.5">
                            {input.trim() && (
                                <button
                                    onClick={handleSend}
                                    disabled={isLoading || isUploading}
                                    className={`p-3 bg-[var(--color-accent-blue)] hover:bg-blue-700 rounded-full transition-colors text-white shadow-sm ${isUploading ? 'cursor-not-allowed opacity-50' : ''}`}
                                >
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                        <line x1="22" y1="2" x2="11" y2="13" />
                                        <polygon points="22 2 15 22 11 13 2 9 22 2" />
                                    </svg>
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}