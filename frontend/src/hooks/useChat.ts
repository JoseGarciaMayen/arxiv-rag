import { useState, useEffect } from 'react'
import type { Message, UploadedFile } from '../types'

export function useChat() {
    const [messages, setMessages] = useState<Message[]>([])
    const [isLoading, setIsLoading] = useState(false)
    const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
    const [documentsError, setDocumentsError] = useState<string | null>(null)
    const [deleteError, setDeleteError] = useState<string | null>(null)

    useEffect(() => {
        fetch('/api/documents')
            .then(r => {
                if (!r.ok) throw new Error(`Failed to load documents (${r.status})`)
                return r.json()
            })
            .then(data => setUploadedFiles(data.documents ?? []))
            .catch((err: unknown) =>
                setDocumentsError(
                    err instanceof Error ? err.message : 'Failed to load documents'
                )
            )
    }, [])

    const sendMessage = async (question: string, baseMessages?: Message[]) => {
        const userMessage: Message = {
            id: crypto.randomUUID(),
            role: 'user',
            content: question,
            timestamp: new Date(),
        }
        setMessages(prev => [...prev, userMessage])
        setIsLoading(true)

        const assistantId = crypto.randomUUID()
        let streamStarted = false

        try {
            const history = (baseMessages ?? messages)
                .slice(-6)
                .map(m => ({ role: m.role, content: m.content }))

            const res = await fetch('/api/query/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, history }),
            })

            if (!res.ok) throw new Error(`API error: ${res.status}`)

            const reader = res.body!.getReader()
            const decoder = new TextDecoder()
            let buffer = ''

            outer: while (true) {
                const { done, value } = await reader.read()
                if (done) break

                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split('\n')
                buffer = lines.pop() ?? ''

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue
                    const payload = line.slice(6)
                    if (payload === '[DONE]') break outer

                    try {
                        const { content } = JSON.parse(payload) as { content: string }
                        if (!content) continue

                        if (!streamStarted) {
                            streamStarted = true
                            setIsLoading(false)
                            setMessages(prev => [
                                ...prev,
                                {
                                    id: assistantId,
                                    role: 'assistant' as const,
                                    content,
                                    timestamp: new Date(),
                                },
                            ])
                        } else {
                            setMessages(prev =>
                                prev.map(m =>
                                    m.id === assistantId
                                        ? { ...m, content: m.content + content }
                                        : m
                                )
                            )
                        }
                    } catch {
                        // malformed SSE line, skip
                    }
                }
            }

            if (!streamStarted) {
                setMessages(prev => [
                    ...prev,
                    {
                        id: assistantId,
                        role: 'assistant',
                        content: 'No response received.',
                        timestamp: new Date(),
                    },
                ])
            }
        } catch {
            setMessages(prev => {
                const hasAssistant = prev.some(m => m.id === assistantId)
                if (hasAssistant) {
                    return prev.map(m =>
                        m.id === assistantId
                            ? { ...m, content: 'Error connecting to the server.' }
                            : m
                    )
                }
                return [
                    ...prev,
                    {
                        id: assistantId,
                        role: 'assistant',
                        content: 'Error connecting to the server.',
                        timestamp: new Date(),
                    },
                ]
            })
        } finally {
            setIsLoading(false)
        }
    }

    const uploadPDF = async (file: File) => {
        const formData = new FormData()
        formData.append('file', file)
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        })
        if (!res.ok) throw new Error('Upload failed')
        const data = await res.json()
        const newFile: UploadedFile = {
            name: file.name,
            chunks: data.chunks,
            uploadedAt: new Date().toISOString(),
        }
        setUploadedFiles(prev => {
            const exists = prev.find(f => f.name === file.name)
            if (exists) return prev.map(f => (f.name === file.name ? newFile : f))
            return [...prev, newFile]
        })
        return data
    }

    const deleteFile = async (filename: string) => {
        try {
            const res = await fetch(`/api/document/${encodeURIComponent(filename)}`, {
                method: 'DELETE',
            })
            if (res.ok) {
                setUploadedFiles(prev => prev.filter(f => f.name !== filename))
            } else {
                setDeleteError(`Could not delete "${filename}"`)
            }
        } catch {
            setDeleteError(`Could not delete "${filename}"`)
        }
    }

    const clearDeleteError = () => setDeleteError(null)

    const clearFiles = async () => {
        const results = await Promise.allSettled(
            uploadedFiles.map(async file => {
                const res = await fetch(
                    `/api/document/${encodeURIComponent(file.name)}`,
                    { method: 'DELETE' }
                )
                if (!res.ok) throw new Error(`Failed to delete ${file.name}`)
                return file.name
            })
        )

        const successfulDeletes = new Set(
            results
                .filter((r): r is PromiseFulfilledResult<string> => r.status === 'fulfilled')
                .map(r => r.value)
        )

        setUploadedFiles(prev => prev.filter(f => !successfulDeletes.has(f.name)))
    }

    const regenerateMessage = (index: number) => {
        let userIndex = index - 1
        while (userIndex >= 0 && messages[userIndex].role !== 'user') {
            userIndex--
        }
        if (userIndex < 0) return

        const question = messages[userIndex].content
        const truncated = messages.slice(0, userIndex)

        setMessages(truncated)
        sendMessage(question, truncated)
    }

    return {
        messages,
        isLoading,
        uploadedFiles,
        documentsError,
        deleteError,
        clearDeleteError,
        sendMessage,
        uploadPDF,
        deleteFile,
        clearFiles,
        regenerateMessage,
    }
}
