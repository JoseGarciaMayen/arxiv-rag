import { useState, useEffect } from 'react'
import type { Message, UploadedFile } from '../types'

export function useChat() {
    const [messages, setMessages] = useState<Message[]>([])
    const [isLoading, setIsLoading] = useState(false)
    const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])

    useEffect(() => {
        fetch('/api/documents')
            .then(r => r.json())
            .then(data => setUploadedFiles(data.documents ?? []))
            .catch(() => { })
    }, [])

    const sendMessage = async (question: string, baseMessages?: Message[]) => {
        const userMessage: Message = {
            id: crypto.randomUUID(),
            role: 'user',
            content: question,
            timestamp: new Date()
        }
        setMessages(prev => [...prev, userMessage])
        setIsLoading(true)

        try {
            const history = (baseMessages ?? messages)
                .slice(-6)
                .map(m => ({ role: m.role, content: m.content }))

            const res = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, history })
            })
            if (!res.ok) throw new Error(`API error: ${res.status}`)
            const data = await res.json()
            setMessages(prev => [...prev, {
                id: crypto.randomUUID(),
                role: 'assistant',
                content: data.answer,
                timestamp: new Date()
            }])
        } catch {
            setMessages(prev => [...prev, {
                id: crypto.randomUUID(),
                role: 'assistant',
                content: 'Error connecting to the server.',
                timestamp: new Date()
            }])
        } finally {
            setIsLoading(false)
        }
    }

    const uploadPDF = async (file: File) => {
        const formData = new FormData()
        formData.append('file', file)
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        if (!res.ok) throw new Error('Upload failed')
        const data = await res.json()
        const newFile: UploadedFile = {
            name: file.name,
            chunks: data.chunks,
            uploadedAt: new Date().toISOString()
        }
        setUploadedFiles(prev => {
            const exists = prev.find(f => f.name === file.name)
            if (exists) return prev.map(f => f.name === file.name ? newFile : f)
            return [...prev, newFile]
        })
        return data
    }

    const deleteFile = async (filename: string) => {
        try {
            const res = await fetch(`/api/document/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            });
            if (res.ok) {
                setUploadedFiles(prev => prev.filter(f => f.name !== filename));
            } else {
                console.error(`Failed to delete ${filename}`);
            }
        } catch (err) {
            console.error(`Error deleting ${filename}:`, err);
        }
    };

    const clearFiles = async () => {
        const results = await Promise.allSettled(
            uploadedFiles.map(async (file) => {
                const res = await fetch(`/api/document/${encodeURIComponent(file.name)}`, {
                    method: 'DELETE'
                });
                if (!res.ok) throw new Error(`Failed to delete ${file.name}`);
                return file.name;
            })
        );

        const successfulDeletes = new Set(
            results
                .filter((r): r is PromiseFulfilledResult<string> => r.status === 'fulfilled')
                .map(r => r.value)
        );

        setUploadedFiles(prev => prev.filter(f => !successfulDeletes.has(f.name)));
    };

    const regenerateMessage = (index: number) => {
        let userIndex = index - 1;
        while (userIndex >= 0 && messages[userIndex].role !== 'user') {
            userIndex--;
        }
        if (userIndex < 0) return;

        const question = messages[userIndex].content;
        const truncated = messages.slice(0, userIndex);

        setMessages(truncated);
        sendMessage(question, truncated);
    }

    return { messages, isLoading, uploadedFiles, sendMessage, uploadPDF, deleteFile, clearFiles, regenerateMessage }
}