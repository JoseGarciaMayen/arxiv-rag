export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export interface UploadedFile {
  name: string
  chunks: number
  uploadedAt: string
}