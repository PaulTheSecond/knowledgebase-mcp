export interface ChatSource {
  repo_id?: string;
  filepath: string;
  lines: string;
  line_start?: number;
  line_end?: number;
}

export interface ChatMessage {
  id?: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  sources?: ChatSource[];
  timestamp?: number;
}
