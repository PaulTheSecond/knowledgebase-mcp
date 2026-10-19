export interface QueueJob {
  id: number;
  repo_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'halted';
  attempts: number;
  last_error?: string | null;
  created_at: string;
  updated_at?: string;
  started_at?: string | null;
  finished_at?: string | null;
  duration?: string | null;
  duration_seconds?: number | null;
}

export interface SystemStatus {
  status: 'idle' | 'processing' | 'syncing' | 'halted' | string;
  last_error?: string | null;
  last_indexed_at?: string | null;
  repositories: RepositoryStatusItem[];
  chat_enabled: boolean;
}

export interface RepositoryStatusItem {
  id: string;
  url: string;
  branch: string;
  files_count: number;
  chunks_count: number;
  vectors_count: number;
}
