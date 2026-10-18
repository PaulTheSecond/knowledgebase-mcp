export interface RepositoryAuth {
  type: 'none' | 'pat' | 'ssh';
  token_env?: string | null;
  token?: string | null;
  private_key_path?: string | null;
}

export interface RepositoryTriggers {
  webhooks?: {
    enabled: boolean;
    secret_env?: string;
  };
  cron?: string | null;
}

export interface Repository {
  id: string;
  url: string;
  branch: string;
  auth?: RepositoryAuth;
  triggers?: RepositoryTriggers;
  files_count?: number;
  chunks_count?: number;
  vectors_count?: number;
}
