import { Repository } from './repository.model';

export interface SystemSettings {
  ui: {
    default_locale: string;
    default_theme: string;
  };
  llm: {
    api_base: string;
    model: string;
    is_env_api_base: boolean;
    is_env_model: boolean;
    chat_enabled: boolean;
  };
  embeddings: {
    enabled: boolean;
    model_name: string;
  };
  integrations: {
    webhook_secret_set: boolean;
    is_env_webhook_secret: boolean;
    github_pat_set: boolean;
    is_env_github_pat: boolean;
  };
  repositories: Repository[];
}

export interface UpdateSettingsPayload {
  default_locale?: string;
  default_theme?: string;
  chat_enabled?: boolean;
  llm_base_url?: string;
  llm_model?: string;
  embeddings_enabled?: boolean;
  embeddings_model?: string;
  webhook_secret?: string;
}
