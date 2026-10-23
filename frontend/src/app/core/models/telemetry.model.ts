export interface CasStats {
  logical_files_count: number;
  files_count?: number;
  unique_contents_count: number;
  physical_contents_count?: number;
  virtual_size_bytes: number;
  real_size_bytes: number;
  physical_size_bytes?: number;
  saved_size_bytes: number;
  saved_bytes?: number;
  saved_percent: number;
  savings_percent?: number;
}

export interface ToolSavings {
  total_baseline_tokens: number;
  total_returned_tokens: number;
  total_saved_tokens: number;
  saved_tokens?: number;
  avg_saved_percent: number;
  saved_percent?: number;
  calls_count: number;
}

export interface TokenSavingsSummary {
  total_saved_tokens: number;
  total_saved?: number;
  saved_tokens?: number;
  avg_saved_percent: number;
  average_saved_percent?: number;
  overall_saved_percent: number;
  total_baseline_tokens: number;
  total_returned_tokens: number;
  total_calls: number;
  by_tool: Record<string, ToolSavings>;
  tools_breakdown?: Record<string, ToolSavings>;
}

export interface WatcherStatus {
  active: boolean;
  watched_paths_count: number;
}

export interface ProjectSummary {
  id: string;
  name: string;
  targets_count: number;
}

export interface TelemetryMetrics {
  cas: CasStats;
  token_savings: TokenSavingsSummary;
  watcher: WatcherStatus;
  projects: ProjectSummary[];
}
