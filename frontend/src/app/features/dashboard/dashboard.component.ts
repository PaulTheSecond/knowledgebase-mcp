import { Component, inject, signal, computed, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { TelemetryMetrics, ToolSavings } from '../../core/models/telemetry.model';

export interface ToolDisplayItem {
  name: string;
  displayName: string;
  stats: ToolSavings;
}

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit, OnDestroy {
  readonly i18n = inject(I18nService);
  private readonly api = inject(ApiService);

  readonly metrics = signal<TelemetryMetrics | null>(null);
  readonly loading = signal<boolean>(true);
  readonly error = signal<string | null>(null);

  private refreshInterval?: any;
  private readonly refreshHandler = () => this.loadMetrics();

  readonly toolsList = computed<ToolDisplayItem[]>(() => {
    const data = this.metrics();
    if (!data || !data.token_savings || !data.token_savings.by_tool) {
      return [];
    }
    const tools = data.token_savings.by_tool;
    return Object.keys(tools).map(key => ({
      name: key,
      displayName: this.getToolDisplayName(key),
      stats: tools[key]
    })).sort((a, b) => (b.stats.total_saved_tokens || 0) - (a.stats.total_saved_tokens || 0));
  });

  readonly casDeduplicationRatio = computed<string>(() => {
    const data = this.metrics();
    if (!data || !data.cas) return '1.0x';
    const logical = data.cas.logical_files_count || 0;
    const unique = data.cas.unique_contents_count || 0;
    if (unique === 0) return '1.0x';
    return (logical / unique).toFixed(1) + 'x';
  });

  ngOnInit(): void {
    this.loadMetrics();
    window.addEventListener('kb:refresh', this.refreshHandler);
    this.refreshInterval = setInterval(() => this.loadMetrics(false), 20000);
  }

  ngOnDestroy(): void {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
    window.removeEventListener('kb:refresh', this.refreshHandler);
  }

  loadMetrics(showSpinner = true): void {
    if (showSpinner) {
      this.loading.set(true);
    }
    this.api.getTelemetryMetrics().subscribe({
      next: (data) => {
        this.metrics.set(data);
        this.error.set(null);
        this.loading.set(false);
      },
      error: (err) => {
        console.error('Failed to load telemetry metrics:', err);
        this.error.set(this.i18n.t('dashboard.error'));
        this.loading.set(false);
      }
    });
  }

  formatTokens(tokens: number | undefined): string {
    if (tokens === undefined || tokens === null) return '0';
    if (tokens >= 1_000_000) {
      return (tokens / 1_000_000).toFixed(1) + 'M';
    }
    if (tokens >= 1_000) {
      return (tokens / 1_000).toFixed(1) + 'K';
    }
    return tokens.toLocaleString();
  }

  formatBytes(bytes: number | undefined): string {
    if (bytes === undefined || bytes === null || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    const val = bytes / Math.pow(k, i);
    return `${val.toFixed(i >= 3 ? 2 : 1)} ${sizes[i]}`;
  }

  formatPercent(pct: number | undefined): string {
    if (pct === undefined || pct === null) return '0.0%';
    return `${pct.toFixed(1)}%`;
  }

  getToolDisplayName(name: string): string {
    switch (name) {
      case 'knowledge_search':
        return 'knowledge_search (RAG / FTS5)';
      case 'knowledge_review_context':
        return 'knowledge_review_context (Blast Radius)';
      case 'knowledge_get_project_rules':
        return 'knowledge_get_project_rules (Rules Hub)';
      case 'knowledge_find_symbol':
        return 'knowledge_find_symbol (Code Graph)';
      case 'knowledge_get_callers':
        return 'knowledge_get_callers (Call Graph)';
      case 'knowledge_get_callees':
        return 'knowledge_get_callees (Call Graph)';
      case 'knowledge_find_tests':
        return 'knowledge_find_tests (TESTED_BY)';
      default:
        return name;
    }
  }
}
