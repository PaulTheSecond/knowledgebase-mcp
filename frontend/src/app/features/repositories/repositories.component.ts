import { Component, inject, signal, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { WebSocketService, SyncProgressData } from '../../core/services/websocket.service';
import { Repository } from '../../core/models/repository.model';
import { RepoDialogComponent } from './repo-dialog/repo-dialog.component';

@Component({
  selector: 'app-repositories',
  standalone: true,
  imports: [CommonModule, RepoDialogComponent],
  templateUrl: './repositories.component.html',
  styleUrls: ['./repositories.component.scss']
})
export class RepositoriesComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  readonly i18n = inject(I18nService);
  private readonly ws = inject(WebSocketService);

  readonly repositories = signal<Repository[]>([]);
  readonly loading = signal(true);
  readonly activeProgresses = signal<Record<string, { progress: number; status: string }>>({});

  readonly showRepoModal = signal(false);
  readonly selectedRepo = signal<Repository | undefined>(undefined);

  private wsSub?: Subscription;

  ngOnInit(): void {
    this.loadRepositories();

    this.wsSub = this.ws.progress$.subscribe((data: SyncProgressData) => {
      this.handleProgress(data);
    });
  }

  ngOnDestroy(): void {
    this.wsSub?.unsubscribe();
  }

  loadRepositories(): void {
    this.loading.set(true);
    this.api.getStatus().subscribe({
      next: (statusData) => {
        this.api.getSettings().subscribe({
          next: (settingsData) => {
            const statusMap = new Map(statusData.repositories.map(r => [r.id, r]));
            const merged = settingsData.repositories.map(repo => {
              const stat = statusMap.get(repo.id);
              return {
                ...repo,
                files_count: stat ? stat.files_count : 0,
                chunks_count: stat ? stat.chunks_count : 0
              };
            });
            this.repositories.set(merged);
            this.loading.set(false);
          },
          error: () => this.loading.set(false)
        });
      },
      error: () => this.loading.set(false)
    });
  }

  private handleProgress(data: SyncProgressData): void {
    const current = { ...this.activeProgresses() };
    if (data.progress === 100 || data.status === 'completed' || data.status === 'failed') {
      delete current[data.repo_id];
      this.activeProgresses.set(current);
      this.loadRepositories();
    } else {
      current[data.repo_id] = {
        progress: data.progress,
        status: data.status_key ? this.i18n.t(data.status_key) : data.status
      };
      this.activeProgresses.set(current);
    }
  }

  openAddDialog(): void {
    this.selectedRepo.set(undefined);
    this.showRepoModal.set(true);
  }

  openEditDialog(repo: Repository): void {
    this.selectedRepo.set(repo);
    this.showRepoModal.set(true);
  }

  onRepoDialogClose(saved: boolean): void {
    this.showRepoModal.set(false);
    if (saved) {
      this.loadRepositories();
    }
  }

  triggerSync(repoId: string): void {
    this.api.triggerSync(repoId).subscribe({
      next: (res) => {
        alert(res.message || this.i18n.t('repos.sync_started', { repo_id: repoId, job_id: res.job_id }));
      },
      error: (err) => {
        alert(err.error?.detail || err.message);
      }
    });
  }

  deleteRepo(repoId: string): void {
    if (!confirm(this.i18n.t('repos.delete_confirm', { repo_id: repoId }))) return;
    const purge = confirm(this.i18n.t('repos.purge_prompt'));

    this.api.deleteRepository(repoId, purge).subscribe({
      next: (res) => {
        alert(res.message);
        this.loadRepositories();
      },
      error: (err) => {
        alert(err.error?.detail || err.message);
      }
    });
  }
}
