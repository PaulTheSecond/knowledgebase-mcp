import { Component, inject, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { QueueJob } from '../../core/models/queue.model';

@Component({
  selector: 'app-queue',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './queue.component.html',
  styleUrls: ['./queue.component.scss']
})
export class QueueComponent implements OnInit {
  private readonly api = inject(ApiService);
  readonly i18n = inject(I18nService);

  readonly jobs = signal<QueueJob[]>([]);
  readonly loading = signal(true);
  readonly retrying = signal(false);

  ngOnInit(): void {
    this.loadQueue();
    window.addEventListener('kb:refresh', () => this.loadQueue());
  }

  loadQueue(): void {
    this.loading.set(true);
    this.api.getQueue().subscribe({
      next: (data) => {
        this.jobs.set(data.jobs || []);
        this.loading.set(false);
      },
      error: () => this.loading.set(false)
    });
  }

  retryFailed(): void {
    this.retrying.set(true);
    this.api.retryFailedJobs().subscribe({
      next: (res) => {
        this.retrying.set(false);
        this.loadQueue();
      },
      error: (err) => {
        this.retrying.set(false);
        alert(err.error?.detail || err.message);
      }
    });
  }

  stopJob(jobId: number): void {
    this.api.stopJob(jobId).subscribe({
      next: () => {
        this.loadQueue();
      },
      error: (err) => {
        alert(err.error?.detail || err.message);
      }
    });
  }

  restartJob(jobId: number): void {
    this.api.restartJob(jobId).subscribe({
      next: () => {
        this.loadQueue();
      },
      error: (err) => {
        alert(err.error?.detail || err.message);
      }
    });
  }

  deleteJob(jobId: number): void {
    if (!confirm(this.i18n.t('queue.confirm_delete', { id: jobId }))) {
      return;
    }
    this.api.deleteJob(jobId).subscribe({
      next: () => {
        this.loadQueue();
      },
      error: (err) => {
        alert(err.error?.detail || err.message);
      }
    });
  }
}
