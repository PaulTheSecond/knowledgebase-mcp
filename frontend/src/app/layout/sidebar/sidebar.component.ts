import { Component, inject, output, signal, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { I18nService } from '../../core/services/i18n.service';
import { ApiService } from '../../core/services/api.service';
import { WebSocketService } from '../../core/services/websocket.service';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.scss']
})
export class SidebarComponent implements OnInit, OnDestroy {
  readonly i18n = inject(I18nService);
  private readonly api = inject(ApiService);
  private readonly ws = inject(WebSocketService);
  readonly auth = inject(AuthService);

  readonly openSettings = output<void>();

  readonly workerStatus = signal<'idle' | 'processing' | 'halted' | 'error'>('idle');
  readonly chatEnabled = signal(true);
  readonly activeQueueCount = signal(0);

  private intervalId?: any;
  private readonly refreshHandler = () => {
    this.updateStatus();
    if (this.auth.isAuthenticated()) {
      this.ws.connect();
    }
  };

  ngOnInit(): void {
    this.updateStatus();
    this.intervalId = setInterval(() => this.updateStatus(), 15000);
    window.addEventListener('kb:refresh', this.refreshHandler);

    if (this.auth.isAuthenticated()) {
      this.ws.connect();
    }
  }

  ngOnDestroy(): void {
    if (this.intervalId) clearInterval(this.intervalId);
    window.removeEventListener('kb:refresh', this.refreshHandler);
  }

  updateStatus(): void {
    if (!this.auth.isAuthenticated()) {
      this.workerStatus.set('error');
      return;
    }

    this.api.getStatus().subscribe({
      next: (data) => {
        if (data.status === 'processing' || data.status === 'syncing') {
          this.workerStatus.set('processing');
        } else if (data.status === 'halted') {
          this.workerStatus.set('halted');
        } else {
          this.workerStatus.set('idle');
        }
        this.chatEnabled.set(data.chat_enabled);
      },
      error: (err) => {
        if (err.status === 401) {
          this.auth.promptLogin();
        }
        this.workerStatus.set('error');
      }
    });

    this.api.getQueue().subscribe({
      next: (data) => {
        const count = (data.jobs || []).filter(j => j.status === 'pending' || j.status === 'processing').length;
        this.activeQueueCount.set(count);
      }
    });
  }

  onSettingsClick(): void {
    this.openSettings.emit();
  }

  logout(): void {
    this.auth.logout();
  }
}
