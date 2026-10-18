import { Injectable, inject } from '@angular/core';
import { Subject, Observable } from 'rxjs';
import { ApiService } from './api.service';

export interface SyncProgressData {
  repo_id: string;
  progress: number;
  status: string;
  status_key?: string;
}

@Injectable({
  providedIn: 'root'
})
export class WebSocketService {
  private readonly api = inject(ApiService);
  private socket: WebSocket | null = null;
  private readonly progressSubject = new Subject<SyncProgressData>();
  readonly progress$: Observable<SyncProgressData> = this.progressSubject.asObservable();

  connect(): void {
    this.api.getAuthToken().subscribe({
      next: ({ token }) => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${protocol}//${window.location.host}/ws/status?token=${token}`;
        
        this.socket = new WebSocket(url);

        this.socket.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'sync_progress' && msg.data) {
              this.progressSubject.next(msg.data);
            }
          } catch (e) {
            console.error('Error parsing WebSocket message:', e);
          }
        };

        this.socket.onclose = () => {
          setTimeout(() => this.connect(), 3000);
        };

        this.socket.onerror = () => {
          if (this.socket) {
            this.socket.close();
          }
        };
      },
      error: () => {
        setTimeout(() => this.connect(), 5000);
      }
    });
  }
}
