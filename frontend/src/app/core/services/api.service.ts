import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { SystemStatus, QueueJob } from '../models/queue.model';
import { SystemSettings, UpdateSettingsPayload } from '../models/settings.model';
import { Repository } from '../models/repository.model';
import { I18nService } from './i18n.service';
import { AuthService } from './auth.service';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly i18n = inject(I18nService);
  private readonly auth = inject(AuthService);

  private getHeaders(): HttpHeaders {
    let headers = new HttpHeaders({
      'Accept-Language': this.i18n.currentLocale()
    });
    const authHeader = this.auth.getAuthHeader();
    if (authHeader) {
      headers = headers.set('Authorization', authHeader);
    }
    return headers;
  }

  checkAuth(username: string, pass: string): Observable<SystemStatus> {
    const encoded = btoa(`${username}:${pass}`);
    const headers = new HttpHeaders({
      'Accept-Language': this.i18n.currentLocale(),
      'Authorization': `Basic ${encoded}`
    });
    return this.http.get<SystemStatus>('/api/status', { headers });
  }

  getStatus(): Observable<SystemStatus> {
    return this.http.get<SystemStatus>('/api/status', { headers: this.getHeaders() });
  }

  getSettings(): Observable<SystemSettings> {
    return this.http.get<SystemSettings>('/api/settings', { headers: this.getHeaders() });
  }

  updateSettings(payload: UpdateSettingsPayload): Observable<{ status: string; message: string }> {
    return this.http.put<{ status: string; message: string }>('/api/settings', payload, {
      headers: this.getHeaders()
    });
  }

  getQueue(): Observable<{ jobs: QueueJob[] }> {
    return this.http.get<{ jobs: QueueJob[] }>('/api/queue', { headers: this.getHeaders() });
  }

  retryFailedJobs(): Observable<{ status: string; count: number; message: string }> {
    return this.http.post<{ status: string; count: number; message: string }>(
      '/api/queue/retry-failed',
      {},
      { headers: this.getHeaders() }
    );
  }

  stopJob(jobId: number): Observable<{ status: string; job_id: number; message: string }> {
    return this.http.post<{ status: string; job_id: number; message: string }>(
      `/api/queue/${jobId}/stop`,
      {},
      { headers: this.getHeaders() }
    );
  }

  restartJob(jobId: number): Observable<{ status: string; job_id: number; message: string }> {
    return this.http.post<{ status: string; job_id: number; message: string }>(
      `/api/queue/${jobId}/restart`,
      {},
      { headers: this.getHeaders() }
    );
  }

  deleteJob(jobId: number): Observable<{ status: string; job_id: number; message: string }> {
    return this.http.delete<{ status: string; job_id: number; message: string }>(
      `/api/queue/${jobId}`,
      { headers: this.getHeaders() }
    );
  }

  triggerSync(repoId: string): Observable<{ status: string; repo_id: string; job_id: number; message: string }> {
    return this.http.post<{ status: string; repo_id: string; job_id: number; message: string }>(
      `/api/sync/${repoId}`,
      {},
      { headers: this.getHeaders() }
    );
  }

  createRepository(repo: Repository): Observable<{ status: string; message: string; repository: Repository }> {
    return this.http.post<{ status: string; message: string; repository: Repository }>(
      '/api/settings/repositories',
      repo,
      { headers: this.getHeaders() }
    );
  }

  updateRepository(repoId: string, repo: Repository): Observable<{ status: string; message: string; repository: Repository }> {
    return this.http.put<{ status: string; message: string; repository: Repository }>(
      `/api/settings/repositories/${repoId}`,
      repo,
      { headers: this.getHeaders() }
    );
  }

  deleteRepository(repoId: string, purgeData: boolean): Observable<{ status: string; message: string; files_purged: number }> {
    return this.http.delete<{ status: string; message: string; files_purged: number }>(
      `/api/settings/repositories/${repoId}?purge_data=${purgeData}`,
      { headers: this.getHeaders() }
    );
  }

  testGit(payload: any): Observable<{ success: boolean; message: string }> {
    return this.http.post<{ success: boolean; message: string }>(
      '/api/settings/test-git',
      payload,
      { headers: this.getHeaders() }
    );
  }

  testLlm(payload: any): Observable<{ success: boolean; message: string }> {
    return this.http.post<{ success: boolean; message: string }>(
      '/api/settings/test-llm',
      payload,
      { headers: this.getHeaders() }
    );
  }

  getAuthToken(): Observable<{ token: string }> {
    return this.http.post<{ token: string }>('/api/auth/token', {}, { headers: this.getHeaders() });
  }

  getFileContent(
    filepath: string,
    startLine = 1,
    endLine?: number,
    repoId?: string,
    fullFile = true
  ): Observable<{
    filepath: string;
    repo_id?: string;
    start_line: number;
    end_line: number;
    total_lines: number;
    content: string;
    is_full_file: boolean;
  }> {
    let url = `/api/chat/file?filepath=${encodeURIComponent(filepath)}&start_line=${startLine}`;
    if (endLine) url += `&end_line=${endLine}`;
    if (repoId) url += `&repo_id=${encodeURIComponent(repoId)}`;
    if (fullFile) url += `&full_file=true`;
    return this.http.get<any>(url, { headers: this.getHeaders() });
  }
}
