import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { Repository } from '../../../core/models/repository.model';
import { ApiService } from '../../../core/services/api.service';
import { I18nService } from '../../../core/services/i18n.service';

@Component({
  selector: 'app-repo-dialog',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './repo-dialog.component.html',
  styleUrls: ['./repo-dialog.component.scss']
})
export class RepoDialogComponent {
  private readonly fb = inject(FormBuilder);
  private readonly api = inject(ApiService);
  readonly i18n = inject(I18nService);

  readonly isEditing = signal(false);
  readonly testing = signal(false);
  readonly saving = signal(false);
  readonly testResult = signal<{ success: boolean; message: string } | null>(null);

  onClose?: (saved: boolean) => void;

  readonly form = this.fb.group({
    id: ['', [Validators.required, Validators.pattern(/^[a-zA-Z0-9_\-]+$/)]],
    url: ['', [Validators.required]],
    branch: ['master', [Validators.required]],
    authType: ['none' as 'none' | 'pat' | 'ssh'],
    tokenEnv: ['GITHUB_PAT'],
    sshPath: [''],
    webhooks: [true],
    cron: ['0 */4 * * *']
  });

  initData(repo?: Repository): void {
    if (repo) {
      this.isEditing.set(true);
      this.form.patchValue({
        id: repo.id,
        url: repo.url,
        branch: repo.branch || 'master',
        authType: repo.auth?.type || 'none',
        tokenEnv: repo.auth?.token_env || 'GITHUB_PAT',
        sshPath: repo.auth?.private_key_path || '',
        webhooks: repo.triggers?.webhooks?.enabled ?? true,
        cron: repo.triggers?.cron || ''
      });
      this.form.get('id')?.disable();
    } else {
      this.isEditing.set(false);
      this.form.reset({
        id: '',
        url: '',
        branch: 'master',
        authType: 'none',
        tokenEnv: 'GITHUB_PAT',
        sshPath: '',
        webhooks: true,
        cron: '0 */4 * * *'
      });
      this.form.get('id')?.enable();
    }
  }

  testConnection(): void {
    if (!this.form.value.url) return;
    this.testing.set(true);
    this.testResult.set(null);

    const payload = {
      url: this.form.value.url,
      branch: this.form.value.branch,
      auth_type: this.form.value.authType,
      token_env: this.form.value.tokenEnv,
      private_key_path: this.form.value.sshPath
    };

    this.api.testGit(payload).subscribe({
      next: (res) => {
        this.testResult.set(res);
        this.testing.set(false);
      },
      error: (err) => {
        this.testResult.set({ success: false, message: err.error?.detail || err.message });
        this.testing.set(false);
      }
    });
  }

  save(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }

    this.saving.set(true);
    const val = this.form.getRawValue();

    const repo: Repository = {
      id: val.id!,
      url: val.url!,
      branch: val.branch || 'master',
      auth: {
        type: val.authType || 'none',
        token_env: val.authType === 'pat' ? val.tokenEnv : null,
        private_key_path: val.authType === 'ssh' ? val.sshPath : null
      },
      triggers: {
        webhooks: {
          enabled: Boolean(val.webhooks),
          secret_env: 'WEBHOOK_SECRET'
        },
        cron: val.cron || null
      }
    };

    const request$ = this.isEditing()
      ? this.api.updateRepository(repo.id, repo)
      : this.api.createRepository(repo);

    request$.subscribe({
      next: () => {
        this.saving.set(false);
        if (this.onClose) this.onClose(true);
      },
      error: (err) => {
        alert(err.error?.detail || err.message);
        this.saving.set(false);
      }
    });
  }

  cancel(): void {
    if (this.onClose) this.onClose(false);
  }
}
