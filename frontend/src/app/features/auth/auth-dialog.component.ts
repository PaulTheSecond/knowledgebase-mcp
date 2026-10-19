import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { AuthService } from '../../core/services/auth.service';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';

@Component({
  selector: 'app-auth-dialog',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './auth-dialog.component.html',
  styleUrls: ['./auth-dialog.component.scss']
})
export class AuthDialogComponent {
  private readonly fb = inject(FormBuilder);
  readonly auth = inject(AuthService);
  private readonly api = inject(ApiService);
  readonly i18n = inject(I18nService);

  readonly loading = signal(false);
  readonly errorMessage = signal<string | null>(null);

  readonly form = this.fb.group({
    username: [this.auth.currentUser() || 'admin', [Validators.required]],
    password: ['admin', [Validators.required]]
  });

  onSubmit(): void {
    if (this.form.invalid) return;

    this.loading.set(true);
    this.errorMessage.set(null);

    const { username, password } = this.form.value;

    this.api.checkAuth(username!, password!).subscribe({
      next: () => {
        this.loading.set(false);
        this.auth.setCredentials(username!, password!);
        // Оповещаем страницу о необходимости перезагрузить данные
        window.dispatchEvent(new CustomEvent('kb:refresh'));
      },
      error: (err) => {
        this.loading.set(false);
        if (err.status === 401) {
          this.errorMessage.set(this.i18n.t('auth.invalid_credentials'));
        } else if (err.status === 403) {
          this.errorMessage.set(this.i18n.t('auth.disabled'));
        } else {
          this.errorMessage.set(err.error?.detail || err.message || 'Connection error');
        }
      }
    });
  }
}
