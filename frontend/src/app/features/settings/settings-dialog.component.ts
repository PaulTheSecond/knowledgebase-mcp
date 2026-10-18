import { Component, inject, signal, output, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, ReactiveFormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';
import { I18nService, Locale } from '../../core/services/i18n.service';
import { ThemeService, ThemeMode } from '../../core/services/theme.service';
import { SystemSettings } from '../../core/models/settings.model';

@Component({
  selector: 'app-settings-dialog',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './settings-dialog.component.html',
  styleUrls: ['./settings-dialog.component.scss']
})
export class SettingsDialogComponent implements OnInit {
  private readonly fb = inject(FormBuilder);
  private readonly api = inject(ApiService);
  readonly i18n = inject(I18nService);
  readonly theme = inject(ThemeService);

  readonly activeTab = signal<'appearance' | 'llm' | 'embeddings' | 'integrations'>('appearance');
  readonly testingLlm = signal(false);
  readonly saving = signal(false);
  readonly llmTestResult = signal<{ success: boolean; message: string } | null>(null);

  readonly selectedLocale = signal<Locale>(this.i18n.currentLocale());
  readonly selectedTheme = signal<ThemeMode>(this.theme.currentTheme());

  readonly isEnvBaseUrl = signal(false);
  readonly isEnvModel = signal(false);
  readonly isEnvWebhook = signal(false);
  readonly isEnvPat = signal(false);

  readonly close = output<boolean>();
  onClose?: (saved: boolean) => void;

  readonly form = this.fb.group({
    chatEnabled: [true],
    llmBaseUrl: [''],
    llmModel: [''],
    embeddingsEnabled: [false],
    embeddingsModel: ['intfloat/multilingual-e5-small'],
    webhookSecret: ['']
  });

  ngOnInit(): void {
    this.selectedLocale.set(this.i18n.currentLocale());
    this.selectedTheme.set(this.theme.currentTheme());
    this.loadSettings();
  }

  loadSettings(): void {
    this.api.getSettings().subscribe({
      next: (data: SystemSettings) => {
        this.form.patchValue({
          chatEnabled: data.llm.chat_enabled,
          llmBaseUrl: data.llm.api_base,
          llmModel: data.llm.model,
          embeddingsEnabled: data.embeddings.enabled,
          embeddingsModel: data.embeddings.model_name
        });

        this.isEnvBaseUrl.set(data.llm.is_env_api_base);
        this.isEnvModel.set(data.llm.is_env_model);
        this.isEnvWebhook.set(data.integrations.is_env_webhook_secret);
        this.isEnvPat.set(data.integrations.is_env_github_pat);
      }
    });
  }

  setTab(tab: 'appearance' | 'llm' | 'embeddings' | 'integrations'): void {
    this.activeTab.set(tab);
  }

  selectLanguage(locale: Locale): void {
    this.selectedLocale.set(locale);
    this.i18n.setLocale(locale);
  }

  selectTheme(mode: ThemeMode): void {
    this.selectedTheme.set(mode);
    this.theme.setTheme(mode);
  }

  testLlm(): void {
    this.testingLlm.set(true);
    this.llmTestResult.set(null);

    const payload = {
      api_base: this.form.value.llmBaseUrl?.trim(),
      model: this.form.value.llmModel?.trim()
    };

    this.api.testLlm(payload).subscribe({
      next: (res) => {
        this.llmTestResult.set(res);
        this.testingLlm.set(false);
      },
      error: (err) => {
        this.llmTestResult.set({ success: false, message: err.error?.detail || err.message });
        this.testingLlm.set(false);
      }
    });
  }

  save(): void {
    this.saving.set(true);
    const val = this.form.value;

    const payload = {
      default_locale: this.selectedLocale(),
      default_theme: this.selectedTheme(),
      chat_enabled: val.chatEnabled ?? true,
      llm_base_url: val.llmBaseUrl?.trim(),
      llm_model: val.llmModel?.trim(),
      embeddings_enabled: val.embeddingsEnabled ?? false,
      embeddings_model: val.embeddingsModel?.trim(),
      webhook_secret: val.webhookSecret?.trim() || undefined
    };

    this.api.updateSettings(payload).subscribe({
      next: (res) => {
        this.saving.set(false);
        alert(res.message || 'Saved successfully');
        this.close.emit(true);
        if (this.onClose) this.onClose(true);
      },
      error: (err) => {
        this.saving.set(false);
        alert(err.error?.detail || err.message);
      }
    });
  }

  cancel(): void {
    this.close.emit(false);
    if (this.onClose) this.onClose(false);
  }
}
