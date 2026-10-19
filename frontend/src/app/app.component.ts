import { Component, inject, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { TuiRoot } from '@taiga-ui/core';
import { SidebarComponent } from './layout/sidebar/sidebar.component';
import { HeaderComponent } from './layout/header/header.component';
import { SettingsDialogComponent } from './features/settings/settings-dialog.component';
import { AuthDialogComponent } from './features/auth/auth-dialog.component';
import { ThemeService, ThemeMode } from './core/services/theme.service';
import { I18nService, Locale } from './core/services/i18n.service';
import { ApiService } from './core/services/api.service';
import { AuthService } from './core/services/auth.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    TuiRoot,
    SidebarComponent,
    HeaderComponent,
    SettingsDialogComponent,
    AuthDialogComponent
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  private readonly themeService = inject(ThemeService);
  private readonly i18nService = inject(I18nService);
  private readonly apiService = inject(ApiService);
  readonly auth = inject(AuthService);

  readonly isSettingsOpen = signal(false);

  ngOnInit(): void {
    this.loadInitSettings();
    window.addEventListener('kb:refresh', () => this.loadInitSettings());
  }

  loadInitSettings(): void {
    if (!this.auth.isAuthenticated()) return;

    this.apiService.getSettings().subscribe({
      next: (settings) => {
        if (settings.ui?.default_locale && !localStorage.getItem('kb_locale')) {
          this.i18nService.setLocale(settings.ui.default_locale as Locale);
        }
        if (settings.ui?.default_theme && !localStorage.getItem('kb_theme')) {
          this.themeService.setTheme(settings.ui.default_theme as ThemeMode);
        }
      },
      error: (err) => {
        if (err.status === 401) {
          this.auth.promptLogin();
        }
      }
    });
  }

  openSettings(): void {
    this.isSettingsOpen.set(true);
  }

  closeSettings(): void {
    this.isSettingsOpen.set(false);
  }

  closeSettingsOnBackdrop(event: MouseEvent): void {
    if ((event.target as HTMLElement).classList.contains('modal-backdrop')) {
      this.closeSettings();
    }
  }
}
