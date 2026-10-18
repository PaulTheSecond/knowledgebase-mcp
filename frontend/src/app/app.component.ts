import { Component, inject, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { TuiRoot } from '@taiga-ui/core';
import { SidebarComponent } from './layout/sidebar/sidebar.component';
import { HeaderComponent } from './layout/header/header.component';
import { SettingsDialogComponent } from './features/settings/settings-dialog.component';
import { ThemeService, ThemeMode } from './core/services/theme.service';
import { I18nService, Locale } from './core/services/i18n.service';
import { ApiService } from './core/services/api.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    TuiRoot,
    SidebarComponent,
    HeaderComponent,
    SettingsDialogComponent
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  private readonly themeService = inject(ThemeService);
  private readonly i18nService = inject(I18nService);
  private readonly apiService = inject(ApiService);

  readonly isSettingsOpen = signal(false);

  ngOnInit(): void {
    // Загрузка дефолтных настроек из базы (локаль и тема по умолчанию)
    this.apiService.getSettings().subscribe({
      next: (settings) => {
        if (settings.ui?.default_locale && !localStorage.getItem('kb_locale')) {
          this.i18nService.setLocale(settings.ui.default_locale as Locale);
        }
        if (settings.ui?.default_theme && !localStorage.getItem('kb_theme')) {
          this.themeService.setTheme(settings.ui.default_theme as ThemeMode);
        }
      },
      error: () => {
        // Fallback: сервисы уже используют значения по умолчанию
      }
    });
  }

  openSettings(): void {
    this.isSettingsOpen.set(true);
  }

  closeSettings(): void {
    this.isSettingsOpen.set(false);
  }
}
