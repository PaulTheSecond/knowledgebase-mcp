import { Injectable, signal, computed } from '@angular/core';

export type ThemeMode = 'dark' | 'light' | 'system';

@Injectable({
  providedIn: 'root'
})
export class ThemeService {
  private readonly themeSignal = signal<ThemeMode>(this.getInitialTheme());
  readonly currentTheme = computed(() => this.themeSignal());
  private readonly mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
  readonly isDarkMode = computed(() => {
    const theme = this.themeSignal();
    return theme === 'system' ? this.mediaQuery.matches : theme === 'dark';
  });

  constructor() {
    this.applyTheme(this.themeSignal());
    this.mediaQuery.addEventListener('change', () => {
      if (this.themeSignal() === 'system') {
        this.applyTheme('system');
      }
    });
  }

  private getInitialTheme(): ThemeMode {
    const saved = (localStorage.getItem('kb_theme') || localStorage.getItem('kmcp_theme')) as ThemeMode;
    if (saved && ['dark', 'light', 'system'].includes(saved)) {
      return saved;
    }
    return 'dark';
  }

  setTheme(theme: ThemeMode): void {
    this.themeSignal.set(theme);
    localStorage.setItem('kb_theme', theme);
    localStorage.setItem('kmcp_theme', theme);
    this.applyTheme(theme);
  }

  private applyTheme(theme: ThemeMode): void {
    const isDark = theme === 'system' ? this.mediaQuery.matches : theme === 'dark';
    if (isDark) {
      document.documentElement.setAttribute('data-theme', 'dark');
      document.body.setAttribute('data-theme', 'dark');
      document.body.classList.remove('tui-theme-light');
      document.body.classList.add('tui-theme-dark');
    } else {
      document.documentElement.setAttribute('data-theme', 'light');
      document.body.setAttribute('data-theme', 'light');
      document.body.classList.remove('tui-theme-dark');
      document.body.classList.add('tui-theme-light');
    }
  }
}
