import { Component, inject, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, NavigationEnd } from '@angular/router';
import { filter } from 'rxjs';
import { toSignal } from '@angular/core/rxjs-interop';
import { I18nService } from '../../core/services/i18n.service';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.scss']
})
export class HeaderComponent {
  readonly i18n = inject(I18nService);
  private readonly router = inject(Router);

  // Отслеживаем текущий URL
  private readonly navEnd = toSignal(
    this.router.events.pipe(filter((e): e is NavigationEnd => e instanceof NavigationEnd))
  );

  readonly currentTitle = computed(() => {
    // В зависимости от текущего роута выводим локализованный заголовок
    const url = this.navEnd()?.urlAfterRedirects || this.router.url || '';
    if (url.includes('/queue')) {
      return this.i18n.t('header.title_queue');
    }
    if (url.includes('/chat')) {
      return this.i18n.t('header.title_chat');
    }
    return this.i18n.t('header.title_repositories');
  });

  onRefresh(): void {
    // Триггерим кастомное глобальное событие обновления, чтобы активные компоненты обновили данные
    window.dispatchEvent(new CustomEvent('kb:refresh'));
  }
}
