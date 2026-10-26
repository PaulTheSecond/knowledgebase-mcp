import { Injectable, inject } from '@angular/core';
import mermaid from 'mermaid';
import { ThemeService } from './theme.service';

/**
 * Получает рабочий инстанс Mermaid с поддержкой как ESM default экспорта, так и пространства имен.
 */
function getMermaid(): any {
  const m = mermaid as any;
  return m?.default || m;
}

/**
 * Сервис для безопасного асинхронного рендеринга Mermaid-диаграмм
 * с автоматической синхронизацией темной и светлой темы.
 */
@Injectable({
  providedIn: 'root'
})
export class MermaidService {
  private readonly themeService = inject(ThemeService);
  private idCounter = 0;

  constructor() {
    this.applyMermaidTheme();
  }

  /**
   * Конфигурирует mermaid в соответствии с текущей темой интерфейса.
   */
  private applyMermaidTheme(): void {
    const isDark = this.themeService.isDarkMode();
    const m = getMermaid();
    if (m && typeof m.initialize === 'function') {
      try {
        m.initialize({
          startOnLoad: false,
          theme: isDark ? 'dark' : 'default',
          securityLevel: 'loose',
          fontFamily: 'Outfit, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
        });
      } catch (err) {
        console.warn('Не удалось инициализировать тему Mermaid:', err);
      }
    }
  }

  /**
   * Рендерит код диаграммы Mermaid в SVG.
   * При возникновении ошибки синтаксиса аккуратно перехватывает её,
   * предотвращая падение интерфейса, и возвращает описание ошибки.
   */
  async render(code: string): Promise<{ svg: string; error?: string }> {
    let cleanCode = code.trim();
    // Очищаем случайные обрамления Markdown ```mermaid ... ```
    cleanCode = cleanCode
      .replace(/^```\s*(mermaid)?\s*/i, '')
      .replace(/\s*```\s*$/, '')
      .trim();

    if (!cleanCode) {
      return { svg: '', error: 'Диаграмма пуста' };
    }

    const id = `mermaid_${++this.idCounter}_${Date.now()}`;
    const m = getMermaid();
    if (!m || typeof m.render !== 'function') {
      return { svg: '', error: 'Модуль Mermaid не загружен' };
    }

    try {
      this.applyMermaidTheme();
      const result = await m.render(id, cleanCode);
      return { svg: result.svg };
    } catch (err: any) {
      const stray = document.getElementById(id) || document.getElementById(`d${id}`);
      if (stray) {
        stray.remove();
      }
      return {
        svg: '',
        error: err?.message || 'Не удалось отрисовать Mermaid диаграмму (синтаксическая ошибка)'
      };
    }
  }
}
