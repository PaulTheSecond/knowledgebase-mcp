import { Injectable, inject } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { MermaidService } from './mermaid.service';

declare const marked: any;
declare const Prism: any;

const LANGUAGE_MAP: Record<string, { name: string; prism: string }> = {
  cs: { name: 'C#', prism: 'csharp' },
  csharp: { name: 'C#', prism: 'csharp' },
  dotnet: { name: '.NET C#', prism: 'csharp' },
  ts: { name: 'TypeScript', prism: 'typescript' },
  typescript: { name: 'TypeScript', prism: 'typescript' },
  js: { name: 'JavaScript', prism: 'javascript' },
  javascript: { name: 'JavaScript', prism: 'javascript' },
  py: { name: 'Python', prism: 'python' },
  python: { name: 'Python', prism: 'python' },
  json: { name: 'JSON', prism: 'json' },
  yaml: { name: 'YAML', prism: 'yaml' },
  yml: { name: 'YAML', prism: 'yaml' },
  sh: { name: 'Bash', prism: 'bash' },
  bash: { name: 'Bash', prism: 'bash' },
  shell: { name: 'Shell', prism: 'bash' },
  sql: { name: 'SQL', prism: 'sql' },
  docker: { name: 'Docker', prism: 'docker' },
  dockerfile: { name: 'Dockerfile', prism: 'docker' },
  md: { name: 'Markdown', prism: 'markdown' },
  markdown: { name: 'Markdown', prism: 'markdown' },
  rust: { name: 'Rust', prism: 'rust' },
  rs: { name: 'Rust', prism: 'rust' },
  go: { name: 'Go', prism: 'go' },
  golang: { name: 'Go', prism: 'go' },
  html: { name: 'HTML', prism: 'markup' },
  xml: { name: 'XML', prism: 'markup' },
  css: { name: 'CSS', prism: 'css' }
};

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Сервис форматирования Markdown и подсветки синтаксиса.
 * Обеспечивает красивое оформление блоков кода с плашками языков,
 * кнопками копирования в буфер и рендеринг схем Mermaid.
 */
@Injectable({
  providedIn: 'root'
})
export class MarkdownService {
  private readonly mermaidService = inject(MermaidService);
  private readonly sanitizer = inject(DomSanitizer);
  private initialized = false;

  constructor() {
    this.configureMarked();
  }

  private configureMarked(): void {
    if (this.initialized || typeof marked === 'undefined') return;

    marked.use({
      renderer: {
        code: (token: any) => {
          const rawText: string = typeof token === 'object' ? token.text : token;
          const rawLang: string = (typeof token === 'object' ? token.lang : '') || '';
          const lang = rawLang.trim().toLowerCase();

          // 1. Обработка блоков диаграмм Mermaid
          if (lang === 'mermaid') {
            const encodedCode = encodeURIComponent(rawText);
            const escapedCode = escapeHtml(rawText);
            return `
              <div class="mermaid-block-wrapper" data-mermaid-code="${encodedCode}">
                <div class="mermaid-block-header">
                  <span class="mermaid-label">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <circle cx="6" cy="6" r="3"/><circle cx="18" cy="18" r="3"/><path d="M6 9v3a3 3 0 0 0 3 3h6"/>
                    </svg>
                    Диаграмма Mermaid
                  </span>
                  <button class="btn-copy-mermaid" type="button" title="Скопировать Mermaid код">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                    </svg>
                    <span>Копировать код</span>
                  </button>
                </div>
                <div class="mermaid-diagram-loading">
                  <span class="spinner-sm"></span> Отрисовка диаграммы...
                </div>
                <div class="mermaid-svg-container"></div>
                <pre class="mermaid-raw-source" style="display:none"><code>${escapedCode}</code></pre>
              </div>
            `;
          }

          // 2. Стандартные блоки исходного кода с плашкой языка и копированием
          const langInfo = LANGUAGE_MAP[lang] || {
            name: lang ? lang.toUpperCase() : 'КОД',
            prism: lang || 'none'
          };
          const escaped = escapeHtml(rawText);

          return `
            <div class="code-block-wrapper">
              <div class="code-block-header">
                <span class="code-lang-label">${langInfo.name}</span>
                <button class="btn-copy-code" type="button" title="Скопировать код">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                  </svg>
                  <span>Копировать</span>
                </button>
              </div>
              <pre class="language-${langInfo.prism}"><code class="language-${langInfo.prism}">${escaped}</code></pre>
            </div>
          `;
        }
      }
    });

    this.initialized = true;
  }

  /**
   * Преобразует Markdown в HTML строку с кастомным оформлением блоков кода и Mermaid
   */
  render(content: string): string {
    if (typeof marked === 'undefined' || !content) {
      return content || '';
    }
    this.configureMarked();
    return marked.parse(content);
  }

  /**
   * Преобразует Markdown в доверенный HTML (SafeHtml) для директивы [innerHTML],
   * предотвращая вырезание Angular DomSanitizer кнопок копирования и атрибутов.
   */
  renderSafe(content: string): SafeHtml {
    const html = this.render(content);
    return this.sanitizer.bypassSecurityTrustHtml(html);
  }

  /**
   * Запускает подсветку синтаксиса Prism для всех блоков в контейнере
   */
  highlightCode(container: HTMLElement): void {
    if (typeof Prism !== 'undefined' && container) {
      Prism.highlightAllUnder(container);
    }
  }

  /**
   * Находит и отрисовывает все непорендеренные диаграммы Mermaid внутри контейнера
   */
  async renderMermaid(container: HTMLElement): Promise<void> {
    if (!container) return;
    const blocks = container.querySelectorAll<HTMLElement>('.mermaid-block-wrapper:not([data-rendered])');

    for (const block of Array.from(blocks)) {
      let code = '';
      const encodedCode = block.getAttribute('data-mermaid-code');
      if (encodedCode) {
        code = decodeURIComponent(encodedCode);
      } else {
        const rawEl = block.querySelector<HTMLElement>('.mermaid-raw-source code');
        if (rawEl) {
          code = rawEl.textContent || '';
        }
      }
      if (!code) continue;

      const loadingEl = block.querySelector<HTMLElement>('.mermaid-diagram-loading');
      const svgContainer = block.querySelector<HTMLElement>('.mermaid-svg-container');

      const result = await this.mermaidService.render(code);
      if (loadingEl) loadingEl.style.display = 'none';

      if (result.svg && svgContainer) {
        svgContainer.innerHTML = result.svg;
        block.setAttribute('data-rendered', 'true');
      } else {
        block.setAttribute('data-rendered', 'error');
        if (svgContainer) {
          svgContainer.innerHTML = `
            <div class="mermaid-error-banner">
              <div class="mermaid-error-title">⚠️ Ошибка синтаксиса диаграммы Mermaid</div>
              <div class="mermaid-error-desc">${escapeHtml(result.error || '')}</div>
              <pre class="mermaid-error-raw"><code>${escapeHtml(code)}</code></pre>
            </div>
          `;
        }
      }
    }
  }

  /**
   * Обработчик кликов на кнопки копирования кода/диаграммы через делегирование
   */
  async handleCopyClick(event: MouseEvent): Promise<boolean> {
    const target = event.target as HTMLElement;
    const copyBtn = target.closest<HTMLButtonElement>('.btn-copy-code, .btn-copy-mermaid');
    if (!copyBtn) return false;

    event.preventDefault();
    event.stopPropagation();

    let textToCopy = '';

    if (copyBtn.classList.contains('btn-copy-mermaid')) {
      const wrapper = copyBtn.closest<HTMLElement>('.mermaid-block-wrapper');
      const encoded = wrapper?.getAttribute('data-mermaid-code');
      if (encoded) {
        textToCopy = decodeURIComponent(encoded);
      } else {
        const rawEl = wrapper?.querySelector<HTMLElement>('.mermaid-raw-source code');
        if (rawEl) {
          textToCopy = rawEl.textContent || '';
        }
      }
    } else {
      const wrapper = copyBtn.closest<HTMLElement>('.code-block-wrapper');
      const codeEl = wrapper?.querySelector('code');
      if (codeEl) {
        textToCopy = codeEl.textContent || '';
      }
    }

    if (!textToCopy) return false;

    try {
      await navigator.clipboard.writeText(textToCopy);
      const span = copyBtn.querySelector('span');
      if (span) {
        const originalText = span.textContent;
        span.textContent = 'Скопировано!';
        copyBtn.classList.add('copied');
        setTimeout(() => {
          span.textContent = originalText;
          copyBtn.classList.remove('copied');
        }, 2000);
      }
      return true;
    } catch (err) {
      console.warn('Не удалось скопировать в буфер обмена:', err);
      return false;
    }
  }
}
