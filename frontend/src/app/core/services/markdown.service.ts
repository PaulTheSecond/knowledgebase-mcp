import { Injectable, inject } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { MermaidService } from './mermaid.service';

declare const marked: any;
declare const Prism: any;

const LANGUAGE_MAP: Record<string, { name: string; prism: string }> = {
  cs: { name: 'C#', prism: 'csharp' },
  csharp: { name: 'C#', prism: 'csharp' },
  'c#': { name: 'C#', prism: 'csharp' },
  dotnet: { name: '.NET C#', prism: 'csharp' },
  c: { name: 'C', prism: 'clike' },
  cpp: { name: 'C++', prism: 'clike' },
  'c++': { name: 'C++', prism: 'clike' },
  cc: { name: 'C++', prism: 'clike' },
  cxx: { name: 'C++', prism: 'clike' },
  ts: { name: 'TypeScript', prism: 'typescript' },
  typescript: { name: 'TypeScript', prism: 'typescript' },
  js: { name: 'JavaScript', prism: 'javascript' },
  javascript: { name: 'JavaScript', prism: 'javascript' },
  mjs: { name: 'JavaScript', prism: 'javascript' },
  cjs: { name: 'JavaScript', prism: 'javascript' },
  jsx: { name: 'React JSX', prism: 'javascript' },
  tsx: { name: 'React TSX', prism: 'typescript' },
  py: { name: 'Python', prism: 'python' },
  python: { name: 'Python', prism: 'python' },
  py3: { name: 'Python', prism: 'python' },
  json: { name: 'JSON', prism: 'json' },
  jsonc: { name: 'JSON', prism: 'json' },
  yaml: { name: 'YAML', prism: 'yaml' },
  yml: { name: 'YAML', prism: 'yaml' },
  sh: { name: 'Bash', prism: 'bash' },
  bash: { name: 'Bash', prism: 'bash' },
  shell: { name: 'Shell', prism: 'bash' },
  zsh: { name: 'Zsh', prism: 'bash' },
  powershell: { name: 'PowerShell', prism: 'bash' },
  ps1: { name: 'PowerShell', prism: 'bash' },
  pwsh: { name: 'PowerShell', prism: 'bash' },
  sql: { name: 'SQL', prism: 'sql' },
  pgsql: { name: 'PostgreSQL', prism: 'sql' },
  postgres: { name: 'PostgreSQL', prism: 'sql' },
  mysql: { name: 'MySQL', prism: 'sql' },
  docker: { name: 'Dockerfile', prism: 'docker' },
  dockerfile: { name: 'Dockerfile', prism: 'docker' },
  md: { name: 'Markdown', prism: 'markdown' },
  markdown: { name: 'Markdown', prism: 'markdown' },
  mermaid: { name: 'Mermaid', prism: 'none' },
  rust: { name: 'Rust', prism: 'rust' },
  rs: { name: 'Rust', prism: 'rust' },
  go: { name: 'Go', prism: 'go' },
  golang: { name: 'Go', prism: 'go' },
  html: { name: 'HTML', prism: 'markup' },
  htm: { name: 'HTML', prism: 'markup' },
  xml: { name: 'XML', prism: 'markup' },
  svg: { name: 'SVG', prism: 'markup' },
  css: { name: 'CSS', prism: 'css' },
  scss: { name: 'SCSS', prism: 'css' },
  sass: { name: 'SASS', prism: 'css' },
  less: { name: 'LESS', prism: 'css' }
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
 * кнопками копирования в буфер и интерактивный рендеринг схем Mermaid.
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

          // 1. Обработка блоков диаграмм Mermaid с кнопками: Код, Во весь экран, Копировать
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
                  <div class="mermaid-header-actions">
                    <button class="btn-toggle-mermaid-code" type="button" title="Переключить просмотр кода диаграммы">
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="16 18 22 12 16 6"/>
                        <polyline points="8 6 2 12 8 18"/>
                      </svg>
                      <span class="btn-toggle-text">Код</span>
                    </button>
                    <button class="btn-fullscreen-mermaid" type="button" title="Развернуть диаграмму на весь экран">
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="15 3 21 3 21 9"/>
                        <polyline points="9 21 3 21 3 15"/>
                        <line x1="21" y1="3" x2="14" y2="10"/>
                        <line x1="3" y1="21" x2="10" y2="14"/>
                      </svg>
                      <span>На весь экран</span>
                    </button>
                    <button class="btn-copy-mermaid" type="button" title="Скопировать Mermaid код">
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                      </svg>
                      <span>Копировать</span>
                    </button>
                  </div>
                </div>
                <div class="mermaid-diagram-loading">
                  <span class="spinner-sm"></span> Отрисовка диаграммы...
                </div>
                <div class="mermaid-svg-container"></div>
                <div class="mermaid-code-container" style="display: none;">
                  <pre class="language-mermaid"><code>${escapedCode}</code></pre>
                </div>
              </div>
            `;
          }

          // 2. Стандартные блоки исходного кода с плашкой языка, подсветкой синтаксиса и номерами строк (IDE-style)
          const langInfo = LANGUAGE_MAP[lang] || {
            name: lang ? lang.toUpperCase() : 'КОД',
            prism: lang || 'none'
          };
          const prismLang = langInfo.prism;

          // Синхронная подсветка синтаксиса через Prism
          let highlighted = '';
          if (typeof Prism !== 'undefined' && Prism.languages && prismLang !== 'none') {
            const grammar = Prism.languages[prismLang] || Prism.languages.clike;
            if (grammar) {
              try {
                highlighted = Prism.highlight(rawText, grammar, prismLang);
              } catch {
                highlighted = escapeHtml(rawText);
              }
            } else {
              highlighted = escapeHtml(rawText);
            }
          } else {
            highlighted = escapeHtml(rawText);
          }

          const linesCount = rawText ? rawText.split('\n').length : 1;
          const gutterPadding = linesCount > 999 ? '4.8em' : (linesCount > 99 ? '4.2em' : '3.8em');
          const encodedCode = encodeURIComponent(rawText);

          return `
            <div class="code-block-wrapper" data-code="${encodedCode}">
              <div class="code-block-header">
                <span class="code-lang-label">${langInfo.name}</span>
                <button class="btn-copy-code" type="button" title="Скопировать чистый код без номеров строк">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                  </svg>
                  <span>Копировать</span>
                </button>
              </div>
              <pre class="line-numbers language-${prismLang}" style="padding-left: ${gutterPadding} !important;"><code class="language-${prismLang}">${highlighted}</code></pre>
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
   * Запускает подсветку синтаксиса Prism и нумерацию строк для всех блоков в контейнере
   */
  highlightCode(container: HTMLElement): void {
    if (typeof Prism !== 'undefined' && container) {
      const pres = container.querySelectorAll<HTMLElement>('pre.line-numbers:not([data-numbered])');
      pres.forEach(pre => {
        pre.setAttribute('data-numbered', 'true');
        const code = pre.querySelector('code');
        if (code) {
          Prism.highlightElement(code);
        }
      });
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
      block.setAttribute('data-rendered', 'rendering');

      let code = '';
      const encodedCode = block.getAttribute('data-mermaid-code');
      if (encodedCode) {
        code = decodeURIComponent(encodedCode);
      } else {
        const rawEl = block.querySelector<HTMLElement>('.mermaid-code-container code');
        if (rawEl) {
          code = rawEl.textContent || '';
        }
      }
      if (!code) {
        block.setAttribute('data-rendered', 'empty');
        continue;
      }

      const loadingEl = block.querySelector<HTMLElement>('.mermaid-diagram-loading');
      const svgContainer = block.querySelector<HTMLElement>('.mermaid-svg-container');

      try {
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
      } catch (err: any) {
        if (loadingEl) loadingEl.style.display = 'none';
        block.setAttribute('data-rendered', 'error');
        if (svgContainer) {
          svgContainer.innerHTML = `
            <div class="mermaid-error-banner">
              <div class="mermaid-error-title">⚠️ Не удалось отобразить диаграмму</div>
              <div class="mermaid-error-desc">${escapeHtml(err?.message || '')}</div>
              <pre class="mermaid-error-raw"><code>${escapeHtml(code)}</code></pre>
            </div>
          `;
        }
      }
    }
  }

  /**
   * Обработчик кликов на кнопки блоков кода и диаграмм через делегирование
   */
  async handleBlockInteraction(event: MouseEvent): Promise<boolean> {
    const target = event.target as HTMLElement;

    // 1. Копирование кода диаграммы Mermaid
    const copyMermaidBtn = target.closest<HTMLButtonElement>('.btn-copy-mermaid');
    if (copyMermaidBtn) {
      event.preventDefault();
      event.stopPropagation();
      const wrapper = copyMermaidBtn.closest<HTMLElement>('.mermaid-block-wrapper');
      const encoded = wrapper?.getAttribute('data-mermaid-code');
      let textToCopy = '';
      if (encoded) {
        textToCopy = decodeURIComponent(encoded);
      } else {
        const rawEl = wrapper?.querySelector<HTMLElement>('.mermaid-code-container code');
        if (rawEl) textToCopy = rawEl.textContent || '';
      }
      if (textToCopy) {
        await navigator.clipboard.writeText(textToCopy);
        const span = copyMermaidBtn.querySelector('span');
        if (span) {
          const prev = span.textContent;
          span.textContent = 'Скопировано!';
          copyMermaidBtn.classList.add('copied');
          setTimeout(() => {
            span.textContent = prev;
            copyMermaidBtn.classList.remove('copied');
          }, 2000);
        }
      }
      return true;
    }

    // 2. Переключение "Код" / "Диаграмма"
    const toggleBtn = target.closest<HTMLButtonElement>('.btn-toggle-mermaid-code');
    if (toggleBtn) {
      event.preventDefault();
      event.stopPropagation();
      const wrapper = toggleBtn.closest<HTMLElement>('.mermaid-block-wrapper');
      if (!wrapper) return true;
      const svgContainer = wrapper.querySelector<HTMLElement>('.mermaid-svg-container');
      const codeContainer = wrapper.querySelector<HTMLElement>('.mermaid-code-container');
      const textSpan = toggleBtn.querySelector('.btn-toggle-text');
      const isCodeVisible = codeContainer && codeContainer.style.display !== 'none';
      if (isCodeVisible) {
        if (codeContainer) codeContainer.style.display = 'none';
        if (svgContainer) svgContainer.style.display = 'flex';
        if (textSpan) textSpan.textContent = 'Код';
        toggleBtn.classList.remove('active');
      } else {
        if (svgContainer) svgContainer.style.display = 'none';
        if (codeContainer) codeContainer.style.display = 'block';
        if (textSpan) textSpan.textContent = 'Диаграмма';
        toggleBtn.classList.add('active');
      }
      return true;
    }

    // 3. Открытие диаграммы на весь экран (Fullscreen Modal с Zoom/Pan)
    const fullscreenBtn = target.closest<HTMLButtonElement>('.btn-fullscreen-mermaid');
    if (fullscreenBtn) {
      event.preventDefault();
      event.stopPropagation();
      const wrapper = fullscreenBtn.closest<HTMLElement>('.mermaid-block-wrapper');
      if (!wrapper) return true;
      const svgContainer = wrapper.querySelector<HTMLElement>('.mermaid-svg-container');
      const svgEl = svgContainer?.querySelector('svg');
      let rawCode = '';
      const encoded = wrapper.getAttribute('data-mermaid-code');
      if (encoded) rawCode = decodeURIComponent(encoded);
      if (svgEl) {
        this.openFullscreenModal(svgEl.outerHTML, rawCode);
      }
      return true;
    }

    // 4. Обычные блоки исходного кода (копирование)
    const copyCodeBtn = target.closest<HTMLButtonElement>('.btn-copy-code');
    if (copyCodeBtn) {
      event.preventDefault();
      event.stopPropagation();
      const wrapper = copyCodeBtn.closest<HTMLElement>('.code-block-wrapper');
      let textToCopy = '';
      const encoded = wrapper?.getAttribute('data-code');
      if (encoded) {
        textToCopy = decodeURIComponent(encoded);
      } else {
        const codeEl = wrapper?.querySelector('code');
        textToCopy = codeEl?.textContent || '';
      }
      if (textToCopy) {
        await navigator.clipboard.writeText(textToCopy);
        const span = copyCodeBtn.querySelector('span');
        if (span) {
          const prev = span.textContent;
          span.textContent = 'Скопировано!';
          copyCodeBtn.classList.add('copied');
          setTimeout(() => {
            span.textContent = prev;
            copyCodeBtn.classList.remove('copied');
          }, 2000);
        }
      }
      return true;
    }

    return false;
  }

  /**
   * Алиас для обратной совместимости
   */
  async handleCopyClick(event: MouseEvent): Promise<boolean> {
    return this.handleBlockInteraction(event);
  }

  /**
   * Открывает полноэкранный модальный просмотр диаграммы с масштабированием
   */
  openFullscreenModal(svgHtml: string, rawCode: string = ''): void {
    const existing = document.querySelector('.mermaid-fullscreen-overlay');
    if (existing) existing.remove();

    // Очищаем инлайновые ограничения max-width Mermaid, чтобы диаграмма масштабировалась в модальном окне
    const cleanSvgHtml = svgHtml
      .replace(/style="[^"]*max-width:[^;"]*;?[^"]*"/gi, '')
      .replace(/width="100%"/gi, '');

    let zoom = 1.0;
    let panX = 0;
    let panY = 0;
    let isDragging = false;
    let startX = 0;
    let startY = 0;

    const overlay = document.createElement('div');
    overlay.className = 'mermaid-fullscreen-overlay';

    overlay.innerHTML = `
      <div class="mermaid-modal-header">
        <div class="mermaid-modal-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="6" cy="6" r="3"/><circle cx="18" cy="18" r="3"/><path d="M6 9v3a3 3 0 0 0 3 3h6"/>
          </svg>
          <span>Диаграмма Mermaid (Полноэкранный просмотр)</span>
        </div>
        <div class="mermaid-modal-controls">
          <span class="mermaid-modal-hint">Зажмите ЛКМ для перемещения • Колесико для масштаба</span>
          <button type="button" class="mermaid-modal-btn btn-icon-only btn-zoom-out" title="Уменьшить масштаб">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
              <line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
          </button>
          <div class="zoom-level-badge" title="Текущий масштаб">
            <span class="zoom-level-text">100%</span>
          </div>
          <button type="button" class="mermaid-modal-btn btn-icon-only btn-zoom-in" title="Увеличить масштаб">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
              <line x1="12" y1="5" x2="12" y2="19"/>
              <line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
          </button>
          <button type="button" class="mermaid-modal-btn btn-zoom-reset" title="Вернуть исходный масштаб (100%) и центрировать">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
              <path d="M3 3v5h5"/>
            </svg>
            <span>Исходный вид</span>
          </button>
          ${rawCode ? `
            <button type="button" class="mermaid-modal-btn btn-modal-copy" title="Скопировать Mermaid код">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
              </svg>
              <span>Копировать код</span>
            </button>
          ` : ''}
          <button type="button" class="mermaid-modal-btn btn-close-modal" title="Закрыть (Esc)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"/>
              <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
            <span>Закрыть</span>
          </button>
        </div>
      </div>
      <div class="mermaid-modal-body">
        <div class="mermaid-modal-svg-wrapper">
          ${cleanSvgHtml}
        </div>
      </div>
    `;

    document.body.appendChild(overlay);

    const svgWrapper = overlay.querySelector<HTMLElement>('.mermaid-modal-svg-wrapper');
    const zoomText = overlay.querySelector<HTMLElement>('.zoom-level-text');
    const bodyEl = overlay.querySelector<HTMLElement>('.mermaid-modal-body');

    const applyTransform = (withTransition: boolean = false) => {
      if (svgWrapper) {
        svgWrapper.style.transition = withTransition ? 'transform 0.2s cubic-bezier(0.16, 1, 0.3, 1)' : 'none';
        svgWrapper.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
      }
      if (zoomText) {
        zoomText.textContent = `${Math.round(zoom * 100)}%`;
      }
    };

    const updateZoom = (newZoom: number, withTransition: boolean = true) => {
      zoom = Math.min(5.0, Math.max(0.2, Math.round(newZoom * 100) / 100));
      applyTransform(withTransition);
    };

    const resetView = () => {
      zoom = 1.0;
      panX = 0;
      panY = 0;
      applyTransform(true);
    };

    overlay.querySelector('.btn-zoom-in')?.addEventListener('click', (e) => {
      e.stopPropagation();
      updateZoom(zoom * 1.25, true);
    });

    overlay.querySelector('.btn-zoom-out')?.addEventListener('click', (e) => {
      e.stopPropagation();
      updateZoom(zoom * 0.8, true);
    });

    overlay.querySelector('.btn-zoom-reset')?.addEventListener('click', (e) => {
      e.stopPropagation();
      resetView();
    });

    overlay.querySelector('.btn-modal-copy')?.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (rawCode) {
        await navigator.clipboard.writeText(rawCode);
        const span = (e.currentTarget as HTMLElement).querySelector('span');
        if (span) {
          const prev = span.textContent;
          span.textContent = '✓ Скопировано!';
          setTimeout(() => span.textContent = prev, 2000);
        }
      }
    });

    // ── Перемещение захватом мыши (Pan / Drag) ──
    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0 && e.pointerType === 'mouse') return;
      isDragging = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
      bodyEl?.classList.add('grabbing');
      bodyEl?.setPointerCapture?.(e.pointerId);
      e.preventDefault();
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!isDragging) return;
      panX = e.clientX - startX;
      panY = e.clientY - startY;
      applyTransform(false);
    };

    const onPointerUp = (e: PointerEvent) => {
      if (!isDragging) return;
      isDragging = false;
      bodyEl?.classList.remove('grabbing');
      try {
        bodyEl?.releasePointerCapture?.(e.pointerId);
      } catch {}
    };

    bodyEl?.addEventListener('pointerdown', onPointerDown);
    bodyEl?.addEventListener('pointermove', onPointerMove);
    bodyEl?.addEventListener('pointerup', onPointerUp);
    bodyEl?.addEventListener('pointercancel', onPointerUp);

    // ── Масштабирование колесиком мыши ──
    bodyEl?.addEventListener('wheel', (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.15 : 0.85;
      updateZoom(zoom * factor, true);
    }, { passive: false });

    // ── Закрытие модального окна ──
    const closeModal = () => {
      document.removeEventListener('keydown', onKeyDown);
      overlay.remove();
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        closeModal();
      }
    };

    document.addEventListener('keydown', onKeyDown);
    overlay.querySelector('.btn-close-modal')?.addEventListener('click', closeModal);
  }
}
