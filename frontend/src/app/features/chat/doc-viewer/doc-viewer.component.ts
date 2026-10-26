import { 
  Component, 
  Input, 
  Output, 
  EventEmitter, 
  inject, 
  signal, 
  ElementRef, 
  ViewChild, 
  AfterViewChecked 
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { SafeHtml } from '@angular/platform-browser';
import { DocTab } from '../../../core/models/doc-viewer.model';
import { MarkdownService } from '../../../core/services/markdown.service';
import { I18nService } from '../../../core/services/i18n.service';

declare const Prism: any;

interface CodeLineDisplay {
  num: number;
  html: string;
  isHighlighted: boolean;
}

@Component({
  selector: 'app-doc-viewer',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './doc-viewer.component.html',
  styleUrls: ['./doc-viewer.component.scss']
})
export class DocViewerComponent implements AfterViewChecked {
  private readonly markdown = inject(MarkdownService);
  readonly i18n = inject(I18nService);

  @Input() tabs: DocTab[] = [];
  @Input() activeTabId: string | null = null;

  @Output() tabSelected = new EventEmitter<string>();
  @Output() tabClosed = new EventEmitter<string>();
  @Output() allTabsClosed = new EventEmitter<void>();
  @Output() retryRequested = new EventEmitter<DocTab>();

  @ViewChild('docBody') private docBody?: ElementRef<HTMLDivElement>;
  @ViewChild('codeContainer') private codeContainer?: ElementRef<HTMLDivElement>;
  @ViewChild('markdownPreview') private markdownPreview?: ElementRef<HTMLDivElement>;

  readonly copiedPath = signal(false);
  readonly copiedContent = signal(false);

  private lastScrolledTabId: string | null = null;
  private needsScroll = false;
  private lastRenderedMarkdownKey: string | null = null;
  private linesCache = new Map<string, CodeLineDisplay[]>();
  private markdownCache = new Map<string, SafeHtml>();

  get activeTab(): DocTab | undefined {
    return this.tabs.find(t => t.id === this.activeTabId);
  }

  ngAfterViewChecked(): void {
    if (this.needsScroll && this.docBody) {
      this.needsScroll = false;
      this.scrollToTargetLine();
    }
    if (this.markdownPreview && this.activeTab?.viewMode === 'rendered') {
      const currentKey = `${this.activeTab.id}_${this.activeTab.scopeMode}_${this.activeTab.fullContent?.length}`;
      if (this.lastRenderedMarkdownKey !== currentKey) {
        this.lastRenderedMarkdownKey = currentKey;
        this.markdown.renderMermaid(this.markdownPreview.nativeElement);
        this.markdown.highlightCode(this.markdownPreview.nativeElement);
      }
    }
  }

  selectTab(tabId: string): void {
    this.tabSelected.emit(tabId);
    if (tabId !== this.lastScrolledTabId) {
      this.needsScroll = true;
      this.lastScrolledTabId = tabId;
    }
  }

  closeTab(tabId: string, event: MouseEvent): void {
    event.stopPropagation();
    this.tabClosed.emit(tabId);
  }

  closeAll(): void {
    this.allTabsClosed.emit();
  }

  retryLoad(tab: DocTab): void {
    this.retryRequested.emit(tab);
  }

  onMarkdownClick(event: MouseEvent): void {
    this.markdown.handleBlockInteraction(event);
  }

  isMarkdown(filepath: string): boolean {
    return /\.(md|markdown)$/i.test(filepath);
  }

  toggleViewMode(tab: DocTab): void {
    tab.viewMode = tab.viewMode === 'rendered' ? 'raw' : 'rendered';
    this.lastRenderedMarkdownKey = null;
    if (tab.viewMode === 'raw') {
      this.needsScroll = true;
    }
  }

  toggleScopeMode(tab: DocTab): void {
    tab.scopeMode = tab.scopeMode === 'full' ? 'chunk' : 'full';
    this.lastRenderedMarkdownKey = null;
    this.needsScroll = true;
  }

  getRenderedMarkdown(tab: DocTab): SafeHtml {
    const raw = tab.fullContent || '';
    if (!raw) return '';

    const cacheKey = `${tab.id}_${tab.scopeMode}_${tab.startLine}_${tab.endLine}_${raw.length}`;
    const cached = this.markdownCache.get(cacheKey);
    if (cached) {
      return cached;
    }

    let textToRender = raw;
    if (tab.scopeMode === 'chunk' && tab.startLine && tab.endLine) {
      const lines = raw.split('\n');
      const start = Math.max(1, tab.startLine) - 1;
      const end = Math.min(lines.length, tab.endLine);
      textToRender = lines.slice(start, end).join('\n');
    }

    const rendered = this.markdown.renderSafe(textToRender);
    if (this.markdownCache.size > 20) {
      this.markdownCache.clear();
    }
    this.markdownCache.set(cacheKey, rendered);
    return rendered;
  }

  getDisplayLines(tab: DocTab): CodeLineDisplay[] {
    const raw = tab.fullContent || '';
    if (!raw) return [];

    const cacheKey = `${tab.id}_${tab.scopeMode}_${tab.startLine}_${tab.endLine}_${raw.length}`;
    const cached = this.linesCache.get(cacheKey);
    if (cached) {
      return cached;
    }

    const lang = this.detectLanguage(tab.filepath);
    const highlightedCode = this.highlightSyntax(raw, lang);
    const allLines = highlightedCode.split('\n');

    const total = allLines.length;
    const start = Math.max(1, tab.startLine || 1);
    const end = Math.min(total, tab.endLine || total);

    let result: CodeLineDisplay[];
    if (tab.scopeMode === 'chunk') {
      result = allLines.slice(start - 1, end).map((html, idx) => {
        const lineNum = start + idx;
        return {
          num: lineNum,
          html: html || '&nbsp;',
          isHighlighted: true
        };
      });
    } else {
      result = allLines.map((html, idx) => {
        const lineNum = idx + 1;
        const isHighlighted = !!(tab.startLine && tab.endLine && lineNum >= start && lineNum <= end);
        return {
          num: lineNum,
          html: html || '&nbsp;',
          isHighlighted
        };
      });
    }

    if (this.linesCache.size > 20) {
      this.linesCache.clear();
    }
    this.linesCache.set(cacheKey, result);
    return result;
  }

  private detectLanguage(filepath: string): string {
    const ext = filepath.split('.').pop()?.toLowerCase() || '';
    const map: Record<string, string> = {
      cs: 'csharp',
      py: 'python',
      ts: 'typescript',
      js: 'javascript',
      json: 'json',
      yaml: 'yaml',
      yml: 'yaml',
      sh: 'bash',
      bash: 'bash',
      sql: 'sql',
      md: 'markdown',
      markdown: 'markdown',
      dockerfile: 'docker',
      rs: 'rust',
      go: 'go',
      xml: 'markup',
      html: 'markup',
      css: 'css'
    };
    return map[ext] || 'clike';
  }

  private highlightSyntax(code: string, lang: string): string {
    if (typeof Prism !== 'undefined' && Prism.languages) {
      const grammar = Prism.languages[lang] || Prism.languages.clike;
      if (grammar) {
        try {
          return Prism.highlight(code, grammar, lang);
        } catch {
          // fallback
        }
      }
    }
    // Fallback: базовое экранирование HTML
    return code
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  private scrollToTargetLine(): void {
    const bodyEl = this.docBody?.nativeElement;
    if (!bodyEl) return;
    const targetEl = bodyEl.querySelector('.code-line-row.line-highlight') as HTMLElement;
    if (targetEl) {
      const containerRect = bodyEl.getBoundingClientRect();
      const targetRect = targetEl.getBoundingClientRect();
      const targetTop = targetRect.top - containerRect.top + bodyEl.scrollTop;
      const targetScrollTop = targetTop - (bodyEl.clientHeight / 2) + (targetRect.height / 2);
      bodyEl.scrollTo({
        top: Math.max(0, targetScrollTop),
        behavior: 'smooth'
      });
    }
  }

  async copyPath(tab: DocTab): Promise<void> {
    try {
      await navigator.clipboard.writeText(tab.filepath);
      this.copiedPath.set(true);
      setTimeout(() => this.copiedPath.set(false), 2000);
    } catch (e) {
      console.warn('Не удалось скопировать путь:', e);
    }
  }

  async copyContent(tab: DocTab): Promise<void> {
    try {
      await navigator.clipboard.writeText(tab.fullContent || '');
      this.copiedContent.set(true);
      setTimeout(() => this.copiedContent.set(false), 2000);
    } catch (e) {
      console.warn('Не удалось скопировать содержимое:', e);
    }
  }

  getFileIcon(filepath: string): string {
    const ext = filepath.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'cs': return 'C#';
      case 'py': return 'PY';
      case 'ts': return 'TS';
      case 'js': return 'JS';
      case 'json': return '{}';
      case 'md': return 'MD';
      case 'sql': return 'SQL';
      case 'yaml':
      case 'yml': return 'YML';
      case 'sh': return 'SH';
      default: return 'FILE';
    }
  }
}
