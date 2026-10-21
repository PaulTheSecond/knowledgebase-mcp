import { 
  Component, 
  Input, 
  Output, 
  EventEmitter, 
  inject, 
  computed, 
  signal, 
  ElementRef, 
  ViewChild, 
  AfterViewChecked 
} from '@angular/core';
import { CommonModule } from '@angular/common';
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

  @ViewChild('codeContainer') private codeContainer?: ElementRef<HTMLDivElement>;
  @ViewChild('markdownPreview') private markdownPreview?: ElementRef<HTMLDivElement>;

  readonly copiedPath = signal(false);
  readonly copiedContent = signal(false);

  private lastScrolledTabId: string | null = null;
  private needsScroll = false;

  get activeTab(): DocTab | undefined {
    return this.tabs.find(t => t.id === this.activeTabId);
  }

  ngAfterViewChecked(): void {
    if (this.needsScroll && this.codeContainer) {
      this.needsScroll = false;
      this.scrollToTargetLine();
    }
    if (this.markdownPreview && this.activeTab?.viewMode === 'rendered') {
      this.markdown.renderMermaid(this.markdownPreview.nativeElement);
      this.markdown.highlightCode(this.markdownPreview.nativeElement);
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

  isMarkdown(filepath: string): boolean {
    return /\.(md|markdown)$/i.test(filepath);
  }

  toggleViewMode(tab: DocTab): void {
    tab.viewMode = tab.viewMode === 'rendered' ? 'raw' : 'rendered';
    if (tab.viewMode === 'raw') {
      this.needsScroll = true;
    }
  }

  toggleScopeMode(tab: DocTab): void {
    tab.scopeMode = tab.scopeMode === 'full' ? 'chunk' : 'full';
    this.needsScroll = true;
  }

  getRenderedMarkdown(content?: string): string {
    return this.markdown.render(content || '');
  }

  getDisplayLines(tab: DocTab): CodeLineDisplay[] {
    const raw = tab.fullContent || '';
    if (!raw) return [];

    const lang = this.detectLanguage(tab.filepath);
    const highlightedCode = this.highlightSyntax(raw, lang);
    const allLines = highlightedCode.split('\n');

    const total = allLines.length;
    const start = Math.max(1, tab.startLine || 1);
    const end = Math.min(total, tab.endLine || total);

    if (tab.scopeMode === 'chunk') {
      return allLines.slice(start - 1, end).map((html, idx) => {
        const lineNum = start + idx;
        return {
          num: lineNum,
          html: html || '&nbsp;',
          isHighlighted: true
        };
      });
    }

    return allLines.map((html, idx) => {
      const lineNum = idx + 1;
      const isHighlighted = !!(tab.startLine && tab.endLine && lineNum >= start && lineNum <= end);
      return {
        num: lineNum,
        html: html || '&nbsp;',
        isHighlighted
      };
    });
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
    if (!this.codeContainer) return;
    const container = this.codeContainer.nativeElement;
    const targetEl = container.querySelector('.code-line-row.line-highlight') as HTMLElement;
    if (targetEl) {
      targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
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
