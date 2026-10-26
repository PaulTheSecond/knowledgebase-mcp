import { 
  Component, 
  inject, 
  signal, 
  computed, 
  ViewChild, 
  ElementRef, 
  AfterViewInit, 
  effect 
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { SafeHtml } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { ThemeService } from '../../core/services/theme.service';
import { MarkdownService } from '../../core/services/markdown.service';
import { ChatStorageService } from '../../core/services/chat-storage.service';
import { ChatMessage, ChatSource } from '../../core/models/chat.model';
import { DocTab } from '../../core/models/doc-viewer.model';
import { DocViewerComponent } from './doc-viewer/doc-viewer.component';

export type { ChatMessage, ChatSource };

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule, DocViewerComponent],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss']
})
export class ChatComponent implements AfterViewInit {
  private readonly api = inject(ApiService);
  readonly i18n = inject(I18nService);
  private readonly theme = inject(ThemeService);
  private readonly markdown = inject(MarkdownService);
  private readonly chatStorage = inject(ChatStorageService);

  @ViewChild('messagesContainer') private messagesContainer!: ElementRef<HTMLDivElement>;
  @ViewChild('chatWorkspace') private chatWorkspace!: ElementRef<HTMLDivElement>;

  // Состояние чата
  readonly messages = signal<ChatMessage[]>(this.loadInitialMessages());
  readonly userInput = signal('');
  readonly isGenerating = signal(false);
  readonly hasSavedHistory = signal<boolean>(false);
  readonly copiedMessageIndex = signal<number | null>(null);

  // Состояние вкладок панели документов (IDE-style)
  readonly docTabs = signal<DocTab[]>([]);
  readonly activeTabId = signal<string | null>(null);
  readonly isViewerOpen = computed(() => this.docTabs().length > 0);

  // Состояние сплиттера (resizable split view)
  readonly viewerWidthPercent = signal<number>(this.loadInitialSplitWidth());
  readonly isDraggingSplitter = signal<boolean>(false);

  constructor() {
    this.hasSavedHistory.set(this.chatStorage.hasHistory());

    // Перерисовывать диаграммы Mermaid при смене темы оформления
    effect(() => {
      this.theme.currentTheme();
      setTimeout(() => {
        if (this.messagesContainer) {
          this.markdown.renderMermaid(this.messagesContainer.nativeElement);
        }
      }, 100);
    });
  }

  ngAfterViewInit(): void {
    setTimeout(() => {
      if (this.messagesContainer) {
        this.markdown.highlightCode(this.messagesContainer.nativeElement);
        this.markdown.renderMermaid(this.messagesContainer.nativeElement);
        this.scrollToBottom();
      }
    }, 100);
  }

  private loadInitialMessages(): ChatMessage[] {
    const saved = this.chatStorage.loadHistory();
    if (saved && saved.length > 0) {
      return saved;
    }
    return [
      {
        role: 'system',
        content: this.i18n.t('chat.welcome')
      }
    ];
  }

  private loadInitialSplitWidth(): number {
    const saved = localStorage.getItem('kb_chat_split_ratio');
    if (saved) {
      const parsed = parseFloat(saved);
      if (!isNaN(parsed) && parsed >= 20 && parsed <= 80) {
        return parsed;
      }
    }
    return 45; // 45% по умолчанию
  }

  // ── Обработка перетаскивания разделителя (Splitter Drag-to-Resize) ──
  onSplitterMouseDown(event: MouseEvent): void {
    event.preventDefault();
    this.isDraggingSplitter.set(true);

    const onMouseMove = (moveEvent: MouseEvent) => {
      const container = this.chatWorkspace?.nativeElement;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const rightDistance = rect.right - moveEvent.clientX;
      const newPercent = (rightDistance / rect.width) * 100;
      const clamped = Math.max(20, Math.min(80, newPercent));
      this.viewerWidthPercent.set(clamped);
    };

    const onMouseUp = () => {
      this.isDraggingSplitter.set(false);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      localStorage.setItem('kb_chat_split_ratio', this.viewerWidthPercent().toFixed(1));
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  }

  resetSplitter(): void {
    this.viewerWidthPercent.set(45);
    localStorage.setItem('kb_chat_split_ratio', '45');
  }

  // ── Управление вкладками документов (Tabs) ──
  openSource(filepath: string, lines: string, repoId?: string): void {
    let start: number | undefined;
    let end: number | undefined;
    if (lines && lines.includes('-')) {
      const parts = lines.split('-');
      start = parseInt(parts[0], 10) || undefined;
      end = parseInt(parts[1], 10) || undefined;
    }

    const tabId = `${repoId || 'local'}:${filepath}`;
    const filename = filepath.split('/').pop()?.split('\\').pop() || filepath;

    const existing = this.docTabs().find(t => t.id === tabId);
    if (existing) {
      // Вкладка уже открыта — обновляем подсвечиваемые строки и делаем активной
      this.docTabs.update(tabs => tabs.map(t => {
        if (t.id === tabId) {
          return {
            ...t,
            startLine: start,
            endLine: end,
            scopeMode: 'full'
          };
        }
        return t;
      }));
      this.activeTabId.set(tabId);
      return;
    }

    // Создаем новую вкладку
    const newTab: DocTab = {
      id: tabId,
      filepath,
      filename,
      repoId,
      startLine: start,
      endLine: end,
      loading: true,
      viewMode: /\.(md|markdown)$/i.test(filepath) ? 'rendered' : 'raw',
      scopeMode: 'full'
    };

    this.docTabs.update(tabs => [...tabs, newTab]);
    this.activeTabId.set(tabId);
    this.loadTabContent(newTab);
  }

  loadTabContent(tab: DocTab): void {
    this.docTabs.update(tabs => tabs.map(t => {
      if (t.id === tab.id) {
        return { ...t, loading: true, error: undefined };
      }
      return t;
    }));

    // Загружаем полный файл с бэкенда для навигации
    this.api.getFileContent(tab.filepath, 1, undefined, tab.repoId, true).subscribe({
      next: (res) => {
        this.docTabs.update(tabs => tabs.map(t => {
          if (t.id === tab.id) {
            return {
              ...t,
              fullContent: res.content,
              totalLines: res.total_lines,
              loading: false,
              error: undefined
            };
          }
          return t;
        }));
      },
      error: (err) => {
        this.docTabs.update(tabs => tabs.map(t => {
          if (t.id === tab.id) {
            return {
              ...t,
              loading: false,
              error: err?.message || 'Не удалось загрузить файл'
            };
          }
          return t;
        }));
      }
    });
  }

  onTabSelected(tabId: string): void {
    this.activeTabId.set(tabId);
  }

  onTabClosed(tabId: string): void {
    const currentTabs = this.docTabs();
    const idx = currentTabs.findIndex(t => t.id === tabId);
    if (idx === -1) return;

    const remaining = currentTabs.filter(t => t.id !== tabId);
    this.docTabs.set(remaining);

    if (this.activeTabId() === tabId) {
      if (remaining.length > 0) {
        const nextActive = remaining[Math.min(idx, remaining.length - 1)];
        this.activeTabId.set(nextActive.id);
      } else {
        this.activeTabId.set(null);
      }
    }
  }

  onAllTabsClosed(): void {
    this.docTabs.set([]);
    this.activeTabId.set(null);
  }

  // ── Отправка и получение сообщений ──
  sendMessage(): void {
    const query = this.userInput().trim();
    if (!query || this.isGenerating()) return;

    this.userInput.set('');
    const history = [...this.messages(), { role: 'user' as const, content: query }];
    this.messages.set(history);
    this.chatStorage.saveHistory(history);
    this.hasSavedHistory.set(true);
    this.scrollToBottom();

    this.isGenerating.set(true);

    const assistantMsgIndex = history.length;
    this.messages.update(msgs => [...msgs, { role: 'assistant', content: '' }]);

    this.api.getAuthToken().subscribe({
      next: ({ token }) => {
        this.streamResponse(token, history.filter(m => m.role !== 'system'), assistantMsgIndex);
      },
      error: (err) => {
        this.messages.update(msgs => {
          const updated = [...msgs];
          updated[assistantMsgIndex].content = `⚠️ ${err.message}`;
          return updated;
        });
        this.chatStorage.saveHistory(this.messages());
        this.isGenerating.set(false);
      }
    });
  }

  private async streamResponse(token: string, history: Array<{ role: string; content: string }>, assistantMsgIndex: number): Promise<void> {
    try {
      const response = await fetch(`/api/chat/stream?token=${token}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept-Language': this.i18n.currentLocale()
        },
        body: JSON.stringify({ messages: history })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No readable stream');

      const decoder = new TextDecoder();
      let buffer = '';
      let fullText = '';
      let sources: ChatSource[] | undefined;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;

          const dataContent = trimmed.slice(6);
          if (dataContent === '[DONE]') continue;

          try {
            const parsed = JSON.parse(dataContent);
            if (parsed.token) {
              fullText += parsed.token;
              this.updateAssistantMessage(assistantMsgIndex, fullText, sources);
            } else if (parsed.sources) {
              sources = parsed.sources;
              this.updateAssistantMessage(assistantMsgIndex, fullText, sources);
            } else if (parsed.error) {
              throw new Error(parsed.error);
            }
          } catch {
            // chunk parse
          }
        }
      }

      this.updateAssistantMessage(assistantMsgIndex, fullText, sources);
      this.chatStorage.saveHistory(this.messages());
      this.hasSavedHistory.set(true);
      this.highlightAndRender();
    } catch (err: any) {
      this.messages.update(msgs => {
        const updated = [...msgs];
        updated[assistantMsgIndex].content = `⚠️ Ошибка: ${err.message}`;
        return updated;
      });
      this.chatStorage.saveHistory(this.messages());
    } finally {
      this.isGenerating.set(false);
    }
  }

  private updateAssistantMessage(idx: number, text: string, sources?: ChatSource[]): void {
    this.messages.update(msgs => {
      const updated = [...msgs];
      if (updated[idx]) {
        updated[idx] = {
          ...updated[idx],
          content: text,
          sources: sources
        };
      }
      return updated;
    });
    this.scrollToBottom();
  }

  renderMarkdown(content: string): SafeHtml {
    return this.markdown.renderSafe(content);
  }

  private highlightAndRender(): void {
    setTimeout(() => {
      if (this.messagesContainer) {
        this.markdown.highlightCode(this.messagesContainer.nativeElement);
        this.markdown.renderMermaid(this.messagesContainer.nativeElement);
      }
    }, 50);
  }

  onMessagesClick(event: MouseEvent): void {
    this.markdown.handleCopyClick(event);
  }

  copyMessageMarkdown(msg: ChatMessage, index: number, event?: Event): void {
    if (event) {
      event.stopPropagation();
    }
    let text = msg.content;
    if (msg.role === 'assistant' && msg.sources && msg.sources.length > 0) {
      text += '\n\n---\n**' + this.i18n.t('chat.context_sources') + '**\n';
      for (const src of msg.sources) {
        text += `- \`${src.filepath}\` (${src.lines})\n`;
      }
    }
    this.copyToClipboard(text).then(success => {
      if (success) {
        this.copiedMessageIndex.set(index);
        setTimeout(() => {
          if (this.copiedMessageIndex() === index) {
            this.copiedMessageIndex.set(null);
          }
        }, 2000);
      }
    });
  }

  private async copyToClipboard(text: string): Promise<boolean> {
    if (navigator?.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch {
        // Fallback ниже
      }
    }
    try {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      const success = document.execCommand('copy');
      document.body.removeChild(textarea);
      return success;
    } catch {
      return false;
    }
  }

  clearChat(): void {
    const isOnlyDefault = this.messages().length === 1 && this.messages()[0].role === 'system';
    if (isOnlyDefault) return;

    if (confirm(this.i18n.t('chat.clear_confirm'))) {
      this.chatStorage.clearHistory();
      this.messages.set([
        {
          role: 'system',
          content: this.i18n.t('chat.welcome')
        }
      ]);
      this.hasSavedHistory.set(false);
    }
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      if (this.messagesContainer) {
        const el = this.messagesContainer.nativeElement;
        el.scrollTop = el.scrollHeight;
      }
    }, 50);
  }
}
