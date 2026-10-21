import { Component, inject, signal, ViewChild, ElementRef, AfterViewInit, effect } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { ThemeService } from '../../core/services/theme.service';
import { MarkdownService } from '../../core/services/markdown.service';
import { ChatStorageService } from '../../core/services/chat-storage.service';
import { ChatMessage, ChatSource } from '../../core/models/chat.model';

export type { ChatMessage, ChatSource };

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
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

  readonly messages = signal<ChatMessage[]>(this.loadInitialMessages());
  readonly userInput = signal('');
  readonly isGenerating = signal(false);
  readonly hasSavedHistory = signal<boolean>(false);

  readonly modalFile = signal<{ filepath: string; startLine: number; endLine: number; content: string } | null>(null);

  constructor() {
    this.hasSavedHistory.set(this.chatStorage.hasHistory());

    // Перерисовывать диаграммы Mermaid при переключении темы оформления
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

  renderMarkdown(content: string): string {
    return this.markdown.render(content);
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

  openSource(filepath: string, lines: string, repoId?: string): void {
    let start = 1;
    let end: number | undefined;
    if (lines && lines.includes('-')) {
      const parts = lines.split('-');
      start = parseInt(parts[0], 10) || 1;
      end = parseInt(parts[1], 10) || undefined;
    }

    this.api.getFileContent(filepath, start, end, repoId, false).subscribe({
      next: (res) => {
        this.modalFile.set({
          filepath: res.filepath,
          startLine: res.start_line,
          endLine: res.end_line,
          content: res.content
        });
      },
      error: (err) => {
        alert(err.message);
      }
    });
  }

  closeSourceModal(): void {
    this.modalFile.set(null);
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
