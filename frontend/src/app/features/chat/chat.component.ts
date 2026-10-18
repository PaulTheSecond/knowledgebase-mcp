import { Component, inject, signal, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';

declare const marked: any;
declare const Prism: any;

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  sources?: Array<{ filepath: string; lines: string }>;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss']
})
export class ChatComponent {
  private readonly api = inject(ApiService);
  readonly i18n = inject(I18nService);

  @ViewChild('messagesContainer') private messagesContainer!: ElementRef<HTMLDivElement>;

  readonly messages = signal<ChatMessage[]>([
    {
      role: 'system',
      content: this.i18n.t('chat.welcome')
    }
  ]);

  readonly userInput = signal('');
  readonly isGenerating = signal(false);

  readonly modalFile = signal<{ filepath: string; startLine: number; endLine: number; content: string } | null>(null);

  sendMessage(): void {
    const query = this.userInput().trim();
    if (!query || this.isGenerating()) return;

    this.userInput.set('');
    const history = [...this.messages(), { role: 'user' as const, content: query }];
    this.messages.set(history);
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
      let sources: Array<{ filepath: string; lines: string }> | undefined;

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
      this.highlightCode();
    } catch (err: any) {
      this.messages.update(msgs => {
        const updated = [...msgs];
        updated[assistantMsgIndex].content = `⚠️ Ошибка: ${err.message}`;
        return updated;
      });
    } finally {
      this.isGenerating.set(false);
    }
  }

  private updateAssistantMessage(idx: number, text: string, sources?: Array<{ filepath: string; lines: string }>): void {
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
    if (typeof marked !== 'undefined') {
      return marked.parse(content);
    }
    return content;
  }

  private highlightCode(): void {
    setTimeout(() => {
      if (typeof Prism !== 'undefined' && this.messagesContainer) {
        Prism.highlightAllUnder(this.messagesContainer.nativeElement);
      }
    }, 50);
  }

  openSource(filepath: string, lines: string): void {
    let start = 1;
    let end: number | undefined;
    if (lines && lines.includes('-')) {
      const parts = lines.split('-');
      start = parseInt(parts[0], 10) || 1;
      end = parseInt(parts[1], 10) || undefined;
    }

    this.api.getFileContent(filepath, start, end).subscribe({
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
