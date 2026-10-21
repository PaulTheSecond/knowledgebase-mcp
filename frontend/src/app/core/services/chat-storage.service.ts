import { Injectable } from '@angular/core';
import { ChatMessage } from '../models/chat.model';

const STORAGE_KEY = 'kb_chat_history_v1';

/**
 * Сервис локального хранения истории сообщений в браузере (LocalStorage).
 * Обеспечивает сохранение диалога между перезагрузками страницы
 * и предоставляет методы для сброса/очистки истории.
 */
@Injectable({
  providedIn: 'root'
})
export class ChatStorageService {
  /**
   * Загружает сохраненную историю диалога.
   */
  loadHistory(): ChatMessage[] | null {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed;
      }
    } catch (e) {
      console.warn('Не удалось загрузить историю диалога из LocalStorage:', e);
    }
    return null;
  }

  /**
   * Сохраняет историю диалога в LocalStorage.
   */
  saveHistory(messages: ChatMessage[]): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch (e) {
      console.warn('Не удалось сохранить историю диалога в LocalStorage:', e);
    }
  }

  /**
   * Очищает сохраненную историю диалога.
   */
  clearHistory(): void {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      console.warn('Не удалось очистить историю диалога:', e);
    }
  }

  /**
   * Проверяет, сохранена ли история в LocalStorage.
   */
  hasHistory(): boolean {
    return !!localStorage.getItem(STORAGE_KEY);
  }
}
