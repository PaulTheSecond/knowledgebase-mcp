# Story 3.3: Chat UI Implementation

Status: ready-for-dev

## Story

As a разработчик,
I want общаться с базой знаний в удобном чат-интерфейсе в браузере с отображением истории сообщений и автопрокруткой,
so that процесс тестирования RAG-поиска был интерактивным.

## Acceptance Criteria

1. **Given** вкладку "RAG Chat" в панели управления,
   **When** загружается страница дашборда,
   **Then** фронтенд считывает состояние фича-флага чата. Если чат отключен (`enabled: false`), вкладка "RAG Chat" скрывается из меню навигации.
2. **And** при включенном чате при вводе вопроса и нажатии "Send" сообщение добавляется в локальную историю в оперативной памяти браузера, а на бэкенд отправляется полный массив истории.
3. **And** во время ожидания выводится анимированный индикатор загрузки ("model is thinking..."), а по мере прихода токенов по каналу SSE текст ответа плавно рендерится в окне чата с автопрокруткой вниз.

## Tasks / Subtasks

- [ ] Интеграция фича-флага на клиенте (AC: 1)
  - [ ] Обновить эндпоинт `GET /api/status` на бэкенде, чтобы он возвращал значение `chat_enabled`.
  - [ ] В `dashboard/static/js/app.js` считывать `chat_enabled` при загрузке. Если `false`, скрыть элемент меню `#nav-chat` из сайдбара.
- [ ] Верстка чат-интерфейса (AC: 2)
  - [ ] Создать разметку чат-секции в `index.html` (контейнер для сообщений, блок ввода, кнопка отправки).
  - [ ] Оформить стили чата в `styles.css` (сообщения от пользователя выровнены по правому краю, от ассистента — по левому, разные цвета фона, скругленные углы).
- [ ] Реализация чат-клиента на Vanilla JS (AC: 2, 3)
  - [ ] Создать файл `knowledge_mcp/dashboard/static/js/chat.js`.
  - [ ] Объявить массив `let chatHistory = []` в ОЗУ страницы.
  - [ ] Написать обработчик отправки сообщения. Сообщение добавляется в `chatHistory` и выводится на экран.
- [ ] SSE-соединение и рендеринг стриминга (AC: 3)
  - [ ] Сделать запрос на получение временного токена.
  - [ ] Открыть соединение `const eventSource = new EventSource("/api/chat/stream?token=" + token)`.
  - [ ] Написать обработчик `eventSource.onmessage`:
    - При получении `{token: "..."}` добавлять символ/слово в DOM-элемент текущего ответа ассистента.
    - Вызывать плавную прокрутку вниз: `chatContainer.scrollTo({top: chatContainer.scrollHeight, behavior: 'smooth'})`.
    - Скрыть индикатор загрузки ("думает...") при приходе первого токена.

## Dev Notes

История диалога в ОЗУ должна хранить сообщения в формате OpenAI:
```javascript
chatHistory.push({ role: "user", content: question });
```
При отправке запроса SSE используйте `fetch` с методом `POST` для передачи тела запроса (массива сообщений), а затем считывайте тело ответа как поток (ReadableStream) с помощью `Reader`, так как стандартный `EventSource` браузера поддерживает только `GET` запросы без тела.
Пример реализации `fetch` POST SSE:
```javascript
const response = await fetch('/api/chat/stream?token=' + token, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: chatHistory })
});
const reader = response.body.getReader();
const decoder = new TextDecoder();
// Чтение потока в цикле
```
Это позволит передавать историю диалога любой длины на бэкенд в теле запроса!

### References

- [Функциональные требования чата: prd.md#4.4.-Интегрированный-RAG-чат](file:///_bmad-output/planning-artifacts/prds/prd-knowledgebase-mcp-2026-06-30/prd.md#L118-L131)

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
