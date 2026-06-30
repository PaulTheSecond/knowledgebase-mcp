# Story 3.4: Markdown, Highlighting and Sources Rendering

Status: ready-for-dev

## Story

As a разработчик,
I want видеть ответы модели с красивой разметкой Markdown, подсветкой кода и кликабельными ссылками на файлы-источники,
so that я мог легко читать код в ответах и переходить к файлам на диске.

## Acceptance Criteria

1. **Given** завершенный поток SSE ответа,
   **When** фронтенд отображает ответ,
   **Then** он парсит текст с помощью встроенной no-build Markdown библиотеки (например, `marked.js`) и подсвечивает синтаксис блоков кода (например, через `Prism.js`).
2. **And** под блоком ответа выводится список кликабельных тегов с названиями файлов-источников (например, `db.py#L12-45`). При клике на тег открывается окно просмотра содержимого этого файла на указанных строках (или в VS Code по схеме `vscode://`).

## Tasks / Subtasks

- [ ] Подключение No-Build библиотек на клиенте (AC: 1)
  - [ ] Добавить в `index.html` загрузку `marked.min.js` и `prism.js`/`prism.css` через локальные статические копии (сохранить их в `dashboard/static/vendor/`) или через доверенный CDN.
- [ ] Парсинг Markdown и подсветка кода (AC: 1)
  - [ ] В `dashboard/static/js/chat.js` после завершения чтения потока SSE вызвать `marked.parse(rawText)` для преобразования Markdown в HTML.
  - [ ] Вызвать `Prism.highlightAllUnder(messageContainer)` для автоматической подсветки синтаксиса блоков `<pre><code>`.
- [ ] Вывод ссылок на файлы-источники (AC: 2)
  - [ ] При разборе финального пакета SSE `sources` извлечь перечень файлов и диапазонов строк.
  - [ ] Отрендерить список тегов-ссылок под текстом ответа.
- [ ] Логика перехода по ссылкам-источникам (AC: 2)
  - [ ] Написать обработчик клика на тег-источник.
  - [ ] Сделать GET запрос на получение содержимого файла (REST эндпоинт `/api/file/read?path=...&start=...&end=...`) и вывести его в модальном окне дашборда с подсветкой кода.
  - [ ] Или генерировать URI схемы для IDE: `vscode://file/<absolute_path>:<line>`.

## Dev Notes

Пример разметки источников под сообщением:
```html
<div class="sources-container">
  <span class="source-title">Использованные источники:</span>
  <a href="#" class="source-tag" onclick="openFileSource('knowledge_mcp/db.py', 10, 30)">db.py (строки 10-30)</a>
</div>
```
Для реализации просмотра файлов на бэкенде в `routes/status.py` можно добавить эндпоинт:
`GET /api/file/read?filepath=...&start_line=...&end_line=...`, который проверяет права доступа (только файлы внутри локального кэша `/data/clones` или папки проекта, во избежание path traversal уязвимости!) и возвращает фрагмент кода.

### References

- [Требования RAG чата: prd.md#4.4.-Интегрированный-RAG-чат](file:///_bmad-output/planning-artifacts/prds/prd-knowledgebase-mcp-2026-06-30/prd.md#L118-L131)

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
