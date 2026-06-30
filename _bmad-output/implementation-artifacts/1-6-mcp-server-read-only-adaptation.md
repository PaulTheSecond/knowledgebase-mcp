# Story 1.6: MCP Server Read-Only Adaptation

Status: ready-for-dev

## Story

As a ИИ-ассистент в IDE,
I want осуществлять поиск по коду через MCP-инструменты в режиме Read-Only, а операции изменения делегировать демону,
so that мои действия не вызывали конфликтов блокировки SQLite с фоновым воркером.

## Acceptance Criteria

1. **Given** запущенный MCP-сервер,
   **When** я вызываю инструмент удаления репозитория `knowledge_delete_repo` или синхронизации `knowledge_sync_repo`,
   **Then** MCP-сервер отправляет HTTP-запрос к Daemon-процессу на удаление/синхронизацию, а не обращается напрямую к таблицам записи SQLite.
2. **And** все поисковые инструменты (`knowledge_search`, `knowledge_get_symbols`) по-прежнему напрямую читают SQLite БД в режиме WAL.

## Tasks / Subtasks

- [ ] Модификация инструмента удаления репозитория (AC: 1)
  - [ ] Обновить обработчик `knowledge_delete_repo` в `mcp/server.py`.
  - [ ] Вместо вызова `db.delete_repo` совершать HTTP-запрос `DELETE /api/repo/{repo_id}` к Daemon-процессу.
- [ ] Модификация инструмента синхронизации (AC: 1)
  - [ ] Обновить обработчик `knowledge_sync_repo` в `mcp/server.py`.
  - [ ] Отправлять запрос `POST /api/sync/{repo_id}` к Daemon-процессу (чтобы задача встала в БД-очередь).
- [ ] Валидация Read-Only статуса соединения (AC: 2)
  - [ ] Убедиться, что MCP-процесс инициализирует соединение с SQLite как Read-Only, если это возможно в Python `sqlite3.connect` (например, `uri=True` и `file:db.db?mode=ro`).

## Dev Notes

В `sqlite3` режим Read-Only открывается через URI-подключение:
```python
import sqlite3
conn = sqlite3.connect("file:/data/knowledge.db?mode=ro", uri=True)
```
Это гарантирует, что MCP-процесс физически не сможет выполнить команду записи, предотвращая любые риски повреждения БД.
Эндпоинты Daemon-процесса для вызова мутаций:
- `POST http://127.0.0.1:8000/api/sync/{repo_id}`
- `DELETE http://127.0.0.1:8000/api/repo/{repo_id}`

### References

- [Архитектурный инвариант Read-Only MCP: ARCHITECTURE-SPINE.md#AD-1](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md#L29-L37)
- [Ревью безопасности: review-spine.md](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/review-spine.md)

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
