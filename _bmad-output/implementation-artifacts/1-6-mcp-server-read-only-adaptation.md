# Story 1.6: MCP Server Read-Only Adaptation

Status: done

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

- [x] Модификация инструмента удаления репозитория (AC: 1)
  - [x] Обновить обработчик `knowledge_delete_repo` в `mcp/server.py`.
  - [x] Вместо вызова `db.delete_repo` совершать HTTP-запрос `DELETE /api/repo/{repo_id}` к Daemon-процессу.
- [x] Модификация инструмента синхронизации (AC: 1)
  - [x] Обновить обработчик `knowledge_sync_repo` в `mcp/server.py`.
  - [x] Отправлять запрос `POST /api/sync/{repo_id}` к Daemon-процессу (чтобы задача встала в БД-очередь).
- [x] Валидация Read-Only статуса соединения (AC: 2)
  - [x] MCP-процесс уже инициализирует соединение с SQLite как Read-Only (`KnowledgeDB(db_path, read_only=True)`).

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
Antigravity (Claude Opus 4.6)

### Debug Log References
None

### Completion Notes List
- Обновлён `_handle_delete_repo` — теперь делегирует `DELETE /api/repo/{repo_id}` через HTTP.
- Обновлён `_handle_sync_repo` — делегирует `POST /api/sync/{repo_id}` через HTTP.
- Убран `repo_path` из схемы `knowledge_sync_repo` (путь берётся из config.yaml в демоне).
- Подтверждено: `start_mcp_server` уже открывает БД через `KnowledgeDB(db_path, read_only=True)`.
- Все 19 тестов проекта пройдены.

### File List
- Modified: `knowledge_mcp/mcp/server.py`

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build/Linter passes (Ruff check clean).
- [x] Full test suite passes (19/19).

### Code Review Summary
- **Read-Only Enforcement:** MCP-сервер больше не выполняет никаких мутаций SQLite.
- **HTTP Delegation:** Все операции записи (delete, sync) делегируются Daemon API.
- **Schema Update:** `repo_path` убран из схемы sync — корректно, т.к. демон берёт путь из config.yaml.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | Мутации делегируются Daemon API | delete_repo → DELETE /api/repo/{id}, sync_repo → POST /api/sync/{id} | ✅ PASS |
| AC 2 | Поисковые инструменты Read-Only | `KnowledgeDB(db_path, read_only=True)` уже используется | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-01 | 0.1.0 | Dev: Started Story 1.6 implementation | James |
| 2026-07-01 | 1.0.0 | Dev: Delegated delete_repo and sync_repo to Daemon API | James |
| 2026-07-01 | 1.0.1 | CR: Verified Read-Only enforcement and HTTP delegation. PASS | CodeReview |
| 2026-07-01 | 1.0.2 | QA: All 19 tests pass. Status → Done | Quinn |
