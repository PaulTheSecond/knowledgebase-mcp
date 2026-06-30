# Story 1.1: Reorganization, DB Migration and WAL Setup

Status: done

## Story

As a разработчик,
I want разделить исходный плоский код на подмодули core, db, mcp и настроить режим WAL для SQLite,
so that проект был готов к добавлению веб-демона и отсутствовали конфликты параллельного чтения/записи.

## Acceptance Criteria

1. **Given** существующий плоский репозиторий,
   **When** я переношу файлы в подкаталоги `core/`, `db/`, `mcp/` и запускаю тесты,
   **Then** все импорты успешно разрешаются и MCP-сервер работает штатно.
2. **And** при каждом подключении к БД SQLite через `db.py` принудительно выполняются команды `PRAGMA journal_mode = WAL;` и `PRAGMA synchronous = NORMAL;`.
3. **And** в БД создается пустая таблица `system_status` для хранения состояния воркера.

## Tasks / Subtasks

- [x] Реорганизация структуры каталогов (AC: 1)
  - [x] Создать директории `core/`, `db/`, `mcp/` внутри `knowledge_mcp/`.
  - [x] Перенести `code_parser.py`, `markdown_parser.py`, `embeddings.py`, `indexer.py` в `core/`.
  - [x] Перенести `server.py` в `mcp/server.py`.
  - [x] Обновить файлы `__init__.py` для всех пакетов.
- [x] Настройка SQLite WAL-режима (AC: 2)
  - [x] Перенести `db.py` в `db/models.py`.
  - [x] Выделить инициализацию подключения в `db/connection.py`.
  - [x] Добавить выполнение PRAGMA команд `journal_mode=WAL` и `synchronous=NORMAL` при создании каждого коннекта.
- [x] Создание таблицы system_status (AC: 3)
  - [x] Добавить в `db/models.py` SQL-запрос `CREATE TABLE IF NOT EXISTS system_status (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP);`.
  - [x] Реализовать метод автоматического прогона миграций при старте.
- [x] Обновление импортов и main.py (AC: 1)
  - [x] Обновить пути импортов во всех перенесенных Python файлах.
  - [x] Обновить `knowledge_mcp/main.py`, адаптировав его под новые пути импорта.
  - [x] Проверить работоспособность CLI командой запуска MCP-сервера.

## Dev Notes

### Реорганизация структуры
Для корректной работы импортов все новые поддиректории должны содержать пустой файл `__init__.py`. В корневом `main.py` импорты изменятся с:
```python
from .db import Database
# на:
from .db.models import Database
```

### Настройка SQLite WAL
WAL (Write-Ahead Logging) критически важен, так как процесс демона и процесс MCP-клиента будут работать с одним и тем же файлом БД. 
Пример реализации `get_db_connection` in `db/connection.py`:
```python
import sqlite3
import sqlite_vec

def get_db_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn
```

### References

- [Архитектурный инвариант по WAL и Read-Only: ARCHITECTURE-SPINE.md#AD-1](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md#L29-L37)
- [Схема СУБД: solution_design.md#2.2.-Таблица-system_status](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/solution_design.md#L133-L145)

## Dev Agent Record

### Agent Model Used
Antigravity (Google DeepMind)

### Debug Log References
None

### Completion Notes List
- Выполнена реорганизация структуры пакета: файлы перенесенны по подкаталогам core, db, mcp.
- Реализован WAL-режим и Read-Only подключение в db/connection.py.
- Добавлена таблица system_status и автоматический запуск миграций.
- Все импорты в репозитории обновлены и протестированы.

### File List
- Created: `knowledge_mcp/core/__init__.py`
- Created: `knowledge_mcp/db/__init__.py`
- Created: `knowledge_mcp/mcp/__init__.py`
- Created: `knowledge_mcp/db/connection.py`
- Moved/Modified: `knowledge_mcp/core/code_parser.py` (from `code_parser.py`)
- Moved/Modified: `knowledge_mcp/core/markdown_parser.py` (from `markdown_parser.py`)
- Moved/Modified: `knowledge_mcp/core/embeddings.py` (from `embeddings.py`)
- Moved/Modified: `knowledge_mcp/core/indexer.py` (from `indexer.py`)
- Moved/Modified: `knowledge_mcp/db/models.py` (from `db.py`)
- Moved/Modified: `knowledge_mcp/mcp/server.py` (from `server.py`)
- Modified: `knowledge_mcp/main.py`
- Modified root test files (`_test_*.py` and `scratch/*.py`) to correct imports.

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build passes (Ruff check clean).
- [x] Git commit created.

### Code Review Summary
- **Style/PEP 8:** Clean, line limit respected.
- **SQLite Concurrency:** Connection uses WAL mode and synchronous=NORMAL.
- **SQL Injection Safety:** Parameters passed via placeholders (`?`).
- **Read-Only Mode:** Implemented correctly for stdio MCP server.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Methodology
- Automated integration testing via `_test_mcp_client.py` and `_test_delete_repo.py`.
- Manual verification of SQLite WAL and schema initialization.

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | Folder reorganization & import resolution | Ran MCP server handshake and search tool | ✅ PASS |
| AC 2 | SQLite WAL mode & connection settings | Checked WAL configuration & read-only connections | ✅ PASS |
| AC 3 | `system_status` migration table | Verified table creation & seed values in DB | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-06-30 | 1.0.0 | Dev: Reorganized folders, setup WAL & system_status, fixed imports | James |
| 2026-06-30 | 1.0.1 | CR: Verified imports, Python standards compliance, build/test PASS | CodeReview |
| 2026-06-30 | 1.0.2 | QA: Verified AC 1, 2, 3 with client and cascade tests. Status -> Done | Quinn |

