# Story 1.3: FIFO DB Queue & Worker Loop

Status: done

## Story

As a разработчик,
I want чтобы задачи на синхронизацию хранились в надежной БД-очереди и выполнялись последовательно воркером,
so that изменения в кодовой базе отражались в базе знаний без сбоев из-за перезагрузок.

## Acceptance Criteria

1. **Given** базу данных SQLite с таблицей `sync_queue`,
   **When** в очередь добавляется несколько задач на синхронизацию,
   **Then** фоновый воркер выбирает старейшую задачу в статусе `pending` (по возрастанию `created_at`), переводит её в `processing` и запускает обновление кода.
2. **And** перед запуском индексации воркер считывает текущий Git SHA коммита. Если он совпадает с ранее проиндексированным, задача помечается как `completed` за миллисекунды (идемпотентность).

## Tasks / Subtasks

- [x] Создание миграции БД для таблиц очереди и коммитов (AC: 1)
  - [x] Добавить создание таблицы `sync_queue` в `db/models.py`.
  - [x] Создать таблицу `indexed_commits (repo_id TEXT PRIMARY KEY, git_sha TEXT, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP);`.
- [x] Реализация циклического воркера (AC: 1)
  - [x] Создать файл `knowledge_mcp/daemon/worker.py`.
  - [x] Реализовать цикл `while True` с интервалом опроса очереди в 1 секунду.
  - [x] Реализовать выборку задач по FIFO: `SELECT * FROM sync_queue WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1;`.
  - [x] Реализовать перевод задачи в статус `processing` в рамках транзакции.
- [x] Интеграция с Git и индексатором (AC: 1, 2)
  - [x] Для выбранной задачи получить конфиг репозитория из `config.yaml`.
  - [x] Выполнить обновление кода через `core/git_manager.py` в `/data/clones/{repo_id}`.
  - [x] Получить текущий коммит HEAD через команду `git rev-parse HEAD`.
- [x] Реализация идемпотентности по Git SHA (AC: 2)
  - [x] Проверить по таблице `indexed_commits` последний обработанный коммит для этого `repo_id`.
  - [x] Если коммиты совпадают, сразу перевести статус задачи в `completed` и перейти к следующей задаче.
  - [x] Если коммит изменился, запустить индексатор `indexer.py` (парсинг кода и эмбеддинги).
  - [x] После успешного завершения записать новый `git_sha` в `indexed_commits` и перевести задачу в `completed`.

## Dev Notes

### Выборка FIFO и транзакции в SQLite
Для исключения race conditions при выборке задач воркером (особенно при переходе к многопоточности в будущем) обновление статуса должно выполняться сразу после выборки:
```python
def fetch_next_job(conn) -> dict:
    cursor = conn.cursor()
    cursor.execute("BEGIN IMMEDIATE TRANSACTION;")
    try:
        cursor.execute("SELECT id, repo_id, status FROM sync_queue WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1;")
        row = cursor.fetchone()
        if row:
            job_id, repo_id, status = row
            cursor.execute("UPDATE sync_queue SET status = 'processing', updated_at = CURRENT_TIMESTAMP WHERE id = ?;", (job_id,))
            conn.commit()
            return {"id": job_id, "repo_id": repo_id}
        conn.commit()
        return None
    except Exception as e:
        conn.rollback()
        raise e
```

### References

- [Архитектурный инвариант очереди: ARCHITECTURE-SPINE.md#AD-2](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md#L39-L43)
- [Алгоритм идемпотентности по SHA коммита: solution_design.md#3.2.-Алгоритм-обработки-очереди-задач-и-экспоненциального-бэк-оффа](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/solution_design.md#L165-L182)

## Dev Agent Record

### Agent Model Used
Antigravity (Google DeepMind)

### Debug Log References
None

### Completion Notes List
- Создана миграция БД `20260701_07_sync_queue_and_indexed_commits` для таблиц `sync_queue` и `indexed_commits` с индексом по статусу и времени.
- Реализован класс `DaemonWorker` в `knowledge_mcp/daemon/worker.py` с циклом опроса очереди (FIFO) и проверкой состояния воркера (`halted`).
- Реализован атомарный выбор и обновление статуса задачи в `fetch_next_job()` с использованием транзакции `BEGIN IMMEDIATE`.
- Интегрирована проверка Git SHA и контроль идемпотентности: если коммит не изменился, индексация пропускается, а задача сразу завершается.
- Созданы тесты `tests/test_worker.py` с моками `GitManager` и `Indexer`, полностью покрывающие работу воркера и идемпотентность.

### File List
- Created: `knowledge_mcp/daemon/worker.py`
- Created: `knowledge_mcp/daemon/__init__.py`
- Created: `knowledge_mcp/core/config.py`
- Created: `tests/test_worker.py`
- Modified: `knowledge_mcp/db/models.py`

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build/Linter passes (Ruff check clean).
- [x] Unit tests pass successfully.

### Code Review Summary
- **Database Concurrency:** Atomicity of task selection is guaranteed by `BEGIN IMMEDIATE TRANSACTION` in SQLite WAL mode.
- **Idempotency:** SHA checkout prevents repeating slow vectorization steps.
- **Style/Standards:** Correct exception handling, logger redirected to stderr, types and clean imports.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Methodology
- Automated integration test run in `tests/test_worker.py`.
- Checked SQLite tables and schema migration for `sync_queue` and `indexed_commits`.

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | FIFO DB Queue & Worker status translation | Added queue entries, verified FIFO sorting and status updates | ✅ PASS |
| AC 2 | SHA comparison and Idempotency | Confirmed indexing is skipped and status set to completed if SHA is unchanged | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-01 | 0.1.0 | Dev: Started Story 1.3 implementation | James |
| 2026-07-01 | 1.0.0 | Dev: Implemented DaemonWorker, migration and tests | James |
| 2026-07-01 | 1.0.1 | CR: Reviewed SQLite transactions and idempotency logic. PASS | CodeReview |
| 2026-07-01 | 1.0.2 | QA: Verified FIFO and idempotency behavior. Status -> Done | Quinn |
