# Story 1.3: FIFO DB Queue & Worker Loop

Status: ready-for-dev

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

- [ ] Создание миграции БД для таблиц очереди и коммитов (AC: 1)
  - [ ] Добавить создание таблицы `sync_queue` в `db/models.py`.
  - [ ] Создать таблицу `indexed_commits (repo_id TEXT PRIMARY KEY, git_sha TEXT, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP);`.
- [ ] Реализация циклического воркера (AC: 1)
  - [ ] Создать файл `knowledge_mcp/daemon/worker.py`.
  - [ ] Реализовать цикл `while True` с интервалом опроса очереди в 1 секунду.
  - [ ] Реализовать выборку задач по FIFO: `SELECT * FROM sync_queue WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1;`.
  - [ ] Реализовать перевод задачи в статус `processing` в рамках транзакции.
- [ ] Интеграция с Git и индексатором (AC: 1, 2)
  - [ ] Для выбранной задачи получить конфиг репозитория из `config.yaml`.
  - [ ] Выполнить обновление кода через `core/git_manager.py` в `/data/clones/{repo_id}`.
  - [ ] Получить текущий коммит HEAD через команду `git rev-parse HEAD`.
- [ ] Реализация идемпотентности по Git SHA (AC: 2)
  - [ ] Проверить по таблице `indexed_commits` последний обработанный коммит для этого `repo_id`.
  - [ ] Если коммиты совпадают, сразу перевести статус задачи в `completed` и перейти к следующей задаче.
  - [ ] Если коммит изменился, запустить индексатор `indexer.py` (парсинг кода и эмбеддинги).
  - [ ] После успешного завершения записать новый `git_sha` в `indexed_commits` и перевести задачу в `completed`.

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

### Debug Log References

### Completion Notes List

### File List
