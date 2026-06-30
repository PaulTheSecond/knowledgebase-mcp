# Story 2.3: REST API Telemetry and Queue View

Status: ready-for-dev

## Story

As a разработчик,
I want видеть актуальный список репозиториев с их метриками и текущее состояние очереди в таблице,
so that я мог быстро оценить объем проиндексированных данных.

## Acceptance Criteria

1. **Given** эндпоинт `GET /api/status`, возвращающий список репозиториев с их количеством файлов, чанков, векторов и статусом воркера,
   **When** я перехожу на вкладку "Repositories",
   **Then** фронтенд отображает красивую сетку карточек репозиториев с метриками.
2. **And** при переходе на вкладку "Sync Queue" фронтенд запрашивает `GET /api/queue` и выводит Read-Only список задач (состояние, количество попыток, дата создания, ошибки для упавших задач).

## Tasks / Subtasks

- [ ] Реализация REST-эндпоинтов на бэкенде (AC: 1, 2)
  - [ ] Создать файл `knowledge_mcp/daemon/routes/status.py`.
  - [ ] Зарегистрировать эндпоинт `GET /api/status`. Он должен считывать репозитории из `config.yaml` и делать `SELECT COUNT(*)` по таблицам `files`, `chunks` в SQLite для подсчета метрик.
  - [ ] Зарегистрировать эндпоинт `GET /api/queue`. Он делает `SELECT * FROM sync_queue ORDER BY created_at DESC LIMIT 50;`.
- [ ] Отрисовка репозиториев на клиенте (AC: 1)
  - [ ] Написать JS функцию `loadRepositories()` в `dashboard/static/js/app.js`.
  - [ ] Запросить `/api/status` с передачей заголовков Basic Auth.
  - [ ] Отрендерить карточки репозиториев: имя, ветка, количество файлов, векторов, статус.
- [ ] Отрисовка очереди на клиенте (AC: 2)
  - [ ] Написать JS функцию `loadQueue()` в `dashboard/static/js/app.js`.
  - [ ] Отрендерить таблицу очереди: ID, Repo, Status, Attempts, Error (сворачиваемый лог ошибки), Created At.

## Dev Notes

Для подсчета общего числа чанков и векторов по репозиторию необходимо использовать `repo_id` (или хэш пути) в запросах агрегации:
```sql
SELECT COUNT(*) FROM chunks WHERE file_id IN (SELECT id FROM files WHERE repo_id = ?);
```
Фронтенд должен безопасно экранировать текст ошибок из очереди с помощью `textContent` (или аналогичных методов JS) для защиты от XSS.

### References

- [Спецификация REST-эндпоинтов: spec_autonomous_sync.md#3.-Требования-к-API-Web-UI-(REST-эндпоинты)](file:///c:/Repos/knowledgebase-mcp/docs/spec_autonomous_sync.md#L96-L107)

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
