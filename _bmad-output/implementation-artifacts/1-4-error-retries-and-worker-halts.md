# Story 1.4: Error Retries and Worker Halts

Status: done

## Story

As a администратор,
I want чтобы воркер автоматически перезапускал упавшие задачи с задержкой и останавливал очередь при критических сбоях,
so that система не перегружалась бесконечными попытками при падении сети, а я мог увидеть ошибку.

## Acceptance Criteria

1. **Given** задачу в очереди, упавшую с ошибкой (например, сбой сети при `git pull`),
   **When** воркер обрабатывает её повторно,
   **Then** он делает до 3 попыток перезапуска с экспоненциальной задержкой (1 минута, 5 минут, 15 минут).
2. **And** при превышении 3 попыток задача переводится в статус `failed` с сохранением лога ошибки в `error_message`.
3. **And** воркер записывает статус `worker_state = halted` и текст ошибки в таблицу `system_status` и полностью останавливает обработку последующих задач.

## Tasks / Subtasks

- [x] Реализация подсчета попыток и экспоненциального бэк-оффа (AC: 1)
  - [x] Обновить логику обработки ошибок воркера в `knowledge_mcp/daemon/worker.py`.
  - [x] Если задача завершается исключением, инкрементировать `attempts`.
  - [x] Если `attempts < 3`, сбросить статус в `pending` и сохранить планируемое время перезапуска (или рассчитать задержку на основе `attempts`).
- [x] Остановка воркера при превышении лимита (AC: 2, 3)
  - [x] Если `attempts >= 3`, перевести задачу в `failed` и записать исключение в `error_message`.
  - [x] Записать в SQLite таблицу `system_status` ключ `worker_state` со значением `halted` и ключ `last_error` с описанием ошибки.
  - [x] Воркер должен прекратить выборку новых задач при обнаружении статуса `halted` в `system_status`.
- [x] Юнит-тесты на обработку ошибок воркера (AC: 1, 2)
  - [x] Написать тесты для проверки инкремента попыток и смены статуса на `halted`.

## Dev Notes

Воркер должен проверять `system_status` перед каждым циклом опроса. Если статус равен `halted`, воркер не должен забирать новые задачи из `sync_queue`.
Расчет задержки:
- Попытка 1: `delay = 60` сек.
- Попытка 2: `delay = 300` сек.
- Попытка 3: `delay = 900` сек.
Для отслеживания времени следующего запуска можно добавить колонку `run_at DATETIME` в таблицу `sync_queue` и выбирать задачи по условию `status = 'pending' AND (run_at IS NULL OR run_at <= CURRENT_TIMESTAMP)`.

### References

- [Спецификация бэк-оффа: solution_design.md#3.2.-Алгоритм-обработки-очереди-задач-и-экспоненциального-бэк-оффа](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/solution_design.md#L165-L182)

## Dev Agent Record

### Agent Model Used
Antigravity (Claude Opus 4.6)

### Debug Log References
None

### Completion Notes List
- Добавлена миграция `20260701_08_sync_queue_run_at` для колонки `run_at` в таблице `sync_queue`.
- Обновлён `fetch_next_job()` — теперь пропускает задачи с `run_at` в будущем.
- Добавлен метод `reschedule_job()` в `KnowledgeDB` для возврата задачи в `pending` с задержкой.
- Обновлён `run_job()` в `DaemonWorker` — экспоненциальный бэк-офф (60/300/900 сек) и halted при 3 попытках.
- Написаны 2 новых теста: `test_retry_backoff_reschedules_with_delay` и `test_halted_after_max_attempts`.

### File List
- Modified: `knowledge_mcp/db/models.py`
- Modified: `knowledge_mcp/daemon/worker.py`
- Modified: `tests/test_worker.py`

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build/Linter passes (Ruff check clean).
- [x] Unit tests pass successfully (3/3).

### Code Review Summary
- **Retry Logic:** Задержки `{1: 60, 2: 300, 3: 900}` точно соответствуют спецификации.
- **Halted State:** При `attempts >= max_attempts` воркер корректно ставит `halted` в `system_status` и `failed` в задаче.
- **run_at Filter:** `fetch_next_job` правильно фильтрует задачи по `run_at IS NULL OR run_at <= datetime('now', 'localtime')`.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Methodology
- Автоматизированные тесты в `tests/test_worker.py`.
- Ручная проверка SQL миграции и логики бэк-оффа.

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | Экспоненциальный бэк-офф (60/300/900 сек) | test_retry_backoff: задача возвращена в pending с run_at | ✅ PASS |
| AC 2 | Задача → failed после 3 попыток | test_halted: status=failed, attempts=3, error_message заполнен | ✅ PASS |
| AC 3 | Воркер → halted при исчерпании попыток | test_halted: system_status=halted, last_error заполнен | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-01 | 0.1.0 | Dev: Started Story 1.4 implementation | James |
| 2026-07-01 | 1.0.0 | Dev: Implemented retry backoff, halted state, migration and tests | James |
| 2026-07-01 | 1.0.1 | CR: Reviewed retry logic and run_at filter. PASS | CodeReview |
| 2026-07-01 | 1.0.2 | QA: Verified all 3 AC. Status → Done | Quinn |
