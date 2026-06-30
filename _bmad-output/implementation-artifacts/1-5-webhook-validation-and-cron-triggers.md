# Story 1.5: Webhook Validation and Cron Triggers

Status: done

## Story

As a разработчик,
I want чтобы задачи в очередь добавлялись автоматически по вебхукам или расписанию Cron,
so that мне не приходилось запускать синхронизацию вручную.

## Acceptance Criteria

1. **Given** FastAPI-эндпоинт `/webhook/git`,
   **When** на него приходит POST запрос от GitHub, Gitea или GitLab с подписью,
   **Then** бэкенд валидирует HMAC-SHA256 подпись с секретом из переменных окружения. При совпадении подписи задача добавляется в `sync_queue`, при несовпадении возвращается 401 Unauthorized.
2. **And** фоновый планировщик (Cron) считывает расписание из `config.yaml` для каждого репозитория и автоматически ставит новые задачи в `sync_queue` при наступлении триггера.

## Tasks / Subtasks

- [x] Реализация эндпоинта вебхуков (AC: 1)
  - [x] Создать файл `knowledge_mcp/daemon/routes/webhooks.py`.
  - [x] Реализовать эндпоинт `POST /webhook/git`.
  - [x] Добавить парсинг тела запроса для извлечения `repo_id`.
- [x] HMAC-SHA256 валидация подписей (AC: 1)
  - [x] Реализовать проверку заголовка GitHub `X-Hub-Signature-256` и Gitea `X-Gitea-Signature` по алгоритму HMAC-SHA256 с использованием секрета из `secret_env`.
  - [x] Реализовать проверку GitLab токена через заголовок `X-Gitlab-Token`.
  - [x] Возвращать `401 Unauthorized` при несовпадении или отсутствии заголовков.
- [x] Реализация планировщика Cron (AC: 2)
  - [x] Использовать легковесный планировщик на `threading.Timer` (без внешних зависимостей).
  - [x] Считывать поле `triggers.cron` для каждого репозитория из `config.yaml`.
  - [x] Ставить задачу в `sync_queue` при срабатывании триггера.

## Dev Notes

Пример проверки подписи HMAC-SHA256 в Python:
```python
import hmac
import hashlib

def verify_signature(payload: bytes, secret: str, signature: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
```

Секрет извлекается по имени переменной, заданной в конфиге:
```yaml
secret_env: "YURTA_WEBHOOK_SECRET" # Имя переменной окружения
```

### References

- [Требования валидации: prd.md#4.2.-Обработка-вебхуков-и-расписания](file:///_bmad-output/planning-artifacts/prds/prd-knowledgebase-mcp-2026-06-30/prd.md#L98-L105)

## Dev Agent Record

### Agent Model Used
Antigravity (Claude Opus 4.6)

### Debug Log References
None

### Completion Notes List
- Создан модуль `daemon/routes/webhooks.py` с эндпоинтом `POST /webhook/git`.
- Реализована HMAC-SHA256 валидация для GitHub и Gitea, и проверка токена для GitLab.
- Автоматический поиск репозитория по clone URL из payload вебхука.
- Создан `daemon/scheduler.py` — легковесный cron-планировщик на `threading.Timer` (без внешних зависимостей).
- 12 тестов пройдено: HMAC-валидация, cron-парсинг, поиск конфига, планировщик.

### File List
- Created: `knowledge_mcp/daemon/routes/__init__.py`
- Created: `knowledge_mcp/daemon/routes/webhooks.py`
- Created: `knowledge_mcp/daemon/scheduler.py`
- Created: `tests/test_webhooks_and_cron.py`
- Modified: `data/config.yaml`

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build/Linter passes (Ruff check clean).
- [x] Unit tests pass successfully (12/12).

### Code Review Summary
- **HMAC:** Используется `hmac.compare_digest` для timing-safe сравнения.
- **Cron:** Легковесная реализация без внешних зависимостей, daemon threads.
- **Security:** Секреты извлекаются из env vars, не хардкодятся.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | Webhook HMAC-SHA256 валидация | Тесты HMAC valid/invalid/without-prefix + поиск репо по URL | ✅ PASS |
| AC 2 | Cron-планировщик читает config.yaml | Тесты парсинга cron + start/stop планировщика | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-01 | 0.1.0 | Dev: Started Story 1.5 implementation | James |
| 2026-07-01 | 1.0.0 | Dev: Implemented webhooks and cron scheduler | James |
| 2026-07-01 | 1.0.1 | CR: Reviewed HMAC and cron logic. PASS | CodeReview |
| 2026-07-01 | 1.0.2 | QA: Verified 12 tests pass. Status → Done | Quinn |
