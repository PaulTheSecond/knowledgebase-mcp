# Story 1.5: Webhook Validation and Cron Triggers

Status: ready-for-dev

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

- [ ] Реализация эндпоинта вебхуков (AC: 1)
  - [ ] Создать файл `knowledge_mcp/daemon/routes/webhooks.py`.
  - [ ] Реализовать эндпоинт `POST /webhook/git`.
  - [ ] Добавить парсинг тела запроса для извлечения `repo_id`.
- [ ] HMAC-SHA256 валидация подписей (AC: 1)
  - [ ] Реализовать проверку заголовка GitHub `X-Hub-Signature-256` и Gitea `X-Gitea-Signature` по алгоритму HMAC-SHA256 с использованием секрета из `secret_env`.
  - [ ] Реализовать проверку GitLab токена через заголовок `X-Gitlab-Token`.
  - [ ] Возвращать `401 Unauthorized` при несовпадении или отсутствии заголовков.
- [ ] Реализация планировщика Cron (AC: 2)
  - [ ] Использовать библиотеку планирования задач (например, `apscheduler` или легковесный `aiocron`).
  - [ ] Считывать поле `triggers.cron` для каждого репозитория из `config.yaml`.
  - [ ] Ставить задачу в `sync_queue` при срабатывании триггера.

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

### Debug Log References

### Completion Notes List

### File List
