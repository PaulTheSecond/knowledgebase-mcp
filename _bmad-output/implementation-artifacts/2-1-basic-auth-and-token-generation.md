# Story 2.1: Basic Auth and Token Generation

Status: ready-for-dev

## Story

As a администратор,
I want защитить доступ к панели паролем Basic Auth и генерировать безопасные временные токены для вебсокетов,
so that посторонние лица не могли просматривать структуру моих репозиториев и логи.

## Acceptance Criteria

1. **Given** FastAPI-приложение,
   **When** я обращаюсь к REST-эндпоинтам (например, `/api/status`) без заголовка `Authorization` или с неверным паролем,
   **Then** бэкенд возвращает `401 Unauthorized`.
2. **And** если переменная окружения `DASHBOARD_PASSWORD` не задана или пуста, бэкенд принудительно возвращает `403 Forbidden` для любого запроса к панели.
3. **And** при успешном Basic Auth эндпоинт `/api/auth/token` возвращает короткоживущий (60 сек) криптографически подписанный токен для установки соединений WebSocket/SSE.

## Tasks / Subtasks

- [ ] Реализация модуля авторизации (AC: 1, 2)
  - [ ] Создать файл `knowledge_mcp/daemon/auth.py`.
  - [ ] Написать FastAPI Dependency для проверки HTTP Basic Credentials.
  - [ ] Если пароль пустой или отсутствует в окружении (`DASHBOARD_PASSWORD`), выкидывать `HTTPException(status_code=403, detail="Dashboard disabled")`.
- [ ] Генерация подписанных токенов для WS/SSE (AC: 3)
  - [ ] Использовать библиотеку `itsdangerous` (или стандартный `hmac`/`hashlib` с солью) для создания подписи.
  - [ ] Метод `/api/auth/token` должен возвращать токен вида `uuid.signature.timestamp`.
- [ ] Валидация токенов (AC: 3)
  - [ ] Реализовать метод верификации токена. Если время действия (>60 секунд) истекло или подпись невалидна, выбрасывать ошибку авторизации.
- [ ] Тестирование авторизации (AC: 1, 2)
  - [ ] Написать юнит-тесты на проверку пустых паролей и валидацию токенов.

## Dev Notes

Пример использования `itsdangerous` для подписи токенов:
```python
from itsdangerous import SignatureExpired, BadSignature
from itsdangerous.url_safe import URLSafeTimedSerializer

SECRET_KEY = "your-system-secret" # Использовать случайный ключ, генерируемый при старте daemon

def generate_signed_token(username: str) -> str:
    serializer = URLSafeTimedSerializer(SECRET_KEY)
    return serializer.dumps({"user": username})

def verify_signed_token(token: str) -> bool:
    serializer = URLSafeTimedSerializer(SECRET_KEY)
    try:
        data = serializer.loads(token, max_age=60) # Токен валиден 60 секунд
        return True
    except (SignatureExpired, BadSignature):
        return False
```

### References

- [Архитектурный инвариант по токенам: ARCHITECTURE-SPINE.md#AD-3](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md#L44-L50)

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
