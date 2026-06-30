# Story 2.4: WebSockets Real-Time Monitoring

Status: ready-for-dev

## Story

As a разработчик,
I want видеть прогресс-бары синхронизации репозиториев и изменение статусов задач на дашборде в реальном времени,
so that мне не приходилось постоянно вручную обновлять страницу браузера.

## Acceptance Criteria

1. **Given** WebSocket-эндпоинт `/ws/status` на бэкенде,
   **When** фронтенд открывает соединение, передавая временный токен `/ws/status?token=<TOKEN>`,
   **Then** бэкенд успешно валидирует токен и устанавливает сессию (при неверном токене сокет закрывается с кодом `4001`).
2. **And** при изменении состояния воркера или прогресса индексации бэкенд шлет JSON-события в сокет, а фронтенд реактивно перерисовывает статус карточки репозитория и показывает прогресс-бар (0-100%) без перезагрузки страницы.

## Tasks / Subtasks

- [ ] Реализация WebSocket эндпоинта (AC: 1)
  - [ ] Создать WebSocket маршрут `/ws/status` в `knowledge_mcp/daemon/routes/status.py`.
  - [ ] Считать токен из параметров запроса: `token = websocket.query_params.get("token")`.
  - [ ] Проверить подпись токена через `auth.py`. Если невалиден, закрыть сокет с кодом `4001` (Unauthorized WS).
- [ ] Механизм рассылки событий (Broadcasting) (AC: 2)
  - [ ] Создать менеджер подключений `ConnectionManager` для рассылки сообщений всем активным сокетам.
  - [ ] Интегрировать отправку событий прогресса индексации из `indexer.py` / `queue_worker.py` в менеджер сокетов.
- [ ] WebSocket-клиент на фронтенде (AC: 2)
  - [ ] Реализовать в `dashboard/static/js/app.js` функцию `connectWebSocket()`.
  - [ ] Перед подключением получить токен через `POST /api/auth/token`.
  - [ ] Открыть соединение `new WebSocket("ws://" + location.host + "/ws/status?token=" + token)`.
  - [ ] При получении сообщения обновлять DOM-элементы: переключать классы статусов карточек, анимировать ширину прогресс-баров (`transition: width 0.2s ease-in-out`).

## Dev Notes

Воркер должен сообщать процент выполнения индексации. Для этого в `indexer.py` можно добавить callback-функцию прогресса (например, на основе соотношения `processed_files / total_files`), которая шлет сообщения в WebSocket менеджер.
Сообщения в вебсокет должны иметь стандартизированную структуру:
```json
{
  "type": "sync_progress",
  "data": {
    "repo_id": "Yurta.Core.Lib",
    "progress": 45,
    "status": "syncing"
  }
}
```

### References

- [Архитектурный инвариант по WebSockets: ARCHITECTURE-SPINE.md#AD-3](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md#L44-L50)
- [Дизайн-инвариант real-time: ADR-005-web-dashboard-architecture.md](file:///c:/Repos/knowledgebase-mcp/docs/adr-005-web-dashboard-architecture.md)

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
