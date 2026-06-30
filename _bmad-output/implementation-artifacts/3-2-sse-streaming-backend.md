# Story 3.2: SSE Streaming Backend

Status: ready-for-dev

## Story

As a разработчик,
I want получать ответ от модели по мере генерации токенов (стриминг),
so that мне не приходилось ждать 10–15 секунд для вывода всего текста целиком.

## Acceptance Criteria

1. **Given** эндпоинт `POST /api/chat/stream`, авторизованный по временному токену в query-параметрах,
   **When** бэкенд получает ответ от OmniRoute в потоковом режиме (`stream=true`),
   **Then** он ретранслирует этот поток клиенту в формате `text/event-stream` (Server-Sent Events) посимвольно/потокенно.
2. **And** в конце потока отправляется специальный JSON-пакет, содержащий метаданные с перечнем файлов-источников (путь и номера строк), использованных в контексте.

## Tasks / Subtasks

- [ ] Реализация эндпоинта Server-Sent Events (AC: 1)
  - [ ] Создать эндпоинт `POST /api/chat/stream` в `routes/chat.py`.
  - [ ] Авторизовать запрос по токену в query string: `/api/chat/stream?token=<TOKEN>`.
  - [ ] Использовать класс `EventSourceResponse` из библиотеки `sse_starlette` (или напрямую возвращать `StreamingResponse` из FastAPI).
- [ ] Ретрансляция потока от OmniRoute (AC: 1)
  - [ ] Вызвать OpenAI клиент с флагом `stream=True`.
  - [ ] Написать асинхронный генератор `token_generator()`, который считывает куски (`chunks`) из OmniRoute и отдает их в формате:
    `data: {"token": "..."}\n\n`
- [ ] Передача метаданных источников в потоке (AC: 2)
  - [ ] Перед закрытием потока сгенерировать финальное событие `sources`, содержащее сериализованный JSON-список файлов-источников:
    `data: {"sources": [{"filepath": "db.py", "lines": "10-20"}]}\n\n`
  - [ ] Закрыть соединение.

## Dev Notes

Пример асинхронного генератора для ретрансляции SSE в FastAPI:
```python
from fastapi.responses import StreamingResponse

async def event_generator(question: str, messages: list):
    # 1. Поиск контекста и сборка системного промпта
    context = search_chunks_hybrid(question)
    system_prompt = build_system_prompt(context)
    
    # 2. Запрос в OmniRoute
    response = await client.chat.completions.create(
        model="kr/claude-sonnet-4.5",
        messages=[{"role": "system", "content": system_prompt}] + messages,
        stream=True
    )
    
    # 3. Стриминг токенов
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield f"data: {json.dumps({'token': content})}\n\n"
            
    # 4. Стриминг метаданных источников в конце
    sources_data = [{"filepath": c.filepath, "lines": c.lines} for c in context]
    yield f"data: {json.dumps({'sources': sources_data})}\n\n"
```

### References

- [Архитектурный инвариант по транспорту SSE: ARCHITECTURE-SPINE.md#AD-3](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md#L44-L50)

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
