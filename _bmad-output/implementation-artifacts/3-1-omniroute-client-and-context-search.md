# Story 3.1: OmniRoute Client and Context Search

Status: ready-for-dev

## Story

As a разработчик,
I want чтобы бэкенд умел искать релевантные куски кода и отправлять их в LLM-модель через OmniRoute,
so that модель генерировала ответы на основе актуального контекста моей кодовой базы.

## Acceptance Criteria

1. **Given** настроенное подключение к OmniRoute на `http://localhost:20128/v1` с API-ключом `LLM_API_KEY`,
   **When** на эндпоинт `POST /api/chat` приходит запрос пользователя, содержащий текущий вопрос и историю диалога (массив `messages`),
   **Then** бэкенд считывает фича-флаг `server.dashboard.chat.enabled` из `config.yaml`. Если он равен `false`, бэкенд возвращает `403 Forbidden`.
2. **And** при включенном флаге бэкенд выполняет гибридный поиск `search_chunks_hybrid` по SQLite, отбирает топ-5 наиболее релевантных чанков кода и встраивает их в системный промпт (с указанием путей к файлам).
3. **And** бэкенд отправляет промпт вместе с переданной историей диалога в OmniRoute для модели `kr/claude-sonnet-4.5` с провайдером `kiro` и возвращает ответ LLM.

## Tasks / Subtasks

- [ ] Чтение конфигурации и фича-флага чата (AC: 1)
  - [ ] Реализовать проверку параметра `server.dashboard.chat.enabled` в `config.yaml`.
  - [ ] Зарегистрировать HTTP роут `POST /api/chat` в `knowledge_mcp/daemon/routes/chat.py`. Возвращать `403 Forbidden`, если флаг равен `false`.
- [ ] Контекстный гибридный поиск (AC: 2)
  - [ ] Встроить вызов функции гибридного поиска (RRF) из `db/models.py` (или `indexer.py`).
  - [ ] Запросить топ-5 чанков кода по тексту вопроса.
  - [ ] Собрать системный промпт, объединяющий найденный код с метаданными (путь, строки):
    ```
    Используй следующий контекст исходного кода для ответа на вопрос.
    [FILEPATH: path/to/file.cs]
    [LINES: 10-45]
    ```
- [ ] OpenAI-совместимый клиент для OmniRoute (AC: 3)
  - [ ] Использовать стандартную библиотеку `openai` в Python или сделать прямые запросы через `httpx`.
  - [ ] Настроить параметры: `base_url="http://localhost:20128/v1"`, `api_key=LLM_API_KEY`.
  - [ ] Сформировать payload: `model="kr/claude-sonnet-4.5"`, `messages=[{"role": "system", "content": system_prompt}] + user_messages`.
  - [ ] Выполнить запрос и вернуть ответ.

## Dev Notes

OmniRoute является локальным прокси-сервером, совместимым с API OpenAI. Для отправки запросов используем стандартный класс `AsyncOpenAI` из библиотеки `openai`:
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:20128/v1",
    api_key=os.environ.get("LLM_API_KEY", "sk-dummy")
)

response = await client.chat.completions.create(
    model="kr/claude-sonnet-4.5",
    messages=formatted_messages
)
```
Обязательно обработать ошибку отсутствия ключа `LLM_API_KEY` в окружении, возвращая понятное сообщение `500 Internal Server Error`.

### References

- [Совместимость с OmniRoute: prd.md#4.4.-Интегрированный-RAG-чат](file:///_bmad-output/planning-artifacts/prds/prd-knowledgebase-mcp-2026-06-30/prd.md#L118-L131)
- [Статья на Хабре по OmniRoute: https://habr.com/ru/articles/1016426/](https://habr.com/ru/articles/1016426/)

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
