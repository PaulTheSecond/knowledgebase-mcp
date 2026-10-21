---
name: Autonomous Sync & Dashboard Architecture
type: architecture-spine
purpose: build-substrate
altitude: feature
paradigm: process-isolated shared-db
scope: Governs the background sync worker, SQLite job queue, webhook validator, and dashboard Web UI with RAG chat.
status: final
created: 2026-06-30
updated: 2026-06-30
binds: ["v1.3", "v1.4", "v1.5", "v1.6", "FR-1", "FR-2", "FR-3", "FR-4", "FR-5", "FR-6", "FR-7", "FR-8", "FR-9", "FR-10", "FR-11", "FR-12", "FR-13", "FR-14", "FR-15", "FR-16"]
sources: ["_bmad-output/planning-artifacts/prds/prd-knowledgebase-mcp-2026-06-30/prd.md"]
companions: ["_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/solution_design.md"]
---

# Architecture Spine — Autonomous Sync & Dashboard Module

## Design Paradigm

Архитектурная парадигма: **Изолированные процессы с разделяемым состоянием в БД (Process-Isolated Shared-Database)**. 
Система разделена на два независимых процесса, использующих общий файл базы данных SQLite:
1. **Daemon-процесс (Web-сервер и фоновый воркер):** Постоянно запущен в контейнере, обслуживает веб-интерфейс, принимает вебхуки и выполняет фоновые задачи синхронизации в один поток через локальную очередь.
2. **MCP-процесс (STDIO-интерфейс):** Кратковременный процесс, запускаемый внешним клиентом (например, VS Code/Cursor) для мгновенного выполнения поисковых запросов.

---

## Invariants & Rules

### AD-1 — Режим доступа к базе данных SQLite
* **Binds:** `server.py`, `db.py`, `indexer.py`, `main.py`
* **Prevents:** Взаимные блокировки (`database is locked`) при одновременной записи воркером и чтении поисковым MCP-клиентом.
* **Rule:** 
  1. Все подключения к базе данных SQLite должны инициализироваться в режиме **WAL (Write-Ahead Logging)** через PRAGMA-директивы:
     ```sql
     PRAGMA journal_mode = WAL;
     PRAGMA synchronous = NORMAL;
     ```
     Это обеспечивает параллельное бесконфликтное чтение множеством потоков/процессов во время записи.
  2. **Изоляция записи:** Все мутирующие операции записи (запись чанков, символов, удаление репозиториев) разрешены только внутри Daemon-процесса. Процесс MCP должен работать строго в режиме **Read-Only** для SQLite. Команды мутации (такие как `knowledge_delete_repo`) должны делегироваться Daemon-процессу через вызовы его REST API по HTTP, а не выполняться процессом MCP напрямую.

### AD-2 — Управление очередью задач и жизненным циклом
* **Binds:** `FR-1`, `FR-2`, `FR-3`, `FR-4` (очередь и воркер)
* **Prevents:** Потерю задач на синхронизацию при сбоях питания, перегрузку CPU конкурентными сборками, неидемпотентность запусков.
* **Rule:** Очередь задач должна храниться в таблице SQLite `sync_queue`. Воркер забирает задачи строго по FIFO (`created_at` ASC). Статус воркера и ошибки сохраняются в таблице `system_status`. При возникновении ошибки и исчерпании 3 попыток перезапуска (с экспоненциальной задержкой 1/5/15 мин), воркер записывает статус `halted` в `system_status` и останавливает обработку. Перед выполнением задачи воркер сверяет Git SHA репозитория с записанным в базе данных; при совпадении задача завершается моментально (гарантия идемпотентности).

### AD-3 — Авторизация и транспорт реального времени
* **Binds:** `FR-9`, `FR-10`, `FR-11`, `FR-12`, `FR-15` (панель управления и RAG-чат)
* **Prevents:** Несанкционированный доступ к исходному коду и логам, задержки интерфейса при медленном ответе ИИ.
* **Rule:** 
  1. Панель управления использует HTTP Basic Auth для REST эндпоинтов. Подключения WebSocket (для телеметрии) и Server-Sent Events (для стриминга RAG-чата) авторизуются с помощью короткоживущих подписанных токенов, передаваемых в строке запроса `?token=...`.
  2. **Отключаемость чата:** Чат на бэкенде и в UI должен отключаться параметром `server.dashboard.chat.enabled` в `config.yaml`. При значении `false` эндпоинты чата отдают `403 Forbidden` для всех запросов.
  3. **Состояние диалога:** Бэкенд является строго stateless для чат-сессий. Вся история чата хранится на клиенте и передается в теле POST-запроса `/api/chat`. Telemetry передается по WebSocket, RAG-чат стримится по SSE.

### AD-4 — Безопасность токенов авторизации Git
* **Binds:** `FR-1`, `FR-5` (подключение к приватным репозиториям)
* **Prevents:** Компрометацию Personal Access Tokens (PAT) при несанкционированном доступе к дисковым томам контейнера.
* **Rule:** Токены Git (PAT) категорически запрещено записывать в файл `.git/config` клонированных репозиториев на диске. Вместо этого при вызове команд `git clone`, `git fetch` или `git pull` токен должен передаваться динамически в оперативной памяти через конфигурационный параметр Git `http.extraHeader="Authorization: Basic <base64>"` или переменную `GIT_ASKPASS`.

### AD-5 — Инспектор документов и форматирование чата (Doc Viewer & Chat Formatting)
* **Binds:** `FR-15`, `FR-16`, `ADR-007`
* **Prevents:** Ошибки 404 при цитировании локальных репозиториев, потерю контекста диалога при перезагрузке, блокировку действий Angular DomSanitizer при рендеринге разметки.
* **Rule:** 
  1. **Резолв путей файлов:** Эндпоинт `/api/chat/file` обязан резолвить файлы во всех настроенных директориях репозиториев (`local_path`, `clones_dir`, `project_dir`) с поддержкой флага `full_file`.
  2. **Инспектор документов и сплиттер:** Просмотр файлов выполняется в правой панели (`app-doc-viewer`) с IDE-вкладками и разделителем с начальной шириной 45% (сброс по dblclick, сохранение в localStorage).
  3. **Рендеринг и безопасность:** Форматирование ответов чата включает подсветку Prism и Mermaid SVG; интерактивные элементы (кнопки копирования) рендерятся через `sanitizer.bypassSecurityTrustHtml()`.

---

## Consistency Conventions

| Область | Соглашение |
| :--- | :--- |
| **Имена таблиц** | Названия таблиц СУБД пишутся в snake_case во множественном числе (например, `sync_queue`, `system_status`). |
| **Формат API** | Все ответы REST API должны следовать стандарту JSON-API. Ошибки возвращают тело `{"detail": "..."}`. |
| **Формат логов** | Все бэкенд-логи пишутся исключительно в `sys.stderr` с метаданными `[TIMESTAMP] [LEVEL] [COMPONENT] Message`. Вывод в `sys.stdout` разрешен только для протокола MCP. |
| **Версионирование токенов** | Подписанные токены авторизации Web UI генерируются алгоритмом Itsdangerous/HMAC со сроком действия 60 секунд. |

---

## Stack

| Название технологии | Версия |
| :--- | :--- |
| **Python** | >= 3.11 |
| **FastAPI** | >= 0.100.0 |
| **Uvicorn** | >= 0.22.0 |
| **SQLite** | >= 3.38.0 |
| **sqlite-vec** | >= 0.1.0 |
| **Git CLI** | >= 2.34.0 |

---

## Structural Seed

```text
{project-root}/
  _bmad-output/
    planning-artifacts/
      architecture/         # Архитектурные схемы и Solution Design
  knowledge_mcp/
    __init__.py
    main.py                 # Единая точка входа CLI (команды: mcp / serve)
    core/                   # Ядро (парсеры кода, разметка, векторный индекс)
      __init__.py
      code_parser.py
      markdown_parser.py
      embeddings.py
      indexer.py
    db/                     # Слой СУБД (транзакции, миграции, WAL)
      __init__.py
      connection.py
      models.py
    mcp/                    # Слой MCP stdio (Read-Only)
      __init__.py
      server.py
    daemon/                 # Процесс FastAPI + Worker очереди
      __init__.py
      app.py
      auth.py
      worker.py
      routes/
        __init__.py
        webhooks.py
        chat.py
        status.py
    dashboard/              # Статический фронтенд панели управления
      static/
        css/
          styles.css
        js/
          app.js
          chat.js
        index.html
```

---

## Deferred (Отложенные решения)
* **Интеграция с СУБД Postgres/MySQL:** На этапе MVP хранение данных полностью ограничено SQLite. Масштабирование на внешние БД отложено до v2.
* **Ролевая модель пользователей (RBAC):** Панель имеет единого пользователя `admin` с глобальным паролем. Разграничение прав отложено.
