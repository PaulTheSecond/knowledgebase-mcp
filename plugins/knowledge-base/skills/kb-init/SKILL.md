---
name: kb-init
description: Инициализация и первичное подключение репозитория или рабочей области (воркспейса) к Knowledge Base MCP. Используйте, когда пользователь просит подключить новый репозиторий, настроить воркспейс в базе знаний, запустить первичную индексацию (sync/sync-project) или установить пресеты/хуки для Claude Code и Antigravity.
---

# Knowledge Base Workspace & Repository Initialization Skill (`kb-init`)

Этот скилл предназначен для автоматического обнаружения структуры проекта, регистрации репозиториев и воркспейсов в Knowledge Base MCP (`data/config.yaml`), выполнения первичной индексации кода/документации и настройки агентов (Claude Code, Antigravity).

## Когда активировать скилл
- Пользователь просит подключить проект: «подключи этот репозиторий к базе знаний», «инициализируй проект в knowledge-mcp», «настрой воркспейс для базы знаний», «проиндексируй проект».
- Запуск команды `/init-kb` или `/kb-init`.
- Настройка интеграции ассистента с базой знаний на новой машине или в новом воркспейсе.

---

## Процесс инициализации (5 шагов)

### Шаг 1. Анализ целевой рабочей области
Определите параметры проекта в рабочей директории:
1. **Путь к корню**: абсолютный путь к репозиторию или воркспейсу (например, `C:/Repos/DVCN/Orbita` или текущий рабочий каталог).
2. **Тип структуры**:
   - **Одиночный репозиторий**: `.git` находится прямо в корне.
   - **Воркспейс с ворктри (worktree workspace)**: содержит структуру `worktrees/master` и `worktrees/{task}` (как в проекте Орбита).
   - **Монорепозиторий / Multi-repo**: содержит несколько подпапок с собственными репозиториями.
3. **Определение используемых языков и технологий**:
   - `*.sln`, `*.csproj` $\rightarrow$ `csharp` (Roslyn-парсер)
   - `package.json`, `tsconfig.json` $\rightarrow$ `typescript` / `javascript` (TypeScript/Babel AST)
   - `pyproject.toml`, `requirements.txt` $\rightarrow$ `python` (Tree-sitter)
   - `go.mod` $\rightarrow$ `go`
4. **Обнаружение документации и правил**:
   - Проверьте наличие `.kb/`, `CLAUDE.md`, `README.md`, `rules/`, `docs/`, `ONBOARDING.md`, `AGENTS.md`.

---

### Шаг 2. Конфигурация в `data/config.yaml`
Откройте конфигурационный файл базы знаний (`data/config.yaml`).

#### Вариант А: Одиночный репозиторий
Добавьте запись в секцию `repositories`:
```yaml
repositories:
  - id: <repo_name>
    url: <git_remote_url_or_none>
    local_path: <absolute_path_to_repo>
    branch: master
    language: <csharp|typescript|python|go>
    triggers:
      cron: 0 */4 * * *
```

#### Вариант Б: Воркспейс / Проект (worktrees)
Добавьте запись в секцию `projects`:
```yaml
projects:
  - id: <project_id>
    name: <Human-readable Name>
    workspace_root: <absolute_path_to_workspace>
    type: worktree_workspace
    base_path: worktrees/master
    tasks_path: worktrees/{task}
    triggers:
      cron: "*/10 * * * *"
    docs_paths:
      - .kb
      - CLAUDE.md
      - docs
    repositories:
      - id: <sub_repo_1>
        path: <sub_repo_1_dir>
        language: <language>
      - id: <sub_repo_2>
        path: <sub_repo_2_dir>
        language: <language>
```

---

### Шаг 3. Первичная индексация (CAS, AST, Embeddings)
Запустите индексацию через CLI или Docker-контейнер:

- **Для одиночного репозитория**:
  ```bash
  python -m knowledge_mcp.main sync --repo-id <repo_id> --repo-path "<path>" --all
  ```
- **Для проекта/воркспейса**:
  ```bash
  python -m knowledge_mcp.main sync-project --project-id <project_id>
  ```
- **Либо через MCP-инструмент (если сервер уже активен)**:
  Вызовите инструмент `knowledge_sync_repo(repo_id="<id>")`.

Сервер создаст CAS-слепки файлов (`files`, `file_contents`), распарсит AST-символы (`symbols`, `symbol_edges`) и сгенерирует векторные эмбеддинги (`chunks_vec`).

---

### Шаг 4. Настройка пресетов и хуков для агентов
Для автоматического использования базы знаний агентами настройте профили:

1. **Для Claude Code**:
   ```bash
   python -m knowledge_mcp.main preset claude --install
   ```
   Команда объединит настройки в `~/.claude/settings.json`:
   - Подключение MCP-сервера `knowledgebase`.
   - Хук `SessionStart` — опрос статуса базы знаний.
   - Хук `PostToolUse` на `Edit|Write` — JIT-доиндексация измененных файлов на лету.
   - Хук `PreToolUse` на `Grep|Glob` — рекомендация использовать семантический поиск перед широким сканированием файлов.

2. **Для Antigravity**:
   - Включите плагин `knowledge-review` / `knowledge-init` из маркетплейса репозитория или добавьте подключение к `http://localhost:8000/mcp/sse`.

---

### Шаг 5. Верификация (Healthcheck)
Убедитесь, что база знаний готова к работе:
1. **Проверка через статус-команду**:
   ```bash
   python -m knowledge_mcp.main sync-file --check-status
   ```
   Должно вернуть:
   - `Status: OK (Daemon Active)`
   - `Indexed files in CAS: > 0`
   - `Workspaces: >= 1 project(s)`
2. **Проверочный поисковый запрос**:
   Вызовите инструмент `knowledge_search(query="<ключевой символ проекта>", limit=3)`.
3. Сообщите пользователю об успешном подключении и готовности к ревью и поиску.
