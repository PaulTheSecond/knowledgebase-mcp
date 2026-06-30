# Логика оркестратора

## Принцип работы

Оркестратор работает в родительском контексте и НЕ выполняет работу сам.
Он только координирует субагентов и принимает решения на основе их результатов.

**Источник истины:** `sprint-status-epic-{N}.yaml`. Оркестратор синхронизирует его
после каждой фазы. Субагенты правят только `## Status` в story-файле.

## Состояние оркестратора

Состояние оркестратора между вызовами субагентов:

| Поле | Описание |
|------|----------|
| `sprint_status_list` | Список sprint-status файлов из аргументов запуска (один или несколько) |
| `sprint_status_path` | Путь к текущему `sprint-status-epic-{N}.yaml` |
| `sprint_plan_path` | Путь к соответствующему `sprint-plan-epic-{N}.md` |
| `epic_id` | Идентификатор эпика (`epic-{N}`) |
| `stories_queue` | Упорядоченный список историй (из sprint-plan) со статусом != done |
| `current_story` | Текущая история: `{ id, slug, story_path, status }` |
| `current_phase` | Текущая фаза (dev/code_review/qa/validation_qa) |
| `stories_completed` | Счётчик завершённых историй |
| `total_reopens` | Общий счётчик reopens |
| `last_result` | Последний результат субагента |

## Разрешение аргументов команды start-epic

```
start-epic                                  → glob docs/{activeLibrary}/sprint-status-epic-*.yaml
start-epic --epic {N}                       → docs/{activeLibrary}/sprint-status-epic-{N}.yaml
start-epic --sprint <path>                  → явный путь
start-epic --sprint "<glob>"                → glob-раскрытие, несколько эпиков
```

Если найден ровно один файл — работа идёт по нему.
Если несколько — оркестратор отчитывает пользователю найденный список и спрашивает подтверждение
(если явный glob не был передан пользователем).

## Формирование очереди stories

1. Прочитать текущий `sprint_status_path` → секция `development_status`.
2. Извлечь все ключи формата `{N}-{M}-{slug}` с значением != `done`.
3. Прочитать `sprint_plan_path` (markdown) → извлечь порядок из секции «Состав спринта»
   (таблица с колонкой `#`) и/или «Волны выполнения».
4. Отсортировать очередь по этому порядку (sequential-fallback: `3.1 → 3.2 → ... → 3.N`).
5. Первая история в очереди становится `current_story`.

## Логика принятия решений

### Перед запуском DEV (первая фаза story)
- Если `sprint-status.yaml[current_story] == backlog` → обновить на `ready-for-dev`.
  Истории проверяются пользователем на этапе sprint-planning; отдельной SM-валидации в цикле нет.
- Запустить субагент DEV.

### После результата DEV
- **if_success:**
  - Обновить `sprint-status.yaml[current_story] → review`.
  - Запустить субагент Code Reviewer.
- **if_blocked:** HALT с описанием блокера. Статус остаётся `in-progress`.
- **if_failure:** Запросить помощь пользователя.

### После результата Code Review + Workflow Validation
(объединённая фаза: субагент code-reviewer делает и ревью кода, и dev-валидацию)
- **if_pass:** Запустить субагент QA.
- **if_needs_work:**
  - Обновить `sprint-status.yaml[current_story] → in-progress`.
  - Вернуть к DEV. В `required_fixes` — объединённый список из `workflow_issues` + `code_issues` (CRITICAL+HIGH+MEDIUM-block).
  - MEDIUM-advise/LOW/INFO замечания не возвращают историю на доработку; оставить их как рекомендации (см. [../subagents/code-reviewer.md](../subagents/code-reviewer.md) — определение MEDIUM-block vs MEDIUM-advise).

### После результата QA
- **if_success (PASS):** Запустить субагент Validator (qa mode).
- **if_fail (FAIL):**
  - Увеличить `reopen_count`.
  - Обновить `sprint-status.yaml[current_story] → in-progress`.
  - Вернуть к DEV.
- **if_max_reopens:** ESCALATE к пользователю.

### После результата Validation (QA)
- **if_pass:**
  - Обновить `sprint-status.yaml[current_story] → done`.
  - Удалить историю из `stories_queue`.
  - Перейти к `next_story_check`.
- **if_fail:**
  - Вернуть к QA для доработки отчёта (sprint-status не трогается).

### После обработки всех stories эпика
- Обновить `sprint-status.yaml[epic-{N}] → done`.
- Выполнить обязательные фазы PO / Architect / Final Commit.
- Перед `next_epic_check` обнулить специфичные для эпика поля состояния:
  `stories_queue`, `current_story`, `current_phase`, `last_result`.
  Поля `sprint_status_list` и `total_reopens` сохраняются между эпиками.
- Если в `sprint_status_list` есть ещё эпики со статусом != done — перейти к следующему (повторный `init` начнёт его с чистого листа).
- Иначе — HALT в состоянии `completed`.

## Шаблон запуска субагента

Оркестратор ОБЯЗАН запускать каждую фазу через **Agent tool** (реальный субагент с изолированным контекстом).
Субагент имеет доступ ко всем инструментам (Read, Edit, Write, Bash, Grep, Glob и др.).

```
Agent(
  subagent_type: "general-purpose",
  description: "{phase} for story {story_id}",
  prompt: """
    # {Phase} Subagent

    ## ОБЯЗАТЕЛЬНЫЕ ДЕЙСТВИЯ В НАЧАЛЕ
    1. Прочитать инструкцию: {instruction_file}
    2. Прочитать workflow: _bmad/bmm/4-implementation/bmad-epic-dev-cycle/workflow.yaml
    3. Прочитать core-config: _bmad/core/config.yaml
    4. Прочитать sprint-plan (контекст спринта): {sprint_plan_path}
    5. Прочитать sprint-status (read-only): {sprint_status_path}
    6. Прочитать историю: {story_path}

    ## ЗАДАНИЕ
    Следуй всем инструкциям из {instruction_file}.

    ## ГРАНИЦЫ ОТВЕТСТВЕННОСТИ
    - Обновляй ТОЛЬКО ## Status в story-файле.
    - НЕ редактируй {sprint_status_path} — это делает оркестратор.
  """
)
```

## Запись в sprint-status.yaml

Оркестратор обновляет sprint-status **атомарно** после получения результата субагента:

1. Прочитать текущий yaml.
2. Обновить одно поле в `development_status`.
3. Обновить `last_updated: {today}`.
4. Записать файл.

Если yaml заблокирован (редактируется пользователем) — оркестратор сообщает об этом
пользователю и ждёт подтверждения (команда `resume`).

## Обработка результата субагента

После получения результата от субагента:
1. Распарсить YAML результат.
2. Обновить sprint-status.yaml (если переход требует этого — см. таблицу решений выше).
3. Обновить внутреннее состояние оркестратора.
4. Вывести краткий статус пользователю.
5. Принять решение о следующей фазе.

## Формат вывода для пользователя

```
📊 Epic {epic_id} · Story {story_id} — Phase: {phase}
Status: {status} (sprint-status: {sprint_status_value})
Queue: {stories_completed}/{total_stories} done
Summary: {summary}
Next: {next_action}
```
