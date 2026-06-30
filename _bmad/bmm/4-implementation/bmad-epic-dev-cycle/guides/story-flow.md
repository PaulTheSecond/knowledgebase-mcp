# Детальный поток обработки истории

Полный цикл обработки одной истории через все фазы workflow.

## Принципы

- **Источник истины** для статуса — `sprint-status-epic-{N}.yaml`. Его правит ОРКЕСТРАТОР.
- **Story-файл** (`## Status` + Change Log) правят СУБАГЕНТЫ — это видимая для человека/агента метка.
- Дублирование статуса осознанное (конвенция BMad). Оркестратор синхронизирует sprint-status после каждой фазы.

## Шаг 1: Разработка

**Действие:** Запустить субагент DEV
**Инструкция:** `_bmad/bmm/4-implementation/bmad-epic-dev-cycle/subagents/developer.md`
**Вход:** `story_path`, `sprint_plan_path`, `sprint_status_path`
**Ожидание:** result YAML

Перед собственной реализацией DEV обязан найти готовые/близкие решения через `knowledge-mcp`. Если решение найдено — использовать или адаптировать его; если не найдено — зафиксировать отсутствие релевантных результатов и только затем проектировать новую реализацию. Evidence поиска записывается в Dev Agent Record → Completion Notes.

**Sync sprint-status (оркестратор):**
- on_enter (если ещё не in-progress) → `{story_id} → in-progress`
- on_success → `{story_id} → review`

**Переходы:**
- `on_success` → Шаг 2 (Code Review)
- `on_blocked` → halt_with_blocker

---

## Шаг 2: Code Review + Workflow Validation (объединённая фаза)

**Действие:** Запустить субагент Code Reviewer — делает Code Review **и** Workflow Validation (Dev Agent Record) в одном проходе
**Инструкция:** `_bmad/bmm/4-implementation/bmad-epic-dev-cycle/subagents/code-reviewer.md`
**Вход:** `story_path`
**Ожидание:** result YAML с раздельными полями `workflow_issues` и `code_issues`

Code Reviewer обязан выполнить `knowledge-mcp` поиск существующих паттернов/готовых решений перед CRITICAL/HIGH/MEDIUM findings и перед рекомендациями новой реализации. Evidence поиска входит в result YAML и в секцию Code Review Results.

Severity gate: CRITICAL/HIGH/MEDIUM-block замечания возвращают историю на доработку; MEDIUM-advise/LOW/INFO остаются рекомендациями и не блокируют переход в QA. Определение MEDIUM-block vs MEDIUM-advise — см. [../subagents/code-reviewer.md](../subagents/code-reviewer.md).

**Переходы:**
- `on_pass` → Шаг 3 (QA)
- `on_needs_work` → return_to_dev (Шаг 1). В `required_fixes` — объединённый список из `workflow_issues` + `code_issues` (CRITICAL+HIGH+MEDIUM-block)

---

## Шаг 3: QA тестирование

**Действие:** Запустить субагент QA
**Инструкция:** `_bmad/bmm/4-implementation/bmad-epic-dev-cycle/subagents/qa-engineer.md`
**Вход:** `story_path`, `sprint_plan_path`, `sprint_status_path`
**Ожидание:** result YAML

**Sync sprint-status (оркестратор):**
- on_failure → `{story_id} → in-progress`
- on_success → НЕ меняем (ждём валидации QA отчёта в Шаге 3.5)

**Переходы:**
- `on_success` → Шаг 3.5 (Validation QA)
- `on_failure` → Шаг 4 (Reopen)

---

## Шаг 3.5: Workflow Validation (QA)

**Действие:** Запустить субагент QA Validator для проверки QA Results
**Инструкция:** `_bmad/bmm/4-implementation/bmad-epic-dev-cycle/subagents/validator.md`
**Вход:** `story_path`
**Ожидание:** result YAML

**Sync sprint-status (оркестратор):**
- on_pass → `{story_id} → done` (финальное закрытие)

**Переходы:**
- `on_pass` → story_complete
- `on_fail` → step_4_qa_rewrite

---

## Шаг 4 (опционально): QA доработка отчета

**Действие:** Вернуть QA для доработки отчета
**Переход:** → Шаг 3 (QA)

---

## Шаг 4 (опционально): Reopen

**Действие:** Увеличить reopen_count, вернуть к DEV
**Проверка:** `reopen_count < max_reopens`

**Переходы:**
- `if_true` → Шаг 1 (DEV)
- `if_false` → escalate

---

## Завершение истории

**Действие:** Удалить историю из stories_queue.
**Состояние sprint-status:** `{story_id}: done` (уже выставлено в Шаге 3.5).
**Переход:** → next_story_check
- если очередь не пуста → следующая история по порядку sprint-plan (`backlog` → `ready-for-dev`), затем Шаг 1 (DEV)
- если пуста → epic_complete
