# Фазы выполнения Epic Development Cycle

## Phase 1: Инициализация

**Действие:** INIT

**Шаги:**
1. Прочитать `_bmad/core/config.yaml` для определения `activeLibrary`.
2. Определить список sprint-status файлов:
   - Из аргументов `--sprint <path|glob>` или `--epic {N}`.
   - По умолчанию — glob `docs/{activeLibrary}/sprint-status-epic-*.yaml`.
3. Для каждого yaml:
   - Валидировать наличие `development_status`.
   - Извлечь `epic_id` (`epic-{N}`) и разрешить `sprint-plan-epic-{N}.md`.
4. Выбрать **первый** эпик со статусом != `done` → текущий.
5. Прочитать `sprint-plan-epic-{N}.md` → извлечь порядок историй из секции
   «Состав спринта» (колонка `#`) и/или «Волны выполнения».
6. Сформировать очередь: все stories с `development_status != done`,
   отсортированные по порядку из sprint-plan.
7. Если `epic-{N}` в yaml имеет статус `backlog` → обновить на `in-progress` и записать yaml.
8. Для `current_story`: если статус `backlog` → обновить на `ready-for-dev`. Истории проверяются пользователем на этапе sprint-planning; отдельная SM-фаза валидации удалена.

**Выход:** Очередь историй, `sprint_status_path`, `sprint_plan_path` в состоянии оркестратора.

**Переход:** → Phase 2 (Development)

---

## Phase 2: Development

**Агент:** dev (James)
**Инструкция:** [../subagents/developer.md](../subagents/developer.md)
**Действие:** IMPLEMENT_STORY

**При входе (оркестратор):**
- `sprint-status.yaml[{story_id}] → in-progress` (если ещё не in-progress).

**При успехе (субагент):**
- Dev Agent Record содержит `Knowledge MCP Research`: какие запросы выполнены, какие готовые/близкие решения найдены, почему выбран `reuse/adapt/new`.
- Story-файл: `## Status` → `Ready For Review`.
- Git commit создан.

**При успехе (оркестратор):**
- `sprint-status.yaml[{story_id}] → review`.
- Переход → Phase 2.5.

**При провале:**
- HALT с описанием блокера. sprint-status остаётся `in-progress`.

---

## Phase 2.5: Code Review + Workflow Validation (объединённая)

**Агент:** code-reviewer (CodeReview)
**Инструкция:** [../subagents/code-reviewer.md](../subagents/code-reviewer.md)
**Действие:** REVIEW_CODE_AND_VALIDATE_DEV

Объединяет две проверки в одном проходе субагента:
- **BLOCK A — Workflow Validation:** Dev Agent Record, File List, Completion Notes, git commit, build
- **BLOCK B — Code Review:** стандарты, безопасность, производительность, архитектура

BLOCK A обязан проверять наличие `Knowledge MCP Research`; отсутствие evidence поиска готового решения через `knowledge-mcp` — workflow issue и `NEEDS_WORK`.

BLOCK B обязан выполнять `knowledge-mcp` поиск существующих паттернов перед CRITICAL/HIGH/MEDIUM findings и перед рекомендациями новой реализации. Findings без `Knowledge MCP Evidence` считаются неполными и должны быть снижены до гипотезы либо возвращены на доработку ревью.

**Severity gate:** CRITICAL/HIGH/MEDIUM-block замечания возвращают историю на доработку (`NEEDS_WORK`). MEDIUM-advise/LOW/INFO замечания остаются рекомендациями и не блокируют переход в QA.

- **MEDIUM-block** — архитектура, безопасность, корректность данных, blast radius > 1 файла, отсутствие тестов на критичную ветку.
- **MEDIUM-advise** — стиль, локальная читаемость, минорные performance/тесты, не затрагивающие AC.
- См. определения в [../subagents/code-reviewer.md](../subagents/code-reviewer.md).

**Результат:** одно общее решение PASS | NEEDS_WORK; в YAML-результате раздельные поля `workflow_issues` и `code_issues.critical/high/medium_block/medium_advise/low_recommendations`.

**При PASS:**
- Переход → Phase 3 (QA Testing)

**При NEEDS_WORK:**
- Переход → Phase 2 (возврат к разработчику), `required_fixes` содержит объединённый список действий (CRITICAL → HIGH → MEDIUM-block; MEDIUM-advise/LOW/INFO туда не попадают)

---

## Phase 3: QA Testing

**Агент:** qa (Quinn)
**Инструкция:** [../subagents/qa-engineer.md](../subagents/qa-engineer.md)
**Действие:** TEST_STORY

**При PASS (субагент):**
- Story-файл: `## Status` → `Done`.

**При PASS (оркестратор):**
- Переход → Phase 3.5 (sprint-status пока НЕ переключается на done — ждём валидации QA отчёта).

**При FAIL (субагент):**
- Story-файл: `## Status` → `In Progress`.

**При FAIL (оркестратор):**
- `sprint-status.yaml[{story_id}] → in-progress`.
- Увеличить `reopen_count`.
- Переход → Phase 2 (возврат к разработчику).

---

## Phase 3.5: Workflow Validation (QA)

**Агент:** validator (Victor)
**Инструкция:** [../subagents/validator.md](../subagents/validator.md)
**Действие:** VALIDATE_QA_REPORT

**При PASS (оркестратор):**
- `sprint-status.yaml[{story_id}] → done`.
- Переход → Phase 5.

**При FAIL:**
- Переход → Phase 3 (возврат к QA для доработки отчёта).

---

## Phase 4: Переход к следующей истории

**Действие:** NEXT_STORY

**Шаги:**
1. Удалить текущую историю из `stories_queue` оркестратора.
2. Если очередь пуста → Phase 5.
3. Иначе взять следующую историю по порядку из sprint-plan; если её статус `backlog` → поднять на `ready-for-dev`; перейти → Phase 2.

---

## Phase 5: Завершение историй эпика

**Действие:** COMPLETE_STORIES
**Триггер:** Все истории эпика имеют `development_status: done` в sprint-status.yaml.

**Шаги:**
1. `sprint-status.yaml[epic-{N}] → done`.
2. Обновить файл эпика (Status → Done, Success Criteria, Epic Completion Summary).
3. Вывести промежуточный отчёт.

**Переход:** → Phase 6

---

## Phase 6: Обновление документации (PO)

**Агент:** po (Sarah)
**Инструкция:** [../subagents/product-owner.md](../subagents/product-owner.md)
**Действие:** UPDATE_DOCUMENTATION

**Переход:** → Phase 7

---

## Phase 7: Обновление архитектурных документов

**Агент:** architect (Alex)
**Инструкция:** [../subagents/architect.md](../subagents/architect.md)
**Действие:** UPDATE_ARCHITECTURE_DOCS

**Переход:** → Phase 8

---

## Phase 8: Финальный коммит документации

**Действие:** FINAL_DOCUMENTATION_COMMIT

**Шаги:**
1. Создать финальный git commit для документации эпика (изменения от PO/Architect).
2. Вывести финальный отчёт по эпику.

**Формат сообщения:** `docs(epic-{N}): update README/PRD/architecture after epic close`

**Запрещено в сообщении коммита (см. CLAUDE.md / AGENTS.md):**
- Подпись `Generated with Claude Code` или аналогичная.
- Trailer `Co-Authored-By: Claude <noreply@anthropic.com>`.
- Любые auto-generated подписи и AI-trailer'ы.

Сообщение должно содержать только бизнес-смысл без AI-атрибуции.

**Переход:** → Phase 9 (next_epic_check)

---

## Phase 9: Проверка следующего эпика

**Действие:** NEXT_EPIC_CHECK
**Триггер:** При запуске было передано несколько sprint-status файлов (multi-epic режим).

**Шаги:**
1. Проверить `sprint_status_list` оркестратора.
2. Найти первый эпик со статусом != `done`.

**Переходы:**
- Если найден → Phase 1 (init для следующего эпика).
- Если нет → `completed` — HALT, все эпики завершены.

> При запуске на одном эпике эта фаза просто пропускается и сразу — `completed`.
