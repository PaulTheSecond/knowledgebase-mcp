# Epic Planning Cycle — Roadmap улучшений

Документ накапливает предложения по расширению воркфлоу `epic-planning-cycle`. Фиксируется по итогам ретроспектив конкретных эпиков.

---

## Источник

- **Epic 5 (PageShell), 2026-04-19** — Velmoren после полного прохождения цикла запросил три процессных улучшения.

---

## Планируемые фазы

### Фаза 0: `prd_validation` (перед `init`)

**Статус:** planned
**Приоритет:** high
**Зачем:** В Epic 5 PRD page-shell-layout.md попал в цикл в статусе Draft v0.2 без формальной валидации. PO-валидация эпика выявила пробелы, которые на самом деле лежали в PRD (несогласованные требования A11y). Раннее обнаружение дешевле.

**Контракт фазы:**

```
agent:       analyst | pm
instruction: subagents/prd-validator.md (TODO: создать)
description: Валидация PRD по стандартам (полнота, измеримость, согласованность, версионирование)
on_pass:     → init (определение типа эпика)
on_needs_work: → halt_user_help (правки в PRD не входят в epic-planning-cycle)
inputs:      prd_path
checks:
  - Все требования измеримы (FR имеют AC, NFR — метрики)
  - Acceptance criteria на уровне продукта
  - Scope чёткий (IN / OUT / deferred)
  - Нет противоречий между разделами
  - Версия зафиксирована (v1.0+, не Draft)
  - Figma-ноды / wireframes есть (для UI-эпиков)
```

**Как запускать:** перед `plan-epic` команда `*validate-prd {path}` (уже есть в `bmm`), интеграция в orchestrator — добавить вызов автоматически.

---

### Фаза 5.5: `po_stories_review` (между `sm_story_creation` и `architect_documentation`)

**Статус:** planned
**Приоритет:** medium
**Зачем:** PO в фазе 2 валидирует только эпик, но не финальные AC в stories. SM-агент декомпозирует AC из эпика в детализированные Given/When/Then — и эта декомпозиция может расходиться с намерениями продукта. Epic 5 прошёл SM-фазу с 96/100, но часть Given/When/Then условий требовала бы продуктового одобрения (например, выбор иконки «Войти», формулировки русских констант).

**Контракт фазы:**

```
agent:       po
instruction: subagents/po-stories-reviewer.md (TODO: создать)
description: PO-ревью AC всех созданных stories на согласованность с эпиком и PRD
on_pass:     → architect_documentation
on_needs_work: → sm_story_creation (повторно, return_count++)
inputs:      epic_path, stories_dir
checks:
  - AC в stories не расширяют скоуп эпика
  - AC остаются измеримыми (не теряется Given/When/Then в переформулировке)
  - Приоритеты stories согласованы (критический путь логичен)
  - Русские формулировки UI-текстов одобрены (или отмечены как TODO)
  - Нет противоречий между stories (один AC не противоречит другому)
```

---

### Фаза 6.5: `architect_verify` (после `architect_documentation`)

**Статус:** planned
**Приоритет:** medium
**Зачем:** Architect-документ (в случае Epic 5 — новый `page-shell-architecture.md`, 13 разделов) создаётся «на свежую голову» и может содержать ссылки-призраки, несогласованность с эпиком, устаревшие цитаты из PRD. Формальная проверка консистентности необходима.

**Контракт фазы:**

```
agent:       validator | technical-writer
instruction: subagents/architect-verifier.md (TODO: создать)
description: Проверка архитектурных документов на консистентность, link-health, соответствие эпику
on_pass:     → final_validation
on_needs_work: → architect_documentation (return_count++)
inputs:      architecture_dir, epic_path
checks:
  - Все ссылки внутри архитектуры резолвятся (нет dangling links)
  - Имена компонентов/моделей/методов совпадают с эпиком и stories
  - Токены, упомянутые в документе, существуют в Knowledge/tokens.md
  - Не введены новые токены вопреки решению эпика
  - Структура директорий consistent с project-structure.md
  - Package.json / public-api.ts упомянутые пути корректны
```

---

## Планируемая последовательность фаз

```
[0] prd_validation           ← NEW
 ↓
[1] init
 ↓
[2] analyst_epic / architect_epic
 ↓
[3] po_validation
 ↓
[4] pm_validation
 ↓
[5] qa_validation
 ↓
[6] sm_story_creation
 ↓
[6.5] po_stories_review      ← NEW
 ↓
[7] architect_documentation
 ↓
[7.5] architect_verify       ← NEW
 ↓
[8] final_validation
 ↓
[9] final_commit → ready_for_dev
```

---

## Приоритет реализации

| # | Фаза | Приоритет | Оценка | Замечание |
|---|------|-----------|--------|-----------|
| 1 | `prd_validation` | High | 3 часа (субагент + интеграция) | Наибольший ROI — ранний фильтр |
| 2 | `architect_verify` | Medium | 2 часа (link-check + symbol-match) | Можно реализовать скриптом (ripgrep) + subagent для анализа |
| 3 | `po_stories_review` | Medium | 3 часа | Тяжёлый субагент (проход по всем stories) |

---

## История изменений

| Дата | Изменение | Автор |
|------|-----------|-------|
| 2026-04-19 | Документ создан. Зафиксированы 3 планируемые фазы на основе опыта Epic 5. | BMad Master (Velmoren request) |
