# Восстановление контекста оркестратора

## Когда требуется

После сжатия контекста (context compaction), смены сессии или при потере состояния
оркестратор ОБЯЗАН восстановить состояние. Главный источник восстановления —
`sprint-status-epic-{N}.yaml`, поэтому процедура детерминирована и не зависит от
того, помнит ли оркестратор предыдущие шаги.

## Признаки потери контекста

- Потеря информации о текущей истории или фазе.
- Неизвестно, какой агент сейчас активен.
- Непонятно, на каком шаге workflow находимся.
- Сообщение системы о сжатии контекста.

## Процедура восстановления

### Шаг 1. Перечитать конфигурацию
**Действие:** Перечитать `_bmad/core/config.yaml`.
**Цель:** Определить `activeLibrary` и пути к артефактам.

### Шаг 2. Перечитать workflow
**Действие:** Перечитать `_bmad/bmm/4-implementation/bmad-epic-dev-cycle/workflow.yaml`.
**Цель:** Восстановить понимание процесса и границ ответственности.

### Шаг 3. Найти sprint-status файлы
**Действие:** Выполнить glob `docs/{activeLibrary}/sprint-status-epic-*.yaml`.
Если пользователь при запуске явно передавал `--sprint <path|glob>` и это значение
сохранено во внешнем состоянии/команде — использовать его.

### Шаг 4. Выбрать текущий эпик
**Действие:** Прочитать каждый найденный yaml.
**Правило:** Текущий эпик = первый в списке, у которого `epic-{N}` != `done`.

Если все эпики `done` → переход к Шагу 7 (проверка финальных фаз).

### Шаг 5. Найти текущую историю
**Действие:** Прочитать `sprint-plan-epic-{N}.md` (рядом со sprint-status).
Извлечь порядок stories из таблицы «Состав спринта».
**Правило:** Текущая история = первая в этом порядке, у которой
`development_status[{N}-{M}-{slug}] != done`.

### Шаг 6. Определить фазу

Определяется по статусу в sprint-status + содержимому story-файла:

| sprint-status | Содержимое story-файла | Возобновить с фазы |
|---------------|-----------------------|--------------------|
| `backlog` | story-файл отсутствует | HALT — нужен bmad-create-story |
| `backlog` | story-файл есть | оркестратор поднимает статус на `ready-for-dev` → `development` |
| `ready-for-dev` | нет Dev Agent Record | `development` (начало) |
| `in-progress` | Tasks не все `[x]`, Dev Agent Record пустой | `development` (продолжение) |
| `in-progress` | Tasks все `[x]`, нет QA Results | `development` (финальный отчёт) |
| `review` | Есть Dev Agent Record, нет QA Results | `code_review` |
| `review` | Есть Dev Agent Record + QA Results (FAIL) | `development` (после reopen) |
| `review` | Есть QA Results (PASS) без `## Status: Done` | `qa_testing` (завершение) |
| `done` | — | `next_story_check` |

### Шаг 7. КРИТИЧЕСКАЯ проверка финальных фаз эпика

**ОБЯЗАТЕЛЬНО после восстановления контекста!**

Если **все stories** текущего эпика имеют `development_status: done`:

1. **Проверить epic-{N}:**
   - Если `!= done` → перейти к `epic_complete`.

2. **Проверить PO Documentation (фаза 7):**
   - Читаем Change Log файла эпика `docs/{activeLibrary}/epics/epic-{N}-*.md`.
   - Ищем запись `PO documentation updated` / `Product Owner`.
   - Если нет → `po_documentation`.

3. **Проверить Architect Documentation (фаза 8):**
   - Ищем `Architect documentation updated` / `Architecture`.
   - Если нет → `architect_documentation`.

4. **Проверить Final Commit (фаза 9):**
   - `git log --oneline -10` → ищем commit с документацией эпика.
   - Если нет → `final_commit`.

5. **Проверить наличие следующего эпика (multi-epic запуск):**
   - Если в `sprint_status_list` есть ещё yaml со статусом != done → `next_epic_check` → `init`.

6. Только если всё выполнено → `completed`.

**ВАЖНО:** Фазы 7, 8, 9 часто теряются при сжатии контекста. Эта проверка гарантирует их выполнение.

## Обязательные файлы для перечитывания

1. `_bmad/core/config.yaml`
2. `_bmad/bmm/4-implementation/bmad-epic-dev-cycle/workflow.yaml`
3. Все sprint-status yaml по активному glob
4. Sprint-plan текущего эпика
5. Файл текущей истории (определён по шагам 4–5)

## Сообщение после восстановления

```
🔄 КОНТЕКСТ ВОССТАНОВЛЕН

Active Library: {activeLibrary}
Sprint Status: {sprint_status_path}
Sprint Plan:   {sprint_plan_path}
Current Epic:  {epic_id}   (status: {epic_status})
Current Story: {story_id} — {story_slug}
               (status: {story_status}, file: {story_path})
Current Phase: {phase_name}
Current Agent: {agent_name}
Queue:         {done_count}/{total_count} stories done
Next Action:   {what_to_do_next}

Продолжаю выполнение workflow...
```
