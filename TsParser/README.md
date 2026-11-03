# ts-parser: Компиляторный семантический анализатор TypeScript, React и Angular

Выделенный компиляторный парсер семантических связей для экосистемы TypeScript/JavaScript/React в составе `knowledgebase-mcp` (согласно [ADR-009](../docs/adr-009-compiler-semantic-analysis-ts-angular.md)).

Является прямым аналогом `RoslynParser` (.NET) для мира TypeScript: заменяет эвристический текстовый синтаксический анализ Tree-sitter на официальный компиляторный семантический анализ через **TypeScript Compiler API**.

---

## 🎯 Решаемые задачи

1. **Семантика вызовов в JSX**: использование компонентов в JSX вида `<ApproverMatrixSelectionBar />` связывается с декларацией компонента как отношение `USES`, устраняя проблему «разрыва цепочки компонентов» при анализе радиуса влияния (blast radius).
2. **Алиасы путей (`tsconfig.json paths`)**: прозрачное разрешение псевдонимов (`@components/*`, `@shared/*`), реэкспортов через barrel-файлы (`index.ts`) и модулей в монорепозиториях.
3. **100% изоляция от `node_modules`**: в граф связей попадают **строго** символы и вызовы между файлами репозитория; библиотеки из `node_modules` и системные тайпинги `lib.d.ts` физически отфильтровываются по пути декларации.
4. **Субсекундный режим diff для PR-ревью**: компилятор работает в оперативной памяти (`noEmit: true`, `skipLibCheck: true`), обеспечивая время отклика ~15–30 мс на 10 файлов.

---

## 🚀 CLI Контракт

```bash
node dist/ts-parser.js --project <path/to/tsconfig.json> [options]

Параметры:
  --project <path>    Путь к tsconfig.json / jsconfig.json (обязательный)
  --root <path>       Корневая директория репозитория (по умолчанию папка tsconfig.json)
  --files <path>      Файл со списком относительных/абсолютных путей или CSV-список (для diff PR)
  --mode <full|diff>  Режим анализа: full — полная индексация, diff — инкрементальный анализ
  --help, -h          Показать справку
```

### Формат вывода JSON (stdout)

Полностью совместим со структурой `RoslynParser` и SQLite схемой `knowledgebase-mcp`:

```json
{
  "symbols": [
    {
      "ast_node_id": 1,
      "file_path": "/app/frontend/src/components/UserCard.tsx",
      "name": "UserCard",
      "qualified_name": "UserCard",
      "kind": "component",
      "signature": "export const UserCard: React.FC<UserCardProps>",
      "language": "typescript",
      "body": "export const UserCard = ...",
      "line_start": 10,
      "line_end": 28,
      "is_test": false
    }
  ],
  "edges": [
    {
      "source_ast_id": 1,
      "target_qualified_name": "Button",
      "target_file_path": "/app/frontend/src/components/Button.tsx",
      "kind": "USES"
    }
  ],
  "metrics": {
    "files_analyzed": 1,
    "symbols_found": 1,
    "edges_found": 1,
    "duration_ms": 13
  }
}
```

---

## 🛠️ Сборка автономного бандла

Модуль собирается с помощью `esbuild` в единый автономный JS-файл `dist/ts-parser.js` (~9.5 MB), куда упаковывается TypeScript Compiler API. В рантайме внешний `node_modules` **не требуется**:

```bash
npm run build    # Сборка dist/ts-parser.js через esbuild
npm test         # Запуск автоматизированных тестов (node:test)
npm run lint     # Проверка строгой типизации (tsc --noEmit)
```

---

## 📦 Контейнеризация и изоляция

- Node.js рантайм и собранный `dist/ts-parser.js` поставляются **строго внутри Docker-образа** (multi-stage сборка).
- Пользователям и AI-агентам (Claude Code, Antigravity, Cursor) **не требуется** устанавливать Node.js на хост-машину.
