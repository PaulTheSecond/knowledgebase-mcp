# Python Coding Standards for knowledgebase-mcp

Этот документ содержит обязательные стандарты кодирования и правила проектирования для языка Python в рамках данного проекта. Агенты разработки (James) и ревьюеры кода (CodeReview) должны строго соблюдать эти правила.

---

## 🗂️ Содержание стандартов

### [01. Общие принципы и PEP 8](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/01-pep8-and-naming.md)
* Соблюдение PEP 8. Использование `ruff` / `black` для форматирования.
* Правила именования:
  * Классы: `PascalCase` (например, `DatabaseManager`).
  * Функции, методы, переменные: `snake_case` (например, `get_db_connection`).
  * Константы: `UPPER_CASE` (например, `DEFAULT_POLL_INTERVAL`).

### [02. Строгая типизация (Type Hinting)](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/02-type-hinting.md)
* Все сигнатуры функций и методов **обязаны** иметь аннотации типов параметров и возвращаемых значений.
* Использование встроенных типов Python 3.10+ (например, `list[str]`, `dict[str, Any]`, `int | None` вместо `typing.List`, `typing.Dict`, `typing.Optional`).
* Исключение использования `Any` без крайней необходимости. Применение `type: ignore` должно сопровождаться комментарием-пояснением.

### [03. Безопасность при работе с SQLite](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/03-sqlite-safety.md)
* Все запросы записи должны выполняться в транзакциях с блокировкой `BEGIN IMMEDIATE TRANSACTION`.
* Запрещено собирать SQL-запросы конкатенацией строк. Разрешена только биндинг-подстановка параметров (`?`).
* Обязательное использование режима WAL (`journal_mode=WAL` и `synchronous=NORMAL`) для предотвращения блокировок между процессами MCP и Web Daemon.
* Процесс MCP-сервера должен открывать соединение с базой данных в режиме Read-Only.

### [04. Обработка ошибок и логирование](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/04-error-handling-and-logging.md)
* Запрещено использовать пустой `except:`. Ловить только специфичные исключения (например, `except sqlite3.Error:` или `except ValueError:`).
* При перевыбросе ошибок сохранять стек вызовов: `raise CustomException("error") from err`.
* Все логи должны выводиться через стандартный модуль `logging` (в MCP-процессе все логи строго направляются в `sys.stderr`, так как `sys.stdout` зарезервирован под протокол MCP).

### [05. Документирование и Docstrings](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/05-docstrings.md)
* Использование Google-style docstrings для всех публичных модулей, классов и функций.
* Описание аргументов, типов, возвращаемых значений и выбрасываемых исключений.

### [06. Стандарты тестирования (pytest)](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/06-testing.md)
* Использование `pytest` в качестве фреймворка тестирования.
* Написание тестов по паттерну Arrange-Act-Assert (AAA).
* Изоляция внешних вызовов (Git CLI, OmniRoute HTTP API) через mocks/fixtures.

---

## 🛠️ Запуск автоматических проверок (Linter & Formatter)
Перед коммитом разработчик обязан запустить:
```bash
ruff check .
ruff format .
pytest
```
