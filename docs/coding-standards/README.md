# Python Coding Standards for knowledgebase-mcp

This directory outlines mandatory Python coding standards and architectural principles for the `knowledgebase-mcp` project. Both development agents and code reviewers must strictly adhere to these guidelines.

---

## Standards Directory

### [01. General Principles and PEP 8](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/01-pep8-and-naming.md)
* Strict adherence to PEP 8 using `ruff` and `black`.
* Naming conventions:
  * Classes: `PascalCase` (e.g., `DatabaseManager`).
  * Functions, methods, variables: `snake_case` (e.g., `get_db_connection`).
  * Constants: `UPPER_CASE` (e.g., `DEFAULT_POLL_INTERVAL`).

### [02. Strict Type Hinting](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/02-type-hinting.md)
* All public function and method signatures **must** declare parameter and return type annotations.
* Use Python 3.10+ native types (e.g., `list[str]`, `dict[str, Any]`, `int | None` instead of `typing.List`, `typing.Dict`, `typing.Optional`).
* Avoid `Any` whenever possible. Any use of `# type: ignore` must be accompanied by an explanatory comment.

### [03. SQLite Safety & Concurrency](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/03-sqlite-safety.md)
* All mutating transactions must execute under `BEGIN IMMEDIATE TRANSACTION`.
* String concatenation for SQL queries is strictly prohibited; parameter binding (`?`) is required.
* Compulsory WAL mode (`journal_mode=WAL` and `synchronous=NORMAL`) to ensure non-blocking concurrent reads between MCP and Daemon processes.
* The stdio MCP process must open database connections in Read-Only mode.

### [04. Error Handling and Logging](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/04-error-handling-and-logging.md)
* Bare `except:` statements are prohibited; catch explicit exceptions (e.g., `except sqlite3.Error:`).
* Preserve exception tracebacks when re-raising: `raise CustomException("error") from err`.
* All logging must route through standard `logging` modules (in the MCP process, logs write exclusively to `sys.stderr` since `sys.stdout` is reserved for MCP JSON-RPC).

### [05. Documentation and Docstrings](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/05-docstrings.md)
* Google-style docstrings for all public modules, classes, and methods.
* Explicit descriptions of arguments, types, return values, and raised exceptions.

### [06. Testing Standards (pytest)](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/06-testing.md)
* `pytest` as the canonical test framework.
* Arrange-Act-Assert (AAA) pattern for unit and integration tests.
* Isolate external network and system dependencies (Git CLI, OmniRoute HTTP API) using mocks and fixtures.

---

## Automated Verification (Linter & Formatter)

Before committing code, verify quality by running:
```bash
ruff check .
ruff format .
pytest
```
