---
name: bmad-python-standards
description: 'Enforces Python code quality standards, type hints, PEP 8, SQLite WAL safety, and proper exception handling. Use when verifying code structure or running code reviews on Python modules.'
---

# Python Coding Standards Enforcer

## Purpose
This skill validates Python source code compliance with the project-specific and global Python coding standards. It acts as an automated quality gate for both developers (James) and code reviewers (CodeReview).

## Core Verification Rules

### 1. Style & PEP 8 Check
* Run `ruff check .` and `ruff format --check .` to ensure correct formatting and code style compliance.
* Max line length must be 120.

### 2. Type Hinting Compliance
* Check that all functions, methods, and properties have complete parameter and return type annotations.
* No raw `typing.List` or `typing.Union` usage; use modern Python 3.10+ types: `list`, `dict`, and `|`.
* Do not allow `Any` without explicit justification comments.

### 3. SQLite Concurrency & Safety
* Check all DB connection creations: WAL mode (`journal_mode=WAL` and `synchronous=NORMAL`) must be enabled.
* Ensure all SELECT/write SQL commands use placeholder bindings (`?`). String interpolation (f-strings) in SQL is a **CRITICAL BLOCKING ISSUE**.
* Verify that the MCP process opens SQLite databases with `mode=ro` (Read-Only).

### 4. Error Handling and Logging
* Avoid bare `except:`.
* Exceptions must be raised with `from` (e.g. `raise CustomError(...) from original_err`) to preserve stack traces.
* Check that `sys.stdout` is never used for logs inside the MCP server. All diagnostics and log outputs must be routed to `sys.stderr`.

### 5. Docstrings
* Google-style docstrings must be present on all public modules, classes, and methods.

### 6. Testing Structure
* Verify that tests use `pytest` and follow the Arrange-Act-Assert (AAA) pattern.
* Ensure that file operations and external dependencies (Git CLI, OmniRoute REST API) are mocked properly.

---

## How to run checks

Reviewers can trigger automated linting and tests using:
```bash
ruff check .
pytest
```

For manual review, cross-reference the changed files list with the standards defined in:
- [c:\Repos\knowledgebase-mcp\docs\coding-standards\README.md](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/README.md)
- [01-pep8-and-naming.md](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/01-pep8-and-naming.md)
- [02-type-hinting.md](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/02-type-hinting.md)
- [03-sqlite-safety.md](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/03-sqlite-safety.md)
- [04-error-handling-and-logging.md](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/04-error-handling-and-logging.md)
- [05-docstrings.md](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/05-docstrings.md)
- [06-testing.md](file:///c:/Repos/knowledgebase-mcp/docs/coding-standards/06-testing.md)
