# 01. PEP 8 & Naming Standards

## PEP 8 Compliance
1. All Python code must strictly comply with PEP 8 standards.
2. Line length limit is set to 120 characters (improving code readability on widescreen displays).
3. Automated formatting and linting is enforced via `ruff` (configured in `pyproject.toml` or `setup.cfg`).

## Naming Conventions
* **Modules and Packages:** `snake_case`, lowercase letters only. Avoid dashes or uppercase letters (e.g., `db_utils.py` instead of `db-Utils.py`).
* **Classes:** `PascalCase`. Use descriptive nouns (e.g., `RepositorySynchronizer`).
* **Functions and Methods:** `snake_case`. Use verbs describing actions (e.g., `fetch_repository_updates`).
* **Variables:** `snake_case`. Names must be clear and descriptive (e.g., `local_commit_sha` instead of `sha`).
* **Constants:** `UPPER_CASE` with underscore separators (e.g., `MAX_RETRY_ATTEMPTS`).
* **Private Attributes and Methods:** prefixed with a single underscore `_` (e.g., `_authenticate_session`).

## Import Organization
1. Group imports in the following canonical order:
   * Python standard library modules (e.g., `os`, `sys`, `sqlite3`).
   * Third-party libraries (e.g., `fastapi`, `openai`, `pydantic`).
   * Local project modules (e.g., `core.git_manager`, `db.models`).
2. Wildcard imports (e.g., `from module import *`) are strictly prohibited.
