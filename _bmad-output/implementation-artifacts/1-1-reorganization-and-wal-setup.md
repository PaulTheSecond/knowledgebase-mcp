# Story 1.1: Reorganization, DB Migration and WAL Setup

Status: done

## Story

As a developer,
I want to split the original flat codebase into `core`, `db`, and `mcp` submodules and configure WAL mode for SQLite,
so that the project is prepared for adding a web daemon and free of concurrent read/write conflicts.

## Acceptance Criteria

1. **Given** an existing flat repository,
   **When** I move files into `core/`, `db/`, and `mcp/` subdirectories and run tests,
   **Then** all imports resolve successfully and the MCP server functions normally.
2. **And** upon every SQLite connection via `db.py`, `PRAGMA journal_mode = WAL;` and `PRAGMA synchronous = NORMAL;` are enforced.
3. **And** an empty `system_status` table is created in the database to store worker runtime state.

## Tasks / Subtasks

- [x] Directory structure reorganization (AC: 1)
  - [x] Create `core/`, `db/`, `mcp/` directories inside `knowledge_mcp/`.
  - [x] Move `code_parser.py`, `markdown_parser.py`, `embeddings.py`, `indexer.py` into `core/`.
  - [x] Move `server.py` to `mcp/server.py`.
  - [x] Update `__init__.py` files across all packages.
- [x] SQLite WAL mode configuration (AC: 2)
  - [x] Move `db.py` to `db/models.py`.
  - [x] Extract connection initialization into `db/connection.py`.
  - [x] Add execution of PRAGMA commands `journal_mode=WAL` and `synchronous=NORMAL` on connection creation.
- [x] Create `system_status` table (AC: 3)
  - [x] Add SQL query `CREATE TABLE IF NOT EXISTS system_status (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP);` to `db/models.py`.
  - [x] Implement automated migration run at startup.
- [x] Update imports and main.py (AC: 1)
  - [x] Update import paths in all relocated Python files.
  - [x] Update `knowledge_mcp/main.py` to match new import paths.
  - [x] Verify CLI functionality with the MCP server start command.

## Dev Notes

### Structure Reorganization
For imports to resolve cleanly, all new subdirectories must contain an `__init__.py` file. In the root `main.py`, imports change from:
```python
from .db import Database
# to:
from .db.models import Database
```

### SQLite WAL Configuration
WAL (Write-Ahead Logging) is critical because the daemon process and MCP client process will interact with the same database file concurrently.
Example implementation of `get_db_connection` in `db/connection.py`:
```python
import sqlite3
import sqlite_vec

def get_db_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn
```

### References

- [Architectural Invariant on WAL and Read-Only: ARCHITECTURE-SPINE.md#AD-1](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md#L29-L37)
- [Database Schema: solution_design.md#2.2.-Table-system_status](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/solution_design.md#L133-L145)

## Dev Agent Record

### Agent Model Used
Antigravity (Google DeepMind)

### Debug Log References
None

### Completion Notes List
- Completed package structure reorganization: files moved into subdirectories `core`, `db`, and `mcp`.
- Implemented WAL mode and read-only connection logic in `db/connection.py`.
- Added `system_status` table and automatic migration runner.
- All imports updated and verified across the test suite.

### File List
- Created: `knowledge_mcp/core/__init__.py`
- Created: `knowledge_mcp/db/__init__.py`
- Created: `knowledge_mcp/mcp/__init__.py`
- Created: `knowledge_mcp/db/connection.py`
- Moved/Modified: `knowledge_mcp/core/code_parser.py` (from `code_parser.py`)
- Moved/Modified: `knowledge_mcp/core/markdown_parser.py` (from `markdown_parser.py`)
- Moved/Modified: `knowledge_mcp/core/embeddings.py` (from `embeddings.py`)
- Moved/Modified: `knowledge_mcp/core/indexer.py` (from `indexer.py`)
- Moved/Modified: `knowledge_mcp/db/models.py` (from `db.py`)
- Moved/Modified: `knowledge_mcp/mcp/server.py` (from `server.py`)
- Modified: `knowledge_mcp/main.py`
- Modified root test files (`_test_*.py` and `scratch/*.py`) to correct imports.

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build passes (Ruff check clean).
- [x] Git commit created.

### Code Review Summary
- **Style/PEP 8:** Clean, line limit respected.
- **SQLite Concurrency:** Connection uses WAL mode and synchronous=NORMAL.
- **SQL Injection Safety:** Parameters passed via placeholders (`?`).
- **Read-Only Mode:** Implemented correctly for stdio MCP server.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Methodology
- Automated integration testing via `_test_mcp_client.py` and `_test_delete_repo.py`.
- Manual verification of SQLite WAL and schema initialization.

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | Folder reorganization & import resolution | Ran MCP server handshake and search tool | ✅ PASS |
| AC 2 | SQLite WAL mode & connection settings | Checked WAL configuration & read-only connections | ✅ PASS |
| AC 3 | `system_status` migration table | Verified table creation & seed values in DB | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-06-30 | 1.0.0 | Dev: Reorganized folders, setup WAL & system_status, fixed imports | James |
| 2026-06-30 | 1.0.1 | CR: Verified imports, Python standards compliance, build/test PASS | CodeReview |
| 2026-06-30 | 1.0.2 | QA: Verified AC 1, 2, 3 with client and cascade tests. Status -> Done | Quinn |
