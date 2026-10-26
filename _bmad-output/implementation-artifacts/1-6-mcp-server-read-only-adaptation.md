# Story 1.6: MCP Server Read-Only Adaptation

Status: done

## Story

As an IDE AI assistant,
I want to execute code searches through MCP tools in Read-Only mode and delegate mutating operations to the daemon,
so that my actions never trigger SQLite locking conflicts with the background worker.

## Acceptance Criteria

1. **Given** a running MCP server,
   **When** I invoke repository deletion (`knowledge_delete_repo`) or synchronization (`knowledge_sync_repo`),
   **Then** the MCP server delegates via an HTTP request to the Daemon process rather than mutating SQLite tables directly.
2. **And** all search tools (`knowledge_search`, `knowledge_find_symbol`) continue to query SQLite directly in WAL mode.

## Tasks / Subtasks

- [x] Modify repository deletion tool (AC: 1)
  - [x] Update `knowledge_delete_repo` handler in `mcp/server.py`.
  - [x] Delegate via HTTP request `DELETE /api/repo/{repo_id}` to the Daemon process instead of calling `db.delete_repo`.
- [x] Modify synchronization tool (AC: 1)
  - [x] Update `knowledge_sync_repo` handler in `mcp/server.py`.
  - [x] Send request `POST /api/sync/{repo_id}` to the Daemon process (enqueuing the task in the DB queue).
- [x] Validate Read-Only connection status (AC: 2)
  - [x] Ensure the MCP process initializes SQLite in Read-Only mode (`KnowledgeDB(db_path, read_only=True)`).

## Dev Notes

In `sqlite3`, Read-Only mode is enforced via URI connections:
```python
import sqlite3
conn = sqlite3.connect("file:/data/knowledge.db?mode=ro", uri=True)
```
This guarantees the MCP process physically cannot issue write commands, eliminating database corruption and concurrency lock risks.
Daemon mutation endpoints:
- `POST http://127.0.0.1:8000/api/sync/{repo_id}`
- `DELETE http://127.0.0.1:8000/api/repo/{repo_id}`

### References

- [Read-Only MCP Architectural Invariant: ARCHITECTURE-SPINE.md#AD-1](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md#L29-L37)
- [Security Review: review-spine.md](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/review-spine.md)

## Dev Agent Record

### Agent Model Used
Antigravity (Claude Opus 4.6)

### Debug Log References
None

### Completion Notes List
- Updated `_handle_delete_repo` to delegate via `DELETE /api/repo/{repo_id}` over HTTP.
- Updated `_handle_sync_repo` to delegate via `POST /api/sync/{repo_id}` over HTTP.
- Removed `repo_path` from `knowledge_sync_repo` schema (path is resolved by the daemon via `config.yaml`).
- Confirmed: `start_mcp_server` connects to SQLite with `KnowledgeDB(db_path, read_only=True)`.
- All 19 project tests passing cleanly.

### File List
- Modified: `knowledge_mcp/mcp/server.py`

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build/Linter passes (Ruff check clean).
- [x] Full test suite passes (19/19).

### Code Review Summary
- **Read-Only Enforcement:** MCP server no longer performs direct SQLite mutations.
- **HTTP Delegation:** All write operations (delete, sync) are safely routed through Daemon APIs.
- **Schema Update:** `repo_path` removed from sync schema appropriately as the daemon resolves paths from configuration.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | Mutations delegated to Daemon API | delete_repo → DELETE /api/repo/{id}, sync_repo → POST /api/sync/{id} | ✅ PASS |
| AC 2 | Search tools remain Read-Only | `KnowledgeDB(db_path, read_only=True)` confirmed active | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-01 | 0.1.0 | Dev: Started Story 1.6 implementation | James |
| 2026-07-01 | 1.0.0 | Dev: Delegated delete_repo and sync_repo to Daemon API | James |
| 2026-07-01 | 1.0.1 | CR: Verified Read-Only enforcement and HTTP delegation. PASS | CodeReview |
| 2026-07-01 | 1.0.2 | QA: All 19 tests pass. Status → Done | Quinn |
