# Story 1.3: FIFO DB Queue & Worker Loop

Status: done

## Story

As a developer,
I want synchronization jobs to be stored in a resilient database queue and processed sequentially by a background worker,
so that codebase changes are reflected in the knowledge base without data loss from unexpected restarts.

## Acceptance Criteria

1. **Given** an SQLite database with a `sync_queue` table,
   **When** multiple synchronization tasks are queued,
   **Then** the background worker selects the oldest task in `pending` status (ordered ascending by `created_at`), sets its status to `processing`, and triggers codebase synchronization.
2. **And** before triggering indexing, the worker retrieves the current Git commit SHA. If it matches the previously indexed SHA, the task is marked as `completed` within milliseconds (idempotency).

## Tasks / Subtasks

- [x] Create DB migration for queue and commits tables (AC: 1)
  - [x] Add creation of `sync_queue` table in `db/models.py`.
  - [x] Create table `indexed_commits (repo_id TEXT PRIMARY KEY, git_sha TEXT, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP);`.
- [x] Implement worker polling loop (AC: 1)
  - [x] Create file `knowledge_mcp/daemon/worker.py`.
  - [x] Implement a `while True` loop with a 1-second queue polling interval.
  - [x] Implement FIFO task selection: `SELECT * FROM sync_queue WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1;`.
  - [x] Implement task status transition to `processing` within an atomic transaction.
- [x] Integrate with Git and indexer (AC: 1, 2)
  - [x] Retrieve repository config from `config.yaml` for the selected task.
  - [x] Execute code update via `core/git_manager.py` in `/data/clones/{repo_id}`.
  - [x] Retrieve current HEAD commit via `git rev-parse HEAD`.
- [x] Implement idempotency check using Git SHA (AC: 2)
  - [x] Check `indexed_commits` table for the last processed commit for this `repo_id`.
  - [x] If commits match, immediately update task status to `completed` and proceed to the next job.
  - [x] If commit changed, run indexer `indexer.py` (code parsing and embeddings).
  - [x] Upon successful completion, persist the new `git_sha` to `indexed_commits` and mark the task `completed`.

## Dev Notes

### FIFO Selection and SQLite Transactions
To prevent race conditions during job selection by the worker (especially when transitioning to multi-worker setups in the future), the status update must occur atomically within the transaction:
```python
def fetch_next_job(conn) -> dict:
    cursor = conn.cursor()
    cursor.execute("BEGIN IMMEDIATE TRANSACTION;")
    try:
        cursor.execute("SELECT id, repo_id, status FROM sync_queue WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1;")
        row = cursor.fetchone()
        if row:
            job_id, repo_id, status = row
            cursor.execute("UPDATE sync_queue SET status = 'processing', updated_at = CURRENT_TIMESTAMP WHERE id = ?;", (job_id,))
            conn.commit()
            return {"id": job_id, "repo_id": repo_id}
        conn.commit()
        return None
    except Exception as e:
        conn.rollback()
        raise e
```

### References

- [Queue Architectural Invariant: ARCHITECTURE-SPINE.md#AD-2](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md#L39-L43)
- [Commit SHA Idempotency Algorithm: solution_design.md#3.2.-Queue-Processing-and-Exponential-Backoff-Algorithm](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/solution_design.md#L165-L182)

## Dev Agent Record

### Agent Model Used
Antigravity (Google DeepMind)

### Debug Log References
None

### Completion Notes List
- Created DB migration `20260701_07_sync_queue_and_indexed_commits` for tables `sync_queue` and `indexed_commits` with status/timestamp indexing.
- Implemented `DaemonWorker` class in `knowledge_mcp/daemon/worker.py` with FIFO polling loop and worker state check (`halted`).
- Implemented atomic task fetch and status update in `fetch_next_job()` using `BEGIN IMMEDIATE` transactions.
- Integrated Git SHA comparison and idempotency check: when unchanged, indexing is skipped and the job completes immediately.
- Created unit tests in `tests/test_worker.py` with mocks for `GitManager` and `Indexer`, fully covering worker lifecycle and idempotency.

### File List
- Created: `knowledge_mcp/daemon/worker.py`
- Created: `knowledge_mcp/daemon/__init__.py`
- Created: `knowledge_mcp/core/config.py`
- Created: `tests/test_worker.py`
- Modified: `knowledge_mcp/db/models.py`

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build/Linter passes (Ruff check clean).
- [x] Unit tests pass successfully.

### Code Review Summary
- **Database Concurrency:** Atomicity of task selection is guaranteed by `BEGIN IMMEDIATE TRANSACTION` in SQLite WAL mode.
- **Idempotency:** SHA checkout prevents repeating slow vectorization steps.
- **Style/Standards:** Correct exception handling, logger redirected to stderr, types and clean imports.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Methodology
- Automated integration test run in `tests/test_worker.py`.
- Checked SQLite tables and schema migration for `sync_queue` and `indexed_commits`.

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | FIFO DB Queue & Worker status translation | Added queue entries, verified FIFO sorting and status updates | ✅ PASS |
| AC 2 | SHA comparison and Idempotency | Confirmed indexing is skipped and status set to completed if SHA is unchanged | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-01 | 0.1.0 | Dev: Started Story 1.3 implementation | James |
| 2026-07-01 | 1.0.0 | Dev: Implemented DaemonWorker, migration and tests | James |
| 2026-07-01 | 1.0.1 | CR: Reviewed SQLite transactions and idempotency logic. PASS | CodeReview |
| 2026-07-01 | 1.0.2 | QA: Verified FIFO and idempotency behavior. Status -> Done | Quinn |
