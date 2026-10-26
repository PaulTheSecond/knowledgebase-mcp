# Story 1.4: Error Retries and Worker Halts

Status: done

## Story

As an administrator,
I want the worker to automatically retry failed jobs with exponential backoff and halt the queue on critical errors,
so that the system does not enter infinite failure loops during network outages and I can inspect the error.

## Acceptance Criteria

1. **Given** a queued job that fails with an exception (e.g., network failure during `git pull`),
   **When** the worker processes it again,
   **Then** it performs up to 3 retries with exponential backoff delays (1 minute, 5 minutes, 15 minutes).
2. **And** upon exceeding 3 attempts, the job is transitioned to `failed` status and the exception traceback is stored in `error_message`.
3. **And** the worker writes `worker_state = halted` and the error details into `system_status`, immediately pausing subsequent job execution.

## Tasks / Subtasks

- [x] Implement attempt counting and exponential backoff (AC: 1)
  - [x] Update error handling logic in `knowledge_mcp/daemon/worker.py`.
  - [x] If a job raises an exception, increment `attempts`.
  - [x] If `attempts < 3`, reset status to `pending` and set target retry timestamp based on `attempts`.
- [x] Worker halt on maximum attempts exceeded (AC: 2, 3)
  - [x] If `attempts >= 3`, transition job to `failed` and record exception in `error_message`.
  - [x] Record `worker_state` with value `halted` and `last_error` in `system_status`.
  - [x] Ensure the worker loop halts job fetching when `worker_state == halted`.
- [x] Unit tests for error recovery (AC: 1, 2)
  - [x] Add tests verifying attempt incrementation, retry scheduling, and transition to `halted`.

## Dev Notes

The worker must check `system_status` before each polling cycle. If status is `halted`, it must not fetch any new jobs from `sync_queue`.
Backoff schedule:
- Attempt 1: `delay = 60` seconds.
- Attempt 2: `delay = 300` seconds.
- Attempt 3: `delay = 900` seconds.
Jobs are queried using `status = 'pending' AND (run_at IS NULL OR run_at <= CURRENT_TIMESTAMP)` to respect the delayed execution timestamp.

### References

- [Backoff Specification: solution_design.md#3.2.-Queue-Processing-and-Exponential-Backoff-Algorithm](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/solution_design.md#L165-L182)

## Dev Agent Record

### Agent Model Used
Antigravity (Claude Opus 4.6)

### Debug Log References
None

### Completion Notes List
- Added DB migration `20260701_08_sync_queue_run_at` introducing the `run_at` column in `sync_queue`.
- Updated `fetch_next_job()` to ignore jobs scheduled for future execution (`run_at`).
- Added `reschedule_job()` helper in `KnowledgeDB` to return jobs to `pending` with exponential delay.
- Updated `run_job()` in `DaemonWorker` to handle backoff (60/300/900s) and halt on 3 failed attempts.
- Added 2 unit tests: `test_retry_backoff_reschedules_with_delay` and `test_halted_after_max_attempts`.

### File List
- Modified: `knowledge_mcp/db/models.py`
- Modified: `knowledge_mcp/daemon/worker.py`
- Modified: `tests/test_worker.py`

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build/Linter passes (Ruff check clean).
- [x] Unit tests pass successfully (3/3).

### Code Review Summary
- **Retry Logic:** Backoff delays `{1: 60, 2: 300, 3: 900}` match the specification exactly.
- **Halted State:** Upon `attempts >= max_attempts`, the worker sets `halted` in `system_status` and marks the job as `failed`.
- **run_at Filter:** `fetch_next_job` correctly filters tasks by `run_at IS NULL OR run_at <= datetime('now', 'localtime')`.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Methodology
- Automated test runs in `tests/test_worker.py`.
- Manual verification of SQL schema migration and backoff intervals.

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | Exponential backoff (60/300/900s) | test_retry_backoff: job returned to pending with future run_at | ✅ PASS |
| AC 2 | Job marked failed after 3 attempts | test_halted: status=failed, attempts=3, error_message populated | ✅ PASS |
| AC 3 | Worker halted when attempts exhausted | test_halted: system_status=halted, last_error populated | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-01 | 0.1.0 | Dev: Started Story 1.4 implementation | James |
| 2026-07-01 | 1.0.0 | Dev: Implemented retry backoff, halted state, migration and tests | James |
| 2026-07-01 | 1.0.1 | CR: Reviewed retry logic and run_at filter. PASS | CodeReview |
| 2026-07-01 | 1.0.2 | QA: Verified all 3 AC. Status → Done | Quinn |
