# Story 1.5: Webhook Validation and Cron Triggers

Status: done

## Story

As a developer,
I want synchronization jobs to be enqueued automatically via webhooks or Cron schedules,
so that I do not need to manually trigger indexing.

## Acceptance Criteria

1. **Given** FastAPI endpoint `/webhook/git`,
   **When** a signed POST request arrives from GitHub, Gitea, or GitLab,
   **Then** the backend validates the HMAC-SHA256 signature against the secret stored in environment variables. If valid, a task is added to `sync_queue`; if invalid, HTTP 401 Unauthorized is returned.
2. **And** a background scheduler (Cron) parses the schedule from `config.yaml` for each repository and automatically enqueues tasks into `sync_queue` upon trigger intervals.

## Tasks / Subtasks

- [x] Implement webhook endpoint (AC: 1)
  - [x] Create file `knowledge_mcp/daemon/routes/webhooks.py`.
  - [x] Implement endpoint `POST /webhook/git`.
  - [x] Add request payload parsing to extract repository details.
- [x] HMAC-SHA256 signature validation (AC: 1)
  - [x] Implement verification for GitHub `X-Hub-Signature-256` and Gitea `X-Gitea-Signature` headers using HMAC-SHA256 with secrets configured via `secret_env`.
  - [x] Implement GitLab token check via header `X-Gitlab-Token`.
  - [x] Return `401 Unauthorized` on missing or invalid headers.
- [x] Implement Cron scheduler (AC: 2)
  - [x] Use lightweight scheduler based on `threading.Timer` (zero external dependencies).
  - [x] Read `triggers.cron` per repository from `config.yaml`.
  - [x] Enqueue tasks into `sync_queue` when triggered.

## Dev Notes

Example of HMAC-SHA256 validation in Python:
```python
import hmac
import hashlib

def verify_signature(payload: bytes, secret: str, signature: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
```

Secret is resolved dynamically via the environment variable name defined in the config:
```yaml
secret_env: "YURTA_WEBHOOK_SECRET" # Environment variable name
```

### References

- [Validation Requirements: prd.md#4.2.-Webhook-and-Schedule-Processing](file:///_bmad-output/planning-artifacts/prds/prd-knowledgebase-mcp-2026-06-30/prd.md#L98-L105)

## Dev Agent Record

### Agent Model Used
Antigravity (Claude Opus 4.6)

### Debug Log References
None

### Completion Notes List
- Created `daemon/routes/webhooks.py` with endpoint `POST /webhook/git`.
- Implemented HMAC-SHA256 signature verification for GitHub/Gitea and token checking for GitLab.
- Added automatic repository lookup by clone URL from webhook payloads.
- Implemented `daemon/scheduler.py` — lightweight cron scheduler using `threading.Timer` without external dependencies.
- 12 unit tests passing: HMAC verification, cron parsing, config resolution, and scheduling.

### File List
- Created: `knowledge_mcp/daemon/routes/__init__.py`
- Created: `knowledge_mcp/daemon/routes/webhooks.py`
- Created: `knowledge_mcp/daemon/scheduler.py`
- Created: `tests/test_webhooks_and_cron.py`
- Modified: `data/config.yaml`

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build/Linter passes (Ruff check clean).
- [x] Unit tests pass successfully (12/12).

### Code Review Summary
- **HMAC:** Uses `hmac.compare_digest` for timing-attack-safe comparisons.
- **Cron:** Zero-dependency implementation with daemon threads.
- **Security:** Secrets extracted from environment variables rather than hardcoded.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | Webhook HMAC-SHA256 validation | Tests for valid, invalid, and prefix-stripped HMAC plus repo lookup | ✅ PASS |
| AC 2 | Cron scheduler reads config.yaml | Cron expression parsing and scheduler start/stop tests | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-01 | 0.1.0 | Dev: Started Story 1.5 implementation | James |
| 2026-07-01 | 1.0.0 | Dev: Implemented webhooks and cron scheduler | James |
| 2026-07-01 | 1.0.1 | CR: Reviewed HMAC and cron logic. PASS | CodeReview |
| 2026-07-01 | 1.0.2 | QA: Verified 12 tests pass. Status → Done | Quinn |
