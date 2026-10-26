# Story 2.1: Basic Auth and Token Generation

Status: done

## Story

As an administrator,
I want to protect dashboard access using Basic Auth and generate secure temporary tokens for WebSockets,
so that unauthorized users cannot inspect my repository structure or execution logs.

## Acceptance Criteria

1. **Given** a FastAPI application,
   **When** I access REST endpoints (e.g., `/api/status`) without an `Authorization` header or with an invalid password,
   **Then** the backend returns `401 Unauthorized`.
2. **And** if environment variable `DASHBOARD_PASSWORD` is unset or empty, the backend strictly returns `403 Forbidden` for all dashboard requests.
3. **And** upon successful Basic Auth, endpoint `/api/auth/token` returns a short-lived (60s) cryptographically signed token for establishing WebSocket/SSE connections.

## Tasks / Subtasks

- [x] Implement authorization module (AC: 1, 2)
  - [x] Create file `knowledge_mcp/daemon/auth.py`.
  - [x] Write FastAPI Dependency to verify HTTP Basic Credentials.
  - [x] If password is empty or missing from environment (`DASHBOARD_PASSWORD`), raise `HTTPException(status_code=403, detail="Dashboard disabled")`.
- [x] Generate signed tokens for WS/SSE (AC: 3)
  - [x] Use cryptographic HMAC-SHA256 with salt to sign tokens without third-party dependencies.
  - [x] Endpoint `/api/auth/token` returns a token formatted as `timestamp.signature`.
- [x] Validate tokens (AC: 3)
  - [x] Implement token verification method. If expired (>60s) or signature invalid, reject connection.
- [x] Test authorization (AC: 1, 2)
  - [x] Write unit tests verifying empty password rejection and token expiration validation.

## Dev Notes

Example of token signing logic:
```python
import hmac
import hashlib
import time

SECRET_KEY = "system-secret-key"

def generate_signed_token(username: str) -> str:
    ts = str(int(time.time()))
    sig = hmac.new(SECRET_KEY.encode(), f"{username}:{ts}".encode(), hashlib.sha256).hexdigest()
    return f"{ts}.{sig}"

def verify_signed_token(token: str, username: str) -> bool:
    try:
        ts_str, sig = token.split(".", 1)
        if time.time() - int(ts_str) > 60:
            return False
        expected = hmac.new(SECRET_KEY.encode(), f"{username}:{ts_str}".encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)
    except Exception:
        return False
```

### References

- [Token Architectural Invariant: ARCHITECTURE-SPINE.md#AD-3](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md#L44-L50)

## Dev Agent Record

### Agent Model Used
Antigravity (Gemini 3.5 Flash)

### Debug Log References
None

### Completion Notes List
- **Knowledge MCP Research**: Conducted symbol and code exploration for auth mechanisms; confirmed no prior auth implementation existed and designed fresh secure module.
- Implemented `knowledge_mcp/daemon/auth.py` providing Basic Auth via `DASHBOARD_PASSWORD`. Returns `403 Forbidden` if password is unset.
- Implemented cryptographic signature for temporary (60s) tokens using standard library HMAC-SHA256, eliminating external dependencies.
- Added REST endpoints in `knowledge_mcp/main.py`: `/api/auth/token`, `/api/status`, `/api/sync/{repo_id}`, `/api/repo/{repo_id}`.
- Added 14 unit tests in `tests/test_auth.py` testing tokens, Basic Auth, and FastAPI endpoints with `TestClient`.

### File List
- [NEW] [auth.py](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/daemon/auth.py)
- [NEW] [test_auth.py](file:///c:/Repos/knowledgebase-mcp/tests/test_auth.py)
- [MODIFY] [main.py](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/main.py)
- [MODIFY] [server.py](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/mcp/server.py)

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build/Linter passes (Ruff check clean).
- [x] Unit tests pass successfully (14/14).

### Code Review Summary
- **Timing Attacks:** `hmac.compare_digest` ensures constant-time verification of credentials and signatures.
- **Dependency Safety:** Built with standard library `hmac` + `hashlib` avoiding dependencies.
- **403 Forbidden:** If `DASHBOARD_PASSWORD` is absent, requests are rejected immediately.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | Access control without Basic Auth | Test `test_api_unauthorized_with_wrong_pass` verifies 401 | ✅ PASS |
| AC 2 | DASHBOARD_PASSWORD not configured -> 403 | Test `test_api_disabled_by_default` verifies 403 | ✅ PASS |
| AC 3 | /api/auth/token generates short token | Test `test_api_get_token` verifies valid signed token issuance | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-02 | 0.1.0 | Dev: Started Story 2.1 implementation | James |
| 2026-07-02 | 1.0.0 | Dev: Implemented Basic Auth, HMAC Token generation & tests | James |
| 2026-07-02 | 1.0.1 | CR: Verified comparison safety and dependency avoidance. PASS | CodeReview |
| 2026-07-02 | 1.0.2 | QA: Verified auth API behaviors and tokens expiration. Status -> Done | Quinn |
