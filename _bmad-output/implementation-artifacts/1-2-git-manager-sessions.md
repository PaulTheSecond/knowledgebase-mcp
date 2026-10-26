# Story 1.2: Git CLI Session Manager

Status: done

## Story

As a system administrator,
I want to securely clone and update private repositories without persisting authorization tokens to disk,
so that my credentials (PAT) are not leaked if the container filesystem is compromised.

## Acceptance Criteria

1. **Given** a private repository with PAT authorization (configured via environment variable),
   **When** a clone or update method is invoked,
   **Then** `git clone` / `git pull` executes successfully with the token injected in-memory via `-c http.extraHeader="Authorization: Basic ..."`.
2. **And** in the `.git/config` file on disk, the repository URL is stored clean without secrets.
3. **And** when authenticating over SSH, the mounted SSH key is used with StrictHostKeyChecking disabled.

## Tasks / Subtasks

- [x] Create Git manager module (AC: 1, 3)
  - [x] Create file `knowledge_mcp/core/git_manager.py`.
  - [x] Implement functions/methods for safe `git clone` and `git pull`.
- [x] Implement HTTPS PAT in-memory authentication (AC: 1, 2)
  - [x] Write token encoding helper: `base64("x-token-auth:" + GITHUB_PAT)`.
  - [x] Execute clone using `git -c http.extraHeader="Authorization: Basic <base64>" clone <url_without_creds> <path>`.
  - [x] Execute pull using `git -c http.extraHeader="Authorization: Basic <base64>" pull origin <branch>`.
  - [x] Verify that `.git/config` in the cloned directory contains no secrets.
- [x] Implement SSH authentication (AC: 3)
  - [x] Pass key via `git -c core.sshCommand="ssh -i <key_path> -o StrictHostKeyChecking=no"`.
- [x] Write tests (AC: 1, 3)
  - [x] Create mock unit tests in `tests/test_git_manager.py` to simulate Git commands with PAT and SSH keys.

## Dev Notes

### Secure Cloning via extraHeader
When using GitHub PAT, the header should contain username `x-token-auth` (or `token`) and the Base64-encoded token:
```python
import base64
import subprocess

def get_git_auth_args(token: str) -> list:
    creds = f"x-token-auth:{token}"
    b64_creds = base64.b64encode(creds.encode("utf-8")).decode("utf-8")
    return ["-c", f"http.extraHeader=Authorization: Basic {b64_creds}"]
```
Example clone command:
```python
auth_args = get_git_auth_args(pat_token)
cmd = ["git"] + auth_args + ["clone", "https://github.com/your-org/repo.git", dest_path]
subprocess.run(cmd, check=True)
```
This prevents token leakage to `.git/config` on disk.

### References

- [Git Security Architectural Invariant: ARCHITECTURE-SPINE.md#AD-4](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md#L49-L53)
- [Git Security Algorithm: solution_design.md#3.1.-Algorithm-for-Secure-Git-Interaction-(PAT-/-SSH)](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/solution_design.md#L150-L163)

## Dev Agent Record

### Agent Model Used
Antigravity (Google DeepMind)

### Debug Log References
None

### Completion Notes List
- Created `GitManager` class in `knowledge_mcp/core/git_manager.py` for safe Git repository interactions.
- Implemented HTTPS PAT authentication with `http.extraHeader` avoiding persistent credentials in `.git/config`.
- Implemented SSH authentication using `core.sshCommand` with disabled `StrictHostKeyChecking`.
- Added unit tests in `tests/test_git_manager.py` with `unittest.mock` covering all scenarios and credential masking.

### File List
- Created: `knowledge_mcp/core/git_manager.py`
- Created: `tests/test_git_manager.py`

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
- **Security:** Using HTTP headers in-memory ensures GITHUB_PAT is never persisted to `.git/config` on disk. Masking base64 credentials prevents accidental leaks in error logs.
- **SSH Auth:** Passing `-o StrictHostKeyChecking=no` via `core.sshCommand` successfully bypasses interactive prompts.
- **Imports:** Verified that `knowledge_mcp.core.git_manager` is imported cleanly without unused packages.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Methodology
- Ran mock unit tests covering all target methods in `tests/test_git_manager.py`.
- Verified masking functionality under simulated command failures.

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | Clone/pull via http.extraHeader PAT | Verified command args containing basic auth header | ✅ PASS |
| AC 2 | No secrets saved in `.git/config` | Checked that headers are basic-auth parameters (not in repo URL) | ✅ PASS |
| AC 3 | SSH auth with StrictHostKeyChecking=no | Verified core.sshCommand is correctly appended | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-01 | 0.1.0 | Dev: Started Story 1.2 implementation | James |
| 2026-07-01 | 1.0.0 | Dev: Completed GitManager and unit tests | James |
| 2026-07-01 | 1.0.1 | CR: Checked security, standards and code style. PASS | CodeReview |
| 2026-07-01 | 1.0.2 | QA: Verified mock execution and mask patterns. Status -> Done | Quinn |
