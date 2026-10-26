---
name: Autonomous Sync & Dashboard Architecture
type: architecture-spine
purpose: build-substrate
altitude: feature
paradigm: process-isolated shared-db
scope: Governs the background sync worker, SQLite job queue, webhook validator, and dashboard Web UI with RAG chat.
status: final
created: 2026-06-30
updated: 2026-10-26
binds: ["v1.3", "v1.4", "v1.5", "v1.6", "v1.9", "v1.10", "FR-1", "FR-2", "FR-3", "FR-4", "FR-5", "FR-6", "FR-7", "FR-8", "FR-9", "FR-10", "FR-11", "FR-12", "FR-13", "FR-14", "FR-15", "FR-16", "AD-6", "AD-7"]
sources: ["_bmad-output/planning-artifacts/prds/prd-knowledgebase-mcp-2026-06-30/prd.md"]
companions: ["_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/solution_design.md"]
---

# Architecture Spine — Autonomous Sync & Dashboard Module

## Design Paradigm

Architectural Paradigm: **Process-Isolated Shared-Database**.
The system is divided into two independent processes sharing a common SQLite database file:
1. **Daemon Process (Web Server and Background Worker):** Continuously running inside the container; serves the web interface, accepts incoming Git webhooks, and executes FIFO synchronization tasks sequentially.
2. **MCP Process (STDIO Interface):** Ephemeral process launched by external clients (e.g., VS Code / Cursor / Claude Code) for instant search and graph query resolution.

---

## Invariants & Rules

### AD-1 — SQLite Database Access Mode
* **Binds:** `server.py`, `db.py`, `indexer.py`, `main.py`
* **Prevents:** Deadlocks (`database is locked`) when the background worker writes while concurrent MCP search clients read.
* **Rule:** 
  1. All SQLite database connections must initialize in **WAL (Write-Ahead Logging)** mode via PRAGMA directives:
     ```sql
     PRAGMA journal_mode = WAL;
     PRAGMA synchronous = NORMAL;
     ```
     This ensures non-blocking concurrent reads across multiple processes and threads during write transactions.
  2. **Write Isolation:** Mutating write operations (writing chunks, symbols, purging repositories) are strictly restricted to the Daemon process. The stdio MCP process must operate strictly in **Read-Only** mode. Mutation requests (such as `knowledge_delete_repo`) must be delegated to the Daemon process via internal HTTP REST calls rather than executed by the MCP process directly.

### AD-2 — Task Queue Lifecycle and Backoff
* **Binds:** `FR-1`, `FR-2`, `FR-3`, `FR-4` (queue and worker)
* **Prevents:** Sync task loss during container restarts, CPU starvation under concurrent builds, and non-idempotent re-runs.
* **Rule:** Tasks reside in the SQLite `sync_queue` table. The worker fetches tasks in strict FIFO order (`created_at` ASC). Worker health and fatal errors persist in `system_status`. On failure, the worker performs up to 3 retry attempts with exponential backoff (1, 5, 15 minutes). If all retries fail, it writes `halted` to `system_status` and halts subsequent execution. Before executing a task, the worker checks the repository Git SHA; matching SHAs complete immediately with zero redundant computation.

### AD-3 — Authentication and Real-Time Transport
* **Binds:** `FR-9`, `FR-10`, `FR-11`, `FR-12`, `FR-15` (dashboard and RAG chat)
* **Prevents:** Unauthorized access to proprietary code, and frontend UI lockup during slow LLM generation.
* **Rule:** 
  1. The dashboard uses HTTP Basic Auth for REST endpoints. WebSocket (telemetry) and Server-Sent Events (RAG chat) connections authenticate using short-lived signed tokens passed in the query parameter `?token=...`.
  2. **Chat Feature Flag:** The chat subsystem must be entirely toggleable via `server.dashboard.chat.enabled` in `config.yaml`. When `false`, chat endpoints immediately return `403 Forbidden`.
  3. **Stateless Dialogue State:** The backend is strictly stateless regarding conversation history. Dialogue history resides in client memory/storage and is transmitted in the body of `POST /api/chat`. Telemetry streams over WebSockets; chat tokens stream over SSE.

### AD-4 — Git Credential Security in Memory
* **Binds:** `FR-1`, `FR-5` (private repository synchronization)
* **Prevents:** Compromising Personal Access Tokens (PATs) in cloned `.git/config` files on container volumes.
* **Rule:** Git tokens must never be written to `.git/config` on disk. When executing `git clone`, `fetch`, or `pull`, tokens must be injected in-memory using Git configuration directives: `http.extraHeader="Authorization: Basic <base64>"` or `GIT_ASKPASS`.

### AD-5 — Document Inspector & Chat Formatting
* **Binds:** `FR-15`, `FR-16`, `ADR-007`
* **Prevents:** 404 errors when citing local files, conversation context loss on page reload, and Angular DomSanitizer UI button sanitization.
* **Rule:** 
  1. **File Path Resolution:** The `/api/chat/file` endpoint must resolve paths across all configured repository roots (`local_path`, `clones_dir`, `project_dir`) and support `full_file=true`.
  2. **Inspector and Splitter:** Document inspection takes place in a dedicated right-hand panel (`app-doc-viewer`) with IDE tabs and a resizable splitter defaulting to 45% width (persisting in localStorage).
  3. **Rendering & Safety:** LLM responses render Prism syntax highlighting and Mermaid SVG diagrams; interactive elements (copy buttons) pass via `sanitizer.bypassSecurityTrustHtml()`.

### AD-6 — LLM Provider Profiles & Protocol Adaptation
* **Binds:** `v1.9`, `daemon/routes/settings.py`, `daemon/routes/chat.py`
* **Prevents:** Hardcoded gateway lock-in, connection failures when API keys are absent for local models, and UI state divergence.
* **Rule:** 
  1. **Centralized Configuration:** Model and provider settings reside strictly inside the Dashboard Settings interface, keeping the RAG chat focused on code inspection.
  2. **Universal Protocols:** Native compatibility with OpenAI Chat Completions (`/v1/chat/completions`), Ollama API, and local inference proxies (vLLM, LM Studio, LocalAI, OmniRoute, LiteLLM). Passwordless operation for local endpoints is a first-class citizen.
  3. **Ownership and Isolation:** Profiles persist with ownership metadata (`owner`, default `'admin'`) and an `is_shared: bool` flag, guaranteeing compatibility with multi-user access without data migrations.

### AD-7 — Multi-User Authentication & RBAC
* **Binds:** `v1.10`, `daemon/auth.py`, `daemon/routes/users.py`, `db/models.py`
* **Prevents:** Unauthorized access to sensitive codebases in multi-developer teams, credential sharing, and user context bleed.
* **Rule:** 
  1. **User Identity Storage:** Authentication transitions from the single-password `DASHBOARD_PASSWORD` to a `users` table with Argon2/bcrypt password hashing and session JWT tokens.
  2. **Role Hierarchy:** Three standard roles: `Admin` (full system and repository control), `Developer` (querying, triggering sync, managing personal LLM profiles), `Viewer` (read-only search and documentation exploration).
  3. **State Partitioning:** Personal LLM profiles, conversation histories, and interface preferences are partitioned by `user_id`.

---

## Consistency Conventions

| Area | Convention |
| :--- | :--- |
| **Table Names** | Plural `snake_case` (e.g., `sync_queue`, `system_status`, `llm_profiles`, `users`). |
| **API Format** | REST API adheres to JSON-API. Errors return `{"detail": "..."}`. |
| **Logging Streams** | All backend logging writes to `sys.stderr` with metadata `[TIMESTAMP] [LEVEL] [COMPONENT] Message`. `sys.stdout` is reserved strictly for MCP JSON-RPC. |
| **Token Versioning** | Signed Web UI temporary tokens use HMAC-SHA256 with a 60-second validity window. |

---

## Stack

| Technology | Version |
| :--- | :--- |
| **Python** | >= 3.11 |
| **FastAPI** | >= 0.100.0 |
| **Uvicorn** | >= 0.22.0 |
| **SQLite** | >= 3.38.0 |
| **sqlite-vec** | >= 0.1.0 |
| **Git CLI** | >= 2.34.0 |

---

## Structural Seed

```text
{project-root}/
  _bmad-output/
    planning-artifacts/
      architecture/         # Architectural schemas & Solution Design
  knowledge_mcp/
    __init__.py
    main.py                 # Single CLI entry point (commands: mcp / serve / preset)
    core/                   # Engine (parsers, chunking, vector indexing)
      __init__.py
      code_parser.py
      markdown_parser.py
      embeddings.py
      indexer.py
    db/                     # Persistence layer (transactions, migrations, WAL)
      __init__.py
      connection.py
      models.py
    mcp/                    # Stdio MCP Layer (Read-Only)
      __init__.py
      server.py
    daemon/                 # FastAPI process + queue worker
      __init__.py
      app.py
      auth.py
      worker.py
      routes/
        __init__.py
        webhooks.py
        chat.py
        status.py
        settings.py
    dashboard/              # Static frontend delivery
      static/
        css/
        js/
        index.html
```

---

## Deferred Architectural Decisions
* **External RDBMS Integration (Postgres/MySQL):** In v1.x, storage is strictly bounded to SQLite (with WAL mode and sqlite-vec). Distributed database backends are deferred to v2+.
* **Role-Based Access Control (RBAC):** Transitioned from indefinite deferral into planned milestone **v1.10** (invariant `AD-7`). In v1.3–v1.9, the system operates in single-user `admin` transitional mode with ownership tagging (`owner_user`).
