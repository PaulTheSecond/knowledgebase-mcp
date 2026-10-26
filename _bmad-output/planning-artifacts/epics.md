---
stepsCompleted:
  - "Requirements Extraction"
  - "Epic Design"
  - "Story Generation"
  - "Final Validation"
inputDocuments:
  - "c:/Repos/knowledgebase-mcp/_bmad-output/planning-artifacts/prds/prd-knowledgebase-mcp-2026-06-30/prd.md"
  - "c:/Repos/knowledgebase-mcp/_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md"
  - "c:/Repos/knowledgebase-mcp/_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/solution_design.md"
  - "c:/Repos/knowledgebase-mcp/docs/spec_autonomous_sync.md"
  - "c:/Repos/knowledgebase-mcp/docs/adr-004-autonomous-sync-triggers-credentials.md"
  - "c:/Repos/knowledgebase-mcp/docs/adr-005-web-dashboard-architecture.md"
  - "c:/Repos/knowledgebase-mcp/docs/roadmap_autonomous_sync.md"
---

# knowledgebase-mcp - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for knowledgebase-mcp, decomposing requirements from the PRD, UX Design, and Architecture invariants into implementable stories.

## Requirements Inventory

### Functional Requirements

* **FR-1:** SQLite Job Queue (table `sync_queue`: id, repo_id, status, attempts, error_message, created_at, updated_at).
* **FR-2:** FIFO Worker Processing (pop by `created_at` ASC in `pending` status, update to `processing`).
* **FR-3:** Execution Idempotency Guarantee (compare Git SHA against latest indexed SHA in the database).
* **FR-4:** Worker Error Halting (after 3 attempts with exponential backoff of 1, 5, 15 minutes, worker records `halted` status and error in `system_status`, halting processing).
* **FR-5:** Containerized Git Client (clone/update in `/data/clones/{repo_id}` utilizing mounted SSH keys or environment PATs).
* **FR-6:** Cryptographic Webhook Validation (HMAC-SHA256 signature for GitHub/Gitea, X-Gitlab-Token for GitLab; 401 Unauthorized on mismatch).
* **FR-7:** Cron Scheduler (periodic execution of sync tasks according to Cron expressions in `config.yaml`).
* **FR-8:** Lightweight SPA Frontend (clean HTML5, Vanilla JS, Vanilla CSS, served from FastAPI).
* **FR-9:** Basic Auth Protection (dashboard credentials; if unset, requests are rejected with 403 Forbidden).
* **FR-10:** Read-Only Queue Viewer in Web UI (display statuses, attempt counts, errors).
* **FR-11:** Reactive WebSocket Telemetry (real-time repository metrics, progress bars, task queue states).
* **FR-12:** Browser Chat UI (text input, history, auto-scroll, Markdown rendering, and code syntax highlighting).
* **FR-13:** Hybrid Context Retrieval (execute `search_chunks_hybrid`, select top 5 chunks, assemble system prompt).
* **FR-14:** OpenAI-Compatible Client (local or remote endpoint, configurable models and keys).
* **FR-15:** Character/Token SSE Streaming for RAG Chat.
* **FR-16:** Source Citations (clickable file tags with line ranges beneath assistant responses).

### Non-Functional Requirements

* **NFR-1 (Data Locality):** Cloned repositories and cache storage reside strictly within local container volumes (`/data/clones/`). No code or metadata is sent to external cloud services.
* **NFR-2 (Resilience):** Sync failures do not disrupt concurrent read-only search operations over existing data.
* **NFR-3 (RAM Limits):** Background indexing and vectorization enforce sub-batch limits and garbage collection to prevent Out Of Memory (OOM).

### Additional Requirements

* **AD-1 (WAL Mode & Read-Only MCP):** SQLite WAL mode (`PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL`). The MCP process operates strictly in Read-Only mode. All mutations delegate to the Daemon process via internal HTTP REST calls.
* **AD-2 (Global Status):** Worker state persists in table `system_status` (keys `worker_state`, `last_error`).
* **AD-3 (Temporary Tokens for WebSockets/SSE):** HTTP Basic Auth protects REST endpoints. WebSocket and SSE connections authenticate via short-lived signed tokens in query parameters `?token=...`.
* **AD-4 (In-Memory Git Token Security):** Git PAT tokens are never written to disk in `.git/config`. They are passed in memory via `http.extraHeader` or `GIT_ASKPASS`.
* **REORG (Directory Structure Modularization):** Split package into modular subpackages: `core/`, `db/`, `mcp/`, `daemon/`, and `dashboard/`.

### UX Design Requirements

* **UX-DR-1:** Responsive dashboard with dark theme support and clean layout.
* **UX-DR-2:** Read-only task queue table (repository, status, attempts, error messages).
* **UX-DR-3:** Real-time indexing progress bars over WebSockets.
* **UX-DR-4:** RAG chat dialog with message history, auto-scrolling, and code highlighting.
* **UX-DR-5:** Smooth SSE token streaming with a loading indicator ("model is thinking...").
* **UX-DR-6:** Clickable source citation tags with filenames and line ranges below answers.

---

## Epic List

### Epic 1: Autonomous Background Sync Engine
Background autonomous synchronization engine powered by SQLite. Handles webhook ingestion and cryptographic validation, Cron scheduling, FIFO task queue processing with exponential error backoff, secure private Git cloning (in-memory SSH/PAT), and Git SHA idempotency.
* **FRs covered:** FR-1, FR-2, FR-3, FR-4, FR-5, FR-6, FR-7, NFR-1, NFR-2, NFR-3, AD-1, AD-2, AD-4, REORG

### Epic 2: Web Dashboard & Real-Time Monitoring
Web dashboard with static asset delivery from FastAPI, protected by Basic Auth and session tokens. Provides real-time queue inspection and live WebSocket telemetry for repository statuses and indexing progress bars.
* **FRs covered:** FR-8, FR-9, FR-10, FR-11, AD-3, UX-DR-1, UX-DR-2, UX-DR-3

### Epic 3: Interactive RAG Chat & LLM Integration
Embedded conversational interface for querying the knowledge base directly from the dashboard. Performs hybrid search for top 5 chunks, constructs context prompts, communicates with OpenAI-compatible inference servers, streams tokens over SSE, and renders clickable source references.
* **FRs covered:** FR-12, FR-13, FR-14, FR-15, FR-16, UX-DR-4, UX-DR-5, UX-DR-6

---

## Epic 1: Autonomous Background Sync Engine

### Story 1.1: Package Reorganization, DB Initialization, and WAL Mode

As a developer,
I want to split the flat codebase into modular subpackages (core, db, mcp) and enforce SQLite WAL mode,
So that the codebase is ready for the web daemon without concurrent read/write lock contention.

**Acceptance Criteria:**

**Given** the existing repository structure,
**When** files are organized into `core/`, `db/`, `mcp/` and tests run,
**Then** all imports resolve cleanly and the MCP server functions normally.
**And** every SQLite connection executes `PRAGMA journal_mode = WAL;` and `PRAGMA synchronous = NORMAL;`.
**And** the `system_status` table is initialized for tracking worker health.

### Story 1.2: Git Manager Implementation with Secure Sessions

As a system administrator,
I want to clone and update private repositories securely without storing credentials on disk,
So that secrets (PATs) cannot leak if container volumes are inspected.

**Acceptance Criteria:**

**Given** a private repository with PAT authentication (in environment variables),
**When** the clone or fetch method executes,
**Then** `git clone` / `git pull` succeeds using in-memory parameters `-c http.extraHeader="Authorization: Basic ..."`.
**And** the `.git/config` file on disk contains only the clean repository URL without credentials.
**And** SSH authentication utilizes mounted private keys with `StrictHostKeyChecking=no`.

### Story 1.3: Background Worker and SQLite FIFO Queue

As a developer,
I want synchronization tasks stored in a durable database queue and processed sequentially,
So that codebase modifications reflect reliably in the knowledge base across restarts.

**Acceptance Criteria:**

**Given** an SQLite database with table `sync_queue`,
**When** multiple sync tasks are queued,
**Then** the background worker consumes the oldest `pending` task (`created_at` ASC), transitions it to `processing`, and initiates updates.
**And** before indexing, the worker validates the current Git commit SHA; matching SHAs complete in milliseconds (idempotency).

### Story 1.4: Error Backoff and Worker Halting

As an administrator,
I want the worker to retry failed tasks with exponential backoff and halt on persistent failures,
So that the system avoids infinite retry loops during network outages and alerts me to errors.

**Acceptance Criteria:**

**Given** a task that failed with an error (e.g. transient network timeout during `git pull`),
**When** the worker retries,
**Then** it performs up to 3 retries with exponential backoff (1m, 5m, 15m).
**And** upon exceeding 3 attempts, the task transitions to `failed` and logs the error in `error_message`.
**And** the worker writes `worker_state = halted` into `system_status` and pauses subsequent task processing.

### Story 1.5: HMAC Webhook Validator and Cron Scheduler

As a developer,
I want tasks enqueued automatically via webhooks or Cron schedules,
So that I never have to run manual synchronization commands.

**Acceptance Criteria:**

**Given** the FastAPI endpoint `/webhook/git`,
**When** an incoming POST request arrives with a signature from GitHub, Gitea, or GitLab,
**Then** the backend validates the HMAC-SHA256 signature using the configured secret; matching signatures enqueue the job into `sync_queue`, while mismatches return 401 Unauthorized.
**And** the background scheduler evaluates Cron expressions from `config.yaml` for each repository and enqueues jobs when triggered.

### Story 1.6: MCP Server Read-Only Adaptation

As an IDE AI assistant,
I want to query code through MCP tools in Read-Only mode and delegate mutations to the daemon,
So that my queries never trigger SQLite write locks against the background worker.

**Acceptance Criteria:**

**Given** a running MCP server,
**When** mutation tools like `knowledge_delete_repo` or `knowledge_sync_repo` are invoked,
**Then** the MCP server delegates the action via HTTP REST calls to the Daemon rather than writing to SQLite directly.
**And** all search tools (`knowledge_search`, `knowledge_find_symbol`) continue reading directly from SQLite in WAL mode.

---

## Epic 2: Web Dashboard & Real-Time Monitoring

### Story 2.1: Basic Auth and Short-Lived Session Tokens

As an administrator,
I want to protect dashboard access with Basic Auth and issue signed temporary tokens for WebSockets,
So that unauthorized parties cannot inspect repository structures or sync logs.

**Acceptance Criteria:**

**Given** the FastAPI application,
**When** accessing REST endpoints without valid authorization headers,
**Then** the backend returns `401 Unauthorized`.
**And** if `DASHBOARD_PASSWORD` is unset or empty, the backend returns `403 Forbidden` for all dashboard requests.
**And** upon successful Basic Auth, `/api/auth/token` returns a short-lived (60s) signed token for establishing WebSocket and SSE connections.

### Story 2.2: Modern SPA Dashboard Shell

As a developer,
I want an intuitive, responsive web interface with dark theme styling,
So that monitoring repository statuses is clean and efficient.

**Acceptance Criteria:**

**Given** the static SPA frontend in `dashboard/static/`,
**When** opening the dashboard in a browser,
**Then** the page renders with a modern dark theme and responsive layout.
**And** the navigation bar allows switching between "Repositories", "Sync Queue", and "RAG Chat" tabs without full-page reloads.

### Story 2.3: REST Telemetry API & Queue View

As a developer,
I want to see connected repositories with metrics and current queue tasks in a table,
So that I can quickly verify the volume and health of indexed code.

**Acceptance Criteria:**

**Given** `GET /api/status` returning repository lists with counts of files, chunks, vectors, and worker states,
**When** navigating to "Repositories",
**Then** the UI renders a responsive card grid with repository statistics.
**And** navigating to "Sync Queue" calls `GET /api/queue` and displays a read-only table of tasks (status, attempts, created date, error messages).

### Story 2.4: Real-Time WebSocket Telemetry

As a developer,
I want to see live indexing progress bars and task status updates without manual page refreshes,
So that I can monitor sync progress in real time.

**Acceptance Criteria:**

**Given** the WebSocket endpoint `/ws/status`,
**When** the frontend connects with a valid temporary token (`?token=<TOKEN>`),
**Then** the backend validates the token and establishes the session (invalid tokens close with code 4001).
**And** whenever worker state or indexing progress updates, the backend pushes JSON events over the socket, dynamically updating repository cards and progress bars (0-100%).

---

## Epic 3: Interactive RAG Chat & LLM Integration

### Story 3.1: LLM Gateway Integration & Hybrid Retrieval

As a developer,
I want the backend to retrieve relevant code chunks and dispatch them to LLMs via standard inference gateways,
So that the model generates answers grounded in the current codebase context.

**Acceptance Criteria:**

**Given** an inference endpoint configured via `LLM_BASE_URL` with key `LLM_API_KEY`,
**When** `POST /api/chat` receives a query and message history array (`messages`),
**Then** the backend checks `server.dashboard.chat.enabled`; if `false`, returns `403 Forbidden`.
**And** if enabled, executes `search_chunks_hybrid`, extracts the top 5 chunks, and injects them into the system prompt with file paths.
**And** dispatches the prompt and conversation history to the model, returning the assistant response.

### Story 3.2: SSE Streaming Backend

As a developer,
I want to stream tokens as they are generated by the model,
So that I do not experience 10-15 second delays waiting for complete responses.

**Acceptance Criteria:**

**Given** `POST /api/chat/stream` authorized via temporary query token,
**When** the backend receives streaming tokens from the inference endpoint (`stream=true`),
**Then** it forwards them via Server-Sent Events (`text/event-stream`) in real time.
**And** upon stream completion, emits a final payload containing source citation metadata (paths and line ranges).

### Story 3.3: Web UI Conversational Chat Interface

As a developer,
I want to interact with the knowledge base via an in-browser chat interface with message history and auto-scrolling,
So that testing RAG search is intuitive and fast.

**Acceptance Criteria:**

**Given** the "RAG Chat" tab in the dashboard,
**When** the dashboard loads,
**Then** the frontend evaluates the chat feature flag; if disabled, the tab is hidden from navigation.
**And** when enabled, submitting a message appends it to client-persisted conversation history and sends the payload to the backend.
**And** a thinking indicator renders during retrieval, and tokens stream smoothly with automatic downward scrolling.

### Story 3.4: Markdown Rendering, Syntax Highlighting & Citations

As a developer,
I want assistant answers rendered with rich Markdown, code syntax highlighting, and clickable file citations,
So that I can read formatted code and inspect referenced files directly.

**Acceptance Criteria:**

**Given** a completed SSE answer stream,
**When** the frontend renders the response,
**Then** it parses Markdown and highlights code blocks with language labels and copy buttons.
**And** beneath each response, renders clickable source tags with file paths and line ranges (e.g. `db.py#L12-45`), opening the file inspector at those exact lines.
