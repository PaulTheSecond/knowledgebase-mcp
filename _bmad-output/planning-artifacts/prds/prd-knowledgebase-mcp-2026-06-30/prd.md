---
title: Autonomous Sync & Dashboard Module
status: final
created: 2026-06-30
updated: 2026-10-26
---

# PRD: Autonomous Sync & Dashboard Module (v1.3 - v1.6)

## 0. Document Purpose

This document defines the product requirements for autonomous background synchronization, incoming trigger validation (webhooks), secure credential storage, and the Web UI Dashboard for the `knowledge-mcp` project.

It serves developers implementing the features and system architects verifying quality standards. This PRD builds on [spec_autonomous_sync.md](file:///c:/Repos/knowledgebase-mcp/docs/spec_autonomous_sync.md) and architectural decisions [ADR-004](file:///c:/Repos/knowledgebase-mcp/docs/adr-004-autonomous-sync-triggers-credentials.md) and [ADR-005](file:///c:/Repos/knowledgebase-mcp/docs/adr-005-web-dashboard-architecture.md).

---

## 1. Vision

In `knowledge-mcp` v1.2, synchronizing the knowledge base required manual commands, causing context drift and hallucinated responses in downstream AI assistants.

This milestone transforms the knowledge base into an autonomous system that keeps itself continuously synchronized with Git events. Developers push their code, and the container pulls delta changes securely via SSH/PAT, parses AST syntax trees incrementally, and updates vector embeddings.

Users gain a Web UI Dashboard for visual monitoring of indexing health, SQLite job queues, and an embedded RAG chat for immediate semantic verification via local models and inference gateways.

---

## 2. Personas and User Journeys

### 2.1. Jobs To Be Done
* **As a developer**, I want my IDE assistant to always possess fresh context without requiring manual re-indexing commands.
* **As a team lead**, I want a web dashboard to verify indexing status, inspect pending queue jobs, and troubleshoot parse errors.
* **As a developer**, I want to test semantic search across the entire codebase directly in my browser using powerful local models.

### 2.2. User Journeys

* **UJ-1: Alex Observes Automatic Knowledge Base Updates Post-Push**
  * **Context:** Alex (Senior Developer) pushes code to `main` in `Yurta.Core.Lib`.
  * **Precondition:** MCP server is running; credentials and webhooks are configured.
  * **Flow:**
    1. Alex pushes changes: `git push origin main`.
    2. Git server dispatches a webhook to `/webhook/git`.
    3. Webhook validator verifies the HMAC-SHA256 signature and enqueues the job.
    4. Background worker consumes the FIFO job, executes delta updates in `/data/clones/Yurta.Core.Lib`, and commits changes to SQLite.
  * **Culmination:** Alex queries Codex in his IDE and receives precise answers citing the lines just pushed.
  * **Resolution:** Knowledge base updated autonomously in the background.

* **UJ-2: Ivan Configures and Tests a Private Repository**
  * **Context:** Ivan (DevOps) sets up integration with a private Git repository.
  * **Precondition:** Ivan opens the Web UI Dashboard and authenticates.
  * **Flow:**
    1. Ivan clicks "Add Repository", enters the Git URL, and selects "SSH".
    2. Specifies the mounted private key path inside the container.
    3. Clicks "Sync Now". The backend creates a queue job.
  * **Culmination:** Ivan watches the job progress in real time via the queue table and monitors the initial indexing progress bar.
  * **Resolution:** Repository status transitions to `Active`.

* **UJ-3: Maria Queries the Integrated RAG Chat**
  * **Context:** Maria (Frontend Developer) seeks code examples for NotificationService in backend repos.
  * **Precondition:** Maria opens the "RAG Chat" tab in the dashboard.
  * **Flow:**
    1. Maria submits query: *"How do I invoke NotificationService.Send?"*.
    2. Backend triggers hybrid search, assembling top 5 chunks.
    3. Backend dispatches the prompt via local gateway to the configured model.
  * **Culmination:** Response streams token by token with code highlighting and clickable file citations.
  * **Resolution:** Maria reviews the answer and navigates directly to the cited source file.

---

## 3. Glossary

* **Local Codebase Cache:** Container directory `/data/clones/{repo_id}` where repositories are cloned for delta synchronization.
* **Autonomous Sync:** Automated indexing triggered by incoming webhooks or scheduled Cron expressions.
* **Job Queue:** SQLite table `sync_queue` managing pending and in-progress synchronization tasks with retry counters.
* **Webhook Validator:** Module cryptographically validating HMAC-SHA256 signatures of incoming Git server HTTP payloads.
* **OmniRoute:** OpenAI-compatible local API gateway (`http://localhost:20128/v1`) routing RAG queries to configured models.
* **RAG Chat:** Web UI conversational interface for testing knowledge base retrieval with streaming and citations.
* **Task Idempotency:** Guarantee that re-running tasks for identical commits creates no redundant embeddings or duplicated rows.

---

## 4. Functional Requirements

### 4.1. Resilient Database-Backed Background Sync
* **FR-1: SQLite Job Queue:** Persist all sync requests in table `sync_queue` (`id`, `repo_id`, `status`, `attempts`, `error_message`, `created_at`, `updated_at`).
* **FR-2: FIFO Worker Consumption:** Background worker pops the oldest `pending` task (`created_at` ASC) and marks it `processing`.
* **FR-3: Execution Idempotency:** Worker validates Git commit SHAs; identical snapshots complete immediately without re-calculating vectors.
* **FR-4: Backoff and Error Halting:** On failure, the worker performs up to 3 retries with exponential backoff (1m, 5m, 15m). Upon exceeding limits, the job transitions to `failed`, the worker sets `system_status` to `halted`, and processing pauses.
* **FR-5: Containerized Git Client:** Worker executes clone/pull operations in `/data/clones/{repo_id}` using mounted SSH keys or environment PATs.

### 4.2. Webhook and Schedule Ingestion
* **FR-6: Cryptographic Validation:** Verify HMAC-SHA256 for GitHub (`X-Hub-Signature-256`) and Gitea (`X-Gitea-Signature`), and token matching for GitLab (`X-Gitlab-Token`). Invalid signatures reject immediately with `401 Unauthorized`.
* **FR-7: Cron Scheduler:** Support recurring sync triggers via standard Cron expressions in `config.yaml`.

### 4.3. Web UI Dashboard
* **FR-8: Lightweight SPA Frontend:** Pure HTML5/CSS/JS interface served statically from FastAPI.
* **FR-9: Basic Auth Protection:** Login and password protection. If `DASHBOARD_PASSWORD` is unset, access is locked with `403 Forbidden`.
* **FR-10: Read-Only Queue Viewer:** Table view of all queue jobs, statuses, retry counts, and error details.
* **FR-11: Reactive WebSocket Telemetry:** Real-time worker states, repository metrics, and progress bars push over WebSockets.

### 4.4. Integrated RAG Chat
* **FR-12: Browser Chat UI:** Text input, message history, auto-scrolling, Markdown code rendering.
  * **Feature Flag:** Fully toggleable via `server.dashboard.chat.enabled` (true/false) in `config.yaml`.
  * **Dialogue Memory:** Conversation history persists on the client and is transmitted per request.
  * **Access Control:** API endpoints authenticate using the same signed session tokens as WebSockets.
* **FR-13: Hybrid Context Retrieval:** Call `search_chunks_hybrid`, select top 5 matches, and assemble system prompt context.
* **FR-14: OpenAI-Compatible Protocol:** Dispatch requests to local or remote endpoints conforming to OpenAI Chat Completions standards.
* **FR-15: Token Streaming:** Stream text generation to the UI via Server-Sent Events (SSE).
* **FR-16: Source Citations:** Render clickable file tags with line numbers below every assistant reply.

---

## 5. Success Metrics

* **SM-1 (Queue Reliability):** 100% of SQLite tasks are cleanly processed or marked failed. Following restarts, workers resume from uncompleted jobs.
* **SM-2 (Idempotency):** Re-syncing an unchanged repository finishes in <1 second with zero duplicated records.
* **SM-3 (RAG Latency):** Time to first streaming token in the chat is <2 seconds from submission.

---

## 6. Assumptions

* **[ASSUMPTION: Gateway API]** Local inference gateways support standard OpenAI JSON token streaming (`stream=true`).
* **[ASSUMPTION: SQLite WAL]** SQLite WAL mode allows concurrent background worker writes while MCP search clients read simultaneously.
