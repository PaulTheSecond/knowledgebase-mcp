# Roadmap: Autonomous Operation & Knowledge Management

This document outlines the planned evolution of the knowledge base MCP server toward fully autonomous operation — automatically reacting to changes in connected repositories and providing deep code comprehension tools.

Current version: **v1.7.0**

---

## Roadmap Visualization

```mermaid
timeline
    title Project Evolution
    v0.9 : Early Versions
         : Basic full-text search (FTS)
         : No dependency graph
    v1.2 : Stable Core
         : Hybrid search & Graph indexing (Roslyn/Tree-sitter)
    v1.3 - v1.4 : Triggers & Webhooks
         : Config & env secret management (PAT/SSH)
         : Cryptographic webhook validation (HMAC-SHA256)
         : Cron scheduling
    v1.5 - v1.6 : Autonomous Lifecycle & Web Dashboard
         : SQLite FIFO queue & resilient worker with backoff
         : No-Build SPA Dashboard (Basic Auth, WebSockets)
         : Interactive RAG Chat with local LLMs (OmniRoute)
    v1.6.1 : Maintenance & Fast Delta-Sync
         : Payload commit file paths extraction
         : Docker Compose hardening & relative volume paths
         : Comprehensive documentation update
    v1.7.0 : Interactive RAG Chat & IDE Doc Inspector
         : Multi-tab doc viewer & citation line highlighting
         : Resizable splitter (45% default) & LocalStorage memory
         : Mermaid SVG rendering & Prism code headers
    v1.8 : Hardening & Resource Telemetry
         : Memory and disk consumption telemetry in Web UI
         : Queue filtering & retry controls
    v1.9 : Multi-Branch & Release Tags
         : Multi-branch & release tag indexing support
    v2.0 (Horizon) : Visual Knowledge Graph Explorer
         : In-browser interactive dependency & call graph (vis.js/Cytoscape)
         : Blast-radius impact visualization in Web UI
```

---

## v1.3: Base Configuration (Env / Config File) — ✅ *Completed*
- Configure subscriptions and triggers via environment variables (`.env`) or a config file (`config.yaml`).
- Enables autonomous operation at container startup without requiring any visual interface.

## v1.3.1: Update Sources — ✅ *Completed*
- Define source parameters for Git repositories in configuration.
- Secure credential management in RAM (PAT via `http.extraHeader`, SSH without saving tokens on disk).

## v1.4: Trigger Configuration — ✅ *Completed*
- HTTP endpoint `/webhook/git` supporting GitHub, Gitea (`X-Hub-Signature-256`, `X-Gitea-Signature`) and GitLab (`X-Gitlab-Token`).
- Periodic scheduled updates (Cron expressions).

## v1.5: Autonomous Processing Lifecycle — ✅ *Completed*
- SQLite-backed FIFO queue (`sync_queue`) with persistent status tracking.
- Worker process with exponential backoff retries (1m, 5m, 15m), fatal halt protection, and Git SHA idempotency.

## v1.6: Management Dashboard & RAG Chat — ✅ *Completed*
- Lightweight No-Build SPA web interface on FastAPI (HTML5, Vanilla CSS/JS).
- Basic Auth security with short-lived session tokens for WebSockets and SSE.
- Real-time indexing telemetry with progress bars and queue viewer.
- Interactive RAG Chat with SSE streaming, code syntax highlighting, and clickable file sources.

---

## v1.6.1: Maintenance, Storage Optimization & Large-Scale Indexing — ✅ *Completed*
- **Payload file parsing:** Extract modified and added file paths directly from webhook commit payloads (`commits[].added`, `commits[].modified`, `commits[].removed`) and pass them to the indexer for instant delta-indexing without scanning entire repositories.
- **Docker Compose portability:** Relative volume mounts (`./data:/data`) and `.env` configuration file ingestion.
- **Storage optimization (ADR-006):** FTS5 external content table (`content='chunks'`) eliminating 100% text duplication; `PRAGMA auto_vacuum = FULL;` for automatic file compaction.
- **Transaction batching & resilience:** Periodic batch commits (`BATCH_COMMIT_SIZE = 50`) and SQLite engine tuning (`temp_store = MEMORY`, `cache_size = -64000`, `busy_timeout = 60000`) enabling seamless indexing of 1,400+ files / 23,000+ chunks without disk I/O errors.
- **Local repository support (`local_path`):** Direct indexing of locally mounted folders without requiring remote Git clone/pull or credentials.
- **Mutation testing:** Setup `mutmut` for testing mutation resilience across the database and persistence layers.
- **Documentation refresh:** Comprehensive update of `README.md`, ADRs, and coding standards.

## v1.7.0: Interactive RAG Chat, IDE-Style Doc Inspector & Diagramming — ✅ *Completed*
- **IDE-style Tabbed Document Viewer (ADR-007):** Dedicated right-hand inspector panel with multi-tab support for simultaneous inspection of referenced files.
- **Citation line range highlighting & auto-scroll:** Exact line ranges (e.g. lines 6–9) highlighted in yellow-orange (`.line-highlight`) with smooth auto-scroll to the citation; toggle between full document view and isolated chunk snippet.
- **Rendered Markdown & Raw Code View:** Switch seamlessly between formatted Markdown preview (with checklist and table rendering) and raw syntax-highlighted code with line numbers.
- **Mermaid diagram rendering:** Rich SVG diagram rendering for Mermaid blocks with Dark/Light theme auto-sync, styled code headers with programming language labels and one-click clipboard copying.
- **Resizable Splitter & Dialogue Memory:** Drag-to-resize split view defaulting to 45% with double-click reset; browser-persisted chat history with a "New Dialog" reset action.
- **Configured Repo Path Resolution:** Backend `/api/chat/file` path resolution across all configured repository roots (`local_path`, `clones_dir`, `project_dir`) supporting `full_file=true`.

## v1.8: Operational Hardening & Health Metrics — 📋 *Planned*
*Estimate: Minor update*
- Memory and disk consumption telemetry in Web UI.
- Queue management extensions (filter by status, manual retry for halted tasks).
- Configurable token context limits for RAG chat.

## v1.9: Multi-Branch & Release Tag Indexing — 📋 *Planned*
*Estimate: Minor update*
- Support for indexing multiple designated branches (e.g., `main`, `staging`, `develop`) and release tags simultaneously.
- Branch-aware context retrieval in MCP search tools.

---

## v2.0 (Future Horizon): Visual Knowledge Graph & Architecture Explorer — 🔭 *Future Horizon*
*Estimate: Major milestone*
- **Interactive Graph UI:** In-browser visualizer (using Cytoscape.js or vis.js) embedded in Web Dashboard.
- **Dependency & Call Navigation:** Click-through navigation across symbols, inheritance hierarchies, and method call chains.
- **Visual Impact Analysis:** Interactive blast-radius visualization highlighting modules and symbols affected by proposed code changes.
- **Architecture Export:** Exporting architectural diagrams in SVG and Markdown formats.
