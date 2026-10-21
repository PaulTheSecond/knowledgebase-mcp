# Knowledge MCP

![Version v1.6.2](https://img.shields.io/badge/version-v1.6.2-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)
![SQLite-vec](https://img.shields.io/badge/SQLite--vec-WAL%20enabled-orange.svg)

High-precision autonomous local knowledge base server (RAG) that implements the **Model Context Protocol (MCP)**. It enables AI coding assistants (Codex, Claude Code, Gemini CLI, Cursor, etc.) to navigate, search, and reason about complex codebases using a hybrid approach combining semantic, lexical, and structural call-graph analysis.

Now includes a fully **autonomous background sync daemon**, **Git webhooks support**, a **Modern Angular 21 Web Dashboard**, an **Interactive RAG Chat** with an **IDE-Style Tabbed Document Inspector**, **Mermaid diagram rendering**, and **storage-optimized FTS5 External Content**.

---

## 🚀 Key Features

1. **Hybrid Triple Search**:
   - Fuses three retrieval channels through **Reciprocal Rank Fusion (RRF)**:
     - **Full-Text Search (FTS5 External Content)**: Zero text duplication (`content='chunks'`), auto-vacuum (`FULL`), and instant keyword/symbol search saving ~50% database storage.
     - **Semantic Vector Search (`sqlite-vec`)**: High-accuracy conceptual matching with multilingual embeddings (`intfloat/multilingual-e5-small`).
     - **Graph-Based Retrieval**: 2x relevance boost to actual code definitions and call chains.
2. **Deep Structural & Semantic Indexing**:
   - **C# / .NET**: Native **Roslyn** workspace analysis extracting full symbol hierarchies and method invocations across large `.sln` solutions.
   - **Polyglot Code**: **Tree-sitter** engine for Python, TypeScript, JavaScript, Go, Rust, and more.
   - **Markdown**: Section-aware chunking for architecture decisions and guides.
3. **Autonomous Background Lifecycle & Queue**:
   - In-database SQLite FIFO queue (`sync_queue`) ensuring persistent state across restarts.
   - Background worker with exponential backoff (1m, 5m, 15m), fatal halt protection, and Git commit SHA idempotency.
   - **Transaction Batching (`BATCH_COMMIT_SIZE = 50`)**: Resilient indexing of enterprise monorepos (1,400+ files, 23,000+ chunks) without disk I/O pressure.
   - **Direct Local Repository Mounts (`local_path`)**: Direct indexing of pre-mounted local directories without remote clone or token requirements.
   - In-memory credential safety: PAT tokens and SSH keys are kept in RAM and never leaked into `.git/config`.
4. **Git Webhooks & Intelligent Delta Sync**:
   - Built-in HTTP receiver `/webhook/git` supporting GitHub, Gitea (HMAC-SHA256 signature validation), and GitLab (`X-Gitlab-Token`).
   - Parses commit payloads (`added`, `modified`, `removed`) for sub-second targeted delta re-indexing.
   - Scheduled periodic re-indexing via standard Cron expressions.
5. **Modern Web Dashboard & Telemetry (Angular 21 + Taiga UI 5)**:
   - Modern, high-performance SPA dashboard built with Angular 21 and Taiga UI 5 design system.
   - **Bilingual & Localization (RU / EN)**: Full multilingual support on both frontend and backend (`Accept-Language`), with default locale stored in SQLite.
   - **Theming**: Dark, Light, and System color schemes with reactive design-token styling.
   - **In-Browser Settings & Management**: Manage LLM endpoints, embeddings, and Git repositories with instant connection testing (`test-git`, `test-llm`) directly from the UI.
   - Protected by HTTP Basic Auth and short-lived crypto tokens.
   - Real-time reactive WebSockets (`/ws/status`) showing repo metrics, sync progress bars, and queue status.
6. **Interactive RAG Chat & IDE-Style Doc Inspector**:
   - Chat with your codebase in the browser tab powered by local LLMs (OmniRoute, vLLM, Ollama, Kiro, etc.) with character-by-character Server-Sent Events (SSE) streaming.
   - **IDE-Style Document Viewer (Inspector)**: Clickable source citations open in a dedicated right-hand inspector panel with multi-tab support, allowing developers to inspect multiple referenced files simultaneously.
   - **Citation Highlighting & Line Gutter**: Exact line ranges (e.g. lines 6–9) highlighted in yellow-orange with smooth auto-scroll to the citation; toggle between full document view and isolated chunk snippet.
   - **Rendered Markdown & Raw Code View**: Switch seamlessly between formatted Markdown preview and raw syntax-highlighted code.
   - **Mermaid Diagrams & Code Headers**: Rich SVG diagram rendering for Mermaid blocks with Dark/Light theme auto-sync, styled code headers with programming language labels and one-click clipboard copying.
   - **Resizable Splitter & Dialogue Memory**: Drag-to-resize split view defaulting to 45% with double-click reset; browser-persisted chat history with a "New Dialog" reset action.

---

## 🛠 Available MCP Tools

Connected AI agents have access to rich read-only tools:

### Search & Retrieval
- **`knowledge_search`**: Triple hybrid search across all configured repositories. Returns code chunks with trust levels (`verified` for source code, `hint` for docs).
- **`knowledge_get_chunk`**: Retrieve complete chunk content and associated metadata by UUID.
- **`knowledge_read_file`**: Retrieve the complete raw content of an indexed file from the knowledge base by its path.

### Symbol & Graph Navigation
- **`knowledge_find_symbol`**: Search classes, interfaces, and methods using wildcards (e.g. `*Repository`).
- **`knowledge_get_callers` / `knowledge_get_callees`**: Inspect method invocation call trees.
- **`knowledge_get_hierarchy`**: Explore class inheritance and interface implementation graphs.
- **`knowledge_impact_analysis`**: Recursive dependency analysis calculating the blast radius of symbol changes.

### Management
- **`knowledge_sync_repo`**: Enqueue an asynchronous background sync job for a repository.
- **`knowledge_delete_repo`**: Safely remove indexed repository data from the knowledge base.

---

## 🐳 Quick Start (Docker)

### 1. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Adjust the values in `.env`:
```env
REPOS_DIR=C:\Repos
DB_PATH=/data/knowledge.db
DASHBOARD_PASSWORD=your_secure_password
WEBHOOK_SECRET=your_webhook_hmac_secret
GITHUB_PAT=ghp_your_github_token
LLM_BASE_URL=http://host.docker.internal:8005/v1
LLM_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
```

### 2. Configure Repositories

Copy the example repository configuration:
```bash
cp config.yaml.example data/config.yaml
```

Configure your repositories, authentication, and webhook triggers:
```yaml
server:
  port: 8000
  host: "0.0.0.0"
  dashboard:
    enabled: true
    password_env: "DASHBOARD_PASSWORD"
    chat:
      enabled: true

embeddings:
  enabled: false # Set true to enable vector search

repositories:
  # Remote repository with PAT authentication
  - id: "my-project"
    url: "https://github.com/org/my-project.git"
    auth:
      type: "pat"
      token_env: "GITHUB_PAT"
    branch: "master"
    triggers:
      webhooks:
        enabled: true
        secret_env: "WEBHOOK_SECRET"
      cron: "0 */4 * * *"

  # Locally mounted repository (no clone/pull, direct delta-sync)
  - id: "ImpactOS.Core.Lib"
    local_path: "/repos/ImpactOS.Core.Lib"
    auth:
      type: "none"
    branch: "master"
```

### 3. Launch the Daemon

```bash
docker compose up -d
```

Access the Web Dashboard at: **`http://localhost:8000/`** (Credentials: `admin` / your `DASHBOARD_PASSWORD`).

---

## 💻 Local Development & CLI

### Installation

Requires **Python 3.10+** and **.NET 8.0 SDK** (for C# Roslyn analysis).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### CLI Commands

- **Start Background Daemon & Web Dashboard:**
  ```bash
  python -m knowledge_mcp.main serve --host 0.0.0.0 --port 8000
  ```

- **Run CLI Repository Sync (Incremental or Force):**
  ```bash
  # Incremental sync (skips unchanged files)
  python -m knowledge_mcp.main sync --repo-id my-project --repo-path C:\Repos\my-project --all

  # Force re-indexing of all files ignoring hash/mtime cache
  python -m knowledge_mcp.main sync --repo-id my-project --repo-path C:\Repos\my-project --all --force
  ```

- **Run MCP Stdio Server (for AI IDE integration):**
  ```bash
  # Fast startup with FTS5 + Graph analysis (zero neural memory overhead)
  python -m knowledge_mcp.main mcp

  # With vector embeddings enabled (multilingual-e5-small)
  python -m knowledge_mcp.main mcp --with-embeddings
  ```

### Frontend Development (Angular 21 + Taiga UI 5)

```bash
cd frontend

# Install dependencies
npm install

# Start local dev server with HMR and proxy to backend (http://localhost:4200)
npm start

# Build production bundle for FastAPI / Docker
npm run build
```

### Testing & Quality Assurance

- **Run Unit & Integration Tests (92 passing tests):**
  ```bash
  pytest
  ```

- **Run Mutation Testing (`mutmut`):**
  ```bash
  mutmut run
  mutmut results
  ```

---

## 🌐 REST & WebSocket API

The daemon provides a complete REST and real-time WebSocket API for CI/CD, webhooks, and automation:

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/` | Basic Auth | Angular 21 SPA Web Dashboard. |
| `POST` | `/api/auth/token` | Basic Auth | Issue a temporary cryptographic session token for WebSockets / SSE. |
| `GET` | `/api/status` | Basic Auth | System telemetry: state (`idle`/`syncing`/`halted`), error, repo metrics (files, chunks, vectors). |
| `GET` | `/api/queue` | Basic Auth | List recent background synchronization queue tasks and their states. |
| `POST` | `/api/queue/retry-failed` | Basic Auth | Reset all failed jobs back to `pending` status for immediate retry. |
| `POST` | `/api/sync/{repo_id}` | Basic Auth | Enqueue an immediate synchronization job for a repository. |
| `DELETE` | `/api/repo/{repo_id}` | Basic Auth | Delete all indexed data, chunks, and symbols for a repository. |
| `GET` | `/api/settings` | Basic Auth | Retrieve system settings (theme, locale, LLM, embeddings, repos). |
| `PUT` | `/api/settings` | Basic Auth | Update settings and persisted config. |
| `POST` | `/api/settings/repositories` | Basic Auth | Add new repository to configuration and initialize tracking. |
| `PUT` | `/api/settings/repositories/{id}` | Basic Auth | Update repository parameters. |
| `DELETE` | `/api/settings/repositories/{id}` | Basic Auth | Remove repository with optional data purging. |
| `POST` | `/api/settings/test-git` | Basic Auth | Validate Git credentials and remote branch accessibility. |
| `POST` | `/api/settings/test-llm` | Basic Auth | Verify connectivity and health of configured LLM API. |
| `POST` | `/api/chat` | Basic Auth | Interactive RAG Chat endpoint with character-by-character SSE streaming. |
| `GET` | `/api/chat/file` | Basic Auth | Retrieve file content or line range (`path`, `start_line`, `end_line`, `full_file`, `repo_id`) across configured repository roots. |
| `WS` | `/ws/status?token=...` | Temp Token | Real-time WebSocket feed of progress bars, telemetry, and queue updates. |
| `POST` | `/webhook/git` | HMAC / Token | Universal webhook receiver for GitHub (`X-Hub-Signature-256`), Gitea, and GitLab (`X-Gitlab-Token`). |

---

## 🔗 Git Webhook Setup

To configure automatic background re-indexing upon `git push`:
1. In your Git server (GitHub / Gitea / GitLab), go to **Settings** → **Webhooks** → **Add webhook**.
2. Set **Payload URL**: `http://<YOUR_HOST>:8000/webhook/git`
3. Set **Content type**: `application/json`
4. Set **Secret**: value matching `WEBHOOK_SECRET` in your `.env`.
5. Select events: **Just the `push` event**.
6. Set **Active**: checked.

When commits are pushed, Knowledge MCP extracts the touched files and performs targeted sub-second delta indexing.

---

## 🤖 Connecting to AI Agents (MCP Client Config)

Add the following to your AI client configuration (e.g. `mcp.json` or Antigravity config):

```json
{
  "mcpServers": {
    "knowledge-mcp": {
      "command": "python",
      "args": ["-m", "knowledge_mcp.main", "mcp"],
      "cwd": "C:/Repos/knowledgebase-mcp"
    }
  }
}
```

Or when running via Docker:

```json
{
  "mcpServers": {
    "knowledge-mcp": {
      "command": "docker",
      "args": ["exec", "-i", "knowledge-mcp", "python", "-m", "knowledge_mcp.main", "mcp"]
    }
  }
}
```

---

## 📐 Architecture & Roadmap

- [Roadmap: Autonomous Sync, Dashboard & Horizons](docs/roadmap_autonomous_sync.md)
- [ADR-001: Local MCP RAG Foundation](docs/adr-001-local-mcp-rag-knowledge-base.md)
- [ADR-002: Hybrid Semantic & Graph Indexing](docs/adr-002-hybrid-semantic-graph-indexing.md)
- [ADR-004: Autonomous Sync, Triggers & Credentials](docs/adr-004-autonomous-sync-triggers-credentials.md)
- [ADR-005: Web Dashboard Architecture](docs/adr-005-web-dashboard-architecture.md)
- [ADR-006: FTS5 External Content, Storage Optimization & Batching](docs/adr-006-fts5-external-content-and-large-repo-batching.md)
- [ADR-007: Interactive Chat Doc Inspector, Mermaid & Formatting](docs/adr-007-interactive-chat-doc-inspector-and-mermaid.md)
