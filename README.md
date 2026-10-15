# Knowledge MCP

![Version v1.6](https://img.shields.io/badge/version-v1.6-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)
![SQLite-vec](https://img.shields.io/badge/SQLite--vec-WAL%20enabled-orange.svg)

High-precision autonomous local knowledge base server (RAG) that implements the **Model Context Protocol (MCP)**. It enables AI coding assistants (Codex, Claude Code, Gemini CLI, Cursor, etc.) to navigate, search, and reason about complex codebases using a hybrid approach combining semantic, lexical, and structural call-graph analysis.

Now includes a fully **autonomous background sync daemon**, **Git webhooks support**, a **No-Build SPA Web Dashboard**, and an **Interactive RAG Chat** powered by local LLMs.

---

## 🚀 Key Features

1. **Hybrid Triple Search**:
   - Fuses three retrieval channels through **Reciprocal Rank Fusion (RRF)**:
     - **Full-Text Search (FTS5)**: Exact symbol names, identifiers, and keyword queries.
     - **Semantic Vector Search (`sqlite-vec`)**: High-accuracy conceptual matching with multilingual embeddings (`intfloat/multilingual-e5-small`).
     - **Graph-Based Retrieval**: 2x relevance boost to actual code definitions and call chains.
2. **Deep Structural & Semantic Indexing**:
   - **C# / .NET**: Native **Roslyn** workspace analysis extracting full symbol hierarchies and method invocations.
   - **Polyglot Code**: **Tree-sitter** engine for Python, TypeScript, JavaScript, Go, Rust, and more.
   - **Markdown**: Section-aware chunking for architecture decisions and guides.
3. **Autonomous Background Lifecycle & Queue**:
   - In-database SQLite FIFO queue (`sync_queue`) ensuring persistent state across restarts.
   - Background worker with exponential backoff (1m, 5m, 15m), fatal halt protection, and Git commit SHA idempotency.
   - In-memory credential safety: PAT tokens and SSH keys are kept in RAM and never leaked into `.git/config`.
4. **Git Webhooks & Intelligent Delta Sync**:
   - Built-in HTTP receiver `/webhook/git` supporting GitHub, Gitea (HMAC-SHA256 signature validation), and GitLab (`X-Gitlab-Token`).
   - Parses commit payloads (`added`, `modified`, `removed`) for sub-second targeted delta re-indexing.
   - Scheduled periodic re-indexing via standard Cron expressions.
5. **Web Dashboard & Real-Time Telemetry**:
   - Lightweight No-Build SPA (HTML5, Vanilla CSS/JS) served directly from FastAPI.
   - Protected by HTTP Basic Auth and short-lived crypto tokens.
   - Real-time reactive WebSockets (`/ws/status`) showing repo metrics, sync progress bars, and queue status.
6. **Interactive RAG Chat & Local LLM Integration**:
   - Chat with your codebase in the browser tab.
   - Compatible with OpenAI API endpoints (OmniRoute, vLLM, Ollama, Kiro, etc.).
   - Character-by-character Server-Sent Events (SSE) streaming, syntax highlighting, and clickable file source tags.

---

## 🛠 Available MCP Tools

Connected AI agents have access to rich read-only tools:

### Search & Retrieval
- **`knowledge_search`**: Triple hybrid search across all configured repositories. Returns code chunks with trust levels (`verified` for source code, `hint` for docs).
- **`knowledge_get_chunk`**: Retrieve complete chunk content and associated metadata by UUID.

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

- **Run CLI Repository Sync:**
  ```bash
  python -m knowledge_mcp.main sync --repo-id my-project --repo-path C:\Repos\my-project --all
  ```

- **Run MCP Stdio Server (for IDE integration):**
  ```bash
  python -m knowledge_mcp.main mcp
  ```

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
