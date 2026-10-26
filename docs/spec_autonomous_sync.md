# Specification: Autonomous Sync, Dashboard, and RAG Chat (v1.3 - v1.6.2)

This specification defines the functional and technical requirements for transitioning `knowledge-mcp` to autonomous operation, including automatic repository synchronization mechanisms, private repository credential management, and the web monitoring dashboard.

---

## 1. Use Cases

### UC-1: Trigger-Based Automated Updates (Webhooks)
* **Actor:** Remote Git Server (GitHub, GitLab, Gitea).
* **Precondition:** The `knowledge-mcp` daemon is running, the webhook ingress port is reachable, and the HMAC secret token is configured.
* **Flow:**
  1. A developer pushes commits to the target repository branch.
  2. The Git server dispatches an HTTP POST request (Webhook) to `/webhook/git`.
  3. `knowledge-mcp` validates the cryptographic HMAC-SHA256 signature against the configured secret.
  4. The server parses the payload and extracts modified file paths (`added`, `modified`, `removed`).
  5. A background worker triggers an incremental delta-sync on the modified files only.

### UC-2: Scheduled Synchronization (Cron)
* **Actor:** Background Task Scheduler.
* **Precondition:** A Cron schedule expression is configured for the target repository in `config.yaml`.
* **Flow:**
  1. The Cron trigger fires.
  2. The scheduler enqueues a sync job into the FIFO SQLite queue.
  3. The worker executes `git fetch` / `git pull`.
  4. The knowledge base incrementally updates altered AST symbols and chunks.

### UC-3: Secure Private Repository Access
* **Actor:** Developer / System Administrator.
* **Precondition:** Access to the private repository is configured via an SSH private key or Personal Access Token (PAT).
* **Flow:**
  1. At container startup or when adding a repository, the system validates the authentication scheme.
  2. For SSH: the mounted private key is utilized with verified permissions.
  3. For PAT: the token is injected into memory as an authorization header or HTTPS clone URL.
  4. The local clone cache (`/data/clones/{repo_id}`) clones or pulls updates cleanly.

### UC-4: Visual Monitoring via Web UI
* **Actor:** Developer / DevOps Engineer.
* **Precondition:** Dashboard option is enabled in configuration (`server.dashboard.enabled = true`).
* **Flow:**
  1. User navigates to `http://localhost:8000` (or configured host:port).
  2. Authenticates via HTTP Basic Auth.
  3. Views the dashboard displaying connected repositories, operational statuses (`Active`, `Error`, `Syncing`), last update timestamps, and metrics (file count, chunk count, symbol count, vectors).
  4. Can trigger manual re-indexing or inspect sync queue logs.

### UC-5: Interactive Dialogue (RAG Chat) & Document Inspector
* **Actor:** Developer.
* **Precondition:** Chat option is enabled (`server.dashboard.chat.enabled = true`) and LLM inference endpoint is reachable.
* **Flow:**
  1. User enters a query regarding the codebase in the chat window.
  2. The backend performs hybrid retrieval across the knowledge base, constructs prompt context, and streams the response via Server-Sent Events (SSE).
  3. Code in the response renders with syntax highlighting, language headers, and copy buttons; Mermaid blocks render as vector SVG diagrams.
  4. User clicks a cited source tag (e.g., `story-1.8.md (6-9)`).
  5. The right-hand document inspector opens with IDE-style tabs, highlighting lines 6–9 and smoothly auto-scrolling to the citation.
  6. The user can switch between tabs, adjust the resizable splitter (45% default), and toggle Preview/Source and Full File/Snippet views.

---

## 2. Configuration Schema (`config.yaml`)

The configuration file is located at `/data/config.yaml` by default (or configured via the `CONFIG_PATH` environment variable).

```yaml
server:
  port: 8000
  host: "0.0.0.0"
  dashboard:
    enabled: true
    port: 8080
    password_env: "DASHBOARD_PASSWORD" # Environment variable containing auth password

# Global embedding settings for background vectorization
embeddings:
  enabled: true
  model_name: "intfloat/multilingual-e5-small"

# Tracked repositories
repositories:
  - id: "Yurta.Core.Lib"
    url: "git@github.com:your-org/Yurta.Core.Lib.git"
    auth:
      type: "ssh"
      private_key_path: "/root/.ssh/id_rsa"
    branch: "main"
    # Autonomous triggers
    triggers:
      webhooks:
        enabled: true
        provider: "github" # github | gitlab | custom
        secret_env: "YURTA_WEBHOOK_SECRET"
      cron: "0 */4 * * *" # Every 4 hours
      
  - id: "knowledgebase-mcp"
    url: "https://github.com/PaulTheSecond/knowledgebase-mcp.git"
    auth:
      type: "pat"
      token_env: "GITHUB_PAT_TOKEN"
    branch: "master"
    triggers:
      webhooks:
        enabled: false
      cron: "30 2 * * *" # Daily at 02:30 AM
```

---

## 3. Web UI REST API Endpoints

The dashboard communicates with the backend through the following endpoints:

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/api/status` | System health check and repository list with metrics. |
| `POST` | `/api/sync/{repo_id}` | Manually enqueue an incremental sync job for a repository. |
| `DELETE` | `/api/repo/{repo_id}` | Purge a repository's indexed symbols and chunks from the database. |
| `POST` | `/api/chat` | Streaming RAG chat endpoint (SSE) with hybrid context retrieval. |
| `GET` | `/api/chat/file` | Retrieve file content (`path`, `start_line`, `end_line`, `full_file`, `repo_id`) for the Document Inspector. |

---

## 4. Non-Functional Requirements

1. **Data Locality:** Cloned repositories and cache storage reside strictly within local container volumes (`/data/clones/`). Source code and AST metadata are never transmitted to external cloud services.
2. **Resilience:** If a `git pull` or clone fails (e.g., transient network or credential issues), the repository status transitions to `Error`, the failure is logged, and the MCP server continues servicing queries against the last successful snapshot.
3. **RAM and Resource Efficiency:** Background indexing must avoid Out of Memory (OOM) conditions. Vectorization processes strictly limit sub-batch sizes and invoke PyTorch cache clearing / Python garbage collection after processing each repository batch.
