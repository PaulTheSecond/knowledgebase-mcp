# ADR-005: Web UI Dashboard Architecture

| Metadata | Value |
| :--- | :--- |
| **Status** | Accepted |
| **Date** | 2026-06-30 |
| **Author** | Antigravity / User |
| **Affects** | Dashboard UI, FastAPI Server, REST API Contract |
| **Breaking Change** | No |

---

## Context

In accordance with milestone `v1.6` of the project roadmap, the `knowledge-mcp` system requires a lightweight, functional Web UI Dashboard. This interface provides visual monitoring of knowledge base health, real-time background sync telemetry, error inspection, and manual repository management without requiring direct terminal or MCP client access.

The dashboard must run within the same Docker container as the MCP server, minimize resource consumption, and avoid bloating the final production image with heavy frontend build chains or persistent Node.js runtime processes.

---

## Problem Statement

1. **Docker Image Footprint:** Heavy modern SPA toolchains (React, Angular, Vue) typically introduce complex build pipelines (Vite, Webpack), hundreds of npm dependencies, and multi-stage Docker complexity with large cache footprints.
2. **Dashboard Security:** Access to indexing controls, logs, and telemetry must be strictly secured to prevent unauthorized access or denial-of-service (DoS) vulnerabilities if the container network port is exposed.
3. **Real-time Telemetry:** AST indexing and embedding generation can take minutes. Standard HTTP polling introduces unnecessary latency and server load.

---

## Decision

### 1. Lightweight No-Build SPA Frontend
The dashboard frontend was originally designed as a **No-Build SPA**:
* **Markup:** Semantic HTML5.
* **Styling:** Clean Vanilla CSS with a modern dark theme, subtle glassmorphism effects, smooth transitions, and standard typography (Inter).
* **Logic:** Modern ES6+ JavaScript modules served directly via `<script type="module">`.
* **Serving:** Static assets located under `knowledge_mcp/dashboard/static` are served directly by FastAPI via `FastAPI.mount("/static", StaticFiles(directory="..."))`, removing the need for a separate Nginx or Express container. *(Note: in v1.8, the dashboard was enhanced into an Angular 21 application built in a multi-stage Docker layer while preserving lightweight static delivery from FastAPI).*

### 2. Authentication and Security Model
* **HTTP Basic Authentication:** Standard, battle-tested server-side basic authentication checking `Authorization: Basic ...` headers.
* **Environment Configuration:** Credentials are configured via `DASHBOARD_USERNAME` (defaulting to `admin`) and `DASHBOARD_PASSWORD`. If `DASHBOARD_PASSWORD` is not configured, dashboard access is immediately blocked with `403 Forbidden` for security.
* **Local Binding by Default:** The dashboard listens on `127.0.0.1` by default, making it accessible locally or behind an authenticated reverse proxy (Nginx / Traefik).

### 3. Real-Time Telemetry over WebSockets
* Instead of aggressive HTTP polling, the client establishes a persistent **WebSocket** connection to the backend at `ws://<host>:<port>/ws/status`.
* Whenever sync states change (e.g., transition from `Idle` to `Syncing` or progress updates during vectorization), the backend broadcasts telemetry packets. The frontend updates progress bars and metrics dynamically without full-page reloads.

---

## UI Layout Wireframe

```
+-----------------------------------------------------------------------------+
|  [K-MCP] Knowledge MCP Dashboard                      [Status: Online]      |
+-----------------------------------------------------------------------------+
|                                                                             |
|  +-- Repositories Status ------------------------------------------------+  |
|  |  [Active] Yurta.Core.Lib (Branch: main)                              |  |
|  |  Files: 342 | Chunks: 1,204 | Chunks (Vec): 1,204                     |  |
|  |  Last Sync: 2026-06-30 22:00 (Success)    [Sync Now]  [Full Reindex]  |  |
|  |                                                                       |  |
|  |  [Syncing: 45%] ImpactOS.Core (Branch: master)                       |  |
|  |  Files Processed: 120/300  ||||||||||||..........                     |  |
|  |  Running background AST analysis...                      [Cancel]     |  |
|  +-----------------------------------------------------------------------+  |
|                                                                             |
|  +-- Realtime Event Logs ------------------------------------------------+  |
|  |  22:00:15 - [Yurta.Core.Lib] Sync completed successfully.             |  |
|  |  22:01:02 - [ImpactOS.Core] Started delta-sync (triggered by webhook).|  |
|  |  22:01:05 - [ImpactOS.Core] Warning: Parse error in file Program.cs.  |  |
|  +-----------------------------------------------------------------------+  |
|                                                                             |
+-----------------------------------------------------------------------------+
```

---

## Alternatives Considered

### Option A: React/Vite + Docker Multi-stage Build
*Rejected.* React introduces significant overhead and runtime complexity for backend developers maintaining the service. Build times increase due to external dependency downloads. Vanilla/lightweight architectures fulfill all real-time queue and repository management requirements with zero build overhead.

### Option B: Server-Side Rendering (SSR) via Jinja2 Templates
*Rejected.* Traditional SSR requires full-page reloads or extensive AJAX glue code for dynamic metrics. WebSocket communication with client-side reactive DOM updates provides a vastly superior user experience (UX) without burdening the server with HTML rendering.
