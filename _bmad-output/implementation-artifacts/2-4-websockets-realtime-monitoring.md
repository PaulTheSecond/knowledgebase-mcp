# Story 2.4: WebSockets Real-Time Monitoring

Status: done

## Story

As a developer,
I want to observe repository synchronization progress bars and task status transitions on the dashboard in real time,
so that I do not need to manually refresh the browser page.

## Acceptance Criteria

1. **Given** WebSocket endpoint `/ws/status` on the backend,
   **When** the frontend initiates a connection passing a temporary token `/ws/status?token=<TOKEN>`,
   **Then** the backend validates the token and accepts the session (closing with code `4001` if invalid).
2. **And** upon changes in worker state or indexing progress, the backend broadcasts JSON events to connected sockets, allowing the frontend to reactively update repository cards and animate progress bars (0–100%) without page reload.

## Tasks / Subtasks

- [x] Implement WebSocket endpoint (AC: 1)
  - [x] Create WebSocket route `/ws/status` in `knowledge_mcp/daemon/routes/status.py`.
  - [x] Read token from query parameters: `token = websocket.query_params.get("token")`.
  - [x] Verify token signature via `auth.py`. If invalid, close connection with code `4001` (Unauthorized WS).
- [x] Implement event broadcasting (AC: 2)
  - [x] Create `ConnectionManager` to broadcast events across active client connections.
  - [x] Integrate indexing progress event emission from `indexer.py` / `queue_worker.py` into the socket manager.
- [x] Client-side WebSocket integration (AC: 2)
  - [x] Implement `connectWebSocket()` in `dashboard/static/js/app.js`.
  - [x] Obtain token prior to handshake via `POST /api/auth/token`.
  - [x] Open connection `new WebSocket("ws://" + location.host + "/ws/status?token=" + token)`.
  - [x] Update DOM on inbound messages: toggle status badges, animate progress bar width (`transition: width 0.2s ease-in-out`).

## Dev Notes

The worker reports indexing completion percentages. In `indexer.py`, a progress callback (calculated via `processed_files / total_files`) sends events to the WebSocket manager.
Message schema:
```json
{
  "type": "sync_progress",
  "data": {
    "repo_id": "Yurta.Core.Lib",
    "progress": 45,
    "status": "syncing"
  }
}
```

### References

- [WebSocket Architectural Invariant: ARCHITECTURE-SPINE.md#AD-3](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md#L44-L50)
- [Real-time UI Invariant: ADR-005-web-dashboard-architecture.md](file:///c:/Repos/knowledgebase-mcp/docs/adr-005-web-dashboard-architecture.md)

## Dev Agent Record

### Agent Model Used
Antigravity (Gemini 3.5 Flash)

### Debug Log References
None

### Completion Notes List
- **Knowledge MCP Research**: Investigated WebSocket best practices in FastAPI/uvicorn; implemented thread-safe `ConnectionManager` for worker threads.
- Created `daemon/status_ws.py` with `ConnectionManager` and helper `broadcast_sync` for safe event dispatch from synchronous worker threads.
- Added `/ws/status` route in `main.py` with one-time token verification (code `4001` on rejection).
- Integrated progress dispatch into `Indexer.sync_repo()` (scanning, vectorization, and completion stages).
- Enhanced `app.js` with WebSocket session establishment via `POST /api/auth/token` and animated progress bars.
- Verified test suite: 37/37 unit and WebSocket integration tests passing cleanly.

### File List
- [NEW] [status_ws.py](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/daemon/status_ws.py)
- [MODIFY] [main.py](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/main.py)
- [MODIFY] [indexer.py](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/core/indexer.py)
- [MODIFY] [app.js](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/dashboard/static/js/app.js)
- [MODIFY] [styles.css](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/dashboard/static/css/styles.css)
- [MODIFY] [test_auth.py](file:///c:/Repos/knowledgebase-mcp/tests/test_auth.py)

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build/Linter passes (Ruff check clean).
- [x] Unit tests pass successfully (37/37).

### Code Review Summary
- **Thread-Safety:** `asyncio.run_coroutine_threadsafe` ensures reliable thread bridging between background workers and FastAPI's event loop.
- **WebSocket Handshake:** Closure code `4001` for unauthorized handshakes follows RFC 6455 conventions.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | WebSocket token verification | Test `test_websocket_connection_unauthorized` verifies close code 4001 on bad tokens | ✅ PASS |
| AC 2 | Progress event delivery | Test `test_websocket_connection_success_and_broadcast` verifies `sync_progress` event receipt | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-02 | 0.1.0 | Dev: Started Story 2.4 implementation | James |
| 2026-07-02 | 1.0.0 | Dev: Implemented ConnectionManager, FastAPI /ws/status router, indexer progress hooks, client-side animations | James |
| 2026-07-02 | 1.0.1 | CR: Reviewed async loop thread-safe calling and WS closure codes. PASS | CodeReview |
| 2026-07-02 | 1.0.2 | QA: Verified WS handshake token check and message delivery. Status -> Done | Quinn |
