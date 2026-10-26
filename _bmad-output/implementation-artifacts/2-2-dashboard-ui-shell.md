# Story 2.2: Dashboard UI Shell

Status: done

## Story

As a developer,
I want an elegant, modern web dashboard with dark theme ergonomics,
so that operating and monitoring the system in the browser is pleasant and efficient.

## Acceptance Criteria

1. **Given** a static SPA frontend (clean HTML5, Vanilla CSS, Vanilla JS) located in `dashboard/static/`,
   **When** I open the dashboard root in a browser,
   **Then** a page with a premium dark theme, curated HSL color palette, and micro-transitions renders smoothly.
2. **And** the navigation menu enables switching between "Repositories", "Sync Queue", and "RAG Chat" tabs without page reloads (SPA).

## Tasks / Subtasks

- [x] Create HTML5 markup (AC: 1, 2)
  - [x] Create file `knowledge_mcp/dashboard/static/index.html`.
  - [x] Structure sidebar navigation and main content containers with distinct DOM IDs.
- [x] Style CSS3 design (AC: 1)
  - [x] Create file `knowledge_mcp/dashboard/static/css/styles.css`.
  - [x] Load `Outfit` typography via Google Fonts.
  - [x] Configure CSS variables for dark theme (charcoal, slate, and vibrant accent blues).
  - [x] Add smooth transitions (`transition: all 0.3s ease`) on interactive UI components.
- [x] Implement Vanilla JS SPA routing (AC: 2)
  - [x] Create file `knowledge_mcp/dashboard/static/js/app.js`.
  - [x] Implement client-side router toggling section visibility: `<section id="repositories-view">`, `<section id="queue-view">`, `<section id="chat-view">`.
  - [x] Handle Basic Auth browser credentials prompt upon initial load.
- [x] Integrate static assets into FastAPI (AC: 1)
  - [x] Mount static assets in `knowledge_mcp/daemon/app.py` via `FastAPI.mount("/static", StaticFiles(directory="knowledge_mcp/dashboard/static"), name="static")`.
  - [x] Serve `index.html` at root endpoint `/`.

## Dev Notes

The UI embraces glassmorphism, rounded corners (`border-radius: 12px`), and subtle neon accent glows.
CSS theme variables:
```css
:root {
  --bg-primary: #0d0e12;
  --bg-secondary: #161820;
  --accent-color: #3b82f6;
  --accent-glow: rgba(59, 130, 246, 0.15);
  --text-primary: #f3f4f6;
  --text-secondary: #9ca3af;
}
```

### References

- [Architecture Requirements: ADR-005-web-dashboard-architecture.md](file:///c:/Repos/knowledgebase-mcp/docs/adr-005-web-dashboard-architecture.md)

## Dev Agent Record

### Agent Model Used
Antigravity (Gemini 3.5 Flash)

### Debug Log References
None

### Completion Notes List
- **Knowledge MCP Research**: Researched existing HTML/CSS styles; designed fresh zero-dependency SPA dashboard.
- Created static bundle `index.html`, `styles.css`, and `app.js` in `knowledge_mcp/dashboard/static/`.
- Built modern responsive SPA dark layout with `Outfit` typography and micro-interactions.
- Implemented hash-based SPA client router handling Repositories, Sync Queue, and RAG Chat tabs.
- Mounted static directory in `knowledge_mcp/main.py` via `StaticFiles` and mapped root `/` to `index.html`.
- Added automated integration tests for static delivery in `tests/test_auth.py`.

### File List
- [NEW] [index.html](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/dashboard/static/index.html)
- [NEW] [styles.css](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/dashboard/static/css/styles.css)
- [NEW] [app.js](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/dashboard/static/js/app.js)
- [MODIFY] [main.py](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/main.py)
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
- [x] Unit tests pass successfully (34/34).

### Code Review Summary
- **SPA Router:** Hash-based routing (`window.location.hash`) prevents page reloads.
- **FastAPI Integration:** Proper non-blocking mounting with `StaticFiles`.
- **Design:** Zero third-party JS runtime dependencies keeps the bundle lightweight and maintainable.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | Dark-theme static SPA delivery | Test `test_static_dashboard_rendering` verifies HTTP 200 on index.html and styles.css | ✅ PASS |
| AC 2 | Seamless tab switching | `hashchange` handler switches visibility of views without reload | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-02 | 0.1.0 | Dev: Started Story 2.2 implementation | James |
| 2026-07-02 | 1.0.0 | Dev: Implemented SPA Dashboard HTML/CSS/JS and mounted in FastAPI | James |
| 2026-07-02 | 1.0.1 | CR: Reviewed SPA routing and FastAPI StaticFiles mounting. PASS | CodeReview |
| 2026-07-02 | 1.0.2 | QA: Verified static endpoints and design. Status -> Done | Quinn |
