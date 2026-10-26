# Story 3.1: OmniRoute Client and Context Search

Status: done

## Story

As a developer,
I want the backend to retrieve relevant codebase chunks and forward them to an LLM via OmniRoute,
so that model responses are grounded in current, authoritative context from my repositories.

## Acceptance Criteria

1. **Given** a configured OmniRoute connection at `http://localhost:20128/v1` with API key `LLM_API_KEY`,
   **When** endpoint `POST /api/chat` receives a user query along with conversation history (array `messages`),
   **Then** the backend reads the feature flag `server.dashboard.chat.enabled` from `config.yaml`. If `false`, it returns `403 Forbidden`.
2. **And** when enabled, the backend performs hybrid search `search_chunks_hybrid` against SQLite, selects the top 5 relevant code chunks, and injects them into the system prompt with file paths and line ranges.
3. **And** the backend dispatches the augmented prompt and history to OmniRoute for model `kr/claude-sonnet-4.5` (kiro provider) and returns the LLM response.

## Tasks / Subtasks

- [x] Read configuration and chat feature flag (AC: 1)
  - [x] Validate `server.dashboard.chat.enabled` in `config.yaml`.
  - [x] Register HTTP route `POST /api/chat` in `knowledge_mcp/daemon/routes/chat.py`. Return `403 Forbidden` if disabled.
- [x] Contextual hybrid search (AC: 2)
  - [x] Integrate hybrid search (RRF) from `db/models.py`.
  - [x] Query top 5 code chunks matching query semantics and text.
  - [x] Format system prompt uniting code chunks with metadata:
    ```
    Use the following codebase context to answer the user query.
    [FILEPATH: path/to/file.cs]
    [LINES: 10-45]
    ```
- [x] OpenAI-compatible client for OmniRoute (AC: 3)
  - [x] Issue direct async HTTP requests via `httpx`.
  - [x] Configure parameters: `base_url="http://localhost:20128/v1"`, `api_key=LLM_API_KEY`.
  - [x] Build payload: `model="kr/claude-sonnet-4.5"`, `messages=[{"role": "system", "content": system_prompt}] + user_messages`.
  - [x] Execute request and return generated response.

## Dev Notes

OmniRoute acts as a local OpenAI-compatible API gateway. Direct async HTTP requests using `httpx.AsyncClient` keep dependencies minimal:
```python
import httpx

async def call_omniroute(messages: list) -> str:
    url = "http://localhost:20128/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"}
    payload = {
        "model": "kr/claude-sonnet-4.5",
        "messages": messages,
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(url, json=payload, headers=headers, timeout=60.0)
        return res.json()["choices"][0]["message"]["content"]
```

### References

- [OmniRoute Compatibility: prd.md#4.4.-Integrated-RAG-Chat](file:///_bmad-output/planning-artifacts/prds/prd-knowledgebase-mcp-2026-06-30/prd.md#L118-L131)

## Dev Agent Record

### Agent Model Used
Antigravity (Gemini 3.5 Flash)

### Debug Log References
None

### Completion Notes List
- **Knowledge MCP Research**: Researched hybrid RRF search `search_chunks_hybrid` in `models.py` and `LocalEmbedder` vector generation.
- Implemented `knowledge_mcp/daemon/routes/chat.py` with endpoint `POST /api/chat`.
- Checked feature flag `server.dashboard.chat.enabled` returning `403 Forbidden` when disabled.
- Linked `LocalEmbedder` vector generation and SQLite hybrid search to retrieve top 5 source code chunks.
- Implemented OpenAI-compatible completions call to OmniRoute via `httpx` without heavy external packages.
- Added 3 unit tests in `tests/test_chat.py` passing cleanly.

### File List
- [NEW] [chat.py](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/daemon/routes/chat.py)
- [MODIFY] [main.py](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/main.py)
- [NEW] [test_chat.py](file:///c:/Repos/knowledgebase-mcp/tests/test_chat.py)

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build/Linter passes (Ruff check clean).
- [x] Unit tests pass successfully (3/3).

### Code Review Summary
- **HTTP Client Choice:** Using `httpx.AsyncClient` avoids external bulky SDKs while providing high performance.
- **Prompt Isolation:** System prompt is strictly segregated from user conversation turns to prevent prompt injection.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | Feature flag enforcement | Test `test_chat_disabled_by_feature_flag` confirms 403 on disabled flag | ✅ PASS |
| AC 2 | Hybrid chunk retrieval | Test `test_chat_success` verifies `search_chunks_hybrid` execution and context injection | ✅ PASS |
| AC 3 | OmniRoute integration | Test `test_chat_success` with mocks validates completion delivery and parsing | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-02 | 0.1.0 | Dev: Started Story 3.1 implementation | James |
| 2026-07-02 | 1.0.0 | Dev: Implemented chat routes, SQLite hybrid search integration and HTTP client completions | James |
| 2026-07-02 | 1.0.1 | CR: Reviewed httpx client usage and custom exception handling. PASS | CodeReview |
| 2026-07-02 | 1.0.2 | QA: Verified mock completions response and sources block output. Status -> Done | Quinn |
