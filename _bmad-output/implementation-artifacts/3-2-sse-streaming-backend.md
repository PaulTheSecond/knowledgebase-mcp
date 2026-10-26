# Story 3.2: SSE Streaming Backend

Status: done

## Story

As a developer,
I want to receive model responses token-by-token as they are generated (streaming),
so that I do not have to wait 10–15 seconds for the entire text block to complete.

## Acceptance Criteria

1. **Given** endpoint `POST /api/chat/stream`, authorized via a temporary token in query parameters,
   **When** the backend receives a streaming completion from OmniRoute (`stream=true`),
   **Then** it relays the stream to the client in `text/event-stream` (Server-Sent Events) format token by token.
2. **And** upon completion, a final JSON event is dispatched containing metadata with the list of source files (paths and line ranges) referenced in the prompt context.

## Tasks / Subtasks

- [x] Implement Server-Sent Events endpoint (AC: 1)
  - [x] Create endpoint `POST /api/chat/stream` in `routes/chat.py`.
  - [x] Authorize request via query token: `/api/chat/stream?token=<TOKEN>`.
  - [x] Return `StreamingResponse` from FastAPI with `media_type="text/event-stream"`.
- [x] Relay stream from OmniRoute (AC: 1)
  - [x] Issue completion request to OmniRoute with `stream=True`.
  - [x] Implement async generator `token_generator()` yielding SSE events:
    `data: {"token": "..."}\n\n`
- [x] Transmit source metadata in stream (AC: 2)
  - [x] Prior to stream termination, yield final `sources` event containing serialized JSON of source files:
    `data: {"sources": [{"filepath": "db.py", "lines": "10-20"}]}\n\n`
  - [x] Close connection cleanly.

## Dev Notes

Example of async event generator for SSE streaming in FastAPI:
```python
from fastapi.responses import StreamingResponse

async def event_generator(question: str, messages: list):
    # 1. Retrieve context and build prompt
    context = search_chunks_hybrid(question)
    system_prompt = build_system_prompt(context)
    
    # 2. Issue request to OmniRoute
    response = await client.chat.completions.create(
        model="kr/claude-sonnet-4.5",
        messages=[{"role": "system", "content": system_prompt}] + messages,
        stream=True
    )
    
    # 3. Stream tokens
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield f"data: {json.dumps({'token': content})}\n\n"
            
    # 4. Stream sources at stream completion
    sources_data = [{"filepath": c.filepath, "lines": c.lines} for c in context]
    yield f"data: {json.dumps({'sources': sources_data})}\n\n"
```

### References

- [SSE Transport Architectural Invariant: ARCHITECTURE-SPINE.md#AD-3](file:///_bmad-output/planning-artifacts/architecture/architecture-knowledgebase-mcp-2026-06-30/ARCHITECTURE-SPINE.md#L44-L50)

## Dev Agent Record

### Agent Model Used
Antigravity (Gemini 3.5 Flash)

### Debug Log References
None

### Completion Notes List
- Implemented streaming endpoint `POST /api/chat/stream` serving Server-Sent Events (SSE).
- Configured authorization via query parameter `?token=<TOKEN>` with temporary token validation.
- Created async generator `event_generator()` using `httpx.AsyncClient().stream()`.
- Implemented live SSE token forwarding from OmniRoute to the connected client.
- Appended source file metadata in JSON format at the conclusion of the stream: `data: {"sources": [{"filepath": "...", "lines": "..."}]}`.
- Added tests in `tests/test_chat.py` covering token auth and streaming (5/5 tests passing).

### File List
- [MODIFY] [chat.py](file:///c:/Repos/knowledgebase-mcp/knowledge_mcp/daemon/routes/chat.py)
- [MODIFY] [test_chat.py](file:///c:/Repos/knowledgebase-mcp/tests/test_chat.py)

---

## Code Review + Workflow Validation

**Reviewer:** CodeReview
**Decision:** ✅ **PASS**

### Workflow Validation
- [x] Status changed to "In Progress" in Change Log.
- [x] Tasks and checklist marked as complete.
- [x] Dev Agent Record filled out.
- [x] Build/Linter passes (Ruff check clean).
- [x] Unit tests pass successfully (42/42 tests pass).

### Code Review Summary
- **Resource Management:** Using async context manager `async with client.stream(...)` guarantees prompt connection cleanup without dangling socket handles.
- **Error Handling inside Stream:** Generator exceptions are caught and formatted into structured JSON error payloads, preventing abrupt client drops.

---

## QA Results

**Reviewer:** Quinn (QA Engineer)
**Decision:** ✅ **PASS**

### Acceptance Criteria Verification
| AC | Description | Verification | Result |
|----|-------------|--------------|--------|
| AC 1 | SSE Streaming API | Test `test_stream_success` verifies `text/event-stream` handshake and token delivery | ✅ PASS |
| AC 2 | Source metadata transfer | Test `test_stream_success` verifies trailing `sources` payload with file paths and line ranges | ✅ PASS |

---

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-07-02 | 0.1.0 | Dev: Started Story 3.2 implementation | James |
| 2026-07-02 | 1.0.0 | Dev: Implemented SSE stream router with async httpx pipeline | James |
| 2026-07-02 | 1.0.1 | CR: Verified async connections security and exceptions handling. PASS | CodeReview |
| 2026-07-02 | 1.0.2 | QA: Verified streaming chunks, token auth checks. Status -> Done | Quinn |
