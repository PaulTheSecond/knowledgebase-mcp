# ADR-006: FTS5 External Content Table, Auto-Vacuum and Transaction Batching

* **Status:** Accepted
* **Date:** 2026-10-18
* **Context:** Version 1.6.1 Maintenance & Storage Optimization

---

## 1. Context and Problem Statement

As repositories connected to `knowledge-mcp` grew to enterprise size (e.g. `ImpactOS.Core.Lib` with 48 C# projects, 1,464 files, and 23,000+ chunks), two architectural challenges emerged:

1. **Storage Inefficiency (Text Duplication):**
   By default, SQLite FTS5 stores a full copy of the indexed text inside internal shadow tables (`chunks_fts_content`). Since the raw code content was already stored in the main `chunks` table, 100% of the text was duplicated, doubling the database file size. When deleting or re-indexing files, deleted pages were marked free but not returned to the OS filesystem, leading to database file bloat.

2. **Windows Docker Bind-Mount Transaction Exhaustion (`disk I/O error`):**
   In `Indexer.sync_repo`, scanning 1,400+ files occurred inside a single uncommitted transaction. On Docker volumes mounted from Windows hosts (`bind mount` via WSL2 / 9P / VirtioFS), holding hundreds of thousands of chunk, symbol, and FTS5 b-tree modifications in a single uncommitted WAL state exhausted page buffers and lock timeouts, resulting in `sqlite3.OperationalError: disk I/O error` and catastrophic rollback of all progress.

3. **Enterprise Monorepos & Local Source Direct Indexing:**
   Organizations frequently have source repositories already cloned or mounted into Docker containers locally. Requiring Git tokens (`GITHUB_PAT`) or remote URLs for mounted local paths was redundant and blocked offline / on-premises indexing.

---

## 2. Decision

### 2.1 FTS5 External Content Table (`content='chunks'`)
We migrated `chunks_fts` to an **External Content Table**:
```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='rowid'
);
```
- **Zero Content Duplication:** FTS5 stores only tokenized inverted index b-trees (`doclists`), reading the actual text on demand directly from the `chunks` table via `rowid`.
- **Automatic Triggers:** Synchronized via `chunks_ai` (AFTER INSERT), `chunks_ad` (AFTER DELETE), and `chunks_au` (AFTER UPDATE) with the external content protocol.
- **Auto-Vacuum (`PRAGMA auto_vacuum = FULL`):** Automatically compacts SQLite freelist pages when chunks are deleted or replaced, returning unused space to the disk.

### 2.2 Transaction Batching (`BATCH_COMMIT_SIZE = 50`)
We divided the indexing loop into periodic transaction batches:
- Every 50 files, `self.db.commit_transaction()` is executed followed by `self.db.begin_transaction()`.
- Flushes WAL frames incrementally, keeping memory and lock footprints small.
- Ensures all previously processed files remain safely persisted even if a subsequent batch encounters an unexpected error.
- Made `commit_transaction()` and `rollback_transaction()` idempotent and protected against exceptions.

### 2.3 SQLite Engine Pragmas
For write connections:
- `PRAGMA busy_timeout = 60000;` (waits up to 60s for locks, avoiding lock contention failures).
- `PRAGMA temp_store = MEMORY;` (stores temporary tables and indices in RAM).
- `PRAGMA cache_size = -64000;` (allocates 64 MB of page cache in RAM for rapid b-tree reads/writes).

### 2.4 Support for Local Repositories (`local_path`)
The daemon worker now checks for `local_path` (or `path`) in `config.yaml`:
- If configured and the folder exists on disk (or in `/repos/...`), the worker directly invokes `sync_repo` without requiring Git clone/pull or credentials.
- HEAD commit SHA is resolved locally via `git rev-parse HEAD`.

---

## 3. Consequences

* **Database Size:** Database file size reduced by ~45–50% due to elimination of duplicate text in FTS5.
* **Indexing Reliability:** 1,464 files in `ImpactOS.Core.Lib` indexed with 100% success in 6.5 minutes without any `disk I/O error`.
* **Cross-Repo References:** Graph-based symbol linking successfully resolved across projects.
* **Backward Compatibility:** All existing MCP tools (`knowledge_search`, `knowledge_find_symbol`, etc.) and the RAG chat continue to function identically with zero API changes.
