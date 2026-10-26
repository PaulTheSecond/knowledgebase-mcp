# 03. SQLite Concurrency & Safety Standards

## WAL (Write-Ahead Logging) Mode
Because the SQLite database is accessed concurrently by the Web Daemon (reads/writes) and the MCP stdio server (read-only queries from IDEs), connection PRAGMAs must be enforced:
```python
def configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
```
WAL mode guarantees that writers do not block readers, and readers never block writers.

## SQL Injection Protection
1. **Strictly Prohibited:** Constructing SQL statements via string concatenation or formatted f-strings:
   ```python
   # PROHIBITED! CRITICAL VULNERABILITY!
   cursor.execute(f"SELECT * FROM files WHERE path = '{user_input}';")
   ```
2. Parameterized bindings using standard SQLite placeholders (`?`) are mandatory:
   ```python
   # COMPLIANT
   cursor.execute("SELECT * FROM files WHERE path = ?;", (user_input,))
   ```

## Concurrent Write Access and Transaction Boundaries
1. All write transactions (INSERT, UPDATE, DELETE) must execute inside an explicit `BEGIN IMMEDIATE TRANSACTION` block to acquire the write lock up-front and eliminate deadlock risks:
   ```python
   conn.execute("BEGIN IMMEDIATE TRANSACTION;")
   ```
2. Every connection must specify an explicit busy timeout (`timeout=30.0` or `timeout=60.0`), preventing immediate `database is locked` exceptions during brief concurrent writes.
3. Database connections must always be managed via `with` context managers:
   ```python
   with get_db_connection(db_path) as conn:
       cursor = conn.cursor()
       cursor.execute("...")
   ```

## Read-Only Connections for MCP
The MCP server process running inside IDE environments executes read queries only. To prevent lock contention and accidental writes, connect using URI Read-Only mode:
```python
conn = sqlite3.connect("file:/data/knowledge.db?mode=ro", uri=True)
```

## Production Performance Pragmas
Write connections initialized in `get_db_connection` configure the following production pragmas:
```python
conn.execute("PRAGMA foreign_keys = ON;")
conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")
conn.execute("PRAGMA temp_store = MEMORY;")
conn.execute("PRAGMA cache_size = -64000;")  # 64 MB page cache in RAM
conn.execute("PRAGMA auto_vacuum = FULL;")
conn.execute("PRAGMA journal_mode = WAL;")
conn.execute("PRAGMA synchronous = NORMAL;")
```

## Batch Commit Strategy for Large Repositories
When batch indexing hundreds or thousands of files (e.g., in `Indexer.sync_repo`), do not hold an open write transaction across the entire repository.
Use periodic batch commits (`BATCH_COMMIT_SIZE = 50`):
```python
if processed % BATCH_COMMIT_SIZE == 0:
    self.db.commit_transaction()
    self.db.begin_transaction()
```
This bounds WAL file growth and prevents I/O errors (`sqlite3.OperationalError: disk I/O error`) on virtualized Docker bind-mount storage.
