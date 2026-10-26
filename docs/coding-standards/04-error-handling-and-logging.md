# 04. Error Handling & Logging Standards

## Exception Handling
1. **Strictly Prohibited:** Catching all exceptions with a bare `except:` block:
   ```python
   # PROHIBITED! Hides syntax errors, NameError, and KeyboardInterrupt
   try:
       do_something()
   except:
       pass
   ```
2. If catching a top-level error, explicitly target `Exception` (which preserves system interrupts):
   ```python
   # COMPLIANT (for top-level boundary logging only)
   try:
       do_something()
   except Exception as e:
       logger.error(f"Unexpected error: {e}")
   ```
3. Always catch the most specific exception appropriate for the failure mode (e.g., `FileNotFoundError`, `sqlite3.IntegrityError`).
4. When re-raising exceptions, preserve the original traceback using explicit exception chaining:
   ```python
   try:
       conn.execute("INSERT ...")
   except sqlite3.IntegrityError as e:
       raise QueueInsertError("Task already exists in queue") from e
   ```

## Logging
1. Do not use `print()` for diagnostic logging or error reporting. Always use the standard Python `logging` module.
2. Initialize loggers using module-level scopes:
   ```python
   import logging
   logger = logging.getLogger(__name__)
   ```
3. **Critical Invariant for the MCP Server:**
   The stdio MCP server communicates with the client IDE via standard input and output streams (`sys.stdout` and `sys.stdin`).
   * Any direct write to `sys.stdout` (via `print` or an incorrectly configured root logger) **corrupts the MCP JSON-RPC protocol stream**.
   * All log records, errors, and debug statements must route exclusively to standard error (`sys.stderr`).
   * Example logger configuration for stdio MCP:
     ```python
     logging.basicConfig(stream=sys.stderr, level=logging.INFO)
     ```
4. Choose appropriate log levels:
   * `DEBUG`: detailed diagnostic info for local troubleshooting.
   * `INFO`: major lifecycle milestones (daemon startup, job enqueued, index completed).
   * `WARNING`: non-fatal anomalies (webhook signature mismatch, transient retry attempt).
   * `ERROR`: fatal failures requiring administrative attention (job exhausted all 3 retries, database write failure).
