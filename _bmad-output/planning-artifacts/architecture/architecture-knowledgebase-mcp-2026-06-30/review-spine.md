# Architecture Review — Autonomous Sync & Dashboard Spine

## Overall Verdict
The architectural spine is well-designed. Invariants AD-1 through AD-4 clearly define process boundaries and security requirements. Critical adversarial review identified one operational vulnerability: direct SQLite write access from the stdio MCP process caused transaction deadlocks and race conditions against the background worker.

---

## 1. Technology & Reality Check
All selected technologies and version constraints are valid and production-grade:
* `sqlite-vec` represents the modern standard for local vector indexing in SQLite, replacing the deprecated `sqlite-vss`.
* Python >= 3.11 requirement is justified by native `tomllib` usage.
* Git `-c http.extraHeader` is supported across all modern Git versions (Git 2.9+), making the AD-4 in-memory credential injection highly portable.

---

## 2. Adversarial Review & Attack Simulation
* **Attack Vector:**
  * Process 1 (Daemon) executes background synchronization of repository `Yurta.Core.Lib`, reading AST nodes and writing vector chunks to SQLite.
  * Process 2 (stdio MCP client in IDE) receives a command to delete this repository (`knowledge_delete_repo`). In unhardened designs, the MCP process issues deletes directly over its SQLite connection (`db.delete_repo(...)`).
  * **Result:** Transaction lockup (`database is locked`) or data corruption (cascading deletes while the worker commits).
* **Vulnerability Remediation:**
  * **Rule for AD-1:** All mutating database operations (including purging repositories via `knowledge_delete_repo`) must execute **exclusively through Web Daemon HTTP endpoints**. The stdio MCP process operates strictly in Read-Only mode for all search tools and delegates mutations to the Web Daemon over HTTP.
