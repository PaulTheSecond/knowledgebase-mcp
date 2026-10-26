# 05. Docstrings & Documentation Standards

## Google-Style Docstrings
All public modules, classes, methods, and functions must be documented using Google-Style docstrings.

### Function Docstring Example:
```python
def fetch_git_commit_sha(repo_path: str, branch: str = "main") -> str:
    """Retrieves the latest commit hash for the specified Git branch.

    Args:
        repo_path: Absolute filesystem path to the cloned repository.
        branch: Target branch name. Defaults to "main".

    Returns:
        String containing the full 40-character SHA-1 commit hash.

    Raises:
        ValueError: If the specified branch cannot be found in the repository.
        subprocess.SubprocessError: If the underlying Git CLI command fails.
    """
    # Implementation
```

### Class Docstring Example:
```python
class DatabaseManager:
    """Manages connections and transactions for the SQLite knowledge base.

    Attributes:
        db_path: Path to the SQLite database file on disk.
        connection: Currently open database connection or None.
    """

    def __init__(self, db_path: str) -> None:
        """Initializes the database manager with the given database path."""
        self.db_path = db_path
        self.connection = None
```

## Documentation Guidelines
1. **One-Line Docstrings:** Permissible for trivial properties or getters where meaning is unambiguous:
   ```python
   @property
   def is_active(self) -> bool:
       """Returns True if the worker is currently processing tasks."""
   ```
2. **Synchronization:** Whenever method signatures or behaviors change, update docstrings synchronously in the same commit.
3. **In-Code Comments:** Comments must explain *why* non-obvious code was written (edge-case workarounds, performance invariants) rather than narrating *what* the code does. Code structure itself should be clean and self-explanatory.
