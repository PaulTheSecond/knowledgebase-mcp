# 06. Testing & Quality Standards

## The pytest Framework
All unit and integration tests are authored and executed using `pytest`.

## Arrange-Act-Assert (AAA) Pattern
Structure test cases according to the AAA convention:
* **Arrange:** set up test inputs, fixtures, mocks, and environment variables.
* **Act:** invoke the target function or unit under test.
* **Assert:** verify return values, system state, and exception expectations.

### Example:
```python
def test_webhook_signature_validation_success():
    # Arrange
    payload = b'{"repo_id": "test"}'
    secret = "my-secret"
    valid_signature = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    
    # Act
    is_valid = verify_signature(payload, secret, valid_signature)
    
    # Assert
    assert is_valid is True
```

## Mocks and Fixtures
1. Tests must never mutate production codebases or make live external network requests.
2. Interacting with the Git CLI or HTTP endpoints (e.g., OmniRoute / external LLM APIs) must be mocked using `unittest.mock` or `pytest-mock`:
   ```python
   def test_git_manager_clone(mocker):
       # Mock subprocess.run
       mock_run = mocker.patch("subprocess.run")
       
       # Trigger clone
       clone_repo("https://github.com/org/repo", "/tmp/dest")
       
       # Verify Git command was executed properly
       mock_run.assert_called_once()
       args = mock_run.call_args[0][0]
       assert "clone" in args
   ```
3. For database unit tests, use SQLite in-memory databases (`:memory:`) or isolated temporary database files automatically cleaned up post-test.
