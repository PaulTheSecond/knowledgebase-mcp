"""Архитектурные фитнес-тесты (Fitness Functions) кодовой базы Knowledge Base MCP.

Гарантируют:
1. Соблюдение лимитов на размер файлов (никаких монолитов > 800 строк в отрефакторенных модулях).
2. Чистоту доменного слоя Domain от веб/MCP фреймворков (DDD boundary rule).
3. Соответствие репозиториев инфраструктурному интерфейсу.
"""

from pathlib import Path
import pytest


def test_refactored_modules_size_limits():
    """Проверяет, что ключевые отрефакторенные модули не превышают лимит строк."""
    base_dir = Path(__file__).parent.parent / "knowledge_mcp"

    server_lines = len((base_dir / "mcp" / "server.py").read_text(encoding="utf-8").splitlines())
    assert server_lines <= 800, f"server.py разросся до {server_lines} строк (лимит: 800)"

    models_lines = len((base_dir / "db" / "models.py").read_text(encoding="utf-8").splitlines())
    assert models_lines <= 800, f"models.py разросся до {models_lines} строк (лимит: 800)"


def test_domain_layer_isolation_from_frameworks():
    """Проверяет, что доменный слой domain/review не зависит от MCP или веб-фреймворков."""
    domain_dir = Path(__file__).parent.parent / "knowledge_mcp" / "domain"
    forbidden_tokens = ["import mcp", "from mcp", "fastmcp", "starlette", "fastapi"]

    for py_file in domain_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8").lower()
        for token in forbidden_tokens:
            assert token not in content, (
                f"Нарушение чистоты DDD: доменный файл {py_file.name} содержит зависимость от '{token}'"
            )


def test_repositories_inherit_base():
    """Проверяет, что все репозитории наследуются от BaseSqliteRepository."""
    from knowledge_mcp.infrastructure.db.repositories.base import BaseSqliteRepository
    from knowledge_mcp.infrastructure.db.repositories.document_repo import SqliteDocumentRepository
    from knowledge_mcp.infrastructure.db.repositories.job_queue_repo import SqliteJobQueueRepository
    from knowledge_mcp.infrastructure.db.repositories.symbol_repo import SqliteSymbolRepository
    from knowledge_mcp.infrastructure.db.repositories.vector_repo import SqliteVectorRepository

    for repo_cls in [
        SqliteDocumentRepository,
        SqliteSymbolRepository,
        SqliteVectorRepository,
        SqliteJobQueueRepository,
    ]:
        assert issubclass(repo_cls, BaseSqliteRepository), f"{repo_cls.__name__} не наследует BaseSqliteRepository"
