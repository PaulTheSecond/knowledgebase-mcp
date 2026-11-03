"""
Unit-тесты для репозитория документов SqliteDocumentRepository:
knowledge_mcp/infrastructure/db/repositories/document_repo.py
"""

from pathlib import Path
import pytest
from knowledge_mcp.db.models import KnowledgeDB


def test_document_repository_crud(tmp_path: Path):
    """Проверка основных операций репозитория документов (upsert, get, delete, stats, clean)."""
    db_file = tmp_path / "test_doc_repo.db"
    db = KnowledgeDB(db_file)
    doc_repo = db.documents

    # 1. upsert_file_content
    sha = doc_repo.upsert_file_content("hello world content", language="python")
    assert isinstance(sha, str)
    assert len(sha) == 64

    # 2. upsert_file
    file_id = doc_repo.upsert_file(
        repo_id="test_repo",
        path="src/main.py",
        mtime=123456.0,
        file_hash=sha,
        content_hash=sha,
        project_id="default",
        branch="master"
    )
    assert file_id > 0
    doc_repo.invalidate_stats_cache()

    # 3. has_indexed_files
    assert doc_repo.has_indexed_files("default", "test_repo", "master") is True
    assert doc_repo.has_indexed_files("default", "non_existent", "master") is False

    # 4. get_file and get_file_content
    f_info = doc_repo.get_file("test_repo", "src/main.py", project_id="default", branch="master")
    assert f_info is not None
    assert f_info["hash"] == sha

    content = doc_repo.get_file_content("src/main.py", repo_id="test_repo", project_id="default", branch="master")
    assert content == "hello world content"

    # get_file_content напрямую по хэшу
    content_by_hash = doc_repo.get_file_content(sha)
    assert isinstance(content_by_hash, dict)
    assert content_by_hash["content"] == "hello world content"

    # 5. get_known_files
    known = doc_repo.get_known_files("test_repo", project_id="default", branch="master")
    assert "src/main.py" in known
    assert known["src/main.py"]["hash"] == sha

    # 6. get_cas_stats and get_all_repo_stats
    cas_stats = doc_repo.get_cas_stats(project_id="default", force_refresh=True)
    assert cas_stats["logical_files_count"] >= 1

    all_stats = doc_repo.get_all_repo_stats(force_refresh=True)
    assert ("test_repo", "default") in all_stats

    repo_stats = doc_repo.get_repo_stats("test_repo", project_id="default", branch="master")
    assert repo_stats["files_count"] >= 1

    # 7. delete_file
    del_ok = doc_repo.delete_file("test_repo", "src/main.py", project_id="default", branch="master")
    assert del_ok is True
    assert doc_repo.get_file("test_repo", "src/main.py", project_id="default", branch="master") is None

    # 8. clean_unreferenced_contents
    cleaned = doc_repo.clean_unreferenced_contents()
    assert cleaned >= 1

    # 9. delete_repo
    # Добавляем файл заново и удаляем весь репозиторий
    doc_repo.upsert_file("repo_to_del", "file.txt", 123.0, sha, "default", "master", sha)
    del_count = doc_repo.delete_repo("repo_to_del", project_id="default")
    assert del_count >= 1

    db.close()
