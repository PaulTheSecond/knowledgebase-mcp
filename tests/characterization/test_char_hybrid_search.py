"""
Характеризационные тесты для гибридного поиска и RRF:
search_chunks_hybrid в knowledge_mcp.db.models.KnowledgeDB.
Фиксирует слияние FTS5, Vector и Graph поиска перед рефакторингом.
"""

import tempfile
from pathlib import Path
import pytest

from knowledge_mcp.db.models import KnowledgeDB


class TestCharacterizationHybridSearch:

    @pytest.fixture
    def test_db(self):
        """Создает изолированную базу данных с FTS5 и таблицами миграций."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_search.db"
            db = KnowledgeDB(db_path, read_only=False)
            yield db
            db.close()

    def test_fts5_text_search_ranks_matching_chunks(self, test_db):
        """Проверяет полнотекстовый поиск FTS5 по ключевым словам."""
        file_id = test_db.upsert_file(
            repo_id="repo1",
            path="docs/architecture.md",
            mtime=1000.0,
            file_hash="hash_doc"
        )
        test_db.add_chunk(
            chunk_id="chunk_auth",
            file_id=file_id,
            content="Authentication service uses JWT tokens for security and auth verification",
            source_kind="markdown",
            trust="verified"
        )
        test_db.add_chunk(
            chunk_id="chunk_db",
            file_id=file_id,
            content="Database connection pool configuration with SQLite WAL mode",
            source_kind="markdown",
            trust="verified"
        )

        results = test_db.search_chunks_hybrid(query="Authentication JWT", query_vector=None)
        assert len(results) >= 1
        assert results[0]["id"] == "chunk_auth"

    def test_rrf_combines_vector_and_fts_scores(self, test_db):
        """
        Проверяет, что чанк B, подтвержденный обоими источниками (FTS5 + Vector),
        получает сумму очков RRF и побеждает чанк A, найденный только в FTS5.
        """
        file_id = test_db.upsert_file(
            repo_id="repo1",
            path="src/Payments.cs",
            mtime=1000.0,
            file_hash="hash_pay"
        )

        # Чанк A: содержит ключевое слово, но не имеет вектора
        c_a = test_db.add_chunk("chunk_A", file_id, "BillingProcessor handles legacy transactions", "code", "verified")
        # Чанк B: содержит ключевое слово И имеет векторное совпадение
        c_b = test_db.add_chunk("chunk_B", file_id, "BillingProcessor executes modern credit card payments", "code", "verified")

        # Добавим вектор только для чанка B
        q_vec = [0.1] * 384
        vec_b = [0.1] * 384
        test_db.add_embedding(c_b, vec_b)

        # Выполняем гибридный поиск с вектором запроса
        results = test_db.search_chunks_hybrid(query="BillingProcessor", query_vector=q_vec, limit=5)

        assert len(results) >= 2
        # Чанк B получил скор из FTS5 (1/(60+2)) + скор из Vector (1/(60+1)) -> побеждает чанк A (только 1/(60+1))
        assert results[0]["id"] == "chunk_B"
        assert results[1]["id"] == "chunk_A"

    def test_search_filters_by_repo_id(self, test_db):
        """Проверяет изоляцию результатов по repo_ids."""
        f1 = test_db.upsert_file(repo_id="repo_frontend", path="src/App.tsx", mtime=1.0, file_hash="h1")
        f2 = test_db.upsert_file(repo_id="repo_backend", path="src/App.cs", mtime=1.0, file_hash="h2")

        test_db.add_chunk("chunk_fe", f1, "CommonApplicationComponent in frontend", "code", "verified")
        test_db.add_chunk("chunk_be", f2, "CommonApplicationComponent in backend", "code", "verified")

        fe_results = test_db.search_chunks_hybrid(query="CommonApplicationComponent", repo_ids=["repo_frontend"])
        assert len(fe_results) == 1
        assert fe_results[0]["id"] == "chunk_fe"
        assert fe_results[0]["repo_id"] == "repo_frontend"
