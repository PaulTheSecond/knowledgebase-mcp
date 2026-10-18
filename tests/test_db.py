"""Тесты для KnowledgeDB: миграции, FTS5 external content table и auto_vacuum."""
import tempfile
import pytest
from pathlib import Path

from knowledge_mcp.db.models import KnowledgeDB


@pytest.fixture
def temp_db():
    """Создает временную БД и возвращает инстанс KnowledgeDB."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "test_knowledge.db"
        db = KnowledgeDB(db_file)
        yield db, db_file
        db.close()


class TestKnowledgeDB:
    def test_auto_vacuum_is_full(self, temp_db):
        """Проверяет, что auto_vacuum установлен в FULL (1)."""
        db, _ = temp_db
        mode = db.get_vacuum_mode()
        assert mode == 1, f"Expected auto_vacuum=1 (FULL), got {mode}"

    def test_fts5_external_content_table_no_duplicate_content(self, temp_db):
        """Проверяет, что chunks_fts_content отсутствует (нет дублирования текстов)."""
        db, _ = temp_db
        cursor = db.conn.cursor()
        tables = [
            row["name"]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "chunks_fts" in tables
        assert "chunks_fts_content" not in tables, "chunks_fts_content should NOT exist in external content mode"

    def test_fts5_insert_and_search(self, temp_db):
        """Проверяет полнотекстовый поиск по вставленным чанкам."""
        db, _ = temp_db
        file_id = db.upsert_file("repo1", "src/auth.py", 100.0, "hash1")
        db.add_chunk(
            chunk_id="chk_1",
            file_id=file_id,
            content="def authenticate_user(username, password):\n    return True",
            line_start=1,
            line_end=2,
            source_kind="code",
            trust="high",
        )

        # Поиск по термину из чанка
        results = db.search_chunks_fts("authenticate_user")
        assert len(results) == 1
        assert results[0]["id"] == "chk_1"
        assert "authenticate_user" in results[0]["content"]
        assert results[0]["path"] == "src/auth.py"

    def test_fts5_update_triggers(self, temp_db):
        """Проверяет обновление FTS5 при изменении контента чанка."""
        db, _ = temp_db
        file_id = db.upsert_file("repo1", "src/service.py", 100.0, "hash2")
        db.add_chunk(
            chunk_id="chk_2",
            file_id=file_id,
            content="class LegacyPaymentGateway: pass",
            line_start=1,
            line_end=1,
            source_kind="code",
            trust="high",
        )

        assert len(db.search_chunks_fts("LegacyPaymentGateway")) == 1

        # Обновляем чанк
        cursor = db.conn.cursor()
        cursor.execute(
            "UPDATE chunks SET content = 'class ModernPaymentGateway: pass' WHERE id = 'chk_2'"
        )
        db.conn.commit()

        # Старый термин не должен находиться
        assert len(db.search_chunks_fts("LegacyPaymentGateway")) == 0
        # Новый термин должен находиться
        modern_results = db.search_chunks_fts("ModernPaymentGateway")
        assert len(modern_results) == 1
        assert modern_results[0]["id"] == "chk_2"

    def test_fts5_delete_file_cascade(self, temp_db):
        """Проверяет, что удаление файла каскадно удаляет чанки из FTS5."""
        db, _ = temp_db
        file_id = db.upsert_file("repo1", "src/obsolete.py", 100.0, "hash3")
        db.add_chunk(
            chunk_id="chk_3",
            file_id=file_id,
            content="def obsolete_algorithm(): return 42",
            line_start=1,
            line_end=1,
            source_kind="code",
            trust="high",
        )

        assert len(db.search_chunks_fts("obsolete_algorithm")) == 1

        # Удаляем файл
        db.delete_file(file_id)

        # Чанк должен исчезнуть из FTS5
        assert len(db.search_chunks_fts("obsolete_algorithm")) == 0

    def test_vacuum_and_optimize(self, temp_db):
        """Проверяет успешное выполнение vacuum и optimize."""
        db, _ = temp_db
        # vacuum и optimize не должны выбрасывать исключений
        db.vacuum()
        db.optimize()
