"""Тесты для CAS (Content-Addressable Storage), миграции multi-worktree и телеметрии экономии токенов."""
import tempfile
import pytest
from pathlib import Path

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.db.connection import get_db_connection


@pytest.fixture
def temp_db():
    """Создает временную БД и возвращает инстанс KnowledgeDB."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "test_cas_knowledge.db"
        db = KnowledgeDB(db_file)
        yield db, db_file
        db.close()


class TestCASModels:
    def test_migration_20261101_12_cas_and_multi_worktree(self, temp_db):
        """Тест применения миграции 20261101_12_cas_and_multi_worktree и проверки схемы."""
        db, _ = temp_db
        cursor = db.conn.cursor()

        # 1. Проверяем регистрацию в __migrations_history
        cursor.execute("SELECT id FROM __migrations_history WHERE id = '20261101_12_cas_and_multi_worktree'")
        row = cursor.fetchone()
        assert row is not None, "Migration 20261101_12_cas_and_multi_worktree should be applied"

        # 2. Проверяем наличие новых таблиц
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r["name"] for r in cursor.fetchall()}
        assert "file_contents" in tables, "Table file_contents should exist"
        assert "token_savings_log" in tables, "Table token_savings_log should exist"

        # 3. Проверяем колонки в files
        cursor.execute("PRAGMA table_info(files)")
        files_cols = {r["name"] for r in cursor.fetchall()}
        assert "project_id" in files_cols, "files should have project_id"
        assert "branch" in files_cols, "files should have branch"
        assert "content_hash" in files_cols, "files should have content_hash"

        # 4. Проверяем колонку content_hash в chunks и symbols
        cursor.execute("PRAGMA table_info(chunks)")
        chunks_cols = {r["name"] for r in cursor.fetchall()}
        assert "content_hash" in chunks_cols, "chunks should have content_hash"

        cursor.execute("PRAGMA table_info(symbols)")
        symbols_cols = {r["name"] for r in cursor.fetchall()}
        assert "content_hash" in symbols_cols, "symbols should have content_hash"

        # 5. Проверяем колонки в token_savings_log
        cursor.execute("PRAGMA table_info(token_savings_log)")
        ts_cols = {r["name"] for r in cursor.fetchall()}
        expected_ts_cols = {"id", "timestamp", "tool_name", "project_id", "repo_id", "branch", "baseline_tokens", "returned_tokens", "saved_tokens", "saved_percent"}
        assert expected_ts_cols.issubset(ts_cols)

    def test_migration_from_legacy_data(self):
        """Тест бережной миграции существующих данных: файлы, чанки и символы связываются с content_hash."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "legacy_migration.db"

            # 1. Создаем базу с начальной схемой (до CAS-миграции)
            conn = get_db_connection(db_file)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS __migrations_history (
                    id TEXT PRIMARY KEY,
                    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    hash TEXT NOT NULL,
                    UNIQUE(repo_id, path)
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    file_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    line_start INTEGER,
                    line_end INTEGER,
                    source_kind TEXT NOT NULL,
                    trust TEXT NOT NULL,
                    sha TEXT,
                    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    qualified_name TEXT,
                    kind TEXT NOT NULL,
                    language TEXT NOT NULL,
                    line_start INTEGER,
                    line_end INTEGER,
                    signature TEXT,
                    chunk_id TEXT,
                    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
                );
            """)
            # Помечаем первые миграции как примененные, кроме CAS
            for mid in [
                "20260319_01_initial", "20260319_02_fts_and_vec", "20260326_03_fix_triggers",
                "20260414_04_symbols_graph", "20260420_05_upgrade_to_e5_small", "20260630_06_system_status",
                "20260701_07_sync_queue_and_indexed_commits", "20260701_08_sync_queue_run_at",
                "20261015_09_idx_chunks_file", "20261015_10_sync_queue_changed_files",
                "20261018_11_fts5_external_content_and_autovacuum", "20261018_12_app_settings",
                "20261019_13_sync_queue_durations"
            ]:
                cursor.execute("INSERT INTO __migrations_history (id) VALUES (?)", (mid,))

            # Вставляем устаревшие данные
            cursor.execute("INSERT INTO files (id, repo_id, path, mtime, hash) VALUES (1, 'legacy_repo', 'main.py', 123.0, 'sha256_legacy_1')")
            cursor.execute("INSERT INTO chunks (id, file_id, content, line_start, line_end, source_kind, trust) VALUES ('c1', 1, 'print(\"legacy\")', 1, 1, 'code', 'high')")
            cursor.execute("INSERT INTO symbols (id, file_id, name, kind, language) VALUES (1, 1, 'main', 'function', 'python')")
            conn.commit()
            conn.close()

            # 2. Открываем через KnowledgeDB, что триггерит применение миграции 20261101_12_cas_and_multi_worktree
            db = KnowledgeDB(db_file)

            # Проверяем файлы
            file_row = db.get_file("legacy_repo", "main.py")
            assert file_row is not None
            assert file_row["project_id"] == "default"
            assert file_row["branch"] == "master"
            assert file_row["content_hash"] == "sha256_legacy_1"

            # Проверяем file_contents
            fc = db.get_file_content("sha256_legacy_1")
            assert fc is not None
            assert fc["content"] == 'print("legacy")'

            # Проверяем foreign keys
            fk_check = db.conn.cursor().execute("PRAGMA foreign_key_check").fetchall()
            assert len(fk_check) == 0, f"Foreign key check failed: {fk_check}"

            db.close()

    def test_cas_deduplication_and_stats(self, temp_db):
        """Тест дедупликации контента: добавление одного и того же файла в master и orbit-1035 дает 50% экономии."""
        db, _ = temp_db

        content = "export function calculateTotal(items: number[]) { return items.reduce((a, b) => a + b, 0); }"
        content_hash = db.upsert_file_content(content, language="typescript")
        assert len(content_hash) == 64

        # Добавляем один и тот же файл в две ветки
        f1_id = db.upsert_file(
            repo_id="ImpactOS.Web",
            path="src/calc.ts",
            mtime=100.0,
            file_hash=content_hash,
            project_id="orbita",
            branch="master",
            content_hash=content_hash
        )

        f2_id = db.upsert_file(
            repo_id="ImpactOS.Web",
            path="src/calc.ts",
            mtime=100.0,
            file_hash=content_hash,
            project_id="orbita",
            branch="orbit-1035",
            content_hash=content_hash
        )

        assert f1_id != f2_id, "Files in different branches should have unique file_ids"

        # Проверяем записи на уровне БД: 2 записи в files, 1 запись в file_contents
        cursor = db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM files WHERE project_id = 'orbita'")
        assert cursor.fetchone()[0] == 2

        cursor.execute("SELECT COUNT(*) FROM file_contents WHERE hash = ?", (content_hash,))
        assert cursor.fetchone()[0] == 1

        # Проверяем статистику дедупликации: экономия 50%
        stats = db.get_cas_stats(project_id="orbita")
        assert stats["logical_files_count"] == 2
        assert stats["unique_contents_count"] == 1
        expected_size = len(content.encode("utf-8"))
        assert stats["real_size_bytes"] == expected_size
        assert stats["virtual_size_bytes"] == expected_size * 2
        assert stats["saved_size_bytes"] == expected_size
        assert stats["saved_percent"] == 50.0

        # Статистика без фильтра по проекту также показывает 50%
        stats_all = db.get_cas_stats()
        assert stats_all["saved_percent"] == 50.0

    def test_backward_compatibility(self, temp_db):
        """Тест обратной совместимости: старые вызовы без указания project_id и branch продолжают работать прозрачно."""
        db, _ = temp_db

        # Старый вызов upsert_file с 4 аргументами
        file_id = db.upsert_file("legacy-repo", "src/auth.py", 500.0, "legacy_hash_xyz")
        assert file_id > 0

        # Старый вызов get_known_files
        known = db.get_known_files("legacy-repo")
        assert "src/auth.py" in known
        assert known["src/auth.py"]["hash"] == "legacy_hash_xyz"

        # get_file по умолчанию ищет с project_id='default', branch='master'
        file_row = db.get_file("legacy-repo", "src/auth.py")
        assert file_row is not None
        assert file_row["project_id"] == "default"
        assert file_row["branch"] == "master"

        # Добавляем чанк
        chunk_id = "chk_legacy_99"
        db.add_chunk(chunk_id, file_id, "def authenticate(): return True", "code", "high", line_start=1, line_end=2)

        # get_file_content по пути и repo_id возвращает текст
        content = db.get_file_content("src/auth.py", repo_id="legacy-repo")
        assert content == "def authenticate(): return True"

        # search_chunks_fts без указания project_id и branch
        fts_res = db.search_chunks_fts("authenticate")
        assert len(fts_res) == 1
        assert fts_res[0]["path"] == "src/auth.py"

        # search_chunks универсальный
        search_res = db.search_chunks("authenticate")
        assert len(search_res) == 1
        assert search_res[0]["path"] == "src/auth.py"

        # delete_file по file_id
        assert db.delete_file(file_id) is True
        assert db.get_file("legacy-repo", "src/auth.py") is None

    def test_token_savings_telemetry(self, temp_db):
        """Тест методов телеметрии экономии токенов: log_token_savings и get_token_savings_summary."""
        db, _ = temp_db

        # Логируем вызовы разных инструментов
        id1 = db.log_token_savings(
            tool_name="knowledge_get_project_rules",
            baseline_tokens=25000,
            returned_tokens=1000,
            project_id="orbita",
            repo_id="ImpactOS",
            branch="master"
        )
        assert id1 > 0

        id2 = db.log_token_savings(
            tool_name="knowledge_search",
            baseline_tokens=10000,
            returned_tokens=2000,
            project_id="orbita",
            repo_id="ImpactOS",
            branch="orbit-1035"
        )
        assert id2 > 0

        id3 = db.log_token_savings(
            tool_name="knowledge_read_file",
            baseline_tokens=20000,
            returned_tokens=5000,
            project_id="other_project",
            repo_id="Lib",
            branch="master"
        )
        assert id3 > 0

        # Сводка по проекту 'orbita'
        summary_orbita = db.get_token_savings_summary(days=7, project_id="orbita")
        assert summary_orbita["total_calls"] == 2
        # (25000 - 1000) + (10000 - 2000) = 24000 + 8000 = 32000
        assert summary_orbita["total_saved_tokens"] == 32000
        assert summary_orbita["total_baseline_tokens"] == 35000
        assert summary_orbita["total_returned_tokens"] == 3000
        assert "knowledge_get_project_rules" in summary_orbita["by_tool"]
        assert summary_orbita["by_tool"]["knowledge_get_project_rules"]["total_saved_tokens"] == 24000
        assert "knowledge_search" in summary_orbita["by_tool"]
        assert summary_orbita["by_tool"]["knowledge_search"]["total_saved_tokens"] == 8000

        # Сводка без фильтрации по проекту (все проекты)
        summary_all = db.get_token_savings_summary(days=7)
        assert summary_all["total_calls"] == 3
        # 32000 + (20000 - 5000) = 47000
        assert summary_all["total_saved_tokens"] == 47000

    def test_symbol_edges_tested_by_support(self, temp_db):
        """Тест поддержки kind='TESTED_BY' в symbol_edges и методов get_symbol_callers/callees."""
        db, _ = temp_db

        f_id = db.upsert_file("repo1", "OrderService.cs", 10.0, "hash_ord", project_id="orbita", branch="master")
        s_service = db.add_symbol(f_id, "OrderService", "Orders.OrderService", "class", "csharp", 1, 50, "class OrderService", None)
        s_test = db.add_symbol(f_id, "OrderServiceTests", "Orders.Tests.OrderServiceTests", "class", "csharp", 1, 30, "class OrderServiceTests", None)

        # Добавляем ребро TESTED_BY
        db.add_symbol_edge(s_test, s_service, "TESTED_BY")

        # Проверяем get_symbol_callers с фильтром по branch
        callers = db.get_symbol_callers(s_service, project_id="orbita", branch="master")
        assert len(callers) == 1
        assert callers[0]["id"] == s_test
        assert callers[0]["edge_kind"] == "TESTED_BY"

        # Проверяем get_symbol_callees
        callees = db.get_symbol_callees(s_test, project_id="orbita", branch="master")
        assert len(callees) == 1
        assert callees[0]["id"] == s_service
        assert callees[0]["edge_kind"] == "TESTED_BY"

        # Проверяем старый метод get_callers
        old_callers = db.get_callers(s_service)
        assert len(old_callers) == 1
        assert old_callers[0]["id"] == s_test

    def test_cas_get_and_clean_unreferenced(self, temp_db):
        """Тест методов upsert_file_content, get_file_content(hash) и clean_unreferenced_contents."""
        db, _ = temp_db

        content = "print('temp unreferenced')"
        h = db.upsert_file_content(content, language="python")

        # get_file_content по hash возвращает dict
        res = db.get_file_content(h)
        assert isinstance(res, dict)
        assert res["hash"] == h
        assert res["content"] == content
        assert res["language"] == "python"
        assert res["size"] == len(content.encode("utf-8"))

        # Контент не привязан ни к одному файлу -> clean_unreferenced_contents удаляет его
        cleaned = db.clean_unreferenced_contents()
        assert cleaned == 1
        assert db.get_file_content(h) is None
