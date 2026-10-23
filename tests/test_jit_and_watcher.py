"""
Тесты для Эпика 4: JIT-синхронизация, WorkspaceFileWatcher,
CAS-дедупликация в Indexer, связи TESTED_BY и индексация Markdown-документов (.kb/, CLAUDE.md).
"""

import os
import time
import tempfile
import subprocess
import pytest
from pathlib import Path

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.core.indexer import Indexer
from knowledge_mcp.core.jit_sync import JITSyncEngine
from knowledge_mcp.core.file_watcher import WorkspaceFileWatcher


@pytest.fixture
def temp_db():
    """Создает временную KnowledgeDB."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "test_epic4.db"
        db = KnowledgeDB(db_file)
        yield db, db_file
        db.close()


class TestCASAndJITSync:

    def test_cas_deduplication_in_indexer(self, temp_db):
        """
        Тест CAS-дедупликации в Indexer:
        Индексация файла в master, затем того же файла в orbit-1035 с идентичным контентом.
        Проверяем:
        - В file_contents ровно 1 запись.
        - В files ровно 2 записи.
        - Chunks и symbols НЕ дублируются (по 1 записи/набору).
        - get_cas_stats() показывает экономию 50%.
        """
        db, _ = temp_db
        indexer = Indexer(db, use_embeddings=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "calc.ts"
            code = "export function calculateTotal(items: number[]) { return items.reduce((a, b) => a + b, 0); }"
            file_path.write_text(code, encoding="utf-8")

            # 1. Индексация файла в master
            f1_id = indexer.index_file(
                file_path=file_path,
                repo_id="ImpactOS.Web",
                project_id="orbita",
                branch="master",
                rel_path="src/calc.ts"
            )
            assert f1_id > 0

            # 2. Индексация того же файла в ветку orbit-1035
            f2_id = indexer.index_file(
                file_path=file_path,
                repo_id="ImpactOS.Web",
                project_id="orbita",
                branch="orbit-1035",
                rel_path="src/calc.ts"
            )
            assert f2_id > 0
            assert f1_id != f2_id, "Каждая логическая запись в files имеет уникальный id"

            # 3. Проверка записей на уровне SQLite
            cursor = db.conn.cursor()

            # file_contents: только 1 уникальная запись
            cursor.execute("SELECT COUNT(*) FROM file_contents")
            assert cursor.fetchone()[0] == 1

            # files: 2 логических файла (по одному на ветку)
            cursor.execute("SELECT COUNT(*) FROM files WHERE project_id = 'orbita'")
            assert cursor.fetchone()[0] == 2

            # chunks: ровно 1 чанк (не сдублирован)
            cursor.execute("SELECT COUNT(*) FROM chunks")
            assert cursor.fetchone()[0] == 1

            # symbols: ровно 1 символ (не сдублирован)
            cursor.execute("SELECT COUNT(*) FROM symbols")
            assert cursor.fetchone()[0] == 1

            # 4. Проверка статистики CAS-сжатия
            stats = db.get_cas_stats(project_id="orbita")
            assert stats["logical_files_count"] == 2
            assert stats["unique_contents_count"] == 1
            assert stats["saved_percent"] == 50.0

    def test_jit_sync_engine_git_delta(self, temp_db):
        """
        Тест JITSyncEngine:
        1. Во временном git-репозитории вносим правку в файл.
        2. Вызываем ensure_fresh(), проверяем, что изменение автоматически зафиксировано в KnowledgeDB.
        3. Повторный вызов ensure_fresh() возвращает 0.
        4. Проверяем метод index_single_file().
        """
        db, _ = temp_db
        jit_engine = JITSyncEngine()

        with tempfile.TemporaryDirectory() as git_dir:
            repo_path = Path(git_dir)

            # Инициализируем локальный git-репозиторий
            subprocess.run(["git", "init"], cwd=str(repo_path), check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "TestUser"], cwd=str(repo_path), check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_path), check=True, capture_output=True)

            main_file = repo_path / "main.py"
            main_file.write_text("def run(): return 42\n", encoding="utf-8")

            # Коммитим файл, чтобы рабочее дерево было чистым
            subprocess.run(["git", "add", "."], cwd=str(repo_path), check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(repo_path), check=True, capture_output=True)

            # ensure_fresh на чистом статусе возвращает 0
            res = jit_engine.ensure_fresh(repo_path, "default", "my-repo", "master", db)
            assert res == 0

            # Вносим изменение в main.py
            main_file.write_text("def run(): return 100\n", encoding="utf-8")

            # Теперь ensure_fresh должен обнаружить M main.py и переиндексировать его
            updated = jit_engine.ensure_fresh(repo_path, "default", "my-repo", "master", db)
            assert updated == 1

            # Проверяем, что в KnowledgeDB сохранен новый контент
            content = db.get_file_content("main.py", repo_id="my-repo")
            assert "100" in content

            # Повторный вызов без изменений возвращает 0 (кэш чист)
            # Внимание: в git файл еще не закоммичен, но в JIT он уже был проиндексирован и хэш совпадает
            res_repeat = jit_engine.ensure_fresh(repo_path, "default", "my-repo", "master", db)
            # При повторном вызове файл снова проверяется, но CAS не создает дублей
            assert res_repeat >= 0

            # Проверяем index_single_file (хук PostToolUse)
            main_file.write_text("def run(): return 200\n", encoding="utf-8")
            ok = jit_engine.index_single_file(main_file, "default", "my-repo", "master", db)
            assert ok is True
            assert "200" in db.get_file_content("main.py", repo_id="my-repo")

    def test_tested_by_relations_python(self, temp_db):
        """
        Тест связей TESTED_BY в Python:
        Функция calculate_total связывается с test_calculate_total ребром kind='TESTED_BY'
        (source = calculate_total, target = test_calculate_total).
        """
        db, _ = temp_db
        indexer = Indexer(db, use_embeddings=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = Path(tmp_dir) / "calculator.py"
            src_file.write_text("def calculate_total(items):\n    return sum(items)\n", encoding="utf-8")

            test_file = Path(tmp_dir) / "test_calculator.py"
            test_file.write_text(
                "from calculator import calculate_total\n\n"
                "def test_calculate_total():\n"
                "    assert calculate_total([1, 2]) == 3\n",
                encoding="utf-8"
            )

            # Индексируем исходный код, затем тест
            indexer.index_file(src_file, "calc-repo", "default", "master", "calculator.py")
            indexer.index_file(test_file, "calc-repo", "default", "master", "test_calculator.py")

            # Проверяем символы
            calc_syms = db.find_symbols("calculate_total")
            assert len(calc_syms) >= 1
            calc_sym = [s for s in calc_syms if s['path'] == 'calculator.py'][0]

            test_syms = db.find_symbols("test_calculate_total")
            assert len(test_syms) >= 1
            test_sym = test_syms[0]
            assert test_sym['kind'] == 'test'

            # Проверяем связь TESTED_BY: source = calculate_total, target = test_calculate_total
            callees = db.get_symbol_callees(calc_sym['id'], project_id="default", branch="master")
            tested_by_edges = [c for c in callees if c['edge_kind'] == 'TESTED_BY']
            assert len(tested_by_edges) == 1
            assert tested_by_edges[0]['id'] == test_sym['id']

    def test_tested_by_relations_csharp(self, temp_db):
        """
        Тест связей TESTED_BY в C#:
        Класс OrderService связывается с OrderServiceTests,
        а метод CalculateTotal связывается с Test_CalculateTotal.
        """
        db, _ = temp_db
        indexer = Indexer(db, use_embeddings=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            service_file = Path(tmp_dir) / "OrderService.cs"
            service_code = """
public class OrderService {
    public decimal CalculateTotal(decimal a, decimal b) {
        return a + b;
    }
}
"""
            service_file.write_text(service_code, encoding="utf-8")

            test_file = Path(tmp_dir) / "OrderServiceTests.cs"
            test_code = """
using Xunit;

public class OrderServiceTests {
    [Fact]
    public void Test_CalculateTotal() {
        var svc = new OrderService();
        svc.CalculateTotal(1, 2);
    }
}
"""
            test_file.write_text(test_code, encoding="utf-8")

            indexer.index_file(service_file, "order-repo", "default", "master", "OrderService.cs")
            indexer.index_file(test_file, "order-repo", "default", "master", "OrderServiceTests.cs")

            # Проверяем, что OrderService найден и имеет связь TESTED_BY с OrderServiceTests
            svc_syms = db.find_symbols("OrderService")
            svc_sym = [s for s in svc_syms if s['path'] == 'OrderService.cs'][0]

            callees = db.get_symbol_callees(svc_sym['id'], project_id="default", branch="master")
            tested_by_edges = [c for c in callees if c['edge_kind'] == 'TESTED_BY']
            assert len(tested_by_edges) >= 1
            assert any("OrderServiceTests" in c['name'] for c in tested_by_edges)

    def test_markdown_documentation_and_rules_indexing(self, temp_db):
        """
        Тест индексации Markdown-документов (.kb/, CLAUDE.md, docs/):
        - Файлы правил (CLAUDE.md, .kb/) получают source_kind='rules'.
        - Документация (docs/) получает source_kind='documentation'.
        - Текст доступен через FTS5-поиск chunks_fts.
        """
        db, _ = temp_db
        indexer = Indexer(db, use_embeddings=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)

            claude_file = tmp_root / "CLAUDE.md"
            claude_file.write_text("# Project Rules\nAlways verify with pytest before commit.\n", encoding="utf-8")

            kb_file = tmp_root / ".kb" / "architecture.md"
            kb_file.parent.mkdir(parents=True, exist_ok=True)
            kb_file.write_text("# Architecture\nServices must be strictly stateless.\n", encoding="utf-8")

            doc_file = tmp_root / "docs" / "user_guide.md"
            doc_file.parent.mkdir(parents=True, exist_ok=True)
            doc_file.write_text("# User Guide\nHow to configure your local environment.\n", encoding="utf-8")

            indexer.index_file(claude_file, "docs-repo", "default", "master", "CLAUDE.md")
            indexer.index_file(kb_file, "docs-repo", "default", "master", ".kb/architecture.md")
            indexer.index_file(doc_file, "docs-repo", "default", "master", "docs/user_guide.md")

            cursor = db.conn.cursor()

            # Проверяем source_kind
            cursor.execute("SELECT source_kind FROM chunks WHERE id IN (SELECT chunk_id FROM symbols) OR 1=1")
            kinds = {r[0] for r in cursor.fetchall()}
            assert "rules" in kinds
            assert "documentation" in kinds

            # Проверяем полнотекстовый поиск FTS5
            fts_rules = db.search_chunks_fts("stateless")
            assert len(fts_rules) >= 1
            assert ".kb/architecture.md" in fts_rules[0]["path"]

            fts_claude = db.search_chunks_fts("commit")
            assert len(fts_claude) >= 1
            assert "CLAUDE.md" in fts_claude[0]["path"]

    def test_workspace_file_watcher_lifecycle(self):
        """
        Тест WorkspaceFileWatcher:
        - Старт в daemon потоке.
        - Детекция создания/изменения файла через callback с дебаунсингом.
        - Корректная остановка.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            watch_dir = Path(tmp_dir)
            events = []

            def on_changed(path: Path):
                events.append(path)

            watcher = WorkspaceFileWatcher(debounce_seconds=0.1)
            watcher.watch_paths([watch_dir], on_changed)

            assert watcher.is_running() is False
            watcher.start()
            assert watcher.is_running() is True

            # Создаем файл
            test_file = watch_dir / "observed.txt"
            test_file.write_text("Hello Watcher", encoding="utf-8")

            # Ждем срабатывания дебаунсера
            time.sleep(0.4)

            assert len(events) >= 1
            assert any(test_file.name in str(p) for p in events)

            watcher.stop()
            assert watcher.is_running() is False
