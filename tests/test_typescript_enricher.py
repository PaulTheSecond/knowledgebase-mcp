"""
Тесты для TypeScriptSemanticEnricher и двухфазной схемы индексации (ADR-009).
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.core.indexer import Indexer
from knowledge_mcp.core.enrichers.typescript_enricher import TypeScriptSemanticEnricher


class TestTypeScriptEnricher(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_knowledge.db"
        self.db = KnowledgeDB(self.db_path)
        self.enricher = TypeScriptSemanticEnricher(self.db)
        self.app_root = Path(__file__).resolve().parents[1]
        self.fixtures_dir = self.app_root / "TsParser" / "tests" / "fixtures"

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_can_enrich_detection(self):
        """Проверка детекции tsconfig.json в проекте."""
        self.assertTrue(self.enricher.can_enrich(self.fixtures_dir))

        # В пустой временной папке tsconfig отсутствует
        empty_dir = Path(self.temp_dir) / "empty_repo"
        empty_dir.mkdir()
        self.assertFalse(self.enricher.can_enrich(empty_dir))

    def test_full_pipeline_with_indexer(self):
        """Проверка полного пайплайна: Фаза 1 (Tree-sitter) + Фаза 2 (ts-parser обогащение)."""
        indexer = Indexer(self.db, use_embeddings=False)

        repo_id = "test-frontend"
        indexer.sync_repo(
            repo_id=repo_id,
            repo_path=self.fixtures_dir,
            project_id="default",
            branch="master",
            force=True
        )

        cursor = self.db.conn.cursor()

        # 1. Проверяем, что файлы были проиндексированы
        cursor.execute("SELECT path FROM files WHERE repo_id = ?", (repo_id,))
        indexed_files = [r["path"].replace("\\", "/") for r in cursor.fetchall()]
        self.assertTrue(any("UserCard.tsx" in f for f in indexed_files))
        self.assertTrue(any("Button.tsx" in f for f in indexed_files))

        # 2. Проверяем, что в БД появились точные компиляторные символы
        cursor.execute("""
            SELECT s.name, s.kind, s.qualified_name
            FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE f.repo_id = ?
        """, (repo_id,))
        symbols = {r["name"]: dict(r) for r in cursor.fetchall()}

        self.assertIn("UserCard", symbols, "Компонент UserCard должен быть проиндексирован")
        self.assertEqual(symbols["UserCard"]["kind"], "component")

        self.assertIn("Button", symbols, "Компонент Button должен быть проиндексирован")
        self.assertEqual(symbols["Button"]["kind"], "component")

        self.assertIn("useUserData", symbols, "Хук useUserData должен быть проиндексирован")
        self.assertEqual(symbols["useUserData"]["kind"], "hook")

        # 3. Проверяем наличие семантических связей в графе
        cursor.execute("""
            SELECT s1.name as source, s2.name as target, e.kind
            FROM symbol_edges e
            JOIN symbols s1 ON e.source_id = s1.id
            JOIN symbols s2 ON e.target_id = s2.id
        """)
        edges = [(r["source"], r["target"], r["kind"]) for r in cursor.fetchall()]

        # Проверяем также unresolved_refs, если ссылки еще не сошлись
        cursor.execute("""
            SELECT s.name as source, u.target_qualified_name as target, u.kind
            FROM unresolved_refs u
            JOIN symbols s ON u.source_id = s.id
        """)
        unresolved = [(r["source"], r["target"], r["kind"]) for r in cursor.fetchall()]
        all_relations = edges + unresolved

        # Связь JSX: UserCard использует Button (<Button ... />)
        has_button_usage = any(
            rel[0] == "UserCard" and rel[1] == "Button" and rel[2] == "USES"
            for rel in all_relations
        )
        self.assertTrue(has_button_usage, f"Связь USES между UserCard и Button должна присутствовать в графе. Найдено: {all_relations}")

        # Связь вызова: UserCard вызывает хук useUserData
        has_hook_call = any(
            rel[0] == "UserCard" and rel[1] == "useUserData" and rel[2] == "CALLS"
            for rel in all_relations
        )
        self.assertTrue(has_hook_call, f"Связь CALLS между UserCard и useUserData должна присутствовать в графе. Найдено: {all_relations}")


if __name__ == "__main__":
    unittest.main()
