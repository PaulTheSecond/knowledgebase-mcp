import unittest
from unittest.mock import patch
import os
import tempfile
from pathlib import Path

from knowledge_mcp.db.models import KnowledgeDB


class TestIndexerSyncAndRoslyn(unittest.TestCase):
    def setUp(self):
        # Создаем временную БД
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.db = KnowledgeDB(self.db_path)

        # Создаем временный каталог репозитория
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.db.close()
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.temp_dir.cleanup()

    def test_roslyn_path_resolution(self):
        """Проверка корректного вычисления путей к RoslynParser относительно корня проекта."""
        from knowledge_mcp.core.indexer import Path as IndexerPath
        app_root = IndexerPath(__file__).resolve().parents[1]
        parser_dll = app_root / "RoslynParser" / "bin" / "Release" / "net8.0" / "RoslynParser.dll"
        parser_csproj = app_root / "RoslynParser" / "RoslynParser.csproj"

        # В корне репозитория должен существовать RoslynParser.csproj
        self.assertTrue(parser_csproj.exists(), f"RoslynParser.csproj не найден по пути: {parser_csproj}")
        # Проверяем структуру пути к DLL
        self.assertEqual(parser_dll.name, "RoslynParser.dll")
        self.assertIn("RoslynParser", parser_dll.parts)

    def test_get_known_files_with_chunks_and_symbols_count(self):
        """Проверка возврата chunks_count и symbols_count в get_known_files."""
        repo_id = "test-repo"
        file_path = "Services/TestService.cs"

        # 1. Добавляем запись о файле без чанков и символов
        file_id = self.db.upsert_file(repo_id, file_path, 123456.0, "fakehash123")
        known = self.db.get_known_files(repo_id)

        self.assertIn(file_path, known)
        self.assertEqual(known[file_path]["chunks_count"], 0)
        self.assertEqual(known[file_path]["symbols_count"], 0)

        # 2. Добавляем чанк
        self.db.add_chunk("chunk-1", file_id, "public class TestService {}", "code", "verified", 1, 10)
        known = self.db.get_known_files(repo_id)
        self.assertEqual(known[file_path]["chunks_count"], 1)
        self.assertEqual(known[file_path]["symbols_count"], 0)

        # 3. Добавляем символ
        self.db.add_symbol(file_id, "TestService", "TestNamespace.TestService", "class", "csharp", 1, 10, "public class TestService", "chunk-1")
        known = self.db.get_known_files(repo_id)
        self.assertEqual(known[file_path]["chunks_count"], 1)
        self.assertEqual(known[file_path]["symbols_count"], 1)

    @patch("knowledge_mcp.core.indexer.LocalEmbedder")
    def test_orphan_csharp_file_forces_reindex(self, mock_embedder):
        """Проверка дельта-скана: файл-сирота .cs (0 чанков или 0 символов) переводится в updated даже при совпадающем mtime."""
        from knowledge_mcp.core.indexer import Indexer

        indexer = Indexer(self.db, use_embeddings=False)

        # Создаем .cs файл в тестовом репозитории
        cs_file = self.repo_dir / "Test.cs"
        cs_file.write_text("public class Test { public void Run() {} }", encoding="utf-8")
        mtime = cs_file.stat().st_mtime

        repo_id = "repo-orphan-test"
        # Регистрируем файл в таблице files (как будто парсер упал в прошлый раз)
        self.db.upsert_file(repo_id, "Test.cs", mtime, "somehash")

        # Наполняем кэш Roslyn (как если бы RoslynParser успешно починили и он отдал символы)
        indexer.csharp_cache = {
            cs_file.resolve(): {
                "symbols": [
                    {
                        "name": "Test",
                        "qualified_name": "Test",
                        "kind": "class",
                        "language": "csharp",
                        "line_start": 1,
                        "line_end": 1,
                        "signature": "public class Test",
                        "body": "public class Test { public void Run() {} }",
                        "ast_node_id": "test_id"
                    }
                ],
                "edges": []
            }
        }

        # Запускаем синхронизацию БЕЗ флага force
        with patch.object(indexer, "_run_roslyn_analysis", return_value=(True, None)):
            indexer.sync_repo(repo_id, self.repo_dir, force=False)

        # Проверяем, что файл был переиндексирован и теперь имеет чанки и символы
        known = self.db.get_known_files(repo_id)
        self.assertEqual(known["Test.cs"]["chunks_count"], 1)
        self.assertEqual(known["Test.cs"]["symbols_count"], 1)

    @patch("knowledge_mcp.core.indexer.LocalEmbedder")
    def test_force_flag_reindexes_unchanged_files(self, mock_embedder):
        """Проверка: флаг force=True переиндексирует неизмененные файлы."""
        from knowledge_mcp.core.indexer import Indexer

        indexer = Indexer(self.db, use_embeddings=False)

        py_file = self.repo_dir / "main.py"
        py_file.write_text("print('hello world')", encoding="utf-8")
        mtime = py_file.stat().st_mtime

        repo_id = "repo-force-test"
        file_id = self.db.upsert_file(repo_id, "main.py", mtime, "hash123")
        self.db.add_chunk("chunk-py-1", file_id, "print('hello world')", "code", "verified", 1, 1)

        # При force=False файл неизменен
        with patch.object(indexer, "_flush_parsed_file") as mock_flush:
            indexer.sync_repo(repo_id, self.repo_dir, force=False)
            mock_flush.assert_not_called()

        # При force=True файл обязательно отправляется на парсинг и запись
        with patch.object(indexer, "_flush_parsed_file") as mock_flush:
            mock_flush.return_value = 1  # ReindexStatus
            indexer.sync_repo(repo_id, self.repo_dir, force=True)
            mock_flush.assert_called()

    def test_cli_non_tty_defaults_to_batch_mode(self):
        """Проверка: отсутствие TTY (sys.stdin.isatty() == False) переводит CLI sync в пакетный режим без вызова questionary."""
        with patch("sys.stdin.isatty", return_value=False), \
             patch("sys.argv", ["main.py", "--db-path", self.db_path, "sync", "--repo-id", "cli-test", "--repo-path", str(self.repo_dir)]), \
             patch("knowledge_mcp.main.init_components"), \
             patch("knowledge_mcp.main.indexer") as mock_indexer:

            from knowledge_mcp.main import main
            # Не должно упасть с EOFError или запросить stdin
            main()

            # sync_repo должен быть вызван с allowed_top_level=None (все каталоги) и force=False
            mock_indexer.sync_repo.assert_called_once_with("cli-test", str(self.repo_dir), allowed_top_level=None, force=False)
