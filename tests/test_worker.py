import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import time
from pathlib import Path

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.daemon.worker import DaemonWorker

class TestDaemonWorker(unittest.TestCase):
    def setUp(self):
        # Создаем временную БД
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.db = KnowledgeDB(self.db_path)
        
        # Создаем временный каталог для клонов
        self.temp_dir = tempfile.TemporaryDirectory()
        self.clones_dir = Path(self.temp_dir.name)
        
        # Переопределяем переменную окружения для клонов в тесте
        os.environ["CLONES_DIR"] = str(self.clones_dir)

    def tearDown(self):
        self.db.close()
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.temp_dir.cleanup()

    @patch("knowledge_mcp.daemon.worker.get_repository_config")
    @patch("knowledge_mcp.daemon.worker.GitManager")
    @patch("knowledge_mcp.daemon.worker.Indexer")
    @patch("subprocess.run")
    def test_worker_flow_and_idempotency(self, mock_sub_run, mock_indexer_class, mock_git_mgr_class, mock_get_config):
        # Setup mocks
        mock_get_config.return_value = {
            "id": "test-repo",
            "url": "https://github.com/mock/repo.git",
            "branch": "master",
            "auth": {"type": "pat", "token_env": "GITHUB_PAT"}
        }
        os.environ["GITHUB_PAT"] = "ghp_testtoken"

        mock_git_mgr = MagicMock()
        mock_git_mgr_class.return_value = mock_git_mgr

        mock_indexer = MagicMock()
        mock_indexer_class.return_value = mock_indexer

        # Mock Git SHA return
        mock_sub_run.return_value = MagicMock(stdout="commit_sha_12345\n", returncode=0)

        # 1. Инициализируем воркер
        worker = DaemonWorker(self.db_path)
        worker.indexer = mock_indexer  # Заменяем на мокнутый индексер

        # 2. Добавляем задачу в очередь
        job_id = self.db.add_to_queue("test-repo")
        
        # Проверяем, что задача в статусе pending
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT status FROM sync_queue WHERE id = ?", (job_id,))
        self.assertEqual(cursor.fetchone()[0], "pending")

        # 3. Извлекаем и запускаем задачу
        job = self.db.fetch_next_job()
        self.assertIsNotNone(job)
        self.assertEqual(job["repo_id"], "test-repo")

        # После извлечения статус должен измениться на processing
        cursor.execute("SELECT status FROM sync_queue WHERE id = ?", (job_id,))
        self.assertEqual(cursor.fetchone()[0], "processing")

        # Запускаем задачу
        worker.run_job(job)

        # Проверяем вызовы
        mock_git_mgr.clone.assert_called_once_with("https://github.com/mock/repo.git", self.clones_dir / "test-repo")
        mock_indexer.sync_repo.assert_called_once_with("test-repo", str(self.clones_dir / "test-repo"))
        
        # Задача должна завершиться успешно
        cursor.execute("SELECT status, error_message FROM sync_queue WHERE id = ?", (job_id,))
        status, err = cursor.fetchone()
        self.assertEqual(status, "completed")
        self.assertIsNone(err)

        # SHA зафиксирован в indexed_commits
        self.assertEqual(self.db.get_indexed_commit("test-repo"), "commit_sha_12345")

        # --- ТЕСТ ИДЕМПОТЕНТНОСТИ ---
        # Снова добавляем задачу для того же репозитория
        job_id_2 = self.db.add_to_queue("test-repo")
        
        # Сбросим моки, чтобы проверить, что индексер не вызывался повторно
        mock_indexer.sync_repo.reset_mock()
        mock_git_mgr.clone.reset_mock()
        # Для повторного запуска git_manager.pull должен вызываться вместо clone
        mock_git_mgr.pull.return_value = None
        
        # Нам нужно убедиться, что каталог существует, чтобы сработал pull
        (self.clones_dir / "test-repo").mkdir(parents=True, exist_ok=True)
        (self.clones_dir / "test-repo" / ".git").mkdir(parents=True, exist_ok=True)

        job_2 = self.db.fetch_next_job()
        worker.run_job(job_2)

        # git_manager.pull должен быть вызван
        mock_git_mgr.pull.assert_called_once_with(self.clones_dir / "test-repo", branch="master")
        # Но индексер sync_repo НЕ должен вызываться, так как SHA не изменился
        mock_indexer.sync_repo.assert_not_called()

        # Задача 2 сразу помечена как completed
        cursor.execute("SELECT status FROM sync_queue WHERE id = ?", (job_id_2,))
        self.assertEqual(cursor.fetchone()[0], "completed")
        worker.db.close()

if __name__ == "__main__":
    unittest.main()
