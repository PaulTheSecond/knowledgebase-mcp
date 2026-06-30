"""Тесты для webhook-обработчика и cron-планировщика (Story 1.5)."""
import unittest
from unittest.mock import patch
import hmac
import hashlib
import os
import tempfile

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.daemon.scheduler import _parse_cron_interval, CronScheduler


class TestCronIntervalParser(unittest.TestCase):
    """Тесты парсера cron-выражений."""

    def test_every_5_minutes(self):
        self.assertEqual(_parse_cron_interval("*/5 * * * *"), 300)

    def test_every_4_hours(self):
        self.assertEqual(_parse_cron_interval("0 */4 * * *"), 14400)

    def test_daily_midnight(self):
        self.assertEqual(_parse_cron_interval("0 0 * * *"), 86400)

    def test_complex_fallback(self):
        # Сложное выражение — fallback 4 часа
        self.assertEqual(_parse_cron_interval("30 2 1 * *"), 14400)

    def test_invalid_expression(self):
        self.assertIsNone(_parse_cron_interval("bad"))


class TestCronScheduler(unittest.TestCase):
    """Тесты cron-планировщика."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.db = KnowledgeDB(self.db_path)

    def tearDown(self):
        self.db.close()
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    @patch("knowledge_mcp.daemon.scheduler.load_config")
    def test_scheduler_start_and_stop(self, mock_config):
        """Проверяет, что планировщик корректно создаёт и останавливает таймеры."""
        mock_config.return_value = {
            "repositories": [
                {
                    "id": "test-repo",
                    "triggers": {"cron": "*/5 * * * *"}
                }
            ]
        }

        scheduler = CronScheduler(self.db)
        scheduler.start()

        # Должен быть создан один таймер
        self.assertEqual(len(scheduler._timers), 1)
        self.assertTrue(scheduler._running)

        # Останавливаем
        scheduler.stop()
        self.assertFalse(scheduler._running)
        self.assertEqual(len(scheduler._timers), 0)


class TestWebhookSignature(unittest.TestCase):
    """Тесты HMAC-SHA256 валидации подписей."""

    def test_verify_hmac_sha256_valid(self):
        from knowledge_mcp.daemon.routes.webhooks import _verify_hmac_sha256

        payload = b'{"ref":"refs/heads/master"}'
        secret = "my-secret"
        sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

        self.assertTrue(_verify_hmac_sha256(payload, secret, sig))

    def test_verify_hmac_sha256_invalid(self):
        from knowledge_mcp.daemon.routes.webhooks import _verify_hmac_sha256

        payload = b'{"ref":"refs/heads/master"}'
        secret = "my-secret"

        self.assertFalse(_verify_hmac_sha256(payload, secret, "sha256=invalid_signature"))

    def test_verify_hmac_sha256_without_prefix(self):
        from knowledge_mcp.daemon.routes.webhooks import _verify_hmac_sha256

        payload = b'{"ref":"refs/heads/master"}'
        secret = "my-secret"
        # Подпись без префикса sha256=
        sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

        self.assertTrue(_verify_hmac_sha256(payload, secret, sig))


class TestFindRepoConfig(unittest.TestCase):
    """Тесты поиска конфигурации репозитория по URL."""

    @patch("knowledge_mcp.daemon.routes.webhooks.load_config")
    def test_find_by_exact_url(self, mock_config):
        from knowledge_mcp.daemon.routes.webhooks import _find_repo_config_by_url

        mock_config.return_value = {
            "repositories": [
                {"id": "my-repo", "url": "https://github.com/user/repo.git"}
            ]
        }

        result = _find_repo_config_by_url("https://github.com/user/repo.git")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "my-repo")

    @patch("knowledge_mcp.daemon.routes.webhooks.load_config")
    def test_find_without_git_suffix(self, mock_config):
        from knowledge_mcp.daemon.routes.webhooks import _find_repo_config_by_url

        mock_config.return_value = {
            "repositories": [
                {"id": "my-repo", "url": "https://github.com/user/repo.git"}
            ]
        }

        # URL без .git — должен тоже найтись
        result = _find_repo_config_by_url("https://github.com/user/repo")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "my-repo")

    @patch("knowledge_mcp.daemon.routes.webhooks.load_config")
    def test_not_found(self, mock_config):
        from knowledge_mcp.daemon.routes.webhooks import _find_repo_config_by_url

        mock_config.return_value = {
            "repositories": [
                {"id": "my-repo", "url": "https://github.com/user/repo.git"}
            ]
        }

        result = _find_repo_config_by_url("https://github.com/other/repo.git")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
