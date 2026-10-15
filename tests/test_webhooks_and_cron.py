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
    def test_find_ssh_config_matched_by_https_webhook(self, mock_config):
        """Проверяет, что репозиторий с SSH URL в config.yaml корректно находится по HTTPS clone URL из вебхука."""
        from knowledge_mcp.daemon.routes.webhooks import _find_repo_config_by_url

        mock_config.return_value = {
            "repositories": [
                {"id": "ssh-repo", "url": "git@github.com:PaulTheSecond/knowledgebase-mcp.git"}
            ]
        }

        # Вебхук от GitHub присылает HTTPS clone_url
        result = _find_repo_config_by_url("https://github.com/PaulTheSecond/knowledgebase-mcp.git")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "ssh-repo")

    def test_normalize_git_url(self):
        from knowledge_mcp.daemon.routes.webhooks import normalize_git_url

        expected = "github.com/org/repo"
        self.assertEqual(normalize_git_url("https://github.com/org/repo.git"), expected)
        self.assertEqual(normalize_git_url("http://github.com/org/repo/"), expected)
        self.assertEqual(normalize_git_url("git@github.com:org/repo.git"), expected)
        self.assertEqual(normalize_git_url("ssh://git@github.com/org/repo.git"), expected)
        self.assertEqual(normalize_git_url("ssh://git@github.com:22/org/repo.git"), expected)

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


class TestWebhookEndpoint(unittest.TestCase):
    """Интеграционные тесты эндпоинта /webhook/git."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.db = KnowledgeDB(self.db_path)

        from fastapi import FastAPI
        from knowledge_mcp.daemon.routes.webhooks import init_webhook_router
        self.app = FastAPI()
        self.app.include_router(init_webhook_router(self.db))

        from starlette.testclient import TestClient
        self.client = TestClient(self.app)
        os.environ["WEBHOOK_SECRET"] = "super_secret_123"

    def tearDown(self):
        self.db.close()
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if "WEBHOOK_SECRET" in os.environ:
            del os.environ["WEBHOOK_SECRET"]

    @patch("knowledge_mcp.daemon.routes.webhooks.load_config")
    def test_github_webhook_success(self, mock_config):
        """Проверяет успешный прием и постановку в очередь вебхука GitHub."""
        mock_config.return_value = {
            "repositories": [
                {
                    "id": "my-lib",
                    "url": "https://github.com/PaulTheSecond/knowledgebase-mcp.git",
                    "triggers": {
                        "webhooks": {
                            "enabled": True,
                            "secret_env": "WEBHOOK_SECRET"
                        }
                    }
                }
            ]
        }
        import json
        payload_data = {"repository": {"clone_url": "https://github.com/PaulTheSecond/knowledgebase-mcp.git"}}
        raw_body = json.dumps(payload_data).encode("utf-8")
        sig = "sha256=" + hmac.new(b"super_secret_123", raw_body, hashlib.sha256).hexdigest()

        response = self.client.post(
            "/webhook/git",
            content=raw_body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "queued")
        self.assertEqual(data["repo_id"], "my-lib")

        # Проверяем наличие задачи в БД
        job = self.db.fetch_next_job()
        self.assertIsNotNone(job)
        self.assertEqual(job["repo_id"], "my-lib")

    @patch("knowledge_mcp.daemon.routes.webhooks.load_config")
    def test_webhook_disabled_for_repo(self, mock_config):
        """Проверяет возврат 403, если вебхуки отключены (enabled: false)."""
        mock_config.return_value = {
            "repositories": [
                {
                    "id": "disabled-repo",
                    "url": "https://github.com/PaulTheSecond/knowledgebase-mcp.git",
                    "triggers": {
                        "webhooks": {
                            "enabled": False,
                            "secret_env": "WEBHOOK_SECRET"
                        }
                    }
                }
            ]
        }
        import json
        payload_data = {"repository": {"clone_url": "https://github.com/PaulTheSecond/knowledgebase-mcp.git"}}
        raw_body = json.dumps(payload_data).encode("utf-8")
        sig = "sha256=" + hmac.new(b"super_secret_123", raw_body, hashlib.sha256).hexdigest()

        response = self.client.post(
            "/webhook/git",
            content=raw_body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig
            }
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("disabled", response.json()["detail"])

    def test_webhook_missing_signature_header(self):
        """Проверяет возврат 401 при отсутствии заголовков подписи."""
        response = self.client.post("/webhook/git", json={"repository": {"clone_url": "https://github.com/user/repo"}})
        self.assertEqual(response.status_code, 401)
        self.assertIn("Missing signature", response.json()["detail"])

    @patch("knowledge_mcp.daemon.routes.webhooks.load_config")
    def test_webhook_invalid_signature(self, mock_config):
        """Проверяет возврат 401 при неверной подписи HMAC."""
        mock_config.return_value = {
            "repositories": [
                {
                    "id": "my-lib",
                    "url": "https://github.com/user/repo.git",
                    "triggers": {
                        "webhooks": {"enabled": True, "secret_env": "WEBHOOK_SECRET"}
                    }
                }
            ]
        }
        response = self.client.post(
            "/webhook/git",
            json={"repository": {"clone_url": "https://github.com/user/repo.git"}},
            headers={"X-Hub-Signature-256": "sha256=wrongsignature123"}
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid signature", response.json()["detail"])

    @patch("knowledge_mcp.daemon.routes.webhooks.load_config")
    def test_gitlab_webhook_success(self, mock_config):
        """Проверяет успешный прием вебхука от GitLab с токеном."""
        mock_config.return_value = {
            "repositories": [
                {
                    "id": "gitlab-lib",
                    "url": "https://gitlab.com/group/repo.git",
                    "triggers": {
                        "webhooks": {"enabled": True, "secret_env": "WEBHOOK_SECRET"}
                    }
                }
            ]
        }
        payload = {"project": {"git_http_url": "https://gitlab.com/group/repo.git"}}
        response = self.client.post(
            "/webhook/git",
            json=payload,
            headers={"X-Gitlab-Token": "super_secret_123"}
        )
        self.assertEqual(response.status_code, 200)
    @patch("knowledge_mcp.daemon.routes.webhooks.load_config")
    def test_webhook_omitted_enabled_defaults_to_false(self, mock_config):
        """Проверяет, что при отсутствии ключа enabled по умолчанию считается False (403)."""
        mock_config.return_value = {
            "repositories": [
                {
                    "id": "unspecified-repo",
                    "url": "https://github.com/PaulTheSecond/knowledgebase-mcp.git",
                    "triggers": {
                        "webhooks": {"secret_env": "WEBHOOK_SECRET"}
                    }
                }
            ]
        }
        import json
        payload_data = {"repository": {"clone_url": "https://github.com/PaulTheSecond/knowledgebase-mcp.git"}}
        raw_body = json.dumps(payload_data).encode("utf-8")
        sig = "sha256=" + hmac.new(b"super_secret_123", raw_body, hashlib.sha256).hexdigest()

        response = self.client.post(
            "/webhook/git",
            content=raw_body,
            headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig}
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("disabled", response.json()["detail"])

    @patch("knowledge_mcp.daemon.routes.webhooks.load_config")
    def test_gitea_webhook_success(self, mock_config):
        """Проверяет успешный прием вебхука от Gitea."""
        mock_config.return_value = {
            "repositories": [
                {
                    "id": "gitea-lib",
                    "url": "https://gitea.local/group/repo.git",
                    "triggers": {
                        "webhooks": {"enabled": True, "secret_env": "WEBHOOK_SECRET"}
                    }
                }
            ]
        }
        import json
        payload = {"repository": {"clone_url": "https://gitea.local/group/repo.git"}}
        raw_body = json.dumps(payload).encode("utf-8")
        sig = "sha256=" + hmac.new(b"super_secret_123", raw_body, hashlib.sha256).hexdigest()

        response = self.client.post(
            "/webhook/git",
            content=raw_body,
            headers={"Content-Type": "application/json", "X-Gitea-Signature": sig}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provider"], "gitea")
        self.assertEqual(response.json()["repo_id"], "gitea-lib")


if __name__ == "__main__":
    unittest.main()

