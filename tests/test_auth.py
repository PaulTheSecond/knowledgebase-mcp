"""Тесты для модуля аутентификации и авторизации (Story 2.1)."""
import unittest
from unittest.mock import patch, MagicMock
import os
import time
from fastapi import HTTPException
from fastapi.testclient import TestClient

from knowledge_mcp.daemon.auth import (
    get_dashboard_password,
    verify_basic_auth,
    generate_temp_token,
    verify_temp_token,
)
from fastapi.security import HTTPBasicCredentials
from knowledge_mcp.main import app
import knowledge_mcp.main as main_mod


class TestAuth(unittest.TestCase):
    def setUp(self):
        self.original_password = os.environ.get("DASHBOARD_PASSWORD")
        if "DASHBOARD_PASSWORD" in os.environ:
            del os.environ["DASHBOARD_PASSWORD"]

    def tearDown(self):
        if self.original_password is not None:
            os.environ["DASHBOARD_PASSWORD"] = self.original_password
        elif "DASHBOARD_PASSWORD" in os.environ:
            del os.environ["DASHBOARD_PASSWORD"]

    def test_get_dashboard_password(self):
        self.assertIsNone(get_dashboard_password())
        os.environ["DASHBOARD_PASSWORD"] = "secret123"
        self.assertEqual(get_dashboard_password(), "secret123")

    def test_verify_basic_auth_disabled(self):
        # Если пароль отсутствует в env, должно быть 403 Forbidden
        credentials = HTTPBasicCredentials(username="admin", password="password")
        with self.assertRaises(HTTPException) as context:
            verify_basic_auth(credentials)
        self.assertEqual(context.exception.status_code, 403)
        self.assertIn("Dashboard disabled", context.exception.detail)

    def test_verify_basic_auth_incorrect(self):
        os.environ["DASHBOARD_PASSWORD"] = "correct_password"
        credentials = HTTPBasicCredentials(username="admin", password="wrong_password")
        with self.assertRaises(HTTPException) as context:
            verify_basic_auth(credentials)
        self.assertEqual(context.exception.status_code, 401)

    def test_verify_basic_auth_success(self):
        os.environ["DASHBOARD_PASSWORD"] = "correct_password"
        credentials = HTTPBasicCredentials(username="admin", password="correct_password")
        username = verify_basic_auth(credentials)
        self.assertEqual(username, "admin")

    def test_generate_and_verify_token_success(self):
        token = generate_temp_token("admin")
        self.assertEqual(len(token.split(".")), 3)
        self.assertTrue(verify_temp_token(token))

    def test_verify_token_invalid_signature(self):
        token = generate_temp_token("admin")
        parts = token.split(".")
        # Меняем сигнатуру
        parts[2] = "wrong_signature_123"
        invalid_token = ".".join(parts)
        self.assertFalse(verify_temp_token(invalid_token))

    def test_verify_token_expired(self):
        # Имитируем, что токен был создан 65 секунд назад
        with patch("time.time", return_value=time.time() - 65):
            expired_token = generate_temp_token("admin")
        # Проверяем в текущем времени (токен просрочен)
        self.assertFalse(verify_temp_token(expired_token))

    def test_verify_token_bad_format(self):
        self.assertFalse(verify_temp_token("invalid-format-token"))
        self.assertFalse(verify_temp_token("one.two"))
        self.assertFalse(verify_temp_token("one.two.three.four"))



class TestAuthAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.original_password = os.environ.get("DASHBOARD_PASSWORD")
        if "DASHBOARD_PASSWORD" in os.environ:
            del os.environ["DASHBOARD_PASSWORD"]
        
        # Мокаем db в main
        self.mock_db = MagicMock()
        self.original_db = main_mod.db
        main_mod.db = self.mock_db

    def tearDown(self):
        main_mod.db = self.original_db
        if self.original_password is not None:
            os.environ["DASHBOARD_PASSWORD"] = self.original_password
        elif "DASHBOARD_PASSWORD" in os.environ:
            del os.environ["DASHBOARD_PASSWORD"]

    def test_api_disabled_by_default(self):
        # Без пароля DASHBOARD_PASSWORD в env запросы возвращают 403
        response = self.client.get("/api/status", auth=("admin", "pass"))
        self.assertEqual(response.status_code, 403)
        self.assertIn("Dashboard disabled", response.json()["detail"])

    def test_api_unauthorized_with_wrong_pass(self):
        os.environ["DASHBOARD_PASSWORD"] = "secret"
        response = self.client.get("/api/status", auth=("admin", "wrong"))
        self.assertEqual(response.status_code, 401)

    def test_api_status_success(self):
        os.environ["DASHBOARD_PASSWORD"] = "secret"
        self.mock_db.get_system_status.return_value = {
            "status": "idle",
            "last_error": None,
            "last_indexed_at": "2026-07-02 12:00:00"
        }
        response = self.client.get("/api/status", auth=("admin", "secret"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "idle")

    def test_api_get_token(self):
        os.environ["DASHBOARD_PASSWORD"] = "secret"
        response = self.client.post("/api/auth/token", auth=("admin", "secret"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("token", response.json())
        token = response.json()["token"]
        self.assertTrue(verify_temp_token(token))

    def test_api_sync_repo(self):
        os.environ["DASHBOARD_PASSWORD"] = "secret"
        self.mock_db.add_to_queue.return_value = 42
        response = self.client.post("/api/sync/test-repo", auth=("admin", "secret"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job_id"], 42)
        self.mock_db.add_to_queue.assert_called_once_with("test-repo")

    def test_api_delete_repo(self):
        os.environ["DASHBOARD_PASSWORD"] = "secret"
        self.mock_db.delete_repo.return_value = 10
        response = self.client.delete("/api/repo/test-repo", auth=("admin", "secret"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["files_deleted"], 10)
        self.mock_db.delete_repo.assert_called_once_with("test-repo")

    def test_static_dashboard_rendering(self):
        # Проверяем, что корень возвращает страницу Dashboard
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Knowledge Base Dashboard", response.text)

        # Проверяем раздачу статических CSS стилей
        response = self.client.get("/static/css/styles.css")
        self.assertEqual(response.status_code, 200)
        self.assertIn("accent-color", response.text)


if __name__ == "__main__":
    unittest.main()
