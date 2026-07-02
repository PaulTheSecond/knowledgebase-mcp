import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
from fastapi.testclient import TestClient

import knowledge_mcp.daemon.routes.chat
from knowledge_mcp.main import app
import knowledge_mcp.main as main_mod

class TestChatAPI(unittest.TestCase):
    def setUp(self):
        # Вручную регистрируем роутер чата, если он еще не зарегистрирован
        from knowledge_mcp.daemon.routes.chat import router as chat_router
        if not any(route.path.startswith("/api/chat") for route in app.routes):
            app.include_router(chat_router)

        self.client = TestClient(app)
        self.original_password = os.environ.get("DASHBOARD_PASSWORD")
        self.original_api_key = os.environ.get("LLM_API_KEY")
        
        # Задаем DASHBOARD_PASSWORD
        os.environ["DASHBOARD_PASSWORD"] = "secret"
        
        # Мокаем db и indexer в main_mod
        self.mock_db = MagicMock()
        self.mock_indexer = MagicMock()
        self.mock_embedder = MagicMock()
        self.mock_indexer.embedder = self.mock_embedder
        
        self.original_db = main_mod.db
        self.original_indexer = main_mod.indexer
        
        main_mod.db = self.mock_db
        main_mod.indexer = self.mock_indexer

    def tearDown(self):
        # Восстанавливаем окружение
        if self.original_password is not None:
            os.environ["DASHBOARD_PASSWORD"] = self.original_password
        elif "DASHBOARD_PASSWORD" in os.environ:
            del os.environ["DASHBOARD_PASSWORD"]
            
        if self.original_api_key is not None:
            os.environ["LLM_API_KEY"] = self.original_api_key
        elif "LLM_API_KEY" in os.environ:
            del os.environ["LLM_API_KEY"]
            
        main_mod.db = self.original_db
        main_mod.indexer = self.original_indexer

    @patch("knowledge_mcp.daemon.routes.chat.load_config")
    def test_chat_disabled_by_feature_flag(self, mock_load_config):
        mock_load_config.return_value = {
            "server": {
                "dashboard": {
                    "chat": {
                        "enabled": False
                    }
                }
            }
        }
        
        req_body = {
            "messages": [
                {"role": "user", "content": "Hello, code?"}
            ]
        }
        response = self.client.post("/api/chat", json=req_body, auth=("admin", "secret"))
        self.assertEqual(response.status_code, 403)
        self.assertIn("Chat is disabled", response.json()["detail"])

    @patch("knowledge_mcp.daemon.routes.chat.load_config")
    def test_chat_missing_llm_api_key(self, mock_load_config):
        mock_load_config.return_value = {
            "server": {
                "dashboard": {
                    "chat": {
                        "enabled": True
                    }
                }
            }
        }
        if "LLM_API_KEY" in os.environ:
            del os.environ["LLM_API_KEY"]
            
        req_body = {
            "messages": [
                {"role": "user", "content": "Hello, code?"}
            ]
        }
        response = self.client.post("/api/chat", json=req_body, auth=("admin", "secret"))
        self.assertEqual(response.status_code, 500)
        self.assertIn("LLM_API_KEY", response.json()["detail"])

    @patch("httpx.AsyncClient.post")
    @patch("knowledge_mcp.daemon.routes.chat.load_config")
    def test_chat_success(self, mock_load_config, mock_post):
        mock_load_config.return_value = {
            "server": {
                "dashboard": {
                    "chat": {
                        "enabled": True
                    }
                }
            }
        }
        os.environ["LLM_API_KEY"] = "sk-test"
        
        # Настраиваем мок эмбеддера
        self.mock_embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        
        # Настраиваем мок поиска в БД
        self.mock_db.search_chunks_hybrid.return_value = [
            {
                "path": "test.py",
                "line_start": 5,
                "line_end": 10,
                "content": "def hello(): print('hi')"
            }
        ]
        
        # Настраиваем мок ответа httpx
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a response from Claude Sonnet"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        req_body = {
            "messages": [
                {"role": "user", "content": "Explain how hello works"}
            ]
        }
        
        response = self.client.post("/api/chat", json=req_body, auth=("admin", "secret"))
        self.assertEqual(response.status_code, 200)
        
        res_data = response.json()
        self.assertEqual(res_data["message"]["role"], "assistant")
        self.assertEqual(res_data["message"]["content"], "This is a response from Claude Sonnet")
        self.assertEqual(len(res_data["sources"]), 1)
        self.assertEqual(res_data["sources"][0]["filepath"], "test.py")
        
        # Проверяем вызовы
        self.mock_embedder.embed_text.assert_called_once_with("Explain how hello works", is_query=True)
        self.mock_db.search_chunks_hybrid.assert_called_once()
        mock_post.assert_called_once()
