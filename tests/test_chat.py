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
        if not any(getattr(route, "path", "").startswith("/api/chat") for route in app.routes):
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

    def test_stream_unauthorized(self):
        req_body = {
            "messages": [
                {"role": "user", "content": "Hello, code?"}
            ]
        }
        # Без токена
        response = self.client.post("/api/chat/stream", json=req_body)
        self.assertEqual(response.status_code, 401)
        
        # С неверным токеном
        response = self.client.post("/api/chat/stream?token=invalid_token_123", json=req_body)
        self.assertEqual(response.status_code, 401)

    @patch("httpx.AsyncClient.stream")
    @patch("knowledge_mcp.daemon.routes.chat.load_config")
    def test_stream_success(self, mock_load_config, mock_stream):
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
        
        # Генерируем валидный токен
        from knowledge_mcp.daemon.auth import generate_temp_token
        valid_token = generate_temp_token("admin")
        
        # Настраиваем мок эмбеддера
        self.mock_embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        
        # Настраиваем мок поиска в БД
        self.mock_db.search_chunks_hybrid.return_value = [
            {
                "path": "test_stream.py",
                "line_start": 1,
                "line_end": 5,
                "content": "def stream(): pass"
            }
        ]
        
        # Настраиваем мок stream менеджера контекста httpx
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        async def mock_aiter_lines():
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " world"}}]}'
            yield 'data: [DONE]'
            
        mock_response.aiter_lines = mock_aiter_lines
        
        class MockStreamContext:
            async def __aenter__(self):
                return mock_response
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
                
        mock_stream.return_value = MockStreamContext()
        
        req_body = {
            "messages": [
                {"role": "user", "content": "Test stream endpoint"}
            ]
        }
        
        # Делаем запрос к эндпоинту
        response = self.client.post(f"/api/chat/stream?token={valid_token}", json=req_body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        
        # Читаем возвращенный стрим
        lines = [line for line in response.iter_lines() if line]
        
        # Должно быть 3 непустых строки в стриме: "Hello", " world", и "sources"
        self.assertTrue(any("Hello" in line for line in lines))
        self.assertTrue(any(" world" in line for line in lines))
        self.assertTrue(any("sources" in line for line in lines))
        self.assertTrue(any("test_stream.py" in line for line in lines))

    def test_read_file_unauthorized(self):
        response = self.client.get("/api/chat/file?filepath=data/config.yaml")
        self.assertEqual(response.status_code, 401)

    def test_read_file_not_found(self):
        os.environ["DASHBOARD_PASSWORD"] = "secret"
        response = self.client.get("/api/chat/file?filepath=non_existent_file.py", auth=("admin", "secret"))
        self.assertEqual(response.status_code, 404)

    def test_read_file_path_traversal_blocked(self):
        os.environ["DASHBOARD_PASSWORD"] = "secret"
        # Попытка выйти к системным файлам вне проекта
        response = self.client.get("/api/chat/file?filepath=../../../../Windows/System32/drivers/etc/hosts", auth=("admin", "secret"))
        self.assertEqual(response.status_code, 403)

    def test_read_file_success(self):
        os.environ["DASHBOARD_PASSWORD"] = "secret"
        # Читаем data/config.yaml
        response = self.client.get("/api/chat/file?filepath=data/config.yaml&start_line=1&end_line=3", auth=("admin", "secret"))
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["filepath"], "data/config.yaml")
        self.assertEqual(data["start_line"], 1)
        self.assertEqual(data["end_line"], 3)
        self.assertIn("server:", data["content"])
