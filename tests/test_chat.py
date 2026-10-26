import unittest
from unittest.mock import patch, MagicMock
import os
from fastapi.testclient import TestClient

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

    @patch("httpx.AsyncClient.post")
    @patch("knowledge_mcp.daemon.routes.chat.load_config")
    def test_chat_without_embedder_uses_lexical_search(self, mock_load_config, mock_post):
        """Проверяет работу чата, когда embedder не инициализирован (гибридный поиск FTS5+Graph)."""
        mock_load_config.return_value = {
            "server": {"dashboard": {"chat": {"enabled": True}}}
        }
        os.environ["LLM_API_KEY"] = "sk-test"
        self.mock_indexer.embedder = None
        
        self.mock_db.search_chunks_hybrid.return_value = [
            {"path": "core.py", "line_start": 1, "line_end": 10, "content": "class Core: pass"}
        ]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Answer without vector search"}}]
        }
        mock_post.return_value = mock_response

        req_body = {"messages": [{"role": "user", "content": "How Core works"}]}
        response = self.client.post("/api/chat", json=req_body, auth=("admin", "secret"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"]["content"], "Answer without vector search")
        self.mock_db.search_chunks_hybrid.assert_called_once_with("How Core works", None, limit=5)

    @patch("httpx.AsyncClient.post")
    @patch("knowledge_mcp.daemon.routes.chat.load_config")
    def test_chat_llm_connection_error_returns_friendly_message(self, mock_load_config, mock_post):
        """Проверяет корректную обработку ошибки подключения к LLM с возвратом цитат."""
        import httpx
        mock_load_config.return_value = {
            "server": {"dashboard": {"chat": {"enabled": True}}}
        }
        os.environ["LLM_API_KEY"] = "sk-test"
        
        self.mock_db.search_chunks_hybrid.return_value = [
            {"path": "models.py", "line_start": 10, "line_end": 20, "content": "class KnowledgeDB: pass"}
        ]
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        req_body = {"messages": [{"role": "user", "content": "Tell me about KnowledgeDB"}]}
        response = self.client.post("/api/chat", json=req_body, auth=("admin", "secret"))
        self.assertEqual(response.status_code, 200)
        res_json = response.json()
        self.assertIn("Не удалось подключиться к LLM-шлюзу", res_json["message"]["content"])
        self.assertEqual(len(res_json["sources"]), 1)
        self.assertEqual(res_json["sources"][0]["filepath"], "models.py")

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
        response = self.client.get("/api/chat/file?filepath=tests/test_chat.py")
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
        self.assertIn("Access denied", response.json()["detail"])

    def test_read_sensitive_file_blocked(self):
        os.environ["DASHBOARD_PASSWORD"] = "secret"
        # Попытка прочитать конфигурацию или секреты должна блокироваться с 403
        for sensitive in ["data/config.yaml", ".env", "data/knowledge.db"]:
            response = self.client.get(f"/api/chat/file?filepath={sensitive}", auth=("admin", "secret"))
            self.assertEqual(response.status_code, 403)
            self.assertIn("Access denied", response.json()["detail"])

    def test_read_file_success(self):
        os.environ["DASHBOARD_PASSWORD"] = "secret"
        # Читаем не конфиденциальный файл проекта (tests/test_chat.py)
        response = self.client.get("/api/chat/file?filepath=tests/test_chat.py&start_line=1&end_line=3", auth=("admin", "secret"))
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["filepath"], "tests/test_chat.py")
        self.assertEqual(data["start_line"], 1)
        self.assertEqual(data["end_line"], 3)
        self.assertIn("import", data["content"])

    @patch("knowledge_mcp.daemon.routes.chat.load_config")
    def test_read_file_from_configured_repo(self, mock_load_config):
        """Проверяет чтение файла из настроенного репозитория по repo_id и без него."""
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_file = Path(tmp_dir) / "docs" / "guide.md"
            repo_file.parent.mkdir(parents=True, exist_ok=True)
            repo_file.write_text("# Guide\nLine 2\nLine 3\nLine 4\nLine 5", encoding="utf-8")
            
            mock_load_config.return_value = {
                "server": {"dashboard": {"chat": {"enabled": True}}},
                "repositories": [
                    {
                        "id": "my-external-repo",
                        "local_path": tmp_dir
                    }
                ]
            }
            os.environ["DASHBOARD_PASSWORD"] = "secret"
            
            # Чтение с явным указанием repo_id
            response = self.client.get(
                "/api/chat/file?filepath=docs/guide.md&repo_id=my-external-repo&start_line=2&end_line=3",
                auth=("admin", "secret")
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["filepath"], "docs/guide.md")
            self.assertEqual(data["repo_id"], "my-external-repo")
            self.assertEqual(data["start_line"], 2)
            self.assertEqual(data["end_line"], 3)
            self.assertEqual(data["total_lines"], 5)
            self.assertIn("Line 2\nLine 3", data["content"])
            self.assertFalse(data["is_full_file"])
            
            # Чтение всего файла (full_file=true)
            response_full = self.client.get(
                "/api/chat/file?filepath=docs/guide.md&repo_id=my-external-repo&start_line=2&end_line=3&full_file=true",
                auth=("admin", "secret")
            )
            self.assertEqual(response_full.status_code, 200)
            data_full = response_full.json()
            self.assertTrue(data_full["is_full_file"])
            self.assertEqual(data_full["start_line"], 2)
            self.assertEqual(data_full["end_line"], 3)
            self.assertEqual(data_full["total_lines"], 5)
            self.assertIn("# Guide\nLine 2\nLine 3\nLine 4\nLine 5", data_full["content"])
            
            # Чтение без указания repo_id (автоматический поиск по репозиториям)
            response_auto = self.client.get(
                "/api/chat/file?filepath=docs/guide.md",
                auth=("admin", "secret")
            )
            self.assertEqual(response_auto.status_code, 200)
            self.assertEqual(response_auto.json()["repo_id"], "my-external-repo")

    @patch("knowledge_mcp.daemon.routes.chat.load_config")
    def test_read_file_from_project_workspace(self, mock_load_config):
        """Проверяет чтение файла из многорепозиторного проекта (projects) через worktrees и docs_paths."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            ws_root = Path(tmp_dir)
            backend_file = ws_root / "worktrees" / "master" / "backend" / "Services" / "UserService.cs"
            backend_file.parent.mkdir(parents=True, exist_ok=True)
            backend_file.write_text("namespace App;\npublic class UserService {}\n", encoding="utf-8")

            doc_file = ws_root / ".kb" / "guide.md"
            doc_file.parent.mkdir(parents=True, exist_ok=True)
            doc_file.write_text("# Project Guide\nInfo\n", encoding="utf-8")

            mock_load_config.return_value = {
                "server": {"dashboard": {"chat": {"enabled": True}}},
                "repositories": [],
                "projects": [
                    {
                        "id": "my-project",
                        "workspace_root": str(ws_root),
                        "base_path": "worktrees/master",
                        "repositories": [
                            {"id": "backend", "path": "backend", "language": "csharp"}
                        ],
                        "docs_paths": [".kb"]
                    }
                ]
            }
            os.environ["DASHBOARD_PASSWORD"] = "secret"

            # 1. Чтение файла бэкенда по repo_id и относительному пути
            res = self.client.get(
                "/api/chat/file?filepath=Services/UserService.cs&repo_id=backend&full_file=true",
                auth=("admin", "secret")
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["repo_id"], "backend")
            self.assertIn("class UserService", res.json()["content"])

            # 2. Чтение файла бэкенда с префиксом repo_id в filepath
            res_prefix = self.client.get(
                "/api/chat/file?filepath=backend/Services/UserService.cs&repo_id=backend&full_file=true",
                auth=("admin", "secret")
            )
            self.assertEqual(res_prefix.status_code, 200)
            self.assertEqual(res_prefix.json()["repo_id"], "backend")

            # 3. Чтение документации проекта
            res_doc = self.client.get(
                "/api/chat/file?filepath=guide.md&repo_id=docs&full_file=true",
                auth=("admin", "secret")
            )
            self.assertEqual(res_doc.status_code, 200)
            self.assertIn("# Project Guide", res_doc.json()["content"])

    @patch("httpx.AsyncClient.post")
    @patch("knowledge_mcp.daemon.routes.chat.load_config")
    def test_sources_contain_repo_id_and_line_metadata(self, mock_load_config, mock_post):
        """Проверяет наличие repo_id, line_start и line_end в sources."""
        mock_load_config.return_value = {
            "server": {"dashboard": {"chat": {"enabled": True}}}
        }
        os.environ["LLM_API_KEY"] = "sk-test"
        os.environ["DASHBOARD_PASSWORD"] = "secret"
        
        self.mock_db.search_chunks_hybrid.return_value = [
            {
                "repo_id": "test-repo",
                "path": "src/test.py",
                "line_start": 10,
                "line_end": 20,
                "content": "def foo(): pass"
            }
        ]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test answer"}}]
        }
        mock_post.return_value = mock_response
        
        req_body = {"messages": [{"role": "user", "content": "How foo works"}]}
        response = self.client.post("/api/chat", json=req_body, auth=("admin", "secret"))
        self.assertEqual(response.status_code, 200)
        sources = response.json()["sources"]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["repo_id"], "test-repo")
        self.assertEqual(sources[0]["filepath"], "src/test.py")
        self.assertEqual(sources[0]["lines"], "10-20")
        self.assertEqual(sources[0]["line_start"], 10)
        self.assertEqual(sources[0]["line_end"], 20)
