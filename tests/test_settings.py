import os
import tempfile
import base64
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.core.i18n import t, get_locale_from_header
from knowledge_mcp.core.config import (
    load_config,
    save_config,
    add_or_update_repository,
    delete_repository_config,
    mask_secret
)
import knowledge_mcp.main as main_mod

@pytest.fixture
def test_db_and_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_knowledge.db"
        cfg_path = Path(tmpdir) / "config.yaml"
        
        # Начальный конфиг
        initial_cfg = {
            "server": {
                "port": 8000,
                "host": "127.0.0.1",
                "dashboard": {
                    "enabled": True,
                    "chat": {
                        "enabled": True,
                        "api_base": "http://localhost:20128/v1",
                        "model": "kr/claude-sonnet-4.5"
                    }
                }
            },
            "embeddings": {
                "enabled": False,
                "model_name": "intfloat/multilingual-e5-small"
            },
            "repositories": [
                {
                    "id": "repo-1",
                    "url": "https://github.com/test/repo-1.git",
                    "branch": "master",
                    "auth": {"type": "none"}
                }
            ]
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(initial_cfg, f)
            
        old_cfg_path = os.environ.get("CONFIG_PATH")
        os.environ["CONFIG_PATH"] = str(cfg_path)
        os.environ["DASHBOARD_PASSWORD"] = "testpassword123"

        db = KnowledgeDB(db_path)
        old_db = getattr(main_mod, "db", None)
        main_mod.db = db

        from knowledge_mcp.daemon.routes.settings import router as settings_router
        if not any(getattr(route, "path", "").startswith("/api/settings") for route in main_mod.app.routes):
            main_mod.app.include_router(settings_router)

        client = TestClient(main_mod.app)
        auth_header = "Basic " + base64.b64encode(b"admin:testpassword123").decode("utf-8")

        try:
            yield db, cfg_path, client, auth_header
        finally:
            if old_cfg_path:
                os.environ["CONFIG_PATH"] = old_cfg_path
            else:
                os.environ.pop("CONFIG_PATH", None)
            main_mod.db = old_db
            db.close()


def test_db_settings_crud(test_db_and_config):
    db, _, _, _ = test_db_and_config
    assert db.get_setting("default_locale") == "ru"
    assert db.get_setting("default_theme") == "dark"
    
    db.set_setting("default_locale", "en")
    assert db.get_setting("default_locale") == "en"

    db.set_setting("custom_param", "123")
    assert db.get_setting("custom_param") == "123"
    
    all_s = db.get_all_settings()
    assert all_s.get("default_locale") == "en"
    assert all_s.get("custom_param") == "123"

def test_i18n_translation_and_resolution():
    assert get_locale_from_header("en-US,en;q=0.9") == "en"
    assert get_locale_from_header("ru-RU,ru;q=0.9") == "ru"
    assert get_locale_from_header(None, default_locale="ru") == "ru"
    assert get_locale_from_header("fr-FR", default_locale="ru") == "ru"

    assert "Репозиторий" in t("repo.not_found", locale="ru", repo_id="xyz")
    assert "Repository" in t("repo.not_found", locale="en", repo_id="xyz")

def test_config_operations(test_db_and_config):
    _, cfg_path, _, _ = test_db_and_config
    assert mask_secret("ghp_1234567890") == "ghp_****90"
    assert mask_secret("short") == "******"

    new_repo = {
        "id": "repo-new",
        "url": "https://github.com/test/repo-new.git",
        "branch": "main",
        "auth": {"type": "pat", "token": "secret123"}
    }
    add_or_update_repository(new_repo)
    cfg = load_config()
    assert any(r["id"] == "repo-new" for r in cfg["repositories"])

    deleted = delete_repository_config("repo-new")
    assert deleted is True
    cfg = load_config()
    assert not any(r["id"] == "repo-new" for r in cfg["repositories"])

def test_api_get_settings(test_db_and_config):
    _, _, client, auth = test_db_and_config
    res = client.get("/api/settings", headers={"Authorization": auth})
    assert res.status_code == 200
    data = res.json()
    assert data["ui"]["default_locale"] == "ru"
    assert data["llm"]["chat_enabled"] is True
    assert len(data["repositories"]) >= 1

def test_api_update_settings(test_db_and_config):
    db, _, client, auth = test_db_and_config
    payload = {
        "default_locale": "en",
        "default_theme": "light",
        "chat_enabled": False,
        "llm_model": "test-model-v1"
    }
    res = client.put("/api/settings", json=payload, headers={"Authorization": auth, "Accept-Language": "en"})
    assert res.status_code == 200
    assert "successfully" in res.json()["message"]
    assert db.get_setting("default_locale") == "en"
    assert db.get_setting("default_theme") == "light"

    # Проверяем через GET
    get_res = client.get("/api/settings", headers={"Authorization": auth})
    assert get_res.json()["ui"]["default_locale"] == "en"
    assert get_res.json()["llm"]["chat_enabled"] is False
    assert get_res.json()["llm"]["model"] == "test-model-v1"

def test_api_repository_crud(test_db_and_config):
    _, _, client, auth = test_db_and_config
    # 1. Создание
    repo_data = {
        "id": "custom-repo",
        "url": "https://github.com/org/custom-repo.git",
        "branch": "develop",
        "auth": {"type": "none"}
    }
    create_res = client.post("/api/settings/repositories", json=repo_data, headers={"Authorization": auth, "Accept-Language": "en"})
    assert create_res.status_code == 200
    assert create_res.json()["status"] == "success"

    # Повторное создание с тем же ID -> 409
    dup_res = client.post("/api/settings/repositories", json=repo_data, headers={"Authorization": auth, "Accept-Language": "en"})
    assert dup_res.status_code == 409

    # 2. Обновление
    repo_data["branch"] = "production"
    update_res = client.put("/api/settings/repositories/custom-repo", json=repo_data, headers={"Authorization": auth})
    assert update_res.status_code == 200

    # 3. Удаление
    del_res = client.delete("/api/settings/repositories/custom-repo", headers={"Authorization": auth, "Accept-Language": "en"})
    assert del_res.status_code == 200
    assert "deleted successfully" in del_res.json()["message"]

def test_api_test_git_endpoint(test_db_and_config):
    _, _, client, auth = test_db_and_config
    from knowledge_mcp.core.git_manager import GitManager
    with patch.object(GitManager, "test_connection", return_value=(True, "OK")):
        res = client.post("/api/settings/test-git", json={"url": "https://github.com/test/repo.git", "branch": "main"}, headers={"Authorization": auth, "Accept-Language": "en"})
        assert res.status_code == 200
        assert res.json()["success"] is True

    with patch.object(GitManager, "test_connection", return_value=(False, "Auth failed")):
        res = client.post("/api/settings/test-git", json={"url": "https://github.com/test/repo.git", "branch": "main"}, headers={"Authorization": auth, "Accept-Language": "en"})
        assert res.status_code == 200
        assert res.json()["success"] is False
        assert "Git connection error" in res.json()["message"]


def test_spa_dashboard_serving(test_db_and_config):
    _, _, client, _ = test_db_and_config
    
    # 1. Корневой маршрут возвращает index.html с новым заголовком
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "Knowledge Base Dashboard" in res_root.text

    # 2. SPA маршруты отдают index.html
    for spa_path in ["/repositories", "/queue", "/chat"]:
        res = client.get(spa_path)
        assert res.status_code == 200
        assert "<app-root>" in res.text

    # 3. Несуществующий API эндпоинт возвращает 404, а не HTML
    res_404 = client.get("/api/non-existent-endpoint")
    assert res_404.status_code == 404


