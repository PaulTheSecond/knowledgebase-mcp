import os
import tempfile
import base64
import yaml
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from knowledge_mcp.db.models import KnowledgeDB
import knowledge_mcp.main as main_mod


@pytest.fixture
def telemetry_test_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "telemetry_test.db"
        cfg_path = Path(tmpdir) / "config.yaml"

        initial_cfg = {
            "server": {
                "port": 8000,
                "host": "127.0.0.1",
                "dashboard": {"enabled": True}
            },
            "repositories": [
                {
                    "id": "repo-telemetry-test",
                    "url": "https://github.com/test/telemetry.git",
                    "branch": "master"
                }
            ],
            "projects": [
                {
                    "id": "orbita-test",
                    "name": "Orbita Workspace",
                    "type": "worktree_workspace",
                    "workspace_root": str(Path(tmpdir) / "ws"),
                    "base_path": "worktrees/master"
                }
            ]
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(initial_cfg, f)

        old_cfg_path = os.environ.get("CONFIG_PATH")
        os.environ["CONFIG_PATH"] = str(cfg_path)
        os.environ["DASHBOARD_PASSWORD"] = "secret_pass_123"

        db = KnowledgeDB(db_path)
        old_db = getattr(main_mod, "db", None)
        old_watcher = getattr(main_mod, "watcher_instance", None)
        main_mod.db = db

        client = TestClient(main_mod.app)
        auth_header = "Basic " + base64.b64encode(b"admin:secret_pass_123").decode("utf-8")

        try:
            yield db, cfg_path, client, auth_header
        finally:
            if old_cfg_path:
                os.environ["CONFIG_PATH"] = old_cfg_path
            else:
                os.environ.pop("CONFIG_PATH", None)
            main_mod.db = old_db
            main_mod.watcher_instance = old_watcher
            db.close()


def test_telemetry_metrics_auth_required(telemetry_test_env):
    """Проверка требования аутентификации для эндпоинта /api/telemetry/metrics."""
    _, _, client, _ = telemetry_test_env

    # Без заголовка Authorization -> 401
    resp = client.get("/api/telemetry/metrics")
    assert resp.status_code == 401

    # С неверным паролем -> 401
    bad_auth = "Basic " + base64.b64encode(b"admin:wrong").decode("utf-8")
    resp_bad = client.get("/api/telemetry/metrics", headers={"Authorization": bad_auth})
    assert resp_bad.status_code == 401


def test_telemetry_metrics_structure(telemetry_test_env):
    """Проверка структуры и типов JSON ответа /api/telemetry/metrics."""
    db, _, client, auth_header = telemetry_test_env

    resp = client.get("/api/telemetry/metrics", headers={"Authorization": auth_header})
    assert resp.status_code == 200
    data = resp.json()

    # Проверка корневых ключей
    assert "cas" in data
    assert "token_savings" in data
    assert "watcher" in data
    assert "projects" in data

    # Проверка структуры cas
    cas = data["cas"]
    for key in ("logical_files_count", "unique_contents_count", "virtual_size_bytes",
                "real_size_bytes", "saved_size_bytes", "saved_percent"):
        assert key in cas

    # Проверка структуры token_savings
    tokens = data["token_savings"]
    for key in ("total_saved_tokens", "avg_saved_percent", "total_calls", "by_tool"):
        assert key in tokens

    # Проверка структуры watcher
    watcher = data["watcher"]
    assert "active" in watcher
    assert "watched_paths_count" in watcher
    assert isinstance(watcher["active"], bool)
    assert isinstance(watcher["watched_paths_count"], int)

    # Проверка структуры projects
    projects = data["projects"]
    assert isinstance(projects, list)
    assert len(projects) == 1
    assert projects[0]["id"] == "orbita-test"
    assert projects[0]["name"] == "Orbita Workspace"
    assert "targets_count" in projects[0]


def test_telemetry_metrics_live_data(telemetry_test_env):
    """Проверка обновления данных при логировании токенов и файлов."""
    db, _, client, auth_header = telemetry_test_env

    # 1. Логируем экономию токенов
    db.log_token_savings(
        tool_name="knowledge_get_project_rules",
        baseline_tokens=25000,
        returned_tokens=1500,
        project_id="orbita-test",
        repo_id="ImpactOS.Web",
        branch="orbit-1035"
    )
    db.log_token_savings(
        tool_name="knowledge_search",
        baseline_tokens=8000,
        returned_tokens=1200,
        project_id="orbita-test",
        repo_id="ImpactOS.Web",
        branch="orbit-1035"
    )

    # 2. Добавляем контент в CAS
    content = "function hello() { return 'world'; }"
    sha = "c3ab8ff13720e8ad9047dd39466b3c8974e592c2fa383d4a3960714caef0c4f2"
    cursor = db.conn.cursor()
    cursor.execute("""
        INSERT INTO file_contents (hash, content, size, language, updated_at)
        VALUES (?, ?, ?, 'typescript', strftime('%s', 'now'))
    """, (sha, content, len(content)))
    db.conn.commit()

    # Два логических файла ссылаются на один и тот же hash контента
    db.upsert_file("ImpactOS.Web", "src/hello.ts", 100.0, sha, project_id="orbita-test", branch="master", content_hash=sha)
    db.upsert_file("ImpactOS.Web", "src/hello.ts", 100.0, sha, project_id="orbita-test", branch="orbit-1035", content_hash=sha)

    # Делаем запрос к API
    resp = client.get("/api/telemetry/metrics", headers={"Authorization": auth_header})
    assert resp.status_code == 200
    data = resp.json()

    # Проверяем токен-сейвинги
    tokens = data["token_savings"]
    assert tokens["total_calls"] == 2
    assert tokens["total_saved_tokens"] == (25000 - 1500) + (8000 - 1200)
    assert "knowledge_get_project_rules" in tokens["by_tool"]
    assert "knowledge_search" in tokens["by_tool"]

    # Проверяем CAS
    cas = data["cas"]
    assert cas["logical_files_count"] >= 2
    assert cas["unique_contents_count"] >= 1
    assert cas["saved_size_bytes"] > 0
    assert cas["saved_percent"] > 0


def test_telemetry_watcher_status(telemetry_test_env):
    """Проверка отражения активного WorkspaceFileWatcher в телеметрии."""
    _, _, client, auth_header = telemetry_test_env

    mock_watcher = MagicMock()
    mock_watcher.is_running.return_value = True
    mock_watcher._paths = [Path("c:/worktrees/master"), Path("c:/worktrees/orbit-1035")]
    main_mod.watcher_instance = mock_watcher

    resp = client.get("/api/telemetry/metrics", headers={"Authorization": auth_header})
    assert resp.status_code == 200
    data = resp.json()

    assert data["watcher"]["active"] is True
    assert data["watcher"]["watched_paths_count"] == 2


def test_dashboard_spa_route(telemetry_test_env):
    """Проверка доступности SPA маршрута /dashboard и отдачи index.html."""
    _, _, client, _ = telemetry_test_env

    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower() or "<!doctype html>" in resp.text.lower()
