import hmac
import hashlib
import json
import time
import urllib.request
import urllib.error
import base64

BASE_URL = "http://localhost:8000"
PASSWORD = "admin"
WEBHOOK_SECRET = "dev_secret_12345"

def test_dashboard_page():
    print("1. Тест Web Dashboard HTML...")
    req = urllib.request.Request(f"{BASE_URL}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        html = resp.read().decode('utf-8')
        assert "Knowledge Base Dashboard" in html
        assert "Репозитории" in html
        print("   -> OK! Dashboard HTML отдает 200 OK.")

def test_unauthorized_api():
    print("2. Тест защиты API Basic Auth (без кредов)...")
    try:
        urllib.request.urlopen(f"{BASE_URL}/api/status")
        assert False, "Should have returned 401"
    except urllib.error.HTTPError as e:
        assert e.code == 401
        print("   -> OK! Возвращен 401 Unauthorized.")

def test_authorized_api_status():
    print("3. Тест авторизованного /api/status...")
    req = urllib.request.Request(f"{BASE_URL}/api/status")
    auth_str = base64.b64encode(f"admin:{PASSWORD}".encode()).decode()
    req.add_header("Authorization", f"Basic {auth_str}")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        print(f"   -> Статус системы: {data.get('status')}")
        print(f"   -> Репозиториев в конфиге: {len(data.get('repositories', []))}")
        print(f"   -> Чат включен: {data.get('chat_enabled')}")
        assert data.get('chat_enabled') is True
        assert len(data.get('repositories', [])) >= 2
        print("   -> OK! /api/status успешно возвращает метрики.")

def test_get_token():
    print("4. Тест генерации временного токена (/api/auth/token)...")
    req = urllib.request.Request(f"{BASE_URL}/api/auth/token", method="POST")
    auth_str = base64.b64encode(f"admin:{PASSWORD}".encode()).decode()
    req.add_header("Authorization", f"Basic {auth_str}")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        token = data.get("token")
        assert token and len(token.split('.')) == 3
        print(f"   -> Токен: {token[:20]}...")
        print("   -> OK! Временный крипто-токен сгенерирован.")
        return token

def test_send_webhook():
    print("5. Тест отправки подписанного GitHub-вебхука с измененными файлами...")
    payload = {
        "repository": {
            "clone_url": "https://github.com/PaulTheSecond/knowledgebase-mcp.git"
        },
        "commits": [
            {
                "id": "test_commit_abc123",
                "message": "fix: update docs and readme",
                "added": ["docs/new_note.md"],
                "modified": ["README.md", "knowledge_mcp/main.py"],
                "removed": []
            }
        ]
    }
    raw_body = json.dumps(payload).encode("utf-8")
    sig = "sha256=" + hmac.new(WEBHOOK_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()

    req = urllib.request.Request(f"{BASE_URL}/webhook/git", data=raw_body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-Hub-Signature-256", sig)

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        print(f"   -> Ответ вебхука: {data}")
        assert data.get("status") == "queued"
        assert data.get("repo_id") == "knowledgebase-mcp"
        assert data.get("changed_files_count") == 3
        print("   -> OK! Вебхук принят, задача поставлена в очередь (3 файла дельты).")
        return data.get("job_id")

def test_check_queue(job_id):
    print("6. Тест просмотра очереди (/api/queue) и статуса задачи...")
    req = urllib.request.Request(f"{BASE_URL}/api/queue")
    auth_str = base64.b64encode(f"admin:{PASSWORD}".encode()).decode()
    req.add_header("Authorization", f"Basic {auth_str}")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        jobs = data.get("jobs", [])
        print(f"   -> Задач в очереди: {len(jobs)}")
        matching_job = next((j for j in jobs if j["id"] == job_id), None)
        assert matching_job is not None, f"Job {job_id} not found in queue"
        print(f"   -> Задача #{job_id}: статус={matching_job['status']}, попыток={matching_job['attempts']}, файлы={matching_job.get('changed_files')}")
        print("   -> OK! Задача присутствует в очереди с извлеченными файлами дельты.")

def test_websocket(token):
    print("7. Тест подключения к WebSocket (/ws/status?token=...)...")
    import asyncio
    import websockets

    async def _ws_connect():
        ws_url = f"ws://localhost:8000/ws/status?token={token}"
        async with websockets.connect(ws_url) as ws:
            # Ждем начальное приветствие или пинг
            print("   -> WebSocket соединение успешно установлено!")
            return True

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    res = loop.run_until_complete(_ws_connect())
    assert res is True
    print("   -> OK! WebSocket успешно авторизован по временному токену.")

if __name__ == "__main__":
    print("==================================================")
    print("ЗАПУСК E2E ТЕСТИРОВАНИЯ KNOWLEDGE MCP В DOCKER")
    print("==================================================")
    test_dashboard_page()
    test_unauthorized_api()
    test_authorized_api_status()
    token = test_get_token()
    test_websocket(token)
    job_id = test_send_webhook()
    time.sleep(1)
    test_check_queue(job_id)
    print("==================================================")
    print("ВСЕ 7 ТЕСТОВ УСПЕШНО ПРОЙДЕНЫ В РЕАЛЬНОМ КОНТЕЙНЕРЕ!")
    print("==================================================")
