"""Инструменты администрирования MCP (синхронизация и удаление репозиториев через Daemon API)."""

import asyncio
import base64
import logging
import os
import urllib.request
from typing import Dict, List

import mcp.types as types

logger = logging.getLogger("knowledge_mcp.interfaces.mcp.admin")


def get_auth_headers() -> Dict[str, str]:
    """Возвращает заголовок Authorization для Basic Auth, если установлен DASHBOARD_PASSWORD."""
    password = os.environ.get("DASHBOARD_PASSWORD")
    if not password:
        return {}
    creds = f"admin:{password}"
    b64_creds = base64.b64encode(creds.encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {b64_creds}"}


def get_daemon_base_url() -> str:
    """Возвращает базовый URL Daemon API из переменной окружения DAEMON_URL."""
    return os.environ.get("DAEMON_URL", "http://127.0.0.1:8765").rstrip("/")


def _err(msg: str) -> List[types.TextContent]:
    """Формирует сообщение об ошибке в формате TextContent."""
    return [types.TextContent(type="text", text=msg)]


async def handle_delete_repo(arguments: dict) -> List[types.TextContent]:
    """knowledge_delete_repo: делегирует удаление репозитория Daemon-процессу через HTTP API."""
    repo_id = arguments.get("repo_id")
    if not repo_id:
        return _err("Ошибка: обязательный аргумент 'repo_id' не передан.")

    try:
        base_url = get_daemon_base_url()
        url = f"{base_url}/api/repo/{repo_id}"
        headers = get_auth_headers()
        req = urllib.request.Request(url, method="DELETE", headers=headers)

        def _trigger_delete():
            with urllib.request.urlopen(req) as response:
                return response.read()

        logger.info(f"Делегирование удаления '{repo_id}' в Daemon API через DELETE {url}")
        await asyncio.to_thread(_trigger_delete)

        return [types.TextContent(type="text", text=f"Запрос на удаление репозитория '{repo_id}' успешно отправлен в Daemon API.")]
    except Exception as e:
        logger.error(f"Не удалось удалить репозиторий через Daemon API: {e}")
        return _err(f"Не удалось делегировать удаление репозитория в Daemon: {e}")


async def handle_sync_repo(arguments: dict) -> List[types.TextContent]:
    """knowledge_sync_repo: делегирует синхронизацию репозитория Daemon-процессу через HTTP API."""
    repo_id = arguments.get("repo_id")
    if not repo_id:
        return _err("Ошибка: обязательный аргумент 'repo_id' не передан.")

    try:
        base_url = get_daemon_base_url()
        url = f"{base_url}/api/sync/{repo_id}"
        headers = get_auth_headers()
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, method="POST", data=b"", headers=headers)

        def _trigger_sync():
            with urllib.request.urlopen(req) as response:
                return response.read()

        logger.info(f"Делегирование синхронизации '{repo_id}' в Daemon API через POST {url}")
        await asyncio.to_thread(_trigger_sync)

        return [types.TextContent(type="text", text=f"Репозиторий '{repo_id}' успешно поставлен в очередь индексации Daemon API.")]
    except Exception as e:
        logger.error(f"Не удалось запустить синхронизацию через Daemon API: {e}")
        return _err(f"Не удалось делегировать синхронизацию репозитория в Daemon: {e}")
