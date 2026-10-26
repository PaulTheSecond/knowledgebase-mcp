import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends

from knowledge_mcp.daemon.auth import verify_basic_auth
from knowledge_mcp.core.config import get_projects, resolve_workspace_targets

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


def get_db():
    """Получает актуальный инстанс KnowledgeDB."""
    import sys
    import knowledge_mcp.main as main_mod
    return getattr(main_mod, "db", None) or getattr(sys.modules.get("__main__"), "db", None)


def get_watcher_status() -> Dict[str, Any]:
    """Возвращает текущий статус фонового WorkspaceFileWatcher."""
    import sys
    import knowledge_mcp.main as main_mod
    watcher = getattr(main_mod, "watcher_instance", None) or getattr(sys.modules.get("__main__"), "watcher_instance", None)
    if watcher and hasattr(watcher, "is_running") and watcher.is_running():
        paths = getattr(watcher, "_paths", [])
        return {
            "active": True,
            "watched_paths_count": len(paths)
        }
    return {
        "active": False,
        "watched_paths_count": 0
    }


import asyncio
import time

_projects_data_cache: List[Dict[str, Any]] = []
_projects_data_cache_time: float = 0.0

def _get_projects_data_cached() -> List[Dict[str, Any]]:
    global _projects_data_cache, _projects_data_cache_time
    now = time.time()
    if _projects_data_cache and (now - _projects_data_cache_time < 60.0):
        return list(_projects_data_cache)

    projects_data: List[Dict[str, Any]] = []
    try:
        configured_projects = get_projects()
        for p in configured_projects:
            p_id = p.get("id")
            targets = resolve_workspace_targets(p_id)
            projects_data.append({
                "id": p_id,
                "name": p.get("name", p_id),
                "targets_count": len(targets)
            })
    except Exception as ex:
        logger.warning(f"Error resolving projects telemetry: {ex}")

    _projects_data_cache = projects_data
    _projects_data_cache_time = now
    return list(projects_data)

def _get_telemetry_sync(db) -> Dict[str, Any]:
    cas_stats = db.get_cas_stats()
    token_savings = db.get_token_savings_summary(days=7)
    watcher_status = get_watcher_status()
    projects_data = _get_projects_data_cached()

    return {
        "cas": cas_stats,
        "token_savings": token_savings,
        "watcher": watcher_status,
        "projects": projects_data
    }

@router.get("/metrics")
async def get_telemetry_metrics(username: str = Depends(verify_basic_auth)):
    """Возвращает сводные метрики телеметрии: CAS-дедупликацию, экономию токенов LLM,
    статус File Watcher и обзор мульти-репо проектов (non-blocking).
    """
    db = get_db()
    if not db:
        raise HTTPException(status_code=500, detail="Database component is not initialized")

    return await asyncio.to_thread(_get_telemetry_sync, db)
