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


@router.get("/metrics")
async def get_telemetry_metrics(username: str = Depends(verify_basic_auth)):
    """Возвращает сводные метрики телеметрии: CAS-дедупликацию, экономию токенов LLM,
    статус File Watcher и обзор мульти-репо проектов.
    """
    db = get_db()
    if not db:
        raise HTTPException(status_code=500, detail="Database component is not initialized")

    cas_stats = db.get_cas_stats()
    token_savings = db.get_token_savings_summary(days=7)
    watcher_status = get_watcher_status()

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

    return {
        "cas": cas_stats,
        "token_savings": token_savings,
        "watcher": watcher_status,
        "projects": projects_data
    }
