"""Утилиты разрешения путей воркспейса, сопоставления веток и JIT-синхронизации."""

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from knowledge_mcp.core.config import get_project_config, get_repository_config, resolve_workspace_targets
from knowledge_mcp.core.jit_sync import JITSyncEngine

logger = logging.getLogger("knowledge_mcp.core.workspace_context")

jit_engine = JITSyncEngine()
_REPO_PATH_CACHE: Dict[Tuple[str, Optional[str], Optional[str]], Optional[Path]] = {}


def clear_repo_path_cache() -> None:
    """Очищает внутренний кэш путей к репозиториям."""
    _REPO_PATH_CACHE.clear()


def is_branch_match(requested_branch: Optional[str], target_branch: Optional[str], task_name: Optional[str]) -> bool:
    """Точное и безопасное сопоставление веток и ворктри задач."""
    if not requested_branch:
        return True

    req = requested_branch.strip().lower()
    tgt = (target_branch or "").strip().lower()
    task = (task_name or "").strip().lower()

    if req == tgt:
        return True

    req_clean = req.removeprefix("origin/").removeprefix("refs/heads/").removeprefix("dvcn/")
    tgt_clean = tgt.removeprefix("origin/").removeprefix("refs/heads/").removeprefix("dvcn/")
    if req_clean == tgt_clean:
        return True

    def extract_task_number(text: str) -> Optional[str]:
        m = re.search(r'(?:orbit|task|issue|feature)[/-]?(\d+)', text, re.IGNORECASE)
        if m:
            return m.group(1)
        m2 = re.search(r'\b(\d{3,6})\b', text)
        if m2:
            return m2.group(1)
        return None

    req_num = extract_task_number(req)
    tgt_num = extract_task_number(tgt) or (extract_task_number(task) if task else None)

    if req_num and tgt_num:
        return req_num == tgt_num

    if task and (task == req or task == req_clean):
        return True

    if req in ("master", "main") and tgt in ("master", "main"):
        return req == tgt

    return False


def resolve_project_id(
    project_id: Optional[str] = None,
    branch: Optional[str] = None,
    repo_ids: Optional[List[str]] = None,
    repo_id: Optional[str] = None
) -> str:
    """Автоматически сопоставляет project_id по ветке или repo_ids, если project_id не указан явно."""
    if project_id and project_id != "default":
        return project_id

    all_repos = list(repo_ids or [])
    if repo_id and repo_id not in all_repos:
        all_repos.append(repo_id)

    try:
        targets = resolve_workspace_targets(None)
        if targets:
            if all_repos:
                for t in targets:
                    if t.get("repo_id") in all_repos:
                        p = t.get("project_id")
                        if p:
                            return p

            if branch and branch not in ("master", "main", "develop"):
                for t in targets:
                    if is_branch_match(branch, t.get("branch"), t.get("task_name")):
                        p = t.get("project_id")
                        if p:
                            return p
    except Exception as e:
        logger.debug(f"Failed to auto-resolve project_id: {e}")

    return project_id or "default"


def find_repo_path(project_id: str, repo_id: Optional[str], branch: Optional[str] = None) -> Optional[Path]:
    """Находит локальный путь к репозиторию/ворктри на диске с мгновенным кэшированием."""
    cache_key = (project_id, repo_id, branch)
    if cache_key in _REPO_PATH_CACHE:
        cached = _REPO_PATH_CACHE[cache_key]
        if cached and cached.exists():
            return cached

    p_id = resolve_project_id(project_id, branch, [repo_id] if repo_id else None)
    targets = resolve_workspace_targets(p_id)
    if targets and repo_id:
        for t in targets:
            if t.get("repo_id") == repo_id:
                if is_branch_match(branch, t.get("branch"), t.get("task_name")):
                    abs_p = t.get("absolute_path")
                    if abs_p and Path(abs_p).exists():
                        _REPO_PATH_CACHE[cache_key] = Path(abs_p)
                        return Path(abs_p)
        for t in targets:
            if t.get("repo_id") == repo_id and t.get("is_base"):
                abs_p = t.get("absolute_path")
                if abs_p and Path(abs_p).exists():
                    _REPO_PATH_CACHE[cache_key] = Path(abs_p)
                    return Path(abs_p)

    if repo_id:
        repo_cfg = get_repository_config(repo_id)
        if repo_cfg and repo_cfg.get("local_path"):
            p = Path(repo_cfg["local_path"]).resolve()
            if p.exists():
                return p

    proj_cfg = get_project_config(project_id)
    if proj_cfg and proj_cfg.get("workspace_root"):
        from knowledge_mcp.core.workspace_scanner import resolve_workspace_root
        p = resolve_workspace_root(proj_cfg["workspace_root"])
        if p and p.exists():
            _REPO_PATH_CACHE[cache_key] = p
            return p

    _REPO_PATH_CACHE[cache_key] = None
    return None


def run_jit_check(
    repo_ids: Optional[List[str]],
    project_id: str,
    branch: str,
    db_instance: Any
) -> int:
    """Выполняет быструю JIT-проверку для указанных репозиториев перед выполнением поисковых запросов."""
    if not db_instance or getattr(db_instance, "read_only", False):
        return 0
    updated_total = 0
    p_id = resolve_project_id(project_id, branch, repo_ids)
    try:
        if repo_ids:
            for r_id in repo_ids:
                p = find_repo_path(p_id, r_id, branch)
                if p and p.exists() and p.is_dir():
                    updated_total += jit_engine.ensure_fresh(
                        p, project_id=p_id, repo_id=r_id, branch=branch, db=db_instance
                    )
        else:
            targets = resolve_workspace_targets(p_id)
            relevant_targets = [
                t for t in targets
                if not t.get("is_doc") and is_branch_match(branch, t.get("branch"), t.get("task_name"))
            ]
            for t in relevant_targets[:5]:
                p = t.get("absolute_path")
                r_id = t.get("repo_id")
                b_name = t.get("branch") or branch
                if p and Path(p).is_dir() and r_id:
                    updated_total += jit_engine.ensure_fresh(
                        p, project_id=p_id, repo_id=r_id, branch=b_name, db=db_instance
                    )
    except Exception as e:
        logger.debug(f"JIT sync check encountered non-fatal error: {e}")
    return updated_total
