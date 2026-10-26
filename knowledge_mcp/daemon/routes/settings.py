import os
import re
import logging
import httpx
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Header, Query
from pydantic import BaseModel, Field

from knowledge_mcp.daemon.auth import verify_basic_auth
from knowledge_mcp.core.config import (
    load_config,
    save_config,
    add_or_update_repository,
    delete_repository_config,
    get_repository_config,
    mask_secret
)
from knowledge_mcp.core.i18n import t, get_locale_from_header
from knowledge_mcp.core.git_manager import GitManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])

def get_db():
    """Получает актуальный инстанс KnowledgeDB."""
    import sys
    import knowledge_mcp.main as main_mod
    return getattr(main_mod, "db", None) or getattr(sys.modules.get("__main__"), "db", None)

def resolve_locale(accept_language: Optional[str] = Header(None, alias="Accept-Language")) -> str:
    """Определяет язык текущего запроса с учетом БД и заголовка Accept-Language."""
    db = get_db()
    default_locale = db.get_setting("default_locale", "ru") if db else "ru"
    return get_locale_from_header(accept_language, default_locale)

# --- Pydantic Схемы ---

class GeneralSettingsPayload(BaseModel):
    default_locale: Optional[str] = None
    default_theme: Optional[str] = None
    chat_enabled: Optional[bool] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    embeddings_enabled: Optional[bool] = None
    embeddings_model: Optional[str] = None
    webhook_secret: Optional[str] = None
    github_pat: Optional[str] = None

class RepositoryAuthPayload(BaseModel):
    type: str = "none" # none, pat, ssh
    token_env: Optional[str] = None
    token: Optional[str] = None
    private_key_path: Optional[str] = None

class RepositoryTriggersPayload(BaseModel):
    webhooks: Optional[Dict[str, Any]] = None
    cron: Optional[str] = None

class RepositoryPayload(BaseModel):
    id: str
    url: str
    branch: str = "master"
    auth: Optional[RepositoryAuthPayload] = None
    triggers: Optional[RepositoryTriggersPayload] = None

class TestGitPayload(BaseModel):
    url: str
    branch: Optional[str] = "master"
    auth_type: str = "none"
    token: Optional[str] = None
    token_env: Optional[str] = None
    private_key_path: Optional[str] = None

class TestLlmPayload(BaseModel):
    api_base: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None

# --- Эндпоинты ---

@router.get("")
async def get_all_settings(
    username: str = Depends(verify_basic_auth),
    locale: str = Depends(resolve_locale)
):
    """Возвращает текущие настройки системы, LLM, эмбеддингов и источников данных."""
    db = get_db()
    current_locale = db.get_setting("default_locale", "ru") if db else "ru"
    current_theme = db.get_setting("default_theme", "dark") if db else "dark"

    config = load_config()
    server_cfg = config.get("server", {})
    chat_cfg = server_cfg.get("dashboard", {}).get("chat", {})
    emb_cfg = config.get("embeddings", {})

    # Определение эффективных значений LLM
    env_base = os.environ.get("LLM_BASE_URL")
    cfg_base = chat_cfg.get("api_base") or "http://localhost:20128/v1"
    effective_base = env_base or cfg_base

    env_model = os.environ.get("LLM_MODEL")
    cfg_model = chat_cfg.get("model") or "kr/claude-sonnet-4.5"
    effective_model = env_model or cfg_model

    chat_enabled = chat_cfg.get("enabled", False)

    # Интеграции
    env_webhook_sec = os.environ.get("WEBHOOK_SECRET")
    env_github_pat = os.environ.get("GITHUB_PAT")

    repos_config = config.get("repositories", [])
    safe_repos = []
    for r in repos_config:
        safe_r = dict(r)
        # Маскируем токены если они были заданы напрямую
        if "auth" in safe_r and isinstance(safe_r["auth"], dict):
            auth_copy = dict(safe_r["auth"])
            if "token" in auth_copy and auth_copy["token"]:
                auth_copy["token_masked"] = mask_secret(auth_copy["token"])
                del auth_copy["token"]
            safe_r["auth"] = auth_copy
        safe_repos.append(safe_r)

    # Проектные воркспейсы (Multi-repo Worktrees / v1.8)
    for project in config.get("projects", []):
        proj_id = project.get("id")
        ws_root = project.get("workspace_root", "")
        base_path = project.get("base_path", "worktrees/master")
        for p_repo in project.get("repositories", []):
            r_id = p_repo.get("id")
            branch = "master" if r_id != "browser-extension" else "main"
            safe_repos.append({
                "id": f"{proj_id}/{r_id}",
                "url": f"{ws_root}/{base_path}/{p_repo.get('path', r_id)}",
                "branch": branch,
                "auth": {"type": "none"}
            })
        safe_repos.append({
            "id": f"{proj_id}/docs",
            "url": f"{ws_root} (Docs & .kb)",
            "branch": "master",
            "auth": {"type": "none"}
        })

    return {
        "ui": {
            "default_locale": current_locale,
            "default_theme": current_theme
        },
        "llm": {
            "api_base": effective_base,
            "model": effective_model,
            "is_env_api_base": bool(env_base),
            "is_env_model": bool(env_model),
            "chat_enabled": chat_enabled
        },
        "embeddings": {
            "enabled": emb_cfg.get("enabled", False),
            "model_name": emb_cfg.get("model_name", "intfloat/multilingual-e5-small")
        },
        "integrations": {
            "webhook_secret_set": bool(env_webhook_sec or server_cfg.get("webhook_secret")),
            "is_env_webhook_secret": bool(env_webhook_sec),
            "github_pat_set": bool(env_github_pat),
            "is_env_github_pat": bool(env_github_pat)
        },
        "repositories": safe_repos
    }

@router.put("")
async def update_settings(
    payload: GeneralSettingsPayload,
    username: str = Depends(verify_basic_auth),
    locale: str = Depends(resolve_locale)
):
    """Обновляет настройки интерфейса в БД и параметры LLM/эмбеддингов в config.yaml."""
    db = get_db()
    if not db:
        raise HTTPException(
            status_code=500,
            detail=t("system.not_initialized", locale=locale)
        )

    # 1. Сохраняем UI настройки в SQLite
    if payload.default_locale:
        norm_loc = "en" if payload.default_locale.startswith("en") else "ru"
        db.set_setting("default_locale", norm_loc)
        locale = norm_loc
        
    if payload.default_theme:
        db.set_setting("default_theme", payload.default_theme)

    # 2. Обновляем config.yaml
    config = load_config()
    server_cfg = config.setdefault("server", {})
    dash_cfg = server_cfg.setdefault("dashboard", {})
    chat_cfg = dash_cfg.setdefault("chat", {})

    if payload.chat_enabled is not None:
        chat_cfg["enabled"] = payload.chat_enabled
    if payload.llm_base_url is not None:
        chat_cfg["api_base"] = payload.llm_base_url.strip()
        os.environ["LLM_BASE_URL"] = payload.llm_base_url.strip()
    if payload.llm_model is not None:
        chat_cfg["model"] = payload.llm_model.strip()
        os.environ["LLM_MODEL"] = payload.llm_model.strip()

    emb_cfg = config.setdefault("embeddings", {})
    if payload.embeddings_enabled is not None:
        emb_cfg["enabled"] = payload.embeddings_enabled
    if payload.embeddings_model is not None:
        emb_cfg["model_name"] = payload.embeddings_model.strip()

    if payload.webhook_secret is not None:
        server_cfg["webhook_secret"] = payload.webhook_secret.strip()

    save_config(config)

    return {
        "status": "success",
        "message": t("settings.saved", locale=locale)
    }

@router.post("/repositories")
async def create_repository(
    repo: RepositoryPayload,
    username: str = Depends(verify_basic_auth),
    locale: str = Depends(resolve_locale)
):
    """Добавляет новый репозиторий в config.yaml."""
    repo_id = repo.id.strip()
    if not re.match(r"^[a-zA-Z0-9_\-]+$", repo_id):
        raise HTTPException(
            status_code=400,
            detail=t("repo.id_invalid", locale=locale)
        )

    if not repo.url.strip():
        raise HTTPException(
            status_code=400,
            detail=t("repo.url_invalid", locale=locale)
        )

    existing = get_repository_config(repo_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=t("repo.already_exists", locale=locale, repo_id=repo_id)
        )

    repo_dict = repo.model_dump(exclude_none=True)
    add_or_update_repository(repo_dict)

    return {
        "status": "success",
        "message": t("repo.created", locale=locale, repo_id=repo_id),
        "repository": repo_dict
    }

@router.put("/repositories/{repo_id}")
async def update_repository_endpoint(
    repo_id: str,
    repo: RepositoryPayload,
    username: str = Depends(verify_basic_auth),
    locale: str = Depends(resolve_locale)
):
    """Обновляет существующий репозиторий в config.yaml."""
    existing = get_repository_config(repo_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail=t("repo.not_found", locale=locale, repo_id=repo_id)
        )

    repo_dict = repo.model_dump(exclude_none=True)
    repo_dict["id"] = repo_id # Гарантируем неизменность ID при редактировании
    add_or_update_repository(repo_dict)

    return {
        "status": "success",
        "message": t("repo.updated", locale=locale, repo_id=repo_id),
        "repository": repo_dict
    }

@router.delete("/repositories/{repo_id}")
async def delete_repository_endpoint(
    repo_id: str,
    purge_data: bool = Query(False, description="Удалить ли файлы и векторный индекс из SQLite"),
    username: str = Depends(verify_basic_auth),
    locale: str = Depends(resolve_locale)
):
    """Удаляет репозиторий из config.yaml и опционально очищает базу."""
    db = get_db()
    deleted = delete_repository_config(repo_id)
    
    files_deleted = 0
    if purge_data and db:
        files_deleted = db.delete_repo(repo_id)

    if not deleted and files_deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=t("repo.not_found", locale=locale, repo_id=repo_id)
        )

    return {
        "status": "success",
        "message": t("repo.deleted", locale=locale, repo_id=repo_id),
        "repo_id": repo_id,
        "files_purged": files_deleted
    }

@router.post("/test-git")
async def test_git_endpoint(
    payload: TestGitPayload,
    username: str = Depends(verify_basic_auth),
    locale: str = Depends(resolve_locale)
):
    """Тестирует сетевую доступность и валидность учетных данных Git-репозитория."""
    if not payload.url or not payload.url.strip():
        raise HTTPException(
            status_code=400,
            detail=t("git.url_required", locale=locale)
        )

    token = payload.token
    if not token and payload.token_env:
        token = os.environ.get(payload.token_env)
    if not token and payload.auth_type == "pat":
        token = os.environ.get("GITHUB_PAT")

    git_mgr = GitManager(
        pat_token=token if payload.auth_type == "pat" else None,
        ssh_key_path=payload.private_key_path if payload.auth_type == "ssh" else None,
        timeout=15
    )

    success, message = git_mgr.test_connection(payload.url.strip(), branch=payload.branch)
    if success:
        return {
            "success": True,
            "message": t("git.test_success", locale=locale, branch=payload.branch or "master")
        }
    else:
        return {
            "success": False,
            "message": t("git.test_failed", locale=locale, error=message)
        }

@router.post("/test-llm")
async def test_llm_endpoint(
    payload: TestLlmPayload,
    username: str = Depends(verify_basic_auth),
    locale: str = Depends(resolve_locale)
):
    """Тестирует соединение с LLM / OmniRoute / OpenAI эндпоинтом."""
    config = load_config()
    chat_cfg = config.get("server", {}).get("dashboard", {}).get("chat", {})

    base_url = (
        payload.api_base
        or os.environ.get("LLM_BASE_URL")
        or chat_cfg.get("api_base")
        or "http://localhost:20128/v1"
    )
    if os.path.exists("/.dockerenv"):
        base_url = (
            base_url
            .replace("://127.0.0.1:", "://host.docker.internal:")
            .replace("://localhost:", "://host.docker.internal:")
        )
    base_url = base_url.rstrip("/")

    if not base_url.endswith("/chat/completions"):
        endpoint = f"{base_url}/chat/completions"
    else:
        endpoint = base_url

    model = (
        payload.model
        or os.environ.get("LLM_MODEL")
        or chat_cfg.get("model")
        or "kr/claude-sonnet-4.5"
    )

    headers = {"Content-Type": "application/json"}
    api_key = payload.api_key or os.environ.get("LLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(endpoint, json=body, headers=headers)
            if resp.status_code >= 400:
                return {
                    "success": False,
                    "message": t("llm.test_failed", locale=locale, error=f"HTTP {resp.status_code}: {resp.text[:150]}")
                }
            return {
                "success": True,
                "message": t("llm.test_success", locale=locale, model=model)
            }
    except Exception as e:
        return {
            "success": False,
            "message": t("llm.test_failed", locale=locale, error=str(e))
        }
