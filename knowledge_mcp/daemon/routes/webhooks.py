"""Обработчик вебхуков Git-провайдеров (GitHub, Gitea, GitLab).

Реализует HMAC-SHA256 валидацию подписей для GitHub/Gitea
и проверку статического токена для GitLab.
"""
import hmac
import hashlib
import logging
import os
from typing import Optional

from fastapi import APIRouter, Request, HTTPException

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.core.config import load_config

logger = logging.getLogger(__name__)

router = APIRouter()

# Ссылка на БД — инжектируется при подключении роутера
_db: Optional[KnowledgeDB] = None


def init_webhook_router(db: KnowledgeDB) -> APIRouter:
    """Инициализирует роутер вебхуков с указанной БД.

    Args:
        db: Инстанс KnowledgeDB для записи задач в очередь.

    Returns:
        Настроенный APIRouter.
    """
    global _db
    _db = db
    return router


def _verify_hmac_sha256(payload: bytes, secret: str, signature: str) -> bool:
    """Проверяет HMAC-SHA256 подпись запроса.

    Args:
        payload: Тело запроса в байтах.
        secret: Секретный ключ для проверки.
        signature: Значение подписи из заголовка (с или без префикса sha256=).

    Returns:
        True если подпись совпадает.
    """
    expected = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    # Поддерживаем подпись с и без префикса sha256=
    if not signature.startswith("sha256="):
        signature = "sha256=" + signature
    return hmac.compare_digest(expected, signature)


def normalize_git_url(url: str) -> str:
    """Приводит Git URL к единому виду host/path (без протоколов, ssh-префиксов, портов и .git).
    
    Примеры:
        'git@github.com:org/repo.git' -> 'github.com/org/repo'
        'https://github.com/org/repo.git' -> 'github.com/org/repo'
        'ssh://git@gitlab.com:22/org/repo.git' -> 'gitlab.com/org/repo'
    """
    import re
    if not url:
        return ""
    clean = url.strip().rstrip("/")
    if clean.endswith(".git"):
        clean = clean[:-4]

    # Убираем схему протокола (https://, http://, ssh://)
    for prefix in ("https://", "http://", "ssh://"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break

    # Обработка SSH-пользователя (например git@)
    if "@" in clean:
        clean = clean.split("@", 1)[1]

    # Удаляем номер порта (например :22/ -> /)
    clean = re.sub(r":\d+/", "/", clean)
    clean = clean.replace(":", "/")

    while "//" in clean:
        clean = clean.replace("//", "/")

    return clean.lower()


def _find_repo_config_by_url(clone_url: str) -> Optional[dict]:
    """Ищет конфигурацию репозитория по clone URL.

    Поддерживает сопоставление между SSH и HTTPS форматами URL.

    Args:
        clone_url: URL репозитория из payload вебхука.

    Returns:
        Словарь конфигурации репозитория или None.
    """
    config = load_config()
    norm_clone = normalize_git_url(clone_url)

    for repo in config.get("repositories", []):
        repo_url = repo.get("url", "")
        if normalize_git_url(repo_url) == norm_clone:
            return repo
    return None


def _extract_repo_info_github(body: dict) -> Optional[str]:
    """Извлекает clone URL из payload GitHub.

    Args:
        body: Тело вебхука в виде словаря.

    Returns:
        Clone URL репозитория или None.
    """
    repo = body.get("repository", {})
    return repo.get("clone_url") or repo.get("html_url")


def _extract_repo_info_gitea(body: dict) -> Optional[str]:
    """Извлекает clone URL из payload Gitea.

    Args:
        body: Тело вебхука в виде словаря.

    Returns:
        Clone URL репозитория или None.
    """
    repo = body.get("repository", {})
    return repo.get("clone_url") or repo.get("html_url")


def _extract_repo_info_gitlab(body: dict) -> Optional[str]:
    """Извлекает clone URL из payload GitLab.

    Args:
        body: Тело вебхука в виде словаря.

    Returns:
        Clone URL репозитория или None.
    """
    project = body.get("project", {})
    return project.get("git_http_url") or project.get("web_url")


@router.post("/webhook/git")
async def webhook_git(request: Request):
    """Обработчик вебхуков от Git-провайдеров.

    Поддерживает GitHub, Gitea и GitLab. Валидирует подпись/токен
    и ставит задачу синхронизации в очередь.
    """
    if _db is None:
        raise HTTPException(status_code=500, detail="Webhook handler not initialized")

    raw_body = await request.body()
    body = await request.json()

    # Определяем провайдер по заголовкам
    github_sig = request.headers.get("X-Hub-Signature-256")
    gitea_sig = request.headers.get("X-Gitea-Signature")
    gitlab_token = request.headers.get("X-Gitlab-Token")

    clone_url: Optional[str] = None
    provider: str = "unknown"

    if github_sig:
        provider = "github"
        clone_url = _extract_repo_info_github(body)
    elif gitea_sig:
        provider = "gitea"
        clone_url = _extract_repo_info_gitea(body)
    elif gitlab_token:
        provider = "gitlab"
        clone_url = _extract_repo_info_gitlab(body)
    else:
        logger.warning("Webhook received without signature/token headers")
        raise HTTPException(status_code=401, detail="Missing signature or token header")

    if not clone_url:
        raise HTTPException(status_code=400, detail="Could not extract repository URL from webhook payload")

    # Ищем конфигурацию репозитория
    repo_config = _find_repo_config_by_url(clone_url)
    if not repo_config:
        logger.warning(f"No repository config found for URL: {clone_url}")
        raise HTTPException(status_code=404, detail=f"Repository not configured: {clone_url}")

    repo_id = repo_config["id"]

    # Проверяем что вебхуки включены для этого репозитория
    triggers = repo_config.get("triggers", {})
    webhooks_config = triggers.get("webhooks", {})
    if not webhooks_config.get("enabled", False):
        logger.warning(f"Webhooks disabled for repo '{repo_id}'")
        raise HTTPException(status_code=403, detail=f"Webhooks disabled for repository '{repo_id}'")

    # Получаем секрет из переменной окружения
    secret_env = webhooks_config.get("secret_env", "WEBHOOK_SECRET")
    secret = os.environ.get(secret_env, "")

    if not secret:
        logger.error(f"Webhook secret env var '{secret_env}' is not set")
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    # Валидируем подпись/токен
    if provider == "github":
        if not _verify_hmac_sha256(raw_body, secret, github_sig):
            logger.warning(f"Invalid GitHub signature for repo '{repo_id}'")
            raise HTTPException(status_code=401, detail="Invalid signature")
    elif provider == "gitea":
        if not _verify_hmac_sha256(raw_body, secret, gitea_sig):
            logger.warning(f"Invalid Gitea signature for repo '{repo_id}'")
            raise HTTPException(status_code=401, detail="Invalid signature")
    elif provider == "gitlab":
        # GitLab использует простой сравниваемый токен, а не HMAC
        if not hmac.compare_digest(secret, gitlab_token):
            logger.warning(f"Invalid GitLab token for repo '{repo_id}'")
            raise HTTPException(status_code=401, detail="Invalid token")

    # Подпись валидна — ставим задачу в очередь
    job_id = _db.add_to_queue(repo_id)
    logger.info(f"Webhook [{provider}] received for '{repo_id}'. Job {job_id} queued.")

    return {"status": "queued", "repo_id": repo_id, "job_id": job_id, "provider": provider}
