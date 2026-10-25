import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

def get_config_path() -> Path:
    """Возвращает путь к файлу конфигурации на основе переменных окружения."""
    path_str = os.environ.get("CONFIG_PATH", "data/config.yaml")
    return Path(path_str)

def load_config() -> Dict[str, Any]:
    """Загружает конфигурационный файл config.yaml."""
    path = get_config_path()
    if not path.exists():
        logger.warning(f"Configuration file not found at: {path.absolute()}. Returning default values.")
        return {
            "server": {
                "port": 8000,
                "host": "127.0.0.1",
                "dashboard": {
                    "enabled": False,
                    "chat": {
                        "enabled": False
                    }
                }
            },
            "embeddings": {
                "enabled": False,
                "model_name": "intfloat/multilingual-e5-small"
            },
            "repositories": [],
            "projects": []
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                data.setdefault("repositories", [])
                data.setdefault("projects", [])
                return data
            return {"repositories": [], "projects": []}
    except Exception as e:
        logger.error(f"Failed to parse config.yaml: {e}")
        return {"repositories": [], "projects": []}

def save_config(config_data: Dict[str, Any]) -> None:
    """Безопасно и атомарно сохраняет конфигурацию в config.yaml."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        tmp_path.replace(path)
        logger.info(f"Configuration saved successfully to {path}")
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        logger.error(f"Failed to save config.yaml: {e}")
        raise e

# --- Методы для работы с одиночными репозиториями (100% обратная совместимость) ---

def get_repository_config(repo_id: str) -> Optional[Dict[str, Any]]:
    """Находит конфигурацию репозитория по его ID."""
    config = load_config()
    for repo in config.get("repositories", []):
        if repo.get("id") == repo_id:
            return repo
    return None

def add_or_update_repository(repo_dict: Dict[str, Any]) -> None:
    """Добавляет новый репозиторий или обновляет существующий по его ID."""
    config = load_config()
    repos: List[Dict[str, Any]] = config.setdefault("repositories", [])
    
    repo_id = repo_dict.get("id")
    found_idx = None
    for idx, r in enumerate(repos):
        if r.get("id") == repo_id:
            found_idx = idx
            break
            
    if found_idx is not None:
        repos[found_idx] = repo_dict
    else:
        repos.append(repo_dict)
        
    save_config(config)

def delete_repository_config(repo_id: str) -> bool:
    """Удаляет конфигурацию репозитория из config.yaml по ID."""
    config = load_config()
    repos = config.get("repositories", [])
    initial_len = len(repos)
    repos = [r for r in repos if r.get("id") != repo_id]
    
    if len(repos) < initial_len:
        config["repositories"] = repos
        save_config(config)
        return True
    return False

# --- Методы для работы с мульти-репозиторными воркспейсами и проектами ---

def get_projects() -> List[Dict[str, Any]]:
    """Возвращает список сконфигурированных проектов/воркспейсов."""
    config = load_config()
    return config.get("projects", [])

def get_project_config(project_id: str) -> Optional[Dict[str, Any]]:
    """Находит конфигурацию проекта по его ID."""
    config = load_config()
    for proj in config.get("projects", []):
        if proj.get("id") == project_id:
            return proj
    return None

def add_or_update_project(project_dict: Dict[str, Any]) -> None:
    """Добавляет новый проект или обновляет существующий по его ID."""
    config = load_config()
    projects: List[Dict[str, Any]] = config.setdefault("projects", [])
    
    project_id = project_dict.get("id")
    found_idx = None
    for idx, p in enumerate(projects):
        if p.get("id") == project_id:
            found_idx = idx
            break
            
    if found_idx is not None:
        projects[found_idx] = project_dict
    else:
        projects.append(project_dict)
        
    save_config(config)

def delete_project_config(project_id: str) -> bool:
    """Удаляет конфигурацию проекта из config.yaml по ID."""
    config = load_config()
    projects = config.get("projects", [])
    initial_len = len(projects)
    projects = [p for p in projects if p.get("id") != project_id]
    
    if len(projects) < initial_len:
        config["projects"] = projects
        save_config(config)
        return True
    return False

def resolve_workspace_targets(project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Разрешает и возвращает список всех активных целей (ворктри, репозиториев, документации)
    для указанного проекта или для всех настроенных проектов/репозиториев.
    """
    from knowledge_mcp.core.workspace_scanner import WorkspaceScanner
    
    scanner = WorkspaceScanner()
    targets: List[Dict[str, Any]] = []
    
    # 1. Если указан конкретный project_id
    if project_id is not None:
        proj = get_project_config(project_id)
        if proj:
            discovered = scanner.scan_workspace(proj)
            targets.extend(t.to_dict() for t in discovered)
        elif project_id == "default":
            # Разрешаем стандартные одиночные репозитории
            config = load_config()
            for repo in config.get("repositories", []):
                repo_path = Path(repo["local_path"]).resolve() if repo.get("local_path") else None
                branch = repo.get("branch", "master")
                if repo_path and repo_path.exists():
                    branch = scanner.get_git_branch(repo_path)
                targets.append({
                    "project_id": "default",
                    "repo_id": repo.get("id"),
                    "branch": branch,
                    "task_name": None,
                    "absolute_path": repo_path,
                    "is_base": True,
                    "is_doc": False,
                    "language": repo.get("language"),
                })
        return targets

    # 2. Если project_id не указан (все проекты и репозитории)
    for proj in get_projects():
        discovered = scanner.scan_workspace(proj)
        targets.extend(t.to_dict() for t in discovered)

    config = load_config()
    for repo in config.get("repositories", []):
        repo_path = Path(repo["local_path"]).resolve() if repo.get("local_path") else None
        branch = repo.get("branch", "master")
        if repo_path and repo_path.exists():
            branch = scanner.get_git_branch(repo_path)
        targets.append({
            "project_id": repo.get("project_id", "default"),
            "repo_id": repo.get("id"),
            "branch": branch,
            "task_name": None,
            "absolute_path": repo_path,
            "is_base": True,
            "is_doc": False,
            "language": repo.get("language"),
        })

    return targets

def mask_secret(value: Optional[str]) -> str:
    """Маскирует секретные токены и пароли для безопасного отображения в UI."""
    if not value:
        return ""
    val_str = str(value)
    if len(val_str) <= 6:
        return "******"
    return f"{val_str[:4]}****{val_str[-2:]}"
