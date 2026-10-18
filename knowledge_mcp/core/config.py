import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

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
            "repositories": []
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error(f"Failed to parse config.yaml: {e}")
        return {}

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

def mask_secret(value: Optional[str]) -> str:
    """Маскирует секретные токены и пароли для безопасного отображения в UI."""
    if not value:
        return ""
    val_str = str(value)
    if len(val_str) <= 6:
        return "******"
    return f"{val_str[:4]}****{val_str[-2:]}"
