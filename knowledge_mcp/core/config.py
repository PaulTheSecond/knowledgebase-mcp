import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

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
                    "enabled": False
                }
            },
            "embeddings": {
                "enabled": False
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

def get_repository_config(repo_id: str) -> Optional[Dict[str, Any]]:
    """Находит конфигурацию репозитория по его ID."""
    config = load_config()
    for repo in config.get("repositories", []):
        if repo.get("id") == repo_id:
            return repo
    return None
