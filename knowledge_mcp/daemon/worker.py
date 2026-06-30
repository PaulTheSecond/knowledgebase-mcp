import os
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.core.config import get_repository_config
from knowledge_mcp.core.git_manager import GitManager
from knowledge_mcp.core.indexer import Indexer

logger = logging.getLogger(__name__)

class DaemonWorker:
    """Фоновый воркер, опрашивающий БД-очередь задач и выполняющий синхронизацию репозиториев."""

    def __init__(self, db_path: str | Path):
        """Инициализирует воркер.

        Args:
            db_path: Путь к файлу базы данных SQLite.
        """
        self.db_path = Path(db_path)
        self.db = KnowledgeDB(self.db_path)
        self.indexer = Indexer(self.db)
        # Директория клонирования (data/clones по умолчанию)
        self.clones_dir = Path(os.environ.get("CLONES_DIR", "data/clones"))
        self.clones_dir.mkdir(parents=True, exist_ok=True)

    def get_git_sha(self, repo_path: Path) -> Optional[str]:
        """Определяет хэш HEAD коммита репозитория.

        Args:
            repo_path: Путь к локальной директории репозитория.

        Returns:
            SHA коммита или None, если произошла ошибка.
        """
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True
            )
            return res.stdout.strip()
        except Exception as e:
            logger.error(f"Failed to get git sha at {repo_path}: {e}")
            return None

    def run_job(self, job: Dict[str, Any]) -> None:
        """Обрабатывает одну задачу из очереди.

        Args:
            job: Словарь с описанием задачи (id, repo_id, attempts).
        """
        job_id = job["id"]
        repo_id = job["repo_id"]
        attempts = job.get("attempts", 0) + 1

        logger.info(f"Processing job {job_id} for repo '{repo_id}' (attempt {attempts})")

        # Получаем конфигурацию репозитория из config.yaml
        repo_config = get_repository_config(repo_id)
        if not repo_config:
            err = f"Repository config for '{repo_id}' not found in config.yaml"
            logger.error(err)
            self.db.update_job_status(job_id, "failed", attempts=attempts, error_message=err)
            return

        git_url = repo_config.get("url")
        branch = repo_config.get("branch", "master")
        auth = repo_config.get("auth", {})
        auth_type = auth.get("type")

        pat_token = None
        ssh_key_path = None

        if auth_type == "pat":
            token_env = auth.get("token_env", "GITHUB_PAT")
            pat_token = os.environ.get(token_env)
            if not pat_token:
                err = f"PAT token environment variable '{token_env}' is not set."
                logger.error(err)
                self.db.update_job_status(job_id, "failed", attempts=attempts, error_message=err)
                return
        elif auth_type == "ssh":
            ssh_key_path = auth.get("private_key_path")

        # Настраиваем GitManager
        git_mgr = GitManager(pat_token=pat_token, ssh_key_path=ssh_key_path)
        dest_path = self.clones_dir / repo_id

        # Обновляем глобальный статус системы на 'syncing'
        self.db.update_system_status("syncing")

        try:
            # Клонируем или обновляем репозиторий
            if not dest_path.exists() or not (dest_path / ".git").exists():
                logger.info(f"Cloning '{repo_id}' from remote URL")
                git_mgr.clone(git_url, dest_path)
            else:
                logger.info(f"Pulling latest updates for '{repo_id}'")
                git_mgr.pull(dest_path, branch=branch)

            # Получаем HEAD SHA
            current_sha = self.get_git_sha(dest_path)
            if not current_sha:
                raise RuntimeError("Failed to retrieve Git SHA after synchronization.")

            # Проверяем идемпотентность по SHA коммита
            last_indexed_sha = self.db.get_indexed_commit(repo_id)
            if last_indexed_sha == current_sha:
                logger.info(f"Repository '{repo_id}' is already indexed at SHA {current_sha}. Skipping indexer.")
                self.db.update_job_status(job_id, "completed")
                self.db.update_system_status("idle", last_indexed_at=time.time())
                return

            # Запускаем индексацию репозитория
            logger.info(f"SHA changed ({last_indexed_sha} -> {current_sha}). Indexing repo '{repo_id}'...")
            self.indexer.sync_repo(repo_id, str(dest_path))

            # Сохраняем информацию об успешном индексировании коммита
            self.db.update_indexed_commit(repo_id, current_sha)
            self.db.update_job_status(job_id, "completed")
            self.db.update_system_status("idle", last_indexed_at=time.time())
            logger.info(f"Job {job_id} completed successfully for repo '{repo_id}' at SHA {current_sha}")

        except Exception as e:
            logger.exception(f"Error processing job {job_id} for repo '{repo_id}': {e}")
            err_msg = str(e)
            # В Story 1.3 выставляем failed. Ретраи и экспоненциальный back-off будут в Story 1.4
            self.db.update_job_status(job_id, "failed", attempts=attempts, error_message=err_msg)
            self.db.update_system_status("idle", last_error=err_msg)

    def start(self):
        """Запускает бесконечный цикл обработки очереди."""
        logger.info("Starting background daemon worker loop...")
        while True:
            try:
                # Проверяем статус воркера в БД
                status_row = self.db.get_system_status()
                if status_row and status_row["status"] == "halted":
                    logger.debug("Worker is halted. Sleeping...")
                    time.sleep(1.0)
                    continue

                # Извлекаем очередную задачу
                job = self.db.fetch_next_job()
                if job:
                    self.run_job(job)
                else:
                    time.sleep(1.0)
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}")
                time.sleep(1.0)
