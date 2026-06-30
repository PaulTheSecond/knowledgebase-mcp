import base64
import subprocess
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

class GitManager:
    """Управляет безопасным клонированием и обновлением репозиториев Git.

    Поддерживает авторизацию HTTPS PAT (токен хранится только в RAM)
    и SSH-ключи (без интерактивной проверки StrictHostKeyChecking).
    """

    def __init__(self, pat_token: Optional[str] = None, ssh_key_path: Optional[str | Path] = None):
        """Инициализирует GitManager с параметрами авторизации.

        Args:
            pat_token: Персональный токен доступа (PAT) для HTTPS.
            ssh_key_path: Путь к файлу закрытого SSH-ключа.
        """
        self.pat_token = pat_token
        self.ssh_key_path = Path(ssh_key_path) if ssh_key_path else None

    def _get_auth_args(self) -> List[str]:
        """Формирует аргументы конфигурации Git для авторизации.

        Returns:
            Список строковых аргументов для git CLI.
        """
        args = []
        if self.pat_token:
            # Для GitHub PAT заголовок: Authorization: Basic base64("x-token-auth:PAT")
            creds = f"x-token-auth:{self.pat_token}"
            b64_creds = base64.b64encode(creds.encode("utf-8")).decode("utf-8")
            args.extend(["-c", f"http.extraHeader=Authorization: Basic {b64_creds}"])
            logger.info("Configured HTTPS PAT auth in memory via http.extraHeader")

        if self.ssh_key_path and self.ssh_key_path.exists():
            # SSH-подключение с отключением StrictHostKeyChecking
            ssh_path = str(self.ssh_key_path).replace("\\", "/")
            args.extend(["-c", f'core.sshCommand=ssh -i "{ssh_path}" -o StrictHostKeyChecking=no'])
            logger.info("Configured SSH key auth via core.sshCommand")

        return args

    def clone(self, repo_url: str, dest_path: str | Path) -> None:
        """Выполняет безопасное клонирование репозитория.

        Args:
            repo_url: URL удаленного репозитория.
            dest_path: Путь для клонирования локально.

        Raises:
            RuntimeError: Если команда git clone завершилась с ошибкой.
        """
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        auth_args = self._get_auth_args()
        cmd = ["git"] + auth_args + ["clone", repo_url, str(dest_path)]

        logger.info(f"Cloning repository from {repo_url} to {dest_path}")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Clone completed successfully.")
            logger.debug(res.stdout)
        except subprocess.CalledProcessError as e:
            # Маскируем токен в логах ошибок
            err_msg = e.stderr
            if self.pat_token:
                creds = f"x-token-auth:{self.pat_token}"
                b64_creds = base64.b64encode(creds.encode("utf-8")).decode("utf-8")
                err_msg = err_msg.replace(b64_creds, "********")
            logger.error(f"Git clone failed: {err_msg}")
            raise RuntimeError(f"Git clone failed: {err_msg}") from e

    def pull(self, repo_path: str | Path, branch: str = "main") -> None:
        """Выполняет безопасное обновление локального репозитория.

        Args:
            repo_path: Путь к локальному репозиторию.
            branch: Ветка для обновления (по умолчанию 'main').

        Raises:
            FileNotFoundError: Если путь к репозиторию отсутствует на диске.
            RuntimeError: Если команда git pull завершилась с ошибкой.
        """
        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise FileNotFoundError(f"Local repository path does not exist: {repo_path}")

        auth_args = self._get_auth_args()
        cmd = ["git"] + auth_args + ["pull", "origin", branch]

        logger.info(f"Pulling branch '{branch}' in repository at {repo_path}")
        try:
            res = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True, check=True)
            logger.info("Pull completed successfully.")
            logger.debug(res.stdout)
        except subprocess.CalledProcessError as e:
            # Маскируем токен в логах ошибок
            err_msg = e.stderr
            if self.pat_token:
                creds = f"x-token-auth:{self.pat_token}"
                b64_creds = base64.b64encode(creds.encode("utf-8")).decode("utf-8")
                err_msg = err_msg.replace(b64_creds, "********")
            logger.error(f"Git pull failed: {err_msg}")
            raise RuntimeError(f"Git pull failed: {err_msg}") from e
