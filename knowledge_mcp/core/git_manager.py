import os
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
    Предотвращает зависания при помощи явных таймаутов и флагов неинтерактивного режима.
    """

    def __init__(
        self,
        pat_token: Optional[str] = None,
        ssh_key_path: Optional[str | Path] = None,
        timeout: int = 180
    ):
        """Инициализирует GitManager с параметрами авторизации и таймаутом.

        Args:
            pat_token: Персональный токен доступа (PAT) для HTTPS.
            ssh_key_path: Путь к файлу закрытого SSH-ключа.
            timeout: Максимальное время ожидания команд git в секундах (по умолчанию 180).
        """
        self.pat_token = pat_token
        self.ssh_key_path = Path(ssh_key_path) if ssh_key_path else None
        self.timeout = timeout

    def _get_env(self) -> dict[str, str]:
        """Формирует безопасное окружение для неинтерактивного запуска Git."""
        env = os.environ.copy()
        # Запрещаем любые интерактивные запросы паролей / учетных данных в терминале
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_ASKPASS"] = ""
        return env

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

    def _mask_secrets(self, text: str) -> str:
        """Маскирует секреты (токены и base64 заголовки) в строках логов и ошибок."""
        if not self.pat_token:
            return text
        creds = f"x-token-auth:{self.pat_token}"
        b64_creds = base64.b64encode(creds.encode("utf-8")).decode("utf-8")
        return text.replace(b64_creds, "********").replace(self.pat_token, "********")

    def clone(self, repo_url: str, dest_path: str | Path) -> None:
        """Выполняет безопасное клонирование репозитория с таймаутом.

        Args:
            repo_url: URL удаленного репозитория.
            dest_path: Путь для клонирования локально.

        Raises:
            TimeoutError: Если команда git clone превысила время ожидания.
            RuntimeError: Если команда git clone завершилась с ошибкой.
        """
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        auth_args = self._get_auth_args()
        cmd = ["git"] + auth_args + ["clone", repo_url, str(dest_path)]
        env = self._get_env()

        logger.info(f"Cloning repository from {repo_url} to {dest_path}")
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
                env=env
            )
            logger.info("Clone completed successfully.")
            logger.debug(res.stdout)
        except subprocess.TimeoutExpired as e:
            err_msg = f"Git clone timed out after {self.timeout}s for {repo_url}"
            logger.error(err_msg)
            raise TimeoutError(err_msg) from e
        except subprocess.CalledProcessError as e:
            err_msg = self._mask_secrets(e.stderr or str(e))
            logger.error(f"Git clone failed: {err_msg}")
            raise RuntimeError(f"Git clone failed: {err_msg}") from e

    def pull(self, repo_path: str | Path, branch: str = "main") -> None:
        """Выполняет безопасное обновление локального репозитория с таймаутом.

        Args:
            repo_path: Путь к локальному репозиторию.
            branch: Ветка для обновления (по умолчанию 'main').

        Raises:
            FileNotFoundError: Если путь к репозиторию отсутствует на диске.
            TimeoutError: Если команда git pull превысила время ожидания.
            RuntimeError: Если команда git pull завершилась с ошибкой.
        """
        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise FileNotFoundError(f"Local repository path does not exist: {repo_path}")

        auth_args = self._get_auth_args()
        cmd = ["git"] + auth_args + ["pull", "origin", branch]
        env = self._get_env()

        logger.info(f"Pulling branch '{branch}' in repository at {repo_path}")
        try:
            res = subprocess.run(
                cmd,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
                env=env
            )
            logger.info("Pull completed successfully.")
            logger.debug(res.stdout)
        except subprocess.TimeoutExpired as e:
            err_msg = f"Git pull timed out after {self.timeout}s for {repo_path}"
            logger.error(err_msg)
            raise TimeoutError(err_msg) from e
        except subprocess.CalledProcessError as e:
            err_msg = self._mask_secrets(e.stderr or str(e))
            logger.error(f"Git pull failed: {err_msg}")
            raise RuntimeError(f"Git pull failed: {err_msg}") from e

    def test_connection(self, repo_url: str, branch: Optional[str] = None) -> tuple[bool, str]:
        """Проверяет доступность удаленного Git-репозитория и корректность авторизации.

        Args:
            repo_url: URL удаленного репозитория.
            branch: Опциональная ветка для проверки существования.

        Returns:
            (success: bool, message: str)
        """
        auth_args = self._get_auth_args()
        cmd = ["git"] + auth_args + ["ls-remote", "--heads", repo_url]
        if branch:
            cmd.append(branch)

        env = self._get_env()
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=min(self.timeout, 15),
                env=env
            )
            output = res.stdout.strip()
            if branch and output and branch not in output and f"refs/heads/{branch}" not in output:
                return False, f"Branch '{branch}' not found on remote"
            return True, "Connection successful"
        except subprocess.TimeoutExpired:
            return False, "Connection timed out (15s limit)"
        except subprocess.CalledProcessError as e:
            err_msg = self._mask_secrets(e.stderr or str(e)).strip()
            return False, err_msg
        except Exception as e:
            return False, str(e)


