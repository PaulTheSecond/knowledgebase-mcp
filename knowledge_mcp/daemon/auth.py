"""Модуль аутентификации и авторизации (Story 2.1).

Реализует авторизацию Basic Auth для REST API панели управления,
а также генерацию и валидацию подписанных временных токенов для сокетов/SSE.
"""
import hmac
import hashlib
import os
import secrets
import time
import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

logger = logging.getLogger(__name__)

# Инициализируем генератор безопасности
# Используем случайный ключ для этого сеанса демона
SECRET_KEY = os.environ.get("DAEMON_SECRET_KEY", secrets.token_hex(32))

security = HTTPBasic()


def get_dashboard_password() -> Optional[str]:
    """Возвращает пароль для панели управления из переменных окружения.

    Если переменная DASHBOARD_PASSWORD не установлена или пуста, возвращает None.
    """
    password = os.environ.get("DASHBOARD_PASSWORD")
    if not password:
        return None
    return password


def verify_basic_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """FastAPI Dependency для проверки HTTP Basic credentials.

    Проверяет переданные учетные данные. Если пароль DASHBOARD_PASSWORD
    не задан, доступ принудительно блокируется с кодом 403 Forbidden.
    """
    password = get_dashboard_password()
    if not password:
        logger.warning("Access blocked: DASHBOARD_PASSWORD is not set or empty")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Dashboard disabled. DASHBOARD_PASSWORD is not configured.",
        )

    # Проверяем пароль. Имя пользователя может быть любым (например, admin)
    is_correct_password = hmac.compare_digest(
        credentials.password.encode("utf-8"), password.encode("utf-8")
    )

    if not is_correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


def generate_temp_token(username: str) -> str:
    """Генерирует короткоживущий (60 сек) криптографически подписанный токен.

    Формат токена: timestamp.uuid.signature
    """
    timestamp = str(int(time.time()))
    uid = secrets.token_hex(16)
    message = f"{timestamp}.{uid}"
    signature = hmac.new(
        SECRET_KEY.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{timestamp}.{uid}.{signature}"


def verify_temp_token(token: str) -> bool:
    """Проверяет валидность и время жизни временного токена.

    Токен валиден в течение 60 секунд. Подпись проверяется с использованием HMAC-SHA256.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return False

        timestamp_str, uid, signature = parts
        timestamp = int(timestamp_str)

        # Проверка времени жизни (60 секунд)
        now = int(time.time())
        if now - timestamp > 60 or timestamp - now > 5:
            logger.warning("Token expired or invalid timestamp")
            return False

        # Проверка подписи
        message = f"{timestamp_str}.{uid}"
        expected_signature = hmac.new(
            SECRET_KEY.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected_signature.encode("utf-8"), signature.encode("utf-8"))
    except Exception as e:
        logger.error(f"Error validating token: {e}")
        return False
