"""Модуль интернационализации бэкенда (Backend i18n).

Обеспечивает мультиязычность для всех сообщений об ошибках, статусов очередей,
ответов API и нотификаций WebSocket.
"""

from typing import Optional, Dict, Any

SUPPORTED_LOCALES = ("ru", "en")
DEFAULT_LOCALE = "ru"

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "ru": {
        # Аутентификация
        "auth.unauthorized": "Неверный логин или пароль",
        "auth.invalid_token": "Неверный или просроченный токен",
        "auth.forbidden": "Доступ запрещен",

        # Системные компоненты
        "system.not_initialized": "Компоненты системы не инициализированы",
        "system.internal_error": "Внутренняя ошибка сервера: {error}",

        # Репозитории
        "repo.not_found": "Репозиторий '{repo_id}' не найден",
        "repo.already_exists": "Репозиторий с ID '{repo_id}' уже существует",
        "repo.id_invalid": "ID репозитория может содержать только латинские буквы, цифры, дефис и подчеркивание",
        "repo.url_invalid": "Неверный Git URL репозитория",
        "repo.deleted": "Репозиторий '{repo_id}' успешно удален",
        "repo.created": "Репозиторий '{repo_id}' успешно добавлен",
        "repo.updated": "Параметры репозитория '{repo_id}' обновлены",

        # Тестирование Git & LLM
        "git.test_success": "Соединение с Git-репозиторием успешно проверено (ветка '{branch}')",
        "git.test_failed": "Ошибка подключения к Git-репозиторию: {error}",
        "git.url_required": "URL репозитория обязателен для проверки",
        "llm.test_success": "Соединение с LLM успешно проверено. Модель: {model}",
        "llm.test_failed": "Ошибка обращения к LLM: {error}",

        # Чат
        "chat.disabled": "RAG Чат отключен в настройках системы",
        "chat.no_user_message": "Не найдено сообщение пользователя",
        "chat.model_error": "Ошибка модели LLM: {error}",

        # Настройки
        "settings.saved": "Настройки успешно сохранены",
        "settings.invalid_payload": "Некорректные параметры настроек: {error}",

        # Очередь задач и синхронизация
        "queue.sync_queued": "Синхронизация репозитория '{repo_id}' поставлена в очередь (задача #{job_id})",
        "queue.retried": "Сброшено задач для повторной попытки: {count}",
        "queue.job_stopped": "Задача #{job_id} успешно остановлена",
        "queue.job_cannot_stop": "Не удалось остановить задачу #{job_id} (задача уже завершена)",
        "queue.job_restarted": "Задача #{job_id} успешно перезапущена",
        "queue.job_deleted": "Задача #{job_id} успешно удалена",
        "queue.job_not_found": "Задача #{job_id} не найдена в очереди",

        # Статусы фонового воркера
        "worker.idle": "Воркер: Ожидание",
        "worker.processing": "Воркер: Индексация",
        "worker.halted": "Воркер: Остановлен (Критическая ошибка)",
        "worker.error": "Воркер: Ошибка",

        # WebSocket-прогресс индексации
        "ws.syncing": "Синхронизация",
        "ws.cloning": "Клонирование репозитория {repo_id}...",
        "ws.pulling": "Получение обновлений ветки {branch}...",
        "ws.scanning": "Сканирование структуры файлов...",
        "ws.indexing": "Индексация файлов: {current}/{total} ({percent}%)",
        "ws.embeddings": "Вычисление векторных эмбеддингов ({current}/{total})...",
        "ws.completed": "Синхронизация успешно завершена",
        "ws.failed": "Ошибка синхронизации: {error}",
    },
    "en": {
        # Authentication
        "auth.unauthorized": "Invalid username or password",
        "auth.invalid_token": "Invalid or expired token",
        "auth.forbidden": "Access forbidden",

        # System components
        "system.not_initialized": "System components not initialized",
        "system.internal_error": "Internal server error: {error}",

        # Repositories
        "repo.not_found": "Repository '{repo_id}' not found",
        "repo.already_exists": "Repository with ID '{repo_id}' already exists",
        "repo.id_invalid": "Repository ID may only contain alphanumeric characters, hyphens, and underscores",
        "repo.url_invalid": "Invalid Git repository URL",
        "repo.deleted": "Repository '{repo_id}' deleted successfully",
        "repo.created": "Repository '{repo_id}' created successfully",
        "repo.updated": "Repository '{repo_id}' updated successfully",

        # Testing Git & LLM
        "git.test_success": "Git repository connection verified successfully (branch '{branch}')",
        "git.test_failed": "Git connection error: {error}",
        "git.url_required": "Repository URL is required for testing",
        "llm.test_success": "LLM connection verified successfully. Model: {model}",
        "llm.test_failed": "LLM query error: {error}",

        # Chat
        "chat.disabled": "RAG Chat is disabled in system settings",
        "chat.no_user_message": "No user message found",
        "chat.model_error": "LLM model error: {error}",

        # Settings
        "settings.saved": "Settings saved successfully",
        "settings.invalid_payload": "Invalid settings parameters: {error}",

        # Queue
        "queue.sync_queued": "Sync for repository '{repo_id}' queued (job #{job_id})",
        "queue.retried": "Retried failed jobs: {count}",
        "queue.job_stopped": "Job #{job_id} stopped successfully",
        "queue.job_cannot_stop": "Could not stop job #{job_id} (job already finished)",
        "queue.job_restarted": "Job #{job_id} restarted successfully",
        "queue.job_deleted": "Job #{job_id} deleted successfully",
        "queue.job_not_found": "Job #{job_id} not found in queue",

        # Worker status
        "worker.idle": "Worker: Idle",
        "worker.processing": "Worker: Indexing",
        "worker.halted": "Worker: Halted (Critical Error)",
        "worker.error": "Worker: Error",

        # WebSocket indexing progress
        "ws.syncing": "Syncing",
        "ws.cloning": "Cloning repository {repo_id}...",
        "ws.pulling": "Pulling branch {branch} updates...",
        "ws.scanning": "Scanning file structure...",
        "ws.indexing": "Indexing files: {current}/{total} ({percent}%)",
        "ws.embeddings": "Generating vector embeddings ({current}/{total})...",
        "ws.completed": "Synchronization completed successfully",
        "ws.failed": "Synchronization error: {error}",
    }
}


def normalize_locale(locale: Optional[str]) -> str:
    """Нормализует код локали к поддерживаемым значениям ('ru' или 'en')."""
    if not locale:
        return DEFAULT_LOCALE
    loc = locale.lower().strip()
    if loc.startswith("en"):
        return "en"
    if loc.startswith("ru"):
        return "ru"
    return DEFAULT_LOCALE


def get_locale_from_header(accept_language: Optional[str], default_locale: str = DEFAULT_LOCALE) -> str:
    """Извлекает и нормализует предпочтительный язык из HTTP-заголовка Accept-Language."""
    if not accept_language:
        return normalize_locale(default_locale)
    
    # Парсим список языков: ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7
    langs = [part.split(";")[0].strip() for part in accept_language.split(",") if part.strip()]
    for lang in langs:
        norm = normalize_locale(lang)
        if norm in SUPPORTED_LOCALES:
            return norm
            
    return normalize_locale(default_locale)


def t(key: str, locale: str = DEFAULT_LOCALE, **kwargs: Any) -> str:
    """Возвращает локализованную строку по ключу с подстановкой параметров.
    
    Если ключ отсутствует в выбранной локали, пытается взять из дефолтной ('ru').
    Если отсутствует везде, возвращает сам ключ.
    """
    norm_locale = normalize_locale(locale)
    trans_map = TRANSLATIONS.get(norm_locale, TRANSLATIONS[DEFAULT_LOCALE])
    template = trans_map.get(key)
    if template is None and norm_locale != DEFAULT_LOCALE:
        template = TRANSLATIONS[DEFAULT_LOCALE].get(key)
    if template is None:
        template = key
    
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template


def get_text(key: str, locale: str = DEFAULT_LOCALE, **kwargs: Any) -> str:
    """Алиас для функции t()."""
    return t(key, locale=locale, **kwargs)
