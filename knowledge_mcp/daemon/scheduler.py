"""Легковесный Cron-планировщик для периодической синхронизации репозиториев.

Читает расписание из config.yaml и добавляет задачи в sync_queue по расписанию.
Не требует внешних зависимостей — использует стандартный threading.
"""
import logging
import re
import threading
from typing import Optional

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.core.config import load_config

logger = logging.getLogger(__name__)


def _parse_cron_interval(cron_expr: str) -> Optional[int]:
    """Парсит упрощённое cron-выражение и возвращает интервал в секундах.

    Поддерживает стандартные 5-полевые cron-выражения с */N.
    Для более сложных выражений возвращает None.

    Примеры:
        "*/5 * * * *"  -> 300 (каждые 5 минут)
        "0 */4 * * *"  -> 14400 (каждые 4 часа)
        "0 0 * * *"    -> 86400 (каждый день в полночь)

    Args:
        cron_expr: Cron-выражение в формате "минуты часы день месяц день_недели".

    Returns:
        Интервал в секундах или None, если выражение слишком сложное.
    """
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        logger.warning(f"Invalid cron expression '{cron_expr}': expected 5 fields")
        return None

    minute, hour, day, month, weekday = parts

    # Паттерн */N — повторение каждые N единиц
    step_re = re.compile(r"^\*/(\d+)$")

    # Каждые N минут: "*/N * * * *"
    m_match = step_re.match(minute)
    if m_match and hour == "*" and day == "*" and month == "*" and weekday == "*":
        return int(m_match.group(1)) * 60

    # Каждые N часов: "0 */N * * *"
    h_match = step_re.match(hour)
    if minute == "0" and h_match and day == "*" and month == "*" and weekday == "*":
        return int(h_match.group(1)) * 3600

    # Ежедневно: "0 0 * * *" (или "M H * * *")
    if day == "*" and month == "*" and weekday == "*":
        try:
            int(minute)
            int(hour)
            return 86400  # раз в сутки
        except ValueError:
            pass

    logger.warning(f"Complex cron expression '{cron_expr}' not supported, falling back to 4h interval")
    return 14400  # fallback — каждые 4 часа


class CronScheduler:
    """Фоновый планировщик, периодически ставящий задачи синхронизации в очередь."""

    def __init__(self, db: KnowledgeDB):
        """Инициализирует планировщик.

        Args:
            db: Инстанс KnowledgeDB для записи задач в очередь.
        """
        self.db = db
        # Храним строго один активный таймер на каждый репозиторий, предотвращая утечку памяти
        self._active_timers: dict[str, threading.Timer] = {}
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Запускает планировщик, считывая расписание из config.yaml."""
        config = load_config()
        repositories = config.get("repositories", [])

        if not repositories:
            logger.info("No repositories configured, cron scheduler idle")
            return

        self._running = True

        for repo in repositories:
            repo_id = repo.get("id")
            triggers = repo.get("triggers", {})
            cron_expr = triggers.get("cron")

            if not cron_expr:
                logger.debug(f"No cron trigger for repo '{repo_id}', skipping")
                continue

            interval = _parse_cron_interval(cron_expr)
            if interval is None:
                logger.warning(f"Could not parse cron for repo '{repo_id}', skipping")
                continue

            logger.info(f"Scheduling cron for repo '{repo_id}': every {interval}s (expression: {cron_expr})")
            self._schedule_repo(repo_id, interval)

    def _schedule_repo(self, repo_id: str, interval: int) -> None:
        """Создаёт повторяющийся таймер для одного репозитория.

        Args:
            repo_id: Идентификатор репозитория.
            interval: Интервал в секундах.
        """
        def _tick():
            if not self._running:
                return
            try:
                with self._lock:
                    job_id = self.db.add_to_queue(repo_id)
                    logger.info(f"Cron trigger: queued job {job_id} for repo '{repo_id}'")
            except Exception as e:
                logger.error(f"Cron scheduler error for repo '{repo_id}': {e}")
            finally:
                # Перепланируем следующий тик
                if self._running:
                    self._schedule_repo(repo_id, interval)

        with self._lock:
            # Отменяем предыдущий таймер этого репозитория, если он есть
            old_timer = self._active_timers.pop(repo_id, None)
            if old_timer:
                old_timer.cancel()

            timer = threading.Timer(interval, _tick)
            timer.daemon = True
            timer.start()
            self._active_timers[repo_id] = timer

    @property
    def _timers(self) -> list[threading.Timer]:
        """Обратная совместимость со старым свойством _timers."""
        with self._lock:
            return list(self._active_timers.values())

    def stop(self) -> None:
        """Останавливает все таймеры планировщика и очищает хранилище."""
        self._running = False
        with self._lock:
            for timer in self._active_timers.values():
                timer.cancel()
            self._active_timers.clear()
        logger.info("Cron scheduler stopped")
