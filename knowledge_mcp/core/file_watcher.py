"""
Модуль фонового наблюдателя файловой системы (WorkspaceFileWatcher).
Поддерживает Watchdog (Windows ReadDirectoryChangesW) с graceful fallback
на опрос таймштампов, дебаунсинг событий (200-300 мс) и работу в daemon-потоке.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Dict

logger = logging.getLogger(__name__)


class WorkspaceFileWatcher:
    """Фоновый наблюдатель за изменениями файлов в воркспейсах."""

    def __init__(self, debounce_seconds: float = 0.25):
        self.debounce_seconds = debounce_seconds
        self._paths: List[Path] = []
        self._callback: Optional[Callable[[Path], None]] = None
        self._running = False
        self._lock = threading.RLock()
        self._timers: Dict[Path, threading.Timer] = {}
        self._observer = None
        self._poll_thread: Optional[threading.Thread] = None

    def watch_paths(
        self,
        paths: List[Path | str],
        on_changed_callback: Callable[[Path], None]
    ):
        """Регистрирует пути для наблюдения и callback при изменении файлов."""
        with self._lock:
            self._paths = [Path(p).resolve() for p in paths if Path(p).exists()]
            self._callback = on_changed_callback

    def _debounce_notify(self, path: Path):
        """Планирует вызов callback с дебаунсингом (подавление дребезга частых записей)."""
        with self._lock:
            if not self._running or self._callback is None:
                return

            # Игнорируем внутренние системные файлы и папки
            str_path = str(path).replace('\\', '/')
            if any(ign in str_path for ign in ('/.git/', '/__pycache__/', '/.pytest_cache/', '/node_modules/')):
                return
            if path.suffix in ('.tmp', '.swp', '.lock'):
                return

            if path in self._timers:
                self._timers[path].cancel()

            def fire():
                with self._lock:
                    self._timers.pop(path, None)
                try:
                    if self._callback and self._running:
                        self._callback(path)
                except Exception as ex:
                    logger.error(f"Error in file watcher callback for {path}: {ex}")

            timer = threading.Timer(self.debounce_seconds, fire)
            timer.daemon = True
            self._timers[path] = timer
            timer.start()

    def start(self):
        """Запускает наблюдение в daemon-потоке."""
        with self._lock:
            if self._running:
                return
            self._running = True

            try:
                from watchdog.observers import Observer
                from watchdog.events import FileSystemEventHandler

                class Handler(FileSystemEventHandler):
                    def __init__(handler_self, watcher: 'WorkspaceFileWatcher'):
                        super().__init__()
                        handler_self.watcher = watcher

                    def on_any_event(handler_self, event):
                        if event.is_directory:
                            return
                        src_path = Path(event.src_path)
                        handler_self.watcher._debounce_notify(src_path)

                observer = Observer()
                observer.daemon = True
                handler = Handler(self)
                for p in self._paths:
                    if p.is_dir():
                        observer.schedule(handler, str(p), recursive=True)
                    elif p.is_file():
                        observer.schedule(handler, str(p.parent), recursive=False)

                observer.start()
                self._observer = observer
                logger.info(f"WorkspaceFileWatcher started with watchdog for {len(self._paths)} paths.")
            except Exception as e:
                logger.warning(f"Watchdog failed or not available, falling back to polling: {e}")
                self._start_poll_fallback()

    def _start_poll_fallback(self):
        """Fallback: опрос mtime файлов в daemon потоке."""
        def poll_loop():
            mtimes: Dict[Path, float] = {}
            while self._running:
                try:
                    for root in self._paths:
                        if not root.exists():
                            continue
                        files = [root] if root.is_file() else list(root.rglob('*'))
                        for f in files:
                            if not f.is_file():
                                continue
                            try:
                                mt = f.stat().st_mtime
                                if f in mtimes and mtimes[f] != mt:
                                    self._debounce_notify(f)
                                mtimes[f] = mt
                            except OSError:
                                pass
                except Exception as ex:
                    logger.debug(f"Poll loop exception: {ex}")
                time.sleep(1.0)

        self._poll_thread = threading.Thread(target=poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info(f"WorkspaceFileWatcher polling fallback started for {len(self._paths)} paths.")

    def stop(self):
        """Останавливает наблюдение и отменяет все отложенные таймеры."""
        with self._lock:
            self._running = False

            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()

            if self._observer is not None:
                try:
                    self._observer.stop()
                    self._observer.join(timeout=2.0)
                except Exception as ex:
                    logger.debug(f"Error stopping watchdog observer: {ex}")
                self._observer = None

            if self._poll_thread is not None:
                self._poll_thread.join(timeout=2.0)
                self._poll_thread = None

    def is_running(self) -> bool:
        """Возвращает статус работы наблюдателя."""
        with self._lock:
            return self._running
