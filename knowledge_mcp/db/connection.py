import sqlite3
import logging
import time
import os
from pathlib import Path
import sqlite_vec

logger = logging.getLogger(__name__)

def get_db_connection(db_path: str | Path, read_only: bool = False, timeout: float = 30.0) -> sqlite3.Connection:
    """Создает и настраивает соединение с базой данных SQLite.

    Включает режим WAL, проверяет внешние ключи (foreign keys) и загружает расширение sqlite-vec.

    Args:
        db_path: Путь к файлу базы данных SQLite на диске.
        read_only: Если True, открывает соединение в режиме только для чтения.
        timeout: Время ожидания разблокировки БД в секундах.

    Returns:
        Объект sqlite3.Connection.
    """
    path_obj = Path(db_path).absolute()
    
    if read_only:
        # Для Read-Only режима используем URI с mode=ro.
        # Внимание: immutable=1 не используется, так как он отключает отслеживание изменений WAL
        # и вызывает рассинхронизацию кэша страниц и ошибку "database disk image is malformed".
        uri_path = f"{path_obj.as_uri()}?mode=ro"
        logger.info(f"Opening database in Read-Only mode: {uri_path}")
        max_retries = 5
        conn = None
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(uri_path, uri=True, timeout=timeout, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                try:
                    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")
                    conn.execute("PRAGMA temp_store = MEMORY;")
                    conn.execute("PRAGMA cache_size = -64000;")
                except sqlite3.OperationalError:
                    pass
                break
            except sqlite3.OperationalError as e:
                err_str = str(e).lower()
                if ("disk i/o error" in err_str or "database is locked" in err_str or "unable to open" in err_str) and attempt < max_retries - 1:
                    wait_sec = (attempt + 1) * 0.5
                    logger.warning(f"Read-only connection attempt {attempt+1}/{max_retries} failed ({e}). Retrying in {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    raise
    else:
        # Режим записи с автоповторами при конфликтах блокировок 9p/drvfs
        logger.info(f"Opening database in Write mode: {path_obj}")
        max_retries = 5
        conn = None
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(str(path_obj), timeout=timeout, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                try:
                    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")
                    conn.execute("PRAGMA temp_store = MEMORY;")
                    conn.execute("PRAGMA cache_size = -64000;")
                except sqlite3.OperationalError:
                    pass
                break
            except sqlite3.OperationalError as e:
                err_str = str(e).lower()
                if ("disk i/o error" in err_str or "database is locked" in err_str) and attempt < max_retries - 1:
                    wait_sec = (attempt + 1) * 0.5
                    logger.warning(f"Write connection attempt {attempt+1}/{max_retries} failed ({e}). Retrying in {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    raise
    
    # Режим WAL, synchronous = NORMAL и auto_vacuum = FULL для пишущих соединений
    if not read_only:
        try:
            # Настройка auto_vacuum (INCREMENTAL) для безопасной работы поверх DrvFs/9P
            conn.execute("PRAGMA auto_vacuum = INCREMENTAL;")
            # Настройка режима WAL
            journal_mode = conn.execute("PRAGMA journal_mode = WAL;").fetchone()[0]
            conn.execute("PRAGMA synchronous = NORMAL;")
            logger.info(f"Database journal mode set to: {journal_mode}")
        except sqlite3.Error as e:
            logger.warning(f"Failed to configure WAL mode or auto_vacuum: {e}")

    # Загрузка sqlite-vec
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as e:
        logger.error(f"Failed to load sqlite-vec extension: {e}")
        raise

    return conn
