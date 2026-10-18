import sqlite3
import logging
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
        # Для Read-Only режима используем URI
        uri_path = f"{path_obj.as_uri()}?mode=ro"
        logger.info(f"Opening database in Read-Only mode: {uri_path}")
        conn = sqlite3.connect(uri_path, uri=True, timeout=timeout, check_same_thread=False)
    else:
        # Режим записи
        logger.info(f"Opening database in Write mode: {path_obj}")
        conn = sqlite3.connect(str(path_obj), timeout=timeout, check_same_thread=False)

    conn.row_factory = sqlite3.Row
    
    # Всегда включаем внешние ключи
    conn.execute("PRAGMA foreign_keys = ON;")
    
    # Режим WAL, synchronous = NORMAL и auto_vacuum = FULL для пишущих соединений
    if not read_only:
        try:
            # Настройка auto_vacuum (FULL) для новых баз данных
            conn.execute("PRAGMA auto_vacuum = FULL;")
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
