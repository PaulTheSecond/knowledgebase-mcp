"""Менеджер многопоточных соединений SQLite с поддержкой WAL и явных транзакций."""

import logging
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

from knowledge_mcp.db.connection import get_db_connection

logger = logging.getLogger("knowledge_mcp.db.pool")


class SqliteConnectionManager:
    """Управляет жизненным циклом и маршрутизацией соединений SQLite между потоками."""

    def __init__(self, db_path: Path, read_only: bool = False):
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.lock = threading.RLock()
        self._local = threading.local()
        self._writer_thread_id = threading.get_ident()
        self._open_conns: List[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._writer_conn = self._create_conn(read_only=read_only)
        self._in_transaction = False

    def _create_conn(self, read_only: bool) -> sqlite3.Connection:
        c = get_db_connection(self.db_path, read_only=read_only)
        with self._conns_lock:
            self._open_conns.append(c)
        return c

    def register_writer_thread(self, thread_id: Optional[int] = None) -> None:
        """Регистрирует поток как единственный поток-писатель для пишущего соединения."""
        self._writer_thread_id = thread_id or threading.get_ident()

    @property
    def writer_conn(self) -> sqlite3.Connection:
        return self._writer_conn

    @property
    def in_transaction(self) -> bool:
        return self._in_transaction

    @in_transaction.setter
    def in_transaction(self, val: bool) -> None:
        self._in_transaction = val

    @property
    def conn(self) -> sqlite3.Connection:
        """Возвращает безопасное соединение с учетом роли потока и транзакций."""
        if self.read_only:
            if not hasattr(self._local, "conn") or self._local.conn is None:
                self._local.conn = self._create_conn(read_only=True)
            return self._local.conn

        curr_id = threading.get_ident()
        if curr_id == self._writer_thread_id:
            return self._writer_conn

        if getattr(self.lock, "_is_owned", lambda: False)() or self._in_transaction:
            return self._writer_conn

        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._create_conn(read_only=True)
        return self._local.conn

    def close(self) -> None:
        """Закрывает все открытые соединения с БД."""
        with self._conns_lock:
            conns = list(self._open_conns)
            self._open_conns.clear()
        for c in conns:
            try:
                c.close()
            except Exception:
                pass
        if hasattr(self, "_writer_conn") and self._writer_conn:
            try:
                self._writer_conn.close()
            except Exception:
                pass
        if hasattr(self, "_local") and hasattr(self._local, "conn") and self._local.conn:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    def vacuum(self) -> None:
        """Выполняет VACUUM для дефрагментации файла БД."""
        with self.lock:
            self.conn.execute("VACUUM;")

    def optimize(self) -> None:
        """Запускает PRAGMA optimize для актуализации статистики планировщика запросов."""
        with self.lock:
            self.conn.execute("PRAGMA optimize;")

    def get_vacuum_mode(self) -> int:
        """Возвращает текущий режим auto_vacuum (0=NONE, 1=FULL, 2=INCREMENTAL)."""
        cursor = self.conn.cursor()
        return cursor.execute("PRAGMA auto_vacuum;").fetchone()[0]

    def begin_transaction(self) -> None:
        """Начинает явную транзакцию для пакетных операций."""
        self._writer_conn.execute("BEGIN")
        self._in_transaction = True

    def commit_transaction(self) -> None:
        """Фиксирует текущую транзакцию."""
        if self._in_transaction:
            self._writer_conn.commit()
            self._in_transaction = False

    def rollback_transaction(self) -> None:
        """Откатывает текущую транзакцию при ошибке."""
        if self._in_transaction:
            try:
                self._writer_conn.rollback()
            except sqlite3.OperationalError:
                pass
            finally:
                self._in_transaction = False

    def auto_commit(self) -> None:
        """Вызывает commit() только если мы НЕ внутри явной транзакции."""
        if not self._in_transaction:
            with self.lock:
                self._writer_conn.commit()
