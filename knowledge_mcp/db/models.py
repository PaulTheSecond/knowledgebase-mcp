"""Модели данных и доступ к базе знаний KnowledgeDB.

Реализует паттерн Фасад (Facade) над специализированными репозиториями:
- SqliteDocumentRepository: документы, файлы и контентно-адресуемое хранилище (CAS);
- SqliteSymbolRepository: символы кода, ребра графа зависимостей и blast radius;
- SqliteVectorRepository: чанки, FTS5 полнотекстовый и sqlite-vec векторный поиск;
- SqliteJobQueueRepository: очередь задач индексации демона, телеметрия и настройки.
"""

import hashlib
import json
import logging
import sqlite3
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from knowledge_mcp.db.connection import get_db_connection
from knowledge_mcp.infrastructure.db.connection_pool import SqliteConnectionManager
from knowledge_mcp.infrastructure.db.migrations.schema_migrator import apply_schema_migrations
from knowledge_mcp.infrastructure.db.repositories import (
    SqliteDocumentRepository,
    SqliteJobQueueRepository,
    SqliteSymbolRepository,
    SqliteVectorRepository,
)

logger = logging.getLogger(__name__)


class KnowledgeDB:
    """Центральный фасад доступа к локальной базе знаний."""

    def __init__(self, db_path: str | Path, read_only: bool = False):
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.lock = threading.RLock()

        if read_only and not self.db_path.exists():
            logger.info(f"Database file does not exist. Initializing schema at {self.db_path} before opening read-only.")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            temp_conn = get_db_connection(self.db_path, read_only=False)
            self.conn = temp_conn
            self.vec_dim = 384
            self._apply_migrations()
            temp_conn.close()

        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._writer_thread_id = threading.get_ident()
        self._open_conns = []
        self._conns_lock = threading.Lock()
        self._writer_conn = self._create_conn(read_only=read_only)
        self.vec_dim = 384
        self._in_transaction = False
        self._stats_cache: Dict[tuple, dict] = {}
        self._stats_cache_time: float = 0.0
        self._cas_cache: Dict[str, dict] = {}
        self._cas_cache_time: Dict[str, float] = {}

        # Инициализация специализированных репозиториев
        self._documents = SqliteDocumentRepository(self)
        self._symbols = SqliteSymbolRepository(self)
        self._vectors = SqliteVectorRepository(self)
        self._jobs = SqliteJobQueueRepository(self)

        if not read_only:
            self._apply_migrations()

    @property
    def documents(self) -> SqliteDocumentRepository:
        """Репозиторий документов и CAS-контента."""
        return self._documents

    @property
    def symbols(self) -> SqliteSymbolRepository:
        """Репозиторий графа символов и вызовов."""
        return self._symbols

    @property
    def vectors(self) -> SqliteVectorRepository:
        """Репозиторий векторного и полнотекстового поиска."""
        return self._vectors

    @property
    def jobs(self) -> SqliteJobQueueRepository:
        """Репозиторий очередей фоновых задач."""
        return self._jobs

    def _create_conn(self, read_only: bool) -> sqlite3.Connection:
        c = get_db_connection(self.db_path, read_only=read_only)
        with self._conns_lock:
            self._open_conns.append(c)
        return c

    def register_writer_thread(self, thread_id: Optional[int] = None):
        """Регистрирует поток как единственный поток-писатель для пишущего соединения."""
        self._writer_thread_id = thread_id or threading.get_ident()

    @property
    def conn(self) -> sqlite3.Connection:
        if self.read_only:
            if not hasattr(self._local, "conn") or self._local.conn is None:
                self._local.conn = self._create_conn(read_only=True)
            return self._local.conn

        curr_id = threading.get_ident()
        if curr_id == self._writer_thread_id:
            return self._writer_conn

        if getattr(self.lock, "_is_owned", lambda: False)() or getattr(self, "_in_transaction", False):
            return self._writer_conn

        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._create_conn(read_only=True)
        return self._local.conn

    def _apply_migrations(self):
        """Миграции схемы БД через infrastructure.db.migrations."""
        apply_schema_migrations(self.conn, self.vec_dim)

    def close(self):
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

    def vacuum(self):
        with self.lock:
            self.conn.execute("VACUUM;")

    def optimize(self):
        with self.lock:
            self.conn.execute("PRAGMA optimize;")

    def get_vacuum_mode(self) -> int:
        cursor = self.conn.cursor()
        return cursor.execute("PRAGMA auto_vacuum;").fetchone()[0]

    def begin_transaction(self):
        self._writer_conn.execute("BEGIN")
        self._in_transaction = True

    def commit_transaction(self):
        if getattr(self, '_in_transaction', False):
            self._writer_conn.commit()
            self._in_transaction = False

    def rollback_transaction(self):
        if getattr(self, '_in_transaction', False):
            try:
                self._writer_conn.rollback()
            except sqlite3.OperationalError:
                pass
            finally:
                self._in_transaction = False

    def _auto_commit(self):
        if not self._in_transaction:
            with self.lock:
                self._writer_conn.commit()

    # ──────────────────────────────────────────────────────────────────────────
    # Делегирование в DocumentRepository
    # ──────────────────────────────────────────────────────────────────────────

    def get_known_files(self, *args, **kwargs):
        return self._documents.get_known_files(*args, **kwargs)

    def get_file(self, *args, **kwargs):
        return self._documents.get_file(*args, **kwargs)

    def has_indexed_files(self, *args, **kwargs):
        return self._documents.has_indexed_files(*args, **kwargs)

    def upsert_file(self, *args, **kwargs):
        return self._documents.upsert_file(*args, **kwargs)

    def delete_file(self, *args, **kwargs):
        return self._documents.delete_file(*args, **kwargs)

    def delete_repo(self, *args, **kwargs):
        return self._documents.delete_repo(*args, **kwargs)

    def invalidate_stats_cache(self, *args, **kwargs):
        return self._documents.invalidate_stats_cache(*args, **kwargs)

    def get_all_repo_stats(self, *args, **kwargs):
        return self._documents.get_all_repo_stats(*args, **kwargs)

    def get_repo_stats(self, *args, **kwargs):
        return self._documents.get_repo_stats(*args, **kwargs)

    def upsert_file_content(self, *args, **kwargs):
        return self._documents.upsert_file_content(*args, **kwargs)

    def get_file_content(self, *args, **kwargs):
        return self._documents.get_file_content(*args, **kwargs)

    def get_cas_stats(self, *args, **kwargs):
        return self._documents.get_cas_stats(*args, **kwargs)

    def clean_unreferenced_contents(self, *args, **kwargs):
        return self._documents.clean_unreferenced_contents(*args, **kwargs)

    # ──────────────────────────────────────────────────────────────────────────
    # Делегирование в SymbolRepository
    # ──────────────────────────────────────────────────────────────────────────

    def clear_file_symbols(self, *args, **kwargs):
        return self._symbols.clear_file_symbols(*args, **kwargs)

    def add_symbol(self, *args, **kwargs):
        return self._symbols.add_symbol(*args, **kwargs)

    def add_symbols_batch(self, *args, **kwargs):
        return self._symbols.add_symbols_batch(*args, **kwargs)

    def add_symbol_edge(self, *args, **kwargs):
        return self._symbols.add_symbol_edge(*args, **kwargs)

    def add_symbol_edges_batch(self, *args, **kwargs):
        return self._symbols.add_symbol_edges_batch(*args, **kwargs)

    def add_unresolved_ref(self, *args, **kwargs):
        return self._symbols.add_unresolved_ref(*args, **kwargs)

    def add_unresolved_refs_batch(self, *args, **kwargs):
        return self._symbols.add_unresolved_refs_batch(*args, **kwargs)

    def resolve_pending_references(self, *args, **kwargs):
        return self._symbols.resolve_pending_references(*args, **kwargs)

    def get_symbols(self, *args, **kwargs):
        return self._symbols.get_symbols(*args, **kwargs)

    def find_symbols(self, *args, **kwargs):
        return self._symbols.find_symbols(*args, **kwargs)

    def get_symbol_callers(self, *args, **kwargs):
        return self._symbols.get_symbol_callers(*args, **kwargs)

    def get_callers(self, *args, **kwargs):
        return self._symbols.get_callers(*args, **kwargs)

    def get_symbol_callees(self, *args, **kwargs):
        return self._symbols.get_symbol_callees(*args, **kwargs)

    def get_callees(self, *args, **kwargs):
        return self._symbols.get_callees(*args, **kwargs)

    def get_hierarchy(self, *args, **kwargs):
        return self._symbols.get_hierarchy(*args, **kwargs)

    def get_impact_analysis(self, *args, **kwargs):
        return self._symbols.get_impact_analysis(*args, **kwargs)

    # ──────────────────────────────────────────────────────────────────────────
    # Делегирование в VectorRepository
    # ──────────────────────────────────────────────────────────────────────────

    def clear_file_chunks(self, *args, **kwargs):
        return self._vectors.clear_file_chunks(*args, **kwargs)

    def add_chunks_batch(self, *args, **kwargs):
        return self._vectors.add_chunks_batch(*args, **kwargs)

    def add_chunk(self, *args, **kwargs):
        return self._vectors.add_chunk(*args, **kwargs)

    def add_embedding(self, *args, **kwargs):
        return self._vectors.add_embedding(*args, **kwargs)

    def add_embeddings_batch(self, *args, **kwargs):
        return self._vectors.add_embeddings_batch(*args, **kwargs)

    def get_chunks_without_embeddings(self, *args, **kwargs):
        return self._vectors.get_chunks_without_embeddings(*args, **kwargs)

    def get_chunk_by_id(self, *args, **kwargs):
        return self._vectors.get_chunk_by_id(*args, **kwargs)

    def search_chunks_fts(self, *args, **kwargs):
        return self._vectors.search_chunks_fts(*args, **kwargs)

    def search_chunks_graph(self, *args, **kwargs):
        return self._vectors.search_chunks_graph(*args, **kwargs)

    def search_chunks_hybrid(self, *args, **kwargs):
        return self._vectors.search_chunks_hybrid(*args, **kwargs)

    def search_chunks(self, *args, **kwargs):
        return self._vectors.search_chunks(*args, **kwargs)

    # ──────────────────────────────────────────────────────────────────────────
    # Делегирование в JobQueueRepository
    # ──────────────────────────────────────────────────────────────────────────

    def get_queue_jobs(self, *args, **kwargs):
        return self._jobs.get_queue_jobs(*args, **kwargs)

    def log_token_savings(self, *args, **kwargs):
        return self._jobs.log_token_savings(*args, **kwargs)

    def get_token_savings_summary(self, *args, **kwargs):
        return self._jobs.get_token_savings_summary(*args, **kwargs)

    def get_system_status(self, *args, **kwargs):
        return self._jobs.get_system_status(*args, **kwargs)

    def update_system_status(self, *args, **kwargs):
        return self._jobs.update_system_status(*args, **kwargs)

    def get_semantic_status(self, *args, **kwargs):
        return self._jobs.get_semantic_status(*args, **kwargs)

    def update_semantic_status(self, *args, **kwargs):
        return self._jobs.update_semantic_status(*args, **kwargs)

    def add_to_queue(self, *args, **kwargs):
        return self._jobs.add_to_queue(*args, **kwargs)

    def fetch_next_job(self, *args, **kwargs):
        return self._jobs.fetch_next_job(*args, **kwargs)

    def update_job_status(self, *args, **kwargs):
        return self._jobs.update_job_status(*args, **kwargs)

    def get_indexed_commit(self, *args, **kwargs):
        return self._jobs.get_indexed_commit(*args, **kwargs)

    def update_indexed_commit(self, *args, **kwargs):
        return self._jobs.update_indexed_commit(*args, **kwargs)

    def reschedule_job(self, *args, **kwargs):
        return self._jobs.reschedule_job(*args, **kwargs)

    def reset_stale_jobs(self, *args, **kwargs):
        return self._jobs.reset_stale_jobs(*args, **kwargs)

    def stop_job(self, *args, **kwargs):
        return self._jobs.stop_job(*args, **kwargs)

    def restart_job(self, *args, **kwargs):
        return self._jobs.restart_job(*args, **kwargs)

    def delete_job(self, *args, **kwargs):
        return self._jobs.delete_job(*args, **kwargs)

    def retry_failed_jobs(self, *args, **kwargs):
        return self._jobs.retry_failed_jobs(*args, **kwargs)

    def get_setting(self, *args, **kwargs):
        return self._jobs.get_setting(*args, **kwargs)

    def set_setting(self, *args, **kwargs):
        return self._jobs.set_setting(*args, **kwargs)

    def get_all_settings(self, *args, **kwargs):
        return self._jobs.get_all_settings(*args, **kwargs)
