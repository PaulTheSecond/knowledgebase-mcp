"""Базовый класс инфраструктурных репозиториев SQLite."""

from typing import Any


class BaseSqliteRepository:
    """Базовый репозиторий для выполнения операций с SQLite через KnowledgeDB."""

    def __init__(self, db: Any):
        self.db = db

    @property
    def conn(self):
        return self.db.conn

    @property
    def writer_conn(self):
        return self.db._writer_conn

    @property
    def lock(self):
        return self.db.lock

    def auto_commit(self):
        self.db._auto_commit()

    def _auto_commit(self):
        self.db._auto_commit()

    @property
    def vec_dim(self):
        return getattr(self.db, "vec_dim", 384)

    @property
    def embedder(self):
        return getattr(self.db, "embedder", None)

    @property
    def _stats_cache(self):
        return self.db._stats_cache

    @property
    def _stats_cache_time(self):
        return self.db._stats_cache_time

    @_stats_cache_time.setter
    def _stats_cache_time(self, val):
        self.db._stats_cache_time = val

    @property
    def _cas_cache(self):
        return self.db._cas_cache

    @property
    def _cas_cache_time(self):
        return self.db._cas_cache_time

    def __getattr__(self, name: str) -> Any:
        """Перенаправляет отсутствующие методы и атрибуты на фасад KnowledgeDB."""
        return getattr(self.db, name)
