"""Пакет версионирования и миграций схемы SQLite."""

from .schema_migrator import apply_schema_migrations

__all__ = ["apply_schema_migrations"]
