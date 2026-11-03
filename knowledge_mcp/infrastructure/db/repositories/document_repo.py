"""Репозиторий documents (инфраструктурный слой доступа к данным)."""

import hashlib
import json
import logging
import sqlite3
import struct
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None

RRF_K = 60

from knowledge_mcp.infrastructure.db.repositories.base import BaseSqliteRepository

logger = logging.getLogger("knowledge_mcp.db.documents")


class SqliteDocumentRepository(BaseSqliteRepository):
    """Специализированный репозиторий SQLite для documents."""

    def get_cas_stats(self, project_id: Optional[str] = None, force_refresh: bool = False) -> dict:
        """Возвращает статистику CAS-дедупликации с кэшированием (TTL 60 сек):
        - logical_files_count: количество логических файлов в files
        - unique_contents_count: количество уникальных физических контентов в file_contents
        - virtual_size_bytes: виртуальный размер (сумма размеров всех логических файлов)
        - real_size_bytes: реальный физический размер контента в байтах
        - saved_size_bytes: сэкономленный размер на диске в байтах
        - saved_percent: процент сэкономленного дискового пространства
        """
        key = project_id or "__all__"
        now = time.time()
        with self.lock:
            if not force_refresh and key in self._cas_cache and (now - self._cas_cache_time.get(key, 0) < 60.0):
                return dict(self._cas_cache[key])

            cursor = self.conn.cursor()
            if project_id:
                cursor.execute("""
                    SELECT COUNT(f.id) as files_count,
                           COALESCE(SUM(COALESCE(fc.size, 0)), 0) as virtual_size
                    FROM files f
                    LEFT JOIN file_contents fc INDEXED BY idx_file_contents_hash_size ON f.content_hash = fc.hash
                    WHERE f.project_id = ?
                """, (project_id,))
                row_files = cursor.fetchone()
                logical_files = row_files['files_count']
                virtual_size = row_files['virtual_size']

                cursor.execute("""
                    SELECT COUNT(DISTINCT fc.hash) as contents_count,
                           COALESCE(SUM(fc.size), 0) as real_size
                    FROM file_contents fc INDEXED BY idx_file_contents_hash_size
                    WHERE fc.hash IN (
                        SELECT content_hash FROM files WHERE project_id = ? AND content_hash IS NOT NULL
                    )
                """, (project_id,))
                row_contents = cursor.fetchone()
                unique_contents = row_contents['contents_count']
                real_size = row_contents['real_size']
            else:
                cursor.execute("""
                    SELECT COUNT(f.id) as files_count,
                           COALESCE(SUM(COALESCE(fc.size, 0)), 0) as virtual_size
                    FROM files f
                    LEFT JOIN file_contents fc INDEXED BY idx_file_contents_hash_size ON f.content_hash = fc.hash
                """)
                row_files = cursor.fetchone()
                logical_files = row_files['files_count']
                virtual_size = row_files['virtual_size']

                cursor.execute("""
                    SELECT COUNT(DISTINCT fc.hash) as contents_count,
                           COALESCE(SUM(fc.size), 0) as real_size
                    FROM file_contents fc INDEXED BY idx_file_contents_hash_size
                    WHERE fc.hash IN (
                        SELECT content_hash FROM files WHERE content_hash IS NOT NULL
                    )
                """)
                row_contents = cursor.fetchone()
                unique_contents = row_contents['contents_count']
                real_size = row_contents['real_size']

            saved_bytes = max(0, virtual_size - real_size)
            saved_percent = round((saved_bytes / virtual_size * 100.0), 2) if virtual_size > 0 else 0.0

            res = {
                "logical_files_count": logical_files,
                "files_count": logical_files,
                "unique_contents_count": unique_contents,
                "physical_contents_count": unique_contents,
                "virtual_size_bytes": virtual_size,
                "real_size_bytes": real_size,
                "physical_size_bytes": real_size,
                "saved_size_bytes": saved_bytes,
                "saved_bytes": saved_bytes,
                "saved_percent": saved_percent,
                "savings_percent": saved_percent,
            }
            self._cas_cache[key] = res
            self._cas_cache_time[key] = now
            return dict(res)


    def get_all_repo_stats(self, force_refresh: bool = False) -> Dict[tuple, dict]:
        """Возвращает агрегированные метрики для всех репозиториев за один быстрый проход.
        
        Результат кэшируется в оперативной памяти (TTL = 60с), гарантируя время ответа < 1 мс.
        Ключ словаря: (repo_id, project_id).
        """
        now = time.time()
        with self.lock:
            if not force_refresh and self._stats_cache and (now - self._stats_cache_time < 60.0):
                return dict(self._stats_cache)

            cursor = self.conn.cursor()
            
            # 1. Быстрый подсчет файлов по всем репозиториям и проектам
            cursor.execute("""
                SELECT project_id, repo_id, COUNT(*) as cnt
                FROM files
                GROUP BY project_id, repo_id
            """)
            files_map = {(row['repo_id'], row['project_id']): row['cnt'] for row in cursor.fetchall()}

            # 2. Быстрый подсчет чанков через сканирование по первичному ключу files
            cursor.execute("""
                SELECT f.project_id, f.repo_id, COUNT(*) as cnt
                FROM files f INDEXED BY sqlite_autoindex_files_1
                JOIN chunks c ON c.file_id = f.id
                GROUP BY f.project_id, f.repo_id
            """)
            chunks_map = {(row['repo_id'], row['project_id']): row['cnt'] for row in cursor.fetchall()}

            # 3. Быстрая проверка векторов
            vectors_map = {}
            try:
                cursor.execute("SELECT COUNT(*) FROM chunks_vec")
                total_vecs = cursor.fetchone()[0]
                if total_vecs > 0:
                    cursor.execute("""
                        SELECT f.project_id, f.repo_id, COUNT(*) as cnt
                        FROM files f INDEXED BY sqlite_autoindex_files_1
                        JOIN chunks c ON c.file_id = f.id
                        JOIN chunks_vec cv ON cv.rowid = c.rowid
                        GROUP BY f.project_id, f.repo_id
                    """)
                    vectors_map = {(row['repo_id'], row['project_id']): row['cnt'] for row in cursor.fetchall()}
            except Exception:
                pass

            all_keys = set(files_map.keys()) | set(chunks_map.keys())
            res = {}
            for r_id, p_id in all_keys:
                res[(r_id, p_id)] = {
                    "files_count": files_map.get((r_id, p_id), 0),
                    "chunks_count": chunks_map.get((r_id, p_id), 0),
                    "vectors_count": vectors_map.get((r_id, p_id), 0),
                }

            self._stats_cache = res
            self._stats_cache_time = now
            return dict(res)


    def delete_file(
        self,
        file_id_or_repo: int | str,
        path: Optional[str] = None,
        project_id: str = "default",
        branch: str = "master"
    ) -> bool:
        """Удаляет файл по id или по (repo_id, path, project_id, branch)."""
        cursor = self.conn.cursor()
        if isinstance(file_id_or_repo, int):
            cursor.execute("DELETE FROM files WHERE id = ?", (file_id_or_repo,))
        else:
            if path is not None:
                cursor.execute("""
                    DELETE FROM files
                    WHERE project_id = ? AND repo_id = ? AND branch = ? AND path = ?
                """, (project_id, file_id_or_repo, branch, path))
            else:
                cursor.execute("DELETE FROM files WHERE id = ?", (int(file_id_or_repo),))
        deleted = cursor.rowcount > 0
        self._auto_commit()
        if deleted:
            self.invalidate_stats_cache()
        return deleted


    def get_file_content(
        self,
        content_hash_or_path: str,
        repo_id: Optional[str] = None,
        project_id: str = "default",
        branch: str = "master"
    ) -> Optional[dict | str]:
        """Возвращает контент файла.
        
        Если передан content_hash (или запись найдена в file_contents),
        возвращает dict: {'hash', 'content', 'size', 'language', 'updated_at'}.
        
        Если передан path и repo_id (для обратной совместимости с MCP server),
        возвращает контент в виде строки str.
        """
        cursor = self.conn.cursor()
        
        # 1. Если явно передан repo_id, то ищем файл по логическому пути в files
        if repo_id is not None:
            cursor.execute("""
                SELECT f.id, f.content_hash, fc.content
                FROM files f
                LEFT JOIN file_contents fc ON f.content_hash = fc.hash
                WHERE f.project_id = ? AND f.repo_id = ? AND f.branch = ? AND f.path = ?
            """, (project_id, repo_id, branch, content_hash_or_path))
            file_row = cursor.fetchone()
            if not file_row:
                cursor.execute("""
                    SELECT f.id, f.content_hash, fc.content
                    FROM files f
                    LEFT JOIN file_contents fc ON f.content_hash = fc.hash
                    WHERE f.repo_id = ? AND f.path = ?
                    ORDER BY f.id DESC LIMIT 1
                """, (repo_id, content_hash_or_path))
                file_row = cursor.fetchone()
                
            if file_row:
                if file_row['content'] is not None and file_row['content'] != '':
                    return file_row['content']
                cursor.execute("""
                    SELECT content FROM chunks
                    WHERE file_id = ?
                    ORDER BY line_start ASC
                """, (file_row['id'],))
                rows = cursor.fetchall()
                if rows:
                    return "\n".join(r['content'] for r in rows)
            return None

        # 2. Проверяем, существует ли запись в file_contents по hash
        cursor.execute("""
            SELECT hash, content, size, language, updated_at
            FROM file_contents
            WHERE hash = ?
        """, (content_hash_or_path,))
        fc_row = cursor.fetchone()
        if fc_row and fc_row['content'] != '':
            return dict(fc_row)

        # 3. Если в file_contents не найдено (или контент пуст), проверяем путь в files
        cursor.execute("""
            SELECT f.id, f.content_hash, fc.content
            FROM files f
            LEFT JOIN file_contents fc ON f.content_hash = fc.hash
            WHERE f.path = ?
            ORDER BY f.id DESC LIMIT 1
        """, (content_hash_or_path,))
        file_row = cursor.fetchone()
        if file_row:
            if file_row['content'] is not None and file_row['content'] != '':
                return file_row['content']
            cursor.execute("""
                SELECT content FROM chunks
                WHERE file_id = ?
                ORDER BY line_start ASC
            """, (file_row['id'],))
            rows = cursor.fetchall()
            if rows:
                return "\n".join(r['content'] for r in rows)

        if fc_row:
            return dict(fc_row)

        return None


    def get_known_files(
        self,
        repo_id: str,
        project_id: str = "default",
        branch: str = "master"
    ) -> Dict[str, sqlite3.Row]:
        """Возвращает словарь известных файлов репозитория: path -> Row(id, mtime, hash, content_hash, chunks_count, symbols_count)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                f.id, f.project_id, f.repo_id, f.branch, f.path, f.mtime, f.hash, f.content_hash,
                EXISTS(SELECT 1 FROM chunks c WHERE c.file_id = f.id OR (c.content_hash IS NOT NULL AND c.content_hash = f.content_hash)) AS chunks_count,
                EXISTS(SELECT 1 FROM symbols s WHERE s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash)) AS symbols_count
            FROM files f
            WHERE f.project_id = ? AND f.repo_id = ? AND f.branch = ?
        """, (project_id, repo_id, branch))
        return {row['path']: row for row in cursor.fetchall()}


    def has_indexed_files(self, project_id: str, repo_id: str, branch: str) -> bool:
        """Проверяет, есть ли проиндексированные файлы для указанного проекта, репозитория и ветки."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 1 FROM files
            WHERE project_id = ? AND repo_id = ? AND branch = ?
            LIMIT 1
        """, (project_id, repo_id, branch))
        return cursor.fetchone() is not None


    def get_file(
        self,
        repo_id: str,
        path: str,
        project_id: str = "default",
        branch: str = "master"
    ) -> Optional[sqlite3.Row]:
        """Возвращает запись файла по repo_id, path, project_id и branch."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM files
            WHERE project_id = ? AND repo_id = ? AND branch = ? AND path = ?
        """, (project_id, repo_id, branch, path))
        return cursor.fetchone()


    def clean_unreferenced_contents(self) -> int:
        """Очищает записи в file_contents, на которые больше нет ссылок ни из одного файла (CAS GC)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM file_contents
            WHERE hash NOT IN (SELECT DISTINCT content_hash FROM files WHERE content_hash IS NOT NULL)
        """)
        count = cursor.rowcount
        self._auto_commit()
        return count


    def upsert_file(
        self,
        repo_id: str,
        path: str,
        mtime: float,
        file_hash: str,
        project_id: str = "default",
        branch: str = "master",
        content_hash: Optional[str] = None
    ) -> int:
        """Обновляет или вставляет запись о файле с поддержкой project_id, branch и content_hash. Возвращает file_id."""
        if content_hash is None:
            content_hash = file_hash

        cursor = self.conn.cursor()
        if content_hash:
            now = time.time()
            cursor.execute("""
                INSERT OR IGNORE INTO file_contents (hash, content, size, language, updated_at)
                VALUES (?, '', 0, NULL, ?)
            """, (content_hash, now))

        cursor.execute("""
            INSERT INTO files (project_id, repo_id, branch, path, mtime, hash, content_hash, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
            ON CONFLICT(project_id, repo_id, branch, path) DO UPDATE SET
                mtime = excluded.mtime,
                hash = excluded.hash,
                content_hash = excluded.content_hash,
                updated_at = excluded.updated_at
        """, (project_id, repo_id, branch, path, mtime, file_hash, content_hash))
        self._auto_commit()

        cursor.execute("""
            SELECT id FROM files
            WHERE project_id = ? AND repo_id = ? AND branch = ? AND path = ?
        """, (project_id, repo_id, branch, path))
        return cursor.fetchone()['id']


    def delete_repo(self, repo_id: str, project_id: Optional[str] = None) -> int:
        """Удаляет все файлы и чанки (каскадно), связанные с репозиторием."""
        cursor = self.conn.cursor()
        if project_id:
            cursor.execute("DELETE FROM files WHERE repo_id = ? AND project_id = ?", (repo_id, project_id))
        else:
            cursor.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))
        deleted_count = cursor.rowcount
        self._auto_commit()
        if deleted_count > 0:
            self.invalidate_stats_cache()
        return deleted_count


    def get_repo_stats(
        self,
        repo_id: str,
        project_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> dict:
        """Возвращает метрики проиндексированных данных для конкретного репозитория."""
        # Для стандартного опроса без ветки используем быстрый пакетный кэш
        if branch is None:
            all_stats = self.get_all_repo_stats()
            key = (repo_id, project_id)
            if key in all_stats:
                return all_stats[key]
            if project_id is None and (repo_id, "default") in all_stats:
                return all_stats[(repo_id, "default")]
            if project_id == "default" and (repo_id, None) in all_stats:
                return all_stats[(repo_id, None)]

        cursor = self.conn.cursor()
        where_file = ["repo_id = ?"]
        params: list[Any] = [repo_id]
        if project_id:
            where_file.append("project_id = ?")
            params.append(project_id)
        if branch:
            where_file.append("branch = ?")
            params.append(branch)
        where_sql = " AND ".join(where_file)

        # Кол-во файлов
        cursor.execute(f"SELECT COUNT(*) as cnt FROM files WHERE {where_sql}", params)
        files_count = cursor.fetchone()['cnt']
        
        # Кол-во чанков
        cursor.execute(f"""
            SELECT COUNT(c.id) as cnt FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE {where_sql.replace('repo_id', 'f.repo_id').replace('project_id', 'f.project_id').replace('branch', 'f.branch')}
        """, params)
        chunks_count = cursor.fetchone()['cnt']
        
        # Кол-во векторов
        cursor.execute(f"""
            SELECT COUNT(cv.rowid) as cnt FROM chunks_vec cv
            JOIN chunks c ON cv.rowid = c.rowid
            JOIN files f ON c.file_id = f.id
            WHERE {where_sql.replace('repo_id', 'f.repo_id').replace('project_id', 'f.project_id').replace('branch', 'f.branch')}
        """, params)
        vectors_count = cursor.fetchone()['cnt']
        
        return {
            "files_count": files_count,
            "chunks_count": chunks_count,
            "vectors_count": vectors_count
        }


    def invalidate_stats_cache(self):
        """Сбрасывает in-memory кэш метрик репозиториев и CAS-статистики."""
        with self.lock:
            self._stats_cache_time = 0.0
            self._stats_cache.clear()
            self._cas_cache.clear()
            self._cas_cache_time.clear()


    def upsert_file_content(self, content: str, language: Optional[str] = None) -> str:
        """Сохраняет или обновляет сырой контент файла в CAS таблице file_contents.
        
        Возвращает SHA-256 хэш содержимого.
        """
        content_bytes = content.encode('utf-8')
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        size = len(content_bytes)
        now = time.time()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO file_contents (hash, content, size, language, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(hash) DO UPDATE SET
                content = excluded.content,
                size = excluded.size,
                language = COALESCE(excluded.language, file_contents.language),
                updated_at = excluded.updated_at
        """, (content_hash, content, size, language, now))
        self._auto_commit()
        return content_hash


