"""Репозиторий vectors (инфраструктурный слой доступа к данным)."""

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

logger = logging.getLogger("knowledge_mcp.db.vectors")


class SqliteVectorRepository(BaseSqliteRepository):
    """Специализированный репозиторий SQLite для vectors."""

    def add_embeddings_batch(self, records: List[tuple[int, List[float]]]):
        """Пакетно добавляет векторные представления для группы чанков (существенное ускорение)."""
        if not records:
            return
        cursor = self.conn.cursor()
        
        batch_data = [
            (rowid, sqlite_vec.serialize_float32(vector))
            for rowid, vector in records
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO chunks_vec(rowid, embedding)
            VALUES (?, ?)
        """, batch_data)
        self._auto_commit()
        

    def get_chunk_by_id(self, chunk_id: str) -> Optional[sqlite3.Row]:
        """Возвращает чанк вместе с данными файла по chunk_id. None если не найден."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT c.*, f.repo_id, f.path, f.project_id, f.branch
            FROM chunks c
            LEFT JOIN files f ON (c.file_id = f.id OR (c.content_hash IS NOT NULL AND c.content_hash = f.content_hash))
            WHERE c.id = ?
            LIMIT 1
        ''', (chunk_id,))
        return cursor.fetchone()


    def search_chunks_hybrid(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        repo_ids: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        branch: Optional[str] = None,
        limit: int = 10
    ) -> List[sqlite3.Row]:
        """Гибридный поиск RRF: объединяет FTS5, KNN по векторам (если есть), и Graph Search."""
        cursor = self.conn.cursor()
        
        fts_res = self.search_chunks_fts(query, repo_ids, project_id=project_id, branch=branch, limit=50)
        graph_res = self.search_chunks_graph(query, repo_ids, project_id=project_id, branch=branch, limit=20)
        
        vec_res = []
        if query_vector is not None:
            vec_sql = '''
                SELECT DISTINCT c.id, c.content, c.source_kind, c.trust, c.line_start, c.line_end, 
                       f.repo_id, f.path, f.project_id, f.branch, vec_distance_L2(cv.embedding, ?) as distance
                FROM chunks_vec cv
                JOIN chunks c ON cv.rowid = c.rowid
                JOIN files f ON (c.file_id = f.id OR (c.content_hash IS NOT NULL AND c.content_hash = f.content_hash))
            '''
            params: list[Any] = [sqlite_vec.serialize_float32(query_vector)]
            where_clauses = []
            if repo_ids:
                placeholders = ','.join('?' for _ in repo_ids)
                where_clauses.append(f"f.repo_id IN ({placeholders})")
                params.extend(repo_ids)
            if project_id is not None:
                where_clauses.append("f.project_id = ?")
                params.append(project_id)
            if branch is not None:
                where_clauses.append("f.branch = ?")
                params.append(branch)
            if where_clauses:
                vec_sql += " WHERE " + " AND ".join(where_clauses)
                
            vec_sql += " ORDER BY distance ASC LIMIT 50"
            cursor.execute(vec_sql, params)
            vec_res = cursor.fetchall()
        
        # Reciprocal Rank Fusion алгоритм
        K = RRF_K
        scores = {}
        chunks_map = {}
        
        # FTS5
        for i, row in enumerate(fts_res):
            chunk_id = row['id']
            chunks_map[chunk_id] = row
            scores[chunk_id] = 1.0 / (K + i + 1)
            
        # Vector Embeddings
        for i, row in enumerate(vec_res):
            chunk_id = row['id']
            if chunk_id not in chunks_map:
                chunks_map[chunk_id] = row
            scores[chunk_id] = scores.get(chunk_id, 0) + (1.0 / (K + i + 1))
            
        # Graph (с бустом x2 за точное совпадение имени)
        for i, row in enumerate(graph_res):
            chunk_id = row['id']
            if chunk_id not in chunks_map:
                chunks_map[chunk_id] = row
            scores[chunk_id] = scores.get(chunk_id, 0) + (2.0 / (K + i + 1))
            
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [chunks_map[cid] for cid, _ in sorted_results[:limit]]


    def add_embedding(self, chunk_rowid: int, vector: List[float]):
        """Добавляет векторное представление в виртуальную vec0 таблицу."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO chunks_vec(rowid, embedding)
            VALUES (?, ?)
        """, (chunk_rowid, sqlite_vec.serialize_float32(vector)))
        self._auto_commit()


    def clear_file_chunks(self, file_id: int):
        """Удаляет все чанки, привязанные к заданному файлу (перед реиндексацией 'грязного' файла)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        self._auto_commit()


    def search_chunks_fts(
        self,
        query: str,
        repo_ids: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        branch: Optional[str] = None,
        limit: int = 10
    ) -> List[sqlite3.Row]:
        """Полнотекстовый поиск по чанкам с поддержкой project_id и branch."""
        cursor = self.conn.cursor()
        
        safe_terms = [f'"{term}"' for term in query.replace('"', ' ').split() if term]
        safe_query = ' OR '.join(safe_terms)
        if not safe_query:
            return []

        sql = '''
            SELECT DISTINCT c.id, c.content, c.source_kind, c.trust, c.line_start, c.line_end, 
                   f.repo_id, f.path, f.project_id, f.branch, bm25(chunks_fts) as rank
            FROM chunks_fts fts
            JOIN chunks c ON fts.rowid = c.rowid
            JOIN files f ON (c.file_id = f.id OR (c.content_hash IS NOT NULL AND c.content_hash = f.content_hash))
            WHERE chunks_fts MATCH ?
        '''
        params: list[Any] = [safe_query]
        
        if repo_ids:
            placeholders = ','.join('?' for _ in repo_ids)
            sql += f" AND f.repo_id IN ({placeholders})"
            params.extend(repo_ids)

        if project_id is not None:
            sql += " AND f.project_id = ?"
            params.append(project_id)

        if branch is not None:
            sql += " AND f.branch = ?"
            params.append(branch)
            
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        return cursor.fetchall()


    def get_chunks_without_embeddings(
        self,
        repo_id: str,
        project_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> List[tuple[str, int, str]]:
        """Возвращает список чанков, у которых еще нет векторного представления (recovery после сбоя)."""
        cursor = self.conn.cursor()
        sql = '''
            SELECT DISTINCT c.content, c.rowid, f.path
            FROM chunks c
            JOIN files f ON (c.file_id = f.id OR (c.content_hash IS NOT NULL AND c.content_hash = f.content_hash))
            WHERE f.repo_id = ? AND c.rowid NOT IN (SELECT rowid FROM chunks_vec)
        '''
        params: list[Any] = [repo_id]
        if project_id is not None:
            sql += " AND f.project_id = ?"
            params.append(project_id)
        if branch is not None:
            sql += " AND f.branch = ?"
            params.append(branch)
        cursor.execute(sql, params)
        return [(row[0], row[1], row[2]) for row in cursor.fetchall()]


    def add_chunk(
        self,
        chunk_id: str,
        file_id: int,
        content: str, 
        source_kind: str,
        trust: str, 
        line_start: Optional[int] = None,
        line_end: Optional[int] = None, 
        sha: Optional[str] = None,
        content_hash: Optional[str] = None
    ) -> int:
        """Добавляет чанк в базу и возвращает его sqlite internal rowid (нужно для vec0)."""
        cursor = self.conn.cursor()
        if content_hash is None and file_id:
            cursor.execute("SELECT content_hash FROM files WHERE id = ?", (file_id,))
            row = cursor.fetchone()
            if row and row['content_hash']:
                content_hash = row['content_hash']
        cursor.execute("""
            INSERT INTO chunks (id, file_id, content, line_start, line_end, source_kind, trust, sha, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (chunk_id, file_id, content, line_start, line_end, source_kind, trust, sha, content_hash))
        self._auto_commit()
        return cursor.lastrowid


    def search_chunks_graph(
        self,
        query: str,
        repo_ids: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        branch: Optional[str] = None,
        limit: int = 10
    ) -> List[sqlite3.Row]:
        """Графовый поиск: поиск релевантных кусков кода по именам из запроса."""
        import re
        cursor = self.conn.cursor()
        
        words = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query))
        candidates = [w for w in words if len(w) > 3]
        if not candidates:
            return []
            
        placeholders = ','.join('?' for _ in candidates)
        params: list[Any] = list(candidates)
        
        sql = f'''
            SELECT DISTINCT c.id, c.content, c.source_kind, c.trust, c.line_start, c.line_end, 
                   f.repo_id, f.path, f.project_id, f.branch, 1.0 as rank
            FROM symbols s
            JOIN chunks c ON s.chunk_id = c.id
            JOIN files f ON (c.file_id = f.id OR (c.content_hash IS NOT NULL AND c.content_hash = f.content_hash))
            WHERE s.name IN ({placeholders}) COLLATE NOCASE
        '''
        
        if repo_ids:
            repo_placeholders = ','.join('?' for _ in repo_ids)
            sql += f" AND f.repo_id IN ({repo_placeholders})"
            params.extend(repo_ids)

        if project_id is not None:
            sql += " AND f.project_id = ?"
            params.append(project_id)

        if branch is not None:
            sql += " AND f.branch = ?"
            params.append(branch)
            
        sql += " LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        return cursor.fetchall()


    def search_chunks(
        self,
        query: str,
        repo_ids: Optional[List[str]] = None,
        project_id: Optional[str] = "default",
        branch: Optional[str] = "master",
        query_vector: Optional[List[float]] = None,
        limit: int = 10
    ) -> List[sqlite3.Row]:
        """Универсальный метод поиска чанков с поддержкой project_id и branch."""
        if query_vector is not None:
            return self.search_chunks_hybrid(query, query_vector=query_vector, repo_ids=repo_ids, project_id=project_id, branch=branch, limit=limit)
        return self.search_chunks_fts(query, repo_ids=repo_ids, project_id=project_id, branch=branch, limit=limit)



    def add_chunks_batch(self, records: list[tuple]) -> list[int]:
        """Батчевая вставка чанков.
        records: [(chunk_id, file_id, content, line_start, line_end, source_kind, trust, sha), ...]
        или [(chunk_id, file_id, content, line_start, line_end, source_kind, trust, sha, content_hash), ...]
        Возвращает список rowid вставленных чанков."""
        if not records:
            return []
        cursor = self.conn.cursor()
        rowids = []
        for record in records:
            if len(record) == 8:
                chunk_id, file_id, content, line_start, line_end, source_kind, trust, sha = record
                content_hash = None
                if file_id:
                    cursor.execute("SELECT content_hash FROM files WHERE id = ?", (file_id,))
                    row = cursor.fetchone()
                    if row and row['content_hash']:
                        content_hash = row['content_hash']
                cursor.execute("""
                    INSERT INTO chunks (id, file_id, content, line_start, line_end, source_kind, trust, sha, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (chunk_id, file_id, content, line_start, line_end, source_kind, trust, sha, content_hash))
            elif len(record) == 9:
                cursor.execute("""
                    INSERT INTO chunks (id, file_id, content, line_start, line_end, source_kind, trust, sha, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, record)
            rowids.append(cursor.lastrowid)
        self._auto_commit()
        return rowids


