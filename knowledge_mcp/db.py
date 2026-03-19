import sqlite3
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class KnowledgeDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self):
        """Инициализация таблиц для дельта-индексации и FTS5."""
        cursor = self.conn.cursor()
        
        # Таблица отслеживания файлов для дельта-индексации
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT NOT NULL,
                path TEXT NOT NULL,
                mtime REAL NOT NULL,
                hash TEXT NOT NULL,
                UNIQUE(repo_id, path)
            )
        """)

        # Основная таблица чанков знаний
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                file_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                line_start INTEGER,
                line_end INTEGER,
                source_kind TEXT NOT NULL,  -- 'code', 'docs', 'knowledge'
                trust TEXT NOT NULL,        -- 'verified', 'hint'
                sha TEXT,
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
            )
        """)

        # FTS5 виртуальная таблица для гибридного и текстового поиска
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content_rowid=id
            )
        """)

        # Триггеры для поддержки fts5 в актуальном состоянии при дельта-синхронизации (INSERT/DELETE/UPDATE)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(content_rowid, content) VALUES (new.id, new.content);
            END;
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, content_rowid, content) VALUES('delete', old.id, old.content);
            END;
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, content_rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO chunks_fts(content_rowid, content) VALUES (new.id, new.content);
            END;
        """)

        self.conn.commit()
    
    def close(self):
        """Закрывает соединение с БД."""
        self.conn.close()

    def get_known_files(self, repo_id: str) -> Dict[str, sqlite3.Row]:
        """Возвращает словарь известных файлов репозитория: path -> Row(id, mtime, hash)."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, path, mtime, hash FROM files WHERE repo_id = ?", (repo_id,))
        return {row['path']: row for row in cursor.fetchall()}

    def upsert_file(self, repo_id: str, path: str, mtime: float, file_hash: str) -> int:
        """Обновляет или вставляет запись о файле. Возвращает file_id."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO files (repo_id, path, mtime, hash)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(repo_id, path) DO UPDATE SET
                mtime=excluded.mtime,
                hash=excluded.hash
        """, (repo_id, path, mtime, file_hash))
        self.conn.commit()
        
        cursor.execute("SELECT id FROM files WHERE repo_id = ? AND path = ?", (repo_id, path))
        return cursor.fetchone()['id']

    def delete_file(self, file_id: int):
        """Удаляет файл и каскадно (благодаря FOREIGN KEY) удаляет все его чанки."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
        self.conn.commit()

    def clear_file_chunks(self, file_id: int):
        """Удаляет все чанки, привязанные к заданному файлу (перед реиндексацией 'грязного' файла)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        self.conn.commit()

    def add_chunk(self, chunk_id: str, file_id: int, content: str, 
                  source_kind: str, trust: str, 
                  line_start: Optional[int] = None, line_end: Optional[int] = None, 
                  sha: Optional[str] = None):
        """Добавляет чанк в базу."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO chunks (id, file_id, content, line_start, line_end, source_kind, trust, sha)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (chunk_id, file_id, content, line_start, line_end, source_kind, trust, sha))
        self.conn.commit()
        
    def search_chunks_fts(self, query: str, repo_ids: Optional[List[str]] = None, limit: int = 10) -> List[sqlite3.Row]:
        """Полнотекстовый поиск по чанкам."""
        cursor = self.conn.cursor()
        
        sql = '''
            SELECT c.*, f.repo_id, f.path, bm25(chunks_fts) as rank
            FROM chunks_fts fts
            JOIN chunks c ON fts.content_rowid = c.id
            JOIN files f ON c.file_id = f.id
            WHERE chunks_fts MATCH ?
        '''
        params = [query]
        
        if repo_ids:
            placeholders = ','.join('?' for _ in repo_ids)
            sql += f" AND f.repo_id IN ({placeholders})"
            params.extend(repo_ids)
            
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        return cursor.fetchall()
