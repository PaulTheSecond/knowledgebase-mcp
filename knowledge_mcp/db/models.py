import sqlite3
import logging
import threading
import hashlib
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Dict, Any

from knowledge_mcp.db.connection import get_db_connection
import sqlite_vec

# Константа алгоритма Reciprocal Rank Fusion
RRF_K = 60

logger = logging.getLogger(__name__)

class KnowledgeDB:
    def __init__(self, db_path: str | Path, read_only: bool = False):
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.lock = threading.RLock()
        
        # Если БД не существует, инициализируем схему во временном пишущем соединении
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
            
        self.conn = get_db_connection(self.db_path, read_only=read_only)
        self.vec_dim = 384  # Обновлено: размерность для multilingual-e5-small
        self._in_transaction = False  # Флаг: внутри явной транзакции (skip individual commits)
        self._stats_cache: Dict[tuple, dict] = {}
        self._stats_cache_time: float = 0.0
        self._cas_cache: Dict[str, dict] = {}
        self._cas_cache_time: Dict[str, float] = {}
        if not read_only:
            self._apply_migrations()


    def _apply_migrations(self):
        """EF Core стайл миграции схемы БД."""
        cursor = self.conn.cursor()
        
        # Таблица истории миграций
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS __migrations_history (
                id TEXT PRIMARY KEY,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

        cursor.execute("SELECT id FROM __migrations_history")
        applied_migrations = {row['id'] for row in cursor.fetchall()}

        migrations = [
            {
                "id": "20260319_01_initial",
                "up": """
                    CREATE TABLE IF NOT EXISTS files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        repo_id TEXT NOT NULL,
                        path TEXT NOT NULL,
                        mtime REAL NOT NULL,
                        hash TEXT NOT NULL,
                        UNIQUE(repo_id, path)
                    );
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        file_id INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        line_start INTEGER,
                        line_end INTEGER,
                        source_kind TEXT NOT NULL,
                        trust TEXT NOT NULL,
                        sha TEXT,
                        FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
                    );
                """
            },
            {
                "id": "20260319_02_fts_and_vec",
                "up": f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        content
                    );
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                        embedding float[{self.vec_dim}]
                    );

                    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
                    END;
                    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                        DELETE FROM chunks_fts WHERE rowid = old.rowid;
                        DELETE FROM chunks_vec WHERE rowid = old.rowid;
                    END;
                    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                        DELETE FROM chunks_fts WHERE rowid = old.rowid;
                        DELETE FROM chunks_vec WHERE rowid = old.rowid;
                        INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
                    END;
                """
            },
            {
                "id": "20260326_03_fix_triggers",
                "up": """
                    DROP TRIGGER IF EXISTS chunks_ad;
                    DROP TRIGGER IF EXISTS chunks_au;

                    CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                        DELETE FROM chunks_fts WHERE rowid = old.rowid;
                        DELETE FROM chunks_vec WHERE rowid = old.rowid;
                    END;

                    CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
                        DELETE FROM chunks_fts WHERE rowid = old.rowid;
                        DELETE FROM chunks_vec WHERE rowid = old.rowid;
                        INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
                    END;
                """
            }
        ]

        # Добавляем новую миграцию
        migrations.append({
            "id": "20260414_04_symbols_graph",
            "up": """
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    qualified_name TEXT,
                    kind TEXT NOT NULL,
                    language TEXT NOT NULL,
                    line_start INTEGER,
                    line_end INTEGER,
                    signature TEXT,
                    chunk_id TEXT,
                    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS symbol_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES symbols(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES symbols(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS unresolved_refs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    target_qualified_name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES symbols(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
                CREATE INDEX IF NOT EXISTS idx_symbols_qualified ON symbols(qualified_name);
                CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
                CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);
                CREATE INDEX IF NOT EXISTS idx_edges_source ON symbol_edges(source_id);
                CREATE INDEX IF NOT EXISTS idx_edges_target ON symbol_edges(target_id);
                CREATE INDEX IF NOT EXISTS idx_edges_kind ON symbol_edges(kind);
            """
        })

        migrations.append({
            "id": "20260420_05_upgrade_to_e5_small",
            "up": f"""
                -- Полный сброс векторного индекса из-за смены модели и размерности
                DROP TABLE IF EXISTS chunks_vec;
                CREATE VIRTUAL TABLE chunks_vec USING vec0(
                    embedding float[{self.vec_dim}]
                );
            """
        })

        migrations.append({
            "id": "20260630_06_system_status",
            "up": """
                CREATE TABLE IF NOT EXISTS system_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    last_error TEXT,
                    last_indexed_at DATETIME,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Инициализация первой строки
                INSERT INTO system_status (id, status, last_error, last_indexed_at)
                SELECT 1, 'idle', NULL, NULL
                WHERE NOT EXISTS (SELECT 1 FROM system_status WHERE id = 1);
            """
        })

        migrations.append({
            "id": "20260701_07_sync_queue_and_indexed_commits",
            "up": """
                CREATE TABLE IF NOT EXISTS sync_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
                    attempts INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_queue_status_created ON sync_queue(status, created_at);

                CREATE TABLE IF NOT EXISTS indexed_commits (
                    repo_id TEXT PRIMARY KEY,
                    git_sha TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """
        })

        migrations.append({
            "id": "20260701_08_sync_queue_run_at",
            "up": """
                ALTER TABLE sync_queue ADD COLUMN run_at DATETIME;
            """
        })

        migrations.append({
            "id": "20261015_09_idx_chunks_file",
            "up": """
                CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
            """
        })

        migrations.append({
            "id": "20261015_10_sync_queue_changed_files",
            "up": """
                ALTER TABLE sync_queue ADD COLUMN changed_files TEXT;
            """
        })

        migrations.append({
            "id": "20261018_11_fts5_external_content_and_autovacuum",
            "up": """
                -- 1. Удаляем старые триггеры
                DROP TRIGGER IF EXISTS chunks_ai;
                DROP TRIGGER IF EXISTS chunks_ad;
                DROP TRIGGER IF EXISTS chunks_au;

                -- 2. Пересоздаем chunks_fts как внешнюю контент-таблицу (content='chunks', content_rowid='rowid')
                -- Это исключает дублирование текстов кода в chunks_fts_content (~35-45% экономии диска)
                DROP TABLE IF EXISTS chunks_fts;
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    content,
                    content='chunks',
                    content_rowid='rowid'
                );

                -- 3. Заполняем FTS-индекс из существующей таблицы chunks
                INSERT INTO chunks_fts(rowid, content) SELECT rowid, content FROM chunks;

                -- 4. Создаем новые триггеры для внешней контент-таблицы FTS5
                CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
                END;

                CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                    DELETE FROM chunks_vec WHERE rowid = old.rowid;
                END;

                CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                    DELETE FROM chunks_vec WHERE rowid = old.rowid;
                    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
                END;

                -- 5. Включаем auto_vacuum = FULL и выполняем VACUUM для сжатия базы
                PRAGMA auto_vacuum = FULL;
                VACUUM;
            """
        })

        migrations.append({
            "id": "20261018_12_app_settings",
            "up": """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                INSERT OR IGNORE INTO app_settings (key, value) VALUES ('default_locale', 'ru');
                INSERT OR IGNORE INTO app_settings (key, value) VALUES ('default_theme', 'dark');
            """
        })

        migrations.append({
            "id": "20261019_13_sync_queue_durations",
            "up": """
                ALTER TABLE sync_queue ADD COLUMN started_at DATETIME;
                ALTER TABLE sync_queue ADD COLUMN finished_at DATETIME;
            """
        })

        migrations.append({
            "id": "20261101_12_cas_and_multi_worktree",
            "up": """
                -- 1. Таблица file_contents (Content-Addressable Storage)
                CREATE TABLE IF NOT EXISTS file_contents (
                    hash TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    language TEXT,
                    updated_at REAL NOT NULL
                );

                -- Заполняем file_contents для существующих файлов, если они есть
                INSERT OR IGNORE INTO file_contents (hash, content, size, language, updated_at)
                SELECT 
                    f.hash,
                    COALESCE((SELECT group_concat(c.content, x'0A') FROM (SELECT content FROM chunks WHERE file_id = f.id ORDER BY line_start ASC) c), ''),
                    COALESCE((SELECT SUM(length(c.content)) FROM chunks c WHERE c.file_id = f.id), 0),
                    NULL,
                    strftime('%s', 'now')
                FROM files f
                WHERE f.hash IS NOT NULL AND f.hash != '';

                -- 2. Миграция таблицы files на поддержку multi-worktree (project_id, branch, content_hash)
                PRAGMA foreign_keys = OFF;

                CREATE TABLE files_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL DEFAULT 'default',
                    repo_id TEXT NOT NULL,
                    branch TEXT NOT NULL DEFAULT 'master',
                    path TEXT NOT NULL,
                    content_hash TEXT REFERENCES file_contents(hash),
                    mtime REAL NOT NULL,
                    hash TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(project_id, repo_id, branch, path)
                );

                INSERT INTO files_new (id, project_id, repo_id, branch, path, content_hash, mtime, hash)
                SELECT 
                    id,
                    'default',
                    repo_id,
                    'master',
                    path,
                    hash,
                    mtime,
                    hash
                FROM files;

                DROP TABLE files;
                ALTER TABLE files_new RENAME TO files;

                CREATE INDEX IF NOT EXISTS idx_files_lookup ON files(project_id, repo_id, branch);
                CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files(content_hash);

                PRAGMA foreign_keys = ON;

                -- 3. Добавление колонки content_hash в таблицы chunks и symbols для дедупликации
                ALTER TABLE chunks ADD COLUMN content_hash TEXT REFERENCES file_contents(hash);
                ALTER TABLE symbols ADD COLUMN content_hash TEXT REFERENCES file_contents(hash);

                UPDATE chunks
                SET content_hash = (SELECT content_hash FROM files WHERE files.id = chunks.file_id)
                WHERE content_hash IS NULL;

                UPDATE symbols
                SET content_hash = (SELECT content_hash FROM files WHERE files.id = symbols.file_id)
                WHERE content_hash IS NULL;

                CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
                CREATE INDEX IF NOT EXISTS idx_symbols_content_hash ON symbols(content_hash);

                -- 4. Таблица лога экономии токенов (Token Savings Telemetry)
                CREATE TABLE IF NOT EXISTS token_savings_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    tool_name TEXT NOT NULL,
                    project_id TEXT DEFAULT 'default',
                    repo_id TEXT,
                    branch TEXT,
                    baseline_tokens INTEGER NOT NULL,
                    returned_tokens INTEGER NOT NULL,
                    saved_tokens INTEGER NOT NULL,
                    saved_percent INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_token_savings_ts ON token_savings_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_token_savings_project ON token_savings_log(project_id);
            """
        })

        migrations.append({
            "id": "20261102_14_idx_file_contents_hash_size",
            "up": """
                CREATE INDEX IF NOT EXISTS idx_file_contents_hash_size ON file_contents(hash, size);
            """
        })

        for migration in migrations:
            if migration["id"] not in applied_migrations:
                logger.info(f"Applying sqlite migration: {migration['id']}")
                self.conn.executescript(migration["up"])
                cursor.execute("INSERT INTO __migrations_history (id) VALUES (?)", (migration["id"],))
                self.conn.commit()
    
    def close(self):
        """Закрывает соединение с БД."""
        self.conn.close()

    def vacuum(self):
        """Выполняет VACUUM для дефрагментации и сжатия файла БД."""
        with self.lock:
            self.conn.execute("VACUUM;")

    def optimize(self):
        """Запускает PRAGMA optimize для актуализации статистики планировщика запросов."""
        with self.lock:
            self.conn.execute("PRAGMA optimize;")

    def get_vacuum_mode(self) -> int:
        """Возвращает текущий режим auto_vacuum (0=NONE, 1=FULL, 2=INCREMENTAL)."""
        cursor = self.conn.cursor()
        return cursor.execute("PRAGMA auto_vacuum;").fetchone()[0]

    def begin_transaction(self):
        """Начинает явную транзакцию для пакетных операций (избегаем тысяч отдельных commit)."""
        self.conn.execute("BEGIN")
        self._in_transaction = True

    def commit_transaction(self):
        """Фиксирует текущую транзакцию."""
        if getattr(self, '_in_transaction', False):
            self.conn.commit()
            self._in_transaction = False

    def rollback_transaction(self):
        """Откатывает текущую транзакцию при ошибке."""
        if getattr(self, '_in_transaction', False):
            try:
                self.conn.rollback()
            except sqlite3.OperationalError:
                pass
            finally:
                self._in_transaction = False

    def _auto_commit(self):
        """Вызывает commit() только если мы НЕ внутри явной транзакции."""
        if not self._in_transaction:
            self.conn.commit()

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

    def invalidate_stats_cache(self):
        """Сбрасывает in-memory кэш метрик репозиториев и CAS-статистики."""
        with self.lock:
            self._stats_cache_time = 0.0
            self._stats_cache.clear()
            self._cas_cache.clear()
            self._cas_cache_time.clear()

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

    def get_queue_jobs(self, limit: int = 50) -> list:
        """Возвращает последние N задач из очереди синхронизации с расчетом времени выполнения."""
        from datetime import datetime
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, repo_id, status, attempts, error_message as last_error, 
                   run_at, created_at, updated_at, started_at, finished_at, changed_files 
            FROM sync_queue 
            ORDER BY created_at DESC 
            LIMIT ?
            """,
            (limit,)
        )
        jobs = []
        fmt = "%Y-%m-%d %H:%M:%S"
        for row in cursor.fetchall():
            job = dict(row)
            duration_str = None
            duration_secs = None
            started_at_str = job.get("started_at")
            finished_at_str = job.get("finished_at")

            if started_at_str:
                try:
                    clean_start = str(started_at_str).replace("T", " ").split(".")[0]
                    t_start = datetime.strptime(clean_start, fmt)

                    if finished_at_str:
                        clean_end = str(finished_at_str).replace("T", " ").split(".")[0]
                        t_end = datetime.strptime(clean_end, fmt)
                        duration_secs = max(0, int((t_end - t_start).total_seconds()))
                    elif job.get("status") == "processing":
                        duration_secs = max(0, int((datetime.now() - t_start).total_seconds()))

                    if duration_secs is not None:
                        if duration_secs < 60:
                            duration_str = f"{duration_secs}с"
                        elif duration_secs < 3600:
                            mins = duration_secs // 60
                            secs = duration_secs % 60
                            duration_str = f"{mins}м {secs}с"
                        else:
                            hrs = duration_secs // 3600
                            mins = (duration_secs % 3600) // 60
                            duration_str = f"{hrs}ч {mins}м"
                except Exception as e:
                    logger.debug(f"Error calculating duration for job {job.get('id')}: {e}")

            job["duration"] = duration_str
            job["duration_seconds"] = duration_secs
            jobs.append(job)
        return jobs

    def clear_file_chunks(self, file_id: int):
        """Удаляет все чанки, привязанные к заданному файлу (перед реиндексацией 'грязного' файла)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        self._auto_commit()

    def clear_file_symbols(self, file_id: int):
        """Удаляет все символы, привязанные к файлу (связи удалятся каскадно)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))
        self._auto_commit()

    def add_symbol(
        self,
        file_id: int,
        name: str,
        qualified_name: str,
        kind: str,
        language: str,
        line_start: int,
        line_end: int,
        signature: str,
        chunk_id: str,
        content_hash: Optional[str] = None
    ) -> int:
        cursor = self.conn.cursor()
        if content_hash is None and file_id:
            cursor.execute("SELECT content_hash FROM files WHERE id = ?", (file_id,))
            row = cursor.fetchone()
            if row and row['content_hash']:
                content_hash = row['content_hash']
        cursor.execute("""
            INSERT INTO symbols (file_id, name, qualified_name, kind, language, line_start, line_end, signature, chunk_id, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (file_id, name, qualified_name, kind, language, line_start, line_end, signature, chunk_id, content_hash))
        self._auto_commit()
        return cursor.lastrowid

    def add_symbols_batch(self, records: list[tuple]) -> list[int]:
        """Батчевая вставка символов.
        records: [(file_id, name, qname, kind, lang, ls, le, sig, chunk_id), ...]
        или [(file_id, name, qname, kind, lang, ls, le, sig, chunk_id, content_hash), ...]
        Возвращает список rowid вставленных символов."""
        if not records:
            return []
        cursor = self.conn.cursor()
        rowids = []
        for record in records:
            if len(record) == 9:
                file_id, name, qname, kind, lang, ls, le, sig, chunk_id = record
                content_hash = None
                if file_id:
                    cursor.execute("SELECT content_hash FROM files WHERE id = ?", (file_id,))
                    row = cursor.fetchone()
                    if row and row['content_hash']:
                        content_hash = row['content_hash']
                cursor.execute("""
                    INSERT INTO symbols (file_id, name, qualified_name, kind, language, line_start, line_end, signature, chunk_id, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (file_id, name, qname, kind, lang, ls, le, sig, chunk_id, content_hash))
            elif len(record) == 10:
                cursor.execute("""
                    INSERT INTO symbols (file_id, name, qualified_name, kind, language, line_start, line_end, signature, chunk_id, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, record)
            rowids.append(cursor.lastrowid)
        self._auto_commit()
        return rowids

    def add_symbol_edge(self, source_id: int, target_id: int, kind: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO symbol_edges (source_id, target_id, kind)
            VALUES (?, ?, ?)
        """, (source_id, target_id, kind))
        self._auto_commit()

    def add_symbol_edges_batch(self, records: list[tuple]):
        """Батчевая вставка рёбер. records: [(source_id, target_id, kind), ...]"""
        if not records:
            return
        cursor = self.conn.cursor()
        cursor.executemany("""
            INSERT INTO symbol_edges (source_id, target_id, kind)
            VALUES (?, ?, ?)
        """, records)
        self._auto_commit()

    def add_unresolved_ref(self, source_id: int, target_qualified_name: str, kind: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO unresolved_refs (source_id, target_qualified_name, kind)
            VALUES (?, ?, ?)
        """, (source_id, target_qualified_name, kind))
        self._auto_commit()

    def add_unresolved_refs_batch(self, records: list[tuple]):
        """Батчевая вставка unresolved ссылок. records: [(source_id, target_qname, kind), ...]"""
        if not records:
            return
        cursor = self.conn.cursor()
        cursor.executemany("""
            INSERT INTO unresolved_refs (source_id, target_qualified_name, kind)
            VALUES (?, ?, ?)
        """, records)
        self._auto_commit()

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

    def resolve_pending_references(self, limit: int = 5000) -> int:
        """
        Пытается найти таргеты для ссылок из unresolved_refs через быстрый пакетный JOIN.
        Ограничен limit для защиты от зависания при большом числе системных/внешних ссылок.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT u.id, u.source_id, u.kind, s.id as target_id
            FROM unresolved_refs u
            JOIN symbols s ON (s.qualified_name = u.target_qualified_name OR s.name = u.target_qualified_name)
            LIMIT ?
        """, (limit,))
        matches = cursor.fetchall()
        if not matches:
            return 0
        
        edges_to_insert = []
        ids_to_delete = []
        for row in matches:
            ref_id = row['id']
            src_id = row['source_id']
            kind = row['kind']
            target_id = row['target_id']
            if kind == 'TESTED_BY':
                edges_to_insert.append((target_id, src_id, 'TESTED_BY'))
            else:
                edges_to_insert.append((src_id, target_id, kind))
            ids_to_delete.append(ref_id)
            
        if edges_to_insert:
            cursor.executemany(
                "INSERT OR IGNORE INTO symbol_edges (source_id, target_id, kind) VALUES (?, ?, ?)",
                edges_to_insert
            )
            
        step = 900
        for i in range(0, len(ids_to_delete), step):
            batch = ids_to_delete[i:i+step]
            placeholders = ','.join('?' for _ in batch)
            cursor.execute(f"DELETE FROM unresolved_refs WHERE id IN ({placeholders})", batch)
        self._auto_commit()
            
        return len(ids_to_delete)

    def get_symbols(
        self,
        name_pattern: str = "%",
        kind: Optional[str] = None,
        repo_ids: Optional[List[str]] = None,
        project_id: Optional[str] = "default",
        branch: Optional[str] = "master",
        limit: int = 20
    ) -> List[sqlite3.Row]:
        """Поиск символов с фильтрацией по project_id и branch."""
        cursor = self.conn.cursor()
        sql = '''
            SELECT DISTINCT s.*, f.repo_id, f.path, f.project_id, f.branch
            FROM symbols s
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE (s.name LIKE ? OR s.qualified_name LIKE ?)
        '''
        sql_pattern = name_pattern.replace('*', '%')
        if '%' not in sql_pattern:
            sql_pattern = f"%{sql_pattern}%"
            
        params: list[Any] = [sql_pattern, sql_pattern]
        
        if kind:
            sql += " AND s.kind = ?"
            params.append(kind)
            
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
            
        sql += " LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        return cursor.fetchall()

    def find_symbols(
        self,
        name_pattern: str,
        kind: Optional[str] = None,
        repo_ids: Optional[List[str]] = None,
        limit: int = 20,
        project_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> List[sqlite3.Row]:
        """Обратно-совместимый метод поиска символов."""
        return self.get_symbols(
            name_pattern=name_pattern,
            kind=kind,
            repo_ids=repo_ids,
            project_id=project_id,
            branch=branch,
            limit=limit
        )

    def get_symbol_callers(
        self,
        symbol_id: int,
        project_id: Optional[str] = "default",
        branch: Optional[str] = "master"
    ) -> List[sqlite3.Row]:
        """Возвращает вызывающие символы (callers) для symbol_id."""
        cursor = self.conn.cursor()
        sql = '''
            SELECT DISTINCT s.*, f.repo_id, f.path, f.project_id, f.branch, e.kind as edge_kind
            FROM symbol_edges e
            JOIN symbols s ON e.source_id = s.id
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE e.target_id = ?
        '''
        params: list[Any] = [symbol_id]
        if project_id is not None:
            sql += " AND f.project_id = ?"
            params.append(project_id)
        if branch is not None:
            sql += " AND f.branch = ?"
            params.append(branch)
        cursor.execute(sql, params)
        return cursor.fetchall()

    def get_callers(self, symbol_id: int) -> List[sqlite3.Row]:
        """Обратно-совместимый вызов получения callers."""
        return self.get_symbol_callers(symbol_id, project_id=None, branch=None)

    def get_symbol_callees(
        self,
        symbol_id: int,
        project_id: Optional[str] = "default",
        branch: Optional[str] = "master"
    ) -> List[sqlite3.Row]:
        """Возвращает вызываемые символы (callees) для symbol_id."""
        cursor = self.conn.cursor()
        sql = '''
            SELECT DISTINCT s.*, f.repo_id, f.path, f.project_id, f.branch, e.kind as edge_kind
            FROM symbol_edges e
            JOIN symbols s ON e.target_id = s.id
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE e.source_id = ?
        '''
        params: list[Any] = [symbol_id]
        if project_id is not None:
            sql += " AND f.project_id = ?"
            params.append(project_id)
        if branch is not None:
            sql += " AND f.branch = ?"
            params.append(branch)
        cursor.execute(sql, params)
        return cursor.fetchall()

    def get_callees(self, symbol_id: int) -> List[sqlite3.Row]:
        """Обратно-совместимый вызов получения callees."""
        return self.get_symbol_callees(symbol_id, project_id=None, branch=None)

    def get_hierarchy(
        self,
        symbol_id: int,
        project_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> List[sqlite3.Row]:
        cursor = self.conn.cursor()
        sql = '''
            SELECT DISTINCT s.*, f.repo_id, f.path, f.project_id, f.branch, e.kind as edge_kind
            FROM symbol_edges e
            JOIN symbols s ON e.target_id = s.id
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE e.source_id = ? AND e.kind IN ('INHERITS', 'IMPLEMENTS')
        '''
        params: list[Any] = [symbol_id]
        if project_id is not None:
            sql += " AND f.project_id = ?"
            params.append(project_id)
        if branch is not None:
            sql += " AND f.branch = ?"
            params.append(branch)
        cursor.execute(sql, params)
        return cursor.fetchall()

    def get_impact_analysis(
        self,
        symbol_id: int,
        max_depth: int = 3,
        project_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        
        visited = set()
        queue: deque = deque([(symbol_id, 1)])
        impacted = []
        
        while queue:
            current_id, depth = queue.popleft()
            if depth > max_depth:
                continue
                
            sql = '''
                SELECT DISTINCT s.*, f.repo_id, f.path, f.project_id, f.branch, e.kind as edge_kind
                FROM symbol_edges e
                JOIN symbols s ON e.source_id = s.id
                JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
                WHERE e.target_id = ? AND e.kind IN ('CALLS', 'INHERITS', 'IMPLEMENTS', 'IMPORTS', 'USES')
            '''
            params: list[Any] = [current_id]
            if project_id is not None:
                sql += " AND f.project_id = ?"
                params.append(project_id)
            if branch is not None:
                sql += " AND f.branch = ?"
                params.append(branch)
            cursor.execute(sql, params)
            
            callers = cursor.fetchall()
            for r in callers:
                caller_id = r['id']
                if caller_id not in visited:
                    visited.add(caller_id)
                    impacted.append({
                        'symbol': r,
                        'depth': depth,
                        'edge_kind': r['edge_kind']
                    })
                    queue.append((caller_id, depth + 1))
                    
        return impacted

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

    def add_embedding(self, chunk_rowid: int, vector: List[float]):
        """Добавляет векторное представление в виртуальную vec0 таблицу."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO chunks_vec(rowid, embedding)
            VALUES (?, ?)
        """, (chunk_rowid, sqlite_vec.serialize_float32(vector)))
        self._auto_commit()

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
                    SELECT COUNT(*) as contents_count,
                           COALESCE(SUM(size), 0) as real_size
                    FROM file_contents fc INDEXED BY idx_file_contents_hash_size
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

    def log_token_savings(
        self,
        tool_name: str,
        baseline_tokens: int,
        returned_tokens: int,
        project_id: str = "default",
        repo_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> int:
        """Логирует сэкономленные токены контекста при вызове инструментов."""
        saved_tokens = max(0, baseline_tokens - returned_tokens)
        saved_percent = round((saved_tokens / baseline_tokens * 100)) if baseline_tokens > 0 else 0
        
        if getattr(self, "read_only", False):
            return 0

        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO token_savings_log (
                    tool_name, project_id, repo_id, branch,
                    baseline_tokens, returned_tokens, saved_tokens, saved_percent
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (tool_name, project_id, repo_id, branch, baseline_tokens, returned_tokens, saved_tokens, saved_percent))
            self._auto_commit()
            return cursor.lastrowid
        except sqlite3.OperationalError as e:
            if "readonly" in str(e).lower():
                logger.warning(f"Cannot log token savings: database is read-only: {e}")
                return 0
            raise

    def get_token_savings_summary(self, days: int = 7, project_id: Optional[str] = None) -> dict:
        """Возвращает сводку телеметрии экономии токенов за указанный период (в днях)."""
        cursor = self.conn.cursor()
        where_clauses = []
        params: list[Any] = []
        
        if days is not None and days > 0:
            where_clauses.append("timestamp >= datetime('now', ?)")
            params.append(f"-{days} days")
            
        if project_id:
            where_clauses.append("project_id = ?")
            params.append(project_id)
            
        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        
        cursor.execute(f"""
            SELECT 
                COALESCE(SUM(baseline_tokens), 0) as total_baseline,
                COALESCE(SUM(returned_tokens), 0) as total_returned,
                COALESCE(SUM(saved_tokens), 0) as total_saved,
                COALESCE(AVG(saved_percent), 0.0) as avg_saved_percent,
                COUNT(*) as total_calls
            FROM token_savings_log
            {where_sql}
        """, params)
        row = cursor.fetchone()
        
        total_baseline = row['total_baseline']
        total_returned = row['total_returned']
        total_saved = row['total_saved']
        avg_saved_percent = round(float(row['avg_saved_percent']), 2)
        overall_saved_percent = round((total_saved / total_baseline * 100), 2) if total_baseline > 0 else 0.0
        total_calls = row['total_calls']
        
        cursor.execute(f"""
            SELECT 
                tool_name,
                COALESCE(SUM(baseline_tokens), 0) as total_baseline,
                COALESCE(SUM(returned_tokens), 0) as total_returned,
                COALESCE(SUM(saved_tokens), 0) as total_saved,
                COALESCE(AVG(saved_percent), 0.0) as avg_saved_percent,
                COUNT(*) as calls_count
            FROM token_savings_log
            {where_sql}
            GROUP BY tool_name
            ORDER BY total_saved DESC
        """, params)
        
        by_tool = {}
        for r in cursor.fetchall():
            tool_name = r['tool_name']
            t_baseline = r['total_baseline']
            t_saved = r['total_saved']
            by_tool[tool_name] = {
                "total_baseline_tokens": t_baseline,
                "total_returned_tokens": r['total_returned'],
                "total_saved_tokens": t_saved,
                "saved_tokens": t_saved,
                "avg_saved_percent": round(float(r['avg_saved_percent']), 2),
                "saved_percent": round((t_saved / t_baseline * 100), 2) if t_baseline > 0 else 0.0,
                "calls_count": r['calls_count']
            }
            
        return {
            "total_saved_tokens": total_saved,
            "total_saved": total_saved,
            "saved_tokens": total_saved,
            "avg_saved_percent": avg_saved_percent,
            "average_saved_percent": avg_saved_percent,
            "overall_saved_percent": overall_saved_percent,
            "total_baseline_tokens": total_baseline,
            "total_returned_tokens": total_returned,
            "total_calls": total_calls,
            "by_tool": by_tool,
            "tools_breakdown": by_tool
        }

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


    def get_system_status(self) -> Optional[sqlite3.Row]:
        """Возвращает текущий статус системы."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT status, last_error, last_indexed_at, updated_at FROM system_status WHERE id = 1")
            return cursor.fetchone()

    def update_system_status(self, status: str, last_error: Optional[str] = None, last_indexed_at: Optional[float] = None) -> None:
        """Обновляет статус системы."""
        with self.lock:
            cursor = self.conn.cursor()
            if last_indexed_at is not None:
                cursor.execute("""
                    UPDATE system_status 
                    SET status = ?, last_error = ?, last_indexed_at = datetime(?, 'unixepoch', 'localtime'), updated_at = datetime('now', 'localtime')
                    WHERE id = 1
                """, (status, last_error, last_indexed_at))
            else:
                cursor.execute("""
                    UPDATE system_status 
                    SET status = ?, last_error = ?, updated_at = datetime('now', 'localtime')
                    WHERE id = 1
                """, (status, last_error))
            self._auto_commit()
            self.invalidate_stats_cache()

    def add_to_queue(self, repo_id: str, changed_files: Optional[List[str]] = None) -> int:
        """Добавляет репозиторий в очередь на синхронизацию.

        Args:
            repo_id: Идентификатор репозитория.
            changed_files: Опциональный список относительных путей измененных файлов из вебхука.
        """
        import json
        changed_json = json.dumps(changed_files) if changed_files else None
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO sync_queue (repo_id, status, changed_files)
                VALUES (?, 'pending', ?)
            """, (repo_id, changed_json))
            self._auto_commit()
            return cursor.lastrowid

    def fetch_next_job(self) -> Optional[Dict[str, Any]]:
        """Извлекает следующую pending задачу по принципу FIFO и переводит ее в processing в IMMEDIATE транзакции.

        Учитывает колонку run_at: задачи с run_at в будущем пропускаются (экспоненциальный бэк-офф).
        """
        with self.lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE TRANSACTION;")
                cursor.execute("""
                    SELECT id, repo_id, attempts, changed_files
                    FROM sync_queue
                    WHERE status = 'pending'
                      AND (run_at IS NULL OR run_at <= datetime('now', 'localtime'))
                    ORDER BY created_at ASC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    job_id, repo_id, attempts, changed_json = row[0], row[1], row[2], row[3]
                    import json
                    changed_files = json.loads(changed_json) if changed_json else None
                    cursor.execute("""
                        UPDATE sync_queue
                        SET status = 'processing',
                            started_at = datetime('now', 'localtime'),
                            finished_at = NULL,
                            updated_at = datetime('now', 'localtime')
                        WHERE id = ?
                    """, (job_id,))
                    self.conn.commit()
                    return {
                        "id": job_id,
                        "repo_id": repo_id,
                        "attempts": attempts,
                        "changed_files": changed_files
                    }
                self.conn.commit()
                return None
            except Exception as e:
                self.conn.rollback()
                logger.error(f"Error fetching next job from queue: {e}")
                raise e

    def update_job_status(self, job_id: int, status: str, attempts: Optional[int] = None, error_message: Optional[str] = None) -> None:
        """Обновляет статус и параметры задачи в очереди с фиксацией времени завершения."""
        with self.lock:
            cursor = self.conn.cursor()
            finished_clause = ", finished_at = datetime('now', 'localtime')" if status in ("completed", "failed") else ""
            if attempts is not None:
                cursor.execute(f"""
                    UPDATE sync_queue
                    SET status = ?, attempts = ?, error_message = ?, updated_at = datetime('now', 'localtime'){finished_clause}
                    WHERE id = ?
                """, (status, attempts, error_message, job_id))
            else:
                cursor.execute(f"""
                    UPDATE sync_queue
                    SET status = ?, error_message = ?, updated_at = datetime('now', 'localtime'){finished_clause}
                    WHERE id = ?
                """, (status, error_message, job_id))
            self._auto_commit()
            if status in ("completed", "failed"):
                self.invalidate_stats_cache()

    def get_indexed_commit(self, repo_id: str) -> Optional[str]:
        """Возвращает хеш последнего проиндексированного коммита для репозитория."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT git_sha FROM indexed_commits WHERE repo_id = ?", (repo_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def update_indexed_commit(self, repo_id: str, git_sha: str) -> None:
        """Обновляет хеш последнего проиндексированного коммита."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO indexed_commits (repo_id, git_sha, updated_at)
                VALUES (?, ?, datetime('now', 'localtime'))
            """, (repo_id, git_sha))
            self._auto_commit()

    def reschedule_job(self, job_id: int, attempts: int, delay_seconds: int, error_message: str) -> None:
        """Возвращает задачу в pending с заданной задержкой (run_at) для экспоненциального бэк-оффа.

        Args:
            job_id: ID задачи.
            attempts: Текущее число попыток.
            delay_seconds: Задержка в секундах до следующей попытки.
            error_message: Текст ошибки.
        """
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE sync_queue
                SET status = 'pending',
                    attempts = ?,
                    error_message = ?,
                    run_at = datetime('now', 'localtime', '+' || ? || ' seconds'),
                    finished_at = datetime('now', 'localtime'),
                    updated_at = datetime('now', 'localtime')
                WHERE id = ?
            """, (attempts, error_message, delay_seconds, job_id))
            self._auto_commit()

    def reset_stale_jobs(self) -> int:
        """Сбрасывает задачи, оставшиеся в статусе processing при перезапуске сервера/воркера."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE sync_queue
                SET status = 'failed',
                    error_message = 'Прервано: перезапуск сервера во время выполнения',
                    finished_at = datetime('now', 'localtime'),
                    updated_at = datetime('now', 'localtime')
                WHERE status = 'processing'
            """)
            count = cursor.rowcount
            self._auto_commit()
            if count > 0:
                logger.warning(f"Reset {count} stale processing job(s) to failed state.")
                sys_st = self.get_system_status()
                if sys_st and sys_st["status"] == "syncing":
                    self.update_system_status("idle", last_error=f"Сброшено {count} зависших задач при запуске")
            return count

    def stop_job(self, job_id: int) -> bool:
        """Останавливает задачу в очереди (переводит в failed с пометкой об остановке)."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE sync_queue
                SET status = 'failed',
                    error_message = 'Остановлено пользователем',
                    finished_at = datetime('now', 'localtime'),
                    updated_at = datetime('now', 'localtime')
                WHERE id = ? AND status IN ('pending', 'processing')
            """, (job_id,))
            updated = cursor.rowcount > 0
            self._auto_commit()
            if updated:
                sys_st = self.get_system_status()
                if sys_st and sys_st["status"] == "syncing":
                    cursor.execute("SELECT COUNT(*) FROM sync_queue WHERE status = 'processing'")
                    if cursor.fetchone()[0] == 0:
                        self.update_system_status("idle")
            return updated

    def restart_job(self, job_id: int) -> bool:
        """Перезапускает задачу: переводит обратно в pending для повторного выполнения."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE sync_queue
                SET status = 'pending',
                    attempts = 0,
                    error_message = NULL,
                    run_at = NULL,
                    started_at = NULL,
                    finished_at = NULL,
                    updated_at = datetime('now', 'localtime')
                WHERE id = ?
            """, (job_id,))
            updated = cursor.rowcount > 0
            self._auto_commit()
            return updated

    def delete_job(self, job_id: int) -> bool:
        """Удаляет задачу из очереди."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM sync_queue WHERE id = ?", (job_id,))
            deleted = cursor.rowcount > 0
            self._auto_commit()
            return deleted

    def retry_failed_jobs(self) -> int:
        """Сбрасывает задачи со статусом failed/halted обратно в pending с attempts=0."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE sync_queue
                SET status = 'pending',
                    attempts = 0,
                    error_message = NULL,
                    run_at = NULL,
                    started_at = NULL,
                    finished_at = NULL,
                    updated_at = datetime('now', 'localtime')
                WHERE status IN ('failed', 'halted')
            """)
            count = cursor.rowcount
            self._auto_commit()
            return count

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Возвращает значение настройки из таблицы app_settings."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        """Сохраняет или обновляет значение настройки в таблице app_settings."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, datetime('now', 'localtime'))
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
            """, (key, value))
            self._auto_commit()

    def get_all_settings(self) -> Dict[str, str]:
        """Возвращает все сохраненные настройки в виде словаря."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT key, value FROM app_settings")
            return {row[0]: row[1] for row in cursor.fetchall()}

