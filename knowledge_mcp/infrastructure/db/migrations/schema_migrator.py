"""Модуль миграций схемы базы данных SQLite (EF Core style migrations)."""

import logging
import sqlite3
import time
from typing import Any, List, Set

logger = logging.getLogger("knowledge_mcp.db.migrations")


def apply_schema_migrations(conn: sqlite3.Connection, vec_dim: int = 384) -> None:
    """Применяет миграции схемы SQLite с устойчивостью к конфликтам блокировок DrvFs/9P."""
    """EF Core стайл миграции схемы БД с устойчивостью к конфликтам блокировок DrvFs/9P."""
    cursor = conn.cursor()
    
    # Таблица истории миграций с повторными попытками при DrvFs/9P блокировках
    max_retries = 10
    for attempt in range(max_retries):
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS __migrations_history (
                    id TEXT PRIMARY KEY,
                    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            break
        except sqlite3.OperationalError as e:
            err_str = str(e).lower()
            if ("disk i/o error" in err_str or "database is locked" in err_str) and attempt < max_retries - 1:
                wait_sec = (attempt + 1) * 0.5
                logger.warning(f"Migration history lock contention ({e}). Retrying in {wait_sec}s...")
                time.sleep(wait_sec)
            else:
                raise

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
                    embedding float[{vec_dim}]
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
                embedding float[{vec_dim}]
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

    migrations.append({
        "id": "20261104_16_idx_unresolved_refs",
        "up": """
            CREATE TABLE IF NOT EXISTS unresolved_refs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_qualified_name TEXT NOT NULL,
                kind TEXT NOT NULL,
                FOREIGN KEY(source_id) REFERENCES symbols(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_unresolved_refs_target ON unresolved_refs(target_qualified_name, kind);
            CREATE INDEX IF NOT EXISTS idx_unresolved_refs_source ON unresolved_refs(source_id);
        """
    })

    migrations.append({
        "id": "20261103_15_semantic_indexing_status",
        "up": """
            CREATE TABLE IF NOT EXISTS semantic_indexing_status (
                repo_id TEXT NOT NULL,
                project_id TEXT DEFAULT 'default',
                branch TEXT DEFAULT 'master',
                status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed', 'failed')),
                last_full_sync DATETIME,
                last_delta_sync DATETIME,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (repo_id, project_id, branch)
            );

            CREATE INDEX IF NOT EXISTS idx_semantic_status 
            ON semantic_indexing_status(status, updated_at);
        """
    })

    for migration in migrations:
        if migration["id"] not in applied_migrations:
            logger.info(f"Applying sqlite migration: {migration['id']}")
            conn.executescript(migration["up"])
            cursor.execute("INSERT INTO __migrations_history (id) VALUES (?)", (migration["id"],))
            conn.commit()

    # Гарантируем наличие виртуальной таблицы chunks_vec
    try:
        cursor.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[{vec_dim}]);")
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to ensure chunks_vec table: {e}")
    
