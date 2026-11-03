"""Репозиторий job_queue (инфраструктурный слой доступа к данным)."""

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

logger = logging.getLogger("knowledge_mcp.db.job_queue")


class SqliteJobQueueRepository(BaseSqliteRepository):
    """Специализированный репозиторий SQLite для job_queue."""

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

        with self.lock:
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
                if "readonly" in str(e).lower() or "locked" in str(e).lower():
                    logger.warning(f"Cannot log token savings: database issue: {e}")
                    return 0
                raise
            except Exception as e:
                logger.warning(f"Failed to log token savings: {e}")
                return 0


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


    def get_system_status(self) -> Optional[sqlite3.Row]:
        """Возвращает текущий статус системы."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT status, last_error, last_indexed_at, updated_at FROM system_status WHERE id = 1")
            return cursor.fetchone()


    def reset_stale_jobs(self, timeout_minutes: Optional[int] = None) -> int:
        """Сбрасывает задачи, оставшиеся в статусе processing при перезапуске сервера/воркера или по таймауту."""
        with self.lock:
            cursor = self.conn.cursor()
            if timeout_minutes:
                cursor.execute(f"""
                    UPDATE sync_queue
                    SET status = 'failed',
                        error_message = 'Прервано: превышен таймаут выполнения ({timeout_minutes} мин)',
                        finished_at = datetime('now', 'localtime'),
                        updated_at = datetime('now', 'localtime')
                    WHERE status = 'processing'
                      AND (started_at IS NULL OR started_at <= datetime('now', 'localtime', '-{timeout_minutes} minutes'))
                """)
            else:
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


    def get_semantic_status(
        self,
        repo_id: str,
        project_id: str = "default",
        branch: str = "master"
    ) -> Optional[dict]:
        """Возвращает текущий статус семантической индексации Roslyn для репозитория."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT repo_id, project_id, branch, status,
                       last_full_sync, last_delta_sync, error_message, retry_count, updated_at
                FROM semantic_indexing_status
                WHERE repo_id = ? AND project_id = ? AND branch = ?
            """, (repo_id, project_id, branch))
            row = cursor.fetchone()
            return dict(row) if row else None


    def get_indexed_commit(self, repo_id: str) -> Optional[str]:
        """Возвращает хеш последнего проиндексированного коммита для репозитория."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT git_sha FROM indexed_commits WHERE repo_id = ?", (repo_id,))
            row = cursor.fetchone()
            return row[0] if row else None


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


    def get_all_settings(self) -> Dict[str, str]:
        """Возвращает все сохраненные настройки в виде словаря."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT key, value FROM app_settings")
            return {row[0]: row[1] for row in cursor.fetchall()}


    def fetch_next_job(self) -> Optional[Dict[str, Any]]:
        """Извлекает следующую pending задачу по принципу FIFO и переводит ее в processing в IMMEDIATE транзакции.

        Учитывает колонку run_at: задачи с run_at в будущем пропускаются (экспоненциальный бэк-офф).
        При запуске полной синхронизации автоматически очищает накопившиеся дубликаты из очереди.
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
                    # Дедупликация: при старте полного sync удаляем все остальные pending-дубликаты для этого репозитория
                    if changed_files is None:
                        cursor.execute("""
                            DELETE FROM sync_queue
                            WHERE repo_id = ? AND status = 'pending' AND changed_files IS NULL
                        """, (repo_id,))
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


    def delete_job(self, job_id: int) -> bool:
        """Удаляет задачу из очереди."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM sync_queue WHERE id = ?", (job_id,))
            deleted = cursor.rowcount > 0
            self._auto_commit()
            return deleted


    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Возвращает значение настройки из таблицы app_settings."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else default


    def update_indexed_commit(self, repo_id: str, git_sha: str) -> None:
        """Обновляет хеш последнего проиндексированного коммита."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO indexed_commits (repo_id, git_sha, updated_at)
                VALUES (?, ?, datetime('now', 'localtime'))
            """, (repo_id, git_sha))
            self._auto_commit()


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


    def update_semantic_status(
        self,
        repo_id: str,
        status: str,
        project_id: str = "default",
        branch: str = "master",
        is_full: bool = False,
        error_message: Optional[str] = None
    ) -> None:
        """Обновляет статус семантической индексации Roslyn.
        
        Args:
            repo_id: ID репозитория
            status: 'pending' | 'running' | 'completed' | 'failed'
            project_id: ID проекта
            branch: Ветка
            is_full: True, если завершился полный прогон Roslyn
            error_message: Сообщение об ошибке (при status == 'failed')
        """
        with self.lock:
            cursor = self.conn.cursor()
            now_sql = "datetime('now', 'localtime')"
            
            cursor.execute("""
                SELECT retry_count FROM semantic_indexing_status
                WHERE repo_id = ? AND project_id = ? AND branch = ?
            """, (repo_id, project_id, branch))
            row = cursor.fetchone()

            if row:
                current_retry = row["retry_count"] or 0
                retry_count = current_retry + 1 if status == "failed" else (0 if status == "completed" else current_retry)
                if status == "completed":
                    if is_full:
                        cursor.execute(f"""
                            UPDATE semantic_indexing_status
                            SET status = ?, last_full_sync = {now_sql}, error_message = NULL, retry_count = 0, updated_at = {now_sql}
                            WHERE repo_id = ? AND project_id = ? AND branch = ?
                        """, (status, repo_id, project_id, branch))
                    else:
                        cursor.execute(f"""
                            UPDATE semantic_indexing_status
                            SET status = ?, last_delta_sync = {now_sql}, error_message = NULL, retry_count = 0, updated_at = {now_sql}
                            WHERE repo_id = ? AND project_id = ? AND branch = ?
                        """, (status, repo_id, project_id, branch))
                elif status == "failed":
                    cursor.execute(f"""
                        UPDATE semantic_indexing_status
                        SET status = ?, error_message = ?, retry_count = ?, updated_at = {now_sql}
                        WHERE repo_id = ? AND project_id = ? AND branch = ?
                    """, (status, error_message, retry_count, repo_id, project_id, branch))
                else:
                    cursor.execute(f"""
                        UPDATE semantic_indexing_status
                        SET status = ?, updated_at = {now_sql}
                        WHERE repo_id = ? AND project_id = ? AND branch = ?
                    """, (status, repo_id, project_id, branch))
            else:
                retry_count = 1 if status == "failed" else 0
                last_full = now_sql if (status == "completed" and is_full) else "NULL"
                last_delta = now_sql if (status == "completed" and not is_full) else "NULL"
                cursor.execute(f"""
                    INSERT INTO semantic_indexing_status (
                        repo_id, project_id, branch, status,
                        last_full_sync, last_delta_sync, error_message, retry_count, updated_at
                    ) VALUES (?, ?, ?, ?, {last_full}, {last_delta}, ?, ?, {now_sql})
                """, (repo_id, project_id, branch, status, error_message, retry_count))

            self._auto_commit()


    def add_to_queue(self, repo_id: str, changed_files: Optional[List[str]] = None) -> int:
        """Добавляет репозиторий в очередь на синхронизацию с дедупликацией.

        Если в очереди уже ожидает задача для этого repo_id:
        - Если новая задача - полная синхронизация (changed_files is None),
          то существующая pending задача обновляется до полной (changed_files = NULL).
        - Если новая задача точечная (со списком changed_files), и в очереди уже есть точечная,
          списки изменившихся файлов объединяются.
        - Если в очереди уже ожидает полная синхронизация, новая точечная задача поглощается ею.
        Возвращает ID задачи (существующей или вновь созданной).

        Args:
            repo_id: Идентификатор репозитория.
            changed_files: Опциональный список относительных путей измененных файлов из вебхука или хука.
        """
        import json
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, changed_files
                FROM sync_queue
                WHERE repo_id = ? AND status = 'pending'
                ORDER BY created_at ASC
                LIMIT 1
            """, (repo_id,))
            pending_job = cursor.fetchone()

            if pending_job:
                existing_id, existing_files_json = pending_job
                # Сценарий 1: Новая задача — полный sync (changed_files is None)
                if not changed_files:
                    if existing_files_json is not None:
                        cursor.execute("""
                            UPDATE sync_queue
                            SET changed_files = NULL,
                                updated_at = datetime('now', 'localtime')
                            WHERE id = ?
                        """, (existing_id,))
                        self._auto_commit()
                    return existing_id

                # Сценарий 2: Новая задача точечная, но в очереди уже ждет полный sync
                if existing_files_json is None:
                    return existing_id

                # Сценарий 3: Обе задачи точечные — объединяем множества файлов
                try:
                    existing_list = json.loads(existing_files_json) if existing_files_json else []
                except Exception:
                    existing_list = []
                merged = sorted(list(set(existing_list) | set(changed_files)))
                cursor.execute("""
                    UPDATE sync_queue
                    SET changed_files = ?,
                        updated_at = datetime('now', 'localtime')
                    WHERE id = ?
                """, (json.dumps(merged), existing_id))
                self._auto_commit()
                return existing_id

            # Если в pending задач для этого repo_id нет, вставляем новую
            changed_json = json.dumps(changed_files) if changed_files else None
            cursor.execute("""
                INSERT INTO sync_queue (repo_id, status, changed_files)
                VALUES (?, 'pending', ?)
            """, (repo_id, changed_json))
            self._auto_commit()
            return cursor.lastrowid


