"""
Базовый класс и интерфейс для компиляторных семантических обогатителей (ADR-009).
Реализует паттерны Strategy и Template Method для унификации семантического анализа
между различными компиляторными движками (Roslyn для C#, TypeScript Compiler API для TS/JS).
"""

import abc
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# TTL полного семантического сканирования по умолчанию (7 дней)
FULL_SYNC_TTL_DAYS = 7


class BaseSemanticEnricher(abc.ABC):
    """Абстрактный базовый класс компиляторного обогатителя семантики."""

    def __init__(self, db: Any):
        self.db = db

    @property
    @abc.abstractmethod
    def language_name(self) -> str:
        """Название языка для логов и статусов (например, 'c_sharp', 'typescript')."""
        pass

    @property
    @abc.abstractmethod
    def target_extensions(self) -> Set[str]:
        """Множество расширений файлов, обрабатываемых данным обогатителем."""
        pass

    @abc.abstractmethod
    def can_enrich(self, repo_path: Path) -> bool:
        """Проверяет, применим ли данный обогатитель к указанному репозиторию."""
        pass

    @abc.abstractmethod
    def run_analysis(
        self,
        repo_path: Path,
        target_files: Optional[List[Path]] = None
    ) -> Tuple[bool, Optional[str], Dict[Path, Dict[str, Any]]]:
        """
        Запускает компиляторный анализатор (CLI или встроенный).
        
        Возвращает:
            (success, error_message, parsed_cache),
            где parsed_cache: { Path: {'symbols': [...], 'edges': [...]} }
        """
        pass

    def is_stale(self, status_row: Optional[dict]) -> bool:
        """Проверяет, превысил ли возраст последнего полного скана TTL."""
        if not status_row:
            return True
        last_full = status_row.get("last_full_sync")
        if not last_full:
            return True
        try:
            if isinstance(last_full, str):
                dt = datetime.fromisoformat(last_full.replace(" ", "T"))
            else:
                dt = last_full
            age_days = (datetime.now() - dt).total_seconds() / 86400.0
            return age_days >= FULL_SYNC_TTL_DAYS
        except Exception as e:
            logger.warning(f"Failed parsing last_full_sync timestamp '{last_full}': {e}")
            return False

    def apply_to_db(
        self,
        parsed_cache: Dict[Path, Dict[str, Any]],
        repo_path: Path,
        repo_id: str,
        project_id: str = "default",
        branch: str = "master"
    ) -> int:
        """
        Универсальное применение извлеченных компилятором символов и рёбер в SQLite БД.
        Заменяет предварительные синтаксические символы Tree-Sitter, связывая новые
        символы с уже существующими чанками (без пересчета эмбеддингов).
        """
        if not parsed_cache:
            return 0

        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT id, path, content_hash FROM files
            WHERE repo_id = ? AND project_id = ? AND branch = ?
        """, (repo_id, project_id, branch))
        known_files = {row['path']: row for row in cursor.fetchall()}

        enriched_files_count = 0
        all_unresolved_records = []

        self.db.begin_transaction()
        try:
            for resolved_path, file_data in parsed_cache.items():
                try:
                    rel_path = resolved_path.relative_to(repo_path).as_posix()
                except ValueError:
                    rel_path = resolved_path.name

                file_row = known_files.get(rel_path)
                if not file_row:
                    continue

                file_id = file_row['id']
                content_hash = file_row['content_hash']

                cursor.execute("""
                    SELECT id, line_start, line_end FROM chunks
                    WHERE file_id = ?
                    ORDER BY line_start ASC
                """, (file_id,))
                file_chunks = cursor.fetchall()

                def find_chunk_id(line_start: int, line_end: int) -> Optional[str]:
                    if not file_chunks:
                        return None
                    for ch in file_chunks:
                        if ch['line_start'] <= line_start and ch['line_end'] >= line_end:
                            return ch['id']
                    best_id = None
                    best_overlap = -1
                    for ch in file_chunks:
                        overlap = min(ch['line_end'], line_end) - max(ch['line_start'], line_start)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_id = ch['id']
                    return best_id if best_overlap > 0 else file_chunks[0]['id']

                # Очищаем предварительные символы Tree-Sitter (каскадно удаляет связанные рёбра)
                self.db.clear_file_symbols(file_id)

                sym_records = [
                    (
                        file_id,
                        s['name'],
                        s['qualified_name'],
                        s['kind'],
                        s.get('language', self.language_name),
                        s['line_start'],
                        s['line_end'],
                        s['signature'],
                        find_chunk_id(s['line_start'], s['line_end']),
                        content_hash
                    )
                    for s in file_data.get('symbols', [])
                ]
                sym_rowids = self.db.add_symbols_batch(sym_records)

                ast_to_sym_id = {}
                for i, s in enumerate(file_data.get('symbols', [])):
                    if i < len(sym_rowids):
                        ast_to_sym_id[s['ast_node_id']] = sym_rowids[i]

                for e in file_data.get('edges', []):
                    src_ast_id = e.get('source_ast_id')
                    target_qname = e.get('target_qualified_name') or e.get('target_name')
                    kind = e.get('kind', 'CALLS')
                    src_sym_id = ast_to_sym_id.get(src_ast_id)
                    if src_sym_id and target_qname:
                        all_unresolved_records.append((src_sym_id, target_qname, kind))

                enriched_files_count += 1

            if all_unresolved_records:
                self.db.add_unresolved_refs_batch(all_unresolved_records)

            self.db.commit_transaction()
            logger.info(f"{self.language_name} semantic enrichment applied to {enriched_files_count} file(s).")
            return enriched_files_count
        except Exception as e:
            self.db.rollback_transaction()
            logger.error(f"Failed applying {self.language_name} semantic enrichment: {e}")
            raise

    def enrich(
        self,
        repo_path: Path,
        repo_id: str,
        project_id: str = "default",
        branch: str = "master",
        changed_files: Optional[List[Path]] = None,
        force: bool = False
    ) -> bool:
        """
        Шаблонный метод (Template Method) выполнения Фазы 2 семантического обогащения:
        1. Оценивает необходимость полного или дифференциального анализа.
        2. Отправляет статус в WebSocket.
        3. Запускает компилятор.
        4. Обогащает базу данных или выполняет graceful fallback при ошибке.
        """
        if not self.can_enrich(repo_path):
            return False

        status_row = self.db.get_semantic_status(repo_id, project_id=project_id, branch=branch)
        is_stale = self.is_stale(status_row)
        needs_full = force or (not status_row) or status_row.get("status") in ("pending", "failed") or is_stale

        # Фильтруем файлы, относящиеся к целевому языку
        target_exts = self.target_extensions
        target_changed = [
            f for f in (changed_files or [])
            if f.suffix.lower() in target_exts
        ]

        # Если полный анализ не требуется и нет измененных файлов этого языка — пропускаем
        if not needs_full and not target_changed:
            logger.info(f"Phase 2 ({self.language_name}): all files unchanged and index is fresh. Skipping.")
            return True

        from knowledge_mcp.daemon.status_ws import broadcast_sync

        if needs_full:
            logger.info(f"Phase 2 ({self.language_name}): Running full semantic analysis for '{repo_id}'...")
            broadcast_sync({
                "type": "sync_progress",
                "data": {
                    "repo_id": repo_id,
                    "progress": 85,
                    "status": f"{self.language_name}_analysis",
                    "message": f"Phase 2: Compiling {self.language_name} with compiler engine..."
                }
            })
            success, err_msg, cache = self.run_analysis(repo_path, target_files=None)
            if success:
                self.apply_to_db(cache, repo_path, repo_id, project_id, branch)
                self.db.update_semantic_status(repo_id, status='completed', project_id=project_id, branch=branch, is_full=True)
                logger.info(f"Phase 2 ({self.language_name}): Full analysis succeeded for '{repo_id}'.")
                return True
            else:
                self.db.update_semantic_status(repo_id, status='failed', project_id=project_id, branch=branch, error_message=err_msg)
                logger.warning(f"Phase 2 ({self.language_name}): Full analysis failed: {err_msg}. Repository remains operational on Tree-Sitter.")
                return False
        else:
            logger.info(f"Phase 2 ({self.language_name}): Running delta analysis for {len(target_changed)} file(s)...")
            broadcast_sync({
                "type": "sync_progress",
                "data": {
                    "repo_id": repo_id,
                    "progress": 85,
                    "status": f"{self.language_name}_analysis",
                    "message": f"Phase 2: Delta analysis on {len(target_changed)} {self.language_name} file(s)..."
                }
            })
            success, err_msg, cache = self.run_analysis(repo_path, target_files=target_changed)
            if success:
                self.apply_to_db(cache, repo_path, repo_id, project_id, branch)
                self.db.update_semantic_status(repo_id, status='completed', project_id=project_id, branch=branch, is_full=False)
                logger.info(f"Phase 2 ({self.language_name}): Delta analysis succeeded for '{repo_id}'.")
                return True
            else:
                self.db.update_semantic_status(repo_id, status='failed', project_id=project_id, branch=branch, error_message=err_msg)
                logger.warning(f"Phase 2 ({self.language_name}): Delta analysis failed: {err_msg}. Repository remains operational on Tree-Sitter.")
                return False
