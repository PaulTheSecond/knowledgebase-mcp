"""
Модуль JIT (Just-In-Time) микросекундной синхронизации.
Проверяет актуальность рабочей копии через git status --porcelain
перед обработкой запросов MCP-агентов и обеспечивает мгновенную
переиндексацию отдельных файлов после хуков Edit|Write.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.core.indexer import Indexer

logger = logging.getLogger(__name__)


class JITSyncEngine:
    """Движок быстрой Just-In-Time синхронизации для AI-агентов."""

    def __init__(self, indexer: Optional[Indexer] = None):
        self._indexer = indexer

    def _get_indexer(self, db: KnowledgeDB) -> Indexer:
        if self._indexer is not None and self._indexer.db == db:
            return self._indexer
        return Indexer(db, use_embeddings=False)

    def ensure_fresh(
        self,
        target_path: Path | str,
        project_id: str,
        repo_id: str,
        branch: str,
        db: KnowledgeDB
    ) -> int:
        """
        Выполняет быстрый subprocess `git status --porcelain` в target_path (< 5 мс).
        Если статус чистый (нет вывода), возвращает 0 (ничего не изменилось).
        Если есть измененные (M), добавленные (??, A) или удаленные (D) файлы:
          - Для удаленных файлов вызывает db.delete_file(repo_id, rel_path, project_id, branch).
          - Для измененных/добавленных вызывает indexer.index_file() для каждого затронутого файла.
        Возвращает количество обновленных файлов.
        """
        target_path = Path(target_path).resolve()
        if not target_path.exists():
            logger.warning(f"JIT sync target path does not exist: {target_path}")
            return 0

        # Выполняем быстрый git status
        try:
            cmd = ["git", "status", "--porcelain", "-uall"]
            res = subprocess.run(
                cmd,
                cwd=str(target_path),
                capture_output=True,
                text=True,
                timeout=5.0
            )
            if res.returncode != 0:
                logger.debug(f"git status non-zero ({res.returncode}): {res.stderr}")
                return 0
            output = res.stdout.strip()
            if not output:
                return 0
        except Exception as e:
            logger.warning(f"Failed to execute git status in {target_path}: {e}")
            return 0

        indexer = self._get_indexer(db)
        updated_count = 0

        lines = output.splitlines()
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue

            status_code = line[:2]
            file_part = line[2:].strip()

            # Обработка переименования: R  old -> new
            if " -> " in file_part:
                old_path, new_path = file_part.split(" -> ", 1)
                old_path = old_path.strip().strip('"')
                new_path = new_path.strip().strip('"')
                db.delete_file(repo_id, old_path, project_id=project_id, branch=branch)
                full_new = target_path / new_path
                if full_new.is_file():
                    indexer.index_file(
                        full_new,
                        repo_id=repo_id,
                        project_id=project_id,
                        branch=branch,
                        rel_path=new_path,
                        repo_root=target_path
                    )
                    updated_count += 1
                continue

            rel_path = file_part.strip('"')
            full_path = target_path / rel_path

            # Удаление файла: статус содержит 'D'
            if 'D' in status_code:
                db.delete_file(repo_id, rel_path, project_id=project_id, branch=branch)
                updated_count += 1
            # Изменение или добавление: M, A, ??, etc.
            else:
                if full_path.is_file():
                    indexer.index_file(
                        full_path,
                        repo_id=repo_id,
                        project_id=project_id,
                        branch=branch,
                        rel_path=rel_path,
                        repo_root=target_path
                    )
                    updated_count += 1

        return updated_count

    def index_single_file(
        self,
        file_path: Path | str,
        project_id: str,
        repo_id: str,
        branch: str,
        db: KnowledgeDB,
        repo_root: Optional[Path | str] = None,
        rel_path: Optional[str] = None
    ) -> bool:
        """
        Метод для вызова из хуков Claude Code (PostToolUse на Edit|Write)
        и CLI-команды sync-file.
        Мгновенно переиндексирует ровно один измененный файл.
        """
        file_path = Path(file_path).resolve()
        indexer = self._get_indexer(db)

        if rel_path is None:
            if repo_root is not None:
                try:
                    rel_path = file_path.relative_to(Path(repo_root).resolve()).as_posix()
                except ValueError:
                    rel_path = file_path.name
            else:
                rel_path = file_path.name

        rel_path = rel_path.replace("\\", "/")

        try:
            if not file_path.exists():
                db.delete_file(repo_id, rel_path, project_id=project_id, branch=branch)
                return True

            if repo_root is not None:
                spec = indexer._get_ignore_spec(Path(repo_root))
                if spec.match_file(rel_path):
                    logger.info(f"[JIT] Skipped ignored file: {rel_path}")
                    return True

            file_id = indexer.index_file(
                file_path,
                repo_id=repo_id,
                project_id=project_id,
                branch=branch,
                rel_path=rel_path,
                repo_root=repo_root
            )
            return file_id > 0
        except Exception as e:
            logger.error(f"Failed to index single file {file_path}: {e}")
            return False
