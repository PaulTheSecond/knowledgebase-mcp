import os
import subprocess
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Set
from knowledge_mcp.core.git_manager import get_git_cmd_prefix

logger = logging.getLogger(__name__)

def is_subpath(child: Path, parent: Path) -> bool:
    """Проверяет, является ли child путем внутри parent (или тем же самым путем)."""
    try:
        if os.name == 'nt':
            c_parts = tuple(p.lower() for p in child.resolve().parts)
            p_parts = tuple(p.lower() for p in parent.resolve().parts)
            return len(c_parts) >= len(p_parts) and c_parts[:len(p_parts)] == p_parts
        else:
            c = child.resolve()
            p = parent.resolve()
            return c == p or c.is_relative_to(p)
    except Exception:
        return False

def is_same_path(p1: Path, p2: Path) -> bool:
    """Проверяет совпадение путей с учетом регистра на Windows."""
    try:
        if os.name == 'nt':
            return tuple(p.lower() for p in p1.resolve().parts) == tuple(p.lower() for p in p2.resolve().parts)
        return p1.resolve() == p2.resolve()
    except Exception:
        return False

@dataclass
class DiscoveredTarget:
    """Обнаруженная цель индексации: репозиторий, ворктри или документация."""
    project_id: str
    repo_id: str
    branch: str
    task_name: Optional[str]  # e.g. "master", "orbit-1035"
    absolute_path: Path
    is_base: bool  # True если это базовый клон (e.g. worktrees/master)
    is_doc: bool   # True если это markdown-документ или каталог документации (.kb, CLAUDE.md)
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация цели в словарь."""
        return {
            "project_id": self.project_id,
            "repo_id": self.repo_id,
            "branch": self.branch,
            "task_name": self.task_name,
            "absolute_path": self.absolute_path,
            "is_base": self.is_base,
            "is_doc": self.is_doc,
            "language": self.language,
        }

def resolve_workspace_root(workspace_root_val: Optional[str]) -> Optional[Path]:
    """Умное разрешение пути workspace_root с поддержкой кросс-платформенности (Windows хост <-> Linux Docker)."""
    if not workspace_root_val:
        return None
    try:
        p = Path(workspace_root_val)
        if p.exists() and p.is_dir():
            return p.resolve()
    except Exception:
        pass
        
    clean_ws = str(workspace_root_val).replace("\\", "/")
    if "repos/" in clean_ws.lower():
        rel_tail = clean_ws.lower().split("repos/", 1)[-1].lstrip("/")
        repos_dir_env = os.environ.get("REPOS_DIR")
        candidates = []
        if repos_dir_env:
            candidates.append(Path(repos_dir_env) / rel_tail)
        candidates.extend([Path("/repos") / rel_tail, Path("C:/Repos") / rel_tail, Path("c:/repos") / rel_tail])
        for cand in candidates:
            try:
                if cand.exists() and cand.is_dir():
                    return cand.resolve()
            except Exception:
                pass
    return None

class WorkspaceScanner:
    """Сканер воркспейсов, рабочих деревьев Git (worktrees) и документации."""

    @staticmethod
    def get_git_branch(repo_path: Path, default_branch: str = "master") -> str:
        """
        Быстро и надежно определяет ветку:
        1. Сначала прямое чтение файлов .git/HEAD или ссылки worktree (микросекунды).
        2. Fallback на git CLI rev-parse, если HEAD detached или файл не прочитан.
        """
        repo_path = Path(repo_path).resolve()

        # 1. Быстрое прямое чтение файлов .git / worktrees HEAD
        try:
            git_target = repo_path / ".git"
            git_dir = None
            if git_target.is_file():
                # Это worktree: файл .git содержит строку вида "gitdir: <path>"
                content = git_target.read_text(encoding="utf-8", errors="ignore").strip()
                if content.startswith("gitdir:"):
                    raw_gitdir = content[len("gitdir:"):].strip()
                    clean_raw = raw_gitdir.replace("\\", "/")
                    cand = (repo_path / raw_gitdir).resolve()
                    if cand.exists():
                        git_dir = cand
                    else:
                        # Кросс-платформенное сопоставление путей worktrees (Windows <-> Docker Linux)
                        if "worktrees/" in clean_raw.lower():
                            tail = clean_raw.lower().split("worktrees/", 1)[-1]
                            for parent in [repo_path] + list(repo_path.parents):
                                if parent.name.lower() == "worktrees":
                                    cand2 = (parent / tail).resolve()
                                    if cand2.exists():
                                        git_dir = cand2
                                        break
                                elif (parent / "worktrees").exists():
                                    cand3 = (parent / "worktrees" / tail).resolve()
                                    if cand3.exists():
                                        git_dir = cand3
                                        break
                        if not git_dir and "repos/" in clean_raw.lower():
                            rel_tail = clean_raw.lower().split("repos/", 1)[-1].lstrip("/")
                            for base in [Path("/repos"), Path("C:/Repos"), Path("c:/repos")]:
                                cand4 = (base / rel_tail).resolve()
                                if cand4.exists():
                                    git_dir = cand4
                                    break
            elif git_target.is_dir():
                git_dir = git_target

            if git_dir and git_dir.exists():
                head_file = git_dir / "HEAD"
                if head_file.exists():
                    head_content = head_file.read_text(encoding="utf-8", errors="ignore").strip()
                    if head_content.startswith("ref: refs/heads/"):
                        branch_name = head_content[len("ref: refs/heads/"):].strip()
                        if branch_name:
                            return branch_name
        except Exception as e:
            logger.debug(f"Ошибка прямого чтения .git для {repo_path}: {e}")

        # 2. Fallback на git CLI
        try:
            cmd = get_git_cmd_prefix(repo_path) + ["rev-parse", "--abbrev-ref", "HEAD"]
            proc = subprocess.run(
                cmd,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=3,
            )
            if proc.returncode == 0:
                branch = proc.stdout.strip()
                if branch and branch != "HEAD":
                    return branch
        except Exception as e:
            logger.debug(f"Git CLI недоступен или вернул ошибку для {repo_path}: {e}")

        return default_branch

    def scan_workspace(self, project_config: Dict[str, Any]) -> List[DiscoveredTarget]:
        """
        Сканирует проектный воркспейс:
        - Проверяет существование workspace_root.
        - Сканирует base_path (например, worktrees/master/*).
        - Сканирует tasks_path (например, worktrees/*, исключая base_path).
        - Сканирует docs_paths (например, .kb, CLAUDE.md).
        """
        project_id = project_config.get("id", "default")
        workspace_root_val = project_config.get("workspace_root")
        if not workspace_root_val:
            logger.warning(f"У проекта '{project_id}' не указан workspace_root.")
            return []

        workspace_root = resolve_workspace_root(workspace_root_val)
        if not workspace_root or not workspace_root.is_dir():
            logger.warning(f"Workspace root не существует или не является директорией: {workspace_root_val}")
            return []

        configured_repos = project_config.get("repositories", [])
        targets: List[DiscoveredTarget] = []
        seen_paths: Set[Path] = set()

        def _find_repo_spec(folder_name: str, rel_path: str = "") -> Optional[Dict[str, Any]]:
            for r in configured_repos:
                if r.get("path") in (folder_name, rel_path) or r.get("id") == folder_name:
                    return r
            return None

        # 1. Сканирование base_path (например, worktrees/master)
        base_rel = project_config.get("base_path", "worktrees/master")
        base_dir = (workspace_root / base_rel).resolve()
        base_task_name = Path(base_rel).name or "master"

        if base_dir.exists() and base_dir.is_dir():
            # Проверяем: возможно, сам base_dir является отдельным репозиторием
            has_child_repos = any(
                child.is_dir() and ((child / ".git").exists() or _find_repo_spec(child.name) is not None)
                for child in base_dir.iterdir()
            ) if base_dir.exists() else False

            if (base_dir / ".git").exists() and not has_child_repos:
                spec = _find_repo_spec(base_dir.name)
                repo_id = (spec.get("id") if spec else None) or (configured_repos[0].get("id") if configured_repos else project_id)
                language = (spec.get("language") if spec else None) or (configured_repos[0].get("language") if configured_repos else None)
                branch = self.get_git_branch(base_dir, default_branch=base_task_name)
                if base_dir not in seen_paths:
                    targets.append(DiscoveredTarget(
                        project_id=project_id,
                        repo_id=repo_id,
                        branch=branch,
                        task_name=base_task_name,
                        absolute_path=base_dir,
                        is_base=True,
                        is_doc=False,
                        language=language,
                    ))
                    seen_paths.add(base_dir)
            else:
                # Внутри base_dir лежат репозитории
                for child in sorted(base_dir.iterdir()):
                    if child.is_dir() and not child.name.startswith("."):
                        spec = _find_repo_spec(child.name)
                        if (child / ".git").exists() or spec is not None:
                            repo_id = (spec.get("id") if spec else None) or child.name
                            language = spec.get("language") if spec else None
                            branch = self.get_git_branch(child, default_branch=base_task_name)
                            if child not in seen_paths:
                                targets.append(DiscoveredTarget(
                                    project_id=project_id,
                                    repo_id=repo_id,
                                    branch=branch,
                                    task_name=base_task_name,
                                    absolute_path=child,
                                    is_base=True,
                                    is_doc=False,
                                    language=language,
                                ))
                                seen_paths.add(child)

                # Дополнительно проверяем сконфигурированные репозитории по их относительным путям
                for r in configured_repos:
                    r_path = r.get("path")
                    if r_path:
                        cand_dir = (base_dir / r_path).resolve()
                        if cand_dir.exists() and cand_dir not in seen_paths:
                            repo_id = r.get("id", r_path)
                            language = r.get("language")
                            branch = self.get_git_branch(cand_dir, default_branch=base_task_name)
                            targets.append(DiscoveredTarget(
                                project_id=project_id,
                                repo_id=repo_id,
                                branch=branch,
                                task_name=base_task_name,
                                absolute_path=cand_dir,
                                is_base=True,
                                is_doc=False,
                                language=language,
                            ))
                            seen_paths.add(cand_dir)

        # 2. Сканирование tasks_path (например, worktrees/{task} или worktrees/*)
        tasks_rel = project_config.get("tasks_path", "worktrees/{task}")
        if "{task}" in tasks_rel:
            prefix_str = tasks_rel.split("{task}")[0]
            tasks_parent = (workspace_root / prefix_str.rstrip("/\\")).resolve()
        elif "*" in tasks_rel:
            prefix_str = tasks_rel.split("*")[0]
            tasks_parent = (workspace_root / prefix_str.rstrip("/\\")).resolve()
        else:
            tasks_parent = (workspace_root / tasks_rel).resolve()

        if tasks_parent.exists() and tasks_parent.is_dir():
            for task_dir in sorted(tasks_parent.iterdir()):
                if not task_dir.is_dir():
                    continue
                if task_dir.name.startswith("."):
                    continue
                # Исключаем base_path из задач
                if task_dir.resolve() == base_dir.resolve():
                    continue

                task_name = task_dir.name

                has_child_repos = any(
                    child.is_dir() and ((child / ".git").exists() or _find_repo_spec(child.name) is not None)
                    for child in task_dir.iterdir()
                ) if task_dir.exists() else False

                if (task_dir / ".git").exists() and not has_child_repos:
                    spec = _find_repo_spec(task_dir.name)
                    repo_id = (spec.get("id") if spec else None) or (configured_repos[0].get("id") if configured_repos else project_id)
                    language = (spec.get("language") if spec else None) or (configured_repos[0].get("language") if configured_repos else None)
                    branch = self.get_git_branch(task_dir, default_branch=task_name)
                    if task_dir not in seen_paths:
                        targets.append(DiscoveredTarget(
                            project_id=project_id,
                            repo_id=repo_id,
                            branch=branch,
                            task_name=task_name,
                            absolute_path=task_dir,
                            is_base=False,
                            is_doc=False,
                            language=language,
                        ))
                        seen_paths.add(task_dir)
                else:
                    for child in sorted(task_dir.iterdir()):
                        if child.is_dir() and not child.name.startswith("."):
                            spec = _find_repo_spec(child.name)
                            if (child / ".git").exists() or spec is not None:
                                repo_id = (spec.get("id") if spec else None) or child.name
                                language = spec.get("language") if spec else None
                                branch = self.get_git_branch(child, default_branch=task_name)
                                if child not in seen_paths:
                                    targets.append(DiscoveredTarget(
                                        project_id=project_id,
                                        repo_id=repo_id,
                                        branch=branch,
                                        task_name=task_name,
                                        absolute_path=child,
                                        is_base=False,
                                        is_doc=False,
                                        language=language,
                                    ))
                                    seen_paths.add(child)

                    for r in configured_repos:
                        r_path = r.get("path")
                        if r_path:
                            cand_dir = (task_dir / r_path).resolve()
                            if cand_dir.exists() and cand_dir not in seen_paths:
                                repo_id = r.get("id", r_path)
                                language = r.get("language")
                                branch = self.get_git_branch(cand_dir, default_branch=task_name)
                                targets.append(DiscoveredTarget(
                                    project_id=project_id,
                                    repo_id=repo_id,
                                    branch=branch,
                                    task_name=task_name,
                                    absolute_path=cand_dir,
                                    is_base=False,
                                    is_doc=False,
                                    language=language,
                                ))
                                seen_paths.add(cand_dir)

        # 3. Сканирование docs_paths (например, .kb, CLAUDE.md)
        docs_paths = project_config.get("docs_paths", [])
        for doc_rel in docs_paths:
            doc_path = (workspace_root / doc_rel).resolve()
            if not doc_path.exists() and (base_dir / doc_rel).resolve().exists():
                doc_path = (base_dir / doc_rel).resolve()
            if not doc_path.exists():
                continue

            if doc_path.is_file():
                if doc_path not in seen_paths:
                    targets.append(DiscoveredTarget(
                        project_id=project_id,
                        repo_id="docs",
                        branch="master",
                        task_name=base_task_name,
                        absolute_path=doc_path,
                        is_base=True,
                        is_doc=True,
                        language="markdown",
                    ))
                    seen_paths.add(doc_path)
            elif doc_path.is_dir():
                md_files = sorted(list(doc_path.rglob("*.md")) + list(doc_path.rglob("*.markdown")))
                for md_file in md_files:
                    resolved_md = md_file.resolve()
                    if md_file.is_file() and resolved_md not in seen_paths:
                        targets.append(DiscoveredTarget(
                            project_id=project_id,
                            repo_id="docs",
                            branch="master",
                            task_name=base_task_name,
                            absolute_path=resolved_md,
                            is_base=True,
                            is_doc=True,
                            language="markdown",
                        ))
                        seen_paths.add(resolved_md)

        return targets

    @classmethod
    def resolve_active_target(
        cls,
        file_path: Union[str, Path],
        projects: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[DiscoveredTarget]:
        """
        По переданному абсолютному пути к файлу определяет, к какому проекту,
        репозиторию, ворктри/таске и ветке он относится.
        """
        file_path = Path(file_path).resolve()

        if projects is None:
            from knowledge_mcp.core.config import get_projects
            projects = get_projects()

        scanner = cls()
        candidates: List[DiscoveredTarget] = []

        for proj in projects:
            ws_root_val = proj.get("workspace_root")
            if not ws_root_val:
                continue
            ws_root = resolve_workspace_root(ws_root_val)
            if not ws_root or not is_subpath(file_path, ws_root):
                continue

            targets = scanner.scan_workspace(proj)
            for t in targets:
                if t.is_doc:
                    if t.absolute_path.is_file() and is_same_path(file_path, t.absolute_path):
                        candidates.append(t)
                    elif t.absolute_path.is_dir() and is_subpath(file_path, t.absolute_path):
                        candidates.append(t)
                else:
                    if is_subpath(file_path, t.absolute_path):
                        candidates.append(t)

            # Если файл не был найден в scan_workspace (например, новый doc файл в docs_paths)
            if not candidates:
                for doc_rel in proj.get("docs_paths", []):
                    doc_cand = (ws_root / doc_rel).resolve()
                    if is_subpath(file_path, doc_cand) and file_path.suffix.lower() in (".md", ".markdown"):
                        base_task_name = Path(proj.get("base_path", "worktrees/master")).name or "master"
                        candidates.append(DiscoveredTarget(
                            project_id=proj.get("id", "default"),
                            repo_id="docs",
                            branch="master",
                            task_name=base_task_name,
                            absolute_path=file_path,
                            is_base=True,
                            is_doc=True,
                            language="markdown",
                        ))

        if candidates:
            # Сортируем по максимальной специфичности пути (наиболее глубокая директория)
            candidates.sort(key=lambda t: len(t.absolute_path.resolve().parts), reverse=True)
            return candidates[0]

        # Fallback для стандартных одиночных репозиториев
        try:
            from knowledge_mcp.core.config import load_config
            cfg = load_config()
            for repo in cfg.get("repositories", []):
                if repo.get("local_path"):
                    r_path = Path(repo["local_path"]).resolve()
                    if is_subpath(file_path, r_path):
                        branch = repo.get("branch") or scanner.get_git_branch(r_path)
                        return DiscoveredTarget(
                            project_id=repo.get("project_id", "default"),
                            repo_id=repo.get("id", "default"),
                            branch=branch,
                            task_name=None,
                            absolute_path=r_path,
                            is_base=True,
                            is_doc=False,
                            language=repo.get("language"),
                        )
        except Exception as e:
            logger.debug(f"Fallback check for legacy repo failed: {e}")

        return None
