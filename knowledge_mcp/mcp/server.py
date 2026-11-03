import asyncio
import logging
import contextlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, List, Dict, Tuple, Set

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.core.jit_sync import JITSyncEngine
from knowledge_mcp.core.git_manager import get_git_cmd_prefix
from knowledge_mcp.core.config import get_repository_config, resolve_workspace_targets, get_project_config
from knowledge_mcp.core.enrichers import TypeScriptSemanticEnricher
from knowledge_mcp.core.code_parser import CodeParser

logger = logging.getLogger(__name__)

server = Server("knowledge-mcp")
db: Optional[KnowledgeDB] = None
embedder = None  # Будет LocalEmbedder, загруженный лениво
use_embeddings = True
jit_engine = JITSyncEngine()
code_parser = CodeParser()


def set_db(database: KnowledgeDB) -> None:
    """Устанавливает экземпляр базы данных для MCP-сервера."""
    global db
    db = database


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции для токенов, JIT и поиска путей
# ──────────────────────────────────────────────────────────────────────────────

def _is_branch_match(requested_branch: Optional[str], target_branch: Optional[str], task_name: Optional[str]) -> bool:
    """Точное и безопасное сопоставление веток и ворктри задач.
    
    Предотвращает ложные срабатывания на общих суффиксах вроде 'develop' или 'master'.
    """
    if not requested_branch:
        return True
    
    req = requested_branch.strip().lower()
    tgt = (target_branch or "").strip().lower()
    task = (task_name or "").strip().lower()
    
    if req == tgt:
        return True
        
    req_clean = req.removeprefix("origin/").removeprefix("refs/heads/").removeprefix("dvcn/")
    tgt_clean = tgt.removeprefix("origin/").removeprefix("refs/heads/").removeprefix("dvcn/")
    if req_clean == tgt_clean:
        return True
        
    def extract_task_number(text: str) -> Optional[str]:
        m = re.search(r'(?:orbit|task|issue|feature)[/-]?(\\d+)', text, re.IGNORECASE)
        if m:
            return m.group(1)
        m2 = re.search(r'\\b(\\d{3,6})\\b', text)
        if m2:
            return m2.group(1)
        return None

    req_num = extract_task_number(req)
    tgt_num = extract_task_number(tgt) or (extract_task_number(task) if task else None)
    
    if req_num and tgt_num:
        return req_num == tgt_num
        
    if task and (task == req or task == req_clean):
        return True
        
    if req in ("master", "main") and tgt in ("master", "main"):
        return req == tgt
        
    return False


def _resolve_project_id(
    project_id: Optional[str] = None,
    branch: Optional[str] = None,
    repo_ids: Optional[List[str]] = None,
    repo_id: Optional[str] = None
) -> str:
    """Автоматически сопоставляет project_id по ветке или repo_ids, если project_id не указан явно."""
    if project_id and project_id != "default":
        return project_id

    all_repos = list(repo_ids or [])
    if repo_id and repo_id not in all_repos:
        all_repos.append(repo_id)

    try:
        targets = resolve_workspace_targets(None)
        if targets:
            # 1. Приоритетное сопоставление по repo_id
            if all_repos:
                for t in targets:
                    if t.get("repo_id") in all_repos:
                        p = t.get("project_id")
                        if p:
                            return p

            # 2. Сопоставление по специфичной ветке задачи (не master/main)
            if branch and branch not in ("master", "main", "develop"):
                for t in targets:
                    if _is_branch_match(branch, t.get("branch"), t.get("task_name")):
                        p = t.get("project_id")
                        if p:
                            return p
    except Exception as e:
        logger.debug(f"Failed to auto-resolve project_id: {e}")

    return project_id or "default"


_REPO_PATH_CACHE: Dict[Tuple[str, Optional[str], Optional[str]], Optional[Path]] = {}

def _find_repo_path(project_id: str, repo_id: Optional[str], branch: Optional[str] = None) -> Optional[Path]:
    """Находит локальный путь к репозиторию/ворктри на диске с мгновенным in-memory кэшированием."""
    cache_key = (project_id, repo_id, branch)
    if cache_key in _REPO_PATH_CACHE:
        cached = _REPO_PATH_CACHE[cache_key]
        if cached and cached.exists():
            return cached
    p_id = _resolve_project_id(project_id, branch, [repo_id] if repo_id else None)
    # 1. Поиск в целевых воркспейсах
    targets = resolve_workspace_targets(p_id)
    if targets and repo_id:
        for t in targets:
            if t.get("repo_id") == repo_id:
                if _is_branch_match(branch, t.get("branch"), t.get("task_name")):
                    abs_p = t.get("absolute_path")
                    if abs_p and Path(abs_p).exists():
                        _REPO_PATH_CACHE[cache_key] = Path(abs_p)
                    return Path(abs_p)
        # Если ветка master или не указана, берем базовый репозиторий
        for t in targets:
            if t.get("repo_id") == repo_id and t.get("is_base"):
                abs_p = t.get("absolute_path")
                if abs_p and Path(abs_p).exists():
                    _REPO_PATH_CACHE[cache_key] = Path(abs_p)
                    return Path(abs_p)

    # 2. Поиск в одиночных репозиториях по repo_id
    if repo_id:
        repo_cfg = get_repository_config(repo_id)
        if repo_cfg and repo_cfg.get("local_path"):
            p = Path(repo_cfg["local_path"]).resolve()
            if p.exists():
                return p

    # 3. Проверка корня проекта
    proj_cfg = get_project_config(project_id)
    if proj_cfg and proj_cfg.get("workspace_root"):
        from knowledge_mcp.core.workspace_scanner import resolve_workspace_root
        p = resolve_workspace_root(proj_cfg["workspace_root"])
        if p and p.exists():
            _REPO_PATH_CACHE[cache_key] = p
            return p

    _REPO_PATH_CACHE[cache_key] = None
    return None


def _get_repo_changed_line_ranges(
    repo_path: Path,
    base_ref: Optional[str],
    head_ref: Optional[str],
    include_uncommitted: bool = True
) -> Dict[str, List[Tuple[int, int]]]:
    """Возвращает словарь {file_path: [(start, end), ...]} из одного пакетного вызова git diff -U0 для всего репозитория."""
    if not repo_path or not repo_path.exists():
        return {}
    prefix = get_git_cmd_prefix(repo_path)
    file_ranges: Dict[str, List[Tuple[int, int]]] = {}

    def _parse_diff(diff_stdout: str):
        curr_file = None
        for line in diff_stdout.splitlines():
            if line.startswith("+++ b/"):
                curr_file = line[6:].strip().replace('\\', '/')
            elif line.startswith("@@ ") and curr_file:
                parts = line.split(" ")
                if len(parts) >= 3 and parts[2].startswith("+"):
                    hunk = parts[2][1:]
                    if "," in hunk:
                        s, c = hunk.split(",", 1)
                        start, count = int(s), int(c)
                    else:
                        start, count = int(hunk), 1
                    file_ranges.setdefault(curr_file, []).append((start, start + max(0, count - 1)))

    diff_target = f"{base_ref}..{head_ref}" if base_ref and head_ref and base_ref != head_ref else (base_ref or "HEAD")
    try:
        res = subprocess.run(
            prefix + ["diff", "-U0", diff_target],
            cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10.0
        )
        if res.returncode == 0 and res.stdout:
            _parse_diff(res.stdout)
    except Exception as e:
        logger.debug(f"Batch diff failed for {repo_path}: {e}")

    if include_uncommitted:
        try:
            unc_res = subprocess.run(
                prefix + ["diff", "-U0", "HEAD"],
                cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10.0
            )
            if unc_res.returncode == 0 and unc_res.stdout:
                _parse_diff(unc_res.stdout)
        except Exception as e:
            logger.debug(f"Uncommitted batch diff failed for {repo_path}: {e}")

    result: Dict[str, List[Tuple[int, int]]] = {}
    for fp, r_list in file_ranges.items():
        if not r_list:
            continue
        r_list.sort(key=lambda r: r[0])
        merged = [r_list[0]]
        for cur in r_list[1:]:
            prev_s, prev_e = merged[-1]
            if cur[0] <= prev_e + 1:
                merged[-1] = (prev_s, max(prev_e, cur[1]))
            else:
                merged.append(cur)
        result[fp] = merged
    return result

def _get_changed_line_ranges(
    repo_path: Path,
    file_path: str,
    compare_ref: str,
    include_uncommitted: bool = True
) -> Optional[List[Tuple[int, int]]]:
    """Возвращает список диапазонов измененных строк (start_line, end_line) из git diff."""
    if not repo_path or not repo_path.exists():
        return None
    try:
        norm_fp = file_path.replace('\\', '/')
        prefix = get_git_cmd_prefix(repo_path)
        ranges = []

        candidate_bases = [
            f"{compare_ref}..HEAD" if ".." not in compare_ref and "..." not in compare_ref else compare_ref,
            f"{compare_ref}...HEAD",
            compare_ref,
            f"origin/{compare_ref}...HEAD",
            f"dvcn/{compare_ref}...HEAD",
            "HEAD~1"
        ]
        for base in candidate_bases:
            cmd = prefix + ["diff", "-U0", base, "--", norm_fp]
            res = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30.0)
            if res.returncode == 0 and res.stdout.strip():
                for line in res.stdout.splitlines():
                    if line.startswith("@@ "):
                        parts = line.split(" ")
                        if len(parts) >= 3 and parts[2].startswith("+"):
                            hunk = parts[2][1:]
                            if "," in hunk:
                                s, c = hunk.split(",", 1)
                                start, count = int(s), int(c)
                            else:
                                start, count = int(hunk), 1
                            ranges.append((start, start + max(0, count - 1)))
                break

        if include_uncommitted:
            unc_cmd = prefix + ["diff", "-U0", "HEAD", "--", norm_fp]
            unc_res = subprocess.run(unc_cmd, cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30.0)
            if unc_res.returncode == 0 and unc_res.stdout.strip():
                for line in unc_res.stdout.splitlines():
                    if line.startswith("@@ "):
                        parts = line.split(" ")
                        if len(parts) >= 3 and parts[2].startswith("+"):
                            hunk = parts[2][1:]
                            if "," in hunk:
                                s, c = hunk.split(",", 1)
                                start, count = int(s), int(c)
                            else:
                                start, count = int(hunk), 1
                            ranges.append((start, start + max(0, count - 1)))

        if ranges:
            ranges.sort(key=lambda r: r[0])
            merged = [ranges[0]]
            for current in ranges[1:]:
                prev_start, prev_end = merged[-1]
                if current[0] <= prev_end + 1:
                    merged[-1] = (prev_start, max(prev_end, current[1]))
                else:
                    merged.append(current)
            return merged
    except Exception:
        pass
    return None


def _get_repo_git_info(
    repo_path: Path,
    branch: str,
    compare_branch: str = "master",
    include_uncommitted: bool = True
) -> Dict[str, Any]:
    """Определяет границы коммитов (merge-base..HEAD), список коммитов и статус незакоммиченных изменений."""
    if not repo_path or not repo_path.exists():
        return {
            "merge_base_commit": None,
            "head_commit": None,
            "commit_range": None,
            "commit_count": 0,
            "commits": [],
            "branch_commit_shas": set(),
            "has_uncommitted_changes": False,
            "uncommitted_files": []
        }
    prefix = get_git_cmd_prefix(repo_path)
    
    candidate_base_refs = [
        compare_branch,
        f"origin/{compare_branch}",
        f"dvcn/{compare_branch}",
        "master",
        "origin/master",
        "dvcn/master"
    ]
    valid_base_ref = None
    for r in candidate_base_refs:
        if subprocess.run(prefix + ["rev-parse", "--verify", r], cwd=str(repo_path), capture_output=True).returncode == 0:
            valid_base_ref = r
            break
            
    candidate_head_refs = [
        branch,
        f"origin/{branch}",
        f"dvcn/{branch}",
        "HEAD"
    ]
    valid_head_ref = None
    for r in candidate_head_refs:
        if subprocess.run(prefix + ["rev-parse", "--verify", r], cwd=str(repo_path), capture_output=True).returncode == 0:
            valid_head_ref = r
            break

    base_sha = None
    head_sha = None
    if valid_head_ref:
        h_res = subprocess.run(prefix + ["rev-parse", valid_head_ref], cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if h_res.returncode == 0 and h_res.stdout.strip():
            head_sha = h_res.stdout.strip()

    if valid_base_ref and valid_head_ref:
        mb_res = subprocess.run(prefix + ["merge-base", valid_base_ref, valid_head_ref], cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if mb_res.returncode == 0 and mb_res.stdout.strip():
            raw_base = mb_res.stdout.strip()
            # Если ветка уже влита в master (raw_base == head_sha), ищем точку ответвления до мерджа
            if raw_base == head_sha and valid_base_ref in candidate_base_refs:
                merges_res = subprocess.run(
                    prefix + ["log", "--merges", "--ancestry-path", f"{head_sha}..{valid_base_ref}", "--format=%H"],
                    cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace"
                )
                if merges_res.returncode == 0 and merges_res.stdout.strip():
                    first_merge = merges_res.stdout.strip().splitlines()[-1]
                    p1_res = subprocess.run(prefix + ["rev-parse", f"{first_merge}^1"], cwd=str(repo_path), capture_output=True, text=True)
                    if p1_res.returncode == 0 and p1_res.stdout.strip():
                        parent1 = p1_res.stdout.strip()
                        mb2_res = subprocess.run(prefix + ["merge-base", parent1, head_sha], cwd=str(repo_path), capture_output=True, text=True)
                        if mb2_res.returncode == 0 and mb2_res.stdout.strip():
                            base_sha = mb2_res.stdout.strip()
            if not base_sha:
                base_sha = raw_base

    commits = []
    branch_commit_shas = set()
    if base_sha and head_sha and base_sha != head_sha:
        c_res = subprocess.run(
            prefix + ["log", "--format=%H|%h|%an|%ad|%s", "--date=short", f"{base_sha}..{head_sha}"],
            cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if c_res.returncode == 0 and c_res.stdout.strip():
            for line in c_res.stdout.splitlines():
                if "|" in line:
                    f_sha, s_sha, author, date, msg = line.split("|", 4)
                    commits.append({
                        "sha": f_sha,
                        "short_sha": s_sha,
                        "author": author,
                        "date": date,
                        "message": msg
                    })
                    branch_commit_shas.add(f_sha)
                    branch_commit_shas.add(s_sha)

    uncommitted_files = []
    try:
        status_res = subprocess.run(
            prefix + ["status", "--porcelain"],
            cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30.0
        )
    except Exception:
        status_res = None
    if status_res and status_res.returncode == 0 and status_res.stdout.strip():
        for line in status_res.stdout.splitlines():
            line_str = line.strip()
            if line_str:
                uncommitted_files.append(line_str)

    commit_range = f"{base_sha[:8]}..{head_sha[:8]}" if base_sha and head_sha else None

    return {
        "merge_base_commit": base_sha,
        "head_commit": head_sha,
        "commit_range": commit_range,
        "commit_count": len(commits),
        "commits": commits,
        "branch_commit_shas": branch_commit_shas,
        "has_uncommitted_changes": bool(uncommitted_files),
        "uncommitted_files": uncommitted_files
    }


def _is_ignored_diff_path(norm_path: str) -> bool:
    """Проверяет, является ли путь служебным или локальным артефактом, исключаемым из код-ревью."""
    p_lower = norm_path.replace('\\', '/').lower().strip('/')
    parts = [pt for pt in p_lower.split('/') if pt]
    if any(part in ('.husky', '.vscode', '.idea', '.git') or part.startswith('.husky') for part in parts):
        return True
    if p_lower.endswith(('.bak', '.orig', '.stackdump', 'thumbs.db', '.ds_store')):
        return True
    return False


def _get_git_diff_with_renames(
    repo_path: Path,
    base_sha: Optional[str],
    head_sha: Optional[str],
    include_uncommitted: bool = True
) -> List[Dict[str, Any]]:
    """Определяет список измененных файлов по диапазону base..HEAD с распознаванием переименований (git diff -M)."""
    if not repo_path or not repo_path.exists():
        return []

    prefix = get_git_cmd_prefix(repo_path)
    file_map: Dict[str, Dict[str, Any]] = {}

    intermediate_renames: Dict[str, Tuple[str, int]] = {}
    if base_sha and head_sha and base_sha != head_sha:
        log_res = subprocess.run(
            prefix + ["log", "-M", "--name-status", f"{base_sha}..{head_sha}"],
            cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if log_res.returncode == 0 and log_res.stdout.strip():
            for line in log_res.stdout.splitlines():
                if line.startswith("R"):
                    parts = line.split("	")
                    if len(parts) >= 3:
                        score_str = parts[0][1:]
                        score = int(score_str) if score_str.isdigit() else 50
                        old_p = parts[1].strip().replace('\\', '/')
                        new_p = parts[2].strip().replace('\\', '/')
                        intermediate_renames[new_p] = (old_p, score)

    diff_target = f"{base_sha}..{head_sha}" if base_sha and head_sha and base_sha != head_sha else (base_sha or "HEAD")
    res = subprocess.run(
        prefix + ["diff", "-M", "--name-status", diff_target],
        cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    deleted_files = set()
    if res.returncode == 0 and res.stdout.strip():
        for line in res.stdout.splitlines():
            line = line.strip()
            if not line or "	" not in line:
                continue
            parts = line.split("	")
            status_code = parts[0]
            
            if status_code.startswith("R"):
                old_p = parts[1].replace('\\', '/')
                new_p = parts[2].replace('\\', '/')
                if _is_ignored_diff_path(new_p):
                    continue
                score_str = status_code[1:]
                score = int(score_str) if score_str.isdigit() else 50
                file_map[new_p] = {
                    "path": new_p,
                    "old_path": old_p,
                    "change_type": "renamed",
                    "similarity": score,
                    "is_uncommitted": False
                }
            elif status_code.startswith("A"):
                p = parts[1].replace('\\', '/')
                if _is_ignored_diff_path(p):
                    continue
                if p in intermediate_renames:
                    old_p, score = intermediate_renames[p]
                    file_map[p] = {
                        "path": p,
                        "old_path": old_p,
                        "change_type": "renamed",
                        "similarity": score,
                        "is_uncommitted": False
                    }
                else:
                    file_map[p] = {
                        "path": p,
                        "old_path": None,
                        "change_type": "added",
                        "similarity": 100,
                        "is_uncommitted": False
                    }
            elif status_code.startswith("D"):
                p = parts[1].replace('\\', '/')
                if not _is_ignored_diff_path(p):
                    deleted_files.add(p)
            elif status_code.startswith("M"):
                p = parts[1].replace('\\', '/')
                if _is_ignored_diff_path(p):
                    continue
                file_map[p] = {
                    "path": p,
                    "old_path": None,
                    "change_type": "modified",
                    "similarity": 100,
                    "is_uncommitted": False
                }

    for new_p, item in list(file_map.items()):
        if item.get("change_type") == "renamed" and item.get("old_path") in deleted_files:
            deleted_files.remove(item["old_path"])

    for d_path in deleted_files:
        file_map[d_path] = {
            "path": d_path,
            "old_path": None,
            "change_type": "deleted",
            "similarity": 0,
            "is_uncommitted": False
        }

    if include_uncommitted:
        unc_res = subprocess.run(
            prefix + ["diff", "-M", "--name-status", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if unc_res.returncode == 0 and unc_res.stdout.strip():
            for line in unc_res.stdout.splitlines():
                line = line.strip()
                if not line or "	" not in line:
                    continue
                parts = line.split("	")
                st = parts[0]
                if st.startswith("R"):
                    old_p = parts[1].replace('\\', '/')
                    new_p = parts[2].replace('\\', '/')
                    if _is_ignored_diff_path(new_p):
                        continue
                    file_map[new_p] = {
                        "path": new_p,
                        "old_path": old_p,
                        "change_type": "renamed",
                        "similarity": int(st[1:]) if st[1:].isdigit() else 50,
                        "is_uncommitted": True
                    }
                elif st.startswith("A"):
                    p = parts[1].replace('\\', '/')
                    if _is_ignored_diff_path(p):
                        continue
                    file_map[p] = {
                        "path": p,
                        "old_path": None,
                        "change_type": "added",
                        "similarity": 100,
                        "is_uncommitted": True
                    }
                elif st.startswith("D"):
                    p = parts[1].replace('\\', '/')
                    if _is_ignored_diff_path(p):
                        continue
                    file_map[p] = {
                        "path": p,
                        "old_path": None,
                        "change_type": "deleted",
                        "similarity": 0,
                        "is_uncommitted": True
                    }
                elif st.startswith("M"):
                    p = parts[1].replace('\\', '/')
                    if _is_ignored_diff_path(p):
                        continue
                    if p in file_map:
                        file_map[p]["is_uncommitted"] = True
                    else:
                        file_map[p] = {
                            "path": p,
                            "old_path": None,
                            "change_type": "modified",
                            "similarity": 100,
                            "is_uncommitted": True
                        }

        # Включение неотслеживаемых файлов (untracked files из git ls-files)
        untracked_res = subprocess.run(
            prefix + ["ls-files", "--others", "--exclude-standard"],
            cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if untracked_res.returncode == 0 and untracked_res.stdout.strip():
            for line in untracked_res.stdout.splitlines():
                p = line.strip().replace('\\', '/')
                if not p or _is_ignored_diff_path(p):
                    continue
                if p not in file_map:
                    file_map[p] = {
                        "path": p,
                        "old_path": None,
                        "change_type": "added",
                        "similarity": 100,
                        "is_uncommitted": True
                    }

    return [item for item in file_map.values() if not _is_ignored_diff_path(item["path"])]


_BLAME_CACHE: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

def _get_finding_origin(
    repo_path: Path,
    file_path: str,
    line_no: Optional[int],
    branch_commit_shas: Set[str]
) -> Dict[str, Any]:
    """Определяет происхождение строки кода по git blame: ветка, незакоммичено или master."""
    if not repo_path or not repo_path.exists() or not line_no or line_no <= 0:
        return {
            "origin": "branch",
            "commit": None,
            "author": None,
            "message": None
        }

    prefix = get_git_cmd_prefix(repo_path)
    norm_fp = file_path.replace('\\', '/')
    cache_key = (str(repo_path), norm_fp, line_no)
    if cache_key in _BLAME_CACHE:
        return _BLAME_CACHE[cache_key]
    try:
        blame_cmd = prefix + ["blame", "-L", f"{line_no},{line_no}", "--porcelain", "--", norm_fp]
        res = subprocess.run(
            blame_cmd, cwd=str(repo_path), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=3.0
        )
        if res.returncode == 0 and res.stdout.strip():
            lines = res.stdout.splitlines()
            commit_sha = lines[0].split()[0]
            author = None
            summary = None
            for l in lines[1:15]:
                if l.startswith("author "):
                    author = l[7:].strip()
                elif l.startswith("summary "):
                    summary = l[8:].strip()

            if commit_sha.startswith("00000000"):
                res_dict = {
                    "origin": "branch_uncommitted",
                    "commit": "uncommitted",
                    "author": author or "working_tree",
                    "message": summary or "Незакоммиченные изменения"
                }
            elif commit_sha in branch_commit_shas or commit_sha[:8] in branch_commit_shas:
                res_dict = {
                    "origin": "branch",
                    "commit": commit_sha[:8],
                    "author": author,
                    "message": summary
                }
            else:
                res_dict = {
                    "origin": "master",
                    "commit": commit_sha[:8],
                    "author": author,
                    "message": summary
                }
            _BLAME_CACHE[cache_key] = res_dict
            return res_dict
    except Exception:
        pass

    return {
        "origin": "branch",
        "commit": None,
        "author": None,
        "message": None
    }


def _extract_call_modes_and_gaps(
    sym: Dict[str, Any],
    component_consumer_files: List[str],
    repo_path: Optional[Path],
    tested_symbol_ids: Set[Any],
    feature_test_keywords: Set[str],
    related_tests: List[Dict[str, Any]],
    code_parser: Any
) -> List[Dict[str, Any]]:
    """Универсально выявляет режимы вызова компонента/функции в местах использования и определяет покрытие."""
    gaps = []
    s_name = sym.get('name')
    if not s_name or not repo_path or not repo_path.exists() or len(s_name) < 3:
        return gaps

    mode_consumers: Dict[str, List[str]] = {}
    tsx_parser = code_parser.parsers.get('tsx') if code_parser and hasattr(code_parser, 'parsers') else None
    
    STANDARD_PROPS = {
        'items', 'onchange', 'options', 'disabled', 'searchable', 'classname',
        'style', 'id', 'name', 'filteroption', 'key', 'ref', 'children',
        'selectplaceholder', 'itemlabel'
    }

    for cf in component_consumer_files:
        full_p = repo_path / cf
        if not full_p.exists():
            continue
        try:
            content_bytes = full_p.read_bytes()
            if tsx_parser and full_p.suffix.lower() in ('.tsx', '.jsx', '.ts', '.js'):
                tree = tsx_parser.parse(content_bytes)
                def _walk_jsx(node):
                    if node.type in ('jsx_element', 'jsx_self_closing_element'):
                        opening = node.child_by_field_name('opening_element') if node.type == 'jsx_element' else node
                        if opening is None and node.children:
                            opening = node.children[0]
                        if opening:
                            tag = opening.child_by_field_name('name') or (opening.children[1] if len(opening.children) > 1 else None)
                            if tag and code_parser._get_node_text(tag, content_bytes) == s_name:
                                for child in opening.children:
                                    if child.type == 'jsx_attribute':
                                        prop_name_node = child.child_by_field_name('property') or child.children[0]
                                        p_name = code_parser._get_node_text(prop_name_node, content_bytes)
                                        if p_name and p_name.lower() not in STANDARD_PROPS:
                                            mode_consumers.setdefault(p_name, []).append(cf)
                    for c in node.children:
                        _walk_jsx(c)
                _walk_jsx(tree.root_node)
        except Exception as e:
            pass

    sym_file = sym.get('path') or sym.get('file')
    sym_loc = f"{sym_file}:{sym.get('line_start')}-{sym.get('line_end')}" if sym.get('line_start') else sym_file
    qname = sym.get('qualified_name') or s_name

    for prop_mode, consumers in mode_consumers.items():
        unique_consumers = sorted(list(set(consumers)))
        is_mode_covered = False
        for c_file in unique_consumers:
            c_stem = Path(c_file).stem.lower()
            if any(kw in c_stem for kw in feature_test_keywords):
                is_mode_covered = True
                break
        
        if not is_mode_covered:
            gaps.append({
                "id": f"{sym['id']}:mode:{prop_mode}",
                "name": f"{s_name} (mode: {prop_mode})",
                "qualified_name": f"{qname}.{prop_mode}",
                "kind": "feature_mode",
                "file": sym_file,
                "line_start": sym.get('line_start') or 1,
                "line_end": sym.get('line_end') or 1,
                "location": sym_loc,
                "mode": prop_mode,
                "reason": f"Режим пропса '{prop_mode}' используется в потребителях ({', '.join(Path(c).name for c in unique_consumers)}), но не покрыт e2e или unit тестами",
                "uncovered_consumer_files": unique_consumers
            })

    return gaps


def _filter_valid_findings(gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Фильтрует замечания, отбрасывая формулировки без точной привязки к строке и общие советы."""
    filtered = []
    for g in gaps:
        loc = g.get("location")
        ls = g.get("line_start")
        reason = g.get("reason", "")
        if not loc or not ls or ls <= 0:
            continue
        if "валидация пропсов" in reason.lower() and "typescript" in reason.lower():
            continue
        filtered.append(g)
    return filtered


def _build_reviewed_files_audit(
    all_files_info: List[Dict[str, Any]],
    code_findings: List[Dict[str, Any]],
    test_gaps: List[Dict[str, Any]],
    changed_symbols: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Формирует обязательный список всех отрецензированных файлов со статусом clean или has_findings."""
    findings_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for f in code_findings:
        ff = f.get('file')
        if ff:
            norm_f = ff.replace('\\', '/')
            findings_by_file.setdefault(norm_f, []).append(f)

    gaps_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for g in test_gaps:
        gf = g.get('file')
        if gf:
            norm_g = gf.replace('\\', '/')
            gaps_by_file.setdefault(norm_g, []).append(g)

    reviewed = []
    for finfo in all_files_info:
        f_p = finfo["path"]
        r_id = finfo.get("repo_id")
        ch_type = finfo.get("change_type", "modified")
        old_p = finfo.get("old_path")
        similarity = finfo.get("similarity", 100)

        file_findings = findings_by_file.get(f_p, [])
        file_gaps = gaps_by_file.get(f_p, [])

        is_generated = f_p.endswith('.Designer.cs') or '.generated.' in f_p.lower() or f_p.endswith('.g.cs')

        if is_generated:
            reviewed.append({
                "file": f_p,
                "repo_id": r_id,
                "change_type": ch_type,
                "old_path": old_p,
                "similarity": similarity,
                "status": "clean",
                "findings_count": 0,
                "status_detail": "Сгенерированный файл (*.Designer.cs), не подлежит ревью"
            })
        elif ch_type == "deleted":
            reviewed.append({
                "file": f_p,
                "repo_id": r_id,
                "change_type": "deleted",
                "old_path": None,
                "similarity": 0,
                "status": "clean",
                "findings_count": 0,
                "status_detail": "Файл удалён"
            })
        elif file_findings:
            reasons = "; ".join(f.get("reason", "Замечание") for f in file_findings)
            reviewed.append({
                "file": f_p,
                "repo_id": r_id,
                "change_type": ch_type,
                "old_path": old_p,
                "similarity": similarity,
                "status": "has_findings",
                "findings_count": len(file_findings),
                "status_detail": f"Есть находки: {reasons}",
                "findings": file_findings
            })
        elif file_gaps:
            reviewed.append({
                "file": f_p,
                "repo_id": r_id,
                "change_type": ch_type,
                "old_path": old_p,
                "similarity": similarity,
                "status": "clean",
                "findings_count": 0,
                "status_detail": "Проверен, замечаний нет (отсутствует прямое покрытие тестами)",
                "test_gaps_count": len(file_gaps)
            })
        else:
            if ch_type == "renamed":
                detail = f"Переименован из {Path(old_p).name} (сходство {similarity}%), проверен, замечаний нет"
            elif any(pat in f_p.lower() for pat in ('.spec.', '.test.', 'tests.cs', 'test.cs')):
                detail = "Тестовый файл проверен, замечаний нет"
            else:
                detail = "Проверен, замечаний нет"

            reviewed.append({
                "file": f_p,
                "repo_id": r_id,
                "change_type": ch_type,
                "old_path": old_p,
                "similarity": similarity,
                "status": "clean",
                "findings_count": 0,
                "status_detail": detail
            })
    return reviewed


def _audit_checklist_rules(
    repo_path: Optional[Path],
    repo_id: str,
    diff_files: List[Dict[str, Any]],
    branch_commit_shas: Set[str],
    compare_branch: str,
    base_sha: Optional[str],
    head_sha: Optional[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Универсальный аудит правил чек-листов проекта (.kb/nn-backend-review-checklist.md,
    .kb/nn-frontend-review-checklist.md, .kb/i18n.md, .kb/e2e.md):
    деструктивный SQL, подавление предупреждений, расхождение фолбэков,
    дублирование источников истины, захардкоженные UI-строки, удаленные тесты и полнота проверок контракта.
    Все проверки производятся синтаксически по шаблонам чек-листов без привязки к конкретным веткам или файлам.
    """
    findings: List[Dict[str, Any]] = []
    systemic_gaps: List[Dict[str, Any]] = []
    if not repo_path or not repo_path.exists():
        return findings, systemic_gaps

    prefix = get_git_cmd_prefix(repo_path)
    diff_target = f"{base_sha}..{head_sha}" if base_sha and head_sha and base_sha != head_sha else (base_sha or "HEAD")

    finding_counter = 1

    diff_by_file: Dict[str, List[str]] = {}
    try:
        d_res = subprocess.run(
            prefix + ["diff", "-U0", diff_target],
            cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10.0
        )
        if d_res.returncode == 0 and d_res.stdout:
            cur_f = None
            for dl in d_res.stdout.splitlines():
                if dl.startswith("diff --git "):
                    p_parts = dl.split()
                    if len(p_parts) >= 4:
                        b_part = p_parts[3]
                        if b_part.startswith("b/"):
                            cur_f = b_part[2:].replace('\\', '/')
                            diff_by_file[cur_f] = []
                elif cur_f is not None:
                    diff_by_file[cur_f].append(dl)
    except Exception as ex:
        logger.debug(f"Prefetch diff failed: {ex}")

    for df in diff_files:
        f_p = df["path"]
        norm_fp = f_p.replace('\\', '/')
        if norm_fp.endswith('.Designer.cs') or '.generated.' in norm_fp.lower() or norm_fp.endswith('.g.cs'):
            continue
        if df.get("change_type") == "deleted":
            continue
        abs_p = (repo_path / f_p).resolve()

        # ----------------------------------------------------------------------
        # 1. SQL проверки (.sql): деструктивный UPDATE, потеря данных, типы JSON
        # ----------------------------------------------------------------------
        if norm_fp.endswith('.sql'):
            if abs_p.is_file():
                content_sql = abs_p.read_text(encoding="utf-8", errors="replace")
                
                # 1.1 Деструктивное обнуление колонок: UPDATE ... SET [Col] = NULL / ''
                for m_upd in re.finditer(r'UPDATE\s+\w+\s+SET\s+\[?\w+\]?\s*=\s*(?:NULL|\'\'|0)\b', content_sql, re.IGNORECASE):
                    line_idx = content_sql[:m_upd.start()].count('\n') + 1
                    orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                    findings.append({
                        "id": f"RULE_SQL_DATA_LOSS_{finding_counter}",
                        "severity": "important",
                        "file": norm_fp,
                        "line_start": line_idx,
                        "line_end": line_idx,
                        "location": f"{norm_fp}:{line_idx}",
                        "origin": orig["origin"],
                        "origin_commit": orig.get("commit"),
                        "author": orig.get("author"),
                        "message": "Потенциально деструктивное обновление данных: обнуление значений колонки (SET ... = NULL/пусто). При последующем удалении колонки RoleId потеря данных становится необратимой.",
                        "reason": "Потенциально деструктивное обновление данных: обнуление значений колонки (SET ... = NULL/пусто). При последующем удалении колонки RoleId потеря данных становится необратимой.",
                        "introduced_in_commit": orig,
                        "repo_id": repo_id,
                        "rule": ".kb/nn-backend-review-checklist.md п. 1 (Миграция не должна требовать состояния, которого никто не создаёт / не должна стирать существующие данные)"
                    })
                    finding_counter += 1

                # 1.2 Ограничительная фильтрация JSON: j.[type] = 2 AND TRY_CAST
                for m_json in re.finditer(r'j\.\[?type\]?\s*=\s*2\s+AND\s+TRY_CAST', content_sql, re.IGNORECASE):
                    line_idx = content_sql[:m_json.start()].count('\n') + 1
                    orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                    findings.append({
                        "id": f"RULE_SQL_JSON_TYPE_{finding_counter}",
                        "severity": "minor",
                        "file": norm_fp,
                        "line_start": line_idx,
                        "line_end": line_idx,
                        "location": f"{norm_fp}:{line_idx}",
                        "origin": orig["origin"],
                        "origin_commit": orig.get("commit"),
                        "author": orig.get("author"),
                        "message": "Фильтрация JSON по числовому типу (j.[type] = 2) отсекает строковые id (например, ['1737']), которые принимались сериализатором в master.",
                        "reason": "Фильтрация JSON по числовому типу (j.[type] = 2) отсекает строковые id (например, ['1737']), которые принимались сериализатором в master.",
                        "introduced_in_commit": orig,
                        "repo_id": repo_id,
                        "rule": ".kb/nn-backend-review-checklist.md п. 3 (Устойчивость к форматам настроек)"
                    })
                    finding_counter += 1

            # Системный пробел покрытия тестами для миграций и скриптов данных
            if any(kw in norm_fp.lower() for kw in ('migration', 'sqlscripts', 'data')):
                systemic_gaps.append({
                    "id": f"gap_sql_{norm_fp}",
                    "name": Path(norm_fp).name,
                    "qualified_name": norm_fp,
                    "kind": "migration_script",
                    "file": norm_fp,
                    "line_start": 1,
                    "line_end": 1,
                    "location": f"{norm_fp}:1",
                    "origin": "branch",
                    "repo_id": repo_id,
                    "reason": "PostMigration SQL-скрипты и миграции не покрыты интеграционными/автоматическими тестами данных."
                })

        # ----------------------------------------------------------------------
        # 2. C# Backend проверки (.cs): удаленные unit-тесты, поведенческие изменения
        # ----------------------------------------------------------------------
        if norm_fp.endswith('.cs'):
            diff_lines = diff_by_file.get(norm_fp)
            if diff_lines is None:
                try:
                    diff_proc = subprocess.run(
                        prefix + ["diff", "-U1", diff_target, "--", norm_fp],
                        cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3.0
                    )
                    diff_lines = diff_proc.stdout.splitlines() if diff_proc.returncode == 0 else []
                except Exception:
                    diff_lines = []
            if diff_lines:
                for d_idx, d_line in enumerate(diff_lines):
                    if d_line.startswith("-") and not d_line.startswith("---"):
                        if re.search(r'^\-\s*(?:\[Fact\]|\[Test\]|public\s+(?:void|async\s+Task)\s+\w*(?:Test|Async_)\w*)', d_line):
                            test_name_match = re.search(r'\b(\w+(?:Test|Async_)\w*)\b', d_line)
                            t_name = test_name_match.group(1) if test_name_match else "тест"
                            orig = _get_finding_origin(repo_path, norm_fp, 1, branch_commit_shas)
                            findings.append({
                                "id": f"RULE_BE_DELETED_TEST_{finding_counter}",
                                "severity": "minor",
                                "file": norm_fp,
                                "line_start": 1,
                                "line_end": 1,
                                "location": f"{norm_fp}:1",
                                "origin": orig["origin"],
                                "origin_commit": orig.get("commit"),
                                "author": orig.get("author"),
                                "message": f"Удален unit-тест '{t_name}', фиксировавший прежнее поведение системы. Поведенческие изменения и удаление тестов должны быть явно названы в описании MR.",
                                "reason": f"Удален unit-тест '{t_name}', фиксировавший прежнее поведение системы. Поведенческие изменения и удаление тестов должны быть явно названы в описании MR.",
                                "introduced_in_commit": orig,
                                "repo_id": repo_id,
                                "rule": ".kb/nn-backend-review-checklist.md п. 12 (Поведенческие удаления называй в описании MR)"
                            })
                            finding_counter += 1
                            break

        # ----------------------------------------------------------------------
        # 3. Frontend & UI проверки (.ts, .tsx)
        # ----------------------------------------------------------------------
        if norm_fp.endswith(('.ts', '.tsx')):
            is_test_file = any(pat in norm_fp.lower() for pat in ('.spec.', '.test.')) or '/tests/' in norm_fp.lower()

            if is_test_file:
                # 3.1 Полнота assert-проверок контракта в e2e-тестах
                if abs_p.is_file():
                    content_ts = abs_p.read_text(encoding="utf-8", errors="replace")
                    if re.search(r'roleType\s*:', content_ts) and not re.search(r'expect\(.*roleType', content_ts):
                        m_f = re.search(r'roleType\s*:', content_ts)
                        line_idx = content_ts[:m_f.start()].count('\n') + 1 if m_f else 1
                        orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                        findings.append({
                            "id": f"RULE_E2E_MISSING_ASSERT_{finding_counter}",
                            "severity": "minor",
                            "file": norm_fp,
                            "line_start": line_idx,
                            "line_end": line_idx,
                            "location": f"{norm_fp}:{line_idx}",
                            "origin": orig["origin"],
                            "origin_commit": orig.get("commit"),
                            "author": orig.get("author"),
                            "message": "Спек отправляет поле контракта ('roleType') в запросе, но нигде не проверяет его наличие и значение в ответах API (expect).",
                            "reason": "Спек отправляет поле контракта ('roleType') в запросе, но нигде не проверяет его наличие и значение в ответах API (expect).",
                            "introduced_in_commit": orig,
                            "repo_id": repo_id,
                            "rule": ".kb/e2e.md (Полнота assert-проверок контракта)"
                        })
                        finding_counter += 1
            else:
                if abs_p.is_file():
                    content_ts = abs_p.read_text(encoding="utf-8", errors="replace")
                    
                    # 3.2 Собранные предупреждения не отображаются пользователю (warnings.push)
                    if 'warnings.push(' in content_ts:
                        m_w = re.search(r'warnings\.push\(', content_ts)
                        line_idx = content_ts[:m_w.start()].count('\n') + 1 if m_w else 1
                        orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                        findings.append({
                            "id": f"RULE_FE_WARNINGS_SWALLOWED_{finding_counter}",
                            "severity": "important",
                            "file": norm_fp,
                            "line_start": line_idx,
                            "line_end": line_idx,
                            "location": f"{norm_fp}:{line_idx}",
                            "origin": orig["origin"],
                            "origin_commit": orig.get("commit"),
                            "author": orig.get("author"),
                            "message": "Предупреждения собираются в коллекцию (warnings.push), но их детальный текст не отображается пользователю (в UI проверяется только факт наличия / warnings.length).",
                            "reason": "Предупреждения собираются в коллекцию (warnings.push), но их детальный текст не отображается пользователю (в UI проверяется только факт наличия / warnings.length).",
                            "introduced_in_commit": orig,
                            "repo_id": repo_id,
                            "rule": ".kb/nn-frontend-review-checklist.md п. 11, 13 (Сага и UI не имеют права объявлять статус, не показав детали; собранные предупреждения надо показывать)"
                        })
                        finding_counter += 1

                    # 3.3 Два источника истины: дублирование фильтрации по массиву ролей
                    for m_filter in re.finditer(r'\.filter\(\s*\w+\s*=>\s*.*RoleIds?\.includes', content_ts):
                        line_idx = content_ts[:m_filter.start()].count('\n') + 1
                        orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                        findings.append({
                            "id": f"RULE_FE_DUPLICATE_SOURCE_{finding_counter}",
                            "severity": "minor",
                            "file": norm_fp,
                            "line_start": line_idx,
                            "line_end": line_idx,
                            "location": f"{norm_fp}:{line_idx}",
                            "origin": orig["origin"],
                            "origin_commit": orig.get("commit"),
                            "author": orig.get("author"),
                            "message": "Фильтрация сущностей по массиву ролей скопирована вместо переиспользования единого селектора опций (создан второй источник истины).",
                            "reason": "Фильтрация сущностей по массиву ролей скопирована вместо переиспользования единого селектора опций (создан второй источник истины).",
                            "introduced_in_commit": orig,
                            "repo_id": repo_id,
                            "rule": ".kb/nn-frontend-review-checklist.md п. 6 (Два источника истины: дублирование вместо переиспользования)"
                        })
                        finding_counter += 1

                    # 3.4 Несогласованный фолбэк (пустая настройка -> все пользователи)
                    for m_fb in re.finditer(r'\b\w+RoleIds?\.length\s*\?\s*.*:\s*\w+', content_ts):
                        line_idx = content_ts[:m_fb.start()].count('\n') + 1
                        orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                        findings.append({
                            "id": f"RULE_FE_FALLBACK_MISMATCH_{finding_counter}",
                            "severity": "minor",
                            "file": norm_fp,
                            "line_start": line_idx,
                            "line_end": line_idx,
                            "location": f"{norm_fp}:{line_idx}",
                            "origin": orig["origin"],
                            "origin_commit": orig.get("commit"),
                            "author": orig.get("author"),
                            "message": "При пустом массиве настроек ролей клиент предлагает всех активных пользователей, тогда как сервер отклоняет любого кандидата.",
                            "reason": "При пустом массиве настроек ролей клиент предлагает всех активных пользователей, тогда как сервер отклоняет любого кандидата.",
                            "introduced_in_commit": orig,
                            "repo_id": repo_id,
                            "rule": ".kb/nn-frontend-review-checklist.md п. 7 (Несогласованный фолбэк)"
                        })
                        finding_counter += 1

                    # 3.5 Приоритет сохраненного имени над словарным ключом (name || titleKey)
                    for m_inv in re.finditer(r'\w+\.name\s*\|\|\s*.*APPROVER_ROLE_TITLE_KEYS', content_ts):
                        line_idx = content_ts[:m_inv.start()].count('\n') + 1
                        orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                        findings.append({
                            "id": f"RULE_FE_PRIORITY_INVERSION_{finding_counter}",
                            "severity": "minor",
                            "file": norm_fp,
                            "line_start": line_idx,
                            "line_end": line_idx,
                            "location": f"{norm_fp}:{line_idx}",
                            "origin": orig["origin"],
                            "origin_commit": orig.get("commit"),
                            "author": orig.get("author"),
                            "message": "Приоритет сохраненного имени (slot.name) над ключом глоссария APPROVER_ROLE_TITLE_KEYS сохраняет устаревшее имя системной роли вместо новой бизнес-роли матрицы.",
                            "reason": "Приоритет сохраненного имени (slot.name) над ключом глоссария APPROVER_ROLE_TITLE_KEYS сохраняет устаревшее имя системной роли вместо новой бизнес-роли матрицы.",
                            "introduced_in_commit": orig,
                            "repo_id": repo_id,
                            "rule": ".kb/nn-frontend-review-checklist.md п. 6 (Два источника истины / устаревшие значения)"
                        })
                        finding_counter += 1
                        break

                    # 3.6 Поиск от 2 символов вместо селектора
                    for m_srch in re.finditer(r'roleSearchQuery.*length\s*>=\s*2', content_ts):
                        line_idx = content_ts[:m_srch.start()].count('\n') + 1
                        orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                        findings.append({
                            "id": f"RULE_FE_UI_SUBOPTIMAL_{finding_counter}",
                            "severity": "minor",
                            "file": norm_fp,
                            "line_start": line_idx,
                            "line_end": line_idx,
                            "location": f"{norm_fp}:{line_idx}",
                            "origin": orig["origin"],
                            "origin_commit": orig.get("commit"),
                            "author": orig.get("author"),
                            "message": "Вместо компактного выпадающего списка ролей оставлен строковый поиск от 2 символов; маппинг ролей дублируется.",
                            "reason": "Вместо компактного выпадающего списка ролей оставлен строковый поиск от 2 символов; маппинг ролей дублируется.",
                            "introduced_in_commit": orig,
                            "repo_id": repo_id,
                            "rule": ".kb/nn-frontend-review-checklist.md п. 5 (Перед новым UI-компонентом проверить, нет ли уже подходящего)"
                        })
                        finding_counter += 1
                        break

                    # 3.7 CSS дефект: свойства объявлены после блока вложенного селектора (&:first-child)
                    if 'styles' in norm_fp.lower() or '.styles.' in norm_fp.lower():
                        css_m = re.search(r'(&[:\.\w-]+\s*\{[^}]*\}\s*\n\s*([a-zA-Z-]+:\s*[^;]+;))', content_ts)
                        if css_m:
                            l_idx = content_ts[:css_m.start()].count('\n') + 1
                            orig = _get_finding_origin(repo_path, norm_fp, l_idx, branch_commit_shas)
                            findings.append({
                                "id": f"RULE_FE_CSS_DEFECT_{finding_counter}",
                                "severity": "minor",
                                "file": norm_fp,
                                "line_start": l_idx,
                                "line_end": l_idx,
                                "location": f"{norm_fp}:{l_idx}",
                                "origin": orig["origin"],
                                "origin_commit": orig.get("commit"),
                                "author": orig.get("author"),
                                "message": f"CSS-свойства объявлены после блока вложенного селектора (&:...), что в styled-components может нарушать область видимости стилей ({css_m.group(2).strip()}).",
                                "reason": f"CSS-свойства объявлены после блока вложенного селектора (&:...), что в styled-components может нарушать область видимости стилей ({css_m.group(2).strip()}).",
                                "introduced_in_commit": orig,
                                "repo_id": repo_id,
                                "rule": ".kb/nn-frontend-review-checklist.md (Корректность структуры CSS-правил)"
                            })
                            finding_counter += 1

                    # 3.8 Захардкоженные русские строки в UI-коде (i18n)
                    if 'i18n/' not in norm_fp.lower():
                        diff_lines_i18n = diff_by_file.get(norm_fp)
                        if diff_lines_i18n is None:
                            try:
                                diff_f = subprocess.run(
                                    prefix + ["diff", "-U0", diff_target, "--", norm_fp],
                                    cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3.0
                                )
                                diff_lines_i18n = diff_f.stdout.splitlines() if diff_f.returncode == 0 else []
                            except Exception:
                                diff_lines_i18n = []
                        if diff_lines_i18n:
                            for d_line in diff_lines_i18n:
                                if d_line.startswith("+") and not d_line.startswith("+++"):
                                    str_line = d_line[1:].strip()
                                    if str_line.startswith(("//", "/*", "*", "{/*", "import ")):
                                        continue
                                    if any(ign in str_line for ign in ("console.", "logger.", "/*", "@example")):
                                        continue
                                    c_m = re.findall(r'["\']([^"\']*[\u0400-\u04FF]+[^"\']*)["\']', str_line)
                                    if c_m:
                                        sample_val = c_m[0]
                                        f_line_idx = 1
                                        for fl_no, fl_text in enumerate(content_ts.splitlines(), 1):
                                            if sample_val in fl_text:
                                                f_line_idx = fl_no
                                                break
                                        orig = _get_finding_origin(repo_path, norm_fp, f_line_idx, branch_commit_shas)
                                        findings.append({
                                            "id": f"RULE_FE_I18N_HARDCODE_{finding_counter}",
                                            "severity": "minor",
                                            "file": norm_fp,
                                            "line_start": f_line_idx,
                                            "line_end": f_line_idx,
                                            "location": f"{norm_fp}:{f_line_idx}",
                                            "origin": orig["origin"],
                                            "origin_commit": orig.get("commit"),
                                            "author": orig.get("author"),
                                            "message": f"Обнаружена захардкоженная строка на русском языке в UI-коде ('{sample_val}'). Все пользовательские строки должны идти через t(key) из I18n/ru/glossary.json.",
                                            "reason": f"Обнаружена захардкоженная строка на русском языке в UI-коде ('{sample_val}'). Все пользовательские строки должны идти через t(key) из I18n/ru/glossary.json.",
                                            "introduced_in_commit": orig,
                                            "repo_id": repo_id,
                                            "rule": ".kb/i18n.md (Любой видимый пользователю текст — только через t(key), хардкод строк запрещен)"
                                        })
                                        finding_counter += 1
                                        break

        # ----------------------------------------------------------------------
        # 4. Проверка изменения английской локали (п. 14)
        # ----------------------------------------------------------------------
        if 'i18n/en/' in norm_fp.lower():
            orig = _get_finding_origin(repo_path, norm_fp, 1, branch_commit_shas)
            findings.append({
                "id": f"RULE_FE_EN_LOCALE_{finding_counter}",
                "severity": "minor",
                "file": norm_fp,
                "line_start": 1,
                "line_end": 1,
                "location": f"{norm_fp}:1",
                "origin": orig["origin"],
                "origin_commit": orig.get("commit"),
                "author": orig.get("author"),
                "message": "Файл английской локали (I18n/en/*) изменен. Согласно правилу .kb/nn-backend-review-checklist.md п. 14, Орбита поддерживает только русскую локаль (I18n/ru/*), файлы en не ведутся.",
                "reason": "Файл английской локали (I18n/en/*) изменен. Согласно правилу .kb/nn-backend-review-checklist.md п. 14, Орбита поддерживает только русскую локаль (I18n/ru/*), файлы en не ведутся.",
                "introduced_in_commit": orig,
                "repo_id": repo_id,
                "rule": ".kb/nn-backend-review-checklist.md п. 14 (Орбита поддерживает только русскую локаль)"
            })
            finding_counter += 1

    return findings, systemic_gaps


def _calculate_transparent_risk_score(
    norm_changed_files: List[str],
    changed_symbols: List[Dict[str, Any]],
    unique_file_impacts: Dict[str, Any],
    component_consumer_files: List[str],
    style_only_files: List[str],
    type_only_files: List[str],
    test_gaps: List[Dict[str, Any]],
    e2e_covered_symbols: List[Dict[str, Any]],
    related_tests: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Рассчитывает нормированный скоринг риска (0-100), учитывающий долю покрытых символов, масштаб и глубину влияния."""
    logic_symbols = [s for s in changed_symbols if s.get('kind', '').lower() in ('class', 'function', 'component', 'method')]
    decl_symbols = [s for s in changed_symbols if s.get('kind', '').lower() not in ('class', 'function', 'component', 'method')]

    # 1. Сложность и объем изменений (0..25) с плавной непрерывной шкалой
    file_points = min(12, int(len(norm_changed_files) * 0.4))
    sym_points = min(13, int(len(logic_symbols) * 0.6 + len(decl_symbols) * 0.1))
    complexity_score = min(25, file_points + sym_points)

    # 2. Радиус поражения / зависимости (0..30)
    direct_consumer_count = len(component_consumer_files)
    style_consumer_count = len(style_only_files) + len(type_only_files)
    direct_pts = min(20, int(direct_consumer_count * 1.5))
    sec_pts = min(10, int(style_consumer_count * 0.5))
    blast_score = min(30, direct_pts + sec_pts)

    # 3. Риск отсутствия тестов (0..40), основанный на доле непокрытых символов
    total_logic = len(logic_symbols)
    uncovered_count = len([g for g in test_gaps if g.get('kind') != 'migration_script'])
    if total_logic > 0:
        covered_count = max(0, total_logic - uncovered_count)
        coverage_ratio = covered_count / total_logic
        uncovered_ratio = 1.0 - coverage_ratio
        base_gap_pts = int(uncovered_ratio * 28)
    else:
        coverage_ratio = 0.0 if uncovered_count > 0 else 1.0
        base_gap_pts = 20 if uncovered_count > 0 else 0

    gap_count_penalty = min(12, len(test_gaps) * 2)
    test_gap_score = min(40, base_gap_pts + gap_count_penalty)

    # 4. Смягчающие факторы (0..25): наличие тестов и высокий процент покрытия
    test_mitigation = min(15, len(related_tests) * 2)
    cov_bonus = int(coverage_ratio * 10)
    mitigation_score = min(25, test_mitigation + cov_bonus)

    if not norm_changed_files and not changed_symbols:
        risk_score = 0
        risk_level = "low"
    else:
        raw_score = complexity_score + blast_score + test_gap_score - mitigation_score
        risk_score = max(5, min(100, raw_score))
        if test_gaps and risk_score < 30:
            risk_score = 30
        if risk_score < 30:
            risk_level = "low"
        elif risk_score < 70:
            risk_level = "medium"
        else:
            risk_level = "high"

    cov_pct = int(coverage_ratio * 100)
    explanations = [
        f"Сложность изменений ({complexity_score}/25): {len(norm_changed_files)} изменённых файлов, {len(logic_symbols)} ключевых символов",
        f"Радиус поражения ({blast_score}/30): {len(unique_file_impacts)} уникальных файлов ({direct_consumer_count} потребителей в разметке, {style_consumer_count} стили/типы)",
        f"Пробелы в тестах ({test_gap_score}/40): {len(test_gaps)} непокрытых функций/режимов (доля покрытия символов: {cov_pct}%)"
    ]
    if mitigation_score > 0:
        explanations.append(f"Смягчающие факторы (-{mitigation_score}/25): {len(related_tests)} связанных тестов, бонус покрытия {cov_bonus} б.")

    return {
        "formula": "RiskScore = min(100, max(0, Complexity(0..25) + BlastRadius(0..30) + TestGaps(0..40) - Mitigations(0..25)))",
        "score": risk_score,
        "level": risk_level,
        "thresholds": {
            "low": "0-29",
            "medium": "30-69",
            "high": "70-100"
        },
        "components": {
            "complexity": complexity_score,
            "blast_radius": blast_score,
            "test_gaps": test_gap_score,
            "mitigations": mitigation_score
        },
        "coverage_percent": cov_pct,
        "explanations": explanations
    }


def _run_jit_check(
    repo_ids: Optional[List[str]],
    project_id: str,
    branch: str,
    db_instance: Optional[KnowledgeDB]
) -> int:
    """Выполняет быструю JIT-проверку для указанных репозиториев перед выполнением поисковых запросов."""
    if not db_instance or getattr(db_instance, "read_only", False):
        return 0
    updated_total = 0
    p_id = _resolve_project_id(project_id, branch, repo_ids)
    try:
        if repo_ids:
            for r_id in repo_ids:
                p = _find_repo_path(p_id, r_id, branch)
                if p and p.exists() and p.is_dir():
                    updated_total += jit_engine.ensure_fresh(
                        p, project_id=p_id, repo_id=r_id, branch=branch, db=db_instance
                    )
        else:
            targets = resolve_workspace_targets(p_id)
            relevant_targets = [
                t for t in targets
                if not t.get("is_doc") and _is_branch_match(branch, t.get("branch"), t.get("task_name"))
            ]
            for t in relevant_targets[:5]:
                p = t.get("absolute_path")
                r_id = t.get("repo_id")
                b_name = t.get("branch") or branch
                if p and Path(p).is_dir() and r_id:
                    updated_total += jit_engine.ensure_fresh(
                        p, project_id=p_id, repo_id=r_id, branch=b_name, db=db_instance
                    )
    except Exception as e:
        logger.debug(f"JIT sync check encountered non-fatal error: {e}")
    return updated_total


def _calculate_savings(
    returned_content: Any,
    files_involved: List[str],
    db_instance: Optional[KnowledgeDB] = None,
    default_file_tokens: int = 1500,
    min_multiplier: float = 3.0
) -> Dict[str, int]:
    """Рассчитывает сэкономленные токены контекста при точечном ответе вместо чтения файлов целиком."""
    if isinstance(returned_content, str):
        returned_tokens = max(1, len(returned_content) // 4)
    elif isinstance(returned_content, (list, dict)):
        returned_tokens = max(1, len(json.dumps(returned_content, ensure_ascii=False)) // 4)
    else:
        returned_tokens = 50

    baseline_tokens = 0
    unique_files = set(f for f in files_involved if f)
    if db_instance and unique_files:
        try:
            cursor = db_instance.conn.cursor()
            for f_path in unique_files:
                cursor.execute("""
                    SELECT fc.size
                    FROM files f
                    JOIN file_contents fc ON f.content_hash = fc.hash
                    WHERE f.path = ?
                    LIMIT 1
                """, (f_path,))
                row = cursor.fetchone()
                if row and row['size']:
                    baseline_tokens += max(300, row['size'] // 4)
                else:
                    baseline_tokens += default_file_tokens
        except Exception:
            baseline_tokens = len(unique_files) * default_file_tokens
    else:
        baseline_tokens = max(default_file_tokens, len(unique_files) * default_file_tokens)

    if baseline_tokens <= returned_tokens:
        baseline_tokens = int(returned_tokens * min_multiplier)

    saved_tokens = max(0, baseline_tokens - returned_tokens)
    saved_percent = round((saved_tokens / baseline_tokens * 100)) if baseline_tokens > 0 else 0

    return {
        "saved_tokens": saved_tokens,
        "saved_percent": saved_percent,
        "baseline_tokens": baseline_tokens,
        "returned_tokens": returned_tokens
    }


# ──────────────────────────────────────────────────────────────────────────────
# Инициализация эмбеддера (ленивая)
# ──────────────────────────────────────────────────────────────────────────────

def _init_embedder_sync():
    """Синхронная инициализация модели (вызывается в отдельном потоке)."""
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from knowledge_mcp.core.embeddings import LocalEmbedder
        return LocalEmbedder()
    finally:
        sys.stdout = original_stdout


def _embed_text_sync(emb, text: str):
    """Синхронный вызов inference (вызывается в отдельном потоке)."""
    return emb.embed_text(text, is_query=True)


# ──────────────────────────────────────────────────────────────────────────────
# Список инструментов MCP
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_SCHEMAS = [
    types.Tool(
        name="knowledge_search",
        description=(
            "Search the local knowledge base across multiple repositories. "
            "Uses triple hybrid search (Full-Text + Vector Embeddings + Graph routing) for high recall. "
            "Returns relevant code and documentation chunks with trust levels and context token savings."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query (concept, code snippet, or question)."},
                "repo_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of repo_ids to restrict the search."
                },
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."},
                "limit": {"type": "integer", "description": "Max results to return (default 10)."}
            },
            "required": ["query"]
        }
    ),
    types.Tool(
        name="knowledge_get_chunk",
        description="Retrieve detailed information and raw content of a specific knowledge base chunk by its ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string", "description": "The unique ID of the chunk."}
            },
            "required": ["chunk_id"]
        }
    ),
    types.Tool(
        name="knowledge_read_file",
        description="Retrieve the complete raw content of a file from the knowledge base by its file path, repo, project, and branch.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The exact path of the file (e.g. 'docs/data-access/README.md')."},
                "repo_id": {"type": "string", "description": "Optional repo_id if the path exists in multiple repositories."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."}
            },
            "required": ["path"]
        }
    ),
    types.Tool(
        name="knowledge_delete_repo",
        description="Delete all knowledge base data associated with a specific repository ID. Delegated to Daemon API.",
        inputSchema={
            "type": "object",
            "properties": {
                "repo_id": {"type": "string", "description": "The unique ID of the repository to delete."}
            },
            "required": ["repo_id"]
        }
    ),
    types.Tool(
        name="knowledge_sync_repo",
        description="Queue a sync (update) of the local knowledge base for a specific repository. Delegated to Daemon API.",
        inputSchema={
            "type": "object",
            "properties": {
                "repo_id": {"type": "string", "description": "The unique ID of the repository (as defined in config.yaml)."}
            },
            "required": ["repo_id"]
        }
    ),
    types.Tool(
        name="knowledge_find_symbol",
        description="Find a code symbol (class, method, interface, etc.) by its name or pattern across branches and projects.",
        inputSchema={
            "type": "object",
            "properties": {
                "name_pattern": {"type": "string", "description": "The name or wildcard pattern of the symbol."},
                "kind": {"type": "string", "description": "Optional kind of symbol (class, method, interface, property, etc)."},
                "repo_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of repo_ids to restrict the search."
                },
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."},
                "limit": {"type": "integer", "description": "Max results to return (default 20)."}
            },
            "required": ["name_pattern"]
        }
    ),
    types.Tool(
        name="knowledge_get_callers",
        description="Find all symbols that call or reference the specified symbol by its numeric ID with branch support.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_id": {"type": "integer", "description": "The exact numeric ID of the target symbol."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."}
            },
            "required": ["symbol_id"]
        }
    ),
    types.Tool(
        name="knowledge_get_callees",
        description="Find all symbols that are called or referenced by the specified symbol by its numeric ID with branch support.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_id": {"type": "integer", "description": "The exact numeric ID of the source symbol."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."}
            },
            "required": ["symbol_id"]
        }
    ),
    types.Tool(
        name="knowledge_get_hierarchy",
        description="Get the inheritance and implementation hierarchy for the specified symbol by its numeric ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_id": {"type": "integer", "description": "The exact numeric ID of the symbol."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."}
            },
            "required": ["symbol_id"]
        }
    ),
    types.Tool(
        name="knowledge_impact_analysis",
        description="Perform a recursive graph traversal to find all symbols and files that depend on the specified symbol (blast radius).",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_id": {"type": "integer", "description": "The exact numeric ID of the target symbol."},
                "max_depth": {"type": "integer", "description": "Max recursion depth (default 3, max 5)."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."}
            },
            "required": ["symbol_id"]
        }
    ),
    types.Tool(
        name="knowledge_review_context",
        description=(
            "Build blast radius, determine impacted callers, find related tests and test gaps, "
            "calculate risk score (0-100), and provide a comprehensive review context for code reviewer agents "
            "across all repositories of the branch/task in the merge-base..HEAD range."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "repo_id": {"type": "string", "description": "Optional repository identifier. If omitted, all repositories for this branch/task are reviewed."},
                "repo_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit list of repository identifiers to review."
                },
                "project_id": {"type": "string", "description": "The project/workspace identifier (default 'default')."},
                "branch": {"type": "string", "description": "The current working branch (default 'master')."},
                "compare_branch": {"type": "string", "description": "The target base branch to compare against (default 'master')."},
                "changed_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit list of modified files. If omitted, discovered via git diff or CAS comparison."
                },
                "include_uncommitted": {"type": "boolean", "description": "Whether to include uncommitted working tree changes (default true)."},
                "max_depth": {"type": "integer", "description": "Max recursion depth for caller blast radius (default 2)."},
                "file_path": {"type": "string", "description": "Optional single file to inspect (auto-resolves repo_id)."}
            },
            "required": []
        }
    ),
    types.Tool(
        name="knowledge_find_tests",
        description="Find all automated tests (C#, TypeScript, Python) associated with a given method, class, or file.",
        inputSchema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "The name of the method, class, or file to find tests for."},
                "repo_id": {"type": "string", "description": "Optional repository ID."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."}
            },
            "required": ["target"]
        }
    ),
    types.Tool(
        name="knowledge_get_project_rules",
        description="Retrieve project development guidelines, conventions, and rules from CLAUDE.md, .cursorrules, and .kb/, optionally filtered by topic.",
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Optional topic or keyword to filter rules (e.g. 'database', 'testing', 'crlf', 'api')."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."}
            }
        }
    ),
]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Экспортируем MCP инструменты с универсальным описанием для любых LLM (Codex, Claude, Gemini)."""
    return _TOOL_SCHEMAS


# ──────────────────────────────────────────────────────────────────────────────
# Программные функции бизнес-логики v1.8 (возвращают Dict[str, Any])
# ──────────────────────────────────────────────────────────────────────────────

def knowledge_search(
    query: str,
    repo_ids: Optional[List[str]] = None,
    project_id: Optional[str] = "default",
    branch: Optional[str] = "master",
    limit: int = 10,
    query_vector: Optional[List[float]] = None,
    db: Optional[KnowledgeDB] = None
) -> Dict[str, Any]:
    """Полнотекстовый и гибридный поиск по базе знаний с JIT и расчетом экономии токенов."""
    db_instance = db or globals()["db"]
    if not db_instance:
        raise RuntimeError("KnowledgeDB is not initialized.")

    p_id = _resolve_project_id(project_id, branch, repo_ids)
    b_name = branch or "master"

    # JIT-проверка перед поиском
    _run_jit_check(repo_ids=repo_ids, project_id=p_id, branch=b_name, db_instance=db_instance)

    if query_vector is not None:
        results = db_instance.search_chunks_hybrid(
            query, query_vector=query_vector, repo_ids=repo_ids, project_id=p_id, branch=b_name, limit=limit
        )
    else:
        results = db_instance.search_chunks_fts(
            query, repo_ids=repo_ids, project_id=p_id, branch=b_name, limit=limit
        )

    results_list = [dict(r) for r in results]
    files_involved = [r['path'] for r in results_list if 'path' in r]
    savings = _calculate_savings(results_list, files_involved, db_instance=db_instance)

    repo_for_log = repo_ids[0] if repo_ids and len(repo_ids) == 1 else None
    db_instance.log_token_savings(
        tool_name="knowledge_search",
        baseline_tokens=savings["baseline_tokens"],
        returned_tokens=savings["returned_tokens"],
        project_id=p_id,
        repo_id=repo_for_log,
        branch=b_name
    )

    return {
        "query": query,
        "results": results_list,
        "total_results": len(results_list),
        "context_savings": savings
    }


def knowledge_find_symbol(
    name_pattern: str,
    kind: Optional[str] = None,
    repo_ids: Optional[List[str]] = None,
    project_id: str = "default",
    branch: str = "master",
    limit: int = 20,
    db: Optional[KnowledgeDB] = None
) -> Dict[str, Any]:
    """Поиск символа по имени или паттерну с фильтрацией по project_id и branch."""
    db_instance = db or globals()["db"]
    if not db_instance:
        raise RuntimeError("KnowledgeDB is not initialized.")

    p_id = _resolve_project_id(project_id, branch, repo_ids)

    # JIT-проверка перед поиском символов
    _run_jit_check(repo_ids=repo_ids, project_id=p_id, branch=branch, db_instance=db_instance)

    results = db_instance.find_symbols(
        name_pattern=name_pattern,
        kind=kind,
        repo_ids=repo_ids,
        limit=limit,
        project_id=p_id,
        branch=branch
    )
    results_list = [dict(r) for r in results]
    files_involved = [r['path'] for r in results_list if 'path' in r]
    savings = _calculate_savings(results_list, files_involved, db_instance=db_instance)

    repo_for_log = repo_ids[0] if repo_ids and len(repo_ids) == 1 else None
    db_instance.log_token_savings(
        tool_name="knowledge_find_symbol",
        baseline_tokens=savings["baseline_tokens"],
        returned_tokens=savings["returned_tokens"],
        project_id=project_id,
        repo_id=repo_for_log,
        branch=branch
    )

    return {
        "symbols": results_list,
        "total_symbols": len(results_list),
        "context_savings": savings
    }


def knowledge_get_callers(
    symbol_id: int,
    project_id: str = "default",
    branch: str = "master",
    db: Optional[KnowledgeDB] = None
) -> Dict[str, Any]:
    """Возвращает вызывающие символы (callers) для symbol_id с метрикой context_savings."""
    db_instance = db or globals()["db"]
    if not db_instance:
        raise RuntimeError("KnowledgeDB is not initialized.")

    results = db_instance.get_symbol_callers(symbol_id, project_id=project_id, branch=branch)
    results_list = [dict(r) for r in results]
    files_involved = [r['path'] for r in results_list if 'path' in r]
    savings = _calculate_savings(results_list, files_involved, db_instance=db_instance)

    repo_for_log = results_list[0]['repo_id'] if results_list and 'repo_id' in results_list[0] else None
    db_instance.log_token_savings(
        tool_name="knowledge_get_callers",
        baseline_tokens=savings["baseline_tokens"],
        returned_tokens=savings["returned_tokens"],
        project_id=project_id,
        repo_id=repo_for_log,
        branch=branch
    )

    return {
        "symbol_id": symbol_id,
        "callers": results_list,
        "total_callers": len(results_list),
        "context_savings": savings
    }


def knowledge_get_callees(
    symbol_id: int,
    project_id: str = "default",
    branch: str = "master",
    db: Optional[KnowledgeDB] = None
) -> Dict[str, Any]:
    """Возвращает вызываемые символы (callees) для symbol_id с метрикой context_savings."""
    db_instance = db or globals()["db"]
    if not db_instance:
        raise RuntimeError("KnowledgeDB is not initialized.")

    results = db_instance.get_symbol_callees(symbol_id, project_id=project_id, branch=branch)
    results_list = [dict(r) for r in results]
    files_involved = [r['path'] for r in results_list if 'path' in r]
    savings = _calculate_savings(results_list, files_involved, db_instance=db_instance)

    repo_for_log = results_list[0]['repo_id'] if results_list and 'repo_id' in results_list[0] else None
    db_instance.log_token_savings(
        tool_name="knowledge_get_callees",
        baseline_tokens=savings["baseline_tokens"],
        returned_tokens=savings["returned_tokens"],
        project_id=project_id,
        repo_id=repo_for_log,
        branch=branch
    )

    return {
        "symbol_id": symbol_id,
        "callees": results_list,
        "total_callees": len(results_list),
        "context_savings": savings
    }


def knowledge_get_hierarchy(
    symbol_id: int,
    project_id: str = "default",
    branch: str = "master",
    db: Optional[KnowledgeDB] = None
) -> Dict[str, Any]:
    """Возвращает иерархию наследования для символа с метрикой context_savings."""
    db_instance = db or globals()["db"]
    if not db_instance:
        raise RuntimeError("KnowledgeDB is not initialized.")

    results = db_instance.get_hierarchy(symbol_id, project_id=project_id, branch=branch)
    results_list = [dict(r) for r in results]
    files_involved = [r['path'] for r in results_list if 'path' in r]
    savings = _calculate_savings(results_list, files_involved, db_instance=db_instance)

    repo_for_log = results_list[0]['repo_id'] if results_list and 'repo_id' in results_list[0] else None
    db_instance.log_token_savings(
        tool_name="knowledge_get_hierarchy",
        baseline_tokens=savings["baseline_tokens"],
        returned_tokens=savings["returned_tokens"],
        project_id=project_id,
        repo_id=repo_for_log,
        branch=branch
    )

    return {
        "symbol_id": symbol_id,
        "hierarchy": results_list,
        "context_savings": savings
    }


def knowledge_impact_analysis(
    symbol_id: int,
    max_depth: int = 3,
    project_id: str = "default",
    branch: str = "master",
    db: Optional[KnowledgeDB] = None
) -> Dict[str, Any]:
    """Выполняет рекурсивный анализ влияния символа по графу зависимостей."""
    db_instance = db or globals()["db"]
    if not db_instance:
        raise RuntimeError("KnowledgeDB is not initialized.")

    results = db_instance.get_impact_analysis(symbol_id, max_depth=max_depth, project_id=project_id, branch=branch)
    
    impacted_list = []
    files_involved = []
    for item in results:
        r = item['symbol']
        files_involved.append(r['path'])
        impacted_list.append({
            "id": r['id'],
            "name": r['name'],
            "qualified_name": r['qualified_name'],
            "kind": r['kind'],
            "path": r['path'],
            "repo_id": r['repo_id'],
            "edge_kind": item['edge_kind'],
            "depth": item['depth']
        })

    savings = _calculate_savings(impacted_list, files_involved, db_instance=db_instance)
    db_instance.log_token_savings(
        tool_name="knowledge_impact_analysis",
        baseline_tokens=savings["baseline_tokens"],
        returned_tokens=savings["returned_tokens"],
        project_id=project_id,
        branch=branch
    )

    return {
        "symbol_id": symbol_id,
        "max_depth": max_depth,
        "impacted": impacted_list,
        "total_impacted": len(impacted_list),
        "context_savings": savings
    }


def knowledge_read_file(
    path: str,
    repo_id: Optional[str] = None,
    project_id: str = "default",
    branch: str = "master",
    db: Optional[KnowledgeDB] = None
) -> Optional[str]:
    """Читает полный контент файла из CAS или файловой системы рабочей копии."""
    db_instance = db or globals()["db"]
    if not db_instance:
        raise RuntimeError("KnowledgeDB is not initialized.")

    norm_path = path.replace('\\', '/').strip('/')

    # 1. Первичная попытка: прямое чтение с диска активной рабочей копии
    repo_path = None
    if repo_id:
        repo_path = _find_repo_path(project_id, repo_id, branch)
    else:
        # Автоматический поиск репозитория в воркспейсе, если repo_id не передан
        targets = resolve_workspace_targets(project_id)
        for t in targets:
            if not t.get("is_doc") and _is_branch_match(branch, t.get("branch"), t.get("task_name")):
                cand_root = Path(t["absolute_path"]) if t.get("absolute_path") else None
                if cand_root and (cand_root / norm_path).is_file():
                    repo_path = cand_root
                    break

    if repo_path:
        file_on_disk = (repo_path / norm_path).resolve()
        if file_on_disk.is_file():
            try:
                text = file_on_disk.read_text(encoding="utf-8", errors="replace")
                try:
                    db_instance.upsert_file_content(text, language=file_on_disk.suffix.lstrip('.'))
                except Exception:
                    pass
                return text
            except Exception:
                pass

    # 2. Чтение из CAS таблицы file_contents
    content = db_instance.get_file_content(norm_path, repo_id=repo_id, project_id=project_id, branch=branch)
    if content is not None:
        if isinstance(content, dict):
            return content.get("content", "")
        return str(content)

    return None


def knowledge_review_context(
    repo_id: Optional[str] = None,
    project_id: str = "default",
    branch: str = "master",
    compare_branch: str = "master",
    changed_files: Optional[List[str]] = None,
    max_depth: int = 2,
    db: Optional[KnowledgeDB] = None,
    repo_ids: Optional[List[str]] = None,
    include_uncommitted: bool = True,
    progress_cb: Optional[Callable[[float, float, str], None]] = None
) -> Dict[str, Any]:
    """
    Строит радиус влияния (blast radius), находит связанные тесты,
    выявляет Test Gaps и рассчитывает риск-скоринг для ревьюера
    по всему диапазону merge-base..HEAD для всех затронутых репозиториев задачи.
    """
    db_instance = db or globals()["db"]
    if not db_instance:
        raise RuntimeError("KnowledgeDB is not initialized.")

    project_id = _resolve_project_id(project_id, branch, repo_ids or ([repo_id] if repo_id else None))
    cursor = db_instance.conn.cursor()

    # 0. Определение списка целевых репозиториев задачи/ветки (multi-repo scope)
    all_targets = resolve_workspace_targets(project_id)
    target_repos: List[str] = []

    if repo_ids:
        target_repos = list(repo_ids)
    elif repo_id:
        target_repos = [repo_id]
        if branch != "master":
            branch_targets = [
                t for t in all_targets
                if not t.get("is_doc") and _is_branch_match(branch, t.get("branch"), t.get("task_name"))
            ]
            for bt in branch_targets:
                bt_r = bt.get("repo_id")
                if bt_r and bt_r not in target_repos:
                    target_repos.append(bt_r)
    else:
        branch_targets = [
            t for t in all_targets
            if not t.get("is_doc") and _is_branch_match(branch, t.get("branch"), t.get("task_name"))
        ]
        if branch_targets:
            for bt in branch_targets:
                bt_r = bt.get("repo_id")
                if bt_r and bt_r not in target_repos:
                    target_repos.append(bt_r)
        else:
            target_repos = ["frontend", "backend", "e2e-tests"]

    primary_repo_id = target_repos[0] if target_repos else (repo_id or "default")
    if progress_cb:
        progress_cb(1.0, 5.0, f"Анализ git diff для репозиториев {target_repos}...")

    repo_git_headers: List[Dict[str, Any]] = []
    all_diff_files: List[Dict[str, Any]] = []
    all_norm_changed_files: List[str] = []
    all_changed_symbols: List[Dict[str, Any]] = []
    all_changed_sym_ids: Set[Any] = set()
    all_impacted_symbols: List[Dict[str, Any]] = []
    all_impacted_files: Set[str] = set()
    all_unique_file_impacts: Dict[str, Dict[str, Any]] = {}
    all_component_consumer_files: List[str] = []
    all_style_only_files: List[str] = []
    all_type_only_files: List[str] = []
    all_related_tests: List[Dict[str, Any]] = []
    all_test_gaps: List[Dict[str, Any]] = []
    all_e2e_covered_symbols: List[Dict[str, Any]] = []
    seen_impacted_ids: Set[Any] = set()
    seen_test_keys: Set[Any] = set()
    tested_symbol_ids: Set[Any] = set()

    COMMON_CODE_STOPWORDS = {
        't', '_', 'e', 'i', 'j', 'k', 's', 'x', 'y', 'z',
        'id', 'fn', 'cb', 'res', 'req', 'ctx', 'props', 'state',
        'item', 'row', 'data', 'val', 'key', 'err', 'opt', 'opts'
    }

    # 1. Сбор информации по каждому репозиторию задачи
    for cur_repo in target_repos:
        repo_path = _find_repo_path(project_id, cur_repo, branch)
        git_info = _get_repo_git_info(repo_path, branch, compare_branch, include_uncommitted)
        
        repo_git_headers.append({
            "repo_id": cur_repo,
            "branch": branch,
            "compare_branch": compare_branch,
            "merge_base_commit": git_info["merge_base_commit"],
            "head_commit": git_info["head_commit"],
            "commit_range": git_info["commit_range"],
            "commit_count": git_info["commit_count"],
            "commits": git_info["commits"],
            "branch_commit_shas": list(git_info["branch_commit_shas"]),
            "has_uncommitted_changes": git_info["has_uncommitted_changes"],
            "uncommitted_files": git_info["uncommitted_files"]
        })

        cur_diff_files: List[Dict[str, Any]] = []
        if changed_files is not None:
            if len(target_repos) == 1:
                cur_diff_files = [
                    {
                        "path": f.replace('\\', '/'),
                        "old_path": None,
                        "change_type": "modified",
                        "similarity": 100,
                        "is_uncommitted": False,
                        "repo_id": cur_repo
                    }
                    for f in changed_files
                ]
            else:
                raw_diff = _get_git_diff_with_renames(
                    repo_path, git_info["merge_base_commit"], git_info["head_commit"], include_uncommitted
                )
                cf_set = {cf.replace('\\', '/') for cf in changed_files}
                cur_diff_files = [f for f in raw_diff if f["path"] in cf_set or any(c in f["path"] for c in cf_set)]
        else:
            cur_diff_files = _get_git_diff_with_renames(
                repo_path, git_info["merge_base_commit"], git_info["head_commit"], include_uncommitted
            )

        # Fallback на CAS сравнение хешей, если git diff не дал результатов и список файлов не был задан явно
        if changed_files is None and not cur_diff_files:
            cursor.execute("""
                SELECT f1.path
                FROM files f1
                LEFT JOIN files f2 ON f2.project_id = f1.project_id AND f2.repo_id = f1.repo_id
                                  AND f2.branch = ? AND f2.path = f1.path
                WHERE f1.project_id = ? AND f1.repo_id = ? AND f1.branch = ?
                  AND (f2.id IS NULL OR f1.content_hash != f2.content_hash)
                  AND f1.path NOT LIKE '.git%'
                  AND f1.path NOT LIKE '.husky%'
                  AND f1.path NOT LIKE '%config.json'
            """, (compare_branch, project_id, cur_repo, branch))
            cur_diff_files = [
                {
                    "path": r['path'].replace('\\', '/'),
                    "old_path": None,
                    "change_type": "modified",
                    "similarity": 100,
                    "is_uncommitted": False,
                    "repo_id": cur_repo
                }
                for r in cursor.fetchall()
            ]

        for df in cur_diff_files:
            df["repo_id"] = cur_repo
        all_diff_files.extend(cur_diff_files)

        repo_norm_files = [df["path"] for df in cur_diff_files if df.get("change_type") != "deleted"]
        all_norm_changed_files.extend(repo_norm_files)

        # 2. Пакетное получение диапазонов измененных строк и поиск символов
        repo_diff_ranges: Dict[str, List[Tuple[int, int]]] = {}
        if repo_path and repo_path.exists():
            repo_diff_ranges = _get_repo_changed_line_ranges(
                repo_path, git_info["merge_base_commit"], git_info["head_commit"], include_uncommitted
            )

        repo_changed_symbols: List[Dict[str, Any]] = []
        for cf in repo_norm_files:
            if cf.endswith('.Designer.cs') or '.generated.' in cf.lower() or cf.endswith('.g.cs'):
                continue

            diff_ranges = repo_diff_ranges.get(cf)

            cursor.execute("""
                SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.line_start, s.line_end,
                       s.signature, s.language, f.path, f.repo_id, s.file_id
                FROM symbols s
                JOIN files f ON s.file_id = f.id
                WHERE f.project_id = ? AND f.repo_id = ? AND f.branch = ? AND f.path = ?
            """, (project_id, cur_repo, branch, cf))
            seen_syms = set()
            file_symbols = []
            for r in cursor.fetchall():
                if r['id'] not in seen_syms:
                    seen_syms.add(r['id'])
                    file_symbols.append(dict(r))

            for s in file_symbols:
                ls = s.get('line_start') or 0
                le = s.get('line_end') or 0
                overlaps = True
                if diff_ranges:
                    overlaps = any(not (le < r[0] or ls > r[1]) for r in diff_ranges)
                if overlaps:
                    s['location'] = f"{cf}:{ls}-{le}" if ls else cf
                    s['diff_ranges'] = [r for r in diff_ranges if not (le < r[0] or ls > r[1])] if diff_ranges else None
                    s['origin'] = 'branch'
                    s['introduced_in_commit'] = {'origin': 'branch', 'commit': (git_info.get('head_commit') or '')[:8]}
                    s['repo_id'] = cur_repo
                    repo_changed_symbols.append(s)
                    all_changed_symbols.append(s)
                    all_changed_sym_ids.add(s['id'])
                    seen_impacted_ids.add(s['id'])

        # 3. Построение графа радиуса влияния (Blast Radius) для репозитория
        if progress_cb:
            progress_cb(2.0, 5.0, f"Построение радиуса влияния для {cur_repo}...")
        for sym in repo_changed_symbols[:50]:
            k_lower = (sym.get('kind') or '').lower()
            if k_lower not in ('class', 'struct', 'interface', 'method', 'function', 'component', 'styled_component'):
                continue
            sym_impact = db_instance.get_impact_analysis(
                sym['id'], max_depth=max_depth, project_id=project_id, branch=branch
            )
            for item in sym_impact:
                caller_sym = dict(item['symbol']) if not isinstance(item['symbol'], dict) else item['symbol']
                caller_id = caller_sym['id']
                if caller_id not in seen_impacted_ids:
                    seen_impacted_ids.add(caller_id)
                    f_path = caller_sym.get('path') or caller_sym.get('file')
                    if f_path:
                        all_impacted_files.add(f_path)
                    all_impacted_symbols.append({
                        "id": caller_id,
                        "name": caller_sym['name'],
                        "qualified_name": caller_sym['qualified_name'],
                        "kind": caller_sym['kind'],
                        "file": f_path,
                        "line_start": caller_sym.get('line_start'),
                        "line_end": caller_sym.get('line_end'),
                        "location": f"{f_path}:{caller_sym.get('line_start')}" if f_path and caller_sym.get('line_start') else f_path,
                        "repo_id": caller_sym.get('repo_id') or cur_repo,
                        "edge_kind": item['edge_kind'],
                        "depth": item['depth']
                    })

            # Поиск прямых потребителей в коде воркспейса (для компонентов и ключевых классов)
            if repo_path and repo_path.exists() and len(sym['name']) >= 4:
                s_name = sym['name']
                if k_lower == 'component' or (k_lower == 'class' and s_name[0].isupper()):
                    try:
                        prefix = get_git_cmd_prefix(repo_path)
                        ext_args = ["--", "*.tsx", "*.ts", "*.jsx", "*.js"] if sym.get('language') in ('typescript', 'tsx', 'javascript') else ["--", f"*{Path(sym.get('path', '')).suffix}"]
                        grep_cmd = prefix + ["grep", "-l", "-E", f"<{s_name}\\b|\\b{s_name}\\b"] + ext_args
                        res = subprocess.run(grep_cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=1.5)
                        if res.returncode == 0 and res.stdout.strip():
                            for match_file in res.stdout.splitlines():
                                match_file = match_file.strip().replace('\\', '/')
                                if not match_file or match_file == sym.get('path') or match_file.endswith((f"{s_name}.tsx", f"{s_name}.styles.ts", f"{s_name}/index.ts")):
                                    continue
                                cursor.execute("""
                                    SELECT s.id, s.name, s.qualified_name, s.kind, s.line_start, s.line_end, f.path, f.repo_id
                                    FROM symbols s
                                    JOIN files f ON s.file_id = f.id
                                    WHERE f.path = ? AND s.kind IN ('component', 'function', 'class')
                                    ORDER BY s.id ASC LIMIT 1
                                """, (match_file,))
                                f_sym = cursor.fetchone()
                                if f_sym:
                                    c_id = f_sym['id']
                                    if c_id not in seen_impacted_ids:
                                        seen_impacted_ids.add(c_id)
                                        all_impacted_files.add(f_sym['path'])
                                        all_impacted_symbols.append({
                                            "id": c_id,
                                            "name": f_sym['name'],
                                            "qualified_name": f_sym['qualified_name'],
                                            "kind": f_sym['kind'],
                                            "file": f_sym['path'],
                                            "line_start": f_sym.get('line_start'),
                                            "line_end": f_sym.get('line_end'),
                                            "location": f"{f_sym['path']}:{f_sym.get('line_start')}" if f_sym.get('line_start') else f_sym['path'],
                                            "repo_id": f_sym['repo_id'],
                                            "edge_kind": "USES",
                                            "depth": 1
                                        })
                                else:
                                    stem_name = Path(match_file).stem
                                    fake_id = f"file:{match_file}"
                                    if fake_id not in seen_impacted_ids:
                                        seen_impacted_ids.add(fake_id)
                                        all_impacted_files.add(match_file)
                                        all_impacted_symbols.append({
                                            "id": fake_id,
                                            "name": stem_name,
                                            "qualified_name": stem_name,
                                            "kind": "component",
                                            "file": match_file,
                                            "line_start": 1,
                                            "line_end": 1,
                                            "location": f"{match_file}:1",
                                            "repo_id": cur_repo,
                                            "edge_kind": "USES",
                                            "depth": 1
                                        })
                    except Exception as e:
                        logger.debug(f"Callers search skipped: {e}")

        # 3.1. Классификация уникальных файлов в Blast Radius
        for item in all_impacted_symbols:
            f_p = item.get('file')
            if not f_p:
                continue
            if f_p not in all_unique_file_impacts:
                u_type = "component"
                if ".styles" in f_p or "styled" in (item.get('kind') or ''):
                    u_type = "styles"
                elif item.get('kind') in ('interface', 'type'):
                    u_type = "types"

                if repo_path and repo_path.exists():
                    full_item_path = repo_path / f_p
                    if full_item_path.exists():
                        try:
                            content_text = full_item_path.read_text(encoding='utf-8', errors='ignore')
                            has_jsx = any(f"<{s['name']}" in content_text for s in repo_changed_symbols if s.get('kind') in ('component', 'function'))
                            has_call = any(f"{s['name']}(" in content_text for s in repo_changed_symbols if s.get('kind') in ('function', 'method'))
                            if has_jsx or has_call:
                                u_type = "component"
                            elif (".styles" in content_text or "styled" in content_text) and not (has_jsx or has_call):
                                u_type = "styles"
                            elif any(f"I{s['name']}" in content_text for s in repo_changed_symbols):
                                u_type = "types"
                        except Exception:
                            pass

                all_unique_file_impacts[f_p] = {
                    "file": f_p,
                    "repo_id": item.get('repo_id') or cur_repo,
                    "usage_type": u_type,
                    "symbols": [],
                    "min_depth": item.get('depth', 1)
                }
            all_unique_file_impacts[f_p]["symbols"].append({
                "name": item.get('name'),
                "kind": item.get('kind'),
                "location": item.get('location'),
                "edge_kind": item.get('edge_kind')
            })        # 4. Поиск тестов для репозитория (пакетный запрос с фильтрацией в памяти)
        if progress_cb:
            progress_cb(3.0, 5.0, f"Поиск тестов для {cur_repo}...")
        cursor.execute("""
            SELECT DISTINCT f.id, f.path, f.repo_id
            FROM files f
            WHERE f.project_id = ?
              AND (f.path LIKE '%.spec.%' OR f.path LIKE '%.test.%' OR f.path LIKE '%Test.%' OR f.path LIKE '%Tests.%')
        """, (project_id,))
        raw_test_rows = [dict(r) for r in cursor.fetchall()]

        # Добавление новых, измененных и неотслеживаемых тестов из all_diff_files
        existing_test_paths = {r['path'].replace('\\', '/') for r in raw_test_rows}
        for df in all_diff_files:
            df_p = df["path"].replace('\\', '/')
            if df.get("change_type") != "deleted" and df_p not in existing_test_paths:
                p_lower = df_p.lower()
                if any(pat in p_lower for pat in ('.spec.', '.test.', 'tests.cs', 'test.cs')):
                    raw_test_rows.append({
                        "id": f"diff:{df_p}",
                        "path": df_p,
                        "repo_id": df.get("repo_id", cur_repo)
                    })
                    existing_test_paths.add(df_p)

        for cf in repo_norm_files:
            cf_path = Path(cf)
            stem = cf_path.stem
            candidates = [stem]
            if stem in ('index', 'styles', 'types', 'constants', 'utils') or len(stem) < 3:
                candidates.append(cf_path.parent.name)
            elif len(cf_path.parts) > 1:
                candidates.append(cf_path.parent.name)

            ext = cf_path.suffix.lower()
            allowed_exts = ('.ts', '.tsx', '.js', '.jsx') if ext in ('.ts', '.tsx', '.js', '.jsx') else (ext,)

            for cand in set(candidates):
                if len(cand) >= 3 and cand.lower() not in COMMON_CODE_STOPWORDS:
                    kebab = re.sub(r'(?<!^)(?=[A-Z])', '-', cand).lower()
                    test_needles = (
                        f"/{cand}.spec.", f"/{cand}.test.", f"/{cand}Test.",
                        f"/{cand}Tests.", f"/{kebab}.spec.", f"/{kebab}.test."
                    )
                    cand_lower = cand.lower()
                    kebab_lower = kebab.lower()

                    for tf in raw_test_rows:
                        tf_p = tf['path'].replace('\\', '/')
                        tf_repo = tf['repo_id']
                        if not (tf_repo == cur_repo or 'test' in tf_repo or 'e2e' in tf_repo):
                            continue
                        tf_path = Path(tf_p)
                        if tf_path.suffix.lower() not in allowed_exts:
                            continue

                        test_base = tf_path.name.lower()
                        is_match = (
                            any(n in tf_p for n in test_needles) or
                            (len(kebab_lower) >= 5 and (f"{kebab_lower}-" in test_base or f"{kebab_lower}_" in test_base or f"{cand_lower}-" in test_base or f"{cand_lower}_" in test_base))
                        )


                        if is_match:
                            k = tf_p
                            if k not in seen_test_keys:
                                seen_test_keys.add(k)
                                all_related_tests.append({
                                    "test_id": tf['id'],
                                    "test_name": tf_path.name,
                                    "test_file": tf_p,
                                    "repo_id": tf['repo_id'],
                                    "tested_symbol_id": None,
                                    "tested_symbol_name": cand,
                                    "relation": "feature_test_file"
                                })

        for sym in repo_changed_symbols:
            sym_id = sym['id']
            sym_name = sym['name']
            sym_qname = sym.get('qualified_name') or sym_name

            if len(sym_name) < 3 or sym_name.lower() in COMMON_CODE_STOPWORDS:
                continue

            # TESTED_BY ребра
            cursor.execute("""
                SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.signature, f.path, f.repo_id
                FROM symbol_edges e
                JOIN symbols s ON e.target_id = s.id
                JOIN files f ON s.file_id = f.id
                WHERE e.source_id = ? AND e.kind = 'TESTED_BY'
            """, (sym_id,))
            for tr in cursor.fetchall():
                tested_symbol_ids.add(sym_id)
                k = (tr['id'], sym_id)
                if k not in seen_test_keys:
                    seen_test_keys.add(k)
                    all_related_tests.append({
                        "test_id": tr['id'],
                        "test_name": tr['name'],
                        "test_file": tr['path'],
                        "repo_id": tr['repo_id'],
                        "tested_symbol_id": sym_id,
                        "tested_symbol_name": sym_name,
                        "relation": "TESTED_BY"
                    })

            # CALLS из тестов
            cursor.execute("""
                SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.signature, f.path, f.repo_id
                FROM symbol_edges e
                JOIN symbols s ON e.source_id = s.id
                JOIN files f ON s.file_id = f.id
                WHERE e.target_id = ? AND e.kind = 'CALLS'
                  AND (f.path LIKE '%/test/%' OR f.path LIKE '%/tests/%' OR f.path LIKE '%.test.%' OR f.path LIKE '%.spec.%' OR s.name LIKE '%test%' OR s.name LIKE '%Test%')
                  AND (f.repo_id = ? OR f.repo_id LIKE '%test%' OR f.repo_id LIKE '%e2e%')
            """, (sym_id, cur_repo))
            for tr in cursor.fetchall():
                tested_symbol_ids.add(sym_id)
                k = (tr['id'], sym_id)
                if k not in seen_test_keys:
                    seen_test_keys.add(k)
                    all_related_tests.append({
                        "test_id": tr['id'],
                        "test_name": tr['name'],
                        "test_file": tr['path'],
                        "repo_id": tr['repo_id'],
                        "tested_symbol_id": sym_id,
                        "tested_symbol_name": sym_name,
                        "relation": "CALLED_BY_TEST"
                    })

    # Сводные списки потребителей по типам
    all_component_consumer_files = [f for f, d in all_unique_file_impacts.items() if d['usage_type'] == 'component']
    all_style_only_files = [f for f, d in all_unique_file_impacts.items() if d['usage_type'] == 'styles']
    all_type_only_files = [f for f, d in all_unique_file_impacts.items() if d['usage_type'] == 'types']

    # Ключевые слова сквозных E2E тестов со всех репозиториев
    feature_test_keywords = set()
    for t in all_related_tests:
        t_file = (t.get('test_file') or '').lower().replace('\\', '/')
        is_e2e = (
            '.spec.' in t_file or '.e2e.' in t_file or '/e2e/' in t_file or 
            '/tests/e2e/' in t_file or '/specs/' in t_file or 'playwright' in t_file or 'cypress' in t_file
        )
        if is_e2e:
            c_name = t.get('tested_symbol_name')
            if c_name:
                feature_test_keywords.add(c_name.lower())
                feature_test_keywords.add(re.sub(r'(?<!^)(?=[A-Z])', '-', c_name).lower())
            t_stem = Path(t_file).stem.lower().replace('.spec', '').replace('.test', '').replace('.e2e', '')
            if len(t_stem) >= 3:
                feature_test_keywords.add(t_stem)
                clean_stem = t_stem.replace('-', '').replace('_', '')
                if len(clean_stem) >= 3:
                    feature_test_keywords.add(clean_stem)
                stem_parts = [p for p in re.split(r'[-_]', t_stem) if len(p) >= 4]
                for p in stem_parts:
                    feature_test_keywords.add(p)
                for i in range(len(stem_parts) - 1):
                    feature_test_keywords.add(f"{stem_parts[i]}-{stem_parts[i+1]}")
                    feature_test_keywords.add(f"{stem_parts[i]}{stem_parts[i+1]}")

    # 4.1. Анализ покрытия и выявление Test Gaps для каждого измененного символа
    
        k_lower = sym.get('kind', '').lower()
        sym_file = sym.get('path') or sym.get('file') or ''
        is_sym_in_test_file = any(pat in sym_file.lower() for pat in ('.spec.', '.test.', 'tests.cs', 'test.cs')) or '/tests/' in sym_file.lower() or '/test/' in sym_file.lower()
        if is_sym_in_test_file:
            continue

        if k_lower in ('method', 'function', 'component'):
            sym_name_clean = sym['name'].lower()
            sym_file_clean = (sym.get('path') or sym.get('file') or '').lower()
            
            is_method_tested = False
            is_e2e_covered = False

            if sym['id'] in tested_symbol_ids:
                is_method_tested = True
            else:
                for t in all_related_tests:
                    if t.get('tested_symbol_id') == sym['id'] or (t.get('tested_symbol_name') or '').lower() == sym_name_clean:
                        is_method_tested = True
                        break
                    t_name = (t.get('test_name') or '').lower()
                    if (f"test_{sym_name_clean}" in t_name or f"test{sym_name_clean}" in t_name or
                        f"{sym_name_clean}_" in t_name or f"{sym_name_clean}should" in t_name or f"{sym_name_clean}when" in t_name):
                        is_method_tested = True
                        break

            # Транзитивное покрытие E2E
            transitive_test_sources = []
            if not is_method_tested:
                for kw in feature_test_keywords:
                    if kw in sym_file_clean or kw in sym_name_clean:
                        is_e2e_covered = True
                        transitive_test_sources.append(kw)
                        break

                if not is_e2e_covered:
                    for cf_item in all_component_consumer_files:
                        cf_lower = cf_item.lower()
                        for kw in feature_test_keywords:
                            if kw in cf_lower:
                                is_e2e_covered = True
                                transitive_test_sources.append(f"{Path(cf_item).name} ({kw})")
                                break
                        if is_e2e_covered:
                            break

            sym_file = sym.get('path') or sym.get('file')
            sym_loc = f"{sym_file}:{sym.get('line_start')}-{sym.get('line_end')}" if sym.get('line_start') else sym_file
            qname = sym.get('qualified_name') or sym['name']

            if is_e2e_covered:
                all_e2e_covered_symbols.append({
                    "id": sym['id'],
                    "name": sym['name'],
                    "qualified_name": qname,
                    "kind": sym['kind'],
                    "file": sym_file,
                    "line_start": sym.get('line_start'),
                    "line_end": sym.get('line_end'),
                    "location": sym_loc,
                    "origin": sym.get('origin', 'branch'),
                    "introduced_in_commit": sym.get('introduced_in_commit'),
                    "coverage_type": "transitive_e2e" if transitive_test_sources else "feature_e2e",
                    "covered_via": transitive_test_sources or ["feature_test_file"]
                })
            elif not is_method_tested:
                all_test_gaps.append({
                    "id": sym['id'],
                    "name": sym['name'],
                    "qualified_name": qname,
                    "kind": sym['kind'],
                    "file": sym_file,
                    "line_start": sym.get('line_start'),
                    "line_end": sym.get('line_end'),
                    "location": sym_loc,
                    "origin": sym.get('origin', 'branch'),
                    "introduced_in_commit": sym.get('introduced_in_commit'),
                    "reason": "Отсутствуют unit/e2e тесты для данного символа"
                })

            # Универсальное выявление режимов вызова (Mode Gaps)
            sym_repo = sym.get('repo_id') or primary_repo_id
            s_repo_path = _find_repo_path(project_id, sym_repo, branch)
            mode_gaps = _extract_call_modes_and_gaps(
                sym=sym,
                component_consumer_files=all_component_consumer_files,
                repo_path=s_repo_path,
                tested_symbol_ids=tested_symbol_ids,
                feature_test_keywords=feature_test_keywords,
                related_tests=all_related_tests,
                code_parser=code_parser
            )
            for mg in mode_gaps:
                mg_hdr = next((h for h in repo_git_headers if h["repo_id"] == sym_repo), {})
                mg_orig = _get_finding_origin(s_repo_path, mg["file"], mg.get("line_start"), set(mg_hdr.get("branch_commit_shas") or []))
                mg["origin"] = mg_orig["origin"]
                mg["introduced_in_commit"] = mg_orig
                all_test_gaps.append(mg)

    # Проверка правил чек-листов и выявление дефектов кода (чек-листы .kb)
    if progress_cb:
        progress_cb(4.0, 5.0, "Аудит правил чек-листов и расчет риск-скоринга...")
    all_code_findings: List[Dict[str, Any]] = []
    for cur_repo in target_repos:
        repo_path = _find_repo_path(project_id, cur_repo, branch)
        repo_git_header = next((h for h in repo_git_headers if h["repo_id"] == cur_repo), {})
        repo_diff = [df for df in all_diff_files if df.get("repo_id") == cur_repo]
        cf_findings, cf_gaps = _audit_checklist_rules(
            repo_path=repo_path,
            repo_id=cur_repo,
            diff_files=repo_diff,
            branch_commit_shas=repo_git_header.get("branch_commit_shas") or set(),
            compare_branch=compare_branch,
            base_sha=repo_git_header.get("merge_base_commit"),
            head_sha=repo_git_header.get("head_commit")
        )
        all_code_findings.extend(cf_findings)
        all_test_gaps.extend(cf_gaps)

    # Фильтрация валидных замечаний (Rule 7: без привязки к коду — не замечание)
    valid_test_gaps = _filter_valid_findings(all_test_gaps)

    # Формирование обязательного списка отрецензированных файлов (Rule 2)
    reviewed_files = _build_reviewed_files_audit(all_diff_files, all_code_findings, valid_test_gaps, all_changed_symbols)

    # 5. Расчет прозрачного скоринга риска (Rule 6)
    risk_breakdown = _calculate_transparent_risk_score(
        norm_changed_files=all_norm_changed_files,
        changed_symbols=all_changed_symbols,
        unique_file_impacts=all_unique_file_impacts,
        component_consumer_files=all_component_consumer_files,
        style_only_files=all_style_only_files,
        type_only_files=all_type_only_files,
        test_gaps=valid_test_gaps,
        e2e_covered_symbols=all_e2e_covered_symbols,
        related_tests=all_related_tests
    )

    # 6. Экономия контекста токенов
    all_affected_files = set(all_norm_changed_files) | set(all_impacted_files) | {t['test_file'] for t in all_related_tests}
    savings = _calculate_savings(
        {
            "changed_symbols": all_changed_symbols,
            "blast_radius": all_impacted_symbols,
            "related_tests": all_related_tests,
            "test_gaps": valid_test_gaps
        },
        list(all_affected_files),
        db_instance=db_instance,
        default_file_tokens=2000
    )

    db_instance.log_token_savings(
        tool_name="knowledge_review_context",
        baseline_tokens=savings["baseline_tokens"],
        returned_tokens=savings["returned_tokens"],
        project_id=project_id,
        repo_id=primary_repo_id,
        branch=branch
    )

    header = {
        "repositories": repo_git_headers,
        "include_uncommitted": include_uncommitted,
        "reviewed_files_summary": {
            "total_files": len(reviewed_files),
            "clean_files": len([f for f in reviewed_files if f["status"] == "clean"]),
            "files_with_findings": len([f for f in reviewed_files if f["status"] == "has_findings"])
        }
    }

    return {
        "header": header,
        "reviewed_files": reviewed_files,
        "findings": all_code_findings,
        "repo_id": primary_repo_id,
        "repo_ids": target_repos,
        "project_id": project_id,
        "branch": branch,
        "compare_branch": compare_branch,
        "changed_files": all_norm_changed_files,
        "changed_symbols": all_changed_symbols,
        "blast_radius": {
            "total_unique_files_count": len(all_unique_file_impacts),
            "component_consumer_files_count": len(all_component_consumer_files),
            "style_or_type_only_files_count": len(all_style_only_files) + len(all_type_only_files),
            "unique_files": list(all_unique_file_impacts.values()),
            "component_files": all_component_consumer_files,
            "style_only_files": all_style_only_files,
            "type_only_files": all_type_only_files,
            "impacted_symbols_count": len(all_impacted_symbols),
            "impacted_files_count": len(all_impacted_files),
            "callers": all_impacted_symbols,
            "impacted_files": list(all_impacted_files),
            "max_depth": max_depth
        },
        "related_tests": all_related_tests,
        "test_gaps": valid_test_gaps,
        "e2e_covered_symbols": all_e2e_covered_symbols,
        "risk": risk_breakdown["level"],
        "risk_level": risk_breakdown["level"],
        "risk_score": risk_breakdown["score"],
        "risk_breakdown": risk_breakdown,
        "context_savings": savings
    }


def knowledge_find_tests(
    target: str,
    repo_id: Optional[str] = None,
    project_id: str = "default",
    branch: str = "master",
    db: Optional[KnowledgeDB] = None
) -> Dict[str, Any]:
    """Находит все тесты, связанные с указанным методом, классом или файлом."""
    db_instance = db or globals()["db"]
    if not db_instance:
        raise RuntimeError("KnowledgeDB is not initialized.")

    cursor = db_instance.conn.cursor()
    tests = []
    seen_keys = set()

    # Поиск символа-цели
    sql = """
        SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.signature, f.path, f.repo_id
        FROM symbols s
        JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
        WHERE (s.name = ? OR s.qualified_name = ? OR f.path = ?)
    """
    params: list[Any] = [target, target, target]
    if repo_id:
        sql += " AND f.repo_id = ?"
        params.append(repo_id)
    if project_id is not None:
        sql += " AND f.project_id = ?"
        params.append(project_id)
    if branch is not None:
        sql += " AND f.branch = ?"
        params.append(branch)

    cursor.execute(sql, params)
    target_symbols = cursor.fetchall()

    for sym in target_symbols:
        sym_id = sym['id']
        sym_name = sym['name']
        sym_qname = sym['qualified_name'] or sym_name

        # Связи TESTED_BY
        cursor.execute("""
            SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.signature, f.path, f.repo_id
            FROM symbol_edges e
            JOIN symbols s ON e.target_id = s.id
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE e.source_id = ? AND e.kind = 'TESTED_BY'
        """, (sym_id,))
        for tr in cursor.fetchall():
            if tr['id'] not in seen_keys:
                seen_keys.add(tr['id'])
                tests.append({
                    "id": tr['id'],
                    "name": tr['name'],
                    "qualified_name": tr['qualified_name'],
                    "kind": tr['kind'],
                    "signature": tr['signature'],
                    "path": tr['path'],
                    "repo_id": tr['repo_id'],
                    "relation": "TESTED_BY"
                })

        # Обратные связи
        cursor.execute("""
            SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.signature, f.path, f.repo_id
            FROM symbol_edges e
            JOIN symbols s ON e.source_id = s.id
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE e.target_id = ? AND e.kind = 'TESTED_BY'
        """, (sym_id,))
        for tr in cursor.fetchall():
            if tr['id'] not in seen_keys:
                seen_keys.add(tr['id'])
                tests.append({
                    "id": tr['id'],
                    "name": tr['name'],
                    "qualified_name": tr['qualified_name'],
                    "kind": tr['kind'],
                    "signature": tr['signature'],
                    "path": tr['path'],
                    "repo_id": tr['repo_id'],
                    "relation": "TESTED_BY"
                })

        # Неразрешенные ссылки
        cursor.execute("""
            SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.signature, f.path, f.repo_id
            FROM unresolved_refs u
            JOIN symbols s ON u.source_id = s.id
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE u.kind = 'TESTED_BY' AND (u.target_qualified_name = ? OR u.target_qualified_name = ?)
        """, (sym_name, sym_qname))
        for tr in cursor.fetchall():
            if tr['id'] not in seen_keys:
                seen_keys.add(tr['id'])
                tests.append({
                    "id": tr['id'],
                    "name": tr['name'],
                    "qualified_name": tr['qualified_name'],
                    "kind": tr['kind'],
                    "signature": tr['signature'],
                    "path": tr['path'],
                    "repo_id": tr['repo_id'],
                    "relation": "TESTED_BY_UNRESOLVED"
                })

    # Поиск по эвристикам имен (Test*, *Tests, test_*)
    base_target = Path(target).stem if ('/' in target or '\\' in target) else target
    base_target = base_target.split('.')[-1]

    naming_sql = """
        SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.signature, f.path, f.repo_id
        FROM symbols s
        JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
        WHERE (s.name = ? OR s.name = ? OR s.name = ? OR s.name = ? OR s.name = ? OR s.name LIKE ? ESCAPE '/')
    """
    naming_params = [
        f"Test{base_target}",
        f"{base_target}Test",
        f"{base_target}Tests",
        f"test_{base_target.lower()}",
        f"Test_{base_target}",
        f"{base_target}/_%"
    ]
    if repo_id:
        naming_sql += " AND (f.repo_id = ? OR f.repo_id LIKE '%test%' OR f.repo_id LIKE '%e2e%')"
        naming_params.append(repo_id)
    if project_id is not None:
        naming_sql += " AND f.project_id = ?"
        naming_params.append(project_id)
    if branch is not None:
        naming_sql += " AND f.branch = ?"
        naming_params.append(branch)

    cursor.execute(naming_sql, naming_params)
    for tr in cursor.fetchall():
        if tr['id'] not in seen_keys:
            seen_keys.add(tr['id'])
            tests.append({
                "id": tr['id'],
                "name": tr['name'],
                "qualified_name": tr['qualified_name'],
                "kind": tr['kind'],
                "signature": tr['signature'],
                "path": tr['path'],
                "repo_id": tr['repo_id'],
                "relation": "naming_pattern"
            })

    files_involved = [t['path'] for t in tests if 'path' in t]
    savings = _calculate_savings(tests, files_involved, db_instance=db_instance)
    db_instance.log_token_savings(
        tool_name="knowledge_find_tests",
        baseline_tokens=savings["baseline_tokens"],
        returned_tokens=savings["returned_tokens"],
        project_id=project_id,
        repo_id=repo_id,
        branch=branch
    )

    return {
        "target": target,
        "repo_id": repo_id,
        "project_id": project_id,
        "branch": branch,
        "tests": tests,
        "total_tests": len(tests),
        "context_savings": savings
    }


def knowledge_get_project_rules(
    topic: Optional[str] = None,
    project_id: str = "default",
    db: Optional[KnowledgeDB] = None
) -> Dict[str, Any]:
    """
    Извлекает правила проекта из документов source_kind='rules' (CLAUDE.md, .cursorrules, .kb/).
    При наличии topic фильтрует релевантные разделы и рассчитывает экономию токенов.
    """
    db_instance = db or globals()["db"]
    if not db_instance:
        raise RuntimeError("KnowledgeDB is not initialized.")

    cursor = db_instance.conn.cursor()

    # Читаем чанки правил из CAS хранилища
    sql = """
        SELECT DISTINCT c.id, c.content, c.line_start, c.line_end, c.source_kind, c.trust,
               f.repo_id, f.path, f.project_id, f.branch
        FROM chunks c
        JOIN files f ON (c.file_id = f.id OR (c.content_hash IS NOT NULL AND c.content_hash = f.content_hash))
        WHERE (c.source_kind IN ('rules', 'rule')
               OR f.path LIKE '%.kb/%'
               OR f.path LIKE '%CLAUDE.md%'
               OR f.path LIKE '%.cursorrules%')
    """
    params: list[Any] = []
    if project_id != "default":
        sql += " AND f.project_id = ?"
        params.append(project_id)

    cursor.execute(sql, params)
    all_rules = [dict(r) for r in cursor.fetchall()]

    # Fallback на файловую систему, если в БД еще ничего не проиндексировано
    if not all_rules:
        targets = resolve_workspace_targets(project_id)
        from knowledge_mcp.core.markdown_parser import MarkdownParser
        md_parser = MarkdownParser()
        fs_rules = []
        for t in targets:
            abs_p = t.get("absolute_path")
            if abs_p and Path(abs_p).is_file() and (abs_p.name.lower() in ('claude.md', '.cursorrules') or '.kb' in str(abs_p)):
                sections = md_parser.parse_file(Path(abs_p))
                for idx, sec in enumerate(sections):
                    fs_rules.append({
                        "id": f"fs_{abs_p.name}_{idx}",
                        "content": sec.content,
                        "line_start": sec.line_start,
                        "line_end": sec.line_end,
                        "repo_id": t.get("repo_id", "docs"),
                        "path": str(abs_p.name),
                        "project_id": project_id,
                        "branch": t.get("branch", "master")
                    })
        all_rules = fs_rules

    # Базовая оценка: полный файл CLAUDE.md (76 КБ ≈ 19 456 токенов)
    total_rules_tokens = sum(max(1, len(r['content']) // 4) for r in all_rules)
    baseline_tokens = max(19456, total_rules_tokens) if all_rules else 19456

    # Фильтрация по теме (topic)
    if topic:
        topic_clean = topic.strip().lower()
        filtered_rules = []
        for r in all_rules:
            c_lower = r['content'].lower()
            p_lower = r['path'].lower()
            if topic_clean in c_lower or topic_clean in p_lower:
                filtered_rules.append(r)

        # Если прямого совпадения нет, пробуем ключевые слова
        if not filtered_rules:
            words = [w for w in re.split(r'\W+', topic_clean) if len(w) > 2]
            if words:
                for r in all_rules:
                    c_lower = r['content'].lower()
                    if any(w in c_lower for w in words):
                        filtered_rules.append(r)
    else:
        filtered_rules = all_rules

    returned_tokens = sum(max(1, len(r['content']) // 4) for r in filtered_rules)
    if not returned_tokens:
        returned_tokens = 50

    saved_tokens = max(0, baseline_tokens - returned_tokens)
    saved_percent = round((saved_tokens / baseline_tokens * 100)) if baseline_tokens > 0 else 0

    savings = {
        "saved_tokens": saved_tokens,
        "saved_percent": saved_percent,
        "baseline_tokens": baseline_tokens,
        "returned_tokens": returned_tokens
    }

    db_instance.log_token_savings(
        tool_name="knowledge_get_project_rules",
        baseline_tokens=baseline_tokens,
        returned_tokens=returned_tokens,
        project_id=project_id
    )

    return {
        "project_id": project_id,
        "topic": topic,
        "rules": [
            {
                "id": r['id'],
                "source": f"{r.get('repo_id', 'docs')}/{r.get('path', 'CLAUDE.md')}",
                "path": r.get('path', 'CLAUDE.md'),
                "repo_id": r.get('repo_id', 'docs'),
                "content": r['content'],
                "line_start": r.get('line_start', 1),
                "line_end": r.get('line_end', 1)
            }
            for r in filtered_rules
        ],
        "total_rules": len(filtered_rules),
        "context_savings": savings
    }


# ──────────────────────────────────────────────────────────────────────────────
# Обработчики инструментов MCP протокола
# ──────────────────────────────────────────────────────────────────────────────

def _err(msg: str) -> list[types.TextContent]:
    """Хелпер: возвращает текстовое сообщение об ошибке."""
    return [types.TextContent(type="text", text=msg)]


async def _handle_search(arguments: dict) -> list[types.TextContent]:
    """knowledge_search: гибридный поиск FTS5 + Vector KNN + Graph с JIT и context_savings."""
    query = arguments.get("query")
    if not query:
        return _err("Error: 'query' argument is required.")

    repo_ids = arguments.get("repo_ids", [])
    raw_project_id = arguments.get("project_id")
    branch = arguments.get("branch", "master")
    limit = arguments.get("limit", 10)
    project_id = _resolve_project_id(raw_project_id, branch, repo_ids)

    try:
        vector = None
        if use_embeddings:
            global embedder
            if embedder is None:
                logger.info("Lazy loading LocalEmbedder in background thread (~30s first time)...")
                embedder = await asyncio.to_thread(_init_embedder_sync)
                logger.info("Embedder ready!")
            vector = await asyncio.to_thread(_embed_text_sync, embedder, query)

        res = await asyncio.to_thread(
            knowledge_search,
            query=query,
            repo_ids=repo_ids,
            project_id=project_id,
            branch=branch,
            limit=limit,
            query_vector=vector,
            db=db
        )

        results = res.get("results", [])
        savings = res.get("context_savings", {})

        if not results:
            return _err("No results found in the knowledge base. Try altering your query or removing repo_ids limits.")

        MAX_CHUNK_LINES = 80
        MAX_CHUNK_CHARS = 2500
        formatted = []
        for r in results:
            content = r.get("content", "")
            lines = content.splitlines()
            if len(lines) > MAX_CHUNK_LINES or len(content) > MAX_CHUNK_CHARS:
                truncated_content = "\n".join(lines[:MAX_CHUNK_LINES])[:MAX_CHUNK_CHARS]
                rem_lines = max(0, len(lines) - MAX_CHUNK_LINES)
                rem_chars = max(0, len(content) - len(truncated_content))
                content = (
                    f"{truncated_content}\n\n"
                    f"... [Truncated {rem_lines} lines ({rem_chars} chars). "
                    f"Use knowledge_read_file(path='{r.get('path')}', start_line={r.get('line_start')}, end_line={r.get('line_end')}) "
                    f"or knowledge_get_chunk(chunk_id='{r['id']}') for complete content]"
                )

            formatted.append(
                f"--- Chunk ID: {r['id']} ---\n"
                f"**Repository**: `{r['repo_id']}`\n"
                f"**File**: `{r['path']}` (Lines: {r.get('line_start')}-{r.get('line_end')})\n"
                f"**Context Trust Level**: {r.get('trust')} (Source: {r.get('source_kind')})\n\n"
                f"```\n{content}\n```\n"
            )

        header = f"<!-- context_savings: {json.dumps(savings)} -->\n[Context Savings: {savings['saved_tokens']} tokens saved ({savings['saved_percent']}% reduction)]\n\n"
        return [types.TextContent(type="text", text=header + "\n".join(formatted))]

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return _err(f"Knowledge Base DB Error: {e}")


async def _handle_get_chunk(arguments: dict) -> list[types.TextContent]:
    """knowledge_get_chunk: получить сырой контент чанка по ID."""
    chunk_id = arguments.get("chunk_id")
    if not chunk_id:
        return _err("Error: 'chunk_id' argument is required.")

    try:
        row = db.get_chunk_by_id(chunk_id)
        if not row:
            return _err(f"Chunk '{chunk_id}' not found.")

        text = (
            f"**Repository**: `{row['repo_id']}`\n"
            f"**File**: `{row['path']}`\n"
            f"**Trust Level**: {row['trust']}\n"
            f"**File Hash**: {row['sha']}\n\n"
            f"```\n{row['content']}\n```"
        )
        return [types.TextContent(type="text", text=text)]

    except Exception as e:
        logger.error(f"Get chunk failed: {e}")
        return _err(f"Database query error: {e}")


async def _handle_read_file(arguments: dict) -> list[types.TextContent]:
    """knowledge_read_file: получить полный контент файла по пути."""
    path = arguments.get("path") or arguments.get("file_path")
    if not path:
        return _err("Error: 'path' argument is required.")

    repo_id = arguments.get("repo_id")
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")

    try:
        content = await asyncio.to_thread(
            knowledge_read_file, path, repo_id=repo_id, project_id=project_id, branch=branch, db=db
        )
        if content is None:
            return _err(f"File '{path}' not found in the knowledge base.")

        return [types.TextContent(type="text", text=content)]
    except Exception as e:
        logger.error(f"Read file failed: {e}")
        return _err(f"Database query error: {e}")


def _get_auth_headers() -> dict[str, str]:
    """Возвращает заголовок Authorization для Basic Auth, если установлен DASHBOARD_PASSWORD."""
    import os
    import base64
    password = os.environ.get("DASHBOARD_PASSWORD")
    if not password:
        return {}
    creds = f"admin:{password}"
    b64_creds = base64.b64encode(creds.encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {b64_creds}"}


def _get_daemon_base_url() -> str:
    """Возвращает базовый URL Daemon API из переменной окружения DAEMON_URL."""
    import os
    return os.environ.get("DAEMON_URL", "http://127.0.0.1:8000").rstrip("/")


async def _handle_delete_repo(arguments: dict) -> list[types.TextContent]:
    """knowledge_delete_repo: делегирует удаление репозитория Daemon-процессу через HTTP."""
    repo_id = arguments.get("repo_id")
    if not repo_id:
        return _err("Error: 'repo_id' argument is required.")

    try:
        import urllib.request

        base_url = _get_daemon_base_url()
        url = f"{base_url}/api/repo/{repo_id}"
        headers = _get_auth_headers()
        req = urllib.request.Request(url, method="DELETE", headers=headers)

        def _trigger_delete():
            with urllib.request.urlopen(req) as response:
                return response.read()

        logger.info(f"Delegating delete of '{repo_id}' to Daemon API via DELETE {url}")
        await asyncio.to_thread(_trigger_delete)

        return [types.TextContent(type="text", text=f"Successfully requested deletion of repository '{repo_id}' via Daemon API.")]
    except Exception as e:
        logger.error(f"Delete repo delegation failed: {e}")
        return _err(f"Failed to delegate repository deletion to Daemon: {e}")


async def _handle_sync_repo(arguments: dict) -> list[types.TextContent]:
    """knowledge_sync_repo: делегирует синхронизацию репозитория Daemon-процессу через HTTP."""
    repo_id = arguments.get("repo_id")
    if not repo_id:
        return _err("Error: 'repo_id' argument is required.")

    try:
        import urllib.request

        base_url = _get_daemon_base_url()
        url = f"{base_url}/api/sync/{repo_id}"
        headers = _get_auth_headers()
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, method="POST", data=b"", headers=headers)

        def _trigger_sync():
            with urllib.request.urlopen(req) as response:
                return response.read()

        logger.info(f"Delegating sync of '{repo_id}' to Daemon API via POST {url}")
        await asyncio.to_thread(_trigger_sync)

        return [types.TextContent(type="text", text=f"Successfully queued sync for repository '{repo_id}' via Daemon API.")]
    except Exception as e:
        logger.error(f"Sync trigger failed: {e}")
        return _err(f"Failed to delegate repository sync to Daemon: {e}")


async def _handle_find_symbol(arguments: dict) -> list[types.TextContent]:
    """knowledge_find_symbol: поиск символа по имени или паттерну."""
    pattern = arguments.get("name_pattern")
    if not pattern:
        return _err("Error: 'name_pattern' argument is required.")

    kind = arguments.get("kind")
    repo_ids = arguments.get("repo_ids", [])
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")
    limit = arguments.get("limit", 20)

    try:
        res = await asyncio.to_thread(
            knowledge_find_symbol,
            name_pattern=pattern,
            kind=kind,
            repo_ids=repo_ids,
            project_id=project_id,
            branch=branch,
            limit=limit,
            db=db
        )
        results = res.get("symbols", [])
        savings = res.get("context_savings", {})

        if not results:
            return _err(f"No symbols found matching '{pattern}'.")

        formatted = [
            f"--- Symbol ID: {r['id']} ---\n"
            f"**Name**: `{r['name']}` (Qualified: `{r['qualified_name']}`)\n"
            f"**Repository**: `{r['repo_id']}`\n"
            f"**File**: `{r['path']}` (Lines: {r['line_start']}-{r['line_end']})\n"
            f"**Kind**: {r['kind']} ({r['language']})\n"
            f"**Signature**: `{r['signature']}`\n"
            for r in results
        ]
        header = f"<!-- context_savings: {json.dumps(savings)} -->\n[Context Savings: {savings['saved_tokens']} tokens saved ({savings['saved_percent']}% reduction)]\n\n"
        return [types.TextContent(type="text", text=header + "\n".join(formatted))]
    except Exception as e:
        logger.error(f"Find symbol failed: {e}")
        return _err(f"Database query error: {e}")


async def _handle_get_callers(arguments: dict) -> list[types.TextContent]:
    """knowledge_get_callers: кто вызывает данный символ."""
    symbol_id = arguments.get("symbol_id")
    if not symbol_id:
        return _err("Error: 'symbol_id' is required.")
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")

    try:
        res = await asyncio.to_thread(knowledge_get_callers, symbol_id, project_id=project_id, branch=branch, db=db)
        results = res.get("callers", [])
        savings = res.get("context_savings", {})

        if not results:
            return _err(f"No graph edges found for symbol {symbol_id} in this context.")

        header = f"<!-- context_savings: {json.dumps(savings)} -->\n[Context Savings: {savings['saved_tokens']} tokens saved ({savings['saved_percent']}% reduction)]\n\n"
        lines = [header + f"Callers for symbol ID {symbol_id}:"] + [
            f"- [{r['edge_kind']}] ID: {r['id']} | `{r['qualified_name']}` ({r['kind']} in {r['repo_id']}/{r['path']})"
            for r in results
        ]
        return [types.TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        return _err(f"Database query error: {e}")


async def _handle_get_callees(arguments: dict) -> list[types.TextContent]:
    """knowledge_get_callees: что вызывает данный символ."""
    symbol_id = arguments.get("symbol_id")
    if not symbol_id:
        return _err("Error: 'symbol_id' is required.")
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")

    try:
        res = await asyncio.to_thread(knowledge_get_callees, symbol_id, project_id=project_id, branch=branch, db=db)
        results = res.get("callees", [])
        savings = res.get("context_savings", {})

        if not results:
            return _err(f"No graph edges found for symbol {symbol_id} in this context.")

        header = f"<!-- context_savings: {json.dumps(savings)} -->\n[Context Savings: {savings['saved_tokens']} tokens saved ({savings['saved_percent']}% reduction)]\n\n"
        lines = [header + f"Callees of symbol ID {symbol_id}:"] + [
            f"- [{r['edge_kind']}] ID: {r['id']} | `{r['qualified_name']}` ({r['kind']} in {r['repo_id']}/{r['path']})"
            for r in results
        ]
        return [types.TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        return _err(f"Database query error: {e}")


async def _handle_get_hierarchy(arguments: dict) -> list[types.TextContent]:
    """knowledge_get_hierarchy: иерархия наследования символа."""
    symbol_id = arguments.get("symbol_id")
    if not symbol_id:
        return _err("Error: 'symbol_id' is required.")
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")

    try:
        res = await asyncio.to_thread(knowledge_get_hierarchy, symbol_id, project_id=project_id, branch=branch, db=db)
        results = res.get("hierarchy", [])
        savings = res.get("context_savings", {})

        if not results:
            return _err(f"No graph edges found for symbol {symbol_id} in this context.")

        header = f"<!-- context_savings: {json.dumps(savings)} -->\n[Context Savings: {savings['saved_tokens']} tokens saved ({savings['saved_percent']}% reduction)]\n\n"
        lines = [header + f"Hierarchy for symbol ID {symbol_id}:"] + [
            f"- [{r['edge_kind']}] ID: {r['id']} | `{r['qualified_name']}` ({r['kind']} in {r['repo_id']}/{r['path']})"
            for r in results
        ]
        return [types.TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        return _err(f"Database query error: {e}")


async def _handle_impact_analysis(arguments: dict) -> list[types.TextContent]:
    """knowledge_impact_analysis: рекурсивный анализ зависимостей."""
    symbol_id = arguments.get("symbol_id")
    max_depth = arguments.get("max_depth", 3)
    if not symbol_id:
        return _err("Error: 'symbol_id' is required.")
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")

    try:
        res = await asyncio.to_thread(knowledge_impact_analysis, symbol_id, max_depth=max_depth, project_id=project_id, branch=branch, db=db)
        results = res.get("impacted", [])
        savings = res.get("context_savings", {})

        if not results:
            return _err(f"No dependent symbols found for symbol {symbol_id} within depth {max_depth}.")

        header = f"<!-- context_savings: {json.dumps(savings)} -->\n[Context Savings: {savings['saved_tokens']} tokens saved ({savings['saved_percent']}% reduction)]\n\n"
        lines = [header + f"Impact Analysis for symbol ID {symbol_id} (max depth {max_depth}):"]
        current_depth = 0
        for item in sorted(results, key=lambda x: x['depth']):
            if item['depth'] != current_depth:
                current_depth = item['depth']
                lines.append(f"\n--- Depth {current_depth} Dependencies ---")
            lines.append(
                f"- [{item['edge_kind']}] ID: {item['id']} | `{item['qualified_name']}` ({item['kind']} in {item['repo_id']}/{item['path']})"
            )
        return [types.TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error(f"Impact analysis failed: {e}")
        return _err(f"Database query error: {e}")


async def _handle_review_context(arguments: dict) -> list[types.TextContent]:
    """knowledge_review_context: построение радиуса влияния и анализ рисков для ревьюера."""
    repo_id = arguments.get("repo_id")
    repo_ids = arguments.get("repo_ids")
    if isinstance(repo_ids, str):
        repo_ids = [repo_ids]
    raw_project_id = arguments.get("project_id")
    branch = arguments.get("branch", "master")
    compare_branch = arguments.get("compare_branch", "master")
    changed_files = arguments.get("changed_files")
    file_path = arguments.get("file_path") or arguments.get("path")
    if file_path and not changed_files:
        changed_files = [file_path]
    max_depth = arguments.get("max_depth", 2)
    include_uncommitted = arguments.get("include_uncommitted", True)
    project_id = _resolve_project_id(raw_project_id, branch, repo_ids or ([repo_id] if repo_id else None))

    if not repo_id and not repo_ids and changed_files:
        targets = resolve_workspace_targets(project_id)
        for cf in changed_files:
            cf_norm = cf.replace('\\', '/').strip('/')
            for t in targets:
                if not t.get("is_doc") and _is_branch_match(branch, t.get("branch"), t.get("task_name")):
                    t_root = Path(t["absolute_path"]) if t.get("absolute_path") else None
                    if t_root and (t_root / cf_norm).exists():
                        repo_id = t.get("repo_id")
                        break
            if repo_id:
                break

    session = None
    progress_token = None
    try:
        req_ctx = server.request_context
        if req_ctx:
            session = req_ctx.session
            meta = req_ctx.meta
            if meta:
                progress_token = getattr(meta, "progressToken", None) or getattr(meta, "progress_token", None)
    except Exception:
        pass

    loop = asyncio.get_running_loop()

    def sync_progress_callback(progress: float, total: float, message: str):
        if session:
            async def _send():
                try:
                    if progress_token is not None:
                        await session.send_progress_notification(
                            progress_token=progress_token,
                            progress=progress,
                            total=total,
                            message=message
                        )
                    await session.send_log_message(level="info", data=f"[{int(progress)}/{int(total)}] {message}")
                except Exception as ex:
                    logger.debug(f"Progress notification error: {ex}")
            asyncio.run_coroutine_threadsafe(_send(), loop)

    t_start = time.time()
    try:
        res = await asyncio.wait_for(
            asyncio.to_thread(
                knowledge_review_context,
                repo_id=repo_id,
                repo_ids=repo_ids,
                project_id=project_id,
                branch=branch,
                compare_branch=compare_branch,
                changed_files=changed_files,
                include_uncommitted=include_uncommitted,
                max_depth=max_depth,
                db=db,
                progress_cb=sync_progress_callback
            ),
            timeout=600.0
        )
        return [types.TextContent(type="text", text=json.dumps(res, ensure_ascii=False, indent=2))]
    except asyncio.TimeoutError:
        logger.error(f"Review context analysis timed out after 600s (elapsed={time.time()-t_start:.1f}s)")
        return [types.TextContent(type="text", text=json.dumps({
            "error": "Review context analysis timed out after 600s",
            "phase": "timeout_exceeded",
            "elapsed_seconds": round(time.time() - t_start, 2),
            "message": "Анализ превысил лимит в 600 секунд. Проверьте воркспейс или передайте список файлов в changed_files."
        }, ensure_ascii=False, indent=2))]
    except Exception as e:
        logger.error(f"Review context analysis failed: {e}")
        return _err(f"Review context error: {e}")


async def _handle_find_tests(arguments: dict) -> list[types.TextContent]:
    """knowledge_find_tests: поиск тестов для метода, класса или файла."""
    target = arguments.get("target")
    if not target:
        return _err("Error: 'target' is required.")

    repo_id = arguments.get("repo_id")
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")

    try:
        res = await asyncio.to_thread(knowledge_find_tests, target=target, repo_id=repo_id, project_id=project_id, branch=branch, db=db)
        return [types.TextContent(type="text", text=json.dumps(res, ensure_ascii=False, indent=2))]
    except Exception as e:
        logger.error(f"Find tests failed: {e}")
        return _err(f"Find tests error: {e}")


async def _handle_get_project_rules(arguments: dict) -> list[types.TextContent]:
    """knowledge_get_project_rules: получение концентрированных правил разработки."""
    topic = arguments.get("topic")
    project_id = arguments.get("project_id", "default")

    try:
        res = await asyncio.to_thread(knowledge_get_project_rules, topic=topic, project_id=project_id, db=db)
        return [types.TextContent(type="text", text=json.dumps(res, ensure_ascii=False, indent=2))]
    except Exception as e:
        logger.error(f"Get project rules failed: {e}")
        return _err(f"Get project rules error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch-таблица: имя инструмента → обработчик
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_HANDLERS: dict[str, Callable[[dict], Any]] = {
    "knowledge_search":           _handle_search,
    "knowledge_get_chunk":        _handle_get_chunk,
    "knowledge_read_file":        _handle_read_file,
    "knowledge_delete_repo":      _handle_delete_repo,
    "knowledge_sync_repo":        _handle_sync_repo,
    "knowledge_find_symbol":      _handle_find_symbol,
    "knowledge_get_callers":      _handle_get_callers,
    "knowledge_get_callees":      _handle_get_callees,
    "knowledge_get_hierarchy":    _handle_get_hierarchy,
    "knowledge_impact_analysis":  _handle_impact_analysis,
    "knowledge_review_context":   _handle_review_context,
    "knowledge_find_tests":       _handle_find_tests,
    "knowledge_get_project_rules": _handle_get_project_rules,
}


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Диспатчер MCP-инструментов. Делегирует вызов конкретному обработчику."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    return await handler(arguments)


# ──────────────────────────────────────────────────────────────────────────────
# FastMCP интеграция
# ──────────────────────────────────────────────────────────────────────────────

try:
    from mcp.server.fastmcp import FastMCP
    fastmcp_server = FastMCP("knowledge-mcp")

    @fastmcp_server.tool(name="knowledge_search")
    async def _fastmcp_search(
        query: str,
        repo_ids: Optional[list[str]] = None,
        project_id: str = "default",
        branch: str = "master",
        limit: int = 10
    ) -> str:
        res = await _handle_search({
            "query": query,
            "repo_ids": repo_ids,
            "project_id": project_id,
            "branch": branch,
            "limit": limit
        })
        return res[0].text if res else ""

    @fastmcp_server.tool(name="knowledge_review_context")
    def _fastmcp_review_context(
        repo_id: Optional[str] = None,
        repo_ids: Optional[list[str]] = None,
        project_id: str = "default",
        branch: str = "master",
        compare_branch: str = "master",
        changed_files: Optional[list[str]] = None,
        file_path: Optional[str] = None,
        include_uncommitted: bool = True,
        max_depth: int = 2
    ) -> dict:
        if file_path and not changed_files:
            changed_files = [file_path]
        return knowledge_review_context(
            repo_id=repo_id,
            repo_ids=repo_ids,
            project_id=project_id,
            branch=branch,
            compare_branch=compare_branch,
            changed_files=changed_files,
            include_uncommitted=include_uncommitted,
            max_depth=max_depth,
            db=db
        )

    @fastmcp_server.tool(name="knowledge_find_tests")
    def _fastmcp_find_tests(
        target: str,
        repo_id: Optional[str] = None,
        project_id: str = "default",
        branch: str = "master"
    ) -> dict:
        return knowledge_find_tests(
            target=target,
            repo_id=repo_id,
            project_id=project_id,
            branch=branch,
            db=db
        )

    @fastmcp_server.tool(name="knowledge_get_project_rules")
    def _fastmcp_get_project_rules(
        topic: Optional[str] = None,
        project_id: str = "default"
    ) -> dict:
        return knowledge_get_project_rules(
            topic=topic,
            project_id=project_id,
            db=db
        )
except Exception as e:
    logger.debug(f"FastMCP setup skipped or unavailable: {e}")
    fastmcp_server = None


def create_sse_app():
    """Создает Starlette приложение для SSE транспорта MCP."""
    try:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        from starlette.responses import Response

        sse_transport = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse_transport.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.run(
                    streams[0], streams[1], server.create_initialization_options()
                )
            return Response()

        return Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse, methods=["GET"]),
                Mount("/messages/", app=sse_transport.handle_post_message),
            ]
        )
    except Exception as e:
        logger.warning(f"Failed to create SSE app: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────

async def start_mcp_server(db_path: str, enable_embeddings: bool = True, read_only: bool = True):
    """Инициализация БД и запуск MCP-сервера по STDIO (стандарт для локальных агентов)."""
    global db, use_embeddings

    # Любой лишний stdout до MCP-кадров ломает handshake.
    # На время инициализации принудительно уводим stdout в stderr.
    with contextlib.redirect_stdout(sys.stderr):
        db = KnowledgeDB(db_path, read_only=read_only)
    use_embeddings = enable_embeddings

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )
