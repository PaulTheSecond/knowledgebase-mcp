"""Инфраструктурный адаптер для работы с Git через консольный интерфейс (Git CLI).

Предоставляет методы для:
1. Вычисления диапазонов измененных строк (git diff -U0).
2. Сбора информации о ветке, коммитах и слиянии (merge-base, HEAD, git log).
3. Построения диффа файлов с определением переименований (git diff -M, git log -M).
4. Определения авторства и происхождения строк кода через git blame (--porcelain).
"""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from knowledge_mcp.core.git_manager import get_git_cmd_prefix

logger = logging.getLogger("knowledge_mcp.infrastructure.git")

# Кэш результатов git blame: (repo_path_str, file_path, line_no) -> finding_origin_dict
_BLAME_CACHE: Dict[Tuple[str, str, int], Dict[str, Any]] = {}


def clear_blame_cache() -> None:
    """Очищает внутренний кэш результатов git blame."""
    _BLAME_CACHE.clear()


def is_ignored_diff_path(norm_path: str) -> bool:
    """Проверяет, является ли путь служебным или локальным артефактом, исключаемым из код-ревью."""
    p_lower = norm_path.replace('\\', '/').lower().strip('/')
    parts = [pt for pt in p_lower.split('/') if pt]
    if any(part in ('.husky', '.vscode', '.idea', '.git') or part.startswith('.husky') for part in parts):
        return True
    if p_lower.endswith(('.bak', '.orig', '.stackdump', 'thumbs.db', '.ds_store')):
        return True
    return False


def get_repo_changed_line_ranges(
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

    def _parse_diff(diff_stdout: str) -> None:
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
        logger.debug(f"Пакетный diff завершился с ошибкой для {repo_path}: {e}")

    if include_uncommitted:
        try:
            unc_res = subprocess.run(
                prefix + ["diff", "-U0", "HEAD"],
                cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10.0
            )
            if unc_res.returncode == 0 and unc_res.stdout:
                _parse_diff(unc_res.stdout)
        except Exception as e:
            logger.debug(f"Пакетный diff незакоммиченных изменений завершился с ошибкой для {repo_path}: {e}")

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


def get_changed_line_ranges(
    repo_path: Path,
    file_path: str,
    compare_ref: str,
    include_uncommitted: bool = True
) -> Optional[List[Tuple[int, int]]]:
    """Возвращает список диапазонов измененных строк (start_line, end_line) из git diff для конкретного файла."""
    if not repo_path or not repo_path.exists():
        return None
    try:
        norm_fp = file_path.replace('\\', '/')
        prefix = get_git_cmd_prefix(repo_path)
        ranges: List[Tuple[int, int]] = []

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


def get_repo_git_info(
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

    commits: List[Dict[str, Any]] = []
    branch_commit_shas: Set[str] = set()
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

    uncommitted_files: List[str] = []
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


def get_git_diff_with_renames(
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
                    parts = line.split("\t")
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
    deleted_files: Set[str] = set()
    if res.returncode == 0 and res.stdout.strip():
        for line in res.stdout.splitlines():
            line = line.strip()
            if not line or "\t" not in line:
                continue
            parts = line.split("\t")
            status_code = parts[0]

            if status_code.startswith("R"):
                old_p = parts[1].replace('\\', '/')
                new_p = parts[2].replace('\\', '/')
                if is_ignored_diff_path(new_p):
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
                if is_ignored_diff_path(p):
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
                if not is_ignored_diff_path(p):
                    deleted_files.add(p)
            elif status_code.startswith("M"):
                p = parts[1].replace('\\', '/')
                if is_ignored_diff_path(p):
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
                if not line or "\t" not in line:
                    continue
                parts = line.split("\t")
                st = parts[0]
                if st.startswith("R"):
                    old_p = parts[1].replace('\\', '/')
                    new_p = parts[2].replace('\\', '/')
                    if is_ignored_diff_path(new_p):
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
                    if is_ignored_diff_path(p):
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
                    if is_ignored_diff_path(p):
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
                    if is_ignored_diff_path(p):
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
                if not p or is_ignored_diff_path(p):
                    continue
                if p not in file_map:
                    file_map[p] = {
                        "path": p,
                        "old_path": None,
                        "change_type": "added",
                        "similarity": 100,
                        "is_uncommitted": True
                    }

    return [item for item in file_map.values() if not is_ignored_diff_path(item["path"])]


def get_finding_origin(
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


class GitCliAdapter:
    """Адаптер репозиториев для выполнения операций Git CLI."""

    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path

    def get_repo_git_info(self, branch: str, compare_branch: str = "master", include_uncommitted: bool = True) -> Dict[str, Any]:
        if not self.repo_path:
            raise ValueError("repo_path не задан для GitCliAdapter")
        return get_repo_git_info(self.repo_path, branch, compare_branch, include_uncommitted)

    def get_git_diff_with_renames(self, base_sha: Optional[str], head_sha: Optional[str], include_uncommitted: bool = True) -> List[Dict[str, Any]]:
        if not self.repo_path:
            raise ValueError("repo_path не задан для GitCliAdapter")
        return get_git_diff_with_renames(self.repo_path, base_sha, head_sha, include_uncommitted)

    def get_repo_changed_line_ranges(self, base_ref: Optional[str], head_ref: Optional[str], include_uncommitted: bool = True) -> Dict[str, List[Tuple[int, int]]]:
        if not self.repo_path:
            raise ValueError("repo_path не задан для GitCliAdapter")
        return get_repo_changed_line_ranges(self.repo_path, base_ref, head_ref, include_uncommitted)

    def get_changed_line_ranges(self, file_path: str, compare_ref: str, include_uncommitted: bool = True) -> Optional[List[Tuple[int, int]]]:
        if not self.repo_path:
            raise ValueError("repo_path не задан для GitCliAdapter")
        return get_changed_line_ranges(self.repo_path, file_path, compare_ref, include_uncommitted)

    def get_finding_origin(self, file_path: str, line_no: Optional[int], branch_commit_shas: Set[str]) -> Dict[str, Any]:
        if not self.repo_path:
            raise ValueError("repo_path не задан для GitCliAdapter")
        return get_finding_origin(self.repo_path, file_path, line_no, branch_commit_shas)
