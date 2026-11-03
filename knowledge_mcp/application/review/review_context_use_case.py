"""Сценарий построения контекста код-ревью (ReviewContextUseCase).

Реализует сквозной прикладной процесс анализа изменений:
1. Определение мультирепозиторного скоупа задачи/ветки.
2. Сбор истории Git и определение точек слияния (merge-base..HEAD).
3. Извлечение измененных символов (AST) с учетом диапазонов git diff.
4. Расчет радиуса влияния (Blast Radius) по графу вызовов.
5. Обнаружение связанных тестов и анализ тестовых пробелов (Test Gaps / Mode Gaps).
6. Аудит по чек-листам качества (.kb) и правилам безопасности.
7. Расчет итогового прозрачного риск-скоринга и метрик экономии контекста токенов.
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from knowledge_mcp.core.code_parser import CodeParser
from knowledge_mcp.core.config import resolve_workspace_targets
from knowledge_mcp.core.git_manager import get_git_cmd_prefix
from knowledge_mcp.domain.review.gap_analyzer import (
    build_reviewed_files_audit,
    extract_call_modes_and_gaps,
    filter_valid_findings,
)
from knowledge_mcp.domain.review.risk_model import calculate_transparent_risk_score
from knowledge_mcp.domain.review.rules import audit_checklist_rules
from knowledge_mcp.infrastructure.git.git_cli_adapter import (
    get_finding_origin,
    get_git_diff_with_renames,
    get_repo_changed_line_ranges,
    get_repo_git_info,
)

logger = logging.getLogger("knowledge_mcp.application.review")

# Частые базовые имена переменных и идентификаторов, отсекаемые при поиске связанных тестов
COMMON_CODE_STOPWORDS: Set[str] = {
    't', '_', 'e', 'i', 'j', 'k', 's', 'x', 'y', 'z',
    'id', 'fn', 'cb', 'res', 'req', 'ctx', 'props', 'state',
    'item', 'row', 'data', 'val', 'key', 'err', 'opt', 'opts'
}


class ReviewContextUseCase:
    """Прикладной сценарий построения контекста код-ревью."""

    def __init__(
        self,
        db: Any,
        code_parser: Optional[CodeParser] = None,
        find_repo_path_fn: Optional[Callable[[str, Optional[str], Optional[str]], Optional[Path]]] = None,
        resolve_project_id_fn: Optional[Callable[[Optional[str], Optional[str], Optional[List[str]]], str]] = None,
        is_branch_match_fn: Optional[Callable[[Optional[str], Optional[str], Optional[str]], bool]] = None,
        calculate_savings_fn: Optional[Callable[..., Dict[str, int]]] = None
    ):
        self.db = db
        self.code_parser = code_parser or CodeParser()
        self._find_repo_path_fn = find_repo_path_fn
        self._resolve_project_id_fn = resolve_project_id_fn
        self._is_branch_match_fn = is_branch_match_fn
        self._calculate_savings_fn = calculate_savings_fn

    def _resolve_project(self, project_id: Optional[str], branch: Optional[str], repo_ids: Optional[List[str]]) -> str:
        if self._resolve_project_id_fn:
            return self._resolve_project_id_fn(project_id, branch, repo_ids)
        return project_id or "default"

    def _find_repo_path(self, project_id: str, repo_id: Optional[str], branch: Optional[str]) -> Optional[Path]:
        if self._find_repo_path_fn:
            return self._find_repo_path_fn(project_id, repo_id, branch)
        return None

    def _is_branch_match(self, requested_branch: Optional[str], target_branch: Optional[str], task_name: Optional[str]) -> bool:
        if self._is_branch_match_fn:
            return self._is_branch_match_fn(requested_branch, target_branch, task_name)
        if not requested_branch or not target_branch:
            return False
        return requested_branch.lower() == target_branch.lower()

    def execute(
        self,
        repo_id: Optional[str] = None,
        project_id: str = "default",
        branch: str = "master",
        compare_branch: str = "master",
        changed_files: Optional[List[str]] = None,
        max_depth: int = 2,
        repo_ids: Optional[List[str]] = None,
        include_uncommitted: bool = True,
        progress_cb: Optional[Callable[[float, float, str], None]] = None
    ) -> Dict[str, Any]:
        """Выполняет построение полного контекста ревью задачи/ветки."""
        db_instance = self.db
        if not db_instance:
            raise RuntimeError("KnowledgeDB не инициализирована.")

        project_id = self._resolve_project(project_id, branch, repo_ids or ([repo_id] if repo_id else None))
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
                    if not t.get("is_doc") and self._is_branch_match(branch, t.get("branch"), t.get("task_name"))
                ]
                for bt in branch_targets:
                    bt_r = bt.get("repo_id")
                    if bt_r and bt_r not in target_repos:
                        target_repos.append(bt_r)
        else:
            branch_targets = [
                t for t in all_targets
                if not t.get("is_doc") and self._is_branch_match(branch, t.get("branch"), t.get("task_name"))
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

        # 1. Сбор информации по каждому репозиторию задачи
        for cur_repo in target_repos:
            repo_path = self._find_repo_path(project_id, cur_repo, branch)
            git_info = get_repo_git_info(repo_path, branch, compare_branch, include_uncommitted)

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
                    raw_diff = get_git_diff_with_renames(
                        repo_path, git_info["merge_base_commit"], git_info["head_commit"], include_uncommitted
                    )
                    cf_set = {cf.replace('\\', '/') for cf in changed_files}
                    cur_diff_files = [f for f in raw_diff if f["path"] in cf_set or any(c in f["path"] for c in cf_set)]
            else:
                cur_diff_files = get_git_diff_with_renames(
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
                repo_diff_ranges = get_repo_changed_line_ranges(
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
                            logger.debug(f"Поиск потребителей пропущен: {e}")

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
                })

            # 4. Поиск тестов для репозитория (пакетный запрос с фильтрацией в памяти)
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
        for sym in all_changed_symbols:
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
                s_repo_path = self._find_repo_path(project_id, sym_repo, branch)
                mode_gaps = extract_call_modes_and_gaps(
                    sym=sym,
                    component_consumer_files=all_component_consumer_files,
                    repo_path=s_repo_path,
                    tested_symbol_ids=tested_symbol_ids,
                    feature_test_keywords=feature_test_keywords,
                    related_tests=all_related_tests,
                    code_parser=self.code_parser
                )
                for mg in mode_gaps:
                    mg_hdr = next((h for h in repo_git_headers if h["repo_id"] == sym_repo), {})
                    mg_orig = get_finding_origin(s_repo_path, mg["file"], mg.get("line_start"), set(mg_hdr.get("branch_commit_shas") or []))
                    mg["origin"] = mg_orig["origin"]
                    mg["introduced_in_commit"] = mg_orig
                    all_test_gaps.append(mg)

        # Проверка правил чек-листов и выявление дефектов кода (чек-листы .kb)
        if progress_cb:
            progress_cb(4.0, 5.0, "Аудит правил чек-листов и расчет риск-скоринга...")
        all_code_findings: List[Dict[str, Any]] = []
        for cur_repo in target_repos:
            repo_path = self._find_repo_path(project_id, cur_repo, branch)
            repo_git_header = next((h for h in repo_git_headers if h["repo_id"] == cur_repo), {})
            repo_diff = [df for df in all_diff_files if df.get("repo_id") == cur_repo]
            cf_findings, cf_gaps = audit_checklist_rules(
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
        valid_test_gaps = filter_valid_findings(all_test_gaps)

        # Формирование обязательного списка отрецензированных файлов (Rule 2)
        reviewed_files = build_reviewed_files_audit(all_diff_files, all_code_findings, valid_test_gaps, all_changed_symbols)

        # 5. Расчет прозрачного скоринга риска (Rule 6)
        risk_breakdown = calculate_transparent_risk_score(
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
        if self._calculate_savings_fn:
            savings = self._calculate_savings_fn(
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
        else:
            savings = {
                "baseline_tokens": len(all_affected_files) * 2000,
                "returned_tokens": 500,
                "saved_tokens": max(0, len(all_affected_files) * 2000 - 500),
                "multiplier": 4.0
            }

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
