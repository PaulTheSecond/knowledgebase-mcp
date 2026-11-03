"""MCP-инструменты для проведения ревью, поиска тестов и правил проекта (Review Tools)."""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import mcp.types as types

from knowledge_mcp.application.review.review_context_use_case import ReviewContextUseCase
from knowledge_mcp.core.code_parser import CodeParser
from knowledge_mcp.core.config import resolve_workspace_targets
from knowledge_mcp.db.models import KnowledgeDB

logger = logging.getLogger("knowledge_mcp.interfaces.mcp.review")


def _err(msg: str) -> List[types.TextContent]:
    """Формирует сообщение об ошибке в формате TextContent."""
    return [types.TextContent(type="text", text=msg)]


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
    progress_cb: Optional[Callable[[float, float, str], None]] = None,
    code_parser: Optional[CodeParser] = None,
    find_repo_path_fn: Optional[Callable[..., Optional[Path]]] = None,
    resolve_project_id_fn: Optional[Callable[..., str]] = None,
    is_branch_match_fn: Optional[Callable[..., bool]] = None,
    calculate_savings_fn: Optional[Callable[..., Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Делегирует построение контекста ревью в прикладной сценарий ReviewContextUseCase."""
    if not db:
        raise RuntimeError("KnowledgeDB не инициализирована.")

    use_case = ReviewContextUseCase(
        db=db,
        code_parser=code_parser,
        find_repo_path_fn=find_repo_path_fn,
        resolve_project_id_fn=resolve_project_id_fn,
        is_branch_match_fn=is_branch_match_fn,
        calculate_savings_fn=calculate_savings_fn
    )
    return use_case.execute(
        repo_id=repo_id,
        project_id=project_id,
        branch=branch,
        compare_branch=compare_branch,
        changed_files=changed_files,
        max_depth=max_depth,
        repo_ids=repo_ids,
        include_uncommitted=include_uncommitted,
        progress_cb=progress_cb
    )


def knowledge_find_tests(
    target: str,
    repo_id: Optional[str] = None,
    project_id: str = "default",
    branch: str = "master",
    db: Optional[KnowledgeDB] = None,
    calculate_savings_fn: Optional[Callable[..., Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Находит все тесты, связанные с указанным методом, классом или файлом."""
    if not db:
        raise RuntimeError("KnowledgeDB не инициализирована.")

    cursor = db.conn.cursor()
    tests: List[Dict[str, Any]] = []
    seen_keys: set = set()

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
    if calculate_savings_fn:
        savings = calculate_savings_fn(tests, files_involved, db_instance=db)
    else:
        savings = {
            "baseline_tokens": len(files_involved) * 1500,
            "returned_tokens": 300,
            "saved_tokens": max(0, len(files_involved) * 1500 - 300),
            "multiplier": 3.0,
            "saved_percent": 80
        }

    db.log_token_savings(
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
    if not db:
        raise RuntimeError("KnowledgeDB не инициализирована.")

    cursor = db.conn.cursor()

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

    total_rules_tokens = sum(max(1, len(r['content']) // 4) for r in all_rules)
    baseline_tokens = max(19456, total_rules_tokens) if all_rules else 19456

    if topic:
        topic_clean = topic.strip().lower()
        filtered_rules = []
        for r in all_rules:
            c_lower = r['content'].lower()
            p_lower = r['path'].lower()
            if topic_clean in c_lower or topic_clean in p_lower:
                filtered_rules.append(r)

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

    db.log_token_savings(
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


async def handle_review_context(
    arguments: dict,
    db: KnowledgeDB,
    review_context_fn: Callable,
    server_request_context: Any = None,
    resolve_project_id_fn: Optional[Callable[..., str]] = None,
    is_branch_match_fn: Optional[Callable[..., bool]] = None
) -> List[types.TextContent]:
    """Обработчик MCP-запроса knowledge_review_context."""
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

    project_id = resolve_project_id_fn(raw_project_id, branch, repo_ids or ([repo_id] if repo_id else None)) if resolve_project_id_fn else (raw_project_id or "default")

    if not repo_id and not repo_ids and changed_files:
        targets = resolve_workspace_targets(project_id)
        for cf in changed_files:
            cf_norm = cf.replace('\\', '/').strip('/')
            for t in targets:
                match = is_branch_match_fn(branch, t.get("branch"), t.get("task_name")) if is_branch_match_fn else True
                if not t.get("is_doc") and match:
                    t_root = Path(t["absolute_path"]) if t.get("absolute_path") else None
                    if t_root and (t_root / cf_norm).exists():
                        repo_id = t.get("repo_id")
                        break
            if repo_id:
                break

    session = None
    progress_token = None
    try:
        req_ctx = server_request_context
        if req_ctx:
            session = req_ctx.session
            meta = req_ctx.meta
            if meta:
                progress_token = getattr(meta, "progressToken", None) or getattr(meta, "progress_token", None)
    except Exception:
        pass

    loop = asyncio.get_running_loop()

    def sync_progress_callback(progress: float, total: float, message: str) -> None:
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
                review_context_fn,
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


async def handle_find_tests(arguments: dict, db: KnowledgeDB, find_tests_fn: Callable) -> List[types.TextContent]:
    """Обработчик MCP-запроса knowledge_find_tests."""
    target = arguments.get("target")
    if not target:
        return _err("Error: 'target' is required.")

    repo_id = arguments.get("repo_id")
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")

    try:
        res = await asyncio.to_thread(find_tests_fn, target=target, repo_id=repo_id, project_id=project_id, branch=branch, db=db)
        return [types.TextContent(type="text", text=json.dumps(res, ensure_ascii=False, indent=2))]
    except Exception as e:
        logger.error(f"Find tests failed: {e}")
        return _err(f"Find tests error: {e}")


async def handle_get_project_rules(arguments: dict, db: KnowledgeDB, get_rules_fn: Callable) -> List[types.TextContent]:
    """Обработчик MCP-запроса knowledge_get_project_rules."""
    topic = arguments.get("topic")
    project_id = arguments.get("project_id", "default")

    try:
        res = await asyncio.to_thread(get_rules_fn, topic=topic, project_id=project_id, db=db)
        return [types.TextContent(type="text", text=json.dumps(res, ensure_ascii=False, indent=2))]
    except Exception as e:
        logger.error(f"Get project rules failed: {e}")
        return _err(f"Get project rules error: {e}")
