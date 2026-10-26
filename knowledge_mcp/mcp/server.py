import asyncio
import logging
import contextlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, List, Dict

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.core.jit_sync import JITSyncEngine
from knowledge_mcp.core.config import get_repository_config, resolve_workspace_targets, get_project_config

logger = logging.getLogger(__name__)

server = Server("knowledge-mcp")
db: Optional[KnowledgeDB] = None
embedder = None  # Будет LocalEmbedder, загруженный лениво
use_embeddings = True
jit_engine = JITSyncEngine()


def set_db(database: KnowledgeDB) -> None:
    """Устанавливает экземпляр базы данных для MCP-сервера."""
    global db
    db = database


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции для токенов, JIT и поиска путей
# ──────────────────────────────────────────────────────────────────────────────

def _find_repo_path(project_id: str, repo_id: Optional[str], branch: Optional[str] = None) -> Optional[Path]:
    """Находит локальный путь к репозиторию/ворктри на диске."""
    # 1. Поиск в целевых воркспейсах
    targets = resolve_workspace_targets(project_id)
    if targets and repo_id:
        for t in targets:
            if t.get("repo_id") == repo_id and (branch is None or t.get("branch") == branch):
                abs_p = t.get("absolute_path")
                if abs_p and Path(abs_p).exists():
                    return Path(abs_p)
        for t in targets:
            if t.get("repo_id") == repo_id:
                abs_p = t.get("absolute_path")
                if abs_p and Path(abs_p).exists():
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
            return p

    return None


def _run_jit_check(
    repo_ids: Optional[List[str]],
    project_id: str,
    branch: str,
    db_instance: Optional[KnowledgeDB]
) -> int:
    """Выполняет JIT-проверку для указанных репозиториев перед выполнением поисковых запросов."""
    if not db_instance or getattr(db_instance, "read_only", False):
        return 0
    updated_total = 0
    try:
        if repo_ids:
            for r_id in repo_ids:
                p = _find_repo_path(project_id, r_id, branch)
                if p and p.exists():
                    updated_total += jit_engine.ensure_fresh(
                        p, project_id=project_id, repo_id=r_id, branch=branch, db=db_instance
                    )
        else:
            targets = resolve_workspace_targets(project_id)
            for t in targets:
                p = t.get("absolute_path")
                r_id = t.get("repo_id")
                b_name = t.get("branch") or branch
                if p and Path(p).exists() and r_id:
                    updated_total += jit_engine.ensure_fresh(
                        p, project_id=project_id, repo_id=r_id, branch=b_name, db=db_instance
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
            "calculate risk score (0-100), and provide a compact review context for code reviewer agents."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "repo_id": {"type": "string", "description": "The repository identifier."},
                "project_id": {"type": "string", "description": "The project/workspace identifier (default 'default')."},
                "branch": {"type": "string", "description": "The current working branch (default 'master')."},
                "compare_branch": {"type": "string", "description": "The target base branch to compare against (default 'master')."},
                "changed_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit list of modified files. If omitted, discovered via git diff or CAS comparison."
                },
                "max_depth": {"type": "integer", "description": "Max recursion depth for caller blast radius (default 2)."}
            },
            "required": ["repo_id"]
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

    p_id = project_id or "default"
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

    # JIT-проверка перед поиском символов
    _run_jit_check(repo_ids=repo_ids, project_id=project_id, branch=branch, db_instance=db_instance)

    results = db_instance.find_symbols(
        name_pattern=name_pattern,
        kind=kind,
        repo_ids=repo_ids,
        limit=limit,
        project_id=project_id,
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

    content = db_instance.get_file_content(path, repo_id=repo_id, project_id=project_id, branch=branch)
    if content is not None:
        if isinstance(content, dict):
            return content.get("content", "")
        return str(content)

    # Fallback на рабочую копию в файловой системе
    if repo_id:
        repo_path = _find_repo_path(project_id, repo_id, branch)
        if repo_path:
            file_on_disk = (repo_path / path).resolve()
            if file_on_disk.is_file():
                try:
                    return file_on_disk.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
    return None


def knowledge_review_context(
    repo_id: str,
    project_id: str = "default",
    branch: str = "master",
    compare_branch: str = "master",
    changed_files: Optional[List[str]] = None,
    max_depth: int = 2,
    db: Optional[KnowledgeDB] = None
) -> Dict[str, Any]:
    """
    Строит радиус влияния (blast radius), находит связанные тесты,
    выявляет Test Gaps и рассчитывает риск-скоринг для ревьюера.
    """
    db_instance = db or globals()["db"]
    if not db_instance:
        raise RuntimeError("KnowledgeDB is not initialized.")

    cursor = db_instance.conn.cursor()

    # 1. Определение измененных файлов
    if changed_files is None:
        # Попытка через Git CLI
        repo_path = _find_repo_path(project_id, repo_id, branch)
        if repo_path and repo_path.exists():
            try:
                cmd = ["git", "diff", "--name-only", f"{compare_branch}...HEAD"]
                res = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=5.0)
                if res.returncode == 0 and res.stdout.strip():
                    changed_files = [f.strip().replace('\\', '/') for f in res.stdout.splitlines() if f.strip()]
                else:
                    cmd2 = ["git", "diff", "--name-only", compare_branch]
                    res2 = subprocess.run(cmd2, cwd=str(repo_path), capture_output=True, text=True, timeout=5.0)
                    if res2.returncode == 0 and res2.stdout.strip():
                        changed_files = [f.strip().replace('\\', '/') for f in res2.stdout.splitlines() if f.strip()]
            except Exception as e:
                logger.debug(f"Git diff determination skipped: {e}")

        # Попытка через сравнение content_hash в CAS БД
        if changed_files is None:
            cursor.execute("""
                SELECT f1.path
                FROM files f1
                LEFT JOIN files f2 ON f2.project_id = f1.project_id AND f2.repo_id = f1.repo_id
                                  AND f2.branch = ? AND f2.path = f1.path
                WHERE f1.project_id = ? AND f1.repo_id = ? AND f1.branch = ?
                  AND (f2.id IS NULL OR f1.content_hash != f2.content_hash)
            """, (compare_branch, project_id, repo_id, branch))
            changed_files = [r['path'].replace('\\', '/') for r in cursor.fetchall()]

    if changed_files is None:
        changed_files = []

    norm_changed_files = [p.replace('\\', '/') for p in changed_files]

    # 2. Поиск измененных символов во всех измененных файлах
    changed_symbols = []
    changed_sym_ids = set()
    for cf in norm_changed_files:
        cursor.execute("""
            SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.line_start, s.line_end,
                   s.signature, s.language, f.path, f.repo_id
            FROM symbols s
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE f.project_id = ? AND f.repo_id = ? AND f.branch = ? AND f.path = ?
        """, (project_id, repo_id, branch, cf))
        for r in cursor.fetchall():
            sym_dict = dict(r)
            changed_symbols.append(sym_dict)
            changed_sym_ids.add(sym_dict['id'])

    # 3. Построение графа радиуса влияния (blast radius)
    impacted_symbols = []
    impacted_files = set()
    seen_impacted_ids = set(changed_sym_ids)

    for sym in changed_symbols:
        sym_impact = db_instance.get_impact_analysis(
            sym['id'], max_depth=max_depth, project_id=project_id, branch=branch
        )
        for item in sym_impact:
            caller_sym = item['symbol']
            caller_id = caller_sym['id']
            if caller_id not in seen_impacted_ids:
                seen_impacted_ids.add(caller_id)
                impacted_files.add(caller_sym['path'])
                impacted_symbols.append({
                    "id": caller_id,
                    "name": caller_sym['name'],
                    "qualified_name": caller_sym['qualified_name'],
                    "kind": caller_sym['kind'],
                    "file": caller_sym['path'],
                    "repo_id": caller_sym['repo_id'],
                    "edge_kind": item['edge_kind'],
                    "depth": item['depth']
                })

    # 4. Поиск связанных тестов и выявление Test Gaps
    related_tests = []
    seen_test_keys = set()
    tested_symbol_ids = set()

    for sym in changed_symbols:
        sym_id = sym['id']
        sym_name = sym['name']
        sym_qname = sym.get('qualified_name') or sym_name

        # а) Ребра TESTED_BY (source = тестируемый метод, target = тест)
        cursor.execute("""
            SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.signature, f.path, f.repo_id
            FROM symbol_edges e
            JOIN symbols s ON e.target_id = s.id
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE e.source_id = ? AND e.kind = 'TESTED_BY'
        """, (sym_id,))
        for tr in cursor.fetchall():
            tested_symbol_ids.add(sym_id)
            k = (tr['id'], sym_id)
            if k not in seen_test_keys:
                seen_test_keys.add(k)
                related_tests.append({
                    "test_id": tr['id'],
                    "test_name": tr['name'],
                    "test_file": tr['path'],
                    "tested_symbol_id": sym_id,
                    "tested_symbol_name": sym_name,
                    "relation": "TESTED_BY"
                })

        # б) Ребра TESTED_BY обратные (на случай если target = тестируемый символ)
        cursor.execute("""
            SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.signature, f.path, f.repo_id
            FROM symbol_edges e
            JOIN symbols s ON e.source_id = s.id
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE e.target_id = ? AND e.kind = 'TESTED_BY'
        """, (sym_id,))
        for tr in cursor.fetchall():
            tested_symbol_ids.add(sym_id)
            k = (tr['id'], sym_id)
            if k not in seen_test_keys:
                seen_test_keys.add(k)
                related_tests.append({
                    "test_id": tr['id'],
                    "test_name": tr['name'],
                    "test_file": tr['path'],
                    "tested_symbol_id": sym_id,
                    "tested_symbol_name": sym_name,
                    "relation": "TESTED_BY"
                })

        # в) Неразрешенные ссылки unresolved_refs
        cursor.execute("""
            SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.signature, f.path, f.repo_id
            FROM unresolved_refs u
            JOIN symbols s ON u.source_id = s.id
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE u.kind = 'TESTED_BY' AND (u.target_qualified_name = ? OR u.target_qualified_name = ?)
        """, (sym_name, sym_qname))
        for tr in cursor.fetchall():
            tested_symbol_ids.add(sym_id)
            k = (tr['id'], sym_id)
            if k not in seen_test_keys:
                seen_test_keys.add(k)
                related_tests.append({
                    "test_id": tr['id'],
                    "test_name": tr['name'],
                    "test_file": tr['path'],
                    "tested_symbol_id": sym_id,
                    "tested_symbol_name": sym_name,
                    "relation": "TESTED_BY_UNRESOLVED"
                })

        # г) Эвристики именования тестов
        cursor.execute("""
            SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.signature, f.path, f.repo_id
            FROM symbols s
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE (s.name = ? OR s.name = ? OR s.name = ? OR s.name = ?)
              AND f.project_id = ? AND f.repo_id = ?
        """, (f"Test{sym_name}", f"{sym_name}Test", f"{sym_name}Tests", f"test_{sym_name.lower()}", project_id, repo_id))
        for tr in cursor.fetchall():
            tested_symbol_ids.add(sym_id)
            k = (tr['id'], sym_id)
            if k not in seen_test_keys:
                seen_test_keys.add(k)
                related_tests.append({
                    "test_id": tr['id'],
                    "test_name": tr['name'],
                    "test_file": tr['path'],
                    "tested_symbol_id": sym_id,
                    "tested_symbol_name": sym_name,
                    "relation": "naming_pattern"
                })

    # Выявляем Test Gaps: измененные функции и методы БЕЗ тестов
    test_gaps = []
    for sym in changed_symbols:
        k_lower = sym['kind'].lower()
        if k_lower in ('method', 'function') and sym['id'] not in tested_symbol_ids:
            test_gaps.append({
                "id": sym['id'],
                "name": sym['name'],
                "qualified_name": sym.get('qualified_name') or sym['name'],
                "kind": sym['kind'],
                "file": sym['path']
            })

    # 5. Расчет скоринга риска
    base_score = len(norm_changed_files) * 5 + len(changed_symbols) * 3
    blast_score = len(impacted_symbols) * 4
    test_gap_score = len(test_gaps) * 15

    if not norm_changed_files and not changed_symbols:
        risk_score = 0
        risk_level = "low"
    else:
        risk_score = max(10, min(100, base_score + blast_score + test_gap_score))
        if risk_score < 30:
            risk_level = "low"
        elif risk_score < 70:
            risk_level = "medium"
        else:
            risk_level = "high"

    # 6. Расчет экономии токенов
    all_affected_files = set(norm_changed_files) | set(impacted_files) | {t['test_file'] for t in related_tests}
    savings = _calculate_savings(
        {
            "changed_symbols": changed_symbols,
            "blast_radius": impacted_symbols,
            "related_tests": related_tests,
            "test_gaps": test_gaps
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
        repo_id=repo_id,
        branch=branch
    )

    return {
        "repo_id": repo_id,
        "project_id": project_id,
        "branch": branch,
        "compare_branch": compare_branch,
        "changed_files": norm_changed_files,
        "changed_symbols": changed_symbols,
        "blast_radius": {
            "impacted_symbols_count": len(impacted_symbols),
            "impacted_files_count": len(impacted_files),
            "callers": impacted_symbols,
            "impacted_files": list(impacted_files),
            "max_depth": max_depth
        },
        "related_tests": related_tests,
        "test_gaps": test_gaps,
        "risk": risk_level,
        "risk_score": risk_score,
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

    cursor.execute("""
        SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind, s.signature, f.path, f.repo_id
        FROM symbols s
        JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
        WHERE (s.name LIKE ? OR s.name LIKE ? OR s.name LIKE ? OR s.name LIKE ? OR s.name LIKE ?)
    """, (f"Test{base_target}%", f"{base_target}Test%", f"{base_target}Tests%", f"test_{base_target.lower()}%", f"%_{base_target}_%"))
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
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")
    limit = arguments.get("limit", 10)

    try:
        vector = None
        if use_embeddings:
            global embedder
            if embedder is None:
                logger.info("Lazy loading LocalEmbedder in background thread (~30s first time)...")
                embedder = await asyncio.to_thread(_init_embedder_sync)
                logger.info("Embedder ready!")
            vector = await asyncio.to_thread(_embed_text_sync, embedder, query)

        res = knowledge_search(
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

        formatted = [
            f"--- Chunk ID: {r['id']} ---\n"
            f"**Repository**: `{r['repo_id']}`\n"
            f"**File**: `{r['path']}` (Lines: {r['line_start']}-{r['line_end']})\n"
            f"**Context Trust Level**: {r['trust']} (Source: {r['source_kind']})\n\n"
            f"```\n{r['content']}\n```\n"
            for r in results
        ]

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
    path = arguments.get("path")
    if not path:
        return _err("Error: 'path' argument is required.")

    repo_id = arguments.get("repo_id")
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")

    try:
        content = knowledge_read_file(path, repo_id=repo_id, project_id=project_id, branch=branch, db=db)
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
        res = knowledge_find_symbol(
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
        res = knowledge_get_callers(symbol_id, project_id=project_id, branch=branch, db=db)
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
        res = knowledge_get_callees(symbol_id, project_id=project_id, branch=branch, db=db)
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
        res = knowledge_get_hierarchy(symbol_id, project_id=project_id, branch=branch, db=db)
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
        res = knowledge_impact_analysis(symbol_id, max_depth=max_depth, project_id=project_id, branch=branch, db=db)
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
    if not repo_id:
        return _err("Error: 'repo_id' is required.")

    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")
    compare_branch = arguments.get("compare_branch", "master")
    changed_files = arguments.get("changed_files")
    max_depth = arguments.get("max_depth", 2)

    try:
        res = knowledge_review_context(
            repo_id=repo_id,
            project_id=project_id,
            branch=branch,
            compare_branch=compare_branch,
            changed_files=changed_files,
            max_depth=max_depth,
            db=db
        )
        return [types.TextContent(type="text", text=json.dumps(res, ensure_ascii=False, indent=2))]
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
        res = knowledge_find_tests(
            target=target,
            repo_id=repo_id,
            project_id=project_id,
            branch=branch,
            db=db
        )
        return [types.TextContent(type="text", text=json.dumps(res, ensure_ascii=False, indent=2))]
    except Exception as e:
        logger.error(f"Find tests failed: {e}")
        return _err(f"Find tests error: {e}")


async def _handle_get_project_rules(arguments: dict) -> list[types.TextContent]:
    """knowledge_get_project_rules: получение концентрированных правил разработки."""
    topic = arguments.get("topic")
    project_id = arguments.get("project_id", "default")

    try:
        res = knowledge_get_project_rules(
            topic=topic,
            project_id=project_id,
            db=db
        )
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
        repo_id: str,
        project_id: str = "default",
        branch: str = "master",
        compare_branch: str = "master",
        changed_files: Optional[list[str]] = None,
        max_depth: int = 2
    ) -> dict:
        return knowledge_review_context(
            repo_id=repo_id,
            project_id=project_id,
            branch=branch,
            compare_branch=compare_branch,
            changed_files=changed_files,
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
        from starlette.routing import Route
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

        async def handle_messages(request):
            await sse_transport.handle_post_message(
                request.scope, request.receive, request._send
            )
            return Response()

        return Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Route("/messages/", endpoint=handle_messages, methods=["POST"]),
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
