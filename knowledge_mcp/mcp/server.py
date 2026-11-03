"""Сервер Model Context Protocol (MCP) базы знаний Knowledge Base.

Обеспечивает интеграцию со средами разработки и агентами (Claude Code, Cursor, Antigravity)
через протоколы STDIO и SSE. Является тонким роутером-фасадом над слоями Interfaces и Application.
"""

import asyncio
import contextlib
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from knowledge_mcp.core.code_parser import CodeParser
from knowledge_mcp.core.token_accounting import calculate_savings as _calculate_savings
from knowledge_mcp.core.workspace_context import (
    _REPO_PATH_CACHE,
    clear_repo_path_cache,
    find_repo_path as _find_repo_path,
    is_branch_match as _is_branch_match,
    jit_engine,
    resolve_project_id as _resolve_project_id,
    run_jit_check as _run_jit_check,
)
from knowledge_mcp.db.models import KnowledgeDB

# Делегация инфраструктурных Git-операций
from knowledge_mcp.infrastructure.git.git_cli_adapter import (
    _BLAME_CACHE,
    clear_blame_cache,
    get_changed_line_ranges as _get_changed_line_ranges,
    get_finding_origin as _get_finding_origin,
    get_git_diff_with_renames as _get_git_diff_with_renames,
    get_repo_changed_line_ranges as _get_repo_changed_line_ranges,
    get_repo_git_info as _get_repo_git_info,
    is_ignored_diff_path as _is_ignored_diff_path,
)

# Делегация доменной логики ревью
from knowledge_mcp.domain.review.gap_analyzer import (
    build_reviewed_files_audit as _build_reviewed_files_audit,
    extract_call_modes_and_gaps as _extract_call_modes_and_gaps,
    filter_valid_findings as _filter_valid_findings,
)
from knowledge_mcp.domain.review.risk_model import (
    calculate_transparent_risk_score as _calculate_transparent_risk_score,
)
from knowledge_mcp.domain.review.rules import (
    audit_checklist_rules as _audit_checklist_rules,
)

# Модули интерфейса MCP и реестр инструментов
from knowledge_mcp.interfaces.mcp.definitions import TOOL_SCHEMAS as _TOOL_SCHEMAS
from knowledge_mcp.interfaces.mcp.tool_registry import ToolRegistry
from knowledge_mcp.interfaces.mcp.tools.admin_tools import (
    get_auth_headers as _get_auth_headers,
    get_daemon_base_url as _get_daemon_base_url,
    handle_delete_repo as _handle_delete_repo,
    handle_sync_repo as _handle_sync_repo,
)
from knowledge_mcp.interfaces.mcp.tools.search_tools import (
    handle_get_chunk as _handle_get_chunk,
    handle_read_file as _handle_read_file,
    handle_search as _handle_search,
)
from knowledge_mcp.interfaces.mcp.tools.symbol_tools import (
    handle_find_symbol as _handle_find_symbol,
    handle_get_callees as _handle_get_callees,
    handle_get_callers as _handle_get_callers,
    handle_get_hierarchy as _handle_get_hierarchy,
    handle_impact_analysis as _handle_impact_analysis,
)
from knowledge_mcp.interfaces.mcp.tools.review_tools import (
    handle_find_tests as _handle_find_tests,
    handle_get_project_rules as _handle_get_project_rules,
    handle_review_context as _handle_review_context,
)

# Реэкспорт функций бизнес-логики инструментов
import knowledge_mcp.interfaces.mcp.tools as _tools

logger = logging.getLogger(__name__)

server = Server("knowledge-mcp")
db: Optional[KnowledgeDB] = None
embedder = None
use_embeddings = True
code_parser = CodeParser()


def set_db(database: KnowledgeDB) -> None:
    """Устанавливает экземпляр базы данных для MCP-сервера."""
    global db
    db = database


def _init_embedder_sync():
    """Синхронная инициализация модели эмбеддингов."""
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from knowledge_mcp.core.embeddings import LocalEmbedder
        return LocalEmbedder()
    finally:
        sys.stdout = original_stdout


def _embed_text_sync(emb, text: str):
    """Синхронный вызов инференса эмбеддингов."""
    return emb.embed_text(text, is_query=True)


# ──────────────────────────────────────────────────────────────────────────────
# Программные API-функции инструментов (полная обратная совместимость)
# ──────────────────────────────────────────────────────────────────────────────

def knowledge_search(query: str, repo_ids: Optional[List[str]] = None, project_id: Optional[str] = "default", branch: Optional[str] = "master", limit: int = 10, query_vector: Optional[List[float]] = None, db: Optional[KnowledgeDB] = None) -> Dict[str, Any]:
    return _tools.knowledge_search(query=query, repo_ids=repo_ids, project_id=project_id, branch=branch, limit=limit, query_vector=query_vector, db=db or globals()["db"], resolve_project_id_fn=_resolve_project_id, run_jit_check_fn=_run_jit_check, calculate_savings_fn=_calculate_savings)


def knowledge_read_file(path: str, repo_id: Optional[str] = None, project_id: str = "default", branch: str = "master", db: Optional[KnowledgeDB] = None) -> Optional[str]:
    return _tools.knowledge_read_file(path=path, repo_id=repo_id, project_id=project_id, branch=branch, db=db or globals()["db"], find_repo_path_fn=_find_repo_path, is_branch_match_fn=_is_branch_match)


def knowledge_find_symbol(name_pattern: str, kind: Optional[str] = None, repo_ids: Optional[List[str]] = None, project_id: str = "default", branch: str = "master", limit: int = 20, db: Optional[KnowledgeDB] = None) -> Dict[str, Any]:
    return _tools.knowledge_find_symbol(name_pattern=name_pattern, kind=kind, repo_ids=repo_ids, project_id=project_id, branch=branch, limit=limit, db=db or globals()["db"], resolve_project_id_fn=_resolve_project_id, run_jit_check_fn=_run_jit_check, calculate_savings_fn=_calculate_savings)


def knowledge_get_callers(symbol_id: int, project_id: str = "default", branch: str = "master", db: Optional[KnowledgeDB] = None) -> Dict[str, Any]:
    return _tools.knowledge_get_callers(symbol_id=symbol_id, project_id=project_id, branch=branch, db=db or globals()["db"], calculate_savings_fn=_calculate_savings)


def knowledge_get_callees(symbol_id: int, project_id: str = "default", branch: str = "master", db: Optional[KnowledgeDB] = None) -> Dict[str, Any]:
    return _tools.knowledge_get_callees(symbol_id=symbol_id, project_id=project_id, branch=branch, db=db or globals()["db"], calculate_savings_fn=_calculate_savings)


def knowledge_get_hierarchy(symbol_id: int, project_id: str = "default", branch: str = "master", db: Optional[KnowledgeDB] = None) -> Dict[str, Any]:
    return _tools.knowledge_get_hierarchy(symbol_id=symbol_id, project_id=project_id, branch=branch, db=db or globals()["db"], calculate_savings_fn=_calculate_savings)


def knowledge_impact_analysis(symbol_id: int, max_depth: int = 3, project_id: str = "default", branch: str = "master", db: Optional[KnowledgeDB] = None) -> Dict[str, Any]:
    return _tools.knowledge_impact_analysis(symbol_id=symbol_id, max_depth=max_depth, project_id=project_id, branch=branch, db=db or globals()["db"], calculate_savings_fn=_calculate_savings)


def knowledge_review_context(repo_id: Optional[str] = None, project_id: str = "default", branch: str = "master", compare_branch: str = "master", changed_files: Optional[List[str]] = None, max_depth: int = 2, db: Optional[KnowledgeDB] = None, repo_ids: Optional[List[str]] = None, include_uncommitted: bool = True, progress_cb: Optional[Callable[[float, float, str], None]] = None) -> Dict[str, Any]:
    return _tools.knowledge_review_context(repo_id=repo_id, project_id=project_id, branch=branch, compare_branch=compare_branch, changed_files=changed_files, max_depth=max_depth, db=db or globals()["db"], repo_ids=repo_ids, include_uncommitted=include_uncommitted, progress_cb=progress_cb, code_parser=code_parser, find_repo_path_fn=_find_repo_path, resolve_project_id_fn=_resolve_project_id, is_branch_match_fn=_is_branch_match, calculate_savings_fn=_calculate_savings)


def knowledge_find_tests(target: str, repo_id: Optional[str] = None, project_id: str = "default", branch: str = "master", db: Optional[KnowledgeDB] = None) -> Dict[str, Any]:
    return _tools.knowledge_find_tests(target=target, repo_id=repo_id, project_id=project_id, branch=branch, db=db or globals()["db"], calculate_savings_fn=_calculate_savings)


def knowledge_get_project_rules(topic: Optional[str] = None, project_id: str = "default", db: Optional[KnowledgeDB] = None) -> Dict[str, Any]:
    return _tools.knowledge_get_project_rules(topic=topic, project_id=project_id, db=db or globals()["db"])


# ──────────────────────────────────────────────────────────────────────────────
# Реестр MCP и диспетчеризация
# ──────────────────────────────────────────────────────────────────────────────

def _get_server_request_context():
    try:
        return server.request_context
    except (LookupError, Exception):
        return None


def _get_registry() -> ToolRegistry:
    return ToolRegistry(
        db=globals()["db"],
        server_request_context_fn=_get_server_request_context,
        search_fn=knowledge_search,
        read_file_fn=knowledge_read_file,
        find_symbol_fn=knowledge_find_symbol,
        get_callers_fn=knowledge_get_callers,
        get_callees_fn=knowledge_get_callees,
        get_hierarchy_fn=knowledge_get_hierarchy,
        impact_fn=knowledge_impact_analysis,
        review_context_fn=knowledge_review_context,
        find_tests_fn=knowledge_find_tests,
        get_rules_fn=knowledge_get_project_rules,
        resolve_project_id_fn=_resolve_project_id,
        is_branch_match_fn=_is_branch_match
    )


_TOOL_HANDLERS: Dict[str, Callable[[dict], Any]] = {
    "knowledge_search": lambda args: _handle_search(args, db, knowledge_search),
    "knowledge_get_chunk": lambda args: _handle_get_chunk(args, db),
    "knowledge_read_file": lambda args: _handle_read_file(args, db, knowledge_read_file),
    "knowledge_delete_repo": _handle_delete_repo,
    "knowledge_sync_repo": _handle_sync_repo,
    "knowledge_find_symbol": lambda args: _handle_find_symbol(args, db, knowledge_find_symbol),
    "knowledge_get_callers": lambda args: _handle_get_callers(args, db, knowledge_get_callers),
    "knowledge_get_callees": lambda args: _handle_get_callees(args, db, knowledge_get_callees),
    "knowledge_get_hierarchy": lambda args: _handle_get_hierarchy(args, db, knowledge_get_hierarchy),
    "knowledge_impact_analysis": lambda args: _handle_impact_analysis(args, db, knowledge_impact_analysis),
    "knowledge_review_context": lambda args: _handle_review_context(
        args, db, knowledge_review_context,
        server_request_context=_get_server_request_context(),
        resolve_project_id_fn=_resolve_project_id,
        is_branch_match_fn=_is_branch_match
    ),
    "knowledge_find_tests": lambda args: _handle_find_tests(args, db, knowledge_find_tests),
    "knowledge_get_project_rules": lambda args: _handle_get_project_rules(args, db, knowledge_get_project_rules),
}


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Экспорт списка доступных MCP инструментов."""
    return _TOOL_SCHEMAS


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
    async def _fastmcp_search(query: str, repo_ids: Optional[list[str]] = None, project_id: str = "default", branch: str = "master", limit: int = 10) -> str:
        res = await _handle_search({"query": query, "repo_ids": repo_ids, "project_id": project_id, "branch": branch, "limit": limit}, db, knowledge_search)
        return res[0].text if res else ""

    @fastmcp_server.tool(name="knowledge_review_context")
    def _fastmcp_review_context(repo_id: Optional[str] = None, repo_ids: Optional[list[str]] = None, project_id: str = "default", branch: str = "master", compare_branch: str = "master", changed_files: Optional[list[str]] = None, file_path: Optional[str] = None, include_uncommitted: bool = True, max_depth: int = 2) -> dict:
        if file_path and not changed_files:
            changed_files = [file_path]
        return knowledge_review_context(repo_id=repo_id, repo_ids=repo_ids, project_id=project_id, branch=branch, compare_branch=compare_branch, changed_files=changed_files, include_uncommitted=include_uncommitted, max_depth=max_depth, db=db)

    @fastmcp_server.tool(name="knowledge_find_tests")
    def _fastmcp_find_tests(target: str, repo_id: Optional[str] = None, project_id: str = "default", branch: str = "master") -> dict:
        return knowledge_find_tests(target=target, repo_id=repo_id, project_id=project_id, branch=branch, db=db)

    @fastmcp_server.tool(name="knowledge_get_project_rules")
    def _fastmcp_get_project_rules(topic: Optional[str] = None, project_id: str = "default") -> dict:
        return knowledge_get_project_rules(topic=topic, project_id=project_id, db=db)
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
            async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
                await server.run(streams[0], streams[1], server.create_initialization_options())
            return Response()

        return Starlette(routes=[Route("/sse", endpoint=handle_sse, methods=["GET"]), Mount("/messages/", app=sse_transport.handle_post_message)])
    except Exception as e:
        logger.warning(f"Failed to create SSE app: {e}")
        return None


async def start_mcp_server(db_path: str, enable_embeddings: bool = True, read_only: bool = True):
    """Инициализация БД и запуск MCP-сервера по STDIO."""
    global db, use_embeddings

    with contextlib.redirect_stdout(sys.stderr):
        db = KnowledgeDB(db_path, read_only=read_only)
    use_embeddings = enable_embeddings

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
