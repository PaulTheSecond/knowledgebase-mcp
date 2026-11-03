"""Реестр и роутер MCP-инструментов (Tool Registry)."""

import logging
from typing import Any, Callable, Dict, List, Optional

import mcp.types as types

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.interfaces.mcp.definitions import TOOL_SCHEMAS
from knowledge_mcp.interfaces.mcp.tools.admin_tools import (
    handle_delete_repo,
    handle_sync_repo,
)
from knowledge_mcp.interfaces.mcp.tools.review_tools import (
    handle_find_tests,
    handle_get_project_rules,
    handle_review_context,
    knowledge_find_tests,
    knowledge_get_project_rules,
    knowledge_review_context,
)
from knowledge_mcp.interfaces.mcp.tools.search_tools import (
    handle_get_chunk,
    handle_read_file,
    handle_search,
    knowledge_read_file,
    knowledge_search,
)
from knowledge_mcp.interfaces.mcp.tools.symbol_tools import (
    handle_find_symbol,
    handle_get_callees,
    handle_get_callers,
    handle_get_hierarchy,
    handle_impact_analysis,
    knowledge_find_symbol,
    knowledge_get_callees,
    knowledge_get_callers,
    knowledge_get_hierarchy,
    knowledge_impact_analysis,
)

logger = logging.getLogger("knowledge_mcp.interfaces.mcp.registry")


class ToolRegistry:
    """Централизованный диспетчер и реестр обработчиков MCP протокола."""

    def __init__(
        self,
        db: KnowledgeDB,
        server_request_context_fn: Optional[Callable[[], Any]] = None,
        search_fn: Optional[Callable] = None,
        read_file_fn: Optional[Callable] = None,
        find_symbol_fn: Optional[Callable] = None,
        get_callers_fn: Optional[Callable] = None,
        get_callees_fn: Optional[Callable] = None,
        get_hierarchy_fn: Optional[Callable] = None,
        impact_fn: Optional[Callable] = None,
        review_context_fn: Optional[Callable] = None,
        find_tests_fn: Optional[Callable] = None,
        get_rules_fn: Optional[Callable] = None,
        resolve_project_id_fn: Optional[Callable] = None,
        is_branch_match_fn: Optional[Callable] = None
    ):
        self.db = db
        self._server_req_ctx_fn = server_request_context_fn
        self.search_fn = search_fn or knowledge_search
        self.read_file_fn = read_file_fn or knowledge_read_file
        self.find_symbol_fn = find_symbol_fn or knowledge_find_symbol
        self.get_callers_fn = get_callers_fn or knowledge_get_callers
        self.get_callees_fn = get_callees_fn or knowledge_get_callees
        self.get_hierarchy_fn = get_hierarchy_fn or knowledge_get_hierarchy
        self.impact_fn = impact_fn or knowledge_impact_analysis
        self.review_context_fn = review_context_fn or knowledge_review_context
        self.find_tests_fn = find_tests_fn or knowledge_find_tests
        self.get_rules_fn = get_rules_fn or knowledge_get_project_rules
        self.resolve_project_id_fn = resolve_project_id_fn
        self.is_branch_match_fn = is_branch_match_fn

    @property
    def schemas(self) -> List[types.Tool]:
        """Возвращает список схем всех зарегистрированных инструментов."""
        return TOOL_SCHEMAS

    def get_handlers(self) -> Dict[str, Callable[[dict], Any]]:
        """Возвращает отображение имени инструмента в асинхронный обработчик."""
        return {
            "knowledge_search": lambda args: handle_search(args, self.db, self.search_fn),
            "knowledge_get_chunk": lambda args: handle_get_chunk(args, self.db),
            "knowledge_read_file": lambda args: handle_read_file(args, self.db, self.read_file_fn),
            "knowledge_delete_repo": handle_delete_repo,
            "knowledge_sync_repo": handle_sync_repo,
            "knowledge_find_symbol": lambda args: handle_find_symbol(args, self.db, self.find_symbol_fn),
            "knowledge_get_callers": lambda args: handle_get_callers(args, self.db, self.get_callers_fn),
            "knowledge_get_callees": lambda args: handle_get_callees(args, self.db, self.get_callees_fn),
            "knowledge_get_hierarchy": lambda args: handle_get_hierarchy(args, self.db, self.get_hierarchy_fn),
            "knowledge_impact_analysis": lambda args: handle_impact_analysis(args, self.db, self.impact_fn),
            "knowledge_review_context": lambda args: handle_review_context(
                args,
                self.db,
                self.review_context_fn,
                server_request_context=self._server_req_ctx_fn() if self._server_req_ctx_fn else None,
                resolve_project_id_fn=self.resolve_project_id_fn,
                is_branch_match_fn=self.is_branch_match_fn
            ),
            "knowledge_find_tests": lambda args: handle_find_tests(args, self.db, self.find_tests_fn),
            "knowledge_get_project_rules": lambda args: handle_get_project_rules(args, self.db, self.get_rules_fn),
        }

    async def dispatch(self, name: str, arguments: dict) -> List[types.TextContent]:
        """Выполняет вызов инструмента по его имени."""
        handlers = self.get_handlers()
        handler = handlers.get(name)
        if handler is None:
            raise ValueError(f"Неизвестный инструмент: {name}")
        return await handler(arguments)
