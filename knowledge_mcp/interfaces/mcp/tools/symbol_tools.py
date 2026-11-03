"""MCP-инструменты поиска и анализа графа символов (Symbol Tools)."""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

import mcp.types as types

from knowledge_mcp.db.models import KnowledgeDB

logger = logging.getLogger("knowledge_mcp.interfaces.mcp.symbols")


def _err(msg: str) -> List[types.TextContent]:
    """Формирует сообщение об ошибке в формате TextContent."""
    return [types.TextContent(type="text", text=msg)]


def knowledge_find_symbol(
    name_pattern: str,
    kind: Optional[str] = None,
    repo_ids: Optional[List[str]] = None,
    project_id: str = "default",
    branch: str = "master",
    limit: int = 20,
    db: Optional[KnowledgeDB] = None,
    resolve_project_id_fn: Optional[Callable[..., str]] = None,
    run_jit_check_fn: Optional[Callable[..., int]] = None,
    calculate_savings_fn: Optional[Callable[..., Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Поиск символа по имени или паттерну с фильтрацией по project_id и branch."""
    if not db:
        raise RuntimeError("KnowledgeDB не инициализирована.")

    p_id = resolve_project_id_fn(project_id, branch, repo_ids) if resolve_project_id_fn else (project_id or "default")

    # JIT-проверка перед поиском символов
    if run_jit_check_fn:
        run_jit_check_fn(repo_ids=repo_ids, project_id=p_id, branch=branch, db_instance=db)

    results = db.find_symbols(
        name_pattern=name_pattern,
        kind=kind,
        repo_ids=repo_ids,
        limit=limit,
        project_id=p_id,
        branch=branch
    )
    results_list = [dict(r) for r in results]
    files_involved = [r['path'] for r in results_list if 'path' in r]
    if calculate_savings_fn:
        savings = calculate_savings_fn(results_list, files_involved, db_instance=db)
    else:
        savings = {
            "baseline_tokens": len(files_involved) * 1500,
            "returned_tokens": 300,
            "saved_tokens": max(0, len(files_involved) * 1500 - 300),
            "multiplier": 3.0,
            "saved_percent": 80
        }

    repo_for_log = repo_ids[0] if repo_ids and len(repo_ids) == 1 else None
    db.log_token_savings(
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
    db: Optional[KnowledgeDB] = None,
    calculate_savings_fn: Optional[Callable[..., Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Возвращает вызывающие символы (callers) для symbol_id с метрикой context_savings."""
    if not db:
        raise RuntimeError("KnowledgeDB не инициализирована.")

    results = db.get_symbol_callers(symbol_id, project_id=project_id, branch=branch)
    results_list = [dict(r) for r in results]
    files_involved = [r['path'] for r in results_list if 'path' in r]
    if calculate_savings_fn:
        savings = calculate_savings_fn(results_list, files_involved, db_instance=db)
    else:
        savings = {
            "baseline_tokens": len(files_involved) * 1500,
            "returned_tokens": 300,
            "saved_tokens": max(0, len(files_involved) * 1500 - 300),
            "multiplier": 3.0,
            "saved_percent": 80
        }

    repo_for_log = results_list[0]['repo_id'] if results_list and 'repo_id' in results_list[0] else None
    db.log_token_savings(
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
    db: Optional[KnowledgeDB] = None,
    calculate_savings_fn: Optional[Callable[..., Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Возвращает вызываемые символы (callees) для symbol_id с метрикой context_savings."""
    if not db:
        raise RuntimeError("KnowledgeDB не инициализирована.")

    results = db.get_symbol_callees(symbol_id, project_id=project_id, branch=branch)
    results_list = [dict(r) for r in results]
    files_involved = [r['path'] for r in results_list if 'path' in r]
    if calculate_savings_fn:
        savings = calculate_savings_fn(results_list, files_involved, db_instance=db)
    else:
        savings = {
            "baseline_tokens": len(files_involved) * 1500,
            "returned_tokens": 300,
            "saved_tokens": max(0, len(files_involved) * 1500 - 300),
            "multiplier": 3.0,
            "saved_percent": 80
        }

    repo_for_log = results_list[0]['repo_id'] if results_list and 'repo_id' in results_list[0] else None
    db.log_token_savings(
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
    db: Optional[KnowledgeDB] = None,
    calculate_savings_fn: Optional[Callable[..., Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Возвращает иерархию наследования для символа с метрикой context_savings."""
    if not db:
        raise RuntimeError("KnowledgeDB не инициализирована.")

    results = db.get_hierarchy(symbol_id, project_id=project_id, branch=branch)
    results_list = [dict(r) for r in results]
    files_involved = [r['path'] for r in results_list if 'path' in r]
    if calculate_savings_fn:
        savings = calculate_savings_fn(results_list, files_involved, db_instance=db)
    else:
        savings = {
            "baseline_tokens": len(files_involved) * 1500,
            "returned_tokens": 300,
            "saved_tokens": max(0, len(files_involved) * 1500 - 300),
            "multiplier": 3.0,
            "saved_percent": 80
        }

    repo_for_log = results_list[0]['repo_id'] if results_list and 'repo_id' in results_list[0] else None
    db.log_token_savings(
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
    db: Optional[KnowledgeDB] = None,
    calculate_savings_fn: Optional[Callable[..., Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Выполняет рекурсивный анализ влияния символа по графу зависимостей."""
    if not db:
        raise RuntimeError("KnowledgeDB не инициализирована.")

    results = db.get_impact_analysis(symbol_id, max_depth=max_depth, project_id=project_id, branch=branch)

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

    if calculate_savings_fn:
        savings = calculate_savings_fn(impacted_list, files_involved, db_instance=db)
    else:
        savings = {
            "baseline_tokens": len(files_involved) * 1500,
            "returned_tokens": 300,
            "saved_tokens": max(0, len(files_involved) * 1500 - 300),
            "multiplier": 3.0,
            "saved_percent": 80
        }

    db.log_token_savings(
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


async def handle_find_symbol(arguments: dict, db: KnowledgeDB, find_symbol_fn: Callable) -> List[types.TextContent]:
    """Обработчик MCP-запроса knowledge_find_symbol."""
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
            find_symbol_fn,
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
        header = f"<!-- context_savings: {json.dumps(savings)} -->\n[Context Savings: {savings['saved_tokens']} tokens saved ({savings.get('saved_percent', 0)}% reduction)]\n\n"
        return [types.TextContent(type="text", text=header + "\n".join(formatted))]
    except Exception as e:
        logger.error(f"Find symbol failed: {e}")
        return _err(f"Database query error: {e}")


async def handle_get_callers(arguments: dict, db: KnowledgeDB, get_callers_fn: Callable) -> List[types.TextContent]:
    """Обработчик MCP-запроса knowledge_get_callers."""
    symbol_id = arguments.get("symbol_id")
    if not symbol_id:
        return _err("Error: 'symbol_id' is required.")
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")

    try:
        res = await asyncio.to_thread(get_callers_fn, symbol_id, project_id=project_id, branch=branch, db=db)
        results = res.get("callers", [])
        savings = res.get("context_savings", {})

        if not results:
            return _err(f"No graph edges found for symbol {symbol_id} in this context.")

        header = f"<!-- context_savings: {json.dumps(savings)} -->\n[Context Savings: {savings['saved_tokens']} tokens saved ({savings.get('saved_percent', 0)}% reduction)]\n\n"
        lines = [header + f"Callers for symbol ID {symbol_id}:"] + [
            f"- [{r['edge_kind']}] ID: {r['id']} | `{r['qualified_name']}` ({r['kind']} in {r['repo_id']}/{r['path']})"
            for r in results
        ]
        return [types.TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        return _err(f"Database query error: {e}")


async def handle_get_callees(arguments: dict, db: KnowledgeDB, get_callees_fn: Callable) -> List[types.TextContent]:
    """Обработчик MCP-запроса knowledge_get_callees."""
    symbol_id = arguments.get("symbol_id")
    if not symbol_id:
        return _err("Error: 'symbol_id' is required.")
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")

    try:
        res = await asyncio.to_thread(get_callees_fn, symbol_id, project_id=project_id, branch=branch, db=db)
        results = res.get("callees", [])
        savings = res.get("context_savings", {})

        if not results:
            return _err(f"No graph edges found for symbol {symbol_id} in this context.")

        header = f"<!-- context_savings: {json.dumps(savings)} -->\n[Context Savings: {savings['saved_tokens']} tokens saved ({savings.get('saved_percent', 0)}% reduction)]\n\n"
        lines = [header + f"Callees of symbol ID {symbol_id}:"] + [
            f"- [{r['edge_kind']}] ID: {r['id']} | `{r['qualified_name']}` ({r['kind']} in {r['repo_id']}/{r['path']})"
            for r in results
        ]
        return [types.TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        return _err(f"Database query error: {e}")


async def handle_get_hierarchy(arguments: dict, db: KnowledgeDB, get_hierarchy_fn: Callable) -> List[types.TextContent]:
    """Обработчик MCP-запроса knowledge_get_hierarchy."""
    symbol_id = arguments.get("symbol_id")
    if not symbol_id:
        return _err("Error: 'symbol_id' is required.")
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")

    try:
        res = await asyncio.to_thread(get_hierarchy_fn, symbol_id, project_id=project_id, branch=branch, db=db)
        results = res.get("hierarchy", [])
        savings = res.get("context_savings", {})

        if not results:
            return _err(f"No graph edges found for symbol {symbol_id} in this context.")

        header = f"<!-- context_savings: {json.dumps(savings)} -->\n[Context Savings: {savings['saved_tokens']} tokens saved ({savings.get('saved_percent', 0)}% reduction)]\n\n"
        lines = [header + f"Hierarchy for symbol ID {symbol_id}:"] + [
            f"- [{r['edge_kind']}] ID: {r['id']} | `{r['qualified_name']}` ({r['kind']} in {r['repo_id']}/{r['path']})"
            for r in results
        ]
        return [types.TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        return _err(f"Database query error: {e}")


async def handle_impact_analysis(arguments: dict, db: KnowledgeDB, impact_fn: Callable) -> List[types.TextContent]:
    """Обработчик MCP-запроса knowledge_impact_analysis."""
    symbol_id = arguments.get("symbol_id")
    max_depth = arguments.get("max_depth", 3)
    if not symbol_id:
        return _err("Error: 'symbol_id' is required.")
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")

    try:
        res = await asyncio.to_thread(impact_fn, symbol_id, max_depth=max_depth, project_id=project_id, branch=branch, db=db)
        results = res.get("impacted", [])
        savings = res.get("context_savings", {})

        if not results:
            return _err(f"No dependent symbols found for symbol {symbol_id} within depth {max_depth}.")

        header = f"<!-- context_savings: {json.dumps(savings)} -->\n[Context Savings: {savings['saved_tokens']} tokens saved ({savings.get('saved_percent', 0)}% reduction)]\n\n"
        lines = [header + f"Impact Analysis for symbol ID {symbol_id} (max depth {max_depth}):"]
        current_depth = 0
        for item in sorted(results, key=lambda x: x['depth']):
            if item['depth'] != current_depth:
                current_depth = item['depth']
                lines.append(f"\n--- Depth {current_depth} Dependencies ---")
            lines.append(f"- [{item['edge_kind']}] `{item['qualified_name']}` ({item['kind']}) in `{item['repo_id']}/{item['path']}` (ID: {item['id']})")

        return [types.TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error(f"Impact query failed: {e}")
        return _err(f"Database query error: {e}")
