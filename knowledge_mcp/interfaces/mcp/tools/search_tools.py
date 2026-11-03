"""MCP-инструменты поиска и чтения файлов в базе знаний (Search Tools)."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import mcp.types as types

from knowledge_mcp.core.config import resolve_workspace_targets
from knowledge_mcp.db.models import KnowledgeDB

logger = logging.getLogger("knowledge_mcp.interfaces.mcp.search")


def _err(msg: str) -> List[types.TextContent]:
    """Формирует сообщение об ошибке в формате TextContent."""
    return [types.TextContent(type="text", text=msg)]


def knowledge_search(
    query: str,
    repo_ids: Optional[List[str]] = None,
    project_id: Optional[str] = "default",
    branch: Optional[str] = "master",
    limit: int = 10,
    query_vector: Optional[List[float]] = None,
    db: Optional[KnowledgeDB] = None,
    resolve_project_id_fn: Optional[Callable[..., str]] = None,
    run_jit_check_fn: Optional[Callable[..., int]] = None,
    calculate_savings_fn: Optional[Callable[..., Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Полнотекстовый и гибридный поиск по базе знаний с JIT и расчетом экономии токенов."""
    if not db:
        raise RuntimeError("KnowledgeDB не инициализирована.")

    p_id = resolve_project_id_fn(project_id, branch, repo_ids) if resolve_project_id_fn else (project_id or "default")
    b_name = branch or "master"

    # JIT-проверка перед поиском
    if run_jit_check_fn:
        run_jit_check_fn(repo_ids=repo_ids, project_id=p_id, branch=b_name, db_instance=db)

    if query_vector is not None:
        results = db.search_chunks_hybrid(
            query, query_vector=query_vector, repo_ids=repo_ids, project_id=p_id, branch=b_name, limit=limit
        )
    else:
        results = db.search_chunks_fts(
            query, repo_ids=repo_ids, project_id=p_id, branch=b_name, limit=limit
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


def knowledge_read_file(
    path: str,
    repo_id: Optional[str] = None,
    project_id: str = "default",
    branch: str = "master",
    db: Optional[KnowledgeDB] = None,
    find_repo_path_fn: Optional[Callable[..., Optional[Path]]] = None,
    is_branch_match_fn: Optional[Callable[..., bool]] = None
) -> Optional[str]:
    """Читает полный контент файла из CAS или файловой системы рабочей копии."""
    if not db:
        raise RuntimeError("KnowledgeDB не инициализирована.")

    norm_path = path.replace('\\', '/').strip('/')

    # 1. Первичная попытка: прямое чтение с диска активной рабочей копии
    repo_path = None
    if repo_id and find_repo_path_fn:
        repo_path = find_repo_path_fn(project_id, repo_id, branch)
    else:
        # Автоматический поиск репозитория в воркспейсе, если repo_id не передан
        targets = resolve_workspace_targets(project_id)
        for t in targets:
            match = is_branch_match_fn(branch, t.get("branch"), t.get("task_name")) if is_branch_match_fn else True
            if not t.get("is_doc") and match:
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
                    db.upsert_file_content(text, language=file_on_disk.suffix.lstrip('.'))
                except Exception:
                    pass
                return text
            except Exception:
                pass

    # 2. Чтение из CAS таблицы file_contents
    content = db.get_file_content(norm_path, repo_id=repo_id, project_id=project_id, branch=branch)
    if content is not None:
        if isinstance(content, dict):
            return content.get("content", "")
        return str(content)

    return None


async def handle_search(arguments: dict, db: KnowledgeDB, search_fn: Callable) -> List[types.TextContent]:
    """Обработчик MCP-запроса knowledge_search."""
    query = arguments.get("query")
    if not query:
        return _err("Error: 'query' argument is required.")

    repo_ids = arguments.get("repo_ids", [])
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")
    limit = arguments.get("limit", 10)
    use_embeddings = arguments.get("use_embeddings", True)

    vector = None
    if use_embeddings and hasattr(db, "embedder") and db.embedder is not None:
        try:
            vector = await asyncio.to_thread(db.embedder.embed_text, query)
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")

    try:
        res = await asyncio.to_thread(
            search_fn,
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

        header = f"<!-- context_savings: {json.dumps(savings)} -->\n[Context Savings: {savings['saved_tokens']} tokens saved ({savings.get('saved_percent', 0)}% reduction)]\n\n"
        return [types.TextContent(type="text", text=header + "\n".join(formatted))]

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return _err(f"Knowledge Base DB Error: {e}")


async def handle_get_chunk(arguments: dict, db: KnowledgeDB) -> List[types.TextContent]:
    """Обработчик MCP-запроса knowledge_get_chunk: получить контент чанка по ID."""
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


async def handle_read_file(arguments: dict, db: KnowledgeDB, read_file_fn: Callable) -> List[types.TextContent]:
    """Обработчик MCP-запроса knowledge_read_file: получить полный контент файла."""
    path = arguments.get("path") or arguments.get("file_path")
    if not path:
        return _err("Error: 'path' argument is required.")

    repo_id = arguments.get("repo_id")
    project_id = arguments.get("project_id", "default")
    branch = arguments.get("branch", "master")

    try:
        content = await asyncio.to_thread(
            read_file_fn, path, repo_id=repo_id, project_id=project_id, branch=branch, db=db
        )
        if content is None:
            return _err(f"File '{path}' not found in the knowledge base.")

        return [types.TextContent(type="text", text=content)]
    except Exception as e:
        logger.error(f"Read file failed: {e}")
        return _err(f"Database query error: {e}")
