import asyncio
import logging
import contextlib
import sys
from typing import Any, Callable

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from knowledge_mcp.db.models import KnowledgeDB
# НЕ импортируем LocalEmbedder на верхнем уровне!
# embeddings.py -> sentence_transformers -> torch (~30 сек на импорт).
# Импортируем лениво, только при первом вызове knowledge_search.

logger = logging.getLogger(__name__)

server = Server("knowledge-mcp")
db: KnowledgeDB = None
embedder = None  # Будет LocalEmbedder, загруженный лениво
use_embeddings = True


# ──────────────────────────────────────────────────────────────────────────────
# Инициализация эмбеддера (ленивая)
# ──────────────────────────────────────────────────────────────────────────────

def _init_embedder_sync():
    """Синхронная инициализация модели (вызывается в отдельном потоке).

    КРИТИЧЕСКИ ВАЖНО: перенаправляем stdout -> stderr на время загрузки!
    torch и sentence_transformers печатают progress bars и warnings в stdout,
    что ломает MCP JSON-RPC протокол (MCP общается через stdout).
    """
    import sys
    original_stdout = sys.stdout
    sys.stdout = sys.stderr  # Всё что хочет напечатать torch — уйдёт в stderr
    try:
        from knowledge_mcp.core.embeddings import LocalEmbedder
        return LocalEmbedder()
    finally:
        sys.stdout = original_stdout  # Восстанавливаем stdout для MCP


def _embed_text_sync(emb, text: str):
    """Синхронный вызов inference (вызывается в отдельном потоке)."""
    return emb.embed_text(text)


# ──────────────────────────────────────────────────────────────────────────────
# Список инструментов MCP
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_SCHEMAS = [
    types.Tool(
        name="knowledge_search",
        description=(
            "Search the local knowledge base across multiple repositories. "
            "Uses triple hybrid search (Full-Text + Vector Embeddings + Graph routing) for high recall. "
            "Returns relevant code and documentation chunks with trust levels ('verified' for code, 'hint' for docs)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query (concept, code snippet, or natural language question)."},
                "repo_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of repo_ids to restrict the search. If omitted, searches all repositories."
                },
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
        description="Retrieve the complete raw content of a file from the knowledge base by its file path.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The exact path of the file (e.g. 'docs/data-access/README.md')."},
                "repo_id": {"type": "string", "description": "Optional repo_id if the path exists in multiple repositories."}
            },
            "required": ["path"]
        }
    ),
    types.Tool(
        name="knowledge_delete_repo",
        description="Delete all knowledge base data associated with a specific repository ID. This removes all scanned files, chunks, and embeddings for that repository.",
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
        description="Queue a sync (update) of the local knowledge base for a specific repository. The request is delegated to the Daemon API which manages the sync queue.",
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
        description="Find a code symbol (class, method, interface, etc.) by its name or pattern. Use asterisks for wildcards (e.g. '*Repository*').",
        inputSchema={
            "type": "object",
            "properties": {
                "name_pattern": {"type": "string", "description": "The name or wildcard pattern of the symbol."},
                "kind": {"type": "string", "description": "Optional kind of symbol (class, method, interface, property, etc)."},
                "repo_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of repo_ids to restrict the search."
                }
            },
            "required": ["name_pattern"]
        }
    ),
    types.Tool(
        name="knowledge_get_callers",
        description="Find all symbols that call or reference the specified symbol by its numeric ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_id": {"type": "integer", "description": "The exact numeric ID of the target symbol."}
            },
            "required": ["symbol_id"]
        }
    ),
    types.Tool(
        name="knowledge_get_callees",
        description="Find all symbols that are called or referenced by the specified symbol by its numeric ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_id": {"type": "integer", "description": "The exact numeric ID of the source symbol."}
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
                "symbol_id": {"type": "integer", "description": "The exact numeric ID of the symbol."}
            },
            "required": ["symbol_id"]
        }
    ),
    types.Tool(
        name="knowledge_impact_analysis",
        description="Perform a recursive graph traversal to find all symbols and files that depend on the specified symbol. Useful for blast radius estimation.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_id": {"type": "integer", "description": "The exact numeric ID of the target symbol."},
                "max_depth": {"type": "integer", "description": "Max recursion depth (default 3, max 5)."}
            },
            "required": ["symbol_id"]
        }
    ),
]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Экспортируем MCP инструменты с универсальным описанием для любых LLM (Codex, Claude, Gemini)."""
    return _TOOL_SCHEMAS


# ──────────────────────────────────────────────────────────────────────────────
# Обработчики инструментов (один handler = одна ответственность)
# ──────────────────────────────────────────────────────────────────────────────

def _err(msg: str) -> list[types.TextContent]:
    """Хелпер: возвращает текстовое сообщение об ошибке."""
    return [types.TextContent(type="text", text=msg)]


async def _handle_search(arguments: dict) -> list[types.TextContent]:
    """knowledge_search: гибридный поиск FTS5 + Vector KNN + Graph."""
    query = arguments.get("query")
    if not query:
        return _err("Error: 'query' argument is required.")

    repo_ids = arguments.get("repo_ids", [])
    limit = arguments.get("limit", 10)

    try:
        results = []
        if use_embeddings:
            global embedder
            if embedder is None:
                logger.info("Lazy loading LocalEmbedder in background thread (~30s first time)...")
                # asyncio.to_thread: не блокируем event loop, MCP-протокол остаётся живым
                embedder = await asyncio.to_thread(_init_embedder_sync)
                logger.info("Embedder ready!")
            # Inference тоже в отдельном потоке (CPU-bound операция)
            vector = await asyncio.to_thread(_embed_text_sync, embedder, query, is_query=True)
            if vector:
                results = db.search_chunks_hybrid(query, vector, repo_ids=repo_ids, limit=limit)
            else:
                results = db.search_chunks_fts(query, repo_ids=repo_ids, limit=limit)
        else:
            results = db.search_chunks_fts(query, repo_ids=repo_ids, limit=limit)

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
        return [types.TextContent(type="text", text="\n".join(formatted))]

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
    try:
        content = db.get_file_content(path, repo_id)
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


async def _handle_delete_repo(arguments: dict) -> list[types.TextContent]:
    """knowledge_delete_repo: делегирует удаление репозитория Daemon-процессу через HTTP.

    MCP-сервер работает в Read-Only режиме и не может выполнять записи в SQLite.
    Запрос на удаление отправляется Daemon API с поддержкой Basic Auth.
    """
    repo_id = arguments.get("repo_id")
    if not repo_id:
        return _err("Error: 'repo_id' argument is required.")

    try:
        import urllib.request
        import urllib.error

        url = f"http://127.0.0.1:8000/api/repo/{repo_id}"
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
    """knowledge_sync_repo: делегирует синхронизацию репозитория Daemon-процессу через HTTP.

    MCP-сервер работает в Read-Only режиме. Задача добавляется в DB-очередь
    через Daemon API с авторизацией Basic Auth.
    """
    repo_id = arguments.get("repo_id")
    if not repo_id:
        return _err("Error: 'repo_id' argument is required.")

    try:
        import urllib.request
        import urllib.error

        url = f"http://127.0.0.1:8000/api/sync/{repo_id}"
        headers = _get_auth_headers()
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, method="POST", data=b"", headers=headers)

        def _trigger_sync():
            with urllib.request.urlopen(req) as response:
                return response.read()

        logger.info(f"Delegating sync of '{repo_id}' to Daemon API via POST {url}")
        await asyncio.to_thread(_trigger_sync)

        return [types.TextContent(type="text", text=f"Successfully queued sync for repository '{repo_id}' via Daemon API. The background worker will process it shortly.")]
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

    try:
        results = db.find_symbols(pattern, kind=kind, repo_ids=repo_ids, limit=20)
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
        return [types.TextContent(type="text", text="\n".join(formatted))]
    except Exception as e:
        logger.error(f"Find symbol failed: {e}")
        return _err(f"Database query error: {e}")


async def _handle_get_callers(arguments: dict) -> list[types.TextContent]:
    """knowledge_get_callers: кто вызывает данный символ."""
    symbol_id = arguments.get("symbol_id")
    if not symbol_id:
        return _err("Error: 'symbol_id' is required.")
    try:
        results = db.get_callers(symbol_id)
        if not results:
            return _err(f"No graph edges found for symbol {symbol_id} in this context.")
        lines = [f"Callers for symbol ID {symbol_id}:"] + [
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
    try:
        results = db.get_callees(symbol_id)
        if not results:
            return _err(f"No graph edges found for symbol {symbol_id} in this context.")
        lines = [f"Callees of symbol ID {symbol_id}:"] + [
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
    try:
        results = db.get_hierarchy(symbol_id)
        if not results:
            return _err(f"No graph edges found for symbol {symbol_id} in this context.")
        lines = [f"Hierarchy for symbol ID {symbol_id}:"] + [
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

    try:
        results = db.get_impact_analysis(symbol_id, max_depth=max_depth)
        if not results:
            return _err(f"No dependent symbols found for symbol {symbol_id} within depth {max_depth}.")

        lines = [f"Impact Analysis for symbol ID {symbol_id} (max depth {max_depth}):"]
        current_depth = 0
        for item in sorted(results, key=lambda x: x['depth']):
            if item['depth'] != current_depth:
                current_depth = item['depth']
                lines.append(f"\n--- Depth {current_depth} Dependencies ---")
            r = item['symbol']
            lines.append(
                f"- [{item['edge_kind']}] ID: {r['id']} | `{r['qualified_name']}` ({r['kind']} in {r['repo_id']}/{r['path']})"
            )
        return [types.TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error(f"Impact analysis failed: {e}")
        return _err(f"Database query error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch-таблица: имя инструмента → обработчик
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_HANDLERS: dict[str, Callable[[dict], Any]] = {
    "knowledge_search":        _handle_search,
    "knowledge_get_chunk":     _handle_get_chunk,
    "knowledge_read_file":     _handle_read_file,
    "knowledge_delete_repo":   _handle_delete_repo,
    "knowledge_sync_repo":     _handle_sync_repo,
    "knowledge_find_symbol":   _handle_find_symbol,
    "knowledge_get_callers":   _handle_get_callers,
    "knowledge_get_callees":   _handle_get_callees,
    "knowledge_get_hierarchy": _handle_get_hierarchy,
    "knowledge_impact_analysis": _handle_impact_analysis,
}


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Диспатчер MCP-инструментов. Делегирует вызов конкретному обработчику."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    return await handler(arguments)


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────

async def start_mcp_server(db_path: str, enable_embeddings: bool = True):
    """Инициализация БД и запуск MCP-сервера по STDIO (стандарт для локальных агентов)."""
    global db, use_embeddings

    # Любой лишний stdout до MCP-кадров ломает handshake.
    # На время инициализации принудительно уводим stdout в stderr.
    with contextlib.redirect_stdout(sys.stderr):
        db = KnowledgeDB(db_path, read_only=True)
    use_embeddings = enable_embeddings
    # Ленивая загрузка embedder = LocalEmbedder() перенесена внутрь _handle_search,
    # чтобы избежать 10-секундного таймаута при инициализации (handshake) протокола MCP.

    # MCP-протокол общения через потоки stdin/stdout
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )
