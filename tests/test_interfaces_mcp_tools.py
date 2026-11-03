"""
Unit-тесты для интерфейсного слоя MCP (Interfaces Layer):
- knowledge_mcp/interfaces/mcp/tool_registry.py
- knowledge_mcp/interfaces/mcp/tools/admin_tools.py
- knowledge_mcp/interfaces/mcp/tools/search_tools.py
- knowledge_mcp/interfaces/mcp/tools/symbol_tools.py
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from knowledge_mcp.interfaces.mcp.tool_registry import ToolRegistry
from knowledge_mcp.interfaces.mcp.tools.admin_tools import (
    get_auth_headers,
    get_daemon_base_url,
    handle_delete_repo,
    handle_sync_repo,
)
from knowledge_mcp.interfaces.mcp.tools.search_tools import (
    handle_search,
    handle_get_chunk,
    handle_read_file,
)
from knowledge_mcp.interfaces.mcp.tools.symbol_tools import (
    handle_find_symbol,
    handle_get_callers,
    handle_get_callees,
    handle_get_hierarchy,
    handle_impact_analysis,
)


# -----------------------------------------------------------------------------
# 1. ToolRegistry Tests
# -----------------------------------------------------------------------------

def test_tool_registry_schemas_and_handlers():
    """Проверка регистрации схем и генерации обработчиков инструментов."""
    mock_db = MagicMock()
    registry = ToolRegistry(db=mock_db)

    schemas = registry.schemas
    assert len(schemas) > 0
    schema_names = {s.name for s in schemas}
    assert "knowledge_search" in schema_names
    assert "knowledge_find_symbol" in schema_names
    assert "knowledge_review_context" in schema_names

    handlers = registry.get_handlers()
    assert "knowledge_search" in handlers
    assert "knowledge_sync_repo" in handlers
    assert "knowledge_delete_repo" in handlers


@pytest.mark.asyncio
async def test_tool_registry_dispatch_success_and_unknown():
    """Проверка роутинга и ошибки при вызове несуществующего тула."""
    mock_db = MagicMock()
    registry = ToolRegistry(db=mock_db)

    # Диспетчеризация известного тула с некорректными аргументами возвращает сообщение об ошибке
    res = await registry.dispatch("knowledge_search", {})
    assert len(res) == 1
    assert "query" in res[0].text

    # Вызов несуществующего тула возбуждает ValueError
    with pytest.raises(ValueError, match="Неизвестный инструмент"):
        await registry.dispatch("non_existent_tool_xyz", {})


# -----------------------------------------------------------------------------
# 2. Admin Tools Tests
# -----------------------------------------------------------------------------

def test_admin_tools_auth_and_url(monkeypatch):
    """Проверка формирования Basic Auth и URL дашборда."""
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    monkeypatch.delenv("DAEMON_URL", raising=False)

    assert get_auth_headers() == {}
    assert get_daemon_base_url() == "http://127.0.0.1:8765"

    monkeypatch.setenv("DASHBOARD_PASSWORD", "secret123")
    monkeypatch.setenv("DAEMON_URL", "http://mydaemon:9000/")

    headers = get_auth_headers()
    assert "Authorization" in headers
    assert headers["Authorization"].startswith("Basic ")
    assert get_daemon_base_url() == "http://mydaemon:9000"


@pytest.mark.asyncio
async def test_admin_tools_delete_and_sync():
    """Проверка делегирования delete_repo и sync_repo в Daemon API."""
    # 1. Missing repo_id
    del_err = await handle_delete_repo({})
    assert "обязательный аргумент 'repo_id'" in del_err[0].text

    sync_err = await handle_sync_repo({})
    assert "обязательный аргумент 'repo_id'" in sync_err[0].text

    # 2. Успешный вызов
    with patch("urllib.request.urlopen") as mock_url:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"status": "ok"}'
        mock_resp.__enter__.return_value = mock_resp
        mock_url.return_value = mock_resp

        del_ok = await handle_delete_repo({"repo_id": "orbit_core"})
        assert "успешно отправлен" in del_ok[0].text

        sync_ok = await handle_sync_repo({"repo_id": "orbit_core"})
        assert "успешно поставлен" in sync_ok[0].text

    # 3. Исключение при обращении
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        del_fail = await handle_delete_repo({"repo_id": "orbit_core"})
        assert "Не удалось делегировать" in del_fail[0].text

        sync_fail = await handle_sync_repo({"repo_id": "orbit_core"})
        assert "Не удалось делегировать" in sync_fail[0].text


# -----------------------------------------------------------------------------
# 3. Search Tools Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_tools_handle_search():
    """Проверка обработчика handle_search."""
    mock_db = MagicMock()
    mock_search_fn = MagicMock()

    # Ошибка: нет query
    err = await handle_search({}, mock_db, mock_search_fn)
    assert "query" in err[0].text

    # Нет результатов
    mock_search_fn.return_value = {"results": [], "context_savings": {"saved_tokens": 0}}
    empty_res = await handle_search({"query": "hello"}, mock_db, mock_search_fn)
    assert "No results found" in empty_res[0].text

    # Есть результаты
    mock_search_fn.return_value = {
        "results": [
            {
                "id": "c1",
                "repo_id": "r1",
                "path": "test.py",
                "line_start": 1,
                "line_end": 10,
                "content": "def test():\n    pass\n",
                "trust": "high",
                "source_kind": "code"
            }
        ],
        "context_savings": {"saved_tokens": 1200, "saved_percent": 80}
    }
    found = await handle_search({"query": "hello"}, mock_db, mock_search_fn)
    assert "Chunk ID: c1" in found[0].text
    assert "Context Savings: 1200" in found[0].text


@pytest.mark.asyncio
async def test_search_tools_handle_get_chunk_and_read_file():
    """Проверка обработчиков handle_get_chunk и handle_read_file."""
    mock_db = MagicMock()

    # 1. handle_get_chunk
    err_chunk = await handle_get_chunk({}, mock_db)
    assert "chunk_id" in err_chunk[0].text

    mock_db.get_chunk_by_id.return_value = None
    not_found = await handle_get_chunk({"chunk_id": "c999"}, mock_db)
    assert "not found" in not_found[0].text

    mock_db.get_chunk_by_id.return_value = {
        "repo_id": "repo1",
        "path": "src/main.py",
        "trust": "high",
        "sha": "abcdef123456",
        "content": "print('hello')"
    }
    chunk_res = await handle_get_chunk({"chunk_id": "c1"}, mock_db)
    assert "Repository" in chunk_res[0].text
    assert "abcdef123456" in chunk_res[0].text

    # 2. handle_read_file
    mock_read_fn = MagicMock()
    err_file = await handle_read_file({}, mock_db, mock_read_fn)
    assert "path" in err_file[0].text

    mock_read_fn.return_value = None
    not_found_file = await handle_read_file({"path": "unknown.py"}, mock_db, mock_read_fn)
    assert "not found" in not_found_file[0].text

    mock_read_fn.return_value = "content of the file"
    file_res = await handle_read_file({"path": "known.py"}, mock_db, mock_read_fn)
    assert file_res[0].text == "content of the file"


# -----------------------------------------------------------------------------
# 4. Symbol Tools Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_symbol_tools_handlers():
    """Проверка всех обработчиков графа символов."""
    mock_db = MagicMock()

    # 1. handle_find_symbol
    mock_find_fn = MagicMock()
    assert "name_pattern" in (await handle_find_symbol({}, mock_db, mock_find_fn))[0].text

    mock_find_fn.return_value = {"symbols": [], "context_savings": {}}
    assert "No symbols found" in (await handle_find_symbol({"name_pattern": "x"}, mock_db, mock_find_fn))[0].text

    mock_find_fn.return_value = {
        "symbols": [{
            "id": 1,
            "name": "AuthService",
            "qualified_name": "app.AuthService",
            "repo_id": "repo_core",
            "path": "auth.py",
            "line_start": 10,
            "line_end": 50,
            "kind": "class",
            "language": "python",
            "signature": "class AuthService:",
            "docstring": "Service for authentication"
        }],
        "context_savings": {"saved_tokens": 1000}
    }
    sym_res = await handle_find_symbol({"name_pattern": "Auth"}, mock_db, mock_find_fn)
    assert "AuthService" in sym_res[0].text

    # 2. handle_get_callers
    mock_callers_fn = MagicMock()
    assert "symbol_id" in (await handle_get_callers({}, mock_db, mock_callers_fn))[0].text

    mock_callers_fn.return_value = {
        "callers": [{
            "id": 2,
            "name": "login_handler",
            "qualified_name": "app.login_handler",
            "repo_id": "repo_core",
            "path": "routes.py",
            "line_start": 5,
            "line_end": 15,
            "kind": "function",
            "language": "python",
            "edge_kind": "calls"
        }],
        "context_savings": {"saved_tokens": 500}
    }
    caller_res = await handle_get_callers({"symbol_id": 1}, mock_db, mock_callers_fn)
    assert "login_handler" in caller_res[0].text

    # 3. handle_get_callees
    mock_callees_fn = MagicMock()
    assert "symbol_id" in (await handle_get_callees({}, mock_db, mock_callees_fn))[0].text

    mock_callees_fn.return_value = {
        "callees": [{
            "id": 3,
            "name": "hash_password",
            "qualified_name": "app.hash_password",
            "repo_id": "repo_core",
            "path": "utils.py",
            "line_start": 1,
            "line_end": 5,
            "kind": "function",
            "language": "python",
            "edge_kind": "calls"
        }],
        "context_savings": {"saved_tokens": 400}
    }
    callee_res = await handle_get_callees({"symbol_id": 1}, mock_db, mock_callees_fn)
    assert "hash_password" in callee_res[0].text

    # 4. handle_get_hierarchy
    mock_hier_fn = MagicMock()
    assert "symbol_id" in (await handle_get_hierarchy({}, mock_db, mock_hier_fn))[0].text

    mock_hier_fn.return_value = {
        "hierarchy": [{
            "id": 4,
            "name": "BaseService",
            "qualified_name": "app.BaseService",
            "repo_id": "repo_core",
            "path": "base.py",
            "line_start": 1,
            "line_end": 20,
            "kind": "class",
            "language": "python",
            "edge_kind": "inherits"
        }],
        "context_savings": {"saved_tokens": 300}
    }
    hier_res = await handle_get_hierarchy({"symbol_id": 1}, mock_db, mock_hier_fn)
    assert "BaseService" in hier_res[0].text

    # 5. handle_impact_analysis
    mock_impact_fn = MagicMock()
    assert "symbol_id" in (await handle_impact_analysis({}, mock_db, mock_impact_fn))[0].text

    mock_impact_fn.return_value = {
        "impacted": [{
            "id": 5,
            "name": "Controller",
            "qualified_name": "app.Controller",
            "repo_id": "repo_core",
            "kind": "class",
            "path": "controller.py",
            "edge_kind": "calls",
            "depth": 1
        }],
        "context_savings": {"saved_tokens": 800}
    }
# -----------------------------------------------------------------------------
# 5. Core MCP Sync Functions Tests
# -----------------------------------------------------------------------------

def test_symbol_tools_sync_functions():
    """Проверка синхронных функций графа символов (поиск, callers, callees, hierarchy, impact)."""
    from knowledge_mcp.interfaces.mcp.tools.symbol_tools import (
        knowledge_find_symbol,
        knowledge_get_callers,
        knowledge_get_callees,
        knowledge_get_hierarchy,
        knowledge_impact_analysis,
    )
    mock_db = MagicMock()

    # 1. knowledge_find_symbol
    mock_db.find_symbols.return_value = [
        {"id": 10, "name": "Auth", "qualified_name": "app.Auth", "path": "auth.py"}
    ]
    res_sym = knowledge_find_symbol(name_pattern="Auth", db=mock_db)
    assert res_sym["total_symbols"] == 1
    assert mock_db.log_token_savings.called

    # 2. knowledge_get_callers
    mock_db.get_symbol_callers.return_value = [
        {"id": 11, "name": "Caller", "path": "caller.py", "repo_id": "r1"}
    ]
    res_callers = knowledge_get_callers(symbol_id=10, db=mock_db)
    assert res_callers["total_callers"] == 1

    # 3. knowledge_get_callees
    mock_db.get_symbol_callees.return_value = [
        {"id": 12, "name": "Callee", "path": "callee.py", "repo_id": "r1"}
    ]
    res_callees = knowledge_get_callees(symbol_id=10, db=mock_db)
    assert res_callees["total_callees"] == 1

    # 4. knowledge_get_hierarchy
    mock_db.get_hierarchy.return_value = [
        {"id": 13, "name": "Parent", "path": "parent.py", "repo_id": "r1"}
    ]
    res_hier = knowledge_get_hierarchy(symbol_id=10, db=mock_db)
    assert len(res_hier["hierarchy"]) == 1

    # 5. knowledge_impact_analysis
    mock_db.get_impact_analysis.return_value = [
        {
            "symbol": {"id": 14, "name": "Imp", "qualified_name": "app.Imp", "kind": "func", "path": "imp.py", "repo_id": "r1"},
            "edge_kind": "calls",
            "depth": 1
        }
    ]
    res_impact = knowledge_impact_analysis(symbol_id=10, db=mock_db)
    assert res_impact["total_impacted"] == 1


def test_search_tools_sync_functions(tmp_path):
    """Проверка синхронных функций поиска и чтения файлов."""
    from knowledge_mcp.interfaces.mcp.tools.search_tools import (
        knowledge_search,
        knowledge_read_file,
    )
    mock_db = MagicMock()

    # 1. FTS Search
    mock_db.search_chunks_fts.return_value = [
        {"id": "c1", "path": "test.py", "repo_id": "r1"}
    ]
    res_fts = knowledge_search(query="test", db=mock_db)
    assert res_fts["total_results"] == 1

    # 2. Hybrid Search with query_vector
    mock_db.search_chunks_hybrid.return_value = [
        {"id": "c2", "path": "vec.py", "repo_id": "r1"}
    ]
    res_hybrid = knowledge_search(query="test", query_vector=[0.1, 0.2], db=mock_db)
    assert res_hybrid["total_results"] == 1

    # 3. read_file from disk
    sample_file = tmp_path / "hello.py"
    sample_file.write_text("print('hello world')", encoding="utf-8")
    content_disk = knowledge_read_file(
        path="hello.py",
        repo_id="r1",
        db=mock_db,
        find_repo_path_fn=lambda p_id, r_id, b: tmp_path
    )
    assert content_disk == "print('hello world')"

    # 4. read_file from CAS DB
    mock_db.get_file_content.return_value = {"content": "cas content"}
    content_cas = knowledge_read_file(
        path="not_on_disk.py",
        repo_id="r1",
        db=mock_db,
        find_repo_path_fn=lambda p_id, r_id, b: None
    )
    assert content_cas == "cas content"

