"""
Модульные тесты для расширенных MCP инструментов v1.8:
- knowledge_search (project_id, branch, JIT, context_savings)
- knowledge_review_context (blast radius, related tests, test gaps, risk score, context_savings)
- knowledge_find_tests (TESTED_BY, naming patterns, file matching)
- knowledge_get_project_rules (CLAUDE.md / .kb/ rules, topic filtering, token savings)
- token_savings_log телеметрия
- knowledge_read_file с multi-worktree поддержкой
"""

import json
import tempfile
from pathlib import Path
import pytest

from knowledge_mcp.db.models import KnowledgeDB
import knowledge_mcp.mcp.server as mcp_server
from knowledge_mcp.mcp.server import (
    knowledge_search,
    knowledge_find_symbol,
    knowledge_get_callers,
    knowledge_get_callees,
    knowledge_get_hierarchy,
    knowledge_impact_analysis,
    knowledge_read_file,
    knowledge_review_context,
    knowledge_find_tests,
    knowledge_get_project_rules,
    call_tool,
    set_db,
    _TOOL_SCHEMAS,
    _TOOL_HANDLERS
)


@pytest.fixture
def mcp_test_env():
    """Создает изолированную тестовую БД и настраивает MCP-сервер."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_mcp_v18.db"
        db = KnowledgeDB(db_path, read_only=False)
        set_db(db)
        # Отключаем использование тяжелых ML-эмбеддингов для быстрых модульных тестов
        mcp_server.use_embeddings = False
        yield db, db_path
        db.close()


def _populate_test_codebase(db: KnowledgeDB):
    """Наполняет БД тестовым кодом с символами, ребрами и правилами для воркспейса orbita."""
    # 1. Добавляем контент файлов в CAS
    order_svc_content = """
    public class OrderService
    {
        public Order ProcessOrder(int orderId)
        {
            var total = CalculateTotal(orderId);
            return new Order(orderId, total);
        }

        public decimal CalculateTotal(int orderId)
        {
            return 150.0m;
        }
    }
    """
    order_ctrl_content = """
    public class OrderController
    {
        private readonly OrderService _service;
        public OrderController(OrderService service) { _service = service; }

        public IActionResult CreateOrder(int id)
        {
            var res = _service.ProcessOrder(id);
            return Ok(res);
        }
    }
    """
    order_tests_content = """
    public class OrderServiceTests
    {
        [Fact]
        public void Test_ProcessOrder()
        {
            var svc = new OrderService();
            var res = svc.ProcessOrder(42);
            Assert.NotNull(res);
        }
    }
    """
    claude_md_content = """
    # CLAUDE.md Rules
    ## Database Guidelines
    Always use EF Core migrations and transactional operations for schema modifications.
    Do not run raw destructive DDL without backup verification.

    ## Testing Architecture
    All public methods must be accompanied by unit tests. Use xUnit with [Fact] and [Theory].
    Maintain TESTED_BY edge coverage for critical services.

    ## Code Review & CRLF Git Standards
    Ensure Windows CRLF line endings are strictly preserved across git repositories.
    Every pull request must pass automated review context analysis before merge.
    """

    h_svc = db.upsert_file_content(order_svc_content, language="csharp")
    h_ctrl = db.upsert_file_content(order_ctrl_content, language="csharp")
    h_test = db.upsert_file_content(order_tests_content, language="csharp")
    h_claude = db.upsert_file_content(claude_md_content, language="markdown")

    # 2. Регистрируем файлы в ветке orbit-1035 проекта orbita
    f_svc = db.upsert_file(
        repo_id="ImpactOS.Core",
        path="Services/OrderService.cs",
        mtime=100.0,
        file_hash=h_svc,
        project_id="orbita",
        branch="orbit-1035",
        content_hash=h_svc
    )

    f_ctrl = db.upsert_file(
        repo_id="ImpactOS.Core",
        path="Controllers/OrderController.cs",
        mtime=100.0,
        file_hash=h_ctrl,
        project_id="orbita",
        branch="orbit-1035",
        content_hash=h_ctrl
    )

    f_test = db.upsert_file(
        repo_id="ImpactOS.Core",
        path="Tests/OrderServiceTests.cs",
        mtime=100.0,
        file_hash=h_test,
        project_id="orbita",
        branch="orbit-1035",
        content_hash=h_test
    )

    f_claude = db.upsert_file(
        repo_id="docs",
        path="CLAUDE.md",
        mtime=100.0,
        file_hash=h_claude,
        project_id="orbita",
        branch="master",
        content_hash=h_claude
    )

    # 3. Регистрируем чанки
    c1 = db.add_chunk("c_svc_1", f_svc, order_svc_content, "code", "verified", 1, 15, h_svc, h_svc)
    c2 = db.add_chunk("c_ctrl_1", f_ctrl, order_ctrl_content, "code", "verified", 1, 15, h_ctrl, h_ctrl)
    c3 = db.add_chunk("c_test_1", f_test, order_tests_content, "code", "verified", 1, 15, h_test, h_test)
    
    # Чанки правил проекта (source_kind='rules')
    r1 = db.add_chunk("c_rule_db", f_claude, "## Database Guidelines\nAlways use EF Core migrations and transactional operations.", "rules", "verified", 2, 5, h_claude, h_claude)
    r2 = db.add_chunk("c_rule_test", f_claude, "## Testing Architecture\nAll public methods must be accompanied by unit tests with [Fact].", "rules", "verified", 6, 9, h_claude, h_claude)
    r3 = db.add_chunk("c_rule_crlf", f_claude, "## Code Review & CRLF Git Standards\nEnsure Windows CRLF line endings are strictly preserved across git repositories.", "rules", "verified", 10, 14, h_claude, h_claude)

    # 4. Регистрируем символы
    sym_records = [
        # file_id, name, qname, kind, lang, ls, le, sig, chunk_id, content_hash
        (f_svc, "OrderService", "ImpactOS.Services.OrderService", "class", "csharp", 2, 14, "public class OrderService", "c_svc_1", h_svc),
        (f_svc, "ProcessOrder", "ImpactOS.Services.OrderService.ProcessOrder", "method", "csharp", 4, 8, "public Order ProcessOrder(int orderId)", "c_svc_1", h_svc),
        (f_svc, "CalculateTotal", "ImpactOS.Services.OrderService.CalculateTotal", "method", "csharp", 10, 13, "public decimal CalculateTotal(int orderId)", "c_svc_1", h_svc),
        (f_ctrl, "OrderController", "ImpactOS.Controllers.OrderController", "class", "csharp", 2, 13, "public class OrderController", "c_ctrl_1", h_ctrl),
        (f_ctrl, "CreateOrder", "ImpactOS.Controllers.OrderController.CreateOrder", "method", "csharp", 7, 11, "public IActionResult CreateOrder(int id)", "c_ctrl_1", h_ctrl),
        (f_test, "OrderServiceTests", "ImpactOS.Tests.OrderServiceTests", "class", "csharp", 2, 12, "public class OrderServiceTests", "c_test_1", h_test),
        (f_test, "Test_ProcessOrder", "ImpactOS.Tests.OrderServiceTests.Test_ProcessOrder", "method", "csharp", 4, 11, "public void Test_ProcessOrder()", "c_test_1", h_test),
    ]
    sym_ids = db.add_symbols_batch(sym_records)
    # sym_ids: [OrderService, ProcessOrder, CalculateTotal, OrderController, CreateOrder, OrderServiceTests, Test_ProcessOrder]
    id_svc_class, id_process_order, id_calc_total, id_ctrl_class, id_create_order, id_test_class, id_test_method = sym_ids

    # 5. Графовые ребра:
    # - ProcessOrder вызывает CalculateTotal
    # - CreateOrder вызывает ProcessOrder
    # - ProcessOrder TESTED_BY Test_ProcessOrder
    # При этом CalculateTotal НЕ имеет TESTED_BY связи (Test Gap!)
    edges = [
        (id_process_order, id_calc_total, "CALLS"),
        (id_create_order, id_process_order, "CALLS"),
        (id_process_order, id_test_method, "TESTED_BY"),
    ]
    db.add_symbol_edges_batch(edges)

    return {
        "id_process_order": id_process_order,
        "id_calc_total": id_calc_total,
        "id_create_order": id_create_order,
        "id_test_method": id_test_method,
        "f_svc": f_svc
    }


class TestMCPV18Tools:
    """Модульные тесты для расширений MCP v1.8."""

    def test_tool_schemas_and_handlers_registration(self):
        """Проверяет регистрацию всех новых инструментов в _TOOL_SCHEMAS и _TOOL_HANDLERS."""
        schema_names = {t.name for t in _TOOL_SCHEMAS}
        assert "knowledge_search" in schema_names
        assert "knowledge_review_context" in schema_names
        assert "knowledge_find_tests" in schema_names
        assert "knowledge_get_project_rules" in schema_names
        assert "knowledge_find_symbol" in schema_names
        assert "knowledge_get_callers" in schema_names

        assert "knowledge_review_context" in _TOOL_HANDLERS
        assert "knowledge_find_tests" in _TOOL_HANDLERS
        assert "knowledge_get_project_rules" in _TOOL_HANDLERS

    def test_knowledge_search_with_project_and_branch(self, mcp_test_env):
        """Тест knowledge_search с project_id, branch и проверкой метрики context_savings."""
        db, _ = mcp_test_env
        _populate_test_codebase(db)

        # 1. Вызов Python функции
        res = knowledge_search(
            query="CalculateTotal",
            project_id="orbita",
            branch="orbit-1035",
            db=db
        )
        assert res["total_results"] > 0
        assert "context_savings" in res
        savings = res["context_savings"]
        assert savings["saved_tokens"] > 0
        assert savings["saved_percent"] > 0
        assert savings["baseline_tokens"] > savings["returned_tokens"]

        # 2. Вызов через MCP протокол
        import asyncio
        mcp_res = asyncio.run(call_tool("knowledge_search", {
            "query": "CalculateTotal",
            "project_id": "orbita",
            "branch": "orbit-1035"
        }))
        assert len(mcp_res) == 1
        text = mcp_res[0].text
        assert "context_savings" in text
        assert "OrderService.cs" in text

    def test_knowledge_review_context_blast_radius_and_test_gaps(self, mcp_test_env):
        """Тест knowledge_review_context: расчет blast radius, определение test gaps и risk score."""
        db, _ = mcp_test_env
        meta = _populate_test_codebase(db)

        # Анализируем контекст для измененного файла Services/OrderService.cs
        res = knowledge_review_context(
            repo_id="ImpactOS.Core",
            project_id="orbita",
            branch="orbit-1035",
            compare_branch="master",
            changed_files=["Services/OrderService.cs"],
            max_depth=2,
            db=db
        )

        # 1. Проверяем измененные файлы и символы
        assert res["changed_files"] == ["Services/OrderService.cs"]
        changed_sym_names = {s["name"] for s in res["changed_symbols"]}
        assert "ProcessOrder" in changed_sym_names
        assert "CalculateTotal" in changed_sym_names

        # 2. Проверяем радиус поражения (Blast Radius): CreateOrder вызывает ProcessOrder
        blast = res["blast_radius"]
        assert blast["impacted_symbols_count"] >= 1
        caller_names = {c["name"] for c in blast["callers"]}
        assert "CreateOrder" in caller_names
        assert "Controllers/OrderController.cs" in blast["impacted_files"]

        # 3. Проверяем связанные тесты: ProcessOrder покрыт тестом Test_ProcessOrder
        rel_test_names = {t["test_name"] for t in res["related_tests"]}
        assert "Test_ProcessOrder" in rel_test_names

        # 4. Проверяем выявление Test Gaps: CalculateTotal НЕ имеет теста!
        gap_names = {g["name"] for g in res["test_gaps"]}
        assert "CalculateTotal" in gap_names
        assert "ProcessOrder" not in gap_names

        # 5. Проверяем скоринг риска
        assert res["risk"] in ("medium", "high")
        assert res["risk_score"] > 20

        # 6. Проверяем метрику экономии токенов
        savings = res["context_savings"]
        assert savings["saved_tokens"] > 0
        assert savings["saved_percent"] > 0

        # 7. Вызов через MCP протокол call_tool
        import asyncio
        mcp_res = asyncio.run(call_tool("knowledge_review_context", {
            "repo_id": "ImpactOS.Core",
            "project_id": "orbita",
            "branch": "orbit-1035",
            "changed_files": ["Services/OrderService.cs"]
        }))
        assert len(mcp_res) == 1
        data = json.loads(mcp_res[0].text)
        assert data["repo_id"] == "ImpactOS.Core"
        assert len(data["test_gaps"]) >= 1

    def test_knowledge_review_context_zero_changes(self, mcp_test_env):
        """Тест knowledge_review_context при отсутствии изменений: risk=low, score=0."""
        db, _ = mcp_test_env
        _populate_test_codebase(db)

        res = knowledge_review_context(
            repo_id="ImpactOS.Core",
            project_id="orbita",
            branch="orbit-1035",
            changed_files=[],
            db=db
        )
        assert res["risk"] == "low"
        assert res["risk_score"] == 0
        assert len(res["test_gaps"]) == 0
        assert len(res["changed_symbols"]) == 0

    def test_knowledge_find_tests(self, mcp_test_env):
        """Тест knowledge_find_tests: поиск связанных тестов по методу и классу."""
        db, _ = mcp_test_env
        _populate_test_codebase(db)

        # 1. Поиск тестов для метода ProcessOrder (связь TESTED_BY)
        res_method = knowledge_find_tests(
            target="ProcessOrder",
            repo_id="ImpactOS.Core",
            project_id="orbita",
            branch="orbit-1035",
            db=db
        )
        assert res_method["total_tests"] >= 1
        test_names = {t["name"] for t in res_method["tests"]}
        assert "Test_ProcessOrder" in test_names
        assert res_method["context_savings"]["saved_tokens"] > 0

        # 2. Поиск тестов для класса OrderService (эвристика OrderServiceTests)
        res_class = knowledge_find_tests(
            target="OrderService",
            repo_id="ImpactOS.Core",
            project_id="orbita",
            branch="orbit-1035",
            db=db
        )
        assert res_class["total_tests"] >= 1
        class_test_names = {t["name"] for t in res_class["tests"]}
        assert any("OrderServiceTests" in n or "Test_ProcessOrder" in n for n in class_test_names)

        # 3. Вызов через MCP протокол
        import asyncio
        mcp_res = asyncio.run(call_tool("knowledge_find_tests", {
            "target": "ProcessOrder",
            "repo_id": "ImpactOS.Core",
            "project_id": "orbita",
            "branch": "orbit-1035"
        }))
        assert len(mcp_res) == 1
        data = json.loads(mcp_res[0].text)
        assert data["total_tests"] >= 1

    def test_knowledge_get_project_rules(self, mcp_test_env):
        """Тест knowledge_get_project_rules: получение правил целиком и с фильтром по теме."""
        db, _ = mcp_test_env
        _populate_test_codebase(db)

        # 1. Получение всех правил (без topic)
        all_rules_res = knowledge_get_project_rules(
            topic=None,
            project_id="orbita",
            db=db
        )
        assert all_rules_res["total_rules"] == 3
        # Должна быть зафиксирована большая экономия токенов по сравнению с чтением 76 КБ файла
        assert all_rules_res["context_savings"]["baseline_tokens"] >= 19456
        assert all_rules_res["context_savings"]["saved_tokens"] > 18000
        assert all_rules_res["context_savings"]["saved_percent"] >= 90

        # 2. Фильтр по теме: "database"
        db_rules = knowledge_get_project_rules(
            topic="database",
            project_id="orbita",
            db=db
        )
        assert db_rules["total_rules"] == 1
        assert "Database Guidelines" in db_rules["rules"][0]["content"]

        # 3. Фильтр по теме: "crlf"
        crlf_rules = knowledge_get_project_rules(
            topic="crlf",
            project_id="orbita",
            db=db
        )
        assert crlf_rules["total_rules"] == 1
        assert "CRLF" in crlf_rules["rules"][0]["content"]

        # 4. Вызов через MCP протокол
        import asyncio
        mcp_res = asyncio.run(call_tool("knowledge_get_project_rules", {
            "topic": "testing",
            "project_id": "orbita"
        }))
        assert len(mcp_res) == 1
        data = json.loads(mcp_res[0].text)
        assert data["total_rules"] == 1
        assert "Testing" in data["rules"][0]["content"]

    def test_token_savings_logging(self, mcp_test_env):
        """Тест записи метрик экономии токенов в token_savings_log и их агрегации."""
        db, _ = mcp_test_env
        _populate_test_codebase(db)

        # Вызываем несколько инструментов
        knowledge_search("CalculateTotal", project_id="orbita", branch="orbit-1035", db=db)
        knowledge_review_context("ImpactOS.Core", project_id="orbita", branch="orbit-1035", changed_files=["Services/OrderService.cs"], db=db)
        knowledge_find_tests("ProcessOrder", project_id="orbita", branch="orbit-1035", db=db)
        knowledge_get_project_rules(topic="database", project_id="orbita", db=db)

        # Проверяем записи в базе данных
        summary = db.get_token_savings_summary(days=7, project_id="orbita")
        assert summary["total_calls"] >= 4
        assert summary["total_saved"] > 0
        assert summary["overall_saved_percent"] > 0

        breakdown = summary["tools_breakdown"]
        assert "knowledge_search" in breakdown
        assert "knowledge_review_context" in breakdown
        assert "knowledge_find_tests" in breakdown
        assert "knowledge_get_project_rules" in breakdown
        assert breakdown["knowledge_get_project_rules"]["saved_tokens"] > 10000

    def test_knowledge_read_file(self, mcp_test_env):
        """Тест чтения файла с учетом project_id и branch."""
        db, _ = mcp_test_env
        _populate_test_codebase(db)

        # Читаем существующий файл
        content = knowledge_read_file(
            path="Services/OrderService.cs",
            repo_id="ImpactOS.Core",
            project_id="orbita",
            branch="orbit-1035",
            db=db
        )
        assert content is not None
        assert "class OrderService" in content

        # Попытка чтения несуществующего файла
        missing = knowledge_read_file(
            path="NonExistent/File.cs",
            repo_id="ImpactOS.Core",
            project_id="orbita",
            branch="orbit-1035",
            db=db
        )
        assert missing is None

    def test_graph_navigation_tools_with_project_and_branch(self, mcp_test_env):
        """Тест graph navigation инструментов: callers, callees, find_symbol, impact_analysis."""
        db, _ = mcp_test_env
        meta = _populate_test_codebase(db)
        id_process_order = meta["id_process_order"]

        # 1. find_symbol
        sym_res = knowledge_find_symbol(
            name_pattern="ProcessOrder",
            project_id="orbita",
            branch="orbit-1035",
            db=db
        )
        assert sym_res["total_symbols"] >= 1
        assert sym_res["context_savings"]["saved_tokens"] > 0

        # 2. get_callers
        callers_res = knowledge_get_callers(
            symbol_id=id_process_order,
            project_id="orbita",
            branch="orbit-1035",
            db=db
        )
        assert callers_res["total_callers"] >= 1
        assert any(c["name"] == "CreateOrder" for c in callers_res["callers"])
        assert callers_res["context_savings"]["saved_tokens"] > 0

        # 3. get_callees
        callees_res = knowledge_get_callees(
            symbol_id=id_process_order,
            project_id="orbita",
            branch="orbit-1035",
            db=db
        )
        assert callees_res["total_callees"] >= 1
        assert any(c["name"] == "CalculateTotal" for c in callees_res["callees"])

        # 4. impact_analysis
        impact_res = knowledge_impact_analysis(
            symbol_id=meta["id_calc_total"],
            max_depth=3,
            project_id="orbita",
            branch="orbit-1035",
            db=db
        )
        assert impact_res["total_impacted"] >= 1
        impact_names = {i["name"] for i in impact_res["impacted"]}
        assert "ProcessOrder" in impact_names
