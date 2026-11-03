"""
Характеризационные тесты для графа вызовов и анализа влияния:
get_impact_analysis, get_callers, get_callees в knowledge_mcp.db.models.KnowledgeDB.
Фиксирует текущее поведение как Golden Master перед рефакторингом.
"""

import tempfile
from pathlib import Path
import pytest

from knowledge_mcp.db.models import KnowledgeDB


class TestCharacterizationImpactGraph:

    @pytest.fixture
    def test_db(self):
        """Создает изолированную базу данных с миграциями."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_knowledge.db"
            db = KnowledgeDB(db_path, read_only=False)
            yield db
            db.close()

    def test_linear_impact_analysis_respects_max_depth(self, test_db):
        """
        Проверяет обход графа вызовов с разной глубиной max_depth:
        Repo.Insert -> calls -> Service.Create -> calls -> Controller.Post
        """
        # 1. Создаем файлы и чанки
        file_id = test_db.upsert_file(
            repo_id="backend",
            path="src/Order.cs",
            mtime=1000.0,
            file_hash="hash1"
        )
        test_db.add_chunk("chunk_1", file_id, "content 1", "code", "verified", 10, 20)
        test_db.add_chunk("chunk_2", file_id, "content 2", "code", "verified", 30, 40)
        test_db.add_chunk("chunk_3", file_id, "content 3", "code", "verified", 50, 60)

        # 2. Создаем символы
        repo_id = test_db.add_symbol(
            file_id=file_id,
            name="Insert",
            qualified_name="OrderRepository.Insert",
            kind="method",
            language="csharp",
            line_start=10,
            line_end=20,
            signature="void Insert()",
            chunk_id="chunk_1"
        )
        service_id = test_db.add_symbol(
            file_id=file_id,
            name="Create",
            qualified_name="OrderService.Create",
            kind="method",
            language="csharp",
            line_start=30,
            line_end=40,
            signature="void Create()",
            chunk_id="chunk_2"
        )
        controller_id = test_db.add_symbol(
            file_id=file_id,
            name="Post",
            qualified_name="OrderController.Post",
            kind="method",
            language="csharp",
            line_start=50,
            line_end=60,
            signature="void Post()",
            chunk_id="chunk_3"
        )

        # 3. Создаем связи вызовов (Controller -> Service -> Repo)
        test_db.add_symbol_edge(source_id=controller_id, target_id=service_id, kind="CALLS")
        test_db.add_symbol_edge(source_id=service_id, target_id=repo_id, kind="CALLS")

        # 4. Анализ влияния для Repo (кто пострадает при изменении Repo):
        # При max_depth=1: должен вернуться только Service с depth=1
        depth1_impact = test_db.get_impact_analysis(symbol_id=repo_id, max_depth=1)
        depth1_ids = [item["symbol"]["id"] for item in depth1_impact]
        assert service_id in depth1_ids
        assert controller_id not in depth1_ids
        assert depth1_impact[0]["depth"] == 1
        assert depth1_impact[0]["edge_kind"] == "CALLS"

        # При max_depth=2: должны вернуться и Service (depth=1), и Controller (depth=2)
        depth2_impact = test_db.get_impact_analysis(symbol_id=repo_id, max_depth=2)
        depth2_ids = [item["symbol"]["id"] for item in depth2_impact]
        assert service_id in depth2_ids
        assert controller_id in depth2_ids
        depths = {item["symbol"]["id"]: item["depth"] for item in depth2_impact}
        assert depths[service_id] == 1
        assert depths[controller_id] == 2

    def test_cyclic_calls_does_not_loop_infinitely(self, test_db):
        """Проверяет, что взаимно рекурсивные вызовы (A <-> B) не приводят к зацикливанию."""
        file_id = test_db.upsert_file(
            repo_id="backend",
            path="src/PingPong.cs",
            mtime=1000.0,
            file_hash="hash2"
        )
        test_db.add_chunk("chunk_p1", file_id, "ping", "code", "verified", 1, 5)
        test_db.add_chunk("chunk_p2", file_id, "pong", "code", "verified", 10, 15)

        ping_id = test_db.add_symbol(
            file_id=file_id,
            name="Ping",
            qualified_name="PingPong.Ping",
            kind="method",
            language="csharp",
            line_start=1,
            line_end=5,
            signature="void Ping()",
            chunk_id="chunk_p1"
        )
        pong_id = test_db.add_symbol(
            file_id=file_id,
            name="Pong",
            qualified_name="PingPong.Pong",
            kind="method",
            language="csharp",
            line_start=10,
            line_end=15,
            signature="void Pong()",
            chunk_id="chunk_p2"
        )

        # Создаем цикл вызовов
        test_db.add_symbol_edge(source_id=ping_id, target_id=pong_id, kind="CALLS")
        test_db.add_symbol_edge(source_id=pong_id, target_id=ping_id, kind="CALLS")

        # Анализ с большой глубиной
        impact = test_db.get_impact_analysis(symbol_id=ping_id, max_depth=10)
        impact_ids = [item["symbol"]["id"] for item in impact]
        assert pong_id in impact_ids
        # Узлы дедуплицированы (pong_id присутствует ровно 1 раз)
        assert impact_ids.count(pong_id) == 1
