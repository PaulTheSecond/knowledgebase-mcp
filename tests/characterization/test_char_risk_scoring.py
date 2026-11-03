"""
Характеризационные тесты для формулы скоринга рисков:
_calculate_transparent_risk_score в knowledge_mcp.mcp.server.
Фиксирует текущее поведение функции как Golden Master перед рефакторингом.
"""

import pytest
from knowledge_mcp.mcp.server import _calculate_transparent_risk_score


class TestCharacterizationRiskScoring:

    def test_empty_changes_returns_zero_score(self):
        """Пустые изменения должны возвращать score=0 и level='low'."""
        result = _calculate_transparent_risk_score(
            norm_changed_files=[],
            changed_symbols=[],
            unique_file_impacts={},
            component_consumer_files=[],
            style_only_files=[],
            type_only_files=[],
            test_gaps=[],
            e2e_covered_symbols=[],
            related_tests=[]
        )
        assert result["score"] == 0
        assert result["level"] == "low"
        assert result["coverage_percent"] == 100
        assert "formula" in result
        assert result["components"]["complexity"] == 0
        assert result["components"]["blast_radius"] == 0
        assert result["components"]["test_gaps"] == 0

    def test_clean_branch_with_good_coverage(self):
        """Небольшие изменения с хорошим тестовым покрытием дают уровень low."""
        changed_files = ["src/utils.py"]
        changed_symbols = [
            {"name": "format_date", "kind": "function"}
        ]
        related_tests = [
            {"name": "test_format_date", "kind": "function"}
        ]

        result = _calculate_transparent_risk_score(
            norm_changed_files=changed_files,
            changed_symbols=changed_symbols,
            unique_file_impacts={"src/main.py": 1},
            component_consumer_files=[],
            style_only_files=[],
            type_only_files=[],
            test_gaps=[],
            e2e_covered_symbols=[],
            related_tests=related_tests
        )
        assert result["score"] < 30
        assert result["level"] == "low"
        assert result["coverage_percent"] == 100
        assert result["components"]["mitigations"] > 0

    def test_test_gaps_floor_penalty(self):
        """При наличии test_gaps скор не может быть ниже 30 (минимум medium)."""
        result = _calculate_transparent_risk_score(
            norm_changed_files=["src/core.py"],
            changed_symbols=[{"name": "process_order", "kind": "method"}],
            unique_file_impacts={},
            component_consumer_files=[],
            style_only_files=[],
            type_only_files=[],
            test_gaps=[{"name": "process_order", "kind": "method"}],
            e2e_covered_symbols=[],
            related_tests=[]
        )
        assert result["score"] >= 30
        assert result["level"] in ("medium", "high")
        assert result["coverage_percent"] == 0

    def test_wide_blast_radius_elevates_to_high(self):
        """Широкий радиус поражения без тестов поднимает риск в диапазон high (>=70)."""
        changed_files = [f"src/file_{i}.py" for i in range(25)]
        logic_symbols = [{"name": f"method_{i}", "kind": "method"} for i in range(20)]
        consumers = [f"frontend/view_{i}.tsx" for i in range(15)]
        gaps = [{"name": f"method_{i}", "kind": "method"} for i in range(15)]

        result = _calculate_transparent_risk_score(
            norm_changed_files=changed_files,
            changed_symbols=logic_symbols,
            unique_file_impacts={f"impacted_{i}.py": 1 for i in range(20)},
            component_consumer_files=consumers,
            style_only_files=["style.css"],
            type_only_files=["types.ts"],
            test_gaps=gaps,
            e2e_covered_symbols=[],
            related_tests=[]
        )
        assert result["score"] >= 70
        assert result["level"] == "high"
        assert result["components"]["complexity"] == 22
        assert result["components"]["blast_radius"] == 21

    def test_score_boundary_invariants(self):
        """Инвариант границ: при любых экстремальных значениях score строго в [0, 100]."""
        changed_files = [f"file_{i}" for i in range(1000)]
        symbols = [{"name": f"sym_{i}", "kind": "function"} for i in range(1000)]
        gaps = [{"name": f"sym_{i}", "kind": "function"} for i in range(500)]

        result = _calculate_transparent_risk_score(
            norm_changed_files=changed_files,
            changed_symbols=symbols,
            unique_file_impacts={f"f_{i}": 1 for i in range(500)},
            component_consumer_files=[f"c_{i}" for i in range(200)],
            style_only_files=[],
            type_only_files=[],
            test_gaps=gaps,
            e2e_covered_symbols=[],
            related_tests=[]
        )
        assert 0 <= result["score"] <= 100
