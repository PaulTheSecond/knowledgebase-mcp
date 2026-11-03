"""
Сквозной характеризационный тест для review context:
knowledge_review_context в knowledge_mcp.mcp.server.
Фиксирует итоговую структуру JSON-контракта Golden Master перед рефакторингом.
"""

import subprocess
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.mcp.server import knowledge_review_context


def _create_sample_git_repo(repo_dir: Path) -> tuple[str, str]:
    """Создает тестовый git репозиторий с master и feature ветками."""
    def run_git(*args):
        res = subprocess.run(
            ["git", "-C", str(repo_dir)] + list(args),
            capture_output=True, text=True, encoding="utf-8", check=True
        )
        return res.stdout.strip()

    run_git("init")
    run_git("config", "user.email", "author@example.com")
    run_git("config", "user.name", "Author")

    # В master создаем базовый файл
    (repo_dir / "Calculator.cs").write_text(
        "public class Calculator {\n    public int Add(int a, int b) => a + b;\n}\n",
        encoding="utf-8"
    )
    run_git("add", "Calculator.cs")
    run_git("commit", "-m", "Initial calculator in master")
    base_sha = run_git("rev-parse", "HEAD")

    # В feature добавляем новый метод
    run_git("checkout", "-b", "feature/calc-div")
    (repo_dir / "Calculator.cs").write_text(
        "public class Calculator {\n    public int Add(int a, int b) => a + b;\n    public int Div(int a, int b) => a / b;\n}\n",
        encoding="utf-8"
    )
    run_git("add", "Calculator.cs")
    run_git("commit", "-m", "Add Div method")
    head_sha = run_git("rev-parse", "HEAD")

    return base_sha, head_sha


class TestCharacterizationReviewContextE2E:

    @pytest.fixture
    def test_env(self):
        """Создает изолированную БД и тестовый Git-репозиторий."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            db_path = tmp_path / "test_review.db"
            db = KnowledgeDB(db_path, read_only=False)
            repo_path = tmp_path / "calc_repo"
            repo_path.mkdir()
            base_sha, head_sha = _create_sample_git_repo(repo_path)

            yield {
                "db": db,
                "repo_path": repo_path,
                "base_sha": base_sha,
                "head_sha": head_sha
            }
            db.close()

    def test_review_context_contract_and_structure(self, test_env):
        """Проверяет полноту и неизменность ключей JSON-контракта knowledge_review_context."""
        db = test_env["db"]
        repo_path = test_env["repo_path"]

        # Мокаем резолв пути к репозиторию, чтобы server нашел наш временный тестовый репозиторий
        with patch("knowledge_mcp.mcp.server._find_repo_path", return_value=repo_path):
            result = knowledge_review_context(
                repo_id="calc_repo",
                project_id="default",
                branch="feature/calc-div",
                compare_branch="master",
                db=db
            )

        # 1. Проверяем точный набор ключей верхнего уровня контракта
        expected_keys = {
            "header",
            "reviewed_files",
            "findings",
            "repo_id",
            "repo_ids",
            "project_id",
            "branch",
            "compare_branch",
            "changed_files",
            "changed_symbols",
            "blast_radius",
            "related_tests",
            "test_gaps",
            "e2e_covered_symbols",
            "risk",
            "risk_level",
            "risk_score",
            "risk_breakdown",
            "context_savings"
        }
        assert expected_keys.issubset(result.keys()), f"Missing keys: {expected_keys - set(result.keys())}"

        # 2. Проверяем структуру header
        header = result["header"]
        assert "repositories" in header
        assert "reviewed_files_summary" in header
        summary = header["reviewed_files_summary"]
        assert summary["total_files"] == 1
        assert summary["clean_files"] == 1
        assert summary["files_with_findings"] == 0

        # 3. Проверяем reviewed_files
        reviewed_files = result["reviewed_files"]
        assert len(reviewed_files) == 1
        assert reviewed_files[0]["file"] == "Calculator.cs"
        assert reviewed_files[0]["repo_id"] == "calc_repo"
        assert reviewed_files[0]["change_type"] == "modified"

        # 4. Проверяем findings (чек-лист)
        assert isinstance(result["findings"], list)

        # 5. Проверяем risk_score и risk_level
        assert isinstance(result["risk_score"], int)
        assert result["risk_level"] in ("low", "medium", "high")
        components = result["risk_breakdown"]["components"]
        assert "complexity" in components
        assert "blast_radius" in components
        assert "test_gaps" in components
        assert "mitigations" in components

        # 6. Проверяем context_savings
        assert "baseline_tokens" in result["context_savings"]
        assert "returned_tokens" in result["context_savings"]
