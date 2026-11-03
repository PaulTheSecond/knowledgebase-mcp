"""
Характеризационные тесты для аудита универсальных чек-листов:
_audit_checklist_rules в knowledge_mcp.mcp.server.
Фиксирует текущее поведение функции как Golden Master перед рефакторингом.
"""

import subprocess
import tempfile
from pathlib import Path
import pytest

from knowledge_mcp.mcp.server import _audit_checklist_rules


def _init_git_repo(repo_dir: Path) -> tuple[str, str, set[str]]:
    """Инициализирует тестовый git репозиторий с master и feature коммитами."""
    def run_git(*args):
        res = subprocess.run(
            ["git", "-C", str(repo_dir)] + list(args),
            capture_output=True, text=True, encoding="utf-8", check=True
        )
        return res.stdout.strip()

    run_git("init")
    run_git("config", "user.email", "test@example.com")
    run_git("config", "user.name", "Test User")

    # Базовый файл
    (repo_dir / "README.md").write_text("# Test Repo", encoding="utf-8")
    run_git("add", "README.md")
    run_git("commit", "-m", "Initial commit")
    base_sha = run_git("rev-parse", "HEAD")

    # Создаем ветку и коммит с тестовыми файлами
    run_git("checkout", "-b", "feature")

    # 1. SQL с деструктивным UPDATE
    sql_file = repo_dir / "V1__update_data.sql"
    sql_file.write_text("UPDATE Users SET [RoleId] = NULL WHERE Active = 0;\n", encoding="utf-8")

    # 2. SQL миграция (проверка системного пробела)
    mig_file = repo_dir / "V2__migration_script.sql"
    mig_file.write_text("CREATE TABLE TestTable (Id INT);\n", encoding="utf-8")

    # 3. CS файл с удалением теста
    cs_file = repo_dir / "OrderServiceTests.cs"
    cs_file.write_text("// Existing tests\n", encoding="utf-8")

    run_git("add", ".")
    run_git("commit", "-m", "Add feature files")
    head_sha = run_git("rev-parse", "HEAD")

    branch_shas = {head_sha}
    return base_sha, head_sha, branch_shas


class TestCharacterizationChecklistRules:

    def test_audit_sql_data_loss_and_systemic_gaps(self):
        """Проверяет обнаружение деструктивного SQL и системных пробелов миграций."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_path = Path(tmp_dir)
            base_sha, head_sha, branch_shas = _init_git_repo(repo_path)

            diff_files = [
                {"path": "V1__update_data.sql", "change_type": "added"},
                {"path": "V2__migration_script.sql", "change_type": "added"}
            ]

            findings, gaps = _audit_checklist_rules(
                repo_path=repo_path,
                repo_id="test-backend",
                diff_files=diff_files,
                branch_commit_shas=branch_shas,
                compare_branch="master",
                base_sha=base_sha,
                head_sha=head_sha
            )

            # Проверка детекции SQL Data Loss
            sql_loss_findings = [f for f in findings if "SQL_DATA_LOSS" in f.get("id", "")]
            assert len(sql_loss_findings) == 1
            f = sql_loss_findings[0]
            assert f["file"] == "V1__update_data.sql"
            assert f["severity"] == "important"
            assert "деструктивное обновление данных" in f["message"]
            assert f["repo_id"] == "test-backend"

            # Проверка системного пробела миграций
            migration_gaps = [g for g in gaps if g.get("kind") == "migration_script"]
            assert len(migration_gaps) >= 1
            assert any("V2__migration_script.sql" in g["file"] for g in migration_gaps)

    def test_audit_deleted_test_in_cs(self):
        """Проверяет обнаружение удаленного unit-теста в C# файле."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_path = Path(tmp_dir)

            def run_git(*args):
                res = subprocess.run(
                    ["git", "-C", str(repo_path)] + list(args),
                    capture_output=True, text=True, encoding="utf-8", check=True
                )
                return res.stdout.strip()

            run_git("init")
            run_git("config", "user.email", "test@example.com")
            run_git("config", "user.name", "Test User")

            # В master добавляем тест
            test_file = repo_path / "UserServiceTests.cs"
            test_file.write_text("[Test]\npublic void Authenticate_ValidUser_ReturnsTokenTest()\n{\n}\n", encoding="utf-8")
            run_git("add", "UserServiceTests.cs")
            run_git("commit", "-m", "Add test in master")
            base_sha = run_git("rev-parse", "HEAD")

            # В feature удаляем тест
            run_git("checkout", "-b", "feature")
            test_file.write_text("// Test removed\n", encoding="utf-8")
            run_git("add", "UserServiceTests.cs")
            run_git("commit", "-m", "Remove test in feature")
            head_sha = run_git("rev-parse", "HEAD")

            diff_files = [
                {"path": "UserServiceTests.cs", "change_type": "modified"}
            ]

            findings, gaps = _audit_checklist_rules(
                repo_path=repo_path,
                repo_id="test-backend",
                diff_files=diff_files,
                branch_commit_shas={head_sha},
                compare_branch="master",
                base_sha=base_sha,
                head_sha=head_sha
            )

            del_test_findings = [f for f in findings if "RULE_BE_DELETED_TEST" in f.get("id", "")]
            assert len(del_test_findings) == 1
            f = del_test_findings[0]
            assert f["file"] == "UserServiceTests.cs"
            assert "Удален unit-тест" in f["message"]
            assert f["severity"] == "minor"
