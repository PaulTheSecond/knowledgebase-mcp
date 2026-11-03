"""
Unit-тесты для доменных сервисов review:
- knowledge_mcp/domain/review/rules.py (Static Checklist Audit)
- knowledge_mcp/domain/review/gap_analyzer.py (Test Gaps & Reviewed Files Audit)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from knowledge_mcp.domain.review.rules import audit_checklist_rules
from knowledge_mcp.domain.review.gap_analyzer import (
    filter_valid_findings,
    build_reviewed_files_audit,
    extract_call_modes_and_gaps
)


def test_rules_non_existent_repo():
    """Проверка возврата пустых списков, если путь к репозиторию не задан или не существует."""
    findings, gaps = audit_checklist_rules(
        repo_path=None,
        repo_id="test_repo",
        diff_files=[],
        branch_commit_shas=set(),
        compare_branch="master",
        base_sha=None,
        head_sha=None
    )
    assert findings == []
    assert gaps == []

    findings, gaps = audit_checklist_rules(
        repo_path=Path("/non/existent/path/for/sure"),
        repo_id="test_repo",
        diff_files=[],
        branch_commit_shas=set(),
        compare_branch="master",
        base_sha=None,
        head_sha=None
    )
    assert findings == []
    assert gaps == []


def test_rules_sql_checks(tmp_path: Path):
    """Проверка правил SQL: обнуление колонок и фильтрация JSON."""
    sql_dir = tmp_path / "migrations"
    sql_dir.mkdir(parents=True)
    sql_file = sql_dir / "V1__update_data.sql"
    sql_file.write_text(
        "UPDATE Users SET RoleId = NULL WHERE Active = 0;\n"
        "SELECT * FROM Configs WHERE j.[type] = 2 AND TRY_CAST(j.val AS INT) > 0;\n",
        encoding="utf-8"
    )

    diff_files = [
        {"path": "migrations/V1__update_data.sql", "change_type": "modified"}
    ]

    with patch("knowledge_mcp.domain.review.rules._get_finding_origin", return_value={"origin": "branch", "commit": "abc", "author": "dev"}):
        findings, gaps = audit_checklist_rules(
            repo_path=tmp_path,
            repo_id="sql_repo",
            diff_files=diff_files,
            branch_commit_shas={"abc"},
            compare_branch="master",
            base_sha="master",
            head_sha="HEAD"
        )

    # Должно быть 2 замечания по SQL
    rule_ids = [f["id"] for f in findings]
    assert any("RULE_SQL_DATA_LOSS" in rid for rid in rule_ids)
    assert any("RULE_SQL_JSON_TYPE" in rid for rid in rule_ids)

    # И 1 системный пробел (папка migrations)
    assert len(gaps) == 1
    assert gaps[0]["kind"] == "migration_script"


def test_rules_cs_deleted_test(tmp_path: Path):
    """Проверка обнаружения удаления unit-теста в C#."""
    cs_file = tmp_path / "UserServiceTests.cs"
    cs_file.write_text("// C# test file", encoding="utf-8")

    diff_files = [{"path": "UserServiceTests.cs", "change_type": "modified"}]

    mock_diff = (
        "--- a/UserServiceTests.cs\n"
        "+++ b/UserServiceTests.cs\n"
        "- [Fact]\n"
        "- public void LegacyAuth_Test()\n"
        "+ public void NewAuth_Test()\n"
    )

    with patch("subprocess.run") as mock_subproc, \
         patch("knowledge_mcp.domain.review.rules._get_finding_origin", return_value={"origin": "branch"}):
        mock_subproc.return_value = MagicMock(returncode=0, stdout=mock_diff)
        findings, _ = audit_checklist_rules(
            repo_path=tmp_path,
            repo_id="cs_repo",
            diff_files=diff_files,
            branch_commit_shas=set(),
            compare_branch="master",
            base_sha="master",
            head_sha="HEAD"
        )

    assert any("RULE_BE_DELETED_TEST" in f["id"] for f in findings)


def test_rules_frontend_ui_and_i18n(tmp_path: Path):
    """Проверка всех фронтенд правил: e2e assert, warnings, duplication, fallback, inversion, UI search, CSS, i18n, en locale."""
    fe_dir = tmp_path / "src" / "components"
    fe_dir.mkdir(parents=True)

    # 1. e2e test missing assert
    e2e_file = fe_dir / "user.spec.ts"
    e2e_file.write_text("const payload = { roleType: 'admin' };", encoding="utf-8")

    # 2. warnings.push & duplicate source & fallback mismatch & priority inversion & suboptimal search
    comp_file = fe_dir / "RoleSelector.tsx"
    comp_file.write_text(
        "warnings.push('error occurred');\n"
        "const filtered = users.filter(u => targetRoleIds.includes(u.roleId));\n"
        "const effective = allowedRoleIds.length ? allowedRoleIds : allRoles;\n"
        "const title = slot.name || APPROVER_ROLE_TITLE_KEYS[slot.id];\n"
        "if (roleSearchQuery.length >= 2) search(roleSearchQuery);\n",
        encoding="utf-8"
    )

    # 3. CSS defect
    css_file = fe_dir / "RoleSelector.styles.ts"
    css_file.write_text(
        "export const StyledBox = styled.div`\n"
        "  &:first-child {\n"
        "    margin: 0;\n"
        "  }\n"
        "  color: red;\n"
        "`;\n",
        encoding="utf-8"
    )

    # 4. English locale changed
    en_locale = tmp_path / "i18n" / "en" / "glossary.json"
    en_locale.parent.mkdir(parents=True)
    en_locale.write_text('{"ok": "OK"}', encoding="utf-8")

    diff_files = [
        {"path": "src/components/user.spec.ts", "change_type": "modified"},
        {"path": "src/components/RoleSelector.tsx", "change_type": "modified"},
        {"path": "src/components/RoleSelector.styles.ts", "change_type": "modified"},
        {"path": "i18n/en/glossary.json", "change_type": "modified"},
    ]

    diff_output = (
        "diff --git a/src/components/RoleSelector.tsx b/src/components/RoleSelector.tsx\n"
        "+ const label = \"Сохранить изменения\";\n"
    )

    with patch("subprocess.run") as mock_subproc, \
         patch("knowledge_mcp.domain.review.rules._get_finding_origin", return_value={"origin": "branch"}):
        mock_subproc.return_value = MagicMock(returncode=0, stdout=diff_output)

        findings, _ = audit_checklist_rules(
            repo_path=tmp_path,
            repo_id="fe_repo",
            diff_files=diff_files,
            branch_commit_shas=set(),
            compare_branch="master",
            base_sha="master",
            head_sha="HEAD"
        )

    f_ids = [f["id"] for f in findings]
    assert any("RULE_E2E_MISSING_ASSERT" in fid for fid in f_ids)
    assert any("RULE_FE_WARNINGS_SWALLOWED" in fid for fid in f_ids)
    assert any("RULE_FE_DUPLICATE_SOURCE" in fid for fid in f_ids)
    assert any("RULE_FE_FALLBACK_MISMATCH" in fid for fid in f_ids)
    assert any("RULE_FE_PRIORITY_INVERSION" in fid for fid in f_ids)
    assert any("RULE_FE_UI_SUBOPTIMAL" in fid for fid in f_ids)
    assert any("RULE_FE_CSS_DEFECT" in fid for fid in f_ids)
    assert any("RULE_FE_I18N_HARDCODE" in fid for fid in f_ids)
    assert any("RULE_FE_EN_LOCALE" in fid for fid in f_ids)


def test_gap_analyzer_filter_valid_findings():
    """Проверка фильтрации невалидных замечаний."""
    gaps = [
        {"location": "file.ts:10", "line_start": 10, "reason": "Обычная ошибка"},
        {"location": "", "line_start": 10, "reason": "Нет локации"},
        {"location": "file.ts:0", "line_start": 0, "reason": "Нулевая строка"},
        {"location": "file.ts:5", "line_start": 5, "reason": "Валидация пропсов в TypeScript избыточна"},
    ]
    filtered = filter_valid_findings(gaps)
    assert len(filtered) == 1
    assert filtered[0]["reason"] == "Обычная ошибка"


def test_gap_analyzer_build_reviewed_files_audit():
    """Проверка классификации отрецензированных файлов."""
    all_files_info = [
        {"path": "Form.Designer.cs", "repo_id": "r1", "change_type": "modified"},
        {"path": "OldFile.cs", "repo_id": "r1", "change_type": "deleted"},
        {"path": "Buggy.ts", "repo_id": "r1", "change_type": "modified"},
        {"path": "Uncovered.ts", "repo_id": "r1", "change_type": "modified"},
        {"path": "Renamed.ts", "repo_id": "r1", "change_type": "renamed", "old_path": "Orig.ts", "similarity": 95},
        {"path": "Component.test.ts", "repo_id": "r1", "change_type": "modified"},
        {"path": "Clean.ts", "repo_id": "r1", "change_type": "modified"},
    ]

    code_findings = [{"file": "Buggy.ts", "reason": "Дефект логики"}]
    test_gaps = [{"file": "Uncovered.ts", "reason": "Нет покрытия"}]

    audit = build_reviewed_files_audit(all_files_info, code_findings, test_gaps, [])
    assert len(audit) == 7

    status_map = {item["file"]: item for item in audit}
    assert status_map["Form.Designer.cs"]["status"] == "clean"
    assert "Designer" in status_map["Form.Designer.cs"]["status_detail"]

    assert status_map["OldFile.cs"]["status"] == "clean"
    assert status_map["OldFile.cs"]["change_type"] == "deleted"

    assert status_map["Buggy.ts"]["status"] == "has_findings"
    assert status_map["Buggy.ts"]["findings_count"] == 1

    assert status_map["Uncovered.ts"]["status"] == "clean"
    assert status_map["Uncovered.ts"]["test_gaps_count"] == 1

    assert status_map["Renamed.ts"]["status"] == "clean"
    assert "Переименован" in status_map["Renamed.ts"]["status_detail"]

    assert status_map["Component.test.ts"]["status"] == "clean"
    assert "Тестовый файл" in status_map["Component.test.ts"]["status_detail"]

    assert status_map["Clean.ts"]["status"] == "clean"


def test_extract_call_modes_and_gaps_with_tsx(tmp_path: Path):
    """Проверка извлечения кастомных пропсов (режимов вызова) из JSX/TSX."""
    from knowledge_mcp.core.code_parser import CodeParser
    parser = CodeParser()

    consumer_file = tmp_path / "AppConsumer.tsx"
    consumer_file.write_text(
        "import React from 'react';\n"
        "export const App = () => (\n"
        "  <div>\n"
        "    <RoleSelector isMultiSelect={true} customVariant=\"compact\" disabled={false} />\n"
        "  </div>\n"
        ");\n",
        encoding="utf-8"
    )

    sym = {
        "id": 101,
        "name": "RoleSelector",
        "qualified_name": "components.RoleSelector",
        "path": "src/RoleSelector.tsx",
        "line_start": 5,
        "line_end": 45
    }

    gaps = extract_call_modes_and_gaps(
        sym=sym,
        component_consumer_files=["AppConsumer.tsx"],
        repo_path=tmp_path,
        tested_symbol_ids=set(),
        feature_test_keywords={"e2e", "spec"},
        related_tests=[],
        code_parser=parser
    )

    modes = [g["mode"] for g in gaps]
    # 'disabled' - в STANDARD_PROPS, а 'isMultiSelect' и 'customVariant' - кастомные
    assert "isMultiSelect" in modes or "customVariant" in modes
    assert all(g["kind"] == "feature_mode" for g in gaps)

