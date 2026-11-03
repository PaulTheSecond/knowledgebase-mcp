"""
Unit-тесты для инфраструктурного адаптера Git CLI:
knowledge_mcp/infrastructure/git/git_cli_adapter.py
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from knowledge_mcp.infrastructure.git.git_cli_adapter import (
    clear_blame_cache,
    is_ignored_diff_path,
    get_repo_changed_line_ranges,
    get_changed_line_ranges,
    get_repo_git_info,
    get_git_diff_with_renames,
    get_finding_origin,
)


def test_is_ignored_diff_path():
    """Проверка фильтрации служебных путей (.husky, .vscode, .bak и т.д.)."""
    assert is_ignored_diff_path(".husky/pre-commit") is True
    assert is_ignored_diff_path(".vscode/settings.json") is True
    assert is_ignored_diff_path(".idea/workspace.xml") is True
    assert is_ignored_diff_path("file.bak") is True
    assert is_ignored_diff_path("src/utils.orig") is True
    assert is_ignored_diff_path("src/components/Button.tsx") is False
    assert is_ignored_diff_path("Models/User.cs") is False


def test_get_repo_changed_line_ranges(tmp_path: Path):
    """Проверка парсинга диапазонов строк из вывода git diff -U0."""
    clear_blame_cache()
    diff_output = (
        "diff --git a/src/index.ts b/src/index.ts\n"
        "--- a/src/index.ts\n"
        "+++ b/src/index.ts\n"
        "@@ -10,0 +11,5 @@\n"
        "@@ -20,3 +26,1 @@\n"
        "diff --git a/src/single.ts b/src/single.ts\n"
        "--- a/src/single.ts\n"
        "+++ b/src/single.ts\n"
        "@@ -5 +5 @@\n"
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=diff_output)
        ranges = get_repo_changed_line_ranges(
            repo_path=tmp_path,
            base_ref="master",
            head_ref="HEAD",
            include_uncommitted=False
        )

    assert "src/index.ts" in ranges
    assert (11, 15) in ranges["src/index.ts"]
    assert (26, 26) in ranges["src/index.ts"]
    assert "src/single.ts" in ranges
    assert (5, 5) in ranges["src/single.ts"]


def test_get_changed_line_ranges_single_file(tmp_path: Path):
    """Проверка вычисления диапазонов для одиночного файла."""
    diff_output = "@@ -1,0 +1,10 @@\n"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=diff_output)
        res = get_changed_line_ranges(tmp_path, "file.py", compare_ref="master", include_uncommitted=False)
        assert res == [(1, 10)]


def test_get_repo_git_info(tmp_path: Path):
    """Проверка сбора информации о ветке, коммитах и merge-base."""
    def fake_subprocess_run(cmd, *args, **kwargs):
        cmd_str = " ".join(cmd)
        mock = MagicMock(returncode=0)
        if "rev-parse --verify" in cmd_str:
            mock.returncode = 0
            mock.stdout = "some_sha\n"
        elif "rev-parse" in cmd_str:
            mock.stdout = "1111222233334444\n"
        elif "merge-base" in cmd_str:
            mock.stdout = "aaaabbbbccccdddd\n"
        elif "log" in cmd_str:
            mock.stdout = "1111222233334444|11112222|Author|2026-01-01|Commit message\n"
        elif "status --porcelain" in cmd_str:
            mock.stdout = " M src/app.ts\n"
        else:
            mock.stdout = ""
        return mock

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        info = get_repo_git_info(tmp_path, branch="feature/my-task")

    assert info["head_commit"] == "1111222233334444"
    assert info["merge_base_commit"] == "aaaabbbbccccdddd"
    assert "1111222233334444" in info["branch_commit_shas"]
    assert info["has_uncommitted_changes"] is True


def test_get_git_diff_with_renames(tmp_path: Path):
    """Проверка формирования списка измененных и переименованных файлов."""
    diff_output = (
        "R100\told_name.ts\tnew_name.ts\n"
        "A\tadded_file.ts\n"
        "M\tmodified_file.ts\n"
        "D\tdeleted_file.ts\n"
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=diff_output)
        res = get_git_diff_with_renames(tmp_path, base_sha="master", head_sha="HEAD", include_uncommitted=False)

    res_by_path = {item["path"]: item for item in res}
    assert "new_name.ts" in res_by_path
    assert res_by_path["new_name.ts"]["change_type"] == "renamed"
    assert res_by_path["new_name.ts"]["old_path"] == "old_name.ts"

    assert "added_file.ts" in res_by_path
    assert res_by_path["added_file.ts"]["change_type"] == "added"

    assert "modified_file.ts" in res_by_path
    assert res_by_path["modified_file.ts"]["change_type"] == "modified"

    assert "deleted_file.ts" in res_by_path
    assert res_by_path["deleted_file.ts"]["change_type"] == "deleted"


def test_get_finding_origin(tmp_path: Path):
    """Проверка определения авторства строки по git blame."""
    clear_blame_cache()

    # 1. Uncommitted changes (hash 00000000...)
    porcelain_uncommitted = (
        "0000000000000000 1 1 1\n"
        "author Dev User\n"
        "summary Working copy change\n"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=porcelain_uncommitted)
        orig = get_finding_origin(tmp_path, "src/code.py", 10, branch_commit_shas=set())
        assert orig["origin"] == "branch_uncommitted"
        assert orig["commit"] == "uncommitted"

    # 2. Branch commit
    clear_blame_cache()
    porcelain_branch = (
        "abcdef1234567890 1 1 1\n"
        "author Branch Dev\n"
        "summary Add feature\n"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=porcelain_branch)
        orig2 = get_finding_origin(tmp_path, "src/code.py", 10, branch_commit_shas={"abcdef12"})
        assert orig2["origin"] == "branch"
        assert orig2["commit"] == "abcdef12"

    # 3. Master commit
    clear_blame_cache()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=porcelain_branch)
        orig3 = get_finding_origin(tmp_path, "src/code.py", 10, branch_commit_shas={"othercommit"})
        assert orig3["origin"] == "master"
