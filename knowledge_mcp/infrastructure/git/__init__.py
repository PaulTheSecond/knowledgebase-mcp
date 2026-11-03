"""Инфраструктурный пакет адаптеров Git."""

from .git_cli_adapter import (
    GitCliAdapter,
    is_ignored_diff_path,
    get_repo_changed_line_ranges,
    get_changed_line_ranges,
    get_repo_git_info,
    get_git_diff_with_renames,
    get_finding_origin,
    clear_blame_cache,
)

__all__ = [
    "GitCliAdapter",
    "is_ignored_diff_path",
    "get_repo_changed_line_ranges",
    "get_changed_line_ranges",
    "get_repo_git_info",
    "get_git_diff_with_renames",
    "get_finding_origin",
    "clear_blame_cache",
]
