import os
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from knowledge_mcp.core.config import (
    load_config,
    save_config,
    get_projects,
    get_project_config,
    add_or_update_project,
    delete_project_config,
    resolve_workspace_targets,
    get_repository_config,
)
from knowledge_mcp.core.workspace_scanner import (
    WorkspaceScanner,
    DiscoveredTarget,
)

@pytest.fixture
def temp_config_file(tmp_path):
    """Фикстура для изоляции CONFIG_PATH во временном каталоге."""
    cfg_file = tmp_path / "config.yaml"
    old_val = os.environ.get("CONFIG_PATH")
    os.environ["CONFIG_PATH"] = str(cfg_file)
    try:
        yield cfg_file
    finally:
        if old_val:
            os.environ["CONFIG_PATH"] = old_val
        else:
            os.environ.pop("CONFIG_PATH", None)


def test_project_config_crud(temp_config_file, tmp_path):
    """
    Тест CRUD операций над проектами в config.yaml:
    get_projects, add_or_update_project, get_project_config, delete_project_config.
    """
    # 1. Начальное состояние - проектов нет
    assert get_projects() == []
    assert get_project_config("orbita") is None

    project_data = {
        "id": "orbita",
        "name": "Orbita ATS",
        "workspace_root": str(tmp_path / "Orbita"),
        "type": "worktree_workspace",
        "base_path": "worktrees/master",
        "tasks_path": "worktrees/{task}",
        "docs_paths": [".kb", "CLAUDE.md"],
        "repositories": [
            {"id": "backend", "path": "backend", "language": "csharp"},
            {"id": "frontend", "path": "frontend", "language": "typescript"}
        ]
    }

    # 2. Добавление проекта
    add_or_update_project(project_data)
    projects = get_projects()
    assert len(projects) == 1
    assert projects[0]["id"] == "orbita"
    assert projects[0]["name"] == "Orbita ATS"

    # 3. Поиск по ID
    found = get_project_config("orbita")
    assert found is not None
    assert found["name"] == "Orbita ATS"
    assert len(found["repositories"]) == 2

    # 4. Обновление существующего проекта
    project_data["name"] = "Orbita ATS Enterprise"
    add_or_update_project(project_data)
    projects_updated = get_projects()
    assert len(projects_updated) == 1
    assert projects_updated[0]["name"] == "Orbita ATS Enterprise"

    # 5. Удаление проекта
    deleted = delete_project_config("orbita")
    assert deleted is True
    assert get_projects() == []
    assert get_project_config("orbita") is None

    # Повторное удаление возвращает False
    assert delete_project_config("orbita") is False


def test_backward_compatibility_repositories_only(temp_config_file):
    """
    Тест обратной совместимости: загрузка конфига только с секцией repositories:
    возвращает пустой список projects без ошибок.
    """
    legacy_cfg = {
        "server": {"port": 8000, "host": "127.0.0.1"},
        "repositories": [
            {
                "id": "legacy-repo",
                "url": "https://github.com/org/legacy-repo.git",
                "branch": "main",
                "local_path": "C:/Repos/legacy"
            }
        ]
    }
    with open(temp_config_file, "w", encoding="utf-8") as f:
        yaml.dump(legacy_cfg, f)

    # Проверяем, что конфиг загрузился корректно
    cfg = load_config()
    assert len(cfg["repositories"]) == 1
    assert cfg["repositories"][0]["id"] == "legacy-repo"

    # projects отсутствовал в файле, но get_projects() возвращает пустой список без исключений
    projects = get_projects()
    assert projects == []
    assert get_project_config("legacy-repo") is None
    assert get_repository_config("legacy-repo") is not None
    assert delete_project_config("legacy-repo") is False


def test_workspace_scanner_discovery_and_active_target(tmp_path):
    """
    Тест WorkspaceScanner:
    - Создает структуру воркспейса с worktrees/master/backend (.git),
      worktrees/orbit-1035/backend (.git), .kb/rules.md, CLAUDE.md.
    - Сканирует воркспейс и проверяет обнаружение базового клона, ворктри задач и документации.
    - Проверяет resolve_active_target для файла внутри worktrees/orbit-1035/backend/Services/OrderService.cs.
    """
    root = tmp_path / "Orbita"
    root.mkdir()

    # 1. Создаем базовый клон (worktrees/master/backend)
    master_backend = root / "worktrees" / "master" / "backend"
    master_backend.mkdir(parents=True)
    master_git = master_backend / ".git"
    master_git.mkdir()
    (master_git / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")

    # 2. Создаем ворктри задачи (worktrees/orbit-1035/backend)
    task_backend = root / "worktrees" / "orbit-1035" / "backend"
    task_backend.mkdir(parents=True)
    task_git = task_backend / ".git"
    task_git.mkdir()
    (task_git / "HEAD").write_text("ref: refs/heads/orbit-1035\n", encoding="utf-8")

    # Создаем рабочий файл C# внутри ворктри задачи
    order_service_file = task_backend / "Services" / "OrderService.cs"
    order_service_file.parent.mkdir(parents=True)
    order_service_file.write_text("namespace ImpactOS.Orders { public class OrderService {} }", encoding="utf-8")

    # 3. Создаем документацию (.kb/rules.md и CLAUDE.md)
    kb_dir = root / ".kb"
    kb_dir.mkdir()
    rules_file = kb_dir / "rules.md"
    rules_file.write_text("# Project Rules\nAll changes must be tested.", encoding="utf-8")

    claude_file = root / "CLAUDE.md"
    claude_file.write_text("# Claude Code Instructions\nCall knowledge_get_project_rules.", encoding="utf-8")

    project_config = {
        "id": "orbita",
        "name": "Orbita ATS",
        "workspace_root": str(root),
        "type": "worktree_workspace",
        "base_path": "worktrees/master",
        "tasks_path": "worktrees/{task}",
        "docs_paths": [".kb", "CLAUDE.md"],
        "repositories": [
            {"id": "backend", "path": "backend", "language": "csharp"}
        ]
    }

    # Сканирование воркспейса
    scanner = WorkspaceScanner()
    targets = scanner.scan_workspace(project_config)

    # Валидация базового клона
    base_targets = [t for t in targets if t.is_base and not t.is_doc]
    assert len(base_targets) == 1
    base_target = base_targets[0]
    assert base_target.project_id == "orbita"
    assert base_target.repo_id == "backend"
    assert base_target.branch == "master"
    assert base_target.task_name == "master"
    assert base_target.language == "csharp"
    assert base_target.is_base is True
    assert base_target.is_doc is False
    assert base_target.absolute_path == master_backend.resolve()

    # Валидация ворктри задачи
    task_targets = [t for t in targets if not t.is_base and not t.is_doc]
    assert len(task_targets) == 1
    task_target = task_targets[0]
    assert task_target.project_id == "orbita"
    assert task_target.repo_id == "backend"
    assert task_target.branch == "orbit-1035"
    assert task_target.task_name == "orbit-1035"
    assert task_target.language == "csharp"
    assert task_target.is_base is False
    assert task_target.is_doc is False
    assert task_target.absolute_path == task_backend.resolve()

    # Валидация документации
    doc_targets = [t for t in targets if t.is_doc]
    assert len(doc_targets) == 2
    doc_paths = {t.absolute_path for t in doc_targets}
    assert rules_file.resolve() in doc_paths
    assert claude_file.resolve() in doc_paths
    for dt in doc_targets:
        assert dt.language == "markdown"
        assert dt.is_doc is True
        assert dt.is_base is True

    # 4. Проверка resolve_active_target для файла внутри ворктри задачи
    active_target = WorkspaceScanner.resolve_active_target(order_service_file, [project_config])
    assert active_target is not None
    assert active_target.project_id == "orbita"
    assert active_target.repo_id == "backend"
    assert active_target.branch == "orbit-1035"
    assert active_target.task_name == "orbit-1035"
    assert active_target.is_base is False
    assert active_target.is_doc is False
    assert active_target.language == "csharp"
    assert active_target.absolute_path == task_backend.resolve()

    # 5. Проверка resolve_active_target для документа CLAUDE.md
    active_claude = WorkspaceScanner.resolve_active_target(claude_file, [project_config])
    assert active_claude is not None
    assert active_claude.project_id == "orbita"
    assert active_claude.is_doc is True
    assert active_claude.language == "markdown"

    # 6. Проверка resolve_active_target для несуществующего пути
    non_existent = WorkspaceScanner.resolve_active_target(Path("C:/SomeUnknownPath/test.cs"), [project_config])
    assert non_existent is None


def test_worktree_git_pointer_file(tmp_path):
    """
    Тест определения ветки через файл .git с gitdir: ссылкой (реальный формат git worktree).
    """
    main_repo_git = tmp_path / "main_repo" / ".git" / "worktrees" / "task-1042"
    main_repo_git.mkdir(parents=True)
    (main_repo_git / "HEAD").write_text("ref: refs/heads/feature/task-1042\n", encoding="utf-8")

    wt_dir = tmp_path / "worktrees" / "task-1042" / "backend"
    wt_dir.mkdir(parents=True)
    
    # Создаем файл .git с относительной ссылкой gitdir
    git_file = wt_dir / ".git"
    git_file.write_text(f"gitdir: {os.path.relpath(main_repo_git, wt_dir)}\n", encoding="utf-8")

    detected_branch = WorkspaceScanner.get_git_branch(wt_dir)
    assert detected_branch == "feature/task-1042"


def test_resolve_workspace_targets_integration(temp_config_file, tmp_path):
    """
    Интеграционный тест resolve_workspace_targets из config.py.
    """
    root = tmp_path / "MyWorkspace"
    root.mkdir()
    master_dir = root / "worktrees" / "master"
    master_dir.mkdir(parents=True)
    (master_dir / ".git").mkdir()
    (master_dir / ".git" / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")

    project_data = {
        "id": "ws-1",
        "name": "Test Workspace",
        "workspace_root": str(root),
        "type": "worktree_workspace",
        "base_path": "worktrees/master",
        "tasks_path": "worktrees/{task}",
        "docs_paths": [],
        "repositories": []
    }
    add_or_update_project(project_data)

    # 1. Поиск по project_id
    targets = resolve_workspace_targets("ws-1")
    assert len(targets) == 1
    assert targets[0]["project_id"] == "ws-1"
    assert targets[0]["branch"] == "master"
    assert targets[0]["is_base"] is True

    # 2. Поиск по всем проектам
    all_targets = resolve_workspace_targets(None)
    assert len(all_targets) >= 1
    assert any(t["project_id"] == "ws-1" for t in all_targets)

    # 3. Несуществующий проект
    empty_targets = resolve_workspace_targets("nonexistent-ws")
    assert empty_targets == []
