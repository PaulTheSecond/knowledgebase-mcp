"""
Тесты для Эпика 7: Интеграционные пресеты для AI-агентов
(Claude Code, Antigravity, Cursor, Codex) и CLI sync-file.
"""

import json
import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from knowledge_mcp.core.agent_presets import AgentPresetManager, AVAILABLE_AGENTS
from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.main import main


@pytest.fixture
def temp_db():
    """Временная база данных для тестирования CLI sync-file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "test_presets.db"
        db = KnowledgeDB(db_file)
        yield db, db_file
        db.close()


class TestAgentPresets:
    """Набор тестов для каталога шаблонов и менеджера пресетов."""

    def test_preset_templates_exist_and_valid_json(self):
        """Проверяет физическое наличие всех 7 файлов пресетов и валидность JSON."""
        templates_dir = Path(__file__).resolve().parent.parent / "templates" / "agent_presets"
        assert templates_dir.is_dir(), f"Каталог {templates_dir} не найден"

        # 1. Claude Code
        claude_settings = templates_dir / "claude" / "settings.json"
        claude_snippet = templates_dir / "claude" / "CLAUDE.md.snippet"
        assert claude_settings.is_file()
        assert claude_snippet.is_file()

        # Проверка JSON Claude
        with open(claude_settings, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "mcpServers" in data
        assert "knowledgebase" in data["mcpServers"]
        assert "hooks" in data
        assert "SessionStart" in data["hooks"]
        assert "PostToolUse" in data["hooks"]
        assert "PreToolUse" in data["hooks"]
        assert any("sync-file --file" in h.get("command", "") for h in data["hooks"]["PostToolUse"])

        snippet_text = claude_snippet.read_text(encoding="utf-8")
        assert "knowledge_get_project_rules" in snippet_text

        # 2. Antigravity
        agy_mcp = templates_dir / "antigravity" / "mcp_config.json"
        agy_rule = templates_dir / "antigravity" / "rule_snippet.md"
        assert agy_mcp.is_file()
        assert agy_rule.is_file()

        # Проверка JSON Antigravity
        with open(agy_mcp, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "mcpServers" in data
        assert "knowledgebase" in data["mcpServers"]

        rule_text = agy_rule.read_text(encoding="utf-8")
        assert "knowledge_search" in rule_text
        assert "knowledge_get_project_rules" in rule_text

        # 3. Cursor
        cursor_mcp = templates_dir / "cursor" / "cursor_mcp.json"
        cursor_rules = templates_dir / "cursor" / "cursorrules.snippet"
        assert cursor_mcp.is_file()
        assert cursor_rules.is_file()

        # Проверка JSON Cursor
        with open(cursor_mcp, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "mcpServers" in data
        assert "knowledgebase" in data["mcpServers"]

        cursor_text = cursor_rules.read_text(encoding="utf-8")
        assert "knowledge_get_project_rules" in cursor_text

        # 4. Codex
        codex_prompt = templates_dir / "codex" / "system_prompt.snippet"
        assert codex_prompt.is_file()
        codex_text = codex_prompt.read_text(encoding="utf-8")
        assert "knowledge_search" in codex_text
        assert "knowledge_review_context" in codex_text

    def test_agent_preset_manager_get_preset_files(self):
        """Проверяет метод get_preset_files для всех доступных агентов."""
        for agent in AVAILABLE_AGENTS:
            files = AgentPresetManager.get_preset_files(agent)
            assert isinstance(files, dict)
            assert len(files) >= 1
            for name, content in files.items():
                assert isinstance(name, str)
                assert isinstance(content, str)
                assert len(content.strip()) > 0

        with pytest.raises(ValueError, match="Неизвестный агент"):
            AgentPresetManager.get_preset_files("non_existent_agent")

    def test_agent_preset_manager_export_all(self, tmp_path):
        """Проверяет экспорт всех пресетов в директорию."""
        created = AgentPresetManager.export_presets("all", output_dir=tmp_path)
        assert len(created) >= 7

        for ag in AVAILABLE_AGENTS:
            agent_dir = tmp_path / ag
            assert agent_dir.is_dir()
            files_in_dir = list(agent_dir.iterdir())
            assert len(files_in_dir) >= 1

    def test_agent_preset_manager_export_single_agent(self, tmp_path):
        """Проверяет экспорт одного агента в директорию."""
        claude_out = tmp_path / "claude_out"
        created = AgentPresetManager.export_presets("claude", output_dir=claude_out)
        assert len(created) == 2

        # Проверяем синтаксис созданного JSON
        settings_file = [f for f in created if f.name == "settings.json"][0]
        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert "hooks" in data

    def test_agent_preset_manager_print_presets(self, capsys):
        """Проверяет корректность консольного вывода пресетов."""
        AgentPresetManager.print_presets("claude")
        captured = capsys.readouterr()
        assert "ИНТЕГРАЦИОННЫЙ ПРЕСЕТ ДЛЯ АГЕНТА: CLAUDE" in captured.out
        assert "CLAUDE.md.snippet" in captured.out
        assert "settings.json" in captured.out


class TestPresetAndSyncCLI:
    """Тесты интерфейса командной строки main: preset и sync-file."""

    def test_cli_preset_command_output_dir(self, tmp_path, capsys):
        """Тест CLI-команды `preset all --output-dir <dir>`."""
        test_args = ["main.py", "preset", "all", "--output-dir", str(tmp_path)]
        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        assert "Успешно создано" in captured.out
        assert (tmp_path / "claude" / "settings.json").is_file()
        assert (tmp_path / "antigravity" / "mcp_config.json").is_file()
        assert (tmp_path / "cursor" / "cursor_mcp.json").is_file()
        assert (tmp_path / "codex" / "system_prompt.snippet").is_file()

    def test_cli_sync_file_check_status(self, temp_db, capsys):
        """Тест CLI-команды `sync-file --check-status`."""
        _, db_path = temp_db
        test_args = ["main.py", "--db-path", str(db_path), "sync-file", "--check-status"]
        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        assert "[KnowledgeBase MCP] Status: OK" in captured.out
        assert "Database:" in captured.out
        assert "Indexed files in CAS:" in captured.out

    def test_cli_sync_file_index_and_delete(self, temp_db, tmp_path, capsys):
        """Тест CLI-команды `sync-file --file <path>` для индексации и удаления файла."""
        db, db_path = temp_db

        # 1. Создаем тестовый файл с кодом
        test_file = tmp_path / "CalculatorService.py"
        test_file.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

        test_args = [
            "main.py",
            "--db-path", str(db_path),
            "sync-file",
            "--file", str(test_file),
            "--project-id", "test_proj",
            "--repo-id", "test_repo",
            "--branch", "main"
        ]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        assert "[JIT] Indexed CalculatorService.py" in captured.out

        # Проверяем, что файл попал в CAS БД
        db_after = KnowledgeDB(db_path)
        content = db_after.get_file_content("CalculatorService.py", repo_id="test_repo")
        db_after.close()
        assert content is not None
        assert "def add(a, b):" in content

        # 2. Удаляем файл и снова вызываем sync-file
        test_file.unlink()

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        assert "[JIT] Deleted CalculatorService.py" in captured.out

        # Проверяем, что файл удален из БД
        db_after = KnowledgeDB(db_path)
        content_deleted = db_after.get_file_content("CalculatorService.py", repo_id="test_repo")
        db_after.close()
        assert content_deleted is None

    def test_cli_sync_file_missing_file_or_flag(self, capsys):
        """Проверяет выход с ошибкой, если не передан ни --file, ни --check-status."""
        test_args = ["main.py", "sync-file"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
