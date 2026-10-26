import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

AVAILABLE_AGENTS = ["claude", "antigravity", "cursor", "codex"]

# Встроенные эталонные шаблоны (fallback при отсутствии файлов на диске)
EMBEDDED_PRESETS: Dict[str, Dict[str, str]] = {
    "claude": {
        "settings.json": json.dumps({
            "mcpServers": {
                "knowledgebase": {
                    "command": "python",
                    "args": ["-m", "knowledge_mcp.main", "mcp"]
                }
            },
            "hooks": {
                "SessionStart": [
                    {
                        "command": "python -m knowledge_mcp.main sync-file --check-status"
                    }
                ],
                "PostToolUse": [
                    {
                        "matcher": "Edit|Write",
                        "command": "python -m knowledge_mcp.main sync-file --file \"$FILE\""
                    }
                ],
                "PreToolUse": [
                    {
                        "matcher": "Grep|Glob",
                        "command": "echo \"[KnowledgeBase MCP] Prioritize knowledge_search or knowledge_review_context for precise semantic and graph-based analysis before running broad Grep/Glob scans.\""
                    }
                ]
            }
        }, indent=2, ensure_ascii=False) + "\n",
        "CLAUDE.md.snippet": (
            "At session startup or before implementing a task, invoke the KnowledgeBase MCP tool "
            "`knowledge_get_project_rules(topic=...)` to retrieve active coding standards, "
            "test database rules, and test isolation guidelines.\n"
        )
    },
    "antigravity": {
        "mcp_config.json": json.dumps({
            "mcpServers": {
                "knowledgebase": {
                    "command": "python",
                    "args": ["-m", "knowledge_mcp.main", "mcp"],
                    "env": {
                        "PYTHONUNBUFFERED": "1"
                    }
                }
            }
        }, indent=2, ensure_ascii=False) + "\n",
        "rule_snippet.md": (
            "# KnowledgeBase MCP Priority in Antigravity\n\n"
            "When tackling development, codebase navigation, and refactoring tasks, always prioritize "
            "KnowledgeBase MCP tools (`knowledge_search`, `knowledge_find_symbol`, `knowledge_get_callers`, "
            "`knowledge_get_callees`, `knowledge_review_context`). Before implementing any task, "
            "invoke `knowledge_get_project_rules(topic=...)` to load relevant project guidelines, "
            "architectural invariants, and testing standards.\n"
        )
    },
    "cursor": {
        "cursor_mcp.json": json.dumps({
            "mcpServers": {
                "knowledgebase": {
                    "command": "python",
                    "args": ["-m", "knowledge_mcp.main", "mcp"]
                }
            }
        }, indent=2, ensure_ascii=False) + "\n",
        "cursorrules.snippet": (
            "Always use KnowledgeBase MCP tools (knowledge_search, knowledge_find_symbol, knowledge_review_context) "
            "for code exploration and blast-radius analysis. Before implementing changes, invoke "
            "knowledge_get_project_rules(topic=...) to retrieve active coding standards and test isolation guidelines.\n"
        )
    },
    "codex": {
        "system_prompt.snippet": (
            "You have access to the KnowledgeBase MCP tools. Prioritize using `knowledge_search` and `knowledge_find_symbol` "
            "for exploring code architecture, callers, and callees instead of scanning entire directories. "
            "Before executing code changes or creating pull requests, call `knowledge_get_project_rules` with the relevant "
            "topic to load coding standards, and use `knowledge_review_context` to assess blast radius and test coverage.\n"
        )
    }
}


class AgentPresetManager:
    """Менеджер генерации и экспорта интеграционных пресетов для AI-агентов."""

    @classmethod
    def get_templates_dir(cls) -> Optional[Path]:
        """Ищет физический каталог templates/agent_presets."""
        current_dir = Path(__file__).resolve().parent
        candidates = [
            current_dir.parent.parent / "templates" / "agent_presets",
            current_dir.parent / "templates" / "agent_presets",
            Path.cwd() / "templates" / "agent_presets",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return None

    @classmethod
    def get_preset_files(cls, agent: str) -> Dict[str, str]:
        """
        Возвращает словарь {filename: content} для указанного агента.
        Сначала проверяет файлы на диске в templates/agent_presets/<agent>,
        затем использует эталонный fallback из EMBEDDED_PRESETS.
        """
        agent = agent.lower().strip()
        if agent not in AVAILABLE_AGENTS:
            raise ValueError(f"Неизвестный агент: '{agent}'. Доступны: {', '.join(AVAILABLE_AGENTS)}")

        result: Dict[str, str] = {}
        templates_dir = cls.get_templates_dir()

        if templates_dir:
            agent_dir = templates_dir / agent
            if agent_dir.is_dir():
                for p in agent_dir.iterdir():
                    if p.is_file():
                        try:
                            result[p.name] = p.read_text(encoding="utf-8")
                        except Exception as e:
                            logger.warning(f"Ошибка чтения файла пресета {p}: {e}")

        # Дополняем отсутствующие файлы из встроенных эталонов
        embedded = EMBEDDED_PRESETS.get(agent, {})
        for filename, content in embedded.items():
            if filename not in result:
                result[filename] = content

        return result

    @classmethod
    def export_presets(
        cls,
        agent: str = "all",
        output_dir: Optional[Path | str] = None
    ) -> List[Path]:
        """
        Экспортирует пресеты в целевую директорию.
        Если output_dir не передан, экспортирует в текущую директорию.
        Возвращает список созданных путей к файлам.
        """
        agent = agent.lower().strip()
        if agent != "all" and agent not in AVAILABLE_AGENTS:
            raise ValueError(f"Неизвестный агент: '{agent}'. Доступны: {', '.join(AVAILABLE_AGENTS)}, all")

        agents_to_export = AVAILABLE_AGENTS if agent == "all" else [agent]
        base_out = Path(output_dir or ".").resolve()
        created_files: List[Path] = []

        for ag in agents_to_export:
            cleaned_dirname = base_out.name.lower().strip(".")
            if len(agents_to_export) == 1 and cleaned_dirname == ag:
                target_dir = base_out
            else:
                target_dir = base_out / ag

            target_dir.mkdir(parents=True, exist_ok=True)
            files = cls.get_preset_files(ag)

            for filename, content in files.items():
                dest_file = target_dir / filename
                dest_file.write_text(content, encoding="utf-8")
                created_files.append(dest_file)

        return created_files

    @classmethod
    def print_presets(cls, agent: str = "all") -> None:
        """Выводит содержимое пресетов в консоль."""
        agent = agent.lower().strip()
        if agent != "all" and agent not in AVAILABLE_AGENTS:
            raise ValueError(f"Неизвестный агент: '{agent}'. Доступны: {', '.join(AVAILABLE_AGENTS)}, all")

        agents_to_print = AVAILABLE_AGENTS if agent == "all" else [agent]

        for ag in agents_to_print:
            print("=" * 70)
            print(f"  ИНТЕГРАЦИОННЫЙ ПРЕСЕТ ДЛЯ АГЕНТА: {ag.upper()}")
            print("=" * 70)
            files = cls.get_preset_files(ag)
            for filename, content in files.items():
                print(f"\n--- [{ag}] {filename} ---")
                print(content.rstrip())
            print()

    @classmethod
    def install_presets(
        cls,
        agent: str = "all",
        home_dir: Optional[Path] = None,
        python_executable: Optional[str] = None,
        scope: str = "user"
    ) -> List[str]:
        """
        Автоматически и безопасно устанавливает (merge) пресеты в системные профили пользователя:
        - Claude Code: ~/.claude/settings.json + ~/.claude/CLAUDE.md
        - Antigravity: ~/.gemini/antigravity/mcp_config.json + rules/knowledgebase.md
        - Cursor: ~/.cursor/mcp.json + ~/.cursor/rules/knowledgebase.mdc
        - Codex / Copilot: ~/.codex/instructions.md

        Сохраняет резервные копии (.bak) и выполняет умное слияние (safe merge),
        не затирая существующие пользовательские плагины, права и другие MCP-серверы.
        """
        agent = agent.lower().strip()
        if agent != "all" and agent not in AVAILABLE_AGENTS:
            raise ValueError(f"Неизвестный агент: '{agent}'. Доступны: {', '.join(AVAILABLE_AGENTS)}, all")

        agents_to_install = AVAILABLE_AGENTS if agent == "all" else [agent]
        scope = scope.lower().strip()
        base_dir = home_dir if home_dir else (Path.cwd() if scope == "project" else Path.home())
        py_exe = python_executable or sys.executable
        actions_taken: List[str] = []

        repo_root = Path(__file__).resolve().parent.parent.parent
        db_path = os.environ.get("DB_PATH", str(repo_root / "data" / "knowledge.db"))

        for ag in agents_to_install:
            if ag == "claude":
                claude_dir = base_dir / ".claude"
                claude_dir.mkdir(parents=True, exist_ok=True)
                settings_path = claude_dir / "settings.json"

                data: Dict[str, Any] = {}
                if settings_path.exists():
                    try:
                        backup_path = claude_dir / "settings.json.bak"
                        backup_path.write_text(settings_path.read_text(encoding="utf-8"), encoding="utf-8")
                        actions_taken.append(f"Backup created: {backup_path}")
                        data = json.loads(settings_path.read_text(encoding="utf-8"))
                    except Exception as e:
                        logger.warning(f"Failed to read existing settings.json: {e}")
                        data = {}

                # 1. Merge mcpServers
                mcp_servers = data.setdefault("mcpServers", {})
                mcp_servers["knowledgebase"] = {
                    "command": py_exe,
                    "args": ["-m", "knowledge_mcp.main", "--db-path", db_path, "mcp"],
                    "cwd": str(repo_root),
                    "env": {
                        "PYTHONPATH": str(repo_root),
                        "PYTHONUNBUFFERED": "1"
                    }
                }

                # 2. Merge hooks
                hooks = data.setdefault("hooks", {})
                # SessionStart
                session_start = hooks.setdefault("SessionStart", [])
                if not any("sync-file --check-status" in h.get("command", "") for h in session_start):
                    session_start.append({"command": f'"{py_exe}" -m knowledge_mcp.main --db-path "{db_path}" sync-file --check-status'})

                # PostToolUse
                post_tool = hooks.setdefault("PostToolUse", [])
                if not any("sync-file --file" in h.get("command", "") for h in post_tool):
                    post_tool.append({
                        "matcher": "Edit|Write",
                        "command": f'"{py_exe}" -m knowledge_mcp.main --db-path "{db_path}" sync-file --file "$FILE"'
                    })

                # PreToolUse
                pre_tool = hooks.setdefault("PreToolUse", [])
                if not any("knowledge_search" in h.get("command", "") for h in pre_tool):
                    pre_tool.append({
                        "matcher": "Grep|Glob",
                        "command": 'echo "[KnowledgeBase MCP] Prioritize knowledge_search or knowledge_review_context for precise semantic and graph-based analysis before running broad Grep/Glob scans."'
                    })

                settings_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                actions_taken.append(f"Updated Claude settings: {settings_path}")

                # 3. Append to CLAUDE.md
                claude_md = base_dir / "CLAUDE.md" if scope == "project" else claude_dir / "CLAUDE.md"
                md_text = claude_md.read_text(encoding="utf-8") if claude_md.exists() else ""
                if "knowledge_get_project_rules" not in md_text:
                    snippet = (
                        "\n\n## KnowledgeBase MCP\n"
                        "At session startup or before implementing a task, invoke the KnowledgeBase MCP tool "
                        "`knowledge_get_project_rules(topic=...)` to retrieve active coding standards, "
                        "test database rules, and test isolation guidelines.\n"
                    )
                    claude_md.write_text(md_text.rstrip() + snippet, encoding="utf-8")
                    actions_taken.append(f"Updated Claude guidelines: {claude_md}")

            elif ag == "antigravity":
                agy_dir = (base_dir / ".agent") if scope == "project" else (base_dir / ".gemini" / "antigravity")
                agy_dir.mkdir(parents=True, exist_ok=True)
                mcp_path = agy_dir / "mcp_config.json"

                data: Dict[str, Any] = {}
                if mcp_path.exists():
                    try:
                        backup_path = agy_dir / "mcp_config.json.bak"
                        backup_path.write_text(mcp_path.read_text(encoding="utf-8"), encoding="utf-8")
                        actions_taken.append(f"Backup created: {backup_path}")
                        data = json.loads(mcp_path.read_text(encoding="utf-8"))
                    except Exception:
                        data = {}

                mcp_servers = data.setdefault("mcpServers", {})
                mcp_servers["knowledgebase"] = {
                    "command": py_exe,
                    "args": ["-m", "knowledge_mcp.main", "--db-path", db_path, "mcp"],
                    "cwd": str(repo_root),
                    "env": {
                        "PYTHONPATH": str(repo_root),
                        "PYTHONUNBUFFERED": "1"
                    }
                }
                mcp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                actions_taken.append(f"Updated Antigravity MCP config: {mcp_path}")

                rules_dir = agy_dir / "rules"
                rules_dir.mkdir(parents=True, exist_ok=True)
                rule_file = rules_dir / "knowledgebase.md"
                rule_text = cls.get_preset_files("antigravity").get("rule_snippet.md", "")
                rule_file.write_text(rule_text, encoding="utf-8")
                actions_taken.append(f"Updated Antigravity rule: {rule_file}")

            elif ag == "cursor":
                cursor_dir = base_dir / ".cursor"
                cursor_dir.mkdir(parents=True, exist_ok=True)
                mcp_path = cursor_dir / "mcp.json"

                data: Dict[str, Any] = {}
                if mcp_path.exists():
                    try:
                        backup_path = cursor_dir / "mcp.json.bak"
                        backup_path.write_text(mcp_path.read_text(encoding="utf-8"), encoding="utf-8")
                        actions_taken.append(f"Backup created: {backup_path}")
                        data = json.loads(mcp_path.read_text(encoding="utf-8"))
                    except Exception:
                        data = {}

                mcp_servers = data.setdefault("mcpServers", {})
                mcp_servers["knowledgebase"] = {
                    "command": py_exe,
                    "args": ["-m", "knowledge_mcp.main", "--db-path", db_path, "mcp"],
                    "cwd": str(repo_root),
                    "env": {
                        "PYTHONPATH": str(repo_root),
                        "PYTHONUNBUFFERED": "1"
                    }
                }
                mcp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                actions_taken.append(f"Updated Cursor MCP config: {mcp_path}")

                if scope == "project":
                    rule_file = base_dir / ".cursorrules"
                    rule_text = cls.get_preset_files("cursor").get("cursorrules.snippet", "")
                    rule_file.write_text(rule_text, encoding="utf-8")
                    actions_taken.append(f"Updated Cursor rule: {rule_file}")
                else:
                    rules_dir = cursor_dir / "rules"
                    rules_dir.mkdir(parents=True, exist_ok=True)
                    rule_file = rules_dir / "knowledgebase.mdc"
                    rule_text = cls.get_preset_files("cursor").get("cursorrules.snippet", "")
                    rule_file.write_text(rule_text, encoding="utf-8")
                    actions_taken.append(f"Updated Cursor rule: {rule_file}")

            elif ag == "codex":
                codex_dir = base_dir / ".codex"
                codex_dir.mkdir(parents=True, exist_ok=True)
                inst_path = codex_dir / "instructions.md"

                existing = inst_path.read_text(encoding="utf-8") if inst_path.exists() else ""
                snippet = cls.get_preset_files("codex").get("system_prompt.snippet", "")
                if "knowledge_search" not in existing:
                    inst_path.write_text((existing.rstrip() + "\n\n" + snippet).strip() + "\n", encoding="utf-8")
                    actions_taken.append(f"Updated Codex instructions: {inst_path}")

        return actions_taken
