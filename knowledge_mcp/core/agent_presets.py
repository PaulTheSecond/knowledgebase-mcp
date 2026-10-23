"""
Модуль управления интеграционными пресетами для AI-агентов:
Claude Code, Antigravity, Cursor, Codex.
Обеспечивает Zero-Footprint интеграцию и генерацию конфигурационных файлов.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
                        "command": "echo \"[KnowledgeBase MCP] Рекомендуется использовать knowledge_search или knowledge_review_context для точного семантического и графового анализа перед глобальным Grep/Glob.\""
                    }
                ]
            }
        }, indent=2, ensure_ascii=False) + "\n",
        "CLAUDE.md.snippet": (
            "В начале сессии или перед реализацией задачи вызови MCP-инструмент "
            "knowledge_get_project_rules(topic=...) для получения актуальных стандартов кода, "
            "правил работы с тестовой БД и изоляции тестов.\n"
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
            "# Приоритет KnowledgeBase MCP в Antigravity\n\n"
            "При решении задач разработки, навигации по кодовой базе и рефакторинга всегда отдавай приоритет "
            "инструментам KnowledgeBase MCP (`knowledge_search`, `knowledge_find_symbol`, `knowledge_get_callers`, "
            "`knowledge_get_callees`, `knowledge_review_context`). Перед реализацией новой функциональности или запуском "
            "тестов вызови `knowledge_get_project_rules(topic=...)` для загрузки релевантных проектных правил, "
            "архитектурных инвариантов и стандартов тестирования.\n"
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
            # Определение целевой директории для агента:
            # Если пользователь передал каталог, названный в честь агента (например, .claude или antigravity),
            # пишем прямо в него. Иначе создаем поддиректорию с именем агента.
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
