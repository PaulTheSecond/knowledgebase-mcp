"""
Компиляторный семантический обогатитель для C# / .NET проектов на базе Roslyn (ADR-002, ADR-009).
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from knowledge_mcp.core.enrichers.base import BaseSemanticEnricher

logger = logging.getLogger(__name__)


class RoslynSemanticEnricher(BaseSemanticEnricher):
    """Стратегия обогащения семантики для .NET (C#) на базе Roslyn."""

    @property
    def language_name(self) -> str:
        return "c_sharp"

    @property
    def target_extensions(self) -> Set[str]:
        return {".cs"}

    def _find_csharp_projects(self, repo_path: Path) -> List[Path]:
        """Ищет файлы .sln или .csproj в репозитории (исключая внутренний RoslynParser)."""
        sln_files = [
            f for f in repo_path.glob("*.sln")
            if "RoslynParser" not in str(f)
        ]
        if sln_files:
            return sln_files

        # Поиск в поддиректориях с ограничением глубины
        try:
            sln_files = [
                f for f in repo_path.rglob("*.sln")
                if "RoslynParser" not in str(f) and ".git" not in str(f) and "node_modules" not in str(f)
            ]
            if sln_files:
                return sln_files
        except Exception:
            pass

        try:
            csproj_files = [
                f for f in repo_path.rglob("*.csproj")
                if "RoslynParser" not in str(f) and ".git" not in str(f) and "node_modules" not in str(f)
            ]
            return csproj_files
        except Exception:
            return []

    def can_enrich(self, repo_path: Path) -> bool:
        """Проверяет наличие C# решений или проектов в репозитории."""
        return len(self._find_csharp_projects(repo_path)) > 0

    def run_analysis(
        self,
        repo_path: Path,
        target_files: Optional[List[Path]] = None
    ) -> Tuple[bool, Optional[str], Dict[Path, Dict[str, Any]]]:
        """
        Запускает RoslynParser для обнаруженных .sln / .csproj файлов.
        """
        projects = self._find_csharp_projects(repo_path)
        if not projects:
            return True, None, {}

        app_root = Path(__file__).resolve().parents[3]
        parser_dll = app_root / "RoslynParser" / "bin" / "Release" / "net8.0" / "RoslynParser.dll"
        parser_csproj = app_root / "RoslynParser" / "RoslynParser.csproj"

        path_resolved_cache: Dict[str, Path] = {}

        def get_resolved(path_str: str) -> Path:
            if path_str not in path_resolved_cache:
                path_resolved_cache[path_str] = Path(path_str).resolve()
            return path_resolved_cache[path_str]

        files_arg_file = None
        if target_files:
            tf = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt")
            for p in target_files:
                tf.write(str(p.resolve()) + "\n")
            tf.close()
            files_arg_file = tf.name

        all_success = True
        last_error = None
        parsed_cache: Dict[Path, Dict[str, Any]] = {}

        try:
            for proj in projects:
                if "RoslynParser" in str(proj):
                    continue

                if parser_dll.exists():
                    cmd = ["dotnet", str(parser_dll), str(proj)]
                else:
                    cmd = [
                        "dotnet", "run", "-c", "Release",
                        "--project", str(parser_csproj),
                        "--", str(proj)
                    ]

                if files_arg_file:
                    cmd.extend(["--files", files_arg_file])

                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    err_msg = res.stderr.strip() if res.stderr else f"Roslyn exited with code {res.returncode}"
                    logger.error(f"Roslyn analysis failed for {proj}: {err_msg}")
                    all_success = False
                    last_error = err_msg
                    continue

                try:
                    data = json.loads(res.stdout)
                    for s in data.get("symbols", []):
                        fp = get_resolved(s["file_path"])
                        if fp not in parsed_cache:
                            parsed_cache[fp] = {"symbols": [], "edges": []}
                        parsed_cache[fp]["symbols"].append(s)

                    id_to_file = {
                        s["ast_node_id"]: get_resolved(s["file_path"])
                        for s in data.get("symbols", [])
                    }

                    for e in data.get("edges", []):
                        src_id = e.get("source_ast_id")
                        if src_id in id_to_file:
                            fp = id_to_file[src_id]
                            parsed_cache[fp]["edges"].append(e)

                except Exception as ex:
                    logger.error(f"Failed to parse Roslyn output for {proj}: {ex}")
                    all_success = False
                    last_error = str(ex)

        finally:
            if files_arg_file and os.path.exists(files_arg_file):
                try:
                    os.unlink(files_arg_file)
                except Exception:
                    pass

        return all_success, last_error, parsed_cache
