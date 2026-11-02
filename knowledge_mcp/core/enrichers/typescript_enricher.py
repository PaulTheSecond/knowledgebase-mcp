"""
Компиляторный семантический обогатитель для TypeScript, JavaScript и React на базе TypeScript Compiler API (ADR-009).
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


class TypeScriptSemanticEnricher(BaseSemanticEnricher):
    """Стратегия компиляторного семантического обогащения для TS/JS/React (ts-parser)."""

    @property
    def language_name(self) -> str:
        return "typescript"

    @property
    def target_extensions(self) -> Set[str]:
        return {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}

    def _find_configs(self, repo_path: Path) -> List[Path]:
        """Ищет конфигурационные файлы tsconfig.json или jsconfig.json в репозитории."""
        # 1. Сначала проверяем корень репозитория
        root_ts = repo_path / "tsconfig.json"
        if root_ts.exists():
            return [root_ts]

        root_js = repo_path / "jsconfig.json"
        if root_js.exists():
            return [root_js]

        # 2. Поиск в поддиректориях монорепозитория (исключая node_modules, build-директории)
        configs = []
        try:
            for conf_name in ("tsconfig.json", "jsconfig.json"):
                for p in repo_path.rglob(conf_name):
                    p_str = str(p).replace("\\", "/")
                    if any(ignored in p_str for ignored in ("/node_modules/", "/.git/", "/dist/", "/.angular/", "/.next/")):
                        continue
                    configs.append(p)
                if configs:
                    break
        except Exception as e:
            logger.debug(f"Error scanning for tsconfig in {repo_path}: {e}")

        return configs

    def can_enrich(self, repo_path: Path) -> bool:
        """Проверяет наличие tsconfig.json / jsconfig.json в целевом проекте."""
        return len(self._find_configs(repo_path)) > 0

    def _get_parser_path(self) -> Optional[Path]:
        """Определяет путь к собранному бандлу ts-parser.js."""
        app_root = Path(__file__).resolve().parents[3]
        
        # Основной путь (в контейнере и на хосте после сборки)
        bundle_path = app_root / "TsParser" / "dist" / "ts-parser.js"
        if bundle_path.exists():
            return bundle_path

        # Резервный путь в рабочей директории
        cwd_bundle = Path.cwd() / "TsParser" / "dist" / "ts-parser.js"
        if cwd_bundle.exists():
            return cwd_bundle

        return None

    def run_analysis(
        self,
        repo_path: Path,
        target_files: Optional[List[Path]] = None
    ) -> Tuple[bool, Optional[str], Dict[Path, Dict[str, Any]]]:
        """
        Запускает компиляторный анализатор ts-parser для найденных tsconfig.json.
        """
        configs = self._find_configs(repo_path)
        if not configs:
            return True, None, {}

        parser_path = self._get_parser_path()
        if not parser_path:
            err = "ts-parser.js бандл не найден в TsParser/dist/. Запустите 'npm run build' в TsParser/."
            logger.warning(err)
            return False, err, {}

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
            for config in configs:
                cmd = [
                    "node",
                    str(parser_path),
                    "--project", str(config.resolve()),
                    "--root", str(repo_path.resolve())
                ]

                if files_arg_file:
                    cmd.extend(["--files", files_arg_file, "--mode", "diff"])
                else:
                    cmd.extend(["--mode", "full"])

                logger.debug(f"Running ts-parser command: {' '.join(cmd)}")
                res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

                if res.returncode != 0:
                    err_msg = res.stderr.strip() if res.stderr else f"ts-parser exited with code {res.returncode}"
                    logger.error(f"ts-parser analysis failed for {config}: {err_msg}")
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
                    logger.error(f"Failed to parse ts-parser JSON output for {config}: {ex}")
                    all_success = False
                    last_error = str(ex)

        finally:
            if files_arg_file and os.path.exists(files_arg_file):
                try:
                    os.unlink(files_arg_file)
                except Exception:
                    pass

        return all_success, last_error, parsed_cache
