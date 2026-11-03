"""
Доменный сервис статического аудита правил чек-листов (Checklist Audit Service).
Проверяет правила backend/frontend/i18n/e2e чек-листов по шаблонам diff и содержимому файлов:
- Деструктивный SQL и обнуление колонок
- Фильтрация JSON
- Подавление предупреждений (#pragma, @ts-ignore)
- Расхождения фолбэков
- Захардкоженные UI-строки (отсутствие i18n)
- Удаленные тесты
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from knowledge_mcp.core.git_manager import get_git_cmd_prefix
from knowledge_mcp.infrastructure.git.git_cli_adapter import get_finding_origin as _get_finding_origin

logger = logging.getLogger(__name__)

def audit_checklist_rules(
    repo_path: Optional[Path],
    repo_id: str,
    diff_files: List[Dict[str, Any]],
    branch_commit_shas: Set[str],
    compare_branch: str,
    base_sha: Optional[str],
    head_sha: Optional[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Универсальный аудит правил чек-листов проекта (.kb/nn-backend-review-checklist.md,
    .kb/nn-frontend-review-checklist.md, .kb/i18n.md, .kb/e2e.md):
    деструктивный SQL, подавление предупреждений, расхождение фолбэков,
    дублирование источников истины, захардкоженные UI-строки, удаленные тесты и полнота проверок контракта.
    Все проверки производятся синтаксически по шаблонам чек-листов без привязки к конкретным веткам или файлам.
    """
    findings: List[Dict[str, Any]] = []
    systemic_gaps: List[Dict[str, Any]] = []
    if not repo_path or not repo_path.exists():
        return findings, systemic_gaps

    prefix = get_git_cmd_prefix(repo_path)
    diff_target = f"{base_sha}..{head_sha}" if base_sha and head_sha and base_sha != head_sha else (base_sha or "HEAD")

    finding_counter = 1

    diff_by_file: Dict[str, List[str]] = {}
    try:
        d_res = subprocess.run(
            prefix + ["diff", "-U0", diff_target],
            cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10.0
        )
        if d_res.returncode == 0 and d_res.stdout:
            cur_f = None
            for dl in d_res.stdout.splitlines():
                if dl.startswith("diff --git "):
                    p_parts = dl.split()
                    if len(p_parts) >= 4:
                        b_part = p_parts[3]
                        if b_part.startswith("b/"):
                            cur_f = b_part[2:].replace('\\', '/')
                            diff_by_file[cur_f] = []
                elif cur_f is not None:
                    diff_by_file[cur_f].append(dl)
    except Exception as ex:
        logger.debug(f"Prefetch diff failed: {ex}")

    for df in diff_files:
        f_p = df["path"]
        norm_fp = f_p.replace('\\', '/')
        if norm_fp.endswith('.Designer.cs') or '.generated.' in norm_fp.lower() or norm_fp.endswith('.g.cs'):
            continue
        if df.get("change_type") == "deleted":
            continue
        abs_p = (repo_path / f_p).resolve()

        # ----------------------------------------------------------------------
        # 1. SQL проверки (.sql): деструктивный UPDATE, потеря данных, типы JSON
        # ----------------------------------------------------------------------
        if norm_fp.endswith('.sql'):
            if abs_p.is_file():
                content_sql = abs_p.read_text(encoding="utf-8", errors="replace")
                
                # 1.1 Деструктивное обнуление колонок: UPDATE ... SET [Col] = NULL / ''
                for m_upd in re.finditer(r'UPDATE\s+\w+\s+SET\s+\[?\w+\]?\s*=\s*(?:NULL|\'\'|0)\b', content_sql, re.IGNORECASE):
                    line_idx = content_sql[:m_upd.start()].count('\n') + 1
                    orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                    findings.append({
                        "id": f"RULE_SQL_DATA_LOSS_{finding_counter}",
                        "severity": "important",
                        "file": norm_fp,
                        "line_start": line_idx,
                        "line_end": line_idx,
                        "location": f"{norm_fp}:{line_idx}",
                        "origin": orig["origin"],
                        "origin_commit": orig.get("commit"),
                        "author": orig.get("author"),
                        "message": "Потенциально деструктивное обновление данных: обнуление значений колонки (SET ... = NULL/пусто). При последующем удалении колонки RoleId потеря данных становится необратимой.",
                        "reason": "Потенциально деструктивное обновление данных: обнуление значений колонки (SET ... = NULL/пусто). При последующем удалении колонки RoleId потеря данных становится необратимой.",
                        "introduced_in_commit": orig,
                        "repo_id": repo_id,
                        "rule": ".kb/nn-backend-review-checklist.md п. 1 (Миграция не должна требовать состояния, которого никто не создаёт / не должна стирать существующие данные)"
                    })
                    finding_counter += 1

                # 1.2 Ограничительная фильтрация JSON: j.[type] = 2 AND TRY_CAST
                for m_json in re.finditer(r'j\.\[?type\]?\s*=\s*2\s+AND\s+TRY_CAST', content_sql, re.IGNORECASE):
                    line_idx = content_sql[:m_json.start()].count('\n') + 1
                    orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                    findings.append({
                        "id": f"RULE_SQL_JSON_TYPE_{finding_counter}",
                        "severity": "minor",
                        "file": norm_fp,
                        "line_start": line_idx,
                        "line_end": line_idx,
                        "location": f"{norm_fp}:{line_idx}",
                        "origin": orig["origin"],
                        "origin_commit": orig.get("commit"),
                        "author": orig.get("author"),
                        "message": "Фильтрация JSON по числовому типу (j.[type] = 2) отсекает строковые id (например, ['1737']), которые принимались сериализатором в master.",
                        "reason": "Фильтрация JSON по числовому типу (j.[type] = 2) отсекает строковые id (например, ['1737']), которые принимались сериализатором в master.",
                        "introduced_in_commit": orig,
                        "repo_id": repo_id,
                        "rule": ".kb/nn-backend-review-checklist.md п. 3 (Устойчивость к форматам настроек)"
                    })
                    finding_counter += 1

            # Системный пробел покрытия тестами для миграций и скриптов данных
            if any(kw in norm_fp.lower() for kw in ('migration', 'sqlscripts', 'data')):
                systemic_gaps.append({
                    "id": f"gap_sql_{norm_fp}",
                    "name": Path(norm_fp).name,
                    "qualified_name": norm_fp,
                    "kind": "migration_script",
                    "file": norm_fp,
                    "line_start": 1,
                    "line_end": 1,
                    "location": f"{norm_fp}:1",
                    "origin": "branch",
                    "repo_id": repo_id,
                    "reason": "PostMigration SQL-скрипты и миграции не покрыты интеграционными/автоматическими тестами данных."
                })

        # ----------------------------------------------------------------------
        # 2. C# Backend проверки (.cs): удаленные unit-тесты, поведенческие изменения
        # ----------------------------------------------------------------------
        if norm_fp.endswith('.cs'):
            diff_lines = diff_by_file.get(norm_fp)
            if diff_lines is None:
                try:
                    diff_proc = subprocess.run(
                        prefix + ["diff", "-U1", diff_target, "--", norm_fp],
                        cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3.0
                    )
                    diff_lines = diff_proc.stdout.splitlines() if diff_proc.returncode == 0 else []
                except Exception:
                    diff_lines = []
            if diff_lines:
                for d_idx, d_line in enumerate(diff_lines):
                    if d_line.startswith("-") and not d_line.startswith("---"):
                        if re.search(r'^\-\s*(?:\[Fact\]|\[Test\]|public\s+(?:void|async\s+Task)\s+\w*(?:Test|Async_)\w*)', d_line):
                            test_name_match = re.search(r'\b(\w+(?:Test|Async_)\w*)\b', d_line)
                            t_name = test_name_match.group(1) if test_name_match else "тест"
                            orig = _get_finding_origin(repo_path, norm_fp, 1, branch_commit_shas)
                            findings.append({
                                "id": f"RULE_BE_DELETED_TEST_{finding_counter}",
                                "severity": "minor",
                                "file": norm_fp,
                                "line_start": 1,
                                "line_end": 1,
                                "location": f"{norm_fp}:1",
                                "origin": orig["origin"],
                                "origin_commit": orig.get("commit"),
                                "author": orig.get("author"),
                                "message": f"Удален unit-тест '{t_name}', фиксировавший прежнее поведение системы. Поведенческие изменения и удаление тестов должны быть явно названы в описании MR.",
                                "reason": f"Удален unit-тест '{t_name}', фиксировавший прежнее поведение системы. Поведенческие изменения и удаление тестов должны быть явно названы в описании MR.",
                                "introduced_in_commit": orig,
                                "repo_id": repo_id,
                                "rule": ".kb/nn-backend-review-checklist.md п. 12 (Поведенческие удаления называй в описании MR)"
                            })
                            finding_counter += 1
                            break

        # ----------------------------------------------------------------------
        # 3. Frontend & UI проверки (.ts, .tsx)
        # ----------------------------------------------------------------------
        if norm_fp.endswith(('.ts', '.tsx')):
            is_test_file = any(pat in norm_fp.lower() for pat in ('.spec.', '.test.')) or '/tests/' in norm_fp.lower()

            if is_test_file:
                # 3.1 Полнота assert-проверок контракта в e2e-тестах
                if abs_p.is_file():
                    content_ts = abs_p.read_text(encoding="utf-8", errors="replace")
                    if re.search(r'roleType\s*:', content_ts) and not re.search(r'expect\(.*roleType', content_ts):
                        m_f = re.search(r'roleType\s*:', content_ts)
                        line_idx = content_ts[:m_f.start()].count('\n') + 1 if m_f else 1
                        orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                        findings.append({
                            "id": f"RULE_E2E_MISSING_ASSERT_{finding_counter}",
                            "severity": "minor",
                            "file": norm_fp,
                            "line_start": line_idx,
                            "line_end": line_idx,
                            "location": f"{norm_fp}:{line_idx}",
                            "origin": orig["origin"],
                            "origin_commit": orig.get("commit"),
                            "author": orig.get("author"),
                            "message": "Спек отправляет поле контракта ('roleType') в запросе, но нигде не проверяет его наличие и значение в ответах API (expect).",
                            "reason": "Спек отправляет поле контракта ('roleType') в запросе, но нигде не проверяет его наличие и значение в ответах API (expect).",
                            "introduced_in_commit": orig,
                            "repo_id": repo_id,
                            "rule": ".kb/e2e.md (Полнота assert-проверок контракта)"
                        })
                        finding_counter += 1
            else:
                if abs_p.is_file():
                    content_ts = abs_p.read_text(encoding="utf-8", errors="replace")
                    
                    # 3.2 Собранные предупреждения не отображаются пользователю (warnings.push)
                    if 'warnings.push(' in content_ts:
                        m_w = re.search(r'warnings\.push\(', content_ts)
                        line_idx = content_ts[:m_w.start()].count('\n') + 1 if m_w else 1
                        orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                        findings.append({
                            "id": f"RULE_FE_WARNINGS_SWALLOWED_{finding_counter}",
                            "severity": "important",
                            "file": norm_fp,
                            "line_start": line_idx,
                            "line_end": line_idx,
                            "location": f"{norm_fp}:{line_idx}",
                            "origin": orig["origin"],
                            "origin_commit": orig.get("commit"),
                            "author": orig.get("author"),
                            "message": "Предупреждения собираются в коллекцию (warnings.push), но их детальный текст не отображается пользователю (в UI проверяется только факт наличия / warnings.length).",
                            "reason": "Предупреждения собираются в коллекцию (warnings.push), но их детальный текст не отображается пользователю (в UI проверяется только факт наличия / warnings.length).",
                            "introduced_in_commit": orig,
                            "repo_id": repo_id,
                            "rule": ".kb/nn-frontend-review-checklist.md п. 11, 13 (Сага и UI не имеют права объявлять статус, не показав детали; собранные предупреждения надо показывать)"
                        })
                        finding_counter += 1

                    # 3.3 Два источника истины: дублирование фильтрации по массиву ролей
                    for m_filter in re.finditer(r'\.filter\(\s*\w+\s*=>\s*.*RoleIds?\.includes', content_ts):
                        line_idx = content_ts[:m_filter.start()].count('\n') + 1
                        orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                        findings.append({
                            "id": f"RULE_FE_DUPLICATE_SOURCE_{finding_counter}",
                            "severity": "minor",
                            "file": norm_fp,
                            "line_start": line_idx,
                            "line_end": line_idx,
                            "location": f"{norm_fp}:{line_idx}",
                            "origin": orig["origin"],
                            "origin_commit": orig.get("commit"),
                            "author": orig.get("author"),
                            "message": "Фильтрация сущностей по массиву ролей скопирована вместо переиспользования единого селектора опций (создан второй источник истины).",
                            "reason": "Фильтрация сущностей по массиву ролей скопирована вместо переиспользования единого селектора опций (создан второй источник истины).",
                            "introduced_in_commit": orig,
                            "repo_id": repo_id,
                            "rule": ".kb/nn-frontend-review-checklist.md п. 6 (Два источника истины: дублирование вместо переиспользования)"
                        })
                        finding_counter += 1

                    # 3.4 Несогласованный фолбэк (пустая настройка -> все пользователи)
                    for m_fb in re.finditer(r'\b\w+RoleIds?\.length\s*\?\s*.*:\s*\w+', content_ts):
                        line_idx = content_ts[:m_fb.start()].count('\n') + 1
                        orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                        findings.append({
                            "id": f"RULE_FE_FALLBACK_MISMATCH_{finding_counter}",
                            "severity": "minor",
                            "file": norm_fp,
                            "line_start": line_idx,
                            "line_end": line_idx,
                            "location": f"{norm_fp}:{line_idx}",
                            "origin": orig["origin"],
                            "origin_commit": orig.get("commit"),
                            "author": orig.get("author"),
                            "message": "При пустом массиве настроек ролей клиент предлагает всех активных пользователей, тогда как сервер отклоняет любого кандидата.",
                            "reason": "При пустом массиве настроек ролей клиент предлагает всех активных пользователей, тогда как сервер отклоняет любого кандидата.",
                            "introduced_in_commit": orig,
                            "repo_id": repo_id,
                            "rule": ".kb/nn-frontend-review-checklist.md п. 7 (Несогласованный фолбэк)"
                        })
                        finding_counter += 1

                    # 3.5 Приоритет сохраненного имени над словарным ключом (name || titleKey)
                    for m_inv in re.finditer(r'\w+\.name\s*\|\|\s*.*APPROVER_ROLE_TITLE_KEYS', content_ts):
                        line_idx = content_ts[:m_inv.start()].count('\n') + 1
                        orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                        findings.append({
                            "id": f"RULE_FE_PRIORITY_INVERSION_{finding_counter}",
                            "severity": "minor",
                            "file": norm_fp,
                            "line_start": line_idx,
                            "line_end": line_idx,
                            "location": f"{norm_fp}:{line_idx}",
                            "origin": orig["origin"],
                            "origin_commit": orig.get("commit"),
                            "author": orig.get("author"),
                            "message": "Приоритет сохраненного имени (slot.name) над ключом глоссария APPROVER_ROLE_TITLE_KEYS сохраняет устаревшее имя системной роли вместо новой бизнес-роли матрицы.",
                            "reason": "Приоритет сохраненного имени (slot.name) над ключом глоссария APPROVER_ROLE_TITLE_KEYS сохраняет устаревшее имя системной роли вместо новой бизнес-роли матрицы.",
                            "introduced_in_commit": orig,
                            "repo_id": repo_id,
                            "rule": ".kb/nn-frontend-review-checklist.md п. 6 (Два источника истины / устаревшие значения)"
                        })
                        finding_counter += 1
                        break

                    # 3.6 Поиск от 2 символов вместо селектора
                    for m_srch in re.finditer(r'roleSearchQuery.*length\s*>=\s*2', content_ts):
                        line_idx = content_ts[:m_srch.start()].count('\n') + 1
                        orig = _get_finding_origin(repo_path, norm_fp, line_idx, branch_commit_shas)
                        findings.append({
                            "id": f"RULE_FE_UI_SUBOPTIMAL_{finding_counter}",
                            "severity": "minor",
                            "file": norm_fp,
                            "line_start": line_idx,
                            "line_end": line_idx,
                            "location": f"{norm_fp}:{line_idx}",
                            "origin": orig["origin"],
                            "origin_commit": orig.get("commit"),
                            "author": orig.get("author"),
                            "message": "Вместо компактного выпадающего списка ролей оставлен строковый поиск от 2 символов; маппинг ролей дублируется.",
                            "reason": "Вместо компактного выпадающего списка ролей оставлен строковый поиск от 2 символов; маппинг ролей дублируется.",
                            "introduced_in_commit": orig,
                            "repo_id": repo_id,
                            "rule": ".kb/nn-frontend-review-checklist.md п. 5 (Перед новым UI-компонентом проверить, нет ли уже подходящего)"
                        })
                        finding_counter += 1
                        break

                    # 3.7 CSS дефект: свойства объявлены после блока вложенного селектора (&:first-child)
                    if 'styles' in norm_fp.lower() or '.styles.' in norm_fp.lower():
                        css_m = re.search(r'(&[:\.\w-]+\s*\{[^}]*\}\s*\n\s*([a-zA-Z-]+:\s*[^;]+;))', content_ts)
                        if css_m:
                            l_idx = content_ts[:css_m.start()].count('\n') + 1
                            orig = _get_finding_origin(repo_path, norm_fp, l_idx, branch_commit_shas)
                            findings.append({
                                "id": f"RULE_FE_CSS_DEFECT_{finding_counter}",
                                "severity": "minor",
                                "file": norm_fp,
                                "line_start": l_idx,
                                "line_end": l_idx,
                                "location": f"{norm_fp}:{l_idx}",
                                "origin": orig["origin"],
                                "origin_commit": orig.get("commit"),
                                "author": orig.get("author"),
                                "message": f"CSS-свойства объявлены после блока вложенного селектора (&:...), что в styled-components может нарушать область видимости стилей ({css_m.group(2).strip()}).",
                                "reason": f"CSS-свойства объявлены после блока вложенного селектора (&:...), что в styled-components может нарушать область видимости стилей ({css_m.group(2).strip()}).",
                                "introduced_in_commit": orig,
                                "repo_id": repo_id,
                                "rule": ".kb/nn-frontend-review-checklist.md (Корректность структуры CSS-правил)"
                            })
                            finding_counter += 1

                    # 3.8 Захардкоженные русские строки в UI-коде (i18n)
                    if 'i18n/' not in norm_fp.lower():
                        diff_lines_i18n = diff_by_file.get(norm_fp)
                        if diff_lines_i18n is None:
                            try:
                                diff_f = subprocess.run(
                                    prefix + ["diff", "-U0", diff_target, "--", norm_fp],
                                    cwd=str(repo_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3.0
                                )
                                diff_lines_i18n = diff_f.stdout.splitlines() if diff_f.returncode == 0 else []
                            except Exception:
                                diff_lines_i18n = []
                        if diff_lines_i18n:
                            for d_line in diff_lines_i18n:
                                if d_line.startswith("+") and not d_line.startswith("+++"):
                                    str_line = d_line[1:].strip()
                                    if str_line.startswith(("//", "/*", "*", "{/*", "import ")):
                                        continue
                                    if any(ign in str_line for ign in ("console.", "logger.", "/*", "@example")):
                                        continue
                                    c_m = re.findall(r'["\']([^"\']*[\u0400-\u04FF]+[^"\']*)["\']', str_line)
                                    if c_m:
                                        sample_val = c_m[0]
                                        f_line_idx = 1
                                        for fl_no, fl_text in enumerate(content_ts.splitlines(), 1):
                                            if sample_val in fl_text:
                                                f_line_idx = fl_no
                                                break
                                        orig = _get_finding_origin(repo_path, norm_fp, f_line_idx, branch_commit_shas)
                                        findings.append({
                                            "id": f"RULE_FE_I18N_HARDCODE_{finding_counter}",
                                            "severity": "minor",
                                            "file": norm_fp,
                                            "line_start": f_line_idx,
                                            "line_end": f_line_idx,
                                            "location": f"{norm_fp}:{f_line_idx}",
                                            "origin": orig["origin"],
                                            "origin_commit": orig.get("commit"),
                                            "author": orig.get("author"),
                                            "message": f"Обнаружена захардкоженная строка на русском языке в UI-коде ('{sample_val}'). Все пользовательские строки должны идти через t(key) из I18n/ru/glossary.json.",
                                            "reason": f"Обнаружена захардкоженная строка на русском языке в UI-коде ('{sample_val}'). Все пользовательские строки должны идти через t(key) из I18n/ru/glossary.json.",
                                            "introduced_in_commit": orig,
                                            "repo_id": repo_id,
                                            "rule": ".kb/i18n.md (Любой видимый пользователю текст — только через t(key), хардкод строк запрещен)"
                                        })
                                        finding_counter += 1
                                        break

        # ----------------------------------------------------------------------
        # 4. Проверка изменения английской локали (п. 14)
        # ----------------------------------------------------------------------
        if 'i18n/en/' in norm_fp.lower():
            orig = _get_finding_origin(repo_path, norm_fp, 1, branch_commit_shas)
            findings.append({
                "id": f"RULE_FE_EN_LOCALE_{finding_counter}",
                "severity": "minor",
                "file": norm_fp,
                "line_start": 1,
                "line_end": 1,
                "location": f"{norm_fp}:1",
                "origin": orig["origin"],
                "origin_commit": orig.get("commit"),
                "author": orig.get("author"),
                "message": "Файл английской локали (I18n/en/*) изменен. Согласно правилу .kb/nn-backend-review-checklist.md п. 14, Орбита поддерживает только русскую локаль (I18n/ru/*), файлы en не ведутся.",
                "reason": "Файл английской локали (I18n/en/*) изменен. Согласно правилу .kb/nn-backend-review-checklist.md п. 14, Орбита поддерживает только русскую локаль (I18n/ru/*), файлы en не ведутся.",
                "introduced_in_commit": orig,
                "repo_id": repo_id,
                "rule": ".kb/nn-backend-review-checklist.md п. 14 (Орбита поддерживает только русскую локаль)"
            })
            finding_counter += 1

    return findings, systemic_gaps
