"""Доменный модуль анализа тестовых пробелов (Test Gaps) и аудита отрецензированных файлов.

Содержит чистую бизнес-логику для:
1. Выявления непокрытых тестами режимов вызова компонентов/функций (extract_call_modes_and_gaps).
2. Фильтрации замечаний без строгой привязки к строкам кода (filter_valid_findings).
3. Формирования структурированного списка всех отрецензированных файлов (build_reviewed_files_audit).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Стандартные пропсы React/UI компонентов, не требующие специфичных тест-кейсов покрытия
STANDARD_PROPS: Set[str] = {
    'items', 'onchange', 'options', 'disabled', 'searchable', 'classname',
    'style', 'id', 'name', 'filteroption', 'key', 'ref', 'children',
    'selectplaceholder', 'itemlabel'
}


def extract_call_modes_and_gaps(
    sym: Dict[str, Any],
    component_consumer_files: List[str],
    repo_path: Optional[Path],
    tested_symbol_ids: Set[Any],
    feature_test_keywords: Set[str],
    related_tests: List[Dict[str, Any]],
    code_parser: Any
) -> List[Dict[str, Any]]:
    """Универсально выявляет режимы вызова компонента/функции в местах использования и определяет пробелы в тестах."""
    gaps: List[Dict[str, Any]] = []
    s_name = sym.get('name')
    if not s_name or not repo_path or not repo_path.exists() or len(s_name) < 3:
        return gaps

    mode_consumers: Dict[str, List[str]] = {}
    tsx_parser = code_parser.parsers.get('tsx') if code_parser and hasattr(code_parser, 'parsers') else None

    for cf in component_consumer_files:
        full_p = repo_path / cf
        if not full_p.exists():
            continue
        try:
            content_bytes = full_p.read_bytes()
            if tsx_parser and full_p.suffix.lower() in ('.tsx', '.jsx', '.ts', '.js'):
                tree = tsx_parser.parse(content_bytes)

                def _walk_jsx(node: Any) -> None:
                    if node.type in ('jsx_element', 'jsx_self_closing_element'):
                        opening = node.child_by_field_name('opening_element') if node.type == 'jsx_element' else node
                        if opening is None and node.children:
                            opening = node.children[0]
                        if opening:
                            tag = opening.child_by_field_name('name') or (opening.children[1] if len(opening.children) > 1 else None)
                            if tag and code_parser._get_node_text(tag, content_bytes) == s_name:
                                for child in opening.children:
                                    if child.type == 'jsx_attribute':
                                        prop_name_node = child.child_by_field_name('property') or child.children[0]
                                        p_name = code_parser._get_node_text(prop_name_node, content_bytes)
                                        if p_name and p_name.lower() not in STANDARD_PROPS:
                                            mode_consumers.setdefault(p_name, []).append(cf)
                    for c in node.children:
                        _walk_jsx(c)

                _walk_jsx(tree.root_node)
        except Exception:
            pass

    sym_file = sym.get('path') or sym.get('file')
    sym_loc = f"{sym_file}:{sym.get('line_start')}-{sym.get('line_end')}" if sym.get('line_start') else sym_file
    qname = sym.get('qualified_name') or s_name

    for prop_mode, consumers in mode_consumers.items():
        unique_consumers = sorted(list(set(consumers)))
        is_mode_covered = False
        for c_file in unique_consumers:
            c_stem = Path(c_file).stem.lower()
            if any(kw in c_stem for kw in feature_test_keywords):
                is_mode_covered = True
                break

        if not is_mode_covered:
            gaps.append({
                "id": f"{sym['id']}:mode:{prop_mode}",
                "name": f"{s_name} (mode: {prop_mode})",
                "qualified_name": f"{qname}.{prop_mode}",
                "kind": "feature_mode",
                "file": sym_file,
                "line_start": sym.get('line_start') or 1,
                "line_end": sym.get('line_end') or 1,
                "location": sym_loc,
                "mode": prop_mode,
                "reason": f"Режим пропса '{prop_mode}' используется в потребителях ({', '.join(Path(c).name for c in unique_consumers)}), но не покрыт e2e или unit тестами",
                "uncovered_consumer_files": unique_consumers
            })

    return gaps


def filter_valid_findings(gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Фильтрует замечания, отбрасывая формулировки без точной привязки к строке и общие советы."""
    filtered: List[Dict[str, Any]] = []
    for g in gaps:
        loc = g.get("location")
        ls = g.get("line_start")
        reason = g.get("reason", "")
        if not loc or not ls or ls <= 0:
            continue
        if "валидация пропсов" in reason.lower() and "typescript" in reason.lower():
            continue
        filtered.append(g)
    return filtered


def build_reviewed_files_audit(
    all_files_info: List[Dict[str, Any]],
    code_findings: List[Dict[str, Any]],
    test_gaps: List[Dict[str, Any]],
    changed_symbols: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Формирует обязательный список всех отрецензированных файлов со статусом clean или has_findings."""
    findings_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for f in code_findings:
        ff = f.get('file')
        if ff:
            norm_f = ff.replace('\\', '/')
            findings_by_file.setdefault(norm_f, []).append(f)

    gaps_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for g in test_gaps:
        gf = g.get('file')
        if gf:
            norm_g = gf.replace('\\', '/')
            gaps_by_file.setdefault(norm_g, []).append(g)

    reviewed: List[Dict[str, Any]] = []
    for finfo in all_files_info:
        f_p = finfo["path"]
        r_id = finfo.get("repo_id")
        ch_type = finfo.get("change_type", "modified")
        old_p = finfo.get("old_path")
        similarity = finfo.get("similarity", 100)

        file_findings = findings_by_file.get(f_p, [])
        file_gaps = gaps_by_file.get(f_p, [])

        is_generated = f_p.endswith('.Designer.cs') or '.generated.' in f_p.lower() or f_p.endswith('.g.cs')

        if is_generated:
            reviewed.append({
                "file": f_p,
                "repo_id": r_id,
                "change_type": ch_type,
                "old_path": old_p,
                "similarity": similarity,
                "status": "clean",
                "findings_count": 0,
                "status_detail": "Сгенерированный файл (*.Designer.cs), не подлежит ревью"
            })
        elif ch_type == "deleted":
            reviewed.append({
                "file": f_p,
                "repo_id": r_id,
                "change_type": "deleted",
                "old_path": None,
                "similarity": 0,
                "status": "clean",
                "findings_count": 0,
                "status_detail": "Файл удалён"
            })
        elif file_findings:
            reasons = "; ".join(f.get("reason", "Замечание") for f in file_findings)
            reviewed.append({
                "file": f_p,
                "repo_id": r_id,
                "change_type": ch_type,
                "old_path": old_p,
                "similarity": similarity,
                "status": "has_findings",
                "findings_count": len(file_findings),
                "status_detail": f"Есть находки: {reasons}",
                "findings": file_findings
            })
        elif file_gaps:
            reviewed.append({
                "file": f_p,
                "repo_id": r_id,
                "change_type": ch_type,
                "old_path": old_p,
                "similarity": similarity,
                "status": "clean",
                "findings_count": 0,
                "status_detail": "Проверен, замечаний нет (отсутствует прямое покрытие тестами)",
                "test_gaps_count": len(file_gaps)
            })
        else:
            if ch_type == "renamed":
                detail = f"Переименован из {Path(old_p).name} (сходство {similarity}%), проверен, замечаний нет"
            elif any(pat in f_p.lower() for pat in ('.spec.', '.test.', 'tests.cs', 'test.cs')):
                detail = "Тестовый файл проверен, замечаний нет"
            else:
                detail = "Проверен, замечаний нет"

            reviewed.append({
                "file": f_p,
                "repo_id": r_id,
                "change_type": ch_type,
                "old_path": old_p,
                "similarity": similarity,
                "status": "clean",
                "findings_count": 0,
                "status_detail": detail
            })
    return reviewed
