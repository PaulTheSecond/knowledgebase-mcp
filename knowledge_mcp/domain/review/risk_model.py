"""
Доменная модель оценки рисков изменений (Risk Score Calculator).
Реализует прозрачную непрерывную математическую шкалу (0..100)
на основе сложности, радиуса влияния, пробелов в тестах и смягчающих факторов.
"""

from typing import Any, Dict, List


def calculate_transparent_risk_score(
    norm_changed_files: List[str],
    changed_symbols: List[Dict[str, Any]],
    unique_file_impacts: Dict[str, Any],
    component_consumer_files: List[str],
    style_only_files: List[str],
    type_only_files: List[str],
    test_gaps: List[Dict[str, Any]],
    e2e_covered_symbols: List[Dict[str, Any]],
    related_tests: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Рассчитывает нормированный скоринг риска (0-100), учитывающий долю покрытых символов, масштаб и глубину влияния."""
    logic_symbols = [s for s in changed_symbols if s.get('kind', '').lower() in ('class', 'function', 'component', 'method')]
    decl_symbols = [s for s in changed_symbols if s.get('kind', '').lower() not in ('class', 'function', 'component', 'method')]

    # 1. Сложность и объем изменений (0..25) с плавной непрерывной шкалой
    file_points = min(12, int(len(norm_changed_files) * 0.4))
    sym_points = min(13, int(len(logic_symbols) * 0.6 + len(decl_symbols) * 0.1))
    complexity_score = min(25, file_points + sym_points)

    # 2. Радиус поражения / зависимости (0..30)
    direct_consumer_count = len(component_consumer_files)
    style_consumer_count = len(style_only_files) + len(type_only_files)
    direct_pts = min(20, int(direct_consumer_count * 1.5))
    sec_pts = min(10, int(style_consumer_count * 0.5))
    blast_score = min(30, direct_pts + sec_pts)

    # 3. Риск отсутствия тестов (0..40), основанный на доле непокрытых символов
    total_logic = len(logic_symbols)
    uncovered_count = len([g for g in test_gaps if g.get('kind') != 'migration_script'])
    if total_logic > 0:
        covered_count = max(0, total_logic - uncovered_count)
        coverage_ratio = covered_count / total_logic
        uncovered_ratio = 1.0 - coverage_ratio
        base_gap_pts = int(uncovered_ratio * 28)
    else:
        coverage_ratio = 0.0 if uncovered_count > 0 else 1.0
        base_gap_pts = 20 if uncovered_count > 0 else 0

    gap_count_penalty = min(12, len(test_gaps) * 2)
    test_gap_score = min(40, base_gap_pts + gap_count_penalty)

    # 4. Смягчающие факторы (0..25): наличие тестов и высокий процент покрытия
    test_mitigation = min(15, len(related_tests) * 2)
    cov_bonus = int(coverage_ratio * 10)
    mitigation_score = min(25, test_mitigation + cov_bonus)

    if not norm_changed_files and not changed_symbols:
        risk_score = 0
        risk_level = "low"
    else:
        raw_score = complexity_score + blast_score + test_gap_score - mitigation_score
        risk_score = max(5, min(100, raw_score))
        if test_gaps and risk_score < 30:
            risk_score = 30
        if risk_score < 30:
            risk_level = "low"
        elif risk_score < 70:
            risk_level = "medium"
        else:
            risk_level = "high"

    cov_pct = int(coverage_ratio * 100)
    explanations = [
        f"Сложность изменений ({complexity_score}/25): {len(norm_changed_files)} изменённых файлов, {len(logic_symbols)} ключевых символов",
        f"Радиус поражения ({blast_score}/30): {len(unique_file_impacts)} уникальных файлов ({direct_consumer_count} потребителей в разметке, {style_consumer_count} стили/типы)",
        f"Пробелы в тестах ({test_gap_score}/40): {len(test_gaps)} непокрытых функций/режимов (доля покрытия символов: {cov_pct}%)"
    ]
    if mitigation_score > 0:
        explanations.append(f"Смягчающие факторы (-{mitigation_score}/25): {len(related_tests)} связанных тестов, бонус покрытия {cov_bonus} б.")

    return {
        "formula": "RiskScore = min(100, max(0, Complexity(0..25) + BlastRadius(0..30) + TestGaps(0..40) - Mitigations(0..25)))",
        "score": risk_score,
        "level": risk_level,
        "thresholds": {
            "low": "0-29",
            "medium": "30-69",
            "high": "70-100"
        },
        "components": {
            "complexity": complexity_score,
            "blast_radius": blast_score,
            "test_gaps": test_gap_score,
            "mitigations": mitigation_score
        },
        "coverage_percent": cov_pct,
        "explanations": explanations
    }
