"""
Bounded Context: Code Review & Quality Audit.
Чистая доменная логика аудита чек-листов, расчета рисков и выявления тестовых пробелов.
"""

from .risk_model import calculate_transparent_risk_score

__all__ = ["calculate_transparent_risk_score"]
