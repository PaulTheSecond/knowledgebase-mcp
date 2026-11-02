"""
Модуль семантических обогатителей кода (ADR-009).
Паттерн Strategy для унифицированного запуска компиляторных парсеров (Roslyn, TypeScript).
"""

from knowledge_mcp.core.enrichers.base import BaseSemanticEnricher
from knowledge_mcp.core.enrichers.roslyn_enricher import RoslynSemanticEnricher
from knowledge_mcp.core.enrichers.typescript_enricher import TypeScriptSemanticEnricher

__all__ = [
    "BaseSemanticEnricher",
    "RoslynSemanticEnricher",
    "TypeScriptSemanticEnricher"
]
