"""Модуль расчета экономии контекста токенов."""

import json
from typing import Any, Dict, List, Optional


def calculate_savings(
    returned_content: Any,
    files_involved: List[str],
    db_instance: Optional[Any] = None,
    default_file_tokens: int = 1500,
    min_multiplier: float = 3.0
) -> Dict[str, int]:
    """Рассчитывает сэкономленные токены контекста при точечном ответе вместо чтения файлов целиком."""
    if isinstance(returned_content, str):
        returned_tokens = max(1, len(returned_content) // 4)
    elif isinstance(returned_content, (list, dict)):
        returned_tokens = max(1, len(json.dumps(returned_content, ensure_ascii=False)) // 4)
    else:
        returned_tokens = 50

    baseline_tokens = 0
    unique_files = set(f for f in files_involved if f)
    if db_instance and unique_files:
        try:
            cursor = db_instance.conn.cursor()
            for f_path in unique_files:
                cursor.execute("""
                    SELECT fc.size
                    FROM files f
                    JOIN file_contents fc ON f.content_hash = fc.hash
                    WHERE f.path = ?
                    LIMIT 1
                """, (f_path,))
                row = cursor.fetchone()
                if row and row['size']:
                    baseline_tokens += max(300, row['size'] // 4)
                else:
                    baseline_tokens += default_file_tokens
        except Exception:
            baseline_tokens = len(unique_files) * default_file_tokens
    else:
        baseline_tokens = max(default_file_tokens, len(unique_files) * default_file_tokens)

    if baseline_tokens <= returned_tokens:
        baseline_tokens = int(returned_tokens * min_multiplier)

    saved_tokens = max(0, baseline_tokens - returned_tokens)
    saved_percent = round((saved_tokens / baseline_tokens * 100)) if baseline_tokens > 0 else 0

    return {
        "saved_tokens": saved_tokens,
        "saved_percent": saved_percent,
        "baseline_tokens": baseline_tokens,
        "returned_tokens": returned_tokens
    }
