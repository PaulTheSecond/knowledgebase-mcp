# KnowledgeBase MCP Priority in Antigravity

When tackling development, codebase navigation, and refactoring tasks, always prioritize KnowledgeBase MCP tools:
- `knowledge_search` — semantic and hybrid search across the knowledge base and codebase.
- `knowledge_find_symbol` — fast navigation to declarations of classes, methods, and interfaces.
- `knowledge_get_callers` / `knowledge_get_callees` — call graph and dependency analysis.
- `knowledge_review_context` — blast radius assessment, risk analysis, and test gap evaluation before making modifications.

Before implementing any task, invoke `knowledge_get_project_rules(topic=...)` to load relevant project guidelines, architectural invariants, and testing standards.
