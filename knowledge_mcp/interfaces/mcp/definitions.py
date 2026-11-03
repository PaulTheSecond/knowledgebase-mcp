"""Определения схем MCP инструментов (Tool Definitions)."""

from typing import List
import mcp.types as types

TOOL_SCHEMAS: List[types.Tool] = [
    types.Tool(
        name="knowledge_search",
        description=(
            "Search the local knowledge base across multiple repositories. "
            "Uses triple hybrid search (Full-Text + Vector Embeddings + Graph routing) for high recall. "
            "Returns relevant code and documentation chunks with trust levels and context token savings."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query (concept, code snippet, or question)."},
                "repo_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of repo_ids to restrict the search."
                },
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."},
                "limit": {"type": "integer", "description": "Max results to return (default 10)."}
            },
            "required": ["query"]
        }
    ),
    types.Tool(
        name="knowledge_get_chunk",
        description="Retrieve detailed information and raw content of a specific knowledge base chunk by its ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string", "description": "The unique ID of the chunk."}
            },
            "required": ["chunk_id"]
        }
    ),
    types.Tool(
        name="knowledge_read_file",
        description="Retrieve the complete raw content of a file from the knowledge base by its file path, repo, project, and branch.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The exact path of the file (e.g. 'docs/data-access/README.md')."},
                "repo_id": {"type": "string", "description": "Optional repo_id if the path exists in multiple repositories."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."}
            },
            "required": ["path"]
        }
    ),
    types.Tool(
        name="knowledge_delete_repo",
        description="Delete all knowledge base data associated with a specific repository ID. Delegated to Daemon API.",
        inputSchema={
            "type": "object",
            "properties": {
                "repo_id": {"type": "string", "description": "The unique ID of the repository to delete."}
            },
            "required": ["repo_id"]
        }
    ),
    types.Tool(
        name="knowledge_sync_repo",
        description="Queue a sync (update) of the local knowledge base for a specific repository. Delegated to Daemon API.",
        inputSchema={
            "type": "object",
            "properties": {
                "repo_id": {"type": "string", "description": "The unique ID of the repository (as defined in config.yaml)."}
            },
            "required": ["repo_id"]
        }
    ),
    types.Tool(
        name="knowledge_find_symbol",
        description="Find a code symbol (class, method, interface, etc.) by its name or pattern across branches and projects.",
        inputSchema={
            "type": "object",
            "properties": {
                "name_pattern": {"type": "string", "description": "The name or wildcard pattern of the symbol."},
                "kind": {"type": "string", "description": "Optional kind of symbol (class, method, interface, property, etc)."},
                "repo_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of repo_ids to restrict the search."
                },
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."},
                "limit": {"type": "integer", "description": "Max results to return (default 20)."}
            },
            "required": ["name_pattern"]
        }
    ),
    types.Tool(
        name="knowledge_get_callers",
        description="Find all symbols that call or reference the specified symbol by its numeric ID with branch support.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_id": {"type": "integer", "description": "The exact numeric ID of the target symbol."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."}
            },
            "required": ["symbol_id"]
        }
    ),
    types.Tool(
        name="knowledge_get_callees",
        description="Find all symbols that are called or referenced by the specified symbol by its numeric ID with branch support.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_id": {"type": "integer", "description": "The exact numeric ID of the source symbol."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."}
            },
            "required": ["symbol_id"]
        }
    ),
    types.Tool(
        name="knowledge_get_hierarchy",
        description="Get the inheritance and implementation hierarchy for the specified symbol by its numeric ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_id": {"type": "integer", "description": "The exact numeric ID of the symbol."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."}
            },
            "required": ["symbol_id"]
        }
    ),
    types.Tool(
        name="knowledge_impact_analysis",
        description="Perform a recursive graph traversal to find all symbols and files that depend on the specified symbol (blast radius).",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_id": {"type": "integer", "description": "The exact numeric ID of the target symbol."},
                "max_depth": {"type": "integer", "description": "Max recursion depth (default 3, max 5)."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."}
            },
            "required": ["symbol_id"]
        }
    ),
    types.Tool(
        name="knowledge_review_context",
        description=(
            "Build blast radius, determine impacted callers, find related tests and test gaps, "
            "calculate risk score (0-100), and provide a comprehensive review context for code reviewer agents "
            "across all repositories of the branch/task in the merge-base..HEAD range."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "repo_id": {"type": "string", "description": "Optional repository identifier. If omitted, all repositories for this branch/task are reviewed."},
                "repo_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit list of repository identifiers to review."
                },
                "project_id": {"type": "string", "description": "The project/workspace identifier (default 'default')."},
                "branch": {"type": "string", "description": "The current working branch (default 'master')."},
                "compare_branch": {"type": "string", "description": "The target base branch to compare against (default 'master')."},
                "changed_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit list of modified files. If omitted, discovered via git diff or CAS comparison."
                },
                "include_uncommitted": {"type": "boolean", "description": "Whether to include uncommitted working tree changes (default true)."},
                "max_depth": {"type": "integer", "description": "Max recursion depth for caller blast radius (default 2)."},
                "file_path": {"type": "string", "description": "Optional single file to inspect (auto-resolves repo_id)."}
            },
            "required": []
        }
    ),
    types.Tool(
        name="knowledge_find_tests",
        description="Find all automated tests (C#, TypeScript, Python) associated with a given method, class, or file.",
        inputSchema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "The name of the method, class, or file to find tests for."},
                "repo_id": {"type": "string", "description": "Optional repository ID."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."},
                "branch": {"type": "string", "description": "Optional branch name (default 'master')."}
            },
            "required": ["target"]
        }
    ),
    types.Tool(
        name="knowledge_get_project_rules",
        description="Retrieve project development guidelines, conventions, and rules from CLAUDE.md, .cursorrules, and .kb/, optionally filtered by topic.",
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Optional topic or keyword to filter rules (e.g. 'database', 'testing', 'crlf', 'api')."},
                "project_id": {"type": "string", "description": "Optional project_id (default 'default')."}
            }
        }
    ),
]
