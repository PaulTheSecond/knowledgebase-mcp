# ADR-009: Compiler-Based Semantic Graph Analysis for TypeScript, React, and Angular (ts-parser)

| Metadata | Value |
| :--- | :--- |
| **Status** | Implemented (v1.9.0) |
| **Date** | 2026-11-02 |
| **Affects** | Core Indexer Pipeline, Code Review Graph (`knowledge_review_context`), Semantic Enrichers, Docker Container Distribution |
| **Related To** | [ADR-002: Hybrid Semantic and Graph Indexing](adr-002-hybrid-semantic-graph-indexing.md), [ADR-008: Multi-Worktree CAS & Hybrid Sync](adr-008-multi-worktree-cas-and-hybrid-sync.md) |
| **Breaking Change** | No (100% Backward Compatible with Graceful Fallback) |

---

## 1. Context and Problem Statement

In version v1.8 of `knowledgebase-mcp`, the `knowledge_review_context` tool was introduced to analyze code modifications, calculate blast radius, trace call graphs, evaluate change risks, and identify affected automated tests.

For C# / .NET codebases, semantic analysis is powered by the Roslyn compiler engine (`Microsoft.CodeAnalysis`), delivering 100% precision for type hierarchies, method invocations, interface implementations, and dependency injection graphs.

However, for the TypeScript, JavaScript, and React ecosystem, analysis was initially performed using a lightweight Tree-sitter syntax parser. Field testing on enterprise codebases (such as the Orbita frontend) revealed fundamental limitations of purely syntactic parsing:

1. **Missing Call Semantics in JSX:** Component invocations in JSX templates (e.g., `<ApproverMatrixSelectionBar />`) are parsed by Tree-sitter as generic markup syntax tags rather than function/class invocations. This breaks caller-callee chains when evaluating blast radius and downstream dependents.
2. **Path Aliases and Module Resolution (`tsconfig.json paths`):** Enterprise codebases extensively use module aliases (`@components/*`, `@shared/*`), barrel re-exports (`index.ts`), and monorepo packages. A purely syntactic parser cannot resolve the true source file without invoking the TypeScript Language Service and module resolution algorithms.
3. **External Dependency Noise:** Heuristic text search without compiler-grade type resolution risks pulling in external symbols or falsely linking tests with matching prefixes (e.g., matching C# backend tests with a similarly named React component).

---

## 2. Decision

Design and implement a dedicated containerized compiler analyzer, **`ts-parser`**, as a symmetric companion to `RoslynParser` (.NET), executed under Node.js inside the server Docker container.

### 2.1. Architecture and Container Isolation
- **Containerized Environment:** The Node.js execution runtime (v20+ LTS) and the compiled bundle `dist/ts-parser.js` are packaged **strictly inside the Docker image** of the `knowledgebase-mcp` service.
- **Zero Client Footprint:** Developers, local IDEs, and AI agents (Claude Code, Antigravity, Cursor) **do not require** Node.js or specialized toolchains installed on their host machine. All interactions occur transparently through the MCP protocol.
- **Multi-Stage Dockerfile Stages:**
  - Stage 1: Build `RoslynParser` (.NET 8 SDK).
  - Stage 2: Build `ts-parser` (`esbuild` compiles a self-contained single-file JS bundle without external `node_modules`).
  - Stage 3: Runtime image with Python 3.11, .NET 8 runtime, and Node.js runtime.

### 2.2. Operation Modes and CLI Contract
`ts-parser` is invoked by Python semantic enricher services in two operational modes:

```bash
node /app/TsParser/dist/ts-parser.js --project <path/to/tsconfig.json> [options]

Parameters:
  --project <path>    Path to project tsconfig.json / jsconfig.json (required)
  --files <path>      File containing relative/absolute file paths or CSV list (for targeted PR diffs)
  --root <path>       Repository root directory (defaults to tsconfig directory)
  --mode <full|diff>  Mode: 'full' for full repository indexing, 'diff' for incremental PR analysis
```

#### JSON Output Contract (stdout)
The output JSON schema is strictly aligned with `RoslynParser` and the SQLite database schema:

```json
{
  "symbols": [
    {
      "ast_node_id": 1,
      "file_path": "/app/frontend/src/components/UserCard.tsx",
      "name": "UserCard",
      "qualified_name": "UserCard",
      "kind": "component",
      "signature": "export const UserCard: React.FC<UserCardProps>",
      "language": "typescript",
      "body": "export const UserCard = ...",
      "line_start": 10,
      "line_end": 28,
      "is_test": false
    }
  ],
  "edges": [
    {
      "source_ast_id": 1,
      "target_qualified_name": "Button",
      "target_file_path": "/app/frontend/src/components/Button.tsx",
      "kind": "USES"
    }
  ],
  "metrics": {
    "files_analyzed": 1,
    "symbols_found": 1,
    "edges_found": 1,
    "duration_ms": 13
  }
}
```

### 2.3. 100% Isolation from `node_modules` and Third-Party Packages
To prevent external package noise from contaminating the repository graph:
- For every AST node, the physical declaration path is extracted via the TypeScript `TypeChecker` (`Declaration.getSourceFile().fileName`).
- If the path contains `/node_modules/`, standard libraries (`/typescript/lib/`), or resides outside the project root (`!targetFile.startsWith(projectRoot)`), the relation is excluded from the graph.
- The resulting graph contains **strictly in-repository symbols and edges between first-party source files**.

### 2.4. Performance and Compiler Optimization
To ensure minimal compilation overhead:
- `noEmit: true` is enforced (no JS output files written to disk; operations occur entirely in memory).
- `skipLibCheck: true` is enabled (skips type checking in `.d.ts` declaration files, providing a 5–10x speedup).
- Compiler incremental program caching is leveraged (`incremental: true`).
- **Sub-Second PR Diff Mode:** During pull request review, only modified files and their immediate consumers are analyzed (response latency: 15–30 ms for 10 files).

### 2.5. Extensibility to Angular (Ivy) and Vue (Volar)
The parser architecture is modular and extensible:
- **React / TypeScript:** Standard TypeScript Compiler API (`ts.createProgram`, `ts.TypeChecker`).
- **Angular:** Integration via `@angular/compiler-cli` (`AngularCompilerHost`), translating HTML component templates into virtual TypeScript (`.ngtypecheck.ts`) and mapping directives/tags to typed method and class invocations.
- **Vue:** Integration via `@volar/language-core` for virtual Single File Component (`.vue`) decomposition.

### 2.6. Graceful Fallback
If a target repository lacks a `tsconfig.json` or `jsconfig.json` (e.g., pure Python, Go, or unconfigured JavaScript), the indexer transparently falls back to the Tree-sitter syntax parser without throwing errors or halting synchronization.

---

## 3. Consequences

### Positive
- **100% Call Graph Accuracy:** Full resolution of React JSX component hierarchies, custom hooks, and TypeScript type inheritances.
- **Continuous Component Chains:** Eliminates broken caller-callee chains in PR code reviews and blast radius calculations.
- **Unified Contract:** Symmetric architecture between `RoslynParser` (.NET) and `ts-parser` (TS/JS) within the `Two-Phase Strategy Pipeline`.
- **Zero Regex Fragility:** Replaces brittle regexes and heuristics with official compiler APIs.

### Negative / Trade-offs
- **Container Footprint:** Incremental increase in Docker image size to include the Node.js runtime and bundled parser (~25 MB).
- **Maintenance:** Maintenance of TypeScript Compiler API integrations across major TypeScript version bumps.
