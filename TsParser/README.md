# ts-parser: Compiler-Based Semantic Analyzer for TypeScript, React, and Angular

A dedicated compiler-grade semantic dependency analyzer for the TypeScript, JavaScript, and React ecosystem in `knowledgebase-mcp` (per [ADR-009](../docs/adr-009-compiler-semantic-analysis-ts-angular.md)).

`ts-parser` is the direct counterpart to `RoslynParser` (.NET) for the JavaScript/TypeScript ecosystem: it replaces heuristic syntax parsing (Tree-sitter) with authoritative compiler analysis via the **TypeScript Compiler API**.

---

## 🎯 Key Capabilities

1. **JSX Call & Dependency Semantics:** JSX component usages such as `<ApproverMatrixSelectionBar />` are linked to component declarations as `USES` relationships, eliminating broken component dependency chains in blast radius and impact analysis.
2. **Path Aliases (`tsconfig.json paths`):** Transparent resolution of path aliases (`@components/*`, `@shared/*`), barrel file re-exports (`index.ts`), and multi-package monorepos.
3. **100% Isolation from `node_modules`:** Only in-repository symbols and calls are indexed; third-party dependencies (`node_modules`) and runtime typings (`lib.d.ts`) are physically filtered out by declaration source file paths.
4. **Sub-Second PR Diff Mode:** Operates in memory (`noEmit: true`, `skipLibCheck: true`), delivering sub-second response times (~15–30 ms for 10 files) for pull request reviews.

---

## 🚀 CLI Contract

```bash
node dist/ts-parser.js --project <path/to/tsconfig.json> [options]

Parameters:
  --project <path>    Path to tsconfig.json / jsconfig.json (required)
  --root <path>       Repository root directory (defaults to tsconfig directory)
  --files <path>      File containing relative/absolute file paths or CSV list (for targeted PR diffs)
  --mode <full|diff>  Analysis mode: 'full' for complete indexing, 'diff' for incremental analysis
  --help, -h          Show help message
```

### JSON Output Schema (stdout)

Fully compatible with `RoslynParser` and the SQLite schema of `knowledgebase-mcp`:

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

---

## 🛠️ Building Standalone Bundle

The module is built with `esbuild` into a single standalone bundle, `dist/ts-parser.js` (~9.5 MB), bundling the TypeScript Compiler API. External `node_modules` are **not required** at runtime:

```bash
npm run build    # Compile dist/ts-parser.js via esbuild
npm test         # Run automated tests (node:test)
npm run lint     # Strict type checking (tsc --noEmit)
```

---

## 📦 Containerization & Zero Host Footprint

- The Node.js runtime and built `dist/ts-parser.js` bundle are shipped **strictly inside the Docker image** (via multi-stage build).
- End users and AI agents (Claude Code, Antigravity, Cursor) **do not require** Node.js installed on the host machine.
