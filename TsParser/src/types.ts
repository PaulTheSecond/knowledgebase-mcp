// Описание типов данных для TypeScript Compiler API парсера (ts-parser)

export type SymbolKind =
  | 'component'
  | 'styled_component'
  | 'hook'
  | 'function'
  | 'method'
  | 'class'
  | 'interface'
  | 'type'
  | 'enum'
  | 'enum_member'
  | 'variable'
  | 'constant'
  | 'property'
  | 'test';

export type EdgeKind =
  | 'CALLS'
  | 'USES'
  | 'INHERITS'
  | 'IMPLEMENTS'
  | 'IMPORTS'
  | 'TESTED_BY';

/**
 * Описание извлеченного символа (совместимо с моделью symbols в knowledgebase-mcp).
 */
export interface SymbolInfo {
  ast_node_id: number;
  file_path: string;
  name: string;
  qualified_name: string;
  kind: SymbolKind;
  signature: string;
  language: 'typescript' | 'javascript';
  body: string;
  line_start: number; // 1-based
  line_end: number;   // 1-based
  is_test?: boolean;
}

/**
 * Описание семантической связи между символами (вызов, использование в JSX, наследование).
 */
export interface EdgeInfo {
  source_ast_id: number;
  target_qualified_name: string;
  target_file_path?: string;
  kind: EdgeKind;
}

/**
 * Итоговый JSON-результат работы ts-parser (совместим со структурой RoslynParser).
 */
export interface ParserOutput {
  symbols: SymbolInfo[];
  edges: EdgeInfo[];
  metrics?: {
    files_analyzed: number;
    symbols_found: number;
    edges_found: number;
    duration_ms: number;
  };
}

/**
 * Опции CLI и конфигурации парсера.
 */
export interface ParserOptions {
  projectPath: string;
  rootPath: string;
  targetFiles?: Set<string>;
  mode: 'full' | 'diff';
}
