// Ядро компиляторного семантического анализатора на базе TypeScript Compiler API (ADR-009)

import * as ts from 'typescript';
import * as fs from 'fs';
import * as path from 'path';
import {
  ParserOptions,
  ParserOutput,
  SymbolInfo,
  EdgeInfo,
  SymbolKind,
  EdgeKind
} from './types.js';
import {
  normalizePath,
  isInsideRoot,
  isExternalOrLibrary,
  isValidProjectDeclaration
} from './filters.js';
import {
  getLanguageForFile,
  extractSignature,
  getNodeLineSpan,
  classifyFunctionOrVariable,
  isPascalCase
} from './ast-utils.js';

export class TypeScriptSemanticAnalyzer {
  private options: ParserOptions;
  private configDir!: string;
  private program!: ts.Program;
  private typeChecker!: ts.TypeChecker;
  private currentAstId: number = 1;
  private symbols: SymbolInfo[] = [];
  private edges: EdgeInfo[] = [];

  constructor(options: ParserOptions) {
    this.options = options;
  }

  /**
   * Инициализирует компилятор TypeScript и создает программу (ts.Program).
   */
  public initialize(): void {
    const configPath = path.resolve(this.options.projectPath);
    if (!fs.existsSync(configPath)) {
      throw new Error(`tsconfig.json не найден по пути: ${configPath}`);
    }

    this.configDir = path.dirname(configPath);

    // Читаем конфигурацию tsconfig.json
    const configFile = ts.readConfigFile(configPath, ts.sys.readFile);
    if (configFile.error) {
      const msg = ts.flattenDiagnosticMessageText(configFile.error.messageText, '\n');
      throw new Error(`Ошибка чтения tsconfig.json: ${msg}`);
    }

    const parsedConfig = ts.parseJsonConfigFileContent(
      configFile.config,
      ts.sys,
      this.configDir
    );

    // Оптимизация производительности согласно ADR-009 §4:
    // noEmit: true, skipLibCheck: true, skipDefaultLibCheck: true
    const compilerOptions: ts.CompilerOptions = {
      ...parsedConfig.options,
      noEmit: true,
      skipLibCheck: true,
      skipDefaultLibCheck: true,
      allowJs: true,
      jsx: parsedConfig.options.jsx || ts.JsxEmit.ReactJSX
    };

    // Создаем экземпляр программы и TypeChecker
    this.program = ts.createProgram({
      rootNames: parsedConfig.fileNames,
      options: compilerOptions
    });
    this.typeChecker = this.program.getTypeChecker();
  }

  /**
   * Запускает полный или дифференциальный семантический анализ.
   */
  public analyze(): ParserOutput {
    const startTime = Date.now();
    const sourceFiles = this.program.getSourceFiles();

    let filesToProcess: ts.SourceFile[] = [];

    for (const sf of sourceFiles) {
      const absPath = path.isAbsolute(sf.fileName)
        ? sf.fileName
        : path.resolve(this.configDir, sf.fileName);
      const normFileName = normalizePath(absPath);

      // 100% изоляция: отсекаем node_modules, тайпинги и внешние файлы
      if (isExternalOrLibrary(normFileName)) {
        continue;
      }
      if (!isInsideRoot(normFileName, this.options.rootPath)) {
        continue;
      }

      // В режиме diff обрабатываем только указанные измененные файлы
      if (this.options.targetFiles && this.options.targetFiles.size > 0) {
        if (!this.options.targetFiles.has(normFileName)) {
          continue;
        }
      }

      filesToProcess.push(sf);
    }

    // Обходим отобранные исходные файлы
    for (const sf of filesToProcess) {
      this.analyzeSourceFile(sf);
    }

    const duration = Date.now() - startTime;

    return {
      symbols: this.symbols,
      edges: this.edges,
      metrics: {
        files_analyzed: filesToProcess.length,
        symbols_found: this.symbols.length,
        edges_found: this.edges.length,
        duration_ms: duration
      }
    };
  }

  /**
   * Анализирует отдельный файл исходного кода: собирает AST-символы и связывает их.
   */
  private analyzeSourceFile(sourceFile: ts.SourceFile): void {
    const absPath = path.isAbsolute(sourceFile.fileName)
      ? sourceFile.fileName
      : path.resolve(this.configDir, sourceFile.fileName);
    const filePath = normalizePath(absPath);
    const language = getLanguageForFile(filePath);

    // Стек активных родительских символов для контекста вызовов и квалифицированных имен
    const scopeStack: { id: number; qualifiedName: string }[] = [];

    const getCurrentScope = () => {
      return scopeStack.length > 0 ? scopeStack[scopeStack.length - 1] : null;
    };

    const registerSymbol = (
      name: string,
      kind: SymbolKind,
      node: ts.Node,
      customSignature?: string,
      isTest: boolean = false
    ): number => {
      const astId = this.currentAstId++;
      const currentScope = getCurrentScope();
      const qualifiedName = currentScope ? `${currentScope.qualifiedName}.${name}` : name;
      const signature = customSignature || extractSignature(node, sourceFile);
      const span = getNodeLineSpan(node, sourceFile);
      const body = node.getText(sourceFile);

      this.symbols.push({
        ast_node_id: astId,
        file_path: filePath,
        name,
        qualified_name: qualifiedName,
        kind,
        signature,
        language,
        body,
        line_start: span.startLine,
        line_end: span.endLine,
        is_test: isTest
      });

      return astId;
    };

    const addEdge = (
      sourceAstId: number,
      targetName: string,
      targetFilePath: string | undefined,
      kind: EdgeKind
    ) => {
      if (!sourceAstId || !targetName) return;
      this.edges.push({
        source_ast_id: sourceAstId,
        target_qualified_name: targetName,
        target_file_path: targetFilePath,
        kind
      });
    };

    // Рекурсивный обход узлов файла
    const visit = (node: ts.Node) => {
      let registeredId: number | null = null;
      let registeredQualifiedName: string | null = null;

      // ── 1. Объявления классов ──────────────────────────────────────────────
      if (ts.isClassDeclaration(node) && node.name) {
        const className = node.name.text;
        // Детекция React-компонента по наследованию React.Component
        let kind: SymbolKind = 'class';
        if (node.heritageClauses) {
          for (const clause of node.heritageClauses) {
            const text = clause.getText(sourceFile);
            if (text.includes('Component') || text.includes('PureComponent')) {
              kind = 'component';
              break;
            }
          }
        }

        registeredId = registerSymbol(className, kind, node);
        registeredQualifiedName = getCurrentScope()
          ? `${getCurrentScope()!.qualifiedName}.${className}`
          : className;

        // Обработка extends и implements через HeritageClauses
        this.resolveHeritageClauses(node, registeredId, addEdge);
      }

      // ── 2. Объявления функций ──────────────────────────────────────────────
      else if (ts.isFunctionDeclaration(node) && node.name) {
        const fnName = node.name.text;
        const kind = classifyFunctionOrVariable(fnName, node.body, node.type, sourceFile);
        const isTest = fnName.startsWith('test_') || fnName.endsWith('Test') || fnName.endsWith('Tests');

        registeredId = registerSymbol(fnName, isTest ? 'test' : kind, node, undefined, isTest);
        registeredQualifiedName = getCurrentScope()
          ? `${getCurrentScope()!.qualifiedName}.${fnName}`
          : fnName;
      }

      // ── 3. Методы классов и геттеры/сеттеры ────────────────────────────────
      else if (ts.isMethodDeclaration(node) && ts.isIdentifier(node.name)) {
        const methodName = node.name.text;
        const kind: SymbolKind = 'method';
        registeredId = registerSymbol(methodName, kind, node);
        registeredQualifiedName = getCurrentScope()
          ? `${getCurrentScope()!.qualifiedName}.${methodName}`
          : methodName;
      }

      // ── 4. Интерфейсы и типы ──────────────────────────────────────────────
      else if (ts.isInterfaceDeclaration(node)) {
        const ifaceName = node.name.text;
        registeredId = registerSymbol(ifaceName, 'interface', node);
        registeredQualifiedName = getCurrentScope()
          ? `${getCurrentScope()!.qualifiedName}.${ifaceName}`
          : ifaceName;

        this.resolveHeritageClauses(node, registeredId, addEdge);
      } else if (ts.isTypeAliasDeclaration(node)) {
        const typeName = node.name.text;
        registeredId = registerSymbol(typeName, 'type', node);
        registeredQualifiedName = getCurrentScope()
          ? `${getCurrentScope()!.qualifiedName}.${typeName}`
          : typeName;
      } else if (ts.isEnumDeclaration(node)) {
        const enumName = node.name.text;
        registeredId = registerSymbol(enumName, 'enum', node);
        registeredQualifiedName = getCurrentScope()
          ? `${getCurrentScope()!.qualifiedName}.${enumName}`
          : enumName;
      }

      // ── 5. Переменные (React функциональные компоненты, хуки, константы) ──
      else if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name)) {
        const varName = node.name.text;
        const isTopLevel = !getCurrentScope();

        // Регистрируем функции, компоненты, константы и styled-компоненты
        if (node.initializer) {
          const isFn =
            ts.isArrowFunction(node.initializer) || ts.isFunctionExpression(node.initializer);

          if (isFn || isTopLevel) {
            const kind = classifyFunctionOrVariable(
              varName,
              node.initializer,
              node.type,
              sourceFile
            );

            registeredId = registerSymbol(varName, kind, node);
            registeredQualifiedName = getCurrentScope()
              ? `${getCurrentScope()!.qualifiedName}.${varName}`
              : varName;
          }
        }
      }

      // ── 6. Тестовые блоки describe / it / test ─────────────────────────────
      else if (ts.isCallExpression(node) && ts.isIdentifier(node.expression)) {
        const fnName = node.expression.text;
        if (['describe', 'it', 'test'].includes(fnName) && node.arguments.length > 0) {
          const firstArg = node.arguments[0];
          if (ts.isStringLiteral(firstArg) || ts.isNoSubstitutionTemplateLiteral(firstArg)) {
            const testTitle = firstArg.text;
            registeredId = registerSymbol(
              testTitle,
              'test',
              node,
              `${fnName}("${testTitle}")`,
              true
            );
            registeredQualifiedName = getCurrentScope()
              ? `${getCurrentScope()!.qualifiedName}.${testTitle}`
              : testTitle;
          }
        }
      }

      // Если объявлен новый контекстный символ — помещаем его в стек скоупа
      if (registeredId !== null && registeredQualifiedName !== null) {
        scopeStack.push({ id: registeredId, qualifiedName: registeredQualifiedName });
      }

      // ── 7. Семантический резолв вызовов и JSX-тегов через TypeChecker ────
      const currentScope = getCurrentScope();
      if (currentScope) {
        // а) JSX-элементы (<ApproverMatrixSelectionBar />, <CustomButton />)
        if (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) {
          this.resolveJsxElement(node, currentScope.id, addEdge);
        }
        // б) Обычные вызовы функций и методов (service.fetch(), calculate())
        else if (ts.isCallExpression(node)) {
          this.resolveCallExpression(node, currentScope.id, addEdge);
        }
        // в) Создание объектов (new OrderManager())
        else if (ts.isNewExpression(node)) {
          this.resolveNewExpression(node, currentScope.id, addEdge);
        }
      }

      // Рекурсивный обход детей
      ts.forEachChild(node, visit);

      // Выходим из скоупа текущего символа
      if (registeredId !== null) {
        scopeStack.pop();
      }
    };

    ts.forEachChild(sourceFile, visit);
  }

  /**
   * Разрешает JSX-элемент через TypeChecker в физический символ и декларацию (ADR-009 §1, §3).
   */
  private resolveJsxElement(
    element: ts.JsxOpeningElement | ts.JsxSelfClosingElement,
    sourceAstId: number,
    addEdge: (sourceAstId: number, targetName: string, targetFilePath: string | undefined, kind: EdgeKind) => void
  ): void {
    const tagNameNode = element.tagName;

    // Игнорируем стандартные HTML теги (div, span, button, a и т.д.), они начинаются с маленькой буквы
    const tagText = tagNameNode.getText();
    if (!tagText || !isPascalCase(tagText.split('.')[0])) {
      return;
    }

    const symbol = this.typeChecker.getSymbolAtLocation(tagNameNode);
    if (!symbol) return;

    const resolved = this.resolveSymbolDeclaration(symbol);
    if (!resolved) return;

    addEdge(sourceAstId, resolved.name, resolved.filePath, 'USES');
  }

  /**
   * Разрешает вызов функции/метода через TypeChecker.
   */
  private resolveCallExpression(
    callExpr: ts.CallExpression,
    sourceAstId: number,
    addEdge: (sourceAstId: number, targetName: string, targetFilePath: string | undefined, kind: EdgeKind) => void
  ): void {
    const expr = callExpr.expression;
    // Пропускаем тесты describe/it/test чтобы не дублировать
    if (ts.isIdentifier(expr) && ['describe', 'it', 'test'].includes(expr.text)) {
      return;
    }

    const symbol = this.typeChecker.getSymbolAtLocation(expr);
    if (!symbol) return;

    const resolved = this.resolveSymbolDeclaration(symbol);
    if (!resolved) return;

    addEdge(sourceAstId, resolved.name, resolved.filePath, 'CALLS');
  }

  /**
   * Разрешает создание экземпляра класса (new Foo()).
   */
  private resolveNewExpression(
    newExpr: ts.NewExpression,
    sourceAstId: number,
    addEdge: (sourceAstId: number, targetName: string, targetFilePath: string | undefined, kind: EdgeKind) => void
  ): void {
    const expr = newExpr.expression;
    const symbol = this.typeChecker.getSymbolAtLocation(expr);
    if (!symbol) return;

    const resolved = this.resolveSymbolDeclaration(symbol);
    if (!resolved) return;

    addEdge(sourceAstId, resolved.name, resolved.filePath, 'CALLS');
  }

  /**
   * Разрешает наследование и реализацию интерфейсов (extends / implements).
   */
  private resolveHeritageClauses(
    node: ts.ClassDeclaration | ts.InterfaceDeclaration,
    sourceAstId: number,
    addEdge: (sourceAstId: number, targetName: string, targetFilePath: string | undefined, kind: EdgeKind) => void
  ): void {
    if (!node.heritageClauses) return;

    for (const clause of node.heritageClauses) {
      const edgeKind: EdgeKind =
        clause.token === ts.SyntaxKind.ExtendsKeyword ? 'INHERITS' : 'IMPLEMENTS';

      for (const typeNode of clause.types) {
        const symbol = this.typeChecker.getSymbolAtLocation(typeNode.expression);
        if (symbol) {
          const resolved = this.resolveSymbolDeclaration(symbol);
          if (resolved) {
            addEdge(sourceAstId, resolved.name, resolved.filePath, edgeKind);
          }
        }
      }
    }
  }

  /**
   * Разрешает ts.Symbol с разворачиванием псевдонимов (алиасов/re-exports)
   * и 100% проверкой изоляции от node_modules и системных библиотек.
   */
  private resolveSymbolDeclaration(
    symbol: ts.Symbol
  ): { name: string; filePath: string } | null {
    let target = symbol;

    // Если символ является alias (import), разворачиваем его до исходного объявления
    if (target.flags & ts.SymbolFlags.Alias) {
      try {
        const aliased = this.typeChecker.getAliasedSymbol(target);
        if (aliased) {
          target = aliased;
        }
      } catch {
        // Защита от циклических или неразрешимых ссылок
      }
    }

    const declarations = target.getDeclarations();
    if (!declarations || declarations.length === 0) {
      return null;
    }

    // Ищем декларацию внутри проекта (гарантия изоляции от node_modules)
    for (const decl of declarations) {
      if (isValidProjectDeclaration(decl, this.options.rootPath)) {
        const sourceFile = decl.getSourceFile();
        const normFilePath = normalizePath(sourceFile.fileName);
        return {
          name: target.getName(),
          filePath: normFilePath
        };
      }
    }

    return null;
  }
}
