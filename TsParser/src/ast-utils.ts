// Утилиты для анализа TypeScript AST и детекции React компонентов/хуков

import * as ts from 'typescript';
import * as path from 'path';
import { SymbolKind } from './types.js';

/**
 * Определяет язык файла по его расширению.
 */
export function getLanguageForFile(filePath: string): 'typescript' | 'javascript' {
  const ext = path.extname(filePath).toLowerCase();
  if (['.ts', '.tsx', '.mts', '.cts'].includes(ext)) {
    return 'typescript';
  }
  return 'javascript';
}

/**
 * Проверяет, записано ли имя в формате PascalCase (компоненты React, классы, интерфейсы).
 */
export function isPascalCase(name: string): boolean {
  if (!name || name.length === 0) return false;
  const first = name[0];
  // Первая буква заглавная, и строка не состоит целиком из заглавных (не КОНСТАНТА)
  return first === first.toUpperCase() && first !== first.toLowerCase() && name !== name.toUpperCase();
}

/**
 * Проверяет, является ли функция React-хуком по соглашению об именовании (useName).
 */
export function isReactHookName(name: string): boolean {
  if (!name.startsWith('use') || name.length <= 3) return false;
  const fourthChar = name[3];
  return fourthChar === fourthChar.toUpperCase() && fourthChar !== fourthChar.toLowerCase();
}

/**
 * Проверяет, содержит ли AST-узел JSX-элементы (JsxElement, JsxSelfClosingElement, JsxFragment).
 */
export function containsJsx(node: ts.Node): boolean {
  let hasJsx = false;

  function visit(n: ts.Node) {
    if (hasJsx) return;
    if (
      ts.isJsxElement(n) ||
      ts.isJsxSelfClosingElement(n) ||
      ts.isJsxFragment(n)
    ) {
      hasJsx = true;
      return;
    }
    ts.forEachChild(n, visit);
  }

  ts.forEachChild(node, visit);
  return hasJsx;
}

/**
 * Проверяет, является ли переменная styled-компонентом (например, styled.div`...` или styled(Component)`...`).
 */
export function isStyledComponent(initializer: ts.Expression, sourceFile: ts.SourceFile): boolean {
  const text = initializer.getText(sourceFile);
  return text.startsWith('styled.') || text.startsWith('styled(');
}

/**
 * Извлекает краткую строковую сигнатуру для узла AST.
 */
export function extractSignature(node: ts.Node, sourceFile: ts.SourceFile): string {
  if (ts.isFunctionDeclaration(node) && node.body) {
    const end = node.body.getStart(sourceFile);
    return sourceFile.text.slice(node.getStart(sourceFile), end).trim();
  }

  if (ts.isMethodDeclaration(node) && node.body) {
    const end = node.body.getStart(sourceFile);
    return sourceFile.text.slice(node.getStart(sourceFile), end).trim();
  }

  if (ts.isClassDeclaration(node)) {
    // Сигнатура класса до открывающей фигурной скобки
    const text = node.getText(sourceFile);
    const braceIdx = text.indexOf('{');
    return braceIdx !== -1 ? text.substring(0, braceIdx).trim() : text.split('\n')[0].trim();
  }

  if (ts.isInterfaceDeclaration(node)) {
    const text = node.getText(sourceFile);
    const braceIdx = text.indexOf('{');
    return braceIdx !== -1 ? text.substring(0, braceIdx).trim() : text.split('\n')[0].trim();
  }

  if (ts.isTypeAliasDeclaration(node)) {
    const text = node.getText(sourceFile);
    const eqIdx = text.indexOf('=');
    return eqIdx !== -1 ? text.substring(0, eqIdx).trim() : text.split('\n')[0].trim();
  }

  if (ts.isVariableDeclaration(node)) {
    if (node.initializer && (ts.isArrowFunction(node.initializer) || ts.isFunctionExpression(node.initializer))) {
      const fn = node.initializer;
      if (fn.body) {
        const end = fn.body.getStart(sourceFile);
        return sourceFile.text.slice(node.getStart(sourceFile), end).trim();
      }
    }
    const fullText = node.getText(sourceFile);
    return fullText.split('\n')[0].trim();
  }

  const line = node.getText(sourceFile).split('\n')[0].trim();
  return line.length > 120 ? line.substring(0, 117) + '...' : line;
}

/**
 * Извлекает координаты начала и конца узла (1-based).
 */
export function getNodeLineSpan(node: ts.Node, sourceFile: ts.SourceFile): { startLine: number; endLine: number } {
  const startPos = node.getStart(sourceFile);
  const endPos = node.getEnd();
  const start = sourceFile.getLineAndCharacterOfPosition(startPos);
  const end = sourceFile.getLineAndCharacterOfPosition(endPos);
  return {
    startLine: start.line + 1,
    endLine: end.line + 1
  };
}

/**
 * Классифицирует тип функции или переменной (компонент, хук, функция, константа и т.д.).
 */
export function classifyFunctionOrVariable(
  name: string,
  initializerOrBody: ts.Node | undefined,
  typeNode: ts.TypeNode | undefined,
  sourceFile: ts.SourceFile
): SymbolKind {
  // Проверка на React-хук
  if (isReactHookName(name)) {
    return 'hook';
  }

  // Проверка на React-компонент
  if (isPascalCase(name)) {
    // 1. По аннотации типа (React.FC, FC, ComponentType и т.п.)
    if (typeNode) {
      const typeText = typeNode.getText(sourceFile);
      if (typeText.includes('FC') || typeText.includes('Component') || typeText.includes('Element')) {
        return 'component';
      }
    }
    // 2. По наличию JSX разметки внутри функции
    if (initializerOrBody && containsJsx(initializerOrBody)) {
      return 'component';
    }
  }

  // Проверка на styled_component
  if (initializerOrBody && ts.isExpression(initializerOrBody) && isStyledComponent(initializerOrBody, sourceFile)) {
    return 'styled_component';
  }

  // Проверка на константу (ALL_CAPS)
  if (name.length > 1 && name === name.toUpperCase() && !name.startsWith('_')) {
    return 'constant';
  }

  return 'function';
}
