// Фильтрация и изоляция от node_modules, стандартных библиотек и внешних файлов (ADR-009 §3)

import * as path from 'path';
import * as ts from 'typescript';

/**
 * Нормализует путь к файлу: заменяет обратные слэши на прямые,
 * приводит к нижнему регистру на платформах без учета регистра (Windows)
 * и разрешает абсолютный путь.
 */
export function normalizePath(filePath: string): string {
  const resolved = path.resolve(filePath);
  let posixPath = resolved.replace(/\\/g, '/');
  if (process.platform === 'win32') {
    // На Windows регистр путей нечувствителен, нормализуем диск и путь к единому виду
    posixPath = posixPath.toLowerCase();
  }
  return posixPath;
}

/**
 * Проверяет, является ли путь дочерним по отношению к rootPath.
 */
export function isInsideRoot(filePath: string, rootPath: string): boolean {
  const normFile = normalizePath(filePath);
  const normRoot = normalizePath(rootPath);

  // Добавляем слеш к корню для корректного сравнения префикса директории
  const rootWithSlash = normRoot.endsWith('/') ? normRoot : normRoot + '/';
  return normFile === normRoot || normFile.startsWith(rootWithSlash);
}

/**
 * Проверяет, является ли файл внешней зависимостью (node_modules, typescript lib или .d.ts).
 */
export function isExternalOrLibrary(filePath: string): boolean {
  const norm = normalizePath(filePath).toLowerCase();

  // Отсекаем зависимости из node_modules
  if (norm.includes('/node_modules/')) {
    return true;
  }

  // Отсекаем стандартные тайпинги TypeScript (lib.d.ts, lib.es2015.d.ts, lib.dom.d.ts и т.д.)
  if (norm.includes('/typescript/lib/') || norm.includes('/typescript/lib.')) {
    return true;
  }

  // Отсекаем сгенерированные файлы тайпингов и деклараций .d.ts
  if (norm.endsWith('.d.ts') || norm.endsWith('.d.mts') || norm.endsWith('.d.cts')) {
    return true;
  }

  return false;
}

/**
 * Проверяет, является ли декларация символа допустимым внутренним кодом репозитория.
 */
export function isValidProjectDeclaration(declaration: ts.Declaration, rootPath: string): boolean {
  const sourceFile = declaration.getSourceFile();
  if (!sourceFile) {
    return false;
  }

  const fileName = sourceFile.fileName;

  // Изоляция: декларация должна лежать внутри корня проекта и не быть node_modules/lib
  if (!isInsideRoot(fileName, rootPath)) {
    return false;
  }

  if (isExternalOrLibrary(fileName)) {
    return false;
  }

  return true;
}

/**
 * Проверяет, является ли файл исходным кодом TypeScript или JavaScript.
 */
export function isSupportedSourceFile(filePath: string): boolean {
  const ext = path.extname(filePath).toLowerCase();
  return ['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs'].includes(ext);
}
