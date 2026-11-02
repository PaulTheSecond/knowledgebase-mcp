// CLI точка входа для компиляторного парсера ts-parser (ADR-009)

import * as fs from 'fs';
import * as path from 'path';
import { ParserOptions } from './types.js';
import { TypeScriptSemanticAnalyzer } from './analyzer.js';
import { normalizePath } from './filters.js';

function printUsageAndExit(): never {
  console.error(`
ts-parser — Компиляторный анализатор семантических связей TypeScript/React (ADR-009)

Использование:
  node ts-parser.js --project <path/to/tsconfig.json> [options]

Параметры:
  --project <path>    Путь к tsconfig.json проекта (обязательный)
  --root <path>       Корневая директория репозитория (по умолчанию папка tsconfig.json)
  --files <path>      Файл со списком относительных/абсолютных путей или CSV-список (для diff PR)
  --mode <full|diff>  Режим анализа: full — полная индексация, diff — инкрементальный анализ
  --help, -h          Показать данную справку
`);
  process.exit(1);
}

function parseCliArgs(argv: string[]): ParserOptions {
  let projectPath: string | null = null;
  let rootPath: string | null = null;
  let filesArg: string | null = null;
  let mode: 'full' | 'diff' = 'full';

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      printUsageAndExit();
    } else if (arg === '--project' && i + 1 < argv.length) {
      projectPath = argv[++i];
    } else if (arg === '--root' && i + 1 < argv.length) {
      rootPath = argv[++i];
    } else if (arg === '--files' && i + 1 < argv.length) {
      filesArg = argv[++i];
    } else if (arg === '--mode' && i + 1 < argv.length) {
      const modeVal = argv[++i].toLowerCase();
      if (modeVal === 'diff' || modeVal === 'full') {
        mode = modeVal;
      }
    }
  }

  if (!projectPath) {
    console.error('Ошибка: параметр --project <path> обязателен.');
    printUsageAndExit();
  }

  const resolvedProject = path.resolve(projectPath);
  const resolvedRoot = rootPath ? path.resolve(rootPath) : path.dirname(resolvedProject);

  const targetFiles = new Set<string>();

  const resolveTargetFile = (filePath: string): string => {
    if (path.isAbsolute(filePath)) {
      return normalizePath(filePath);
    }
    const fromCwd = path.resolve(process.cwd(), filePath);
    if (fs.existsSync(fromCwd)) {
      return normalizePath(fromCwd);
    }
    const fromRoot = path.resolve(resolvedRoot, filePath);
    return normalizePath(fromRoot);
  };

  if (filesArg) {
    const isFileList =
      fs.existsSync(filesArg) &&
      fs.statSync(filesArg).isFile() &&
      (filesArg.endsWith('.txt') || filesArg.endsWith('.lst') || filesArg.endsWith('.list'));

    if (isFileList) {
      const lines = fs.readFileSync(filesArg, 'utf-8').split(/\r?\n/);
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed && !trimmed.startsWith('#')) {
          targetFiles.add(resolveTargetFile(trimmed));
        }
      }
    } else {
      const parts = filesArg.split(/[,;]/);
      for (const part of parts) {
        const trimmed = part.trim();
        if (trimmed) {
          targetFiles.add(resolveTargetFile(trimmed));
        }
      }
    }
    mode = 'diff';
  }

  return {
    projectPath: resolvedProject,
    rootPath: resolvedRoot,
    targetFiles: targetFiles.size > 0 ? targetFiles : undefined,
    mode
  };
}

function main(): void {
  try {
    const args = process.argv.slice(2);
    if (args.length === 0) {
      printUsageAndExit();
    }

    const options = parseCliArgs(args);
    const analyzer = new TypeScriptSemanticAnalyzer(options);

    analyzer.initialize();
    const result = analyzer.analyze();

    // Выводим структурированный JSON в stdout (аналогично RoslynParser)
    process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  } catch (error: any) {
    process.stderr.write(`[ts-parser error] ${error.message || error}\n`);
    if (error.stack) {
      process.stderr.write(`${error.stack}\n`);
    }
    process.exit(1);
  }
}

main();
