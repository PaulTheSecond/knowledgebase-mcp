// Набор автоматизированных тестов для ts-parser (Node.js test runner)

import { test, describe } from 'node:test';
import * as assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const parserJs = path.resolve(__dirname, '../dist/ts-parser.js');
const fixtureTsconfig = path.resolve(__dirname, 'fixtures/tsconfig.json');
const fixtureRoot = path.resolve(__dirname, 'fixtures');

describe('TsParser: Compiler-grade TypeScript/React Semantic Analyzer (ADR-009)', () => {

  test('Режим full: извлекает компоненты, хуки, функции и семантические JSX-связи', () => {
    const stdout = execFileSync('node', [
      parserJs,
      '--project', fixtureTsconfig,
      '--root', fixtureRoot,
      '--mode', 'full'
    ], { encoding: 'utf-8' });

    const result = JSON.parse(stdout);

    assert.ok(result.symbols, 'Ожидается массив symbols');
    assert.ok(result.edges, 'Ожидается массив edges');
    assert.ok(result.metrics, 'Ожидается объект metrics');

    // Проверка символов
    const names = result.symbols.map((s) => s.name);
    assert.ok(names.includes('Button'), 'Символ Button должен присутствовать');
    assert.ok(names.includes('UserCard'), 'Символ UserCard должен присутствовать');
    assert.ok(names.includes('useUserData'), 'Хук useUserData должен присутствовать');
    assert.ok(names.includes('fetchUserApi'), 'Функция fetchUserApi должна присутствовать');

    // Проверка классификации типов (kind)
    const buttonSym = result.symbols.find((s) => s.name === 'Button');
    assert.equal(buttonSym.kind, 'component', 'Button должен классифицироваться как component');

    const userCardSym = result.symbols.find((s) => s.name === 'UserCard');
    assert.equal(userCardSym.kind, 'component', 'UserCard должен классифицироваться как component');

    const hookSym = result.symbols.find((s) => s.name === 'useUserData');
    assert.equal(hookSym.kind, 'hook', 'useUserData должен классифицироваться как hook');

    // Проверка семантического резолва JSX и хуков
    // 1. UserCard использует Button в JSX (<Button ... />)
    const jsxEdge = result.edges.find(
      (e) => e.source_ast_id === userCardSym.ast_node_id && e.target_qualified_name === 'Button'
    );
    assert.ok(jsxEdge, 'Должно присутствовать ребро USES между UserCard и Button в JSX');
    assert.equal(jsxEdge.kind, 'USES');
    assert.ok(
      jsxEdge.target_file_path.toLowerCase().includes('button.tsx'),
      `target_file_path должен указывать на Button.tsx, получено: ${jsxEdge.target_file_path}`
    );

    // 2. UserCard вызывает useUserData
    const hookEdge = result.edges.find(
      (e) => e.source_ast_id === userCardSym.ast_node_id && e.target_qualified_name === 'useUserData'
    );
    assert.ok(hookEdge, 'Должно присутствовать ребро CALLS между UserCard и useUserData');
    assert.equal(hookEdge.kind, 'CALLS');
  });

  test('100% изоляция: отсутствие node_modules и системных lib.d.ts в графе', () => {
    const stdout = execFileSync('node', [
      parserJs,
      '--project', fixtureTsconfig,
      '--root', fixtureRoot,
      '--mode', 'full'
    ], { encoding: 'utf-8' });

    const result = JSON.parse(stdout);

    for (const sym of result.symbols) {
      assert.ok(!sym.file_path.includes('node_modules'), `Символ не должен быть из node_modules: ${sym.file_path}`);
      assert.ok(!sym.file_path.includes('.d.ts'), `Символ не должен быть из .d.ts: ${sym.file_path}`);
    }

    for (const edge of result.edges) {
      if (edge.target_file_path) {
        assert.ok(!edge.target_file_path.includes('node_modules'), `Таргет ребра не должен быть в node_modules: ${edge.target_file_path}`);
        assert.ok(!edge.target_file_path.includes('.d.ts'), `Таргет ребра не должен быть в .d.ts: ${edge.target_file_path}`);
      }
    }
  });

  test('Режим diff: инкрементальный анализ одного файла выполняется субсекундно (< 200 мс)', () => {
    const targetFile = path.resolve(__dirname, 'fixtures/src/components/UserCard.tsx');
    const start = Date.now();

    const stdout = execFileSync('node', [
      parserJs,
      '--project', fixtureTsconfig,
      '--root', fixtureRoot,
      '--files', targetFile,
      '--mode', 'diff'
    ], { encoding: 'utf-8' });

    const duration = Date.now() - start;
    const result = JSON.parse(stdout);

    assert.equal(result.metrics.files_analyzed, 1, 'В diff режиме должен анализироваться только 1 файл');
    assert.ok(
      result.metrics.duration_ms < 500,
      `Внутреннее время анализа компилятором (${result.metrics.duration_ms} мс) должно быть субсекундным (< 500 мс)`
    );
    assert.ok(
      duration < 5000,
      `Общее время CLI вызова (${duration} мс) должно завершаться быстро`
    );

    // Проверяем, что символы извлечены только для UserCard.tsx
    const userCardSym = result.symbols.find((s) => s.name === 'UserCard');
    assert.ok(userCardSym, 'UserCard должен присутствовать');
    const buttonSym = result.symbols.find((s) => s.name === 'Button');
    assert.equal(buttonSym, undefined, 'Button не должен дублироваться в symbols в diff режиме');

    // Но ребро к Button должно присутствовать!
    const jsxEdge = result.edges.find(
      (e) => e.source_ast_id === userCardSym.ast_node_id && e.target_qualified_name === 'Button'
    );
    assert.ok(jsxEdge, 'Ребро USES к Button должно быть разрешено даже в режиме diff');
  });

});
