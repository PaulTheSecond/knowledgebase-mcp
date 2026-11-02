// Скрипт сборки автономного бандла ts-parser с помощью esbuild
const esbuild = require('esbuild');
const path = require('path');
const fs = require('fs');

async function build() {
  const outDir = path.resolve(__dirname, 'dist');
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  console.log('[ts-parser build] Сборка автономного бандла dist/ts-parser.js...');
  const startTime = Date.now();

  try {
    await esbuild.build({
      entryPoints: [path.resolve(__dirname, 'src/index.ts')],
      bundle: true,
      platform: 'node',
      target: 'node20',
      outfile: path.resolve(outDir, 'ts-parser.js'),
      banner: {
        js: '#!/usr/bin/env node\n'
      },
      minify: false,
      sourcemap: false,
      logLevel: 'info',
      // typescript пакуется в единый бандл без внешних зависимостей
      external: []
    });

    const elapsed = Date.now() - startTime;
    const stats = fs.statSync(path.resolve(outDir, 'ts-parser.js'));
    const sizeMb = (stats.size / (1024 * 1024)).toFixed(2);
    console.log(`[ts-parser build] Успешно собран dist/ts-parser.js (${sizeMb} MB) за ${elapsed} мс.`);
  } catch (err) {
    console.error('[ts-parser build] Ошибка при сборке бандла:', err);
    process.exit(1);
  }
}

build();
