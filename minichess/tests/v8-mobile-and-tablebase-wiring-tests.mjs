import assert from 'node:assert/strict';
import fs from 'node:fs';

const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const css = fs.readFileSync(new URL('../styles.css', import.meta.url), 'utf8');
const app = fs.readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const worker = fs.readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');
const playWorker = fs.readFileSync(new URL('../js/engine/play-worker.js', import.meta.url), 'utf8');
const tablebase = fs.readFileSync(new URL('../js/engine/tablebase.js', import.meta.url), 'utf8');

assert.match(html, /class="mobile-evaluation-rail"/);
assert.match(html, /id="analysisButton"/);
assert.match(html, /<details id="movePanel" class="panel move-panel">/);
assert.match(css, /--board-size:\s*clamp\(196px, 64vw, 268px\)/);
assert.match(css, /aspect-ratio:\s*1\s*\/\s*1/);
assert.match(css, /#analysisButton\s*\{[\s\S]*?display:\s*inline-flex\s*!important/);
assert.match(css, /\.analysis-line:nth-child\(n\+3\)\s*\{\s*display:\s*none/);
assert.match(css, /\.mobile-evaluation-rail\s*\{[\s\S]*?display:\s*block/);
assert.match(app, /matchMedia\('\(max-width: 640px\)'\)/);
assert.match(app, /movePanel\.open = false/);
assert.match(worker, /new GardnerTablebase\(\)/);
assert.match(playWorker, /config\.level >= 8/);
assert.match(tablebase, /tools\/gardner_tablebase\/tables/);
assert.match(tablebase, /DecompressionStream\('gzip'\)/);

console.log('v8 mobile layout and tablebase wiring tests passed.');
