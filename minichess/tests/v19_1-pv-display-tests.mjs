import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const engine = readFileSync(new URL('../js/engine/engine.js', import.meta.url), 'utf8');
const cache = readFileSync(new URL('../js/engine/analysis-cache.js', import.meta.url), 'utf8');
const styles = readFileSync(new URL('../styles.css', import.meta.url), 'utf8');

assert.match(html, /Gardner MiniChess Lab v19\.3/);
assert.match(html, /Orion JS 19\.3/);
assert.match(html, /app\.js\?v=19\.3/);
assert.match(app, /gardner-current-game-v19\.3/);
assert.match(app, /gardner-current-game-v19\.2/);
assert.match(engine, /ENGINE_VERSION = 'Orion JS 19\.3'/);
assert.match(cache, /gardner-analysis-cache-v19\.3/);
assert.match(cache, /gardner-analysis-cache-v19\.2/);

// The app formats every supplied PV ply; presentation must not turn that into
// a one-line preview. The final v19.1 rule wins over earlier compact rules.
assert.match(app, /const pv = Array\.isArray\(line\.pv\) \? line\.pv : \[\];/);
assert.match(app, /pvSan: pvSan\.join\(' '\)/);
const finalPvRule = styles.slice(styles.lastIndexOf('Gardner MiniChess v19.1 PV visibility'));
assert.match(finalPvRule, /\.analysis-pv\s*\{[\s\S]*white-space:\s*normal;[\s\S]*overflow:\s*visible;[\s\S]*text-overflow:\s*clip;[\s\S]*overflow-wrap:\s*anywhere;/);
assert.match(styles, /\.analysis-line:nth-child\(n\+4\) \{ display: none; \}/);
assert.doesNotMatch(styles, /\.analysis-line:nth-child\(n\+3\) \{ display: none; \}/);

console.log('v19.3 compatibility: PV visibility and versioning tests passed.');
