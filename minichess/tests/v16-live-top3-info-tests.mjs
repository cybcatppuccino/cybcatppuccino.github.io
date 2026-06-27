import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { EnginePosition, GardnerSearcher } from '../js/engine/engine.js';

const engineSource = readFileSync(new URL('../js/engine/engine.js', import.meta.url), 'utf8');
const workerSource = readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');
const panelSource = readFileSync(new URL('../js/ui/analysis-panel.js', import.meta.url), 'utf8');
const cacheSource = readFileSync(new URL('../js/engine/analysis-cache.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');

assert.match(html, /Gardner MiniChess Lab v18.1/, 'visible app version should be v18');
assert.match(html, /Orion JS 18.1/, 'default engine label should be Orion JS 18.1');
assert.match(engineSource, /noteLiveRootLine/, 'searcher should record live root candidate snapshots for UI reporting');
assert.match(engineSource, /mergeKnownRootLines/, 'searcher should merge live root candidates with previous known lines');
assert.match(workerSource, /mergeKnownAnalysisResult/, 'worker should merge every refreshed result with previous known candidate lines');
assert.match(panelSource, /Live\$\{line\.liveDepth/, 'analysis panel should mark incomplete-depth live candidate rows');
assert.match(cacheSource, /gardner-analysis-cache-v18.1/, 'v18.1 cache should use a fresh v18.1 bucket');
assert.match(cacheSource, /gardner-analysis-cache-v17\.4/, 'v18.1 cache should explicitly retire stale v17.4 buckets');

const root = EnginePosition.fromFEN('rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1');
const searcher = new GardnerSearcher({ hashEntries: 16_384 });
const partial = searcher.analyze(root, {
  timeMs: 80,
  maxDepth: 8,
  startDepth: 8,
  multipv: 3,
  mateProbeMs: 0,
  endgameProbeMs: 0,
  fortressProbeMs: 0
});
assert.equal(partial.engine, 'Orion JS 18.1');
assert.equal(partial.completed, false, 'interrupted fixed-depth slices should be surfaced as incomplete');
assert.equal(partial.liveUpdate, true, 'incomplete slices should still publish live known root information');
assert.ok(partial.liveDepth >= 8, 'live update should carry the attempted depth');
assert.ok(partial.lines.length >= 1 && partial.lines.length <= 3, 'live update should publish the strongest known candidates up to the requested top three');
assert.ok(partial.lines.some(line => line.liveUpdate && line.liveDepth >= 8), 'at least one displayed line should come from the live incomplete-depth information');

console.log('v18.1 live top-three analysis information tests passed.');
