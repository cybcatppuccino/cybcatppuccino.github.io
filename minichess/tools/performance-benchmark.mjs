import { ENGINE_VERSION, EnginePosition, GardnerSearcher } from '../js/engine/engine.js';

const fen = 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1';
const runs = Math.max(2, Number(process.argv[2] || 5));
const timeMs = Math.max(80, Number(process.argv[3] || 950));
const multipv = Math.max(1, Math.min(5, Number(process.argv[4] || 3)));
const rows = [];

for (let run = 1; run <= runs; run += 1) {
  const searcher = new GardnerSearcher({ hashEntries: 524288 });
  const result = searcher.analyze(EnginePosition.fromFEN(fen), {
    timeMs,
    maxDepth: 40,
    multipv,
    newPosition: true
  });
  rows.push({
    run,
    depth: result.depth,
    selDepth: result.selDepth,
    nodes: result.nodes,
    nps: result.nps,
    elapsed: result.elapsed,
    best: result.lines[0]?.move || '—',
    score: result.lines[0]?.scoreText || '—'
  });
}

const warm = rows.slice(1).sort((a, b) => a.nps - b.nps);
const median = warm[Math.floor(warm.length / 2)];
console.log(`${ENGINE_VERSION} · ${timeMs} ms · MultiPV ${multipv}`);
console.table(rows);
console.log('Warm-run median:', median);
