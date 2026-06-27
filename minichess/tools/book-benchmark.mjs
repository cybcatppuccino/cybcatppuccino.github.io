import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { COORD_SYSTEMS } from '../js/core/constants.js';
import { parsePGN } from '../js/core/pgn.js';
import { moveToUci as coreMoveToUci } from '../js/core/notation.js';
import { ENGINE_VERSION, EnginePosition, GardnerSearcher, generateLegalMoves } from '../js/engine/engine.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const files = [
  'Gardnerwhiteoracle.pgn',
  'Gardnerblackoracle_whitemovesd4.pgn',
  'Gardnerblackoracle_whitemovese4.pgn',
  'Gardnerblackoracle_whitemovesf4.pgn'
];
const maxSamples = Number(process.argv[2] || 24);
const depth = Number(process.argv[3] || 4);
const timeMs = Number(process.argv[4] || 900);
const samples = [];

function standardSourceLabel(file) {
  const raw = file.replace('Gardnerblackoracle_whitemoves', '').replace('.pgn', '');
  return { b4: 'a3', d4: 'c3', e4: 'd3', f4: 'e3' }[raw] || raw;
}
for (const file of files) {
  const study = parsePGN(fs.readFileSync(path.join(root, 'data/pgn', file), 'utf8'), file, { coordSystem: COORD_SYSTEMS.LEGACY_STUDY });
  let node = study.root;
  let ply = 0;
  while (node.children.length && samples.length < maxSamples) {
    const child = node.children[0];
    if (ply >= 1 && ply % 2 === 1) {
      samples.push({ file, ply, fen: node.positionFen, reference: coreMoveToUci(child.move), san: child.san });
    }
    node = child;
    ply += 1;
  }
  if (samples.length >= maxSamples) break;
}

let top1 = 0, top3 = 0, scored = 0, totalDelta = 0, maxDelta = 0;
let within50 = 0, within100 = 0, within150 = 0;
const rows = [];
for (const [index, sample] of samples.entries()) {
  const pos = EnginePosition.fromFEN(sample.fen);
  const legalCount = generateLegalMoves(pos).length;
  const searcher = new GardnerSearcher({ hashEntries: 131072 });
  const result = searcher.analyze(pos, {
    timeMs,
    startDepth: depth,
    maxDepth: depth,
    multipv: legalCount,
    newPosition: true
  });
  const lineIndex = result.lines.findIndex(line => line.move === sample.reference);
  const best = result.lines[0]?.score ?? 0;
  const ref = lineIndex >= 0 ? result.lines[lineIndex].score : null;
  const side = pos.turn;
  // Engine result scores are White-centric; compare from side-to-move perspective.
  const delta = ref == null ? null : (side === 1 ? best - ref : ref - best);
  if (lineIndex === 0) top1 += 1;
  if (lineIndex >= 0 && lineIndex < 3) top3 += 1;
  if (delta != null) {
    scored += 1;
    totalDelta += Math.max(0, delta);
    const loss = Math.max(0, delta);
    maxDelta = Math.max(maxDelta, loss);
    if (loss <= 50) within50 += 1;
    if (loss <= 100) within100 += 1;
    if (loss <= 150) within150 += 1;
  }
  rows.push({
    n: index + 1,
    source: standardSourceLabel(sample.file),
    ply: sample.ply,
    reference: sample.reference,
    rank: lineIndex < 0 ? 'timeout' : lineIndex + 1,
    deltaCp: delta == null ? null : Math.round(delta),
    depth: result.depth,
    best: result.lines[0]?.move || '—'
  });
  process.stdout.write(`\rBenchmarked ${index + 1}/${samples.length}`);
}
process.stdout.write('\n');
console.table(rows);
const summary = {
  engine: ENGINE_VERSION,
  samples: samples.length,
  requestedDepth: depth,
  timePerPositionMs: timeMs,
  top1Rate: samples.length ? top1 / samples.length : 0,
  top3Rate: samples.length ? top3 / samples.length : 0,
  scoredPositions: scored,
  meanReferenceDeltaCp: scored ? totalDelta / scored : null,
  maxReferenceDeltaCp: maxDelta,
  within50CpRate: scored ? within50 / scored : 0,
  within100CpRate: scored ? within100 / scored : 0,
  within150CpRate: scored ? within150 / scored : 0,
  methodology: 'Samples the first/main continuation of supplied oracle PGNs. The engine receives no book moves during the benchmark.'
};
console.log(JSON.stringify(summary, null, 2));
fs.writeFileSync(path.join(root, 'data', 'book-benchmark.json'), JSON.stringify({ generatedAt: new Date().toISOString(), summary, rows }, null, 2));
