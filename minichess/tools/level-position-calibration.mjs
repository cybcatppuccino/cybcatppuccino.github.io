import fs from 'node:fs';
import { parsePGN, flattenTree } from '../js/core/pgn.js';
import { EnginePosition, GardnerSearcher } from '../js/engine/engine.js';
import { AI_LEVELS, selectLineForLevel } from '../js/engine/difficulty.js';

function lcg(seed = 1) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function sampleNodes(path, source, offsets) {
  const text = fs.readFileSync(new URL(path, import.meta.url), 'utf8');
  const study = parsePGN(text, source);
  const nodes = flattenTree(study.root).filter(node => node.ply >= 2 && node.children.length);
  return offsets.map(fraction => nodes[Math.min(nodes.length - 1, Math.floor(nodes.length * fraction))]);
}

const selectedNodes = [
  ...sampleNodes('../data/pgn/Gardneranalysis.pgn', 'Gardner analysis', [0.05, 0.19, 0.43, 0.72]),
  ...sampleNodes('../data/pgn/MalletM25.pgn', 'Mallett M25', [0.17, 0.61])
];

const searcher = new GardnerSearcher({ hashEntries: 262144 });
const suites = [];
for (let index = 0; index < selectedNodes.length; index += 1) {
  const node = selectedNodes[index];
  const position = EnginePosition.fromFEN(node.position.toCompactFEN());
  const result = searcher.analyze(position, {
    timeMs: 260,
    maxDepth: 10,
    multipv: 10,
    newPosition: true
  });
  if (result.lines.length < 2) continue;
  const side = node.position.turn;
  const sign = side === 'b' ? -1 : 1;
  const bestUtility = Math.max(...result.lines.map(line => sign * line.score));
  suites.push({
    id: `${node.source || 'archive'}-ply-${node.ply}-${index + 1}`,
    source: node.source,
    ply: node.ply,
    fen: node.position.toStudyFEN(),
    sideToMove: side,
    searchDepth: result.depth,
    candidates: result.lines.map((line, rank) => ({ rank: rank + 1, move: line.move, scoreCpWhite: line.score, regretCp: bestUtility - sign * line.score }))
  });
}

const levels = [];
for (const config of AI_LEVELS) {
  let rankTotal = 0;
  let regretTotal = 0;
  let severe = 0;
  let samples = 0;
  const byPosition = [];
  for (let p = 0; p < suites.length; p += 1) {
    const suite = suites[p];
    const rng = lcg(0x600d0000 + config.level * 257 + p);
    let localRank = 0;
    let localRegret = 0;
    const trials = config.level === 10 ? 1 : 500;
    for (let trial = 0; trial < trials; trial += 1) {
      const lines = suite.candidates.map(candidate => ({ move: candidate.move, score: candidate.scoreCpWhite }));
      const selected = selectLineForLevel(lines, config, suite.sideToMove, rng);
      const candidate = suite.candidates.find(item => item.move === selected.move) || suite.candidates[0];
      localRank += candidate.rank - 1;
      localRegret += candidate.regretCp;
      if (candidate.regretCp >= 250) severe += 1;
      samples += 1;
    }
    rankTotal += localRank;
    regretTotal += localRegret;
    byPosition.push({ id: suite.id, averageRankZeroBased: localRank / trials, averageRegretCp: localRegret / trials });
  }
  levels.push({
    level: config.level,
    label: config.label,
    samples,
    averageRankZeroBased: samples ? rankTotal / samples : 0,
    averageRegretCp: samples ? regretTotal / samples : 0,
    severeErrorRate: samples ? severe / samples : 0,
    byPosition
  });
}

const report = {
  generatedAt: new Date().toISOString(),
  purpose: 'Behavioral calibration on real Gardner/Mallett archive positions; not an Elo estimate.',
  positionCount: suites.length,
  positions: suites,
  levels
};
fs.writeFileSync(new URL('../data/level-calibration.json', import.meta.url), `${JSON.stringify(report, null, 2)}\n`);
console.table(levels.map(row => ({ level: row.level, avgRank: row.averageRankZeroBased.toFixed(2), avgRegretCp: row.averageRegretCp.toFixed(1), severePct: `${(row.severeErrorRate * 100).toFixed(1)}%` })));
