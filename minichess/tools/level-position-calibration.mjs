import fs from 'node:fs';
import { COORD_SYSTEMS } from '../js/core/constants.js';
import { parsePGN, flattenTree } from '../js/core/pgn.js';
import { EnginePosition, GardnerSearcher } from '../js/engine/engine.js';
import { AI_STYLES, buildMoveStyleProfile, selectLineForStyle } from '../js/engine/difficulty.js';

function sampleNodes(path, source, offsets) {
  const text = fs.readFileSync(new URL(path, import.meta.url), 'utf8');
  const study = parsePGN(text, source, { coordSystem: COORD_SYSTEMS.LEGACY_STUDY });
  const nodes = flattenTree(study.root).filter(node => node.ply >= 2 && node.children.length);
  return offsets.map(fraction => nodes[Math.min(nodes.length - 1, Math.floor(nodes.length * fraction))]);
}

const selectedNodes = [
  ...sampleNodes('../data/pgn/Gardneranalysis.pgn', 'Gardner analysis', [0.05, 0.19, 0.43, 0.72]),
  ...sampleNodes('../data/pgn/MalletM25.pgn', 'Mallett M25', [0.17, 0.61])
];

const searcher = new GardnerSearcher({ hashEntries: 262144 });
const positions = [];
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
  for (const line of result.lines) line.styleProfile = buildMoveStyleProfile(position, line);
  const side = node.position.turn;
  const sign = side === 'b' ? -1 : 1;
  const bestUtility = Math.max(...result.lines.map(line => sign * line.score));
  positions.push({
    id: `${node.source || 'archive'}-ply-${node.ply}-${index + 1}`,
    source: node.source,
    ply: node.ply,
    fen: node.position.toStandardFEN(),
    sideToMove: side,
    searchDepth: result.depth,
    candidates: result.lines.map((line, rank) => ({
      rank: rank + 1,
      move: line.move,
      scoreCpWhite: line.score,
      regretCp: bestUtility - sign * line.score
    })),
    styleSelections: AI_STYLES.map(style => {
      const selected = selectLineForStyle(result.lines, style, side) || result.lines[0];
      const rank = result.lines.findIndex(line => line.move === selected.move) + 1;
      return {
        style: style.id,
        label: style.label,
        move: selected.move,
        rank,
        regretCp: bestUtility - sign * selected.score
      };
    })
  });
}

const styles = AI_STYLES.map(style => {
  const samples = positions.flatMap(position => position.styleSelections.filter(selection => selection.style === style.id));
  return {
    style: style.id,
    label: style.label,
    samples: samples.length,
    averageRankOneBased: samples.length ? samples.reduce((sum, item) => sum + item.rank, 0) / samples.length : 0,
    averageRegretCp: samples.length ? samples.reduce((sum, item) => sum + item.regretCp, 0) / samples.length : 0,
    maxRegretCp: samples.length ? Math.max(...samples.map(item => item.regretCp)) : 0
  };
});

const report = {
  generatedAt: new Date().toISOString(),
  purpose: 'Behavioral calibration on real Gardner/Mallett archive positions; not an Elo estimate. v13 uses compact A1–E5 FEN, standard UCI and closed-position verification.',
  engineCoordinateSystem: 'standard-a1-e5',
  positionCount: positions.length,
  positions,
  styles
};
fs.writeFileSync(new URL('../data/level-calibration.json', import.meta.url), `${JSON.stringify(report, null, 2)}\n`);
console.table(styles.map(row => ({ style: row.style, avgRank: row.averageRankOneBased.toFixed(2), avgRegretCp: row.averageRegretCp.toFixed(1), maxRegretCp: row.maxRegretCp })));
