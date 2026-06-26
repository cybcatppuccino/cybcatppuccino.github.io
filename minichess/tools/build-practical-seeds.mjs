import fs from 'node:fs/promises';
import path from 'node:path';
import zlib from 'node:zlib';
import { promisify } from 'node:util';
import { parsePGN, flattenTree } from '../js/core/pgn.js';

const gzip = promisify(zlib.gzip);
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const library = JSON.parse(await fs.readFile(path.join(root, 'data/library.json'), 'utf8'));
const entries = new Map();
const stats = { parsedMoves: 0, parseErrors: 0, sourceNodes: 0, acceptedNodes: 0 };

for (const source of library.sources) {
  const text = await fs.readFile(path.join(root, source.path), 'utf8');
  const study = parsePGN(text, source.id);
  stats.parsedMoves += study.parsedMoves;
  stats.parseErrors += study.errors.length;
  for (const node of flattenTree(study.root)) {
    stats.sourceNodes += 1;
    const position = node.position;
    const pieceCount = position.board.reduce((sum, piece) => sum + (piece ? 1 : 0), 0);
    if (pieceCount > 6) continue;
    stats.acceptedNodes += 1;
    const key = position.canonicalKey();
    let entry = entries.get(key);
    if (!entry) {
      entry = {
        fen: position.toCompactFEN(),
        pieceCount,
        frequency: 0,
        sources: new Set(),
        minPly: node.ply
      };
      entries.set(key, entry);
    }
    entry.frequency += 1;
    entry.sources.add(source.id);
    entry.minPly = Math.min(entry.minPly, node.ply);
  }
}

const positions = [...entries.values()]
  .map(entry => ({ ...entry, sources: [...entry.sources].sort() }))
  .sort((a, b) => b.frequency - a.frequency || a.pieceCount - b.pieceCount || a.fen.localeCompare(b.fen));

const payload = {
  format: 'GardnerPracticalSeeds',
  version: 1,
  generatedAt: new Date().toISOString(),
  description: 'Reachable <=6-piece positions extracted from the bundled Gardner and Mallett research PGNs.',
  stats: { ...stats, uniquePositions: positions.length },
  positions
};
const json = Buffer.from(JSON.stringify(payload));
const compressed = await gzip(json, { level: 9 });
const output = path.join(root, 'data/practical-seeds.json.gz');
await fs.writeFile(output, compressed);
console.log(JSON.stringify({ output, bytes: compressed.length, ...payload.stats }, null, 2));
