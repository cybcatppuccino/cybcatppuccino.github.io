import assert from 'node:assert/strict';
import fs from 'node:fs';
import { EnginePosition, GardnerSearcher, ENGINE_VERSION, generateLegalMoves, moveToUci, analyzeOnce } from '../js/engine/engine.js';

assert.equal(ENGINE_VERSION, 'Orion JS 17.1');

const app = fs.readFileSync(new URL('../app.js', import.meta.url), 'utf8');
assert.match(app, /let gameMode = 'local';/, 'page boot should force Local mode');
assert.match(app, /clearAiCachesOnBoot\(\)/, 'page boot should clear AI analysis caches');
assert.match(app, /GAME_STATE_STORAGE_KEY = 'gardner-current-game-v17.1'/, 'v17.1 should persist only the current game tree separately');
assert.match(app, /restoreSavedGameState\(\)/, 'v17.1 should restore the current game tree on reload');

const tablebase = fs.readFileSync(new URL('../js/engine/tablebase.js', import.meta.url), 'utf8');
assert.match(tablebase, /MAX_EXACT_TABLEBASE_PIECES = 5/, 'tablebase use should be hard-limited to <=5 pieces');
assert.match(tablebase, /warmExactWdlForPosition/, 'tablebase should support exact block-level lazy WDL warming');
assert.match(tablebase, /warmExactWdlNeighborhood/, 'workers should prefetch only relevant exact WDL neighborhoods');

const worker = fs.readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');
const playWorker = fs.readFileSync(new URL('../js/engine/play-worker.js', import.meta.url), 'utf8');
assert.doesNotMatch(worker + playWorker, /warmExactWdl\(\{ pieceLimit: 4 \}\)/, 'workers must not startup-warm every small WDL table');
assert.match(worker, /isThinPvResume/, 'analysis worker should reject deep cached results with very short PVs');
assert.match(playWorker, /isThinPvResume/, 'play worker should reject deep cached results with very short PVs');

const manifest = JSON.parse(fs.readFileSync(new URL('../tools/gardner_tablebase/tables/manifest.json', import.meta.url), 'utf8'));
assert.equal(manifest.format, 'GardnerTB');
assert.ok(manifest.tables.KRvKNP, 'uploaded exact <=5 table manifest should include KRvKNP');

const fen = '1n2k/p1ppp/2p2/PP1PP/4K w - - 0 2';
const position = EnginePosition.fromFEN(fen);
const searcher = new GardnerSearcher({ hashEntries: 65536 });
const b3 = generateLegalMoves(position, false).find(move => moveToUci(move) === 'b2b3');
assert.ok(b3, 'test position should contain b2-b3');
const risk = searcher.opponentMateRiskAfterRootMove(position, b3, [b3], 5);
assert.ok(risk?.opponentMateThreat, 'b2-b3 should be detected as allowing a short forced mate');
assert.ok(risk.score < -29000, 'short opponent mate should receive a root mate-loss score');
assert.ok(risk.pv.map(moveToUci).join(' ').startsWith('b2b3 c3c2'), 'risk PV should include the forcing black reply');

const result = analyzeOnce(fen, { timeMs: 1200, maxDepth: 9, multipv: 5, mateProbeMs: 300, mateMaxPlies: 21 });
assert.notEqual(result.lines[0]?.move, 'b2b3', 'analysis should not rank b2-b3 as the best move in the mate-trap position');
const b3Line = result.lines.find(line => line.move === 'b2b3');
if (b3Line) assert.ok(b3Line.mateVerified && b3Line.score < -29000, 'if b2-b3 is displayed, it must be shown as a verified mate loss');
assert.ok((result.lines[0]?.pv?.length || 0) >= 6 || result.lines[0]?.mateVerified || result.tablebase, 'best displayed PV should not be a shallow 4-5 ply fragment in a deep non-terminal search');

console.log('v17.1 state, tablebase lazy loading and tactical-safety tests passed.');
