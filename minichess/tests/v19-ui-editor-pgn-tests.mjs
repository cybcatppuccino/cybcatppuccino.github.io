import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { GameTree } from '../js/core/game-tree.js';
import { Position } from '../js/core/position.js';
import { legalMoves } from '../js/core/rules.js';
import { moveToUci } from '../js/core/notation.js';
import { exportCurrentLineMovetext, exportGameTreeMovetext, exportGameTreePGN, parsePGN } from '../js/core/pgn.js';

const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const panel = readFileSync(new URL('../js/ui/analysis-panel.js', import.meta.url), 'utf8');
const moveList = readFileSync(new URL('../js/ui/move-list.js', import.meta.url), 'utf8');
const styles = readFileSync(new URL('../styles.css', import.meta.url), 'utf8');

assert.match(html, /Gardner MiniChess Lab v19\.3/);
assert.match(html, /app\.js\?v=19\.3/);
assert.match(html, /copyLineButton/);
assert.match(html, /copyTreeButton/);
assert.match(html, /newGameButton/);
assert.match(html, /Move is selected by default/);
assert.doesNotMatch(panel, /rootWinRate/);
assert.doesNotMatch(panel, /analysis-winrate/);
assert.match(app, /ANALYSIS_PAINT_INTERVAL_MS\s*=\s*500/);
assert.match(app, /Principal-variation scores are only synchronized here/);
assert.match(app, /editorTool\s*=\s*'move'/);
assert.match(app, /Move piece/);
assert.match(app, /newGameUndoSnapshot/);
assert.match(app, /restoreNewGameUndoSnapshot/);
assert.match(moveList, /move-turn/);
assert.match(styles, /overflow-y:\s*auto/);

const game = new GameTree(Position.initial());
function play(uci) {
  const move = legalMoves(game.current.position).find(candidate => moveToUci(candidate) === uci);
  assert.ok(move, `Expected legal move ${uci}`);
  return game.play(move);
}

const firstWhite = play('a2a3');
const firstBlack = play('b4b3');
game.navigate(game.root);
play('b2b3');
game.navigate(firstBlack);

assert.equal(exportCurrentLineMovetext(game), '1. a3 b3 *', 'Current-line copy should group a full move in conventional PGN notation');
const treeMovetext = exportGameTreeMovetext(game);
assert.match(treeMovetext, /^1\. a3 \(1\. b3\) b3 \*$/, 'Tree copy should retain a PGN recursive-annotation variation');
const pgn = exportGameTreePGN(game);
assert.match(pgn, /\[Variant "Gardner MiniChess"\]/);
assert.match(pgn, /\[FEN "rnbqk\/ppppp\/5\/PPPPP\/RNBQK w - - 0 1"\]/);
const parsed = parsePGN(pgn, 'v19.3-export');
assert.equal(parsed.errors.length, 0, 'The exported PGN should be readable by the supplied PGN parser');
assert.equal(parsed.parsedMoves, 3, 'The exported tree should preserve both main-line plies plus the root variation');

const snapshot = game.captureSnapshot();
game.reset(Position.initial());
assert.equal(game.restoreSnapshot(snapshot), true, 'A New game undo snapshot should restore a complete game tree');
assert.equal(exportCurrentLineMovetext(game), '1. a3 b3 *');
assert.equal(game.root.children.length, 2, 'The restored snapshot should retain alternate branches');
assert.equal(firstWhite.san, 'a3');

console.log('v19.3 compatibility: editor-mode, PGN export, and New game undo tests passed.');
