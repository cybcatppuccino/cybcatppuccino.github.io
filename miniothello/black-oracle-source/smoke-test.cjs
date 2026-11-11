'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');

(async () => {
  const assets = path.join(__dirname, '..', 'assets');
  const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, 'black-opening-manifest.json')));
  const tree = fs.readFileSync(path.join(assets, 'black-opening-tree.dat'));
  const data = fs.readFileSync(path.join(assets, 'black-opening-data.dat'));
  assert.equal(manifest.black_decision_layers, 9);
  assert.equal(manifest.white_response_layers, 8);
  assert.equal(manifest.source_max_ply, 17);
  assert.equal(tree.length, 69184);
  assert.equal(data.length, 553050);
  assert.equal(data.length / 2, 276525);
  assert.equal(manifest.black_decision_nodes, data.length / 2);
  assert.equal(manifest.layer_nodes.reduce((a,b) => a+b, 0), data.length/2);
  const wasm = fs.readFileSync(path.join(assets, 'black-oracle-db.wasm'));
  const { instance } = await WebAssembly.instantiate(wasm, {});
  const e = instance.exports;
  const tp = e.alloc(tree.length), dp = e.alloc(data.length);
  new Uint8Array(e.memory.buffer, tp, tree.length).set(tree);
  new Uint8Array(e.memory.buffer, dp, data.length).set(data);
  assert.equal(e.init_db(tp, tree.length, dp, data.length), 1);
  assert.equal(e.get_node_count(), 276525);
  const mp = e.alloc(64);
  const setHistory = moves => {
    assert(moves.length <= 64);
    new Int8Array(e.memory.buffer, mp, 64).fill(0);
    new Int8Array(e.memory.buffer, mp, moves.length).set(moves);
  };
  const lookupBlack = moves => {
    setHistory(moves);
    const packed = e.black_lookup(mp, moves.length);
    if (!packed) return null;
    const v = (packed >>> 8) & 255;
    return { move: (packed & 255) - 2, score: v >= 128 ? v - 256 : v };
  };
  const lookupWhite = moves => {
    setHistory(moves);
    const count = e.white_replies(mp, moves.length);
    if (count < 0) return null;
    const raw = new Int8Array(e.memory.buffer, e.get_replies_ptr(), count * 2);
    return Array.from({ length: count }, (_, i) => ({ move: raw[i * 2], value: raw[i * 2 + 1] }));
  };
  assert.deepEqual(lookupBlack([]), { move: 13, score: -4 });
  assert(lookupWhite([13]).every(x => x.value !== 127));
  // Original Black Oracle best-move line through the 8th White reply.
  const afterW7 = [13,7,8,9,0,6,1,12,2,3,4,10,5,11];
  assert.deepEqual(lookupBlack(afterW7), { move: 16, score: 36 });
  const afterB8 = [...afterW7,16];
  const w8 = lookupWhite(afterB8);
  assert(w8.length > 0 && w8.every(x => x.value !== 127), 'W8 remains covered by B9 evaluations');
  for (const item of w8) assert.equal(lookupBlack([...afterB8,item.move])?.score, item.value);
  const afterW8 = [...afterB8,17];
  assert.deepEqual(lookupBlack(afterW8), { move: 18, score: 36 }, 'B9 is retained');
  const afterB9 = [...afterW8,18];
  const w9 = lookupWhite(afterB9);
  assert.deepEqual(w9.map(x => x.move), [19,25,26]);
  assert(w9.every(x => x.value === 127), 'W9 has no Black database values');
  for (const item of w9) assert.equal(lookupBlack([...afterB9,item.move]), null, 'B10 absent');
  // At the B9->W9 boundary the unchanged dedicated WASM must compute
  // every legal White move to the actual terminal score, not a heuristic.
  const dirs = [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];
  function flips(board, side, pos) {
    if (pos < 0 || board[pos]) return [];
    const x = pos % 6, y = (pos / 6) | 0, result = [];
    for (const [dx, dy] of dirs) {
      let xx = x + dx, yy = y + dy, line = [];
      while (xx >= 0 && xx < 6 && yy >= 0 && yy < 6 && board[yy * 6 + xx] === -side) {
        line.push(yy * 6 + xx); xx += dx; yy += dy;
      }
      if (line.length && xx >= 0 && xx < 6 && yy >= 0 && yy < 6 && board[yy * 6 + xx] === side) result.push(...line);
    }
    return result;
  }
  const board = Array(36).fill(0);
  board[14] = -1; board[15] = 1; board[20] = 1; board[21] = -1;
  afterB9.forEach((move, ply) => {
    const side = ply % 2 ? -1 : 1, f = flips(board, side, move);
    assert(f.length, `Invalid recorded move at ply ${ply + 1}`);
    board[move] = side; for (const pos of f) board[pos] = side;
  });
  assert.equal(board.filter(Boolean).length, 21);
  const exactBytes = fs.readFileSync(path.join(assets, 'endgame-evaluator.wasm'));
  const endgame = new WebAssembly.Instance(new WebAssembly.Module(exactBytes), {}).exports;
  new Int8Array(endgame.memory.buffer, endgame.board_ptr(), 36).set(board);
  assert.equal(endgame.begin(-1), w9.length);
  const exactMoves = Array.from(new Int8Array(endgame.memory.buffer, endgame.moves_ptr(), w9.length));
  assert.deepEqual(exactMoves, w9.map(item => item.move));
  for (let i = 0; i < w9.length; i++) {
    assert.equal(endgame.solve_next(), i);
    assert.equal(new Int16Array(endgame.memory.buffer, endgame.scores_ptr(), w9.length)[i], -36);
  }
  assert.equal(endgame.result_exact(), 1);

  const whiteTree = fs.readFileSync(path.join(assets,'white-opening-tree.dat'));
  const whiteData = fs.readFileSync(path.join(assets,'white-opening-data.dat'));
  assert.equal(crypto.createHash('sha256').update(whiteTree).digest('hex'),
    'ddd6a49aa51913050181890f42344afdd863ca636727efb5f740ce73c68d5205');
  assert.equal(crypto.createHash('sha256').update(whiteData).digest('hex'),
    '7fa299971433587acbc09cbb42edf4b12677c15c198e6bfbe1835828d9489818');
  assert.equal(whiteData.length/2,328300);
  assert.equal((whiteData[1] << 24) >> 24, 4, 'Recalculated White root score');
  console.log('PASS: Black B9W8, B9 lookup valid, W9 exact endgame fallback, White recomputation SHA verified.');
})().catch(error => { console.error(error); process.exitCode=1; });
