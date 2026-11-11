// Black Oracle: LOUDS opening tree (all White replies; one certified best Black move)
// with a separate, bounded exact WASM solver for positions outside the tree.
let bookPromise = null;
let book = null;
let solverPromise = null;
let solver = null;
let whiteBook = null;
let whiteBookPromise = null;

// The original White Oracle's LOUDS book, queried independently of the
// Orion WASM get_white_move/get_black_moves API (which also searches on a miss).
// Membership is therefore exact: search results are never mislabeled as a DB hit.
function makeWhiteBook(treeBuffer, dataBuffer) {
  const tree = new Uint8Array(treeBuffer);
  const data = new Uint8Array(dataBuffer);
  const view = new DataView(treeBuffer);
  const bitLength = tree.length * 8;
  const nodes = data.length / 2;
  if (tree.length % 64 || data.length % 2) throw new Error('White LOUDS data alignment error');
  const rankBlocks = new Uint32Array(bitLength / 512 + 1);
  function popcount(n) {
    n = n - ((n >>> 1) & 0x55555555);
    n = (n & 0x33333333) + ((n >>> 2) & 0x33333333);
    return (((n + (n >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
  }
  for (let block = 0; block < rankBlocks.length - 1; block++) {
    let sum = rankBlocks[block];
    for (let i = 0; i < 16; i++) sum += popcount(view.getUint32(block * 64 + i * 4, true));
    rankBlocks[block + 1] = sum;
  }
  if (rankBlocks[rankBlocks.length - 1] !== nodes) throw new Error('White LOUDS node count mismatch');
  function rank1(pos) {
    const block = pos >>> 9;
    const wordCount = (pos & 511) >>> 5;
    let result = rankBlocks[block];
    let byteOffset = block * 64;
    for (let i = 0; i < wordCount; i++) result += popcount(view.getUint32(byteOffset + i * 4, true));
    const tail = pos & 31;
    if (tail) result += popcount(view.getUint32(byteOffset + wordCount * 4, true) & (0xffffffff >>> (32 - tail)));
    return result;
  }
  function select0(index) {
    let low = 0, high = bitLength;
    while (low < high) {
      const mid = low + ((high - low) >>> 1);
      if (mid - rank1(mid) < index) low = mid + 1;
      else high = mid;
    }
    return low;
  }
  function testBit(pos) { return (tree[pos >>> 3] >>> (pos & 7)) & 1; }
  function childCount(id) {
    const first = select0(id);
    let count = 0;
    while (first + count < bitLength && testBit(first + count)) count++;
    return count;
  }
  function childId(id, index) { return rank1(select0(id) + 1 + index); }
  function whiteMove(id) { const move = data[(id - 1) * 2]; return move === 36 ? -1 : move; }
  function whiteScore(id) { return (data[(id - 1) * 2 + 1] << 24) >> 24; }
  return { nodes, childCount, childId, whiteMove, whiteScore };
}

async function loadWhiteBook() {
  if (whiteBook) return whiteBook;
  if (!whiteBookPromise) {
    whiteBookPromise = Promise.all([
      fetchBytes('white-opening-tree.dat'), fetchBytes('white-opening-data.dat')
    ]).then(([tree, data]) => {
      whiteBook = makeWhiteBook(tree, data);
      return whiteBook;
    });
  }
  return whiteBookPromise;
}

function whiteTransform(first) {
  // Normalize to White Oracle's canonical E4 (index 22).
  if (first === 8) return 1;   // C2: anti-diagonal
  if (first === 13) return 2;  // B3: rotate 180
  if (first === 22) return 0;  // E4: identity
  if (first === 27) return 3;  // D5: transpose
  return -1;
}
function transformWhiteMove(move, transform) {
  if (move < 0) return -1;
  const x = move % 6, y = (move / 6) | 0;
  if (transform === 1) return (5 - x) * 6 + (5 - y);
  if (transform === 2) return (5 - y) * 6 + (5 - x);
  if (transform === 3) return x * 6 + y;
  return move;
}
function newWhiteBoard() {
  const board = new Int8Array(36);
  board[14] = -1; board[15] = 1; board[20] = 1; board[21] = -1;
  return board;
}
const whiteDirs = [[-1,-1], [0,-1], [1,-1], [-1,0], [1,0], [-1,1], [0,1], [1,1]];
function whiteFlips(board, side, move) {
  if (move < 0 || move > 35 || board[move]) return [];
  const x = move % 6, y = (move / 6) | 0, flips = [];
  for (const [dx,dy] of whiteDirs) {
    let xx = x + dx, yy = y + dy, line = [];
    while (xx >= 0 && xx < 6 && yy >= 0 && yy < 6 && board[yy * 6 + xx] === -side) {
      line.push(yy * 6 + xx); xx += dx; yy += dy;
    }
    if (line.length && xx >= 0 && xx < 6 && yy >= 0 && yy < 6 && board[yy * 6 + xx] === side) flips.push(...line);
  }
  return flips;
}
function whiteLegal(board, side) {
  const moves = [];
  for (let i = 0; i < 36; i++) if (whiteFlips(board, side, i).length) moves.push(i);
  return moves.length ? moves : [-1];
}
function whitePlay(board, side, move) {
  const legal = whiteLegal(board, side);
  if (!legal.includes(move)) return false;
  if (move !== -1) {
    for (const i of whiteFlips(board, side, move)) board[i] = side;
    board[move] = side;
  }
  return true;
}
function lookupWhiteBook(moves, db) {
  if (!Array.isArray(moves) || moves.length > 64 || moves.some(m => !Number.isInteger(m) || m < -1 || m > 35)) return null;
  if (!moves.length) return {
    side: 'black', options: [8,13,22,27].map(move => ({move, blackValue: -db.whiteScore(1)}))
  };
  const transform = whiteTransform(moves[0]);
  if (transform < 0) return null;
  const normalized = moves.map(move => transformWhiteMove(move, transform));
  const board = newWhiteBoard();
  if (!whitePlay(board, 1, normalized[0])) return null;
  let id = 1, index = 1;
  while (true) {
    if (id < 1 || id > db.nodes) return null;
    if (index === normalized.length) {
      if (index % 2) return {
        side: 'white', bestMove: transformWhiteMove(db.whiteMove(id), transform),
        scoreWhite: db.whiteScore(id)
      };
      const legal = whiteLegal(board, 1);
      if (db.childCount(id) !== legal.length) return null;
      return { side: 'black', options: legal.map((move, i) => {
        const child = db.childId(id, i);
        return { move: transformWhiteMove(move, transform), blackValue: -db.whiteScore(child) };
      }) };
    }
    // White's only book continuation is the stored optimal move.
    const whiteMove = db.whiteMove(id);
    if (normalized[index] !== whiteMove || db.childCount(id) === 0 || !whitePlay(board, -1, whiteMove)) return null;
    index++;
    if (index === normalized.length) continue;
    const legal = whiteLegal(board, 1);
    if (db.childCount(id) !== legal.length) return null;
    const chosen = legal.indexOf(normalized[index]);
    if (chosen < 0) return null;
    id = db.childId(id, chosen);
    if (!whitePlay(board, 1, normalized[index])) return null;
    index++;
  }
}

async function fetchBytes(filename) {
  const response = await fetch(new URL(filename, self.location.href));
  if (!response.ok) throw new Error(`Black Oracle asset ${filename}: HTTP ${response.status}`);
  return response.arrayBuffer();
}

async function loadBook() {
  if (book) return book;
  if (!bookPromise) {
    bookPromise = (async () => {
      const [wasmBytes, treeBytes, dataBytes] = await Promise.all([
        fetchBytes('black-oracle-db.wasm'),
        fetchBytes('black-opening-tree.dat'),
        fetchBytes('black-opening-data.dat')
      ]);
      const { instance } = await WebAssembly.instantiate(wasmBytes, {});
      const exports = instance.exports;
      const treePointer = exports.alloc(treeBytes.byteLength);
      const dataPointer = exports.alloc(dataBytes.byteLength);
      if (!treePointer || !dataPointer) throw new Error('Black Oracle WASM database allocation failed');
      new Uint8Array(exports.memory.buffer, treePointer, treeBytes.byteLength).set(new Uint8Array(treeBytes));
      new Uint8Array(exports.memory.buffer, dataPointer, dataBytes.byteLength).set(new Uint8Array(dataBytes));
      if (exports.init_db(treePointer, treeBytes.byteLength, dataPointer, dataBytes.byteLength) !== 1) {
        throw new Error('Black Oracle LOUDS database verification failed');
      }
      const historyPointer = exports.alloc(64);
      if (!historyPointer) throw new Error('Black Oracle history allocation failed');
      book = { wasm: exports, historyPointer };
      self.postMessage({ type: 'ready', nodes: exports.get_node_count() });
      return book;
    })();
  }
  return bookPromise;
}

async function loadSolver() {
  if (solver) return solver;
  if (!solverPromise) {
    solverPromise = (async () => {
      const bytes = await fetchBytes('black-oracle-solver.wasm');
      const { instance } = await WebAssembly.instantiate(bytes, {});
      solver = instance.exports;
      return solver;
    })();
  }
  return solverPromise;
}

function packHistory(moves, state) {
  if (!Array.isArray(moves) || moves.length > 64 || moves.some((m) => !Number.isInteger(m) || m < -1 || m > 35)) return false;
  const target = new Int8Array(state.wasm.memory.buffer, state.historyPointer, 64);
  target.fill(0);
  target.set(moves);
  return true;
}

function unpackMoveScore(packed) {
  if (!packed) return null;
  const move = (packed & 255) - 2;
  const value = (packed >>> 8) & 255;
  return { move, scoreBlack: value < 128 ? value : value - 256 };
}

function dbBlackMove(moves, state) {
  if (!packHistory(moves, state)) return null;
  return unpackMoveScore(state.wasm.black_lookup(state.historyPointer, moves.length));
}

function dbWhiteOptions(moves, state) {
  if (!packHistory(moves, state)) return null;
  const count = state.wasm.white_replies(state.historyPointer, moves.length);
  if (count < 0 || count > 36) return null;
  const raw = new Int8Array(state.wasm.memory.buffer, state.wasm.get_replies_ptr(), count * 2);
  const replies = Array.from({ length: count }, (_, i) => ({ move: raw[i * 2], blackValue: raw[i * 2 + 1] }));
  // 127 is the database-miss sentinel. Never label partial coverage as exact.
  if (replies.some((reply) => reply.blackValue === 127)) return null;
  const options = [];
  for (const reply of replies) {
    const child = dbBlackMove(moves.concat(reply.move), state);
    if (!child || child.scoreBlack !== reply.blackValue) return null;
    options.push({ move: reply.move, blackReply: child.move, blackValue: reply.blackValue });
  }
  return options;
}

function readSolverOptions(wasm, count) {
  const raw = new Int8Array(wasm.memory.buffer, wasm.output_ptr(), count * 3);
  const out = [];
  for (let i = 0; i < count; i++) out.push({
    move: raw[i * 3], blackReply: raw[i * 3 + 1], blackValue: raw[i * 3 + 2]
  });
  return out;
}

self.addEventListener('message', async (event) => {
  const payload = event.data || {};
  if (payload.type !== 'lookup' && payload.type !== 'white-book' && payload.type !== 'solve') return;
  const id = payload.id;
  try {
    if (payload.type === 'lookup') {
      const state = await loadBook();
      const moves = payload.moves || [];
      if (moves.length % 2 === 0) {
        const result = dbBlackMove(moves, state);
        self.postMessage(result ? {
          id, found: true, exact: true, side: 'black', bestMove: result.move,
          scoreBlack: result.scoreBlack, moves, source: 'Black Oracle DB'
        } : { id, found: false, moves });
      } else {
        const options = dbWhiteOptions(moves, state);
        self.postMessage(options !== null ? {
          id, found: true, exact: true, side: 'white', options,
          moves, source: 'Black Oracle DB'
        } : { id, found: false, moves });
      }
      return;
    }

    if (payload.type === 'white-book') {
      const db = await loadWhiteBook();
      const result = lookupWhiteBook(payload.moves || [], db);
      self.postMessage(result ? {
        id, found: true, exact: true, ...result, source: 'White Oracle DB'
      } : { id, found: false });
      return;
    }

    // Solver is loaded lazily. A database hit never starts endgame search.
    const wasm = await loadSolver();
    const board = Array.isArray(payload.board) ? payload.board : [];
    const boardInput = new Int8Array(wasm.memory.buffer, wasm.search_board_ptr(), 36);
    for (let i = 0; i < 36; i++) {
      const cell = board[i];
      boardInput[i] = cell === 'black' || cell === 1 || cell === 'B' ? 1 :
        (cell === 'white' || cell === -1 || cell === 'W' ? -1 : 0);
    }
    const side = payload.side === 'white' ? 'white' : 'black';
    if (side === 'black') {
      const move = wasm.solve_black_position();
      self.postMessage({
        id, found: true, exact: Boolean(wasm.solve_is_exact()), side,
        bestMove: move, scoreBlack: wasm.solve_score_black(),
        nodes: wasm.solve_nodes(), source: 'Black Oracle WASM search'
      });
    } else {
      const count = wasm.solve_white_options();
      self.postMessage({
        id, found: true, exact: Boolean(wasm.solve_is_exact()), side,
        options: readSolverOptions(wasm, count), nodes: wasm.solve_nodes(),
        source: 'Black Oracle WASM search'
      });
    }
  } catch (error) {
    self.postMessage({ id, found: false, error: error?.message || String(error) });
  }
});

loadBook().catch((error) => {
  self.postMessage({ type: 'error', message: error?.message || String(error) });
});
