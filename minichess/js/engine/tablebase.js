import {
  ENGINE_VERSION,
  EngineInternals,
  generateLegalMoves,
  moveToUci,
  scoreToDisplay,
  validateMateResult
} from './engine.js';

const {
  PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK,
  makeMove, undoMove, moveFrom, moveTo, movePromotion, encodeMove,
  MATE
} = EngineInternals;

const MATERIAL_ORDER = Object.freeze([
  ['Q', QUEEN], ['R', ROOK], ['B', BISHOP], ['N', KNIGHT], ['P', PAWN]
]);
const EXACT_MAP = Object.freeze([-1, 0, 1, 2]);
const TRANSFORM_MIRROR_FILES = 1;
const TRANSFORM_ROTATE_SWAP = 2;
const DEFAULT_BASE = new URL('../../tools/gardner_tablebase/tables/', import.meta.url).href;

const COMB = Array.from({ length: 26 }, () => Array(7).fill(0));
for (let n = 0; n <= 25; n += 1) {
  COMB[n][0] = 1;
  for (let k = 1; k <= Math.min(6, n); k += 1) {
    COMB[n][k] = k === n ? 1 : (COMB[n - 1]?.[k - 1] || 0) + (COMB[n - 1]?.[k] || 0);
  }
}

function pieceCount(position) {
  let count = 0;
  for (const piece of position.board) if (piece) count += 1;
  return count;
}

function lexCompare(left, right) {
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    if (left[index] !== right[index]) return left[index] < right[index] ? -1 : 1;
  }
  return left.length === right.length ? 0 : left.length < right.length ? -1 : 1;
}

function sideCounts(board, side) {
  const counts = [0, 0, 0, 0, 0];
  let kings = 0;
  for (const piece of board) {
    if (!piece || Math.sign(piece) !== side) continue;
    const type = Math.abs(piece);
    if (type === KING) kings += 1;
    else {
      const order = type === QUEEN ? 0 : type === ROOK ? 1 : type === BISHOP ? 2 : type === KNIGHT ? 3 : type === PAWN ? 4 : -1;
      if (order >= 0) counts[order] += 1;
    }
  }
  if (kings !== 1) throw new Error('A tablebase position requires exactly one king per side.');
  return counts;
}

function sideText(counts) {
  let text = 'K';
  counts.forEach((count, index) => { if (count) text += MATERIAL_ORDER[index][0].repeat(count); });
  return text;
}

function rotateSwapBoard(board) {
  const output = new Int8Array(25);
  for (let square = 0; square < 25; square += 1) if (board[square]) output[24 - square] = -board[square];
  return output;
}

function mirrorFiles(board) {
  const output = new Int8Array(25);
  for (let square = 0; square < 25; square += 1) {
    const rank = Math.floor(square / 5);
    const file = square % 5;
    output[rank * 5 + (4 - file)] = board[square];
  }
  return output;
}

function transformBoard(board, turn, transform) {
  let output = Int8Array.from(board);
  let outputTurn = turn;
  if (transform & TRANSFORM_ROTATE_SWAP) {
    output = rotateSwapBoard(output);
    outputTurn = -outputTurn;
  }
  if (transform & TRANSFORM_MIRROR_FILES) output = mirrorFiles(output);
  return { board: output, turn: outputTurn };
}

function transformSquare(square, transform) {
  let rank = Math.floor(square / 5);
  let file = square % 5;
  if (transform & TRANSFORM_ROTATE_SWAP) {
    rank = 4 - rank;
    file = 4 - file;
  }
  if (transform & TRANSFORM_MIRROR_FILES) file = 4 - file;
  return rank * 5 + file;
}

function transformPackedMove(move, transform) {
  if (!move || !transform) return move >>> 0;
  return encodeMove(
    transformSquare(moveFrom(move), transform),
    transformSquare(moveTo(move), transform),
    movePromotion(move)
  ) >>> 0;
}

function materialSpec(board) {
  const white = sideCounts(board, WHITE);
  const black = sideCounts(board, BLACK);
  const keep = lexCompare(white, black) > 0 || (lexCompare(white, black) === 0 && sideText(white) >= sideText(black));
  const canonicalWhite = keep ? white : black;
  const canonicalBlack = keep ? black : white;
  return {
    white: canonicalWhite,
    black: canonicalBlack,
    signature: `${sideText(canonicalWhite)}v${sideText(canonicalBlack)}`,
    swapped: !keep,
    pieceCount: 2 + canonicalWhite.reduce((a, b) => a + b, 0) + canonicalBlack.reduce((a, b) => a + b, 0)
  };
}

function specGroups(spec) {
  const codes = [KING, -KING];
  const counts = [1, 1];
  for (const [side, vector] of [[WHITE, spec.white], [BLACK, spec.black]]) {
    vector.forEach((count, index) => {
      if (!count) return;
      codes.push(side * MATERIAL_ORDER[index][1]);
      counts.push(count);
    });
  }
  return { codes, counts };
}

function rankSelectedPositions(positions, count, availableCount) {
  let rank = 0;
  let previous = -1;
  let remaining = count;
  for (let item = 0; item < count; item += 1) {
    const chosen = positions[item];
    for (let candidate = previous + 1; candidate < chosen; candidate += 1) {
      rank += COMB[availableCount - candidate - 1][remaining - 1];
    }
    previous = chosen;
    remaining -= 1;
  }
  return rank;
}

function rankBoard(board, turn, spec) {
  const { codes, counts } = specGroups(spec);
  let available = Array.from({ length: 25 }, (_, index) => index);
  let value = 0;
  for (let group = 0; group < codes.length; group += 1) {
    const code = codes[group];
    const count = counts[group];
    const selectedPositions = [];
    const selectedSquares = new Set();
    for (let position = 0; position < available.length; position += 1) {
      const square = available[position];
      if (board[square] === code) {
        selectedPositions.push(position);
        selectedSquares.add(square);
      }
    }
    if (selectedPositions.length !== count) throw new Error(`Position does not match ${spec.signature}.`);
    const radix = COMB[available.length][count];
    value = value * radix + rankSelectedPositions(selectedPositions, count, available.length);
    available = available.filter(square => !selectedSquares.has(square));
  }
  return value * 2 + (turn === WHITE ? 0 : 1);
}

function exactCanonical(position) {
  const spec = materialSpec(position.board);
  if (!spec.swapped) return { spec, board: Int8Array.from(position.board), turn: position.turn, transform: 0 };
  return { spec, board: rotateSwapBoard(position.board), turn: -position.turn, transform: TRANSFORM_ROTATE_SWAP };
}

function practicalCanonical(position) {
  let best = null;
  for (let transform = 0; transform < 4; transform += 1) {
    let candidate = transformBoard(position.board, position.turn, transform);
    const spec = materialSpec(candidate.board);
    let effectiveTransform = transform;
    if (spec.swapped) {
      candidate = { board: rotateSwapBoard(candidate.board), turn: -candidate.turn };
      effectiveTransform ^= TRANSFORM_ROTATE_SWAP;
    }
    const index = rankBoard(candidate.board, candidate.turn, spec);
    const key = `${spec.signature}\u0000${String(index).padStart(16, '0')}`;
    if (!best || key < best.key) best = { key, spec, index, transform: effectiveTransform };
  }
  return best;
}

function uint16LE(bytes) {
  const count = Math.floor(bytes.byteLength / 2);
  const output = new Uint16Array(count);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < count; index += 1) output[index] = view.getUint16(index * 2, true);
  return output;
}

function uint32LE(bytes) {
  const count = Math.floor(bytes.byteLength / 4);
  const output = new Uint32Array(count);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < count; index += 1) output[index] = view.getUint32(index * 4, true);
  return output;
}

async function fetchBytes(url) {
  const response = await fetch(url, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`Tablebase request failed (${response.status}) for ${url}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`Tablebase metadata request failed (${response.status}) for ${url}`);
  return response.json();
}

async function gunzip(url) {
  const bytes = await fetchBytes(url);
  if (bytes.length < 2 || bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('This browser does not provide native gzip decompression.');
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function binarySearch(array, target) {
  let low = 0;
  let high = array.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (array[middle] < target) low = middle + 1;
    else high = middle;
  }
  return low;
}

function packedValue(value) {
  const states = [-1, 0, 1, 0];
  return {
    wdl: states[value & 3],
    dtmPly: (value >>> 2) & 1023,
    bestMove: (value >>> 12) & 0xffff,
    dtmUpperBound: Boolean((value >>> 28) & 1)
  };
}

class LruCache {
  constructor(limit = 12) {
    this.limit = Math.max(1, limit);
    this.map = new Map();
  }
  get(key) {
    if (!this.map.has(key)) return null;
    const value = this.map.get(key);
    this.map.delete(key);
    this.map.set(key, value);
    return value;
  }
  set(key, value) {
    this.map.delete(key);
    this.map.set(key, value);
    while (this.map.size > this.limit) this.map.delete(this.map.keys().next().value);
  }
}

export class GardnerTablebase {
  constructor({ baseUrl = DEFAULT_BASE, maxCachedBlocks = 12 } = {}) {
    this.baseUrl = new URL(baseUrl, import.meta.url).href;
    this.maxCachedBlocks = maxCachedBlocks;
    this.initialized = false;
    this.available = false;
    this.exactManifest = { tables: {} };
    this.practicalManifest = { tables: {} };
    this.metadata = new Map();
    this.blocks = new LruCache(maxCachedBlocks);
    this.initPromise = null;
    this.lastError = '';
  }

  async init() {
    if (this.initialized) return this.available;
    if (this.initPromise) return this.initPromise;
    this.initPromise = (async () => {
      const exactUrl = new URL('manifest.json', this.baseUrl).href;
      const practicalUrl = new URL('practical-manifest.json', this.baseUrl).href;
      const [exact, practical] = await Promise.allSettled([fetchJson(exactUrl), fetchJson(practicalUrl)]);
      if (exact.status === 'fulfilled') this.exactManifest = exact.value;
      if (practical.status === 'fulfilled') this.practicalManifest = practical.value;
      this.available = Boolean(Object.keys(this.exactManifest.tables || {}).length || Object.keys(this.practicalManifest.tables || {}).length);
      this.initialized = true;
      if (!this.available) {
        const reasons = [exact, practical].filter(item => item.status === 'rejected').map(item => item.reason?.message || String(item.reason));
        this.lastError = reasons.join(' · ') || 'No Gardner tablebase manifests were found.';
      }
      return this.available;
    })();
    return this.initPromise;
  }

  async metadataFor(kind, signature) {
    const key = `${kind}:${signature}`;
    if (this.metadata.has(key)) return this.metadata.get(key);
    const manifest = kind === 'exact' ? this.exactManifest : this.practicalManifest;
    const entry = manifest.tables?.[signature];
    if (!entry) throw new Error(`No ${kind} table for ${signature}.`);
    const metadata = await fetchJson(new URL(entry.path, this.baseUrl).href);
    this.metadata.set(key, metadata);
    return metadata;
  }

  async exactBlock(signature, metadata, blockId) {
    const key = `exact:${signature}:${blockId}`;
    const cached = this.blocks.get(key);
    if (cached) return cached;
    const block = metadata.blocks[blockId];
    if (!block) throw new Error(`Missing exact tablebase block ${signature}/${blockId}.`);
    const tableUrl = new URL(`${signature}/`, this.baseUrl);
    const [wdlBytes, dtmBytes] = await Promise.all([
      gunzip(new URL(block.wdl, tableUrl).href),
      gunzip(new URL(block.dtm, tableUrl).href)
    ]);
    const packed = wdlBytes;
    const wdl = new Int8Array(block.count);
    for (let index = 0; index < block.count; index += 1) {
      const code = (packed[index >>> 2] >>> ((index & 3) * 2)) & 3;
      wdl[index] = EXACT_MAP[code];
    }
    const value = { wdl, dtm: uint16LE(dtmBytes) };
    this.blocks.set(key, value);
    return value;
  }

  async practicalBlock(signature, metadata, blockId) {
    const key = `practical:${signature}:${blockId}`;
    const cached = this.blocks.get(key);
    if (cached) return cached;
    const block = metadata.blocks[blockId];
    if (!block) throw new Error(`Missing practical tablebase block ${signature}/${blockId}.`);
    const tableUrl = new URL(`practical/${signature}/`, this.baseUrl);
    const [indexBytes, valueBytes] = await Promise.all([
      gunzip(new URL(block.indices, tableUrl).href),
      gunzip(new URL(block.values, tableUrl).href)
    ]);
    const encoded = uint32LE(indexBytes);
    const indices = new Uint32Array(encoded.length);
    let total = 0;
    for (let index = 0; index < encoded.length; index += 1) {
      total = metadata.indexEncoding === 'delta-u32' ? (total + encoded[index]) >>> 0 : encoded[index];
      indices[index] = total;
    }
    const value = { indices, values: uint32LE(valueBytes) };
    this.blocks.set(key, value);
    return value;
  }

  async probe(position) {
    if (pieceCount(position) > 6 || !(await this.init())) return null;
    const exact = exactCanonical(position);
    if (this.exactManifest.tables?.[exact.spec.signature]) {
      const metadata = await this.metadataFor('exact', exact.spec.signature);
      const index = rankBoard(exact.board, exact.turn, exact.spec);
      const blockId = Math.floor(index / metadata.blockSize);
      const offset = index % metadata.blockSize;
      const block = await this.exactBlock(exact.spec.signature, metadata, blockId);
      const wdl = block.wdl[offset];
      if (wdl === 2 || wdl === undefined) return null;
      return {
        wdl,
        dtmPly: Number(block.dtm[offset] || 0),
        bestMove: 0,
        dtmUpperBound: false,
        source: 'exact-core',
        signature: exact.spec.signature,
        index
      };
    }

    const practical = practicalCanonical(position);
    if (!this.practicalManifest.tables?.[practical.spec.signature]) return null;
    const metadata = await this.metadataFor('practical', practical.spec.signature);
    const blocks = metadata.blocks || [];
    let low = 0, high = blocks.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (blocks[middle].maxIndex < practical.index) low = middle + 1;
      else high = middle;
    }
    if (low >= blocks.length || practical.index < blocks[low].minIndex) return null;
    const block = await this.practicalBlock(practical.spec.signature, metadata, low);
    const offset = binarySearch(block.indices, practical.index >>> 0);
    if (offset >= block.indices.length || block.indices[offset] !== (practical.index >>> 0)) return null;
    const result = packedValue(block.values[offset]);
    result.bestMove = transformPackedMove(result.bestMove, practical.transform);
    return {
      ...result,
      source: 'practical-verified',
      signature: practical.spec.signature,
      index: practical.index
    };
  }

  async chooseMoves(position, rootProbe, limit = 3) {
    const legal = generateLegalMoves(position, false);
    const candidates = [];
    for (const move of legal) {
      const state = makeMove(position, move);
      let child = null;
      try { child = await this.probe(position); } catch { child = null; }
      undoMove(position, move, state);
      if (!child) continue;
      candidates.push({
        move,
        child,
        wdl: -child.wdl,
        dtmPly: child.dtmPly ? child.dtmPly + 1 : 0
      });
    }

    if (!candidates.length && rootProbe.bestMove && legal.includes(rootProbe.bestMove)) {
      return [{ move: rootProbe.bestMove, child: null, wdl: rootProbe.wdl, dtmPly: rootProbe.dtmPly }];
    }
    const matching = candidates.filter(item => item.wdl === rootProbe.wdl);
    const pool = matching.length ? matching : candidates;
    pool.sort((left, right) => {
      if (left.wdl !== right.wdl) return right.wdl - left.wdl;
      if (rootProbe.wdl > 0) return left.dtmPly - right.dtmPly;
      if (rootProbe.wdl < 0) return right.dtmPly - left.dtmPly;
      return left.dtmPly - right.dtmPly;
    });
    return pool.slice(0, Math.max(1, limit));
  }

  async buildPv(position, firstMove, maxPly = 96) {
    const cursor = position.clone();
    const pv = [];
    let move = firstMove;
    const seen = new Set();
    for (let ply = 0; ply < maxPly && move; ply += 1) {
      const legal = generateLegalMoves(cursor, false);
      if (!legal.includes(move)) break;
      pv.push(move);
      makeMove(cursor, move);
      const nextLegal = generateLegalMoves(cursor, false);
      if (!nextLegal.length) break;
      const key = `${cursor.hashA}:${cursor.hashB}`;
      if (seen.has(key)) break;
      seen.add(key);
      const probe = await this.probe(cursor);
      if (!probe) break;
      const choices = await this.chooseMoves(cursor, probe, 1);
      move = choices[0]?.move || 0;
    }
    return pv;
  }

  async analyze(position, { multipv = 3, maxPvPly = 96 } = {}) {
    const root = position.clone();
    const probe = await this.probe(root);
    if (!probe) return null;
    const choices = await this.chooseMoves(root, probe, multipv);
    const rootSide = root.turn;
    const lines = [];
    for (const choice of choices) {
      const pvMoves = await this.buildPv(root, choice.move, maxPvPly);
      const rootDtm = choice.dtmPly || probe.dtmPly || pvMoves.length;
      const rootScore = probe.wdl > 0
        ? MATE - Math.max(1, rootDtm)
        : probe.wdl < 0
          ? -MATE + Math.max(1, rootDtm)
          : 0;
      const whiteScore = rootSide === WHITE ? rootScore : -rootScore;
      const pv = pvMoves.map(moveToUci);
      const candidate = {
        move: pv[0] || moveToUci(choice.move),
        score: whiteScore,
        scoreText: scoreToDisplay(whiteScore),
        pv: pv.length ? pv : [moveToUci(choice.move)],
        mateVerified: false,
        endgameProof: false,
        tablebase: true,
        tablebaseWdl: probe.wdl,
        dtm: rootDtm,
        dtmUpperBound: Boolean(probe.dtmUpperBound),
        source: probe.source
      };
      if (probe.wdl !== 0 && candidate.pv.length) candidate.mateVerified = validateMateResult(root, { ...candidate, mateVerified: true });
      if (probe.wdl !== 0 && !candidate.mateVerified) {
        const whiteWdl = rootSide === WHITE ? probe.wdl : -probe.wdl;
        candidate.score = whiteWdl > 0 ? 20000 : -20000;
        candidate.scoreText = whiteWdl > 0 ? 'TB win' : 'TB loss';
      }
      lines.push(candidate);
    }
    // Sparse practical records deliberately omit unproved child states. A WDL
    // hit without a legal proved continuation is still useful to the normal
    // search, but it is not sufficient to manufacture a move or a PV. Falling
    // back here avoids presenting an arbitrary legal move as a tablebase line.
    if (!lines.length) return null;
    return {
      engine: ENGINE_VERSION,
      engineLabel: `${ENGINE_VERSION} + GTB`,
      depth: 0,
      selDepth: 0,
      nodes: 0,
      nps: 0,
      elapsed: 0,
      lines,
      terminal: true,
      completed: true,
      tablebase: true,
      tablebaseSource: probe.source,
      tablebaseSignature: probe.signature,
      tablebaseWdl: probe.wdl,
      solved: true,
      nextDepth: 0,
      searchDepth: 0,
      hashfull: 0
    };
  }
}

export const TablebaseInternals = Object.freeze({
  materialSpec,
  exactCanonical,
  practicalCanonical,
  rankBoard,
  transformPackedMove,
  gunzip,
  DEFAULT_BASE
});
