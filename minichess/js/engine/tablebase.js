import {
  ENGINE_VERSION,
  EngineInternals,
  generateLegalMoves,
  isInCheck,
  moveToUci,
  scoreToDisplay,
  validateMateResult,
  uciToMove
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
const TRIVIAL_DRAW_SIGNATURES = Object.freeze(new Set(['KvK', 'KBvK', 'KNvK']));
const MATE_IN_ONE_ONLY_SIGNATURES = Object.freeze(new Set(['KBvKB', 'KBvKN', 'KNNvK', 'KNvKN']));

const COMB = Array.from({ length: 26 }, () => Array(7).fill(0));
for (let n = 0; n <= 25; n += 1) {
  COMB[n][0] = 1;
  for (let k = 1; k <= Math.min(6, n); k += 1) {
    COMB[n][k] = k === n ? 1 : (COMB[n - 1]?.[k - 1] || 0) + (COMB[n - 1]?.[k] || 0);
  }
}

function pieceCount(position) {
  if (Number.isInteger(position?.pieceCount)) return position.pieceCount;
  let count = 0;
  for (let sq = 0; sq < 25; sq += 1) if (position.board[sq]) count += 1;
  return count;
}

function cloneProbeResult(result) {
  return result ? { ...result } : null;
}

function cloneAnalyzeResult(result) {
  if (!result) return null;
  return {
    ...result,
    lines: Array.isArray(result.lines)
      ? result.lines.map(line => ({ ...line, pv: Array.isArray(line.pv) ? line.pv.slice() : [] }))
      : []
  };
}

function tablebaseKey(position) {
  return `${position.hashA >>> 0}:${position.hashB >>> 0}`;
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


function maybeLightweightProbe(position, canonical = null) {
  let exact = canonical;
  try {
    if (!exact) exact = exactCanonical(position);
  } catch {
    return null;
  }
  const signature = exact.spec.signature;
  if (TRIVIAL_DRAW_SIGNATURES.has(signature)) {
    return {
      wdl: 0,
      dtmPly: 0,
      bestMove: 0,
      dtmUpperBound: false,
      source: 'hardcoded-draw',
      signature,
      index: -1
    };
  }
  if (!MATE_IN_ONE_ONLY_SIGNATURES.has(signature)) return null;

  const legal = generateLegalMoves(position, false);
  if (!legal.length) {
    return {
      wdl: isInCheck(position) ? -1 : 0,
      dtmPly: 0,
      bestMove: 0,
      dtmUpperBound: false,
      source: 'hardcoded-mate1-only',
      signature,
      index: -1
    };
  }

  for (const move of legal) {
    const state = makeMove(position, move);
    const opponentMated = isInCheck(position) && generateLegalMoves(position, false).length === 0;
    undoMove(position, move, state);
    if (opponentMated) {
      return {
        wdl: 1,
        dtmPly: 1,
        bestMove: move,
        dtmUpperBound: false,
        source: 'hardcoded-mate1-only',
        signature,
        index: -1
      };
    }
  }

  return {
    wdl: 0,
    dtmPly: 0,
    bestMove: 0,
    dtmUpperBound: false,
    source: 'hardcoded-mate1-only',
    signature,
    index: -1
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

const RANK_AVAILABLE = Int8Array.from({ length: 25 }, (_, index) => index);
const RANK_WORK = new Int8Array(25);
const RANK_SELECTED = new Int8Array(6);

function rankBoard(board, turn, spec) {
  const { codes, counts } = specGroups(spec);
  const available = RANK_WORK;
  available.set(RANK_AVAILABLE);
  let availableLength = 25;
  let value = 0;
  for (let group = 0; group < codes.length; group += 1) {
    const code = codes[group];
    const count = counts[group];
    let selectedCount = 0;
    for (let position = 0; position < availableLength; position += 1) {
      const sq = available[position];
      if (board[sq] === code) RANK_SELECTED[selectedCount++] = position;
    }
    if (selectedCount !== count) throw new Error(`Position does not match ${spec.signature}.`);
    const radix = COMB[availableLength][count];
    value = value * radix + rankSelectedPositions(RANK_SELECTED, count, availableLength);
    if (count) {
      let write = 0;
      for (let read = 0; read < availableLength; read += 1) {
        const sq = available[read];
        if (board[sq] !== code) available[write++] = sq;
      }
      availableLength = write;
    }
  }
  return value * 2 + (turn === WHITE ? 0 : 1);
}

function exactCanonical(position) {
  const spec = materialSpec(position.board);
  if (!spec.swapped) return { spec, board: position.board, turn: position.turn, transform: 0 };
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
    if (!this.map.has(key)) return undefined;
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
    this.wdlBlocks = new Map();
    this.wdlWarmPromise = null;
    this.wdlWarmComplete = false;
    this.probeCache = new LruCache(8192);
    this.analysisCache = new LruCache(512);
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
    if (cached !== undefined) return cached;
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

  async exactWdlOnlyBlock(signature, metadata, blockId) {
    const key = `exact-wdl:${signature}:${blockId}`;
    const cached = this.wdlBlocks.get(key);
    if (cached !== undefined) return cached;
    const block = metadata.blocks[blockId];
    if (!block) throw new Error(`Missing exact WDL tablebase block ${signature}/${blockId}.`);
    const tableUrl = new URL(`${signature}/`, this.baseUrl);
    const wdlBytes = await gunzip(new URL(block.wdl, tableUrl).href);
    const wdl = new Int8Array(block.count);
    for (let index = 0; index < block.count; index += 1) {
      const code = (wdlBytes[index >>> 2] >>> ((index & 3) * 2)) & 3;
      wdl[index] = EXACT_MAP[code];
    }
    this.wdlBlocks.set(key, wdl);
    return wdl;
  }

  async warmExactWdl({ pieceLimit = 4, signatures = null } = {}) {
    if (this.wdlWarmComplete) return true;
    if (this.wdlWarmPromise) return this.wdlWarmPromise;
    this.wdlWarmPromise = (async () => {
      if (!(await this.init())) return false;
      const wanted = signatures ? new Set(signatures) : null;
      const entries = Object.keys(this.exactManifest.tables || {})
        .filter(signature => !wanted || wanted.has(signature))
        .filter(signature => {
          if (TRIVIAL_DRAW_SIGNATURES.has(signature) || MATE_IN_ONE_ONLY_SIGNATURES.has(signature)) return false;
          try {
            const text = signature.replace('v', '');
            return text.length <= pieceLimit;
          } catch {
            return false;
          }
        });
      for (const signature of entries) {
        const metadata = await this.metadataFor('exact', signature);
        const blocks = metadata.blocks || [];
        for (let blockId = 0; blockId < blocks.length; blockId += 1) {
          try {
            await this.exactWdlOnlyBlock(signature, metadata, blockId);
          } catch {
            // A missing/corrupt block should not disable the engine.  Search only
            // consumes WDL blocks that are already present in memory.
          }
          if ((blockId & 3) === 3) await new Promise(resolve => setTimeout(resolve, 0));
        }
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      this.wdlWarmComplete = true;
      return true;
    })().finally(() => { this.wdlWarmPromise = null; });
    return this.wdlWarmPromise;
  }

  probeWdlSync(position) {
    if (pieceCount(position) > 4) return null;
    const exact = exactCanonical(position);
    const lightweight = maybeLightweightProbe(position, exact);
    if (lightweight) return cloneProbeResult({ ...lightweight, source: `${lightweight.source}-sync` });
    const entry = this.exactManifest.tables?.[exact.spec.signature];
    if (!entry) return null;
    const metadata = this.metadata.get(`exact:${exact.spec.signature}`);
    if (!metadata) return null;
    const index = rankBoard(exact.board, exact.turn, exact.spec);
    const blockId = Math.floor(index / metadata.blockSize);
    const offset = index % metadata.blockSize;
    const wdl = this.wdlBlocks.get(`exact-wdl:${exact.spec.signature}:${blockId}`)
      || this.blocks.get(`exact:${exact.spec.signature}:${blockId}`)?.wdl;
    if (!wdl) return null;
    const value = wdl[offset];
    if (value === 2 || value === undefined) return null;
    return {
      wdl: value,
      dtmPly: 0,
      bestMove: 0,
      dtmUpperBound: true,
      source: 'exact-wdl-sync',
      signature: exact.spec.signature,
      index
    };
  }

  async practicalBlock(signature, metadata, blockId) {
    const key = `practical:${signature}:${blockId}`;
    const cached = this.blocks.get(key);
    if (cached !== undefined) return cached;
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
    if (pieceCount(position) > 6) return null;
    const cacheKey = tablebaseKey(position);
    const cached = this.probeCache.get(cacheKey);
    if (cached !== undefined) return cloneProbeResult(cached);
    const exact = exactCanonical(position);
    const lightweight = maybeLightweightProbe(position, exact);
    if (lightweight) {
      this.probeCache.set(cacheKey, lightweight);
      return cloneProbeResult(lightweight);
    }
    if (!(await this.init())) {
      this.probeCache.set(cacheKey, null);
      return null;
    }

    if (this.exactManifest.tables?.[exact.spec.signature]) {
      const metadata = await this.metadataFor('exact', exact.spec.signature);
      const index = rankBoard(exact.board, exact.turn, exact.spec);
      const blockId = Math.floor(index / metadata.blockSize);
      const offset = index % metadata.blockSize;
      const block = await this.exactBlock(exact.spec.signature, metadata, blockId);
      const wdl = block.wdl[offset];
      if (wdl === 2 || wdl === undefined) {
        this.probeCache.set(cacheKey, null);
        return null;
      }
      const result = {
        wdl,
        dtmPly: Number(block.dtm[offset] || 0),
        bestMove: 0,
        dtmUpperBound: false,
        source: 'exact-core',
        signature: exact.spec.signature,
        index
      };
      this.probeCache.set(cacheKey, result);
      return cloneProbeResult(result);
    }

    if (!this.practicalManifest.tables?.[exact.spec.signature]) {
      this.probeCache.set(cacheKey, null);
      return null;
    }
    const practical = practicalCanonical(position);
    if (!this.practicalManifest.tables?.[practical.spec.signature]) {
      this.probeCache.set(cacheKey, null);
      return null;
    }
    const metadata = await this.metadataFor('practical', practical.spec.signature);
    const blocks = metadata.blocks || [];
    let low = 0, high = blocks.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (blocks[middle].maxIndex < practical.index) low = middle + 1;
      else high = middle;
    }
    if (low >= blocks.length || practical.index < blocks[low].minIndex) {
      this.probeCache.set(cacheKey, null);
      return null;
    }
    const block = await this.practicalBlock(practical.spec.signature, metadata, low);
    const offset = binarySearch(block.indices, practical.index >>> 0);
    if (offset >= block.indices.length || block.indices[offset] !== (practical.index >>> 0)) {
      this.probeCache.set(cacheKey, null);
      return null;
    }
    const packed = packedValue(block.values[offset]);
    packed.bestMove = transformPackedMove(packed.bestMove, practical.transform);
    const result = {
      ...packed,
      source: 'practical-verified',
      signature: practical.spec.signature,
      index: practical.index
    };
    this.probeCache.set(cacheKey, result);
    return cloneProbeResult(result);
  }

  async chooseMoves(position, rootProbe, limit = 3) {
    const legal = generateLegalMoves(position, false);
    const childPositions = [];
    for (let i = 0; i < legal.length; i += 1) {
      const move = legal[i];
      const state = makeMove(position, move);
      const child = position.clone();
      undoMove(position, move, state);
      childPositions.push({ move, child });
    }
    const probes = await Promise.all(childPositions.map(item => this.probe(item.child).catch(() => null)));
    const candidates = [];
    for (let i = 0; i < childPositions.length; i += 1) {
      const child = probes[i];
      if (!child) continue;
      candidates.push({
        move: childPositions[i].move,
        child,
        wdl: -child.wdl,
        dtmPly: child.wdl === 0 ? 0 : Number(child.dtmPly || 0) + 1
      });
    }

    if (!candidates.length && rootProbe.bestMove) {
      for (let i = 0; i < legal.length; i += 1) {
        if (legal[i] === rootProbe.bestMove) return [{ move: rootProbe.bestMove, child: null, wdl: rootProbe.wdl, dtmPly: rootProbe.dtmPly }];
      }
    }
    const matching = [];
    for (let i = 0; i < candidates.length; i += 1) if (candidates[i].wdl === rootProbe.wdl) matching.push(candidates[i]);
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


  async dtmBoundForLine(position, line, { maxProbePly = 24 } = {}) {
    if (!line?.pv?.length) return null;
    const root = position.clone();
    const cursor = position.clone();
    const rootSide = root.turn;
    const pv = Array.isArray(line.pv) ? line.pv.slice(0, maxProbePly) : [];

    for (let ply = 0; ply <= pv.length && ply <= maxProbePly; ply += 1) {
      if (pieceCount(cursor) <= 4) {
        const probe = await this.probe(cursor).catch(() => null);
        const dtm = Number(probe?.dtmPly || 0);
        if (probe && probe.wdl !== 0 && Number.isFinite(dtm) && dtm > 0) {
          const winningSide = probe.wdl > 0 ? cursor.turn : -cursor.turn;
          const rootWdl = winningSide === rootSide ? 1 : -1;
          return {
            wdl: rootWdl,
            dtmPly: Math.max(1, ply + dtm),
            tablebaseWdl: probe.wdl,
            tablebaseSource: probe.source,
            tablebaseSignature: probe.signature,
            dtmUpperBound: ply > 0 || Boolean(probe.dtmUpperBound),
            exactDtm: !probe.dtmUpperBound
          };
        }
      }
      if (ply >= pv.length) break;
      const move = uciToMove(cursor, pv[ply]);
      if (!move) break;
      makeMove(cursor, move);
    }
    return null;
  }

  async annotateResultWithDtmBounds(position, result, { maxLines = 5, maxProbePly = 24 } = {}) {
    if (!result?.lines?.length || result.tablebase || result.fortressProof) return result;
    const lines = result.lines.map(line => ({ ...line, pv: Array.isArray(line.pv) ? line.pv.slice() : [] }));
    const rootSide = position.turn;
    let changed = false;
    for (const line of lines.slice(0, Math.max(1, maxLines))) {
      if (line.mateVerified || line.fortressProof) continue;
      const bound = await this.dtmBoundForLine(position, line, { maxProbePly });
      if (!bound) continue;
      const rootScore = bound.wdl > 0
        ? MATE - Math.max(1, bound.dtmPly)
        : -MATE + Math.max(1, bound.dtmPly);
      const whiteScore = rootSide === WHITE ? rootScore : -rootScore;
      line.score = whiteScore;
      line.scoreText = `${scoreToDisplay(whiteScore)} · TB bound`;
      line.dtm = bound.dtmPly;
      line.dtmUpperBound = true;
      line.tablebase = true;
      line.tablebaseBound = true;
      line.tablebaseExactDtm = Boolean(bound.exactDtm);
      line.tablebaseWdl = bound.wdl;
      line.source = bound.tablebaseSource;
      line.tablebaseSource = bound.tablebaseSource;
      line.tablebaseSignature = bound.tablebaseSignature;
      changed = true;
    }
    if (!changed) return result;
    lines.sort((a, b) => {
      const utilityA = rootSide === WHITE ? Number(a.score || 0) : -Number(a.score || 0);
      const utilityB = rootSide === WHITE ? Number(b.score || 0) : -Number(b.score || 0);
      return utilityB - utilityA;
    });
    return {
      ...result,
      lines,
      tablebaseProbeHits: Number(result.tablebaseProbeHits || 0),
      tablebaseDtmBound: true
    };
  }


  async analyze(position, { multipv = 3, maxPvPly = 96 } = {}) {
    const analyzeKey = `${tablebaseKey(position)}:m${Math.max(1, multipv | 0)}:pv${Math.max(1, maxPvPly | 0)}`;
    const cachedAnalysis = this.analysisCache.get(analyzeKey);
    if (cachedAnalysis !== undefined) return cloneAnalyzeResult(cachedAnalysis);
    const root = position.clone();
    const probe = await this.probe(root);
    if (!probe) {
      this.analysisCache.set(analyzeKey, null);
      return null;
    }
    const choices = await this.chooseMoves(root, probe, multipv);
    const rootSide = root.turn;
    const lines = [];
    for (const choice of choices) {
      const pvMoves = await this.buildPv(root, choice.move, maxPvPly);
      const choiceDtm = Number(choice.dtmPly);
      const probeDtm = Number(probe.dtmPly);
      const rootDtm = Number.isFinite(choiceDtm)
        ? choiceDtm
        : Number.isFinite(probeDtm)
          ? probeDtm
          : pvMoves.length;
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
        candidate.tablebaseBound = Boolean(probe.dtmUpperBound);
        candidate.tablebaseExactDtm = !probe.dtmUpperBound;
        candidate.scoreText = `${candidate.scoreText} · TB`;
      }
      lines.push(candidate);
    }
    // Sparse practical records deliberately omit unproved child states. A WDL
    // hit without a legal proved continuation is still useful to the normal
    // search, but it is not sufficient to manufacture a move or a PV. Falling
    // back here avoids presenting an arbitrary legal move as a tablebase line.
    if (!lines.length) {
      this.analysisCache.set(analyzeKey, null);
      return null;
    }
    const result = {
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
    this.analysisCache.set(analyzeKey, cloneAnalyzeResult(result));
    return result;
  }
}

export const TablebaseInternals = Object.freeze({
  materialSpec,
  exactCanonical,
  practicalCanonical,
  rankBoard,
  transformPackedMove,
  maybeLightweightProbe,
  TRIVIAL_DRAW_SIGNATURES,
  MATE_IN_ONE_ONLY_SIGNATURES,
  gunzip,
  DEFAULT_BASE
});
