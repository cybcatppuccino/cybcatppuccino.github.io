import {
  ENGINE_VERSION,
  EngineInternals,
  generateLegalMoves,
  isInCheck, evaluate,
  moveToUci,
  scoreToDisplay,
  validateMateResult,
  uciToMove
} from './engine.js';
import { EMBEDDED_EXACT_MANIFEST } from './tablebase-manifest.js';

const {
  PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK,
  makeMove, undoMove, moveFrom, moveTo, movePromotion, encodeMove,
  isCapture, isPromotion, givesCheck, MATE, sideOf, typeOf, fileOf, rankOf
} = EngineInternals;

const MATERIAL_ORDER = Object.freeze([
  ['Q', QUEEN], ['R', ROOK], ['B', BISHOP], ['N', KNIGHT], ['P', PAWN]
]);
const EXACT_MAP = Object.freeze([-1, 0, 1, 2]);
const TRANSFORM_MIRROR_FILES = 1;
const TRANSFORM_ROTATE_SWAP = 2;
const DEFAULT_BASE = new URL('../../tools/gardner_tablebase/tables/', import.meta.url).href;
const FALLBACK_BASES = Object.freeze([
  new URL('../../tools/gardner_tablebase/tables/', import.meta.url).href,
  new URL('/tools/gardner_tablebase/tables/', globalThis.location?.href || import.meta.url).href,
  new URL('./tools/gardner_tablebase/tables/', globalThis.location?.href || import.meta.url).href
]);
const MAX_EXACT_TABLEBASE_PIECES = 5;
// The published GTB set is a fixed, complete 111-table corpus. Keep a stable
// cache namespace instead of tying binary URLs to each UI release.
const TABLEBASE_CACHE_BUSTER = 'gtb-111-stable';
const MIN_EXPECTED_EXACT_TABLES = 111;
const TRIVIAL_DRAW_SIGNATURES = Object.freeze(new Set(['KvK', 'KBvK', 'KNvK']));
const MATE_IN_ONE_ONLY_SIGNATURES = Object.freeze(new Set(['KBvKB', 'KBvKN', 'KNNvK', 'KNvKN']));

function bridgeControllerKingPressureBonus(position, move, controller) {
  if (!position?.board) return 0;
  if (typeOf(position.board[moveFrom(move)]) !== KING) return 0;
  const to = moveTo(move);
  let nearestEnemyPawn = Infinity;
  let enemyPawns = 0;
  for (let sq = 0; sq < position.board.length; sq += 1) {
    const piece = position.board[sq];
    if (sideOf(piece) !== -controller || typeOf(piece) !== PAWN) continue;
    enemyPawns += 1;
    const distance = Math.max(Math.abs(fileOf(to) - fileOf(sq)), Math.abs(rankOf(to) - rankOf(sq)));
    nearestEnemyPawn = Math.min(nearestEnemyPawn, distance);
  }
  if (!enemyPawns || !Number.isFinite(nearestEnemyPawn)) return 0;
  // In near-tablebase pawn endings, the winning king usually has to either
  // blockade or approach the defender's pawns before the proof can collapse
  // into an exact five-piece leaf.  This is only an ordering bonus for the
  // bounded controller candidate set; defender replies are still exhaustive and
  // every published result is independently verified.
  return Math.max(0, 6 - nearestEnemyPawn) * 250_000;
}

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


function filterExactManifestTables(tables) {
  const output = {};
  for (const [signature, entry] of Object.entries(tables || {})) {
    const pieces = Number(entry?.pieceCount || signature.replace('v', '').length || 0);
    if (pieces > 0 && pieces <= MAX_EXACT_TABLEBASE_PIECES) output[signature] = entry;
  }
  return output;
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

function cacheBustedUrl(url) {
  const output = new URL(url, globalThis.location?.href || import.meta.url);
  output.searchParams.set('tbv', TABLEBASE_CACHE_BUSTER);
  return output.href;
}

function mergeExactManifestTables(fetched = {}) {
  const embedded = filterExactManifestTables(EMBEDDED_EXACT_MANIFEST.tables || {});
  const remote = filterExactManifestTables(fetched.tables || {});
  return {
    ...EMBEDDED_EXACT_MANIFEST,
    ...fetched,
    tables: { ...embedded, ...remote }
  };
}

async function fetchManifestJson(url) {
  try {
    return await fetchJson(cacheBustedUrl(url), { cache: 'force-cache' });
  } catch {
    return fetchJson(url, { cache: 'default' });
  }
}

async function fetchMetadataJson(url) {
  try {
    return await fetchJson(cacheBustedUrl(url), { cache: 'force-cache' });
  } catch {
    return fetchJson(url, { cache: 'default' });
  }
}

function looksLikeTextError(bytes) {
  if (!bytes || !bytes.length) return false;
  const sample = new TextDecoder().decode(bytes.slice(0, Math.min(96, bytes.length))).trimStart().toLowerCase();
  return sample.startsWith('<!doctype')
    || sample.startsWith('<html')
    || sample.startsWith('version https://git-lfs.github.com/spec/');
}

async function fetchBytes(url, { cache = 'force-cache' } = {}) {
  const response = await fetch(url, { cache });
  if (!response.ok) throw new Error(`Tablebase request failed (${response.status}) for ${url}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (looksLikeTextError(bytes)) {
    const sample = new TextDecoder().decode(bytes.slice(0, Math.min(96, bytes.length))).replace(/\s+/g, ' ').trim();
    throw new Error(`Tablebase binary payload was not downloaded correctly for ${url}: ${sample}`);
  }
  return bytes;
}

async function fetchJson(url, { cache = 'force-cache' } = {}) {
  const response = await fetch(url, { cache });
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


function tablebaseBoundScoreText(score) {
  const text = scoreToDisplay(score);
  return Math.abs(Number(score || 0)) >= 29000 ? `≤${text} · TB bound` : `${text} · TB bound`;
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
  constructor({
    baseUrl = DEFAULT_BASE,
    maxCachedBlocks = 128,
    maxCachedWdlBlocks = 96,
    maxConcurrentRequests = 3
  } = {}) {
    this.baseUrl = new URL(baseUrl, import.meta.url).href;
    this.baseCandidates = [...new Set([this.baseUrl, ...FALLBACK_BASES])];
    this.maxCachedBlocks = maxCachedBlocks;
    this.initialized = false;
    this.available = false;
    this.exactManifest = { tables: {} };
    this.metadata = new Map();
    this.blocks = new LruCache(maxCachedBlocks);
    // WDL blocks used to grow without limit. Keep WDL in its own LRU budget so
    // a long study cannot exhaust a worker just by visiting many endgames.
    this.wdlBlocks = new LruCache(maxCachedWdlBlocks);
    this.metadataPromises = new Map();
    this.blockPromises = new Map();
    this.wdlBlockPromises = new Map();
    this.wdlWarmPromise = null;
    this.wdlWarmComplete = false;
    this.wdlWarmSignaturePromises = new Map();
    this.wdlWarmSignatures = new Set();
    this.probeCache = new LruCache(8192);
    this.wdlProbeCache = new LruCache(8192);
    this.analysisCache = new LruCache(512);
    this.initPromise = null;
    this.lastError = '';
    this.maxConcurrentRequests = Math.max(1, Math.min(6, Number(maxConcurrentRequests) || 3));
    this.activeRequests = 0;
    this.requestSequence = 0;
    this.requestQueue = [];
  }

  enqueueRequest(task, priority = 0) {
    return new Promise((resolve, reject) => {
      this.requestQueue.push({ task, priority, sequence: this.requestSequence++, resolve, reject });
      this.requestQueue.sort((left, right) => left.priority - right.priority || left.sequence - right.sequence);
      this.drainRequestQueue();
    });
  }

  drainRequestQueue() {
    while (this.activeRequests < this.maxConcurrentRequests && this.requestQueue.length) {
      const next = this.requestQueue.shift();
      this.activeRequests += 1;
      Promise.resolve()
        .then(next.task)
        .then(next.resolve, next.reject)
        .finally(() => {
          this.activeRequests -= 1;
          this.drainRequestQueue();
        });
    }
  }

  loadGzip(url, priority = 0) {
    return this.enqueueRequest(() => gunzip(url), priority);
  }

  async init() {
    if (this.initialized) return this.available;
    if (this.initPromise) return this.initPromise;
    this.initPromise = (async () => {
      const errors = [];
      for (const candidate of this.baseCandidates) {
        try {
          const exact = await fetchManifestJson(new URL('manifest.json', candidate).href);
          this.baseUrl = candidate;
          const merged = mergeExactManifestTables(exact);
          this.exactManifest = { ...merged, tables: filterExactManifestTables(merged.tables || {}) };
          const tableCount = Object.keys(this.exactManifest.tables || {}).length;
          this.available = tableCount > 0;
          if (tableCount < MIN_EXPECTED_EXACT_TABLES) {
            this.lastError = `Only ${tableCount} exact tablebase tables were discovered; the fixed 111-table fallback manifest remains active.`;
          }
          break;
        } catch (error) {
          errors.push(`${candidate}: ${error?.message || error}`);
        }
      }
      if (!this.available) {
        const embedded = mergeExactManifestTables({});
        // When a deployment omits manifest.json but serves the fixed table
        // folders, use the first non-file candidate as the binary root. Node
        // worker tests and browser workers cannot fetch file:// metadata even
        // though the embedded manifest is available in the JS bundle.
        this.baseUrl = this.baseCandidates.find(candidate => {
          try { return new URL(candidate).protocol !== 'file:'; } catch { return false; }
        }) || this.baseCandidates[0];
        this.exactManifest = { ...embedded, tables: filterExactManifestTables(embedded.tables || {}) };
        this.available = Boolean(Object.keys(this.exactManifest.tables || {}).length);
        if (this.available) this.lastError = `Using embedded fixed 111-table manifest after remote manifest load failed: ${errors.join(' · ')}`;
      }
      this.initialized = true;
      if (!this.available) this.lastError = errors.join(' · ') || 'No Gardner exact tablebase manifest was found.';
      return this.available;
    })();
    return this.initPromise;
  }

  async metadataFor(signature, { priority = 0 } = {}) {
    const key = `exact:${signature}`;
    if (this.metadata.has(key)) return this.metadata.get(key);
    if (this.metadataPromises.has(key)) return this.metadataPromises.get(key);
    const promise = (async () => {
      const entry = this.exactManifest.tables?.[signature];
      if (!entry) throw new Error(`No exact table for ${signature}.`);
      const metadata = await this.enqueueRequest(
        () => fetchMetadataJson(new URL(entry.path, this.baseUrl).href),
        priority
      );
      if (!metadata || !Array.isArray(metadata.blocks) || !Number.isFinite(Number(metadata.blockSize))) {
        throw new Error(`Invalid exact metadata for ${signature}.`);
      }
      this.metadata.set(key, metadata);
      return metadata;
    })().finally(() => { this.metadataPromises.delete(key); });
    this.metadataPromises.set(key, promise);
    return promise;
  }

  async exactBlock(signature, metadata, blockId, { priority = 0 } = {}) {
    const key = `exact:${signature}:${blockId}`;
    const cached = this.blocks.get(key);
    if (cached !== undefined) return cached;
    if (this.blockPromises.has(key)) return this.blockPromises.get(key);
    const promise = (async () => {
      const block = metadata.blocks[blockId];
      if (!block) throw new Error(`Missing exact tablebase block ${signature}/${blockId}.`);
      const tableUrl = new URL(`${signature}/`, this.baseUrl);
      const wdlKey = `exact-wdl:${signature}:${blockId}`;
      let wdl = this.wdlBlocks.get(wdlKey);
      const dtmPromise = this.loadGzip(new URL(block.dtm, tableUrl).href, priority);
      if (!wdl) {
        const wdlBytes = await this.loadGzip(new URL(block.wdl, tableUrl).href, priority);
        if (wdlBytes.length * 4 < block.count) throw new Error(`Short WDL block ${signature}/${blockId}.`);
        wdl = new Int8Array(block.count);
        for (let index = 0; index < block.count; index += 1) {
          const code = (wdlBytes[index >>> 2] >>> ((index & 3) * 2)) & 3;
          wdl[index] = EXACT_MAP[code];
        }
        this.wdlBlocks.set(wdlKey, wdl);
      }
      const dtmBytes = await dtmPromise;
      if (dtmBytes.length < block.count * 2) throw new Error(`Short DTM block ${signature}/${blockId}.`);
      const value = { wdl, dtm: uint16LE(dtmBytes) };
      this.blocks.set(key, value);
      return value;
    })().finally(() => { this.blockPromises.delete(key); });
    this.blockPromises.set(key, promise);
    return promise;
  }

  async exactWdlOnlyBlock(signature, metadata, blockId, { priority = 0 } = {}) {
    const key = `exact-wdl:${signature}:${blockId}`;
    const cached = this.wdlBlocks.get(key);
    if (cached !== undefined) return cached;
    if (this.wdlBlockPromises.has(key)) return this.wdlBlockPromises.get(key);
    const promise = (async () => {
      const full = this.blocks.get(`exact:${signature}:${blockId}`);
      if (full?.wdl) {
        this.wdlBlocks.set(key, full.wdl);
        return full.wdl;
      }
      const block = metadata.blocks[blockId];
      if (!block) throw new Error(`Missing exact WDL tablebase block ${signature}/${blockId}.`);
      const tableUrl = new URL(`${signature}/`, this.baseUrl);
      const wdlBytes = await this.loadGzip(new URL(block.wdl, tableUrl).href, priority);
      if (wdlBytes.length * 4 < block.count) throw new Error(`Short WDL block ${signature}/${blockId}.`);
      const wdl = new Int8Array(block.count);
      for (let index = 0; index < block.count; index += 1) {
        const code = (wdlBytes[index >>> 2] >>> ((index & 3) * 2)) & 3;
        wdl[index] = EXACT_MAP[code];
      }
      this.wdlBlocks.set(key, wdl);
      return wdl;
    })().finally(() => { this.wdlBlockPromises.delete(key); });
    this.wdlBlockPromises.set(key, promise);
    return promise;
  }

  async warmExactWdl({ pieceLimit = MAX_EXACT_TABLEBASE_PIECES, signatures = null } = {}) {
    const partial = Array.isArray(signatures) && signatures.length > 0;
    const normalizeEntries = () => Object.keys(this.exactManifest.tables || {})
      .filter(signature => !partial || signatures.includes(signature))
      .filter(signature => {
        if (TRIVIAL_DRAW_SIGNATURES.has(signature) || MATE_IN_ONE_ONLY_SIGNATURES.has(signature)) return false;
        try {
          const text = signature.replace('v', '');
          return text.length <= pieceLimit;
        } catch {
          return false;
        }
      });

    const warmEntries = async entries => {
      if (!(await this.init())) return false;
      let any = false;
      for (const signature of entries) {
        if (partial && this.wdlWarmSignatures.has(signature)) {
          any = true;
          continue;
        }
        let metadata = null;
        try {
          metadata = await this.metadataFor(signature, { priority: 2 });
        } catch {
          continue;
        }
        const blocks = metadata.blocks || [];
        let signatureOk = blocks.length > 0;
        for (let blockId = 0; blockId < blocks.length; blockId += 1) {
          try {
            await this.exactWdlOnlyBlock(signature, metadata, blockId, { priority: 2 });
          } catch {
            signatureOk = false;
            // A missing/corrupt block should not disable the engine. Search only
            // consumes WDL blocks that are already present in memory.
          }
          if ((blockId & 3) === 3) await new Promise(resolve => setTimeout(resolve, 0));
        }
        if (signatureOk) {
          any = true;
          if (partial) this.wdlWarmSignatures.add(signature);
        }
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      return any;
    };

    if (!partial) {
      if (this.wdlWarmComplete) return true;
      if (this.wdlWarmPromise) return this.wdlWarmPromise;
      this.wdlWarmPromise = (async () => {
        const ok = await warmEntries(normalizeEntries());
        if (ok) this.wdlWarmComplete = true;
        return ok;
      })().finally(() => { this.wdlWarmPromise = null; });
      return this.wdlWarmPromise;
    }

    if (!(await this.init())) return false;
    const entries = normalizeEntries().filter(signature => !this.wdlWarmSignatures.has(signature));
    if (!entries.length) return true;
    const key = entries.slice().sort().join('|');
    if (this.wdlWarmSignaturePromises.has(key)) return this.wdlWarmSignaturePromises.get(key);
    const promise = warmEntries(entries).finally(() => { this.wdlWarmSignaturePromises.delete(key); });
    this.wdlWarmSignaturePromises.set(key, promise);
    return promise;
  }

  probeExactSync(position) {
    // This never performs I/O. It returns an exact WDL+DTM record only when
    // the matching decompressed block is already resident in memory. Search can
    // therefore use it as a safe terminal node without blocking the worker.
    if (pieceCount(position) > MAX_EXACT_TABLEBASE_PIECES) return null;
    let exact;
    try { exact = exactCanonical(position); } catch { return null; }
    const lightweight = maybeLightweightProbe(position, exact);
    if (lightweight) return cloneProbeResult({
      ...lightweight,
      exactDtm: true,
      source: `${lightweight.source}-exact-sync`
    });
    const entry = this.exactManifest.tables?.[exact.spec.signature];
    if (!entry) return null;
    const metadata = this.metadata.get(`exact:${exact.spec.signature}`);
    if (!metadata) return null;
    const index = rankBoard(exact.board, exact.turn, exact.spec);
    const blockId = Math.floor(index / metadata.blockSize);
    const offset = index % metadata.blockSize;
    const block = this.blocks.get(`exact:${exact.spec.signature}:${blockId}`);
    if (!block?.wdl || !block?.dtm) return null;
    const wdl = block.wdl[offset];
    if (wdl === 2 || wdl === undefined) return null;
    return {
      wdl,
      dtmPly: Number(block.dtm[offset] || 0),
      bestMove: 0,
      dtmUpperBound: false,
      exactDtm: true,
      source: 'exact-core-sync',
      signature: exact.spec.signature,
      index
    };
  }

  probeWdlSync(position) {
    if (pieceCount(position) > MAX_EXACT_TABLEBASE_PIECES) return null;
    let exact;
    try { exact = exactCanonical(position); } catch { return null; }
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
      exactDtm: false,
      source: 'exact-wdl-sync',
      signature: exact.spec.signature,
      index
    };
  }

  probeSync(position) {
    // Prefer exact DTM whenever a full block was preloaded; otherwise retain
    // the existing WDL-only fast cut-off. Both results are tablebase exact for
    // W/D/L, but only the first is allowed to drive a DTM/PV tail.
    return this.probeExactSync(position) || this.probeWdlSync(position);
  }

  async warmExactForPosition(position, { priority = 1 } = {}) {
    if (pieceCount(position) > MAX_EXACT_TABLEBASE_PIECES) return false;
    if (!(await this.init())) return false;
    let exact;
    try { exact = exactCanonical(position); } catch { return false; }
    if (!this.exactManifest.tables?.[exact.spec.signature]) return false;
    if (TRIVIAL_DRAW_SIGNATURES.has(exact.spec.signature) || MATE_IN_ONE_ONLY_SIGNATURES.has(exact.spec.signature)) return true;
    try {
      const metadata = await this.metadataFor(exact.spec.signature, { priority });
      const index = rankBoard(exact.board, exact.turn, exact.spec);
      const blockId = Math.floor(index / metadata.blockSize);
      await this.exactBlock(exact.spec.signature, metadata, blockId, { priority });
      return true;
    } catch {
      return false;
    }
  }

  async warmExactFrontier(position, { maxPly = 4, maxStates = 320, priority = 1 } = {}) {
    // A 6-piece root is not itself a GTB root, but it can often force a capture
    // or promotion into a supplied <=5-piece table within a few plies. Warm those
    // first-hit leaves in the background so alpha-beta can cut them off synchronously.
    const root = position?.clone?.();
    if (!root || pieceCount(root) <= MAX_EXACT_TABLEBASE_PIECES || pieceCount(root) > MAX_EXACT_TABLEBASE_PIECES + 1) {
      return { visited: 0, targets: 0, warmed: 0 };
    }
    const targets = new Map();
    const seen = new Set();
    const queue = [{ position: root, ply: 0 }];
    let visited = 0;
    const limit = Math.max(1, Math.floor(Number(maxStates || 320)));
    const plyLimit = Math.max(1, Math.floor(Number(maxPly || 4)));
    while (queue.length && visited < limit) {
      const item = queue.shift();
      const node = item.position;
      const identity = `${node.hashA}:${node.hashB}:t${node.turn}:p${node.pieceCount}`;
      if (seen.has(identity)) continue;
      seen.add(identity);
      visited += 1;
      if (pieceCount(node) <= MAX_EXACT_TABLEBASE_PIECES) {
        targets.set(tablebaseKey(node), node);
        continue;
      }
      if (item.ply >= plyLimit) continue;
      let legal = [];
      try { legal = generateLegalMoves(node, false); } catch { legal = []; }
      for (const move of legal) {
        const state = makeMove(node, move);
        const child = node.clone();
        undoMove(node, move, state);
        if (pieceCount(child) <= MAX_EXACT_TABLEBASE_PIECES) targets.set(tablebaseKey(child), child);
        else if (item.ply + 1 < plyLimit) queue.push({ position: child, ply: item.ply + 1 });
        if (queue.length + targets.size >= limit * 2) break;
      }
    }
    let warmed = 0;
    for (const target of targets.values()) {
      if (await this.warmExactForPosition(target, { priority })) warmed += 1;
      if ((warmed & 3) === 3) await new Promise(resolve => setTimeout(resolve, 0));
    }
    return { visited, targets: targets.size, warmed };
  }


  async warmExactBridgeTables(position, { maxPly = 4, maxStates = 320, maxBlocks = 36, priority = 1, seedSignatures = [] } = {}) {
    // Discover the material signatures at the first <=5-piece frontier, then
    // warm whole small signatures before starting an AND/OR bridge proof.  The
    // proof itself only consumes resident exact blocks, so tablebase I/O never
    // becomes part of the logical search budget.
    const root = position?.clone?.();
    if (!root || pieceCount(root) <= MAX_EXACT_TABLEBASE_PIECES || pieceCount(root) > MAX_EXACT_TABLEBASE_PIECES + 2 || !(await this.init())) {
      return { visited: 0, signatures: [], blocks: 0, warmed: false };
    }
    const queue = [{ position: root, ply: 0 }];
    const seen = new Set();
    const signatures = new Map();
    // A stable PV may enter GTB beyond the bounded frontier scan.  Its exact
    // material signature is still a safe pre-warm seed: it only enables
    // resident probing and never grants proof status by itself.
    for (const signature of Array.isArray(seedSignatures) ? seedSignatures : []) {
      const text = String(signature || '');
      if (text && this.exactManifest.tables?.[text]) signatures.set(text, Number.MAX_SAFE_INTEGER);
    }
    const limit = Math.max(1, Math.floor(Number(maxStates || 320)));
    const plyLimit = Math.max(1, Math.floor(Number(maxPly || 4)));
    let visited = 0;
    while (queue.length && visited < limit) {
      const item = queue.shift();
      const node = item.position;
      const identity = this.bridgePositionKey(node);
      if (seen.has(identity)) continue;
      seen.add(identity);
      visited += 1;
      if (pieceCount(node) <= MAX_EXACT_TABLEBASE_PIECES) {
        try {
          const exact = exactCanonical(node);
          signatures.set(exact.spec.signature, (signatures.get(exact.spec.signature) || 0) + 1);
        } catch {}
        continue;
      }
      if (item.ply >= plyLimit) continue;
      let legal = [];
      try { legal = generateLegalMoves(node, false); } catch { legal = []; }
      for (const move of legal) {
        const state = makeMove(node, move);
        const child = node.clone();
        undoMove(node, move, state);
        if (pieceCount(child) <= MAX_EXACT_TABLEBASE_PIECES) {
          try {
            const exact = exactCanonical(child);
            signatures.set(exact.spec.signature, (signatures.get(exact.spec.signature) || 0) + 1);
          } catch {}
        } else if (item.ply + 1 < plyLimit) queue.push({ position: child, ply: item.ply + 1 });
        if (queue.length >= limit * 2) break;
      }
    }
    let remaining = Math.max(1, Math.floor(Number(maxBlocks || 36)));
    let blocks = 0;
    const warmed = [];
    const ordered = [...signatures.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]));
    for (const [signature] of ordered) {
      let metadata = null;
      try { metadata = await this.metadataFor(signature, { priority }); } catch { continue; }
      const count = metadata.blocks?.length || 0;
      // Proof leaves need full signature residency.  Skip a large family rather
      // than partially warm it and accidentally turn I/O timing into a proof
      // outcome.
      if (!count || count > remaining) continue;
      try {
        await Promise.all(metadata.blocks.map((_, blockId) => this.exactBlock(signature, metadata, blockId, { priority })));
        warmed.push(signature);
        blocks += count;
        remaining -= count;
      } catch {
        // A missing table remains an ordinary-search position.
      }
      if (remaining <= 0) break;
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    return { visited, signatures: warmed, blocks, warmed: warmed.length > 0 };
  }



  async probeWdl(position, { priority = 0 } = {}) {
    if (pieceCount(position) > MAX_EXACT_TABLEBASE_PIECES) return null;
    const cacheKey = tablebaseKey(position);
    const cached = this.wdlProbeCache.get(cacheKey);
    if (cached !== undefined) return cloneProbeResult(cached);
    let exact;
    try {
      exact = exactCanonical(position);
    } catch {
      this.wdlProbeCache.set(cacheKey, null);
      return null;
    }
    const lightweight = maybeLightweightProbe(position, exact);
    if (lightweight) {
      const result = { ...lightweight, source: `${lightweight.source}-wdl` };
      this.wdlProbeCache.set(cacheKey, result);
      return cloneProbeResult(result);
    }
    if (!(await this.init())) return null;
    if (!this.exactManifest.tables?.[exact.spec.signature]) {
      this.wdlProbeCache.set(cacheKey, null);
      return null;
    }
    try {
      const metadata = await this.metadataFor(exact.spec.signature, { priority });
      const index = rankBoard(exact.board, exact.turn, exact.spec);
      const blockId = Math.floor(index / metadata.blockSize);
      const offset = index % metadata.blockSize;
      const wdl = await this.exactWdlOnlyBlock(exact.spec.signature, metadata, blockId, { priority });
      const value = wdl[offset];
      if (value === 2 || value === undefined) {
        this.wdlProbeCache.set(cacheKey, null);
        return null;
      }
      const result = {
        wdl: value,
        dtmPly: 0,
        bestMove: 0,
        dtmUpperBound: true,
        source: 'exact-wdl',
        signature: exact.spec.signature,
        index
      };
      this.wdlProbeCache.set(cacheKey, result);
      return cloneProbeResult(result);
    } catch {
      // Do not memoize transient network/decompression failures as a permanent
      // miss. On GitHub Pages a block can become available after the first
      // request or after cache invalidation, and analysis should retry.
      return null;
    }
  }

  async warmExactWdlForPosition(position, { priority = 2 } = {}) {
    if (pieceCount(position) > MAX_EXACT_TABLEBASE_PIECES) return false;
    if (!(await this.init())) return false;
    const exact = exactCanonical(position);
    if (!this.exactManifest.tables?.[exact.spec.signature]) return false;
    if (TRIVIAL_DRAW_SIGNATURES.has(exact.spec.signature) || MATE_IN_ONE_ONLY_SIGNATURES.has(exact.spec.signature)) return true;
    try {
      const metadata = await this.metadataFor(exact.spec.signature, { priority });
      const index = rankBoard(exact.board, exact.turn, exact.spec);
      const blockId = Math.floor(index / metadata.blockSize);
      await this.exactWdlOnlyBlock(exact.spec.signature, metadata, blockId, { priority });
      return true;
    } catch {
      return false;
    }
  }

  async warmExactWdlNeighborhood(position, { includeLegalChildren = true } = {}) {
    // Background warm-up deliberately submits low-priority work one position at
    // a time. Direct root probes remain ahead of it in the shared request queue.
    let warmed = await this.warmExactWdlForPosition(position, { priority: 2 });
    if (!includeLegalChildren) return warmed;
    try {
      const legal = generateLegalMoves(position, false);
      for (let index = 0; index < legal.length; index += 1) {
        const state = makeMove(position, legal[index]);
        const child = position.clone();
        undoMove(position, legal[index], state);
        warmed = (await this.warmExactWdlForPosition(child, { priority: 2 })) || warmed;
        if ((index & 1) === 1) await new Promise(resolve => setTimeout(resolve, 0));
      }
    } catch {
      // Prefetch is optional; direct tablebase probes remain authoritative.
    }
    return warmed;
  }

  async probe(position, { priority = 0 } = {}) {
    if (pieceCount(position) > MAX_EXACT_TABLEBASE_PIECES) return null;
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
      try {
        const metadata = await this.metadataFor(exact.spec.signature, { priority });
        const index = rankBoard(exact.board, exact.turn, exact.spec);
        const blockId = Math.floor(index / metadata.blockSize);
        const offset = index % metadata.blockSize;
        const block = await this.exactBlock(exact.spec.signature, metadata, blockId, { priority });
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
      } catch {
        return null;
      }
    }

    // v18.3 consults only the complete fixed 111-table corpus. An uncovered
    // position returns null and can use ordinary search.
    this.probeCache.set(cacheKey, null);
    return null;
  }


  async chooseMoves(position, rootProbe, limit = 3) {
    const legal = generateLegalMoves(position, false);
    const childPositions = [];
    for (const move of legal) {
      const state = makeMove(position, move);
      const child = position.clone();
      undoMove(position, move, state);
      childPositions.push({ move, child });
    }

    // For a covered ≤5-piece position, every child is read directly from GTB.
    // Do not manufacture a bound with secondary search: the tablebase WDL is
    // authoritative and its DTM is used only for ordering/display.
    const probes = await Promise.all(childPositions.map(item => this.probe(item.child, { priority: 0 }).catch(() => null)));
    const candidates = [];
    for (let index = 0; index < childPositions.length; index += 1) {
      const child = probes[index];
      if (!child) continue;
      candidates.push({
        move: childPositions[index].move,
        childPosition: childPositions[index].child,
        child,
        wdl: -child.wdl,
        dtmPly: child.wdl === 0 ? 0 : Math.max(1, Number(child.dtmPly || 0) + 1),
        dtmUpperBound: Boolean(child.dtmUpperBound),
        source: child.source
      });
    }
    // A root WDL is only useful when at least one *verified child* preserves
    // it. Do not fall back to an arbitrary readable child (or an unverified
    // stored bestMove) after a transient block failure: that can advertise a
    // winning tablebase result while recommending a losing move.
    const matching = candidates.filter(item => item.wdl === rootProbe.wdl);
    if (!matching.length) return [];
    matching.sort((left, right) => {
      if (left.wdl !== right.wdl) return right.wdl - left.wdl;
      const leftDtm = Number(left.dtmPly || 0);
      const rightDtm = Number(right.dtmPly || 0);
      if (rootProbe.wdl > 0) return leftDtm - rightDtm;
      if (rootProbe.wdl < 0) return rightDtm - leftDtm;
      return moveToUci(left.move).localeCompare(moveToUci(right.move));
    });
    return matching.slice(0, Math.max(1, limit));
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


  tablebaseTailEndsInMate(position, moves) {
    const cursor = position.clone();
    for (const move of moves || []) {
      const legal = generateLegalMoves(cursor, false);
      if (!legal.includes(move)) return false;
      makeMove(cursor, move);
    }
    return generateLegalMoves(cursor, false).length === 0 && isInCheck(cursor);
  }

  async extendLineWithExactTablebaseTail(position, line, { maxProbePly = 36, maxTailPly = 96 } = {}) {
    // Replace only the portion of an already legal principal variation after it
    // first enters a covered GTB. The root score and mate status are intentionally
    // untouched: a single PV hitting GTB is not a root proof.
    if (!position?.clone || !line?.pv?.length || line.tablebase || line.mateVerified) return null;
    const root = position.clone();
    const cursor = position.clone();
    const rootSide = root.turn;
    const sourcePv = Array.isArray(line.pv) ? line.pv.slice() : [];
    const limit = Math.min(sourcePv.length, Math.max(0, Number(maxProbePly || 0)));

    for (let ply = 0; ply <= limit; ply += 1) {
      if (pieceCount(cursor) <= MAX_EXACT_TABLEBASE_PIECES) {
        const probe = await this.probe(cursor, { priority: 0 }).catch(() => null);
        if (!probe) return null;
        const dtm = Number(probe.dtmPly || 0);
        // A draw has no finite DTM tail, but an exact WDL=0 node is still a
        // valid bridge frontier: the v19.7 dual-controller prover may later
        // establish 0.00 only after both colours' strategies cover every reply.
        // Merely seeing this one PV entry is informational and changes no root
        // score or solved flag by itself. Handle it before asking GTB for a
        // continuation, so a terminal-draw leaf is also represented correctly.
        if (probe.wdl === 0) {
          return {
            ...line,
            tablebaseTail: {
              entersAtPly: ply,
              wdl: 0,
              tablebaseWdl: 0,
              dtmPly: 0,
              exactDtm: !probe.dtmUpperBound,
              exactWdl: true,
              bridgeable: true,
              draw: true,
              source: probe.source,
              signature: probe.signature,
              terminal: false
            },
            tablebaseTailComplete: false
          };
        }
        // A non-draw tail is admitted only if GTB itself reaches the terminal
        // checkmate at its exact DTM. A draw may be described but is never used
        // to fabricate a finite terminal continuation.
        const choices = await this.chooseMoves(cursor, probe, 1);
        const firstMove = choices[0]?.move || 0;
        if (!firstMove) return null;
        const tailMoves = await this.buildPv(cursor, firstMove, Math.max(1, Number(maxTailPly || 96)));
        const terminal = dtm > 0
          && tailMoves.length === dtm
          && this.tablebaseTailEndsInMate(cursor, tailMoves);
        if (!terminal) return null;
        const winner = probe.wdl > 0 ? cursor.turn : -cursor.turn;
        const rootWdl = winner === rootSide ? 1 : -1;
        return {
          ...line,
          pv: [...sourcePv.slice(0, ply), ...tailMoves.map(moveToUci)],
          tablebaseTail: {
            entersAtPly: ply,
            wdl: rootWdl,
            tablebaseWdl: probe.wdl,
            dtmPly: dtm,
            exactDtm: !probe.dtmUpperBound,
            exactWdl: true,
            bridgeable: true,
            draw: false,
            source: probe.source,
            signature: probe.signature,
            terminal: true
          },
          tablebaseTailComplete: true
        };
      }
      if (ply >= sourcePv.length) break;
      const move = uciToMove(cursor, sourcePv[ply]);
      if (!move) return null;
      makeMove(cursor, move);
    }
    return null;
  }

  async extendResultWithExactTablebaseTails(position, result, { maxLines = 5, maxProbePly = 36, maxTailPly = 96 } = {}) {
    if (!result?.lines?.length || result.tablebase || result.fortressProof) return result;
    const lines = result.lines.map(line => ({ ...line, pv: Array.isArray(line.pv) ? line.pv.slice() : [] }));
    let changed = false;
    const count = Math.min(lines.length, Math.max(1, Number(maxLines || 1)));
    for (let index = 0; index < count; index += 1) {
      const extended = await this.extendLineWithExactTablebaseTail(position, lines[index], { maxProbePly, maxTailPly });
      if (!extended) continue;
      lines[index] = extended;
      changed = true;
    }
    return changed
      ? { ...result, lines, tablebaseTailHydrated: true }
      : result;
  }



  // v19.7: exact-tablebase bridge proof.
  //
  // This is deliberately an AND/OR proof rather than a PV annotation:
  // - the winning/drawing controller chooses one legal continuation;
  // - the resisting side must have *every* legal continuation covered;
  // - leaves are only immediate terminal positions or exact WDL+DTM GTB nodes.
  //
  // Restricting controller moves can make the prover miss a valid proof, but
  // cannot fabricate one.  Resisting moves are never restricted.
  bridgePositionKey(position) {
    return `${position.hashA >>> 0}:${position.hashB >>> 0}:t${position.turn}:p${pieceCount(position)}`;
  }

  bridgeTerminal(position) {
    const legal = generateLegalMoves(position, false);
    if (legal.length) return null;
    return isInCheck(position) ? -position.turn : 0;
  }

  bridgeWinnerFromProbe(position, probe) {
    if (!probe || probe.wdl === 0) return 0;
    return probe.wdl > 0 ? position.turn : -position.turn;
  }

  bridgeProofMoveOrder(position, controller, preferredMoves = [], { includeExactHints = true, rootCandidateOrdering = false } = {}) {
    const preferred = new Map();
    for (let index = 0; index < preferredMoves.length; index += 1) {
      const text = typeof preferredMoves[index] === 'string' ? preferredMoves[index] : moveToUci(preferredMoves[index]);
      if (text && !preferred.has(text)) preferred.set(text, preferredMoves.length - index);
    }
    const moverIsController = position.turn === controller;
    const entries = [];
    for (const move of generateLegalMoves(position, false)) {
      const uci = moveToUci(move);
      const capture = isCapture(position, move);
      const promotion = isPromotion(move);
      const check = givesCheck(position, move);
      const state = makeMove(position, move);
      const terminal = this.bridgeTerminal(position);
      const hit = includeExactHints && pieceCount(position) <= MAX_EXACT_TABLEBASE_PIECES ? this.probeExactSync(position) : null;
      const childTurn = position.turn;
      let value = evaluate(position) * (controller === WHITE ? 1 : -1);
      undoMove(position, move, state);

      // The expected bridge move is an ordering hint only.  It is deliberately
      // high enough to stay inside the controller's bounded candidate set, but
      // never suppresses any defender reply.
      if (preferred.has(uci)) value += 2_000_000_000 + (preferred.get(uci) || 0);
      if (moverIsController && rootCandidateOrdering) value += bridgeControllerKingPressureBonus(position, move, controller);
      if (terminal === controller) value += 1_000_000_000;
      if (hit) {
        const winner = hit.wdl > 0 ? childTurn : hit.wdl < 0 ? -childTurn : 0;
        if (winner === controller) value += 100_000_000 - Math.min(99_999, Number(hit.dtmPly || 0));
        else if (winner && winner !== controller) value -= 100_000_000 - Math.min(99_999, Number(hit.dtmPly || 0));
      }
      const moverSign = moverIsController ? 1 : -1;
      if (capture) value += moverSign * 1_000_000;
      if (promotion) value += moverSign * 500_000;
      if (check) value += moverSign * 10_000;
      entries.push({ move, uci, priority: value });
    }
    entries.sort((left, right) => {
      const primary = moverIsController
        ? right.priority - left.priority
        : left.priority - right.priority;
      return primary || left.uci.localeCompare(right.uci);
    });
    return entries;
  }

  bridgeExactLeafSync(position, controller, outcome, remaining) {
    const terminal = this.bridgeTerminal(position);
    if (terminal !== null) {
      if (outcome === 'draw') {
        return terminal === 0 ? { kind: 'terminal-draw', distance: 0, terminal: true } : null;
      }
      return terminal === controller ? { kind: 'terminal-mate', distance: 0, terminal: true } : null;
    }
    if (pieceCount(position) > MAX_EXACT_TABLEBASE_PIECES) return null;
    // Bridge proofs are intentionally synchronous after warmExactBridgeTables().
    // A missing resident block is a failed leaf, never an asynchronous fallback.
    const probe = this.probeExactSync(position);
    if (!probe || probe.dtmUpperBound) return null;
    if (outcome === 'draw') {
      return probe.wdl === 0
        ? { kind: 'tablebase-draw', distance: 0, probe, terminal: true }
        : null;
    }
    const winner = this.bridgeWinnerFromProbe(position, probe);
    const distance = Math.max(0, Number(probe.dtmPly || 0));
    if (winner !== controller || !distance || distance > remaining) return null;
    return { kind: 'tablebase-mate', distance, probe, terminal: true };
  }

  async bridgeExactLeaf(position, controller, outcome, remaining, { priority = 0 } = {}) {
    // Kept as an async compatibility wrapper for callers outside the proof core.
    return this.bridgeExactLeafSync(position, controller, outcome, remaining);
  }

  async buildBridgePrincipalVariation(position, proof, { priority = 0, maxTailPly = 128 } = {}) {
    const cursor = position.clone();
    const pv = [];
    let node = proof;
    const seen = new Set();
    while (node) {
      const identity = this.bridgePositionKey(cursor);
      if (seen.has(identity)) return null;
      seen.add(identity);
      if (node.kind === 'choice') {
        if (!generateLegalMoves(cursor, false).includes(node.move)) return null;
        pv.push(moveToUci(node.move));
        makeMove(cursor, node.move);
        node = node.child;
        continue;
      }
      if (node.kind === 'all') {
        if (!node.children?.length) return null;
        const worst = node.children.slice().sort((left, right) => (
          (Number(right.child?.distance || 0) + 1) - (Number(left.child?.distance || 0) + 1)
          || moveToUci(left.move).localeCompare(moveToUci(right.move))
        ))[0];
        if (!worst || !generateLegalMoves(cursor, false).includes(worst.move)) return null;
        pv.push(moveToUci(worst.move));
        makeMove(cursor, worst.move);
        node = worst.child;
        continue;
      }
      if (node.kind === 'tablebase-mate') {
        const probe = this.probeExactSync(cursor) || await this.probe(cursor, { priority });
        if (!probe || probe.wdl === 0 || probe.dtmUpperBound || Number(probe.dtmPly || 0) !== Number(node.distance || 0)) return null;
        const first = (await this.chooseMoves(cursor, probe, 1))[0]?.move || 0;
        if (!first) return null;
        const tail = await this.buildPv(cursor, first, Math.min(maxTailPly, Number(node.distance || 0)));
        if (tail.length !== Number(node.distance || 0) || !this.tablebaseTailEndsInMate(cursor, tail)) return null;
        pv.push(...tail.map(moveToUci));
        return pv;
      }
      // A draw proof intentionally has no finite mating tail.  The bridge
      // route ends at an exact GTB draw node.
      if (node.kind === 'tablebase-draw' || node.kind === 'terminal-draw' || node.kind === 'terminal-mate') return pv;
      return null;
    }
    return null;
  }

  async proveExactBridgeOutcome(position, {
    controller = position?.turn,
    outcome = 'win',
    preferredMoves = [],
    maxPlies = 48,
    maxNodes = 48_000,
    timeMs = 700,
    controllerMoveLimit = 4,
    priority = 1,
    maxTailPly = 128
  } = {}) {
    const root = position?.clone?.();
    if (!root || ![WHITE, BLACK].includes(controller)) return null;
    if (!['win', 'draw'].includes(outcome)) return null;
    if (pieceCount(root) <= MAX_EXACT_TABLEBASE_PIECES || pieceCount(root) > MAX_EXACT_TABLEBASE_PIECES + 2) return null;

    const startedAt = performance.now();
    const deadline = startedAt + Math.max(30, Number(timeMs || 0));
    const depthLimit = Math.max(1, Math.min(128, Math.floor(Number(maxPlies || 0))));
    const nodeLimit = Math.max(100, Math.floor(Number(maxNodes || 0)));
    const moverLimit = Math.max(1, Math.min(12, Math.floor(Number(controllerMoveLimit || 1))));
    const rootPreferred = Array.isArray(preferredMoves) ? String(preferredMoves[0] || '') : '';
    const BRIDGE_ABORT = Symbol('bridge-abort');
    let nodes = 0;
    let leaves = 0;

    const checkBudget = () => {
      nodes += 1;
      // Check wall-clock time periodically so the very cheap exact probes do
      // not turn budget accounting itself into the bottleneck.
      if (nodes > nodeLimit || ((nodes & 1023) === 0 && performance.now() >= deadline)) {
        throw BRIDGE_ABORT;
      }
    };

    // This deliberately mirrors the conservative proof shape used by the
    // validated six-piece certificate: controller nodes choose from a bounded
    // candidate set; resisting nodes enumerate every legal reply.  The
    // candidate bound can lose a proof, but never creates one.
    const solveSubtree = (node, remaining, path, memo) => {
      checkBudget();
      const identity = this.bridgePositionKey(node);
      const memoKey = `${identity}:c${controller}:o${outcome}:r${remaining}`;
      if (memo.has(memoKey)) return memo.get(memoKey);
      if (path.has(identity)) return null;

      const leaf = this.bridgeExactLeafSync(node, controller, outcome, remaining);
      if (leaf) {
        leaves += 1;
        memo.set(memoKey, leaf);
        return leaf;
      }
      // Exact-tablebase territory is a proof boundary.  Continuing to search
      // a losing/drawing/missing <=5-piece node could never strengthen this
      // controller's certificate and would turn one failed leaf into a large
      // irrelevant subtree.  Treat it as a failed branch exactly as the
      // certificate semantics require.
      if (pieceCount(node) <= MAX_EXACT_TABLEBASE_PIECES) {
        memo.set(memoKey, null);
        return null;
      }
      if (remaining <= 0) {
        memo.set(memoKey, null);
        return null;
      }

      const all = this.bridgeProofMoveOrder(node, controller, [], { includeExactHints: false });
      const moves = node.turn === controller ? all.slice(0, moverLimit) : all;
      if (!moves.length) {
        memo.set(memoKey, null);
        return null;
      }

      path.add(identity);
      let answer = null;
      try {
        if (node.turn === controller) {
          for (const item of moves) {
            const state = makeMove(node, item.move);
            const child = solveSubtree(node, remaining - 1, path, memo);
            undoMove(node, item.move, state);
            if (!child) continue;
            const candidate = {
              kind: 'choice',
              move: item.move,
              child,
              distance: 1 + Number(child.distance || 0)
            };
            if (!answer || candidate.distance < answer.distance) answer = candidate;
          }
        } else {
          const children = [];
          let worst = 0;
          let complete = true;
          for (const item of moves) {
            const state = makeMove(node, item.move);
            const child = solveSubtree(node, remaining - 1, path, memo);
            undoMove(node, item.move, state);
            if (!child) {
              complete = false;
              break;
            }
            children.push({ move: item.move, child });
            worst = Math.max(worst, 1 + Number(child.distance || 0));
          }
          if (complete && children.length === moves.length) {
            answer = { kind: 'all', children, distance: worst };
          }
        }
      } finally {
        path.delete(identity);
      }
      memo.set(memoKey, answer);
      return answer;
    };

    const runFromMove = move => {
      const branch = root.clone();
      const state = makeMove(branch, move);
      const rootPath = new Set([this.bridgePositionKey(root)]);
      const proof = solveSubtree(branch, depthLimit - 1, rootPath, new Map());
      undoMove(branch, move, state);
      return proof
        ? { kind: 'choice', move, child: proof, distance: 1 + Number(proof.distance || 0) }
        : null;
    };

    let proof = null;
    try {
      if (root.turn === controller) {
        // The current completed analysis PV is a safe policy *hint*: use its
        // root move first, then fall back to normal bounded candidates.  The
        // hint never removes any defender response from the proof.
        const ordered = this.bridgeProofMoveOrder(root, controller, rootPreferred ? [rootPreferred] : [], { includeExactHints: false, rootCandidateOrdering: true });
        const candidates = ordered.slice(0, moverLimit);
        for (const item of candidates) {
          proof = runFromMove(item.move);
          if (proof) break;
        }
      } else {
        proof = solveSubtree(root.clone(), depthLimit, new Set(), new Map());
      }
    } catch (error) {
      if (error !== BRIDGE_ABORT) throw error;
      if (!proof) {
        this.lastBridgeDiagnostics = {
          reason: 'budget', nodes, leaves, elapsed: Math.round(performance.now() - startedAt)
        };
        return null;
      }
    }

    if (!proof) {
      this.lastBridgeDiagnostics = {
        reason: 'no-proof', nodes, leaves, elapsed: Math.round(performance.now() - startedAt)
      };
      return null;
    }
    const pv = await this.buildBridgePrincipalVariation(root, proof, { priority, maxTailPly });
    if (!pv?.length && outcome === 'win') {
      this.lastBridgeDiagnostics = {
        reason: 'pv-build', nodes, leaves, distance: proof.distance,
        elapsed: Math.round(performance.now() - startedAt)
      };
      return null;
    }
    return {
      tablebaseBridgeProof: outcome === 'win',
      tablebaseBridgeDraw: outcome === 'draw',
      controller,
      wdl: outcome === 'draw' ? 0 : (controller === root.turn ? 1 : -1),
      dtmPly: outcome === 'win' ? Number(proof.distance || 0) : 0,
      exactDtm: false,
      upperBound: outcome === 'win',
      pv,
      proof,
      proofNodes: nodes,
      proofLeaves: leaves,
      elapsed: Math.round(performance.now() - startedAt),
      rootKey: this.bridgePositionKey(root)
    };
  }

  // v20.1: verify a complete in-memory bridge certificate before it is cached.
  // The cache stores the full AND/OR tree, not just the displayed worst PV:
  // controller nodes contain one legal policy move and resisting nodes contain
  // every legal reply.  Leaves must still match a resident exact tablebase or
  // an immediate terminal result.  This is intentionally synchronous because
  // callers warm all required blocks before constructing/caching a proof.
  verifyExactBridgeProof(position, record, { controller = record?.controller, outcome = record?.tablebaseBridgeDraw ? 'draw' : 'win' } = {}) {
    const root = position?.clone?.();
    if (!root || ![WHITE, BLACK].includes(controller) || !['win', 'draw'].includes(outcome)) return false;

    const verifyNode = (node, cursor, path) => {
      if (!node || !cursor) return false;
      const identity = this.bridgePositionKey(cursor);
      if (path.has(identity)) return false;

      const verifyLeaf = () => {
        if (node.kind === 'terminal-mate') {
          return outcome === 'win'
            && this.bridgeTerminal(cursor) === controller
            && Number(node.distance || 0) === 0;
        }
        if (node.kind === 'terminal-draw') {
          return outcome === 'draw'
            && this.bridgeTerminal(cursor) === 0
            && Number(node.distance || 0) === 0;
        }
        const probe = this.probeExactSync(cursor);
        if (!probe || probe.dtmUpperBound) return false;
        if (node.kind === 'tablebase-draw') {
          return outcome === 'draw' && probe.wdl === 0 && Number(node.distance || 0) === 0;
        }
        if (node.kind === 'tablebase-mate') {
          const winner = this.bridgeWinnerFromProbe(cursor, probe);
          return outcome === 'win'
            && winner === controller
            && Number(probe.dtmPly || 0) > 0
            && Number(node.distance || 0) === Number(probe.dtmPly || 0);
        }
        return false;
      };

      if (node.kind === 'tablebase-mate' || node.kind === 'tablebase-draw'
        || node.kind === 'terminal-mate' || node.kind === 'terminal-draw') {
        return verifyLeaf();
      }

      const legal = generateLegalMoves(cursor, false);
      if (!legal.length) return false;
      path.add(identity);
      try {
        if (node.kind === 'choice') {
          if (cursor.turn !== controller || !legal.includes(node.move) || !node.child) return false;
          const state = makeMove(cursor, node.move);
          const valid = verifyNode(node.child, cursor, path);
          undoMove(cursor, node.move, state);
          return valid && Number(node.distance || 0) === 1 + Number(node.child.distance || 0);
        }
        if (node.kind === 'all') {
          if (cursor.turn === controller || !Array.isArray(node.children) || node.children.length !== legal.length) return false;
          const unique = new Set();
          let worst = 0;
          for (const entry of node.children) {
            if (!entry || !legal.includes(entry.move) || !entry.child) return false;
            const uci = moveToUci(entry.move);
            if (unique.has(uci)) return false;
            unique.add(uci);
            const state = makeMove(cursor, entry.move);
            const valid = verifyNode(entry.child, cursor, path);
            undoMove(cursor, entry.move, state);
            if (!valid) return false;
            worst = Math.max(worst, 1 + Number(entry.child.distance || 0));
          }
          return unique.size === legal.length && Number(node.distance || 0) === worst;
        }
        return false;
      } finally {
        path.delete(identity);
      }
    };

    if (outcome === 'draw') {
      const strategies = record?.drawStrategies;
      const white = strategies?.white;
      const black = strategies?.black;
      return Boolean(white?.proof && black?.proof)
        && verifyNode(white.proof, root.clone(), new Set())
        && verifyNode(black.proof, root.clone(), new Set());
    }
    return Boolean(record?.proof) && verifyNode(record.proof, root, new Set());
  }

  async proveExactBridgeDraw(position, options = {}) {
    const root = position?.clone?.();
    if (!root) return null;
    // A single side being able to reach a draw does not establish 0.00: that
    // side may still have a forced win.  Exact bridge draw output requires a
    // draw strategy for *both* colors, each against all opposing replies.
    const white = await this.proveExactBridgeOutcome(root, { ...options, controller: WHITE, outcome: 'draw' });
    if (!white) return null;
    const black = await this.proveExactBridgeOutcome(root, { ...options, controller: BLACK, outcome: 'draw' });
    if (!black) return null;
    const preferred = root.turn === WHITE ? white : black;
    return {
      ...preferred,
      tablebaseBridgeProof: false,
      tablebaseBridgeDraw: true,
      wdl: 0,
      upperBound: false,
      proofNodes: Number(white.proofNodes || 0) + Number(black.proofNodes || 0),
      proofLeaves: Number(white.proofLeaves || 0) + Number(black.proofLeaves || 0),
      drawStrategies: { white, black }
    };
  }

  async dtmBoundForLine(position, line, { maxProbePly = 24 } = {}) {
    if (!line?.pv?.length) return null;
    const root = position.clone();
    const cursor = position.clone();
    const rootSide = root.turn;
    const pv = Array.isArray(line.pv) ? line.pv.slice(0, maxProbePly) : [];

    for (let ply = 0; ply <= pv.length && ply <= maxProbePly; ply += 1) {
      if (pieceCount(cursor) <= MAX_EXACT_TABLEBASE_PIECES) {
        const probe = await this.probe(cursor).catch(() => null);
        const dtm = Number(probe?.dtmPly || 0);
        if (probe && probe.wdl !== 0 && Number.isFinite(dtm) && dtm > 0) {
          const winningSide = probe.wdl > 0 ? cursor.turn : -cursor.turn;
          const rootWdl = winningSide === rootSide ? 1 : -1;
          return {
            // This is a conditional observation along the current PV.  It is
            // not a proof about the root because the opponent may choose an
            // earlier branch outside this variation.
            entersAtPly: ply,
            wdl: rootWdl,
            dtmPly: Math.max(1, ply + dtm),
            tablebaseWdl: probe.wdl,
            tablebaseSource: probe.source,
            tablebaseSignature: probe.signature,
            dtmUpperBound: ply > 0 || Boolean(probe.dtmUpperBound),
            exactDtm: !probe.dtmUpperBound,
            conditional: true
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
    // v19.7: a future tablebase hit may explain a candidate PV, but it must
    // never overwrite the root score / mate state. Keep it in a separate hint
    // field so ordering, caching and completion semantics remain unchanged.
    if (!result?.lines?.length || result.tablebase || result.fortressProof) return result;
    const lines = result.lines.map(line => ({ ...line, pv: Array.isArray(line.pv) ? line.pv.slice() : [] }));
    let changed = false;
    for (const line of lines.slice(0, Math.max(1, maxLines))) {
      if (line.mateVerified || line.fortressProof) continue;
      const bound = await this.dtmBoundForLine(position, line, { maxProbePly });
      if (!bound) continue;
      line.tablebaseHint = {
        entersAtPly: bound.entersAtPly,
        wdl: bound.wdl,
        dtmPly: bound.dtmPly,
        tablebaseWdl: bound.tablebaseWdl,
        source: bound.tablebaseSource,
        signature: bound.tablebaseSignature,
        conditional: true,
        exactDtm: Boolean(bound.exactDtm)
      };
      changed = true;
    }
    if (!changed) return result;
    return {
      ...result,
      lines,
      tablebaseDtmHint: true
    };
  }


  async analyze(position, { multipv = 3, maxPvPly = 96 } = {}) {
    const analyzeKey = `${tablebaseKey(position)}:m${Math.max(1, multipv | 0)}:pv${Math.max(1, maxPvPly | 0)}`;
    const cachedAnalysis = this.analysisCache.get(analyzeKey);
    if (cachedAnalysis !== undefined) return cloneAnalyzeResult(cachedAnalysis);
    const root = position.clone();
    // A covered endgame is terminal analysis: read the complete GTB record
    // immediately and never spend conventional engine time on it.
    const probe = await this.probe(root, { priority: 0 });
    if (!probe) return null;
    const choices = await this.chooseMoves(root, probe, multipv);
    const rootSide = root.turn;
    const lines = [];
    for (const choice of choices) {
      const pvMoves = await this.buildPv(root, choice.move, maxPvPly);
      const choiceDtm = Number(choice.dtmPly);
      const probeDtm = Number(probe.dtmPly);
      const rootDtm = Number.isFinite(choiceDtm) && choiceDtm > 0
        ? choiceDtm
        : Number.isFinite(probeDtm) && probeDtm > 0
          ? probeDtm
          : Math.max(1, pvMoves.length);
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
        scoreKind: 'exact-tablebase',
        scoreNumeric: true,
        pv: pv.length ? pv : [moveToUci(choice.move)],
        mateVerified: false,
        endgameProof: false,
        tablebase: true,
        tablebaseScope: 'root-exact',
        tablebaseWdl: probe.wdl,
        dtm: rootDtm,
        dtmUpperBound: Boolean(choice.dtmUpperBound),
        source: choice.source || probe.source,
        tablebaseExactDtm: !choice.dtmUpperBound,
        pvComplete: true
      };
      // GTB itself proves WDL; mate-PV replay is presentation-only and must not
      // downgrade or delay a direct tablebase result.
      lines.push(candidate);
    }
    // A complete-table hit must have a legal GTB continuation. Do not invent
    // an arbitrary move when a corrupt or unavailable block cannot provide one.
    if (!lines.length) return null;
    const result = {
      engine: ENGINE_VERSION,
      engineLabel: `${ENGINE_VERSION} + GTB`,
      depth: 0,
      selDepth: 0,
      nodes: 0,
      nps: 0,
      elapsed: 0,
      scoreDepth: 0,
      pvDepth: 0,
      pvTarget: 0,
      pvComplete: true,
      lines,
      terminal: true,
      completed: true,
      tablebase: true,
      tablebaseScope: 'root-exact',
      tablebaseSource: probe.source,
      tablebaseSignature: probe.signature,
      tablebaseWdl: probe.wdl,
      tablebaseDtmBound: lines.some(line => line.dtmUpperBound),
      rootTurn: rootSide,
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
  rankBoard,
  transformPackedMove,
  maybeLightweightProbe,
  TRIVIAL_DRAW_SIGNATURES,
  MATE_IN_ONE_ONLY_SIGNATURES,
  gunzip,
  DEFAULT_BASE
});
