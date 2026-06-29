import {
  ENGINE_VERSION,
  EngineInternals,
  generateLegalMoves,
  isInCheck,
  moveToUci,
  scoreToDisplay
} from './engine.js';
import { EMBEDDED_EXACT_MANIFEST } from './tablebase-manifest.js';

const {
  PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK,
  makeMove, undoMove, moveFrom, moveTo, movePromotion, encodeMove,
  MATE, sideOf, typeOf, fileOf, rankOf
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
    // Missing tablebase blocks may be requested by an actual search node, but
    // never by a root-to-tablebase bridge or broad frontier traversal.
    this.passiveWdlRequests = new Map();
    this.maxPassiveWdlRequests = 48;
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

  requestWdlFromSearch(position, { priority = 4 } = {}) {
    // Search is synchronous, like Stockfish's in-memory probe path. When a
    // local block is absent, queue only the precise WDL block for this actual
    // node; the current search continues normally and a later iteration can
    // consume the cached bound. This is cache fill, not a separate search.
    if (!position || pieceCount(position) > MAX_EXACT_TABLEBASE_PIECES) return;
    let exact;
    try { exact = exactCanonical(position); } catch { return; }
    const index = rankBoard(exact.board, exact.turn, exact.spec);
    const key = `${exact.spec.signature}:${index}`;
    if (this.passiveWdlRequests.has(key) || this.passiveWdlRequests.size >= this.maxPassiveWdlRequests) return;
    const request = this.probeWdl(position.clone(), { priority })
      .catch(() => null)
      .finally(() => this.passiveWdlRequests.delete(key));
    this.passiveWdlRequests.set(key, request);
  }

  async probe(position, { priority = 0 } = {}) {
    if (pieceCount(position) > MAX_EXACT_TABLEBASE_PIECES) return null;
    const cacheKey = tablebaseKey(position);
    const cached = this.probeCache.get(cacheKey);
    if (cached !== undefined) return cloneProbeResult(cached);
    const exact = exactCanonical(position);
    const lightweight = maybeLightweightProbe(position, exact);
    if (lightweight) {
      const exactLightweight = { ...lightweight, exactDtm: true };
      this.probeCache.set(cacheKey, exactLightweight);
      return cloneProbeResult(exactLightweight);
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
          exactDtm: true,
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

    // Root tablebase handling follows the Stockfish model: inspect every legal
    // child directly and rank only after the complete root move set is known.
    const probes = await Promise.all(childPositions.map(item => this.probe(item.child, { priority: 0 }).catch(() => null)));
    const candidates = [];
    for (let index = 0; index < childPositions.length; index += 1) {
      const child = probes[index];
      if (!this.isExactDtmProbeResult(child)) continue;
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
    // A direct root result is only publishable after every legal child has an
    // exact DTM record. A partial download must fall back to ordinary search.
    if (candidates.length !== childPositions.length) return [];
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


  isExactDtmProbeResult(probe) {
    if (!probe || probe.wdl === 2 || probe.wdl === undefined) return false;
    if (probe.dtmUpperBound === true) return false;
    if (Math.sign(Number(probe.wdl || 0)) === 0) return true;
    const dtm = Number(probe.dtmPly || 0);
    return Number.isFinite(dtm) && dtm >= 0;
  }

  async probeExactDtm(position, { priority = 0 } = {}) {
    if (pieceCount(position) > MAX_EXACT_TABLEBASE_PIECES) return null;
    const probe = await this.probe(position, { priority });
    if (!this.isExactDtmProbeResult(probe)) return null;
    return probe;
  }


  async analyze(position, { multipv = 3, maxPvPly = 96 } = {}) {
    const analyzeKey = `${tablebaseKey(position)}:m${Math.max(1, multipv | 0)}:pv${Math.max(1, maxPvPly | 0)}`;
    const cachedAnalysis = this.analysisCache.get(analyzeKey);
    if (cachedAnalysis !== undefined) return cloneAnalyzeResult(cachedAnalysis);

    const root = position.clone();
    const probe = await this.probe(root, { priority: 0 });
    // A root answer is direct only when its DTM is exact. WDL-only cache blocks
    // remain useful to the synchronous alpha-beta search as bounds, but never
    // become a fake "mate in N" UI result.
    if (!this.isExactDtmProbeResult(probe)) return null;
    const choices = await this.chooseMoves(root, probe, multipv);
    if (!choices.length) return null;

    const rootSide = root.turn;
    const rootWdl = Math.sign(Number(probe.wdl || 0));
    const rootDtm = rootWdl === 0 ? 0 : Math.max(1, Number(probe.dtmPly || choices[0]?.dtmPly || 1));
    const rootScore = rootWdl > 0 ? MATE - rootDtm : rootWdl < 0 ? -MATE + rootDtm : 0;
    const whiteScore = rootSide === WHITE ? rootScore : -rootScore;

    const lines = [];
    for (const choice of choices) {
      const pvMoves = await this.buildPv(root, choice.move, maxPvPly);
      const pv = pvMoves.map(moveToUci);
      lines.push({
        move: pv[0] || moveToUci(choice.move),
        score: whiteScore,
        scoreText: rootWdl === 0 ? '0.00' : scoreToDisplay(whiteScore),
        scoreKind: rootWdl ? 'mate' : 'evaluation',
        scoreNumeric: true,
        pv: pv.length ? pv : [moveToUci(choice.move)],
        mateVerified: rootWdl !== 0,
        tablebase: true,
        tablebaseRoot: true,
        tablebaseWdl: rootWdl,
        dtm: rootDtm,
        source: choice.source || probe.source || 'gardner-tablebase',
        tablebaseExactDtm: true,
        rootScoreExact: true,
        pvComplete: true,
        resultContract: 'tablebase-root',
        resultKindV2: 'tablebase-root'
      });
    }

    const result = {
      engine: ENGINE_VERSION,
      engineLabel: `${ENGINE_VERSION} + GTB`,
      depth: 0,
      selDepth: 0,
      nodes: 0,
      nps: 0,
      elapsed: 0,
      scoreDepth: 0,
      pvDepth: Math.max(1, ...lines.map(line => line.pv.length)),
      pvTarget: 0,
      pvComplete: true,
      lines,
      terminal: true,
      completed: true,
      tablebase: true,
      tablebaseRoot: true,
      tablebaseSource: probe.source || 'gardner-tablebase',
      tablebaseSignature: probe.signature || '',
      tablebaseWdl: rootWdl,
      rootTurn: rootSide,
      solved: true,
      nextDepth: 0,
      searchDepth: 0,
      hashfull: 0,
      multiPvVerified: true,
      resultContract: 'tablebase-root',
      resultKindV2: 'tablebase-root'
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
