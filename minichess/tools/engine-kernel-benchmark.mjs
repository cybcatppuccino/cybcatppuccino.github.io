#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { analyzeOnce } from '../js/engine/engine.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const require = createRequire(import.meta.url);

const STOCKFISH_DIR = path.join(root, 'vendor/fairy-stockfish');
const STOCKFISH_JS = path.join(STOCKFISH_DIR, 'stockfish.js');
const MATE = 30000;

const SUITE = [
  {
    id: 'initial-balance',
    category: 'opening / baseline speed',
    fen: 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1',
    expected: 'near 0.00, legal opening PV'
  },
  {
    id: 'closed-deadlock-qb1',
    category: 'closed fortress / draw recognition',
    fen: 'rq2k/p1p1p/PpPpP/1B1P1/RQ2K b - - 6 8',
    expected: 'near 0.00 despite white material edge'
  },
  {
    id: 'closed-deadlock-qc1',
    category: 'closed fortress / draw recognition',
    fen: 'rq2k/p1p1p/PpPpP/1B1P1/R1Q1K w - - 5 8',
    expected: 'near 0.00 despite white material edge'
  },
  {
    id: 'quiet-breakthrough',
    category: 'closed tactic / quiet breakthrough',
    fen: 'r2qk/p1p1p/NpPpP/1P1P1/2BQK b - - 0 5',
    expected: 'prefer ...Qb5/rook-heavy breakthrough resources'
  },
  {
    id: 'reported-cycle-line',
    category: 'repetition stability',
    fen: 'r2qk/p1p1p/npPpP/1P1P1/1NBQK w - - 0 5',
    expected: 'avoid hard 0.00 from first twofold cycle'
  },
  {
    id: 'mate-proof',
    category: 'mate search',
    fen: '4k/5/2Q2/2K2/5 w - - 0 1',
    expected: 'forced mate / high winning score'
  },
  {
    id: 'pawn-race-proof',
    category: 'low-material conversion',
    fen: '8/8/8/1p3p2/1k3P2/8/4K3/8 b - - 1 23',
    expected: 'exact low-material mate proof for Orion; Stockfish usually searches normally'
  },
  {
    id: 'book-random-like',
    category: 'middlegame judgement',
    fen: 'r1bqk/pp1pp/2p2/PP1PP/RNBQK b - - 0 2',
    expected: 'sensible legal PV and stable sign'
  }
];

function installNodeFetchForLocalWasm() {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async function patchedFetch(input, init) {
    const raw = String(input?.url || input || '');
    if (raw.startsWith('/') && fs.existsSync(raw)) {
      const buffer = fs.readFileSync(raw);
      return new Response(buffer, {
        headers: { 'Content-Type': raw.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream' }
      });
    }
    return originalFetch(input, init);
  };
}

function scoreFromUci(tokens, rootTurn) {
  const scoreIndex = tokens.indexOf('score');
  if (scoreIndex < 0) return null;
  const kind = tokens[scoreIndex + 1];
  const raw = Number(tokens[scoreIndex + 2] || 0);
  let score = raw;
  if (kind === 'mate') {
    const plies = Math.max(1, Math.abs(raw) * 2 - 1);
    score = raw > 0 ? MATE - plies : -MATE + plies;
  }
  return rootTurn === 'w' ? score : -score;
}

function parseInfoLine(line, rootTurn) {
  const tokens = String(line || '').trim().split(/\s+/);
  if (tokens[0] !== 'info') return null;
  const depthIndex = tokens.indexOf('depth');
  const nodesIndex = tokens.indexOf('nodes');
  const npsIndex = tokens.indexOf('nps');
  const pvIndex = tokens.indexOf('pv');
  const score = scoreFromUci(tokens, rootTurn);
  if (score === null || pvIndex < 0) return null;
  return {
    depth: depthIndex >= 0 ? Number(tokens[depthIndex + 1] || 0) : 0,
    nodes: nodesIndex >= 0 ? Number(tokens[nodesIndex + 1] || 0) : 0,
    nps: npsIndex >= 0 ? Number(tokens[npsIndex + 1] || 0) : 0,
    score,
    pv: tokens.slice(pvIndex + 1)
  };
}

class FairyStockfishNode {
  constructor() {
    this.engine = null;
    this.lines = [];
    this.waits = [];
  }

  async init() {
    installNodeFetchForLocalWasm();
    const Stockfish = require(STOCKFISH_JS);
    this.engine = await Stockfish({ locateFile: file => path.join(STOCKFISH_DIR, file) });
    this.engine.addMessageListener(line => this.handleLine(line));
    await this.commandAndWait('uci', /^uciok$/);
    this.send('setoption name UCI_Variant value gardner');
    this.send('setoption name Threads value 1');
    this.send('setoption name Hash value 32');
    this.send('setoption name Use NNUE value true');
    await this.commandAndWait('isready', /^readyok$/);
  }

  handleLine(line) {
    const text = String(line || '');
    this.lines.push(text);
    for (const wait of [...this.waits]) {
      if (wait.pattern.test(text)) {
        clearTimeout(wait.timer);
        this.waits = this.waits.filter(item => item !== wait);
        wait.resolve(text);
      }
    }
  }

  send(command) {
    this.engine.postMessage(command);
  }

  commandAndWait(command, pattern, timeoutMs = 12000) {
    const promise = new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.waits = this.waits.filter(item => item.resolve !== resolve);
        reject(new Error(`Timed out waiting for ${pattern} after ${command}`));
      }, timeoutMs);
      this.waits.push({ pattern, resolve, timer });
    });
    this.send(command);
    return promise;
  }

  async search(fen, { depth = 8, multipv = 1 } = {}) {
    this.lines = [];
    const rootTurn = /\sb\s/.test(fen) ? 'b' : 'w';
    this.send(`setoption name MultiPV value ${multipv}`);
    this.send('ucinewgame');
    this.send(`position fen ${fen}`);
    const started = performance.now();
    await this.commandAndWait(`go depth ${depth}`, /^bestmove\s+/, 30000);
    const elapsed = Math.max(1, Math.round(performance.now() - started));
    const infos = this.lines.map(line => parseInfoLine(line, rootTurn)).filter(Boolean);
    const best = infos.sort((a, b) => b.depth - a.depth)[0] || { depth: 0, nodes: 0, nps: 0, score: 0, pv: [] };
    const bestmove = (this.lines.find(line => /^bestmove\s+/.test(line)) || '').split(/\s+/)[1] || best.pv[0] || '';
    return { engine: 'Fairy-Stockfish', depth: best.depth, nodes: best.nodes, nps: best.nps, elapsed, score: best.score, move: bestmove, pv: best.pv.slice(0, 8) };
  }

  close() {
    try { this.send('quit'); } catch {}
    try { this.engine?.terminate?.(); } catch {}
  }
}

function orionSearch(fen, depth = 8) {
  const started = performance.now();
  const result = analyzeOnce(fen, {
    timeMs: 100000,
    maxDepth: depth,
    startDepth: depth,
    multipv: 1,
    fortressProbeMs: 120,
    endgameProbeMs: 70,
    mateProbeMs: 500,
    mateMaxPlies: 13
  });
  const elapsed = Math.max(1, Math.round(performance.now() - started));
  const line = result.lines?.[0] || {};
  return { engine: 'Orion JS', depth: result.depth || depth, nodes: result.nodes || 0, nps: Math.round((result.nodes || 0) * 1000 / Math.max(1, elapsed)), elapsed, score: Number(line.score || 0), move: line.move || '', pv: (line.pv || []).slice(0, 8), proof: result.fortressProof ? 'fortress' : line.mateVerified ? 'mate' : result.tablebase ? 'tablebase' : '' };
}

function scoreText(score) {
  if (Math.abs(score) >= 29000) return score > 0 ? '#+' : '#-';
  return `${score >= 0 ? '+' : ''}${(score / 100).toFixed(2)}`;
}

function rowToMarkdown(row) {
  return `| ${row.id} | ${row.category} | ${scoreText(row.orion.score)} / ${row.orion.move || '—'} | ${scoreText(row.fairy.score)} / ${row.fairy.move || '—'} | ${row.orion.elapsed} / ${row.fairy.elapsed} | ${row.comment} |`;
}

function commentFor(item, orion, fairy) {
  if (item.id.includes('deadlock')) {
    return Math.abs(orion.score) <= 10 && Math.abs(fairy.score) > 80
      ? 'Orion recognizes the closed draw; Fairy keeps a static material edge.'
      : Math.abs(orion.score) <= 10 ? 'Orion recognizes the closed draw.' : 'Needs review.';
  }
  if (item.id === 'mate-proof') return orion.proof === 'mate' ? 'Orion proof-search verifies mate; Fairy searches it as a normal UCI engine.' : 'Both search tactically.';
  if (item.id === 'quiet-breakthrough') return 'Checks whether quiet heavy-piece offers appear before repetition lines.';
  return 'Baseline comparison.';
}

async function main() {
  const depth = Number(process.env.BENCH_DEPTH || 8);
  const fairy = new FairyStockfishNode();
  let fairyReady = false;
  try {
    await fairy.init();
    fairyReady = true;
  } catch (error) {
    console.error('Fairy-Stockfish unavailable for benchmark:', error?.message || error);
  }

  const rows = [];
  for (const item of SUITE) {
    const orion = orionSearch(item.fen, depth);
    const fairyResult = fairyReady
      ? await fairy.search(item.fen, { depth, multipv: 1 })
      : { engine: 'Fairy-Stockfish', depth: 0, nodes: 0, nps: 0, elapsed: 0, score: 0, move: '', pv: [] };
    rows.push({ ...item, orion, fairy: fairyResult, comment: commentFor(item, orion, fairyResult) });
  }
  fairy.close();

  const totalOrionNodes = rows.reduce((sum, row) => sum + row.orion.nodes, 0);
  const totalFairyNodes = rows.reduce((sum, row) => sum + row.fairy.nodes, 0);
  const totalOrionElapsed = rows.reduce((sum, row) => sum + row.orion.elapsed, 0);
  const totalFairyElapsed = rows.reduce((sum, row) => sum + row.fairy.elapsed, 0);

  const lines = [];
  lines.push('# Engine kernel benchmark — v14.1');
  lines.push('');
  lines.push(`Depth: ${depth}. Scores are white-centric centipawns. Times are milliseconds from this Node benchmark run.`);
  lines.push('');
  lines.push('| Position | Category | Orion score / move | Fairy score / move | ms Orion / Fairy | Note |');
  lines.push('|---|---|---:|---:|---:|---|');
  for (const row of rows) lines.push(rowToMarkdown(row));
  lines.push('');
  lines.push('## Aggregate speed');
  lines.push('');
  lines.push(`- Orion: ${totalOrionNodes.toLocaleString()} nodes in ${totalOrionElapsed} ms, about ${Math.round(totalOrionNodes * 1000 / Math.max(1, totalOrionElapsed)).toLocaleString()} nps.`);
  lines.push(`- Fairy-Stockfish: ${totalFairyNodes.toLocaleString()} nodes in ${totalFairyElapsed} ms, about ${Math.round(totalFairyNodes * 1000 / Math.max(1, totalFairyElapsed)).toLocaleString()} nps.`);
  lines.push('');
  lines.push('## Practical reading');
  lines.push('');
  lines.push('- Orion JS is stronger for app-specific truth conditions: compact-Gardner legal validation, mate proof flags, tablebase/fortress hooks, and closed-deadlock compression.');
  lines.push('- Fairy-Stockfish is much faster at raw alpha-beta node throughput and is useful as an independent tactical/open-position opinion, but it does not know Orion’s bespoke fortress/deadlock model.');
  lines.push('- In browser deployment, Fairy requires cross-origin isolation headers because this wasm build uses pthreads/SharedArrayBuffer. Use `serve.sh`, `serve.bat`, or equivalent COOP/COEP headers.');
  lines.push('');

  const markdown = lines.join('\n');
  console.log(markdown);
  const out = path.join(root, 'docs/ENGINE-KERNEL-BENCHMARK-v14.1.md');
  fs.writeFileSync(out, markdown);
}

main().catch(error => {
  console.error(error?.stack || error?.message || String(error));
  process.exitCode = 1;
});
