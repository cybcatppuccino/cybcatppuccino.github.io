import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  EnginePosition,
  GardnerSearcher,
  generateLegalMoves,
  isInCheck,
  uciToMove
} from '../js/engine/engine.js';
import { GardnerTablebase } from '../js/engine/tablebase.js';

const engineSource = readFileSync(new URL('../js/engine/engine.js', import.meta.url), 'utf8');
const workerSource = readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');
const tablebaseSource = readFileSync(new URL('../js/engine/tablebase.js', import.meta.url), 'utf8');
const panelSource = readFileSync(new URL('../js/ui/analysis-panel.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const indexSource = readFileSync(new URL('../index.html', import.meta.url), 'utf8');

// Visible/versioned live-analysis plumbing.
assert.match(indexSource, /Gardner MiniChess Lab v19\.3/);
assert.match(indexSource, /Orion JS 19\.3/);
assert.match(indexSource, /app\.js\?v=19\.3/);
assert.match(workerSource, /postLiveProgress/);
assert.match(workerSource, /nodeTarget/);
assert.match(panelSource, /formatNodeProgress/);
assert.doesNotMatch(panelSource, /rootWinRate/);
assert.doesNotMatch(panelSource, /analysis-winrate/);
assert.match(appSource, /candidate\.liveProgress/);

// The low-progress policy remains a hard-draw policy, but v18.4 explicitly
// audits relevant tactical and zugzwang resources with a 25% time multiplier.
assert.match(engineSource, /PROGRESS_MAX_STALENESS_MS\s*=\s*180/);
assert.match(engineSource, /LOW_PROGRESS_AUDIT_MULTIPLIER\s*=\s*1\.25/);
assert.match(engineSource, /sacrificialAttack/);
assert.match(engineSource, /quietKingEntry/);
assert.match(engineSource, /zugzwangProbe/);
assert.match(engineSource, /this\.rootDeadDraw && !fortressProof && !verifiedRootMate/);

// Equivalent repetition-relevant multisets now use a commutative path salt.
assert.match(engineSource, /this\.ttPathSaltA\[index\]\s*=\s*\(previousA \+ contributionA\)/);
assert.match(engineSource, /this\.ttPathSaltB\[index\]\s*=\s*\(previousB \+ contributionB\)/);
{
  const first = EnginePosition.fromFEN('4k/5/5/5/K3R w - - 0 1');
  const second = EnginePosition.fromFEN('4k/5/5/5/K2R1 w - - 0 1');
  const searcher = new GardnerSearcher({ hashEntries: 16_384 });
  searcher.ttHistorySaltA = 0;
  searcher.ttHistorySaltB = 0;
  searcher.recordSearchPath(0, first);
  searcher.recordSearchPath(1, second);
  const forward = [searcher.ttPathSaltA[1], searcher.ttPathSaltB[1]];
  searcher.ttPathSaltA.fill(0);
  searcher.ttPathSaltB.fill(0);
  searcher.recordSearchPath(0, second);
  searcher.recordSearchPath(1, first);
  const reverse = [searcher.ttPathSaltA[1], searcher.ttPathSaltB[1]];
  assert.deepEqual(reverse, forward, 'Equivalent path-state multisets should share the v18.4 TT history salt');
}

// A checked <=5-piece node must still use an already loaded synchronous WDL
// result instead of falling back to a heuristic qsearch evaluation.
{
  const checked = EnginePosition.fromFEN('4k/5/5/5/K3R b - - 0 1');
  assert.equal(checked.pieceCount, 3);
  assert.equal(isInCheck(checked), true);
  let probes = 0;
  const searcher = new GardnerSearcher({
    hashEntries: 16_384,
    tablebaseProbe: () => {
      probes += 1;
      return { wdl: 0 };
    }
  });
  const score = searcher.qsearch(checked.clone(), -32_000, 32_000, 0, 0);
  assert.equal(score, 0);
  assert.equal(probes, 1, 'Checked WDL node should be probed exactly once before move generation');
}

// Direct GTB selection must never pair a winning root WDL with an arbitrary
// readable child if none of the available child probes preserves the root WDL.
{
  const root = EnginePosition.fromFEN('4k/5/5/5/K3R w - - 0 1');
  const legal = generateLegalMoves(root);
  assert.ok(legal.length > 0);
  const tb = new GardnerTablebase();
  tb.probe = async () => ({ wdl: 0, dtmPly: 0 });
  const verifiedFallback = uciToMove(root, 'a1a2');
  assert.ok(verifiedFallback);
  const moves = await tb.chooseMoves(root, { wdl: 1, bestMove: verifiedFallback }, 3);
  assert.deepEqual(moves, [], 'No WDL-preserving child must return an empty tablebase move set, never a random fallback');
}

assert.match(tablebaseSource, /const matching = candidates\.filter\(item => item\.wdl === rootProbe\.wdl\);/);
assert.match(tablebaseSource, /if \(!matching\.length\) return \[\];/);
assert.doesNotMatch(tablebaseSource, /const pool = matching\.length \? matching : candidates;/);

console.log('v19.3 compatibility checks for v18.4 tablebase and TT changes passed.');
