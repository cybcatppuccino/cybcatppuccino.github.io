import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  RESULT_CONTRACT_KIND,
  canDisplayMateBound,
  canDisplayMateIn,
  contractKindForLine,
  isPublishableLine,
  normalizeLineContract,
  normalizeResultContract
} from '../js/engine/result-contract.js';
import { lineHasCompletePv, pvTargetForDepth, withResultQuality } from '../js/engine/result-quality.js';

const bridge = normalizeLineContract({
  score: 29995,
  scoreText: '≤#3',
  tablebaseBridgeProof: true,
  mateVerified: true,
  mateProof: true,
  mateUpperBound: true,
  tablebaseBridgeDtm: 5,
  pv: ['a1a2']
});
assert.equal(bridge.resultContract, RESULT_CONTRACT_KIND.DB_BRIDGE_MATE_BOUND);
assert.equal(bridge.mateVerified, false);
assert.equal(bridge.mateProof, false);
assert.equal(canDisplayMateBound(bridge), true);
assert.equal(canDisplayMateIn(bridge), false);
assert.equal(isPublishableLine(bridge), true);

const fakeMate = normalizeLineContract({
  score: 29997,
  scoreText: '#2',
  mateVerified: true,
  mateProof: false,
  pv: ['a1a2', 'b1b2']
});
assert.equal(fakeMate.resultContract, RESULT_CONTRACT_KIND.MATE_CANDIDATE);
assert.equal(fakeMate.mateVerified, false);
assert.equal(fakeMate.scoreText, '+250.00');
assert.equal(isPublishableLine(fakeMate), true);

const proven = normalizeLineContract({
  score: 29995,
  scoreText: '#3',
  mateVerified: true,
  mateProof: true,
  dtm: 5,
  pv: ['a1a2']
});
assert.equal(contractKindForLine(proven), RESULT_CONTRACT_KIND.FORCED_MATE_EXACT);
assert.equal(canDisplayMateIn(proven), true);

assert.equal(pvTargetForDepth(1), 1);
assert.equal(pvTargetForDepth(8), 8);
assert.equal(pvTargetForDepth(14), 14);
assert.equal(lineHasCompletePv({ score: 12, pv: Array(5).fill('a1a2') }, { depth: 6, lines: [] }), false);
assert.equal(lineHasCompletePv({ score: 12, pv: Array(6).fill('a1a2') }, { depth: 6, lines: [] }), true);

const normalized = withResultQuality({
  engine: 'Test',
  depth: 9,
  lines: [{ score: 29999, scoreText: '#1', mateVerified: true, mateProof: false, pv: ['a1a2'] }]
});
assert.equal(normalized.lines[0].scoreText, '+250.00');
assert.equal(normalized.lines[0].mateVerified, false);
assert.equal(normalized.resultContract, RESULT_CONTRACT_KIND.MATE_CANDIDATE);

const minifish = fs.readFileSync(new URL('../js/engine/minifish.js', import.meta.url), 'utf8');
assert.match(minifish, /Minifish JS 21\.1/);
assert.match(minifish, /pvMateLineOnly/);
assert.doesNotMatch(minifish, /mateProof:\s*mateVerified/);
assert.doesNotMatch(minifish, /lines\[0\]\?\.mateVerified\s*&&/);

const panel = fs.readFileSync(new URL('../js/ui/analysis-panel.js', import.meta.url), 'utf8');
assert.match(panel, /contractKindForLine/);
assert.match(panel, /canDisplayMateIn/);
assert.doesNotMatch(panel, /if \(line\.mateVerified\) return `\$\{line\.scoreText\} · proven`/);

const worker = fs.readFileSync(new URL('../js/engine/worker.js', import.meta.url), 'utf8');
assert.match(worker, /RESULT_CONTRACT_KIND\.DB_BRIDGE_MATE_BOUND/);
assert.match(worker, /mateVerified:\s*false/);

console.log('v21.1 result contract, UI, Minifish and depth-PV integrity tests passed.');
