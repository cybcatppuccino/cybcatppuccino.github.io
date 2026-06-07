const fs = require('fs');
const vm = require('vm');
function assert(cond, msg){ if(!cond) throw new Error(msg); }
const stats = JSON.parse(fs.readFileSync('assets/ries-intsumdb-v11_7-stats.json', 'utf8'));
assert(stats.version === '11.7.2-intsumdb-sign-dedupe-20260607', 'intsum stats version mismatch');
assert(stats.signDedupe && stats.signDedupe.rowsRemoved === 242, 'expected 242 conservative sign-equivalence removals');
assert(stats.signDedupe.ruleCounts['unit-beta/rational-log sign convention'] === 72, 'unit beta/rational sign convention count changed');
assert(stats.signDedupe.ruleCounts['trig-Fourier reflection x->pi-x'] === 170, 'trig Fourier reflection count changed');
assert(Array.isArray(stats.signDedupe.removedIds) && stats.signDedupe.removedIds.length === 242, 'removed id list mismatch');
const ctx = { window:{}, console, atob:(s)=>Buffer.from(s,'base64').toString('binary') };
ctx.window = ctx;
vm.createContext(ctx);
for(const lvl of [4,5,6]) vm.runInContext(fs.readFileSync(`assets/ries-intsumdb-v11_7-level${lvl}.js`, 'utf8'), ctx);
const allIds = new Set();
for(const ch of ctx.RIES_INTSUMDB_V117_CHUNKS){
  for(const id of String(ch.idBlob || '').split('\n').filter(Boolean)) allIds.add(id);
}
assert(allIds.size === stats.rows, `asset id count ${allIds.size} does not match stats rows ${stats.rows}`);
for(const id of stats.signDedupe.removedIds) assert(!allIds.has(id), `removed sign-duplicate ${id} still present in assets`);
for(const sample of stats.signDedupe.sampleCases){
  assert(allIds.has(sample.kept), `kept sign representative ${sample.kept} missing`);
  for(const id of sample.removed) assert(!allIds.has(id), `sample removed row ${id} still present`);
}
console.log('PASS RIES v11.7.2 conservative sign-equivalence dedupe test');
