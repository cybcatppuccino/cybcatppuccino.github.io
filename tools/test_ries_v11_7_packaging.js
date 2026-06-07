const fs = require('fs');
const path = require('path');

function assert(cond, msg){ if(!cond) throw new Error(msg); }
function exists(p){ assert(fs.existsSync(p), `missing ${p}`); }

const html = fs.readFileSync('ries.html', 'utf8');
const script = fs.readFileSync('ries-script.js', 'utf8');
const stats = JSON.parse(fs.readFileSync('assets/ries-intsumdb-v11_7-stats.json', 'utf8'));

assert(html.includes('RIES <em>v11.9</em>'), 'RIES navbar should advertise v11.9');
assert(html.includes('ries-script.js?v=11.9'), 'ries.html should cache-bust ries-script.js with v11.9');
assert(html.includes('data-module-block="moduleIntsumDb"'), 'Integral/sum module block missing from parameters UI');
assert(html.includes('id="moduleIntsumDb"'), 'Integral/sum module toggle missing');
assert(html.includes('id="intsumDbLimit"') && html.includes('id="intsumDb3BudgetMs"'), 'Integral/sum limit/budget controls missing');
assert(!/<script[^>]+ries-intsumdb-v11_7-level[456]\.js/.test(html), 'Integral/sum chunks must be lazy-loaded, not included by ries.html');

assert(script.includes('window.__RIES_INTSUMDB_TEST__'), 'Intsum test hook missing');
assert(script.includes('moduleIntsumDb'), 'readSettings module toggle for intsum missing');
assert(script.includes('intsumDbOptions'), 'readSettings intsum options missing');
assert(script.includes('integral/sum database: ${view.lhsText} ≈'), 'runtime rows should identify the independent intsum module and transformed target lhs');
assert(script.includes("constantDbSource:'intsumdb-v11.7.2'"), 'intsum rows need stable source marker');

for(const level of [4,5,6]) exists(`assets/ries-intsumdb-v11_7-level${level}.js`);
exists('assets/ries-intsumdb-v11_7-stats.json');
exists('tools/build_intsumdb_v11_7.py');
exists('changelog/RIES_v11.7.2_CHANGELOG.md');

assert(stats.rows === 36443, `unexpected intsum row count ${stats.rows}`);
assert(stats.valueSignPairScan && stats.valueSignPairScan.absValue16MixedSignGroups === 742, 'mixed-sign scan count mismatch');
assert(stats.signDedupe && stats.signDedupe.rowsRemoved === 242, 'sign-equivalence dedupe count mismatch');
assert(stats.signDedupe.retainedMixedSignGroupsAsDistinct === 500, 'retained mixed-sign groups count mismatch');
assert(stats.signDedupe.ruleCounts['unit-beta/rational-log sign convention'] === 72, 'unit beta/rational sign-dedupe count mismatch');
assert(stats.signDedupe.ruleCounts['trig-Fourier reflection x->pi-x'] === 170, 'trig Fourier sign-dedupe count mismatch');
assert(stats.level4Rows === 6789, `unexpected level4 row count ${stats.level4Rows}`);
assert(stats.level5AdditionalRows === 29654, `unexpected level5 additional rows ${stats.level5AdditionalRows}`);
assert(stats.level4Fraction > 0.18 && stats.level4Fraction < 0.19, 'level4 fraction should be about one fifth');
assert(stats.level5CumulativeRows === 36443 && stats.level6CumulativeRows === 36443, 'levels 5/6 should expose full row table');
assert(stats.multiplierStageCounts['1'] === 1200, 'stage-1 multiplier count changed');
assert(stats.multiplierStageCounts['2'] === 5300, 'stage-2 multiplier count changed');
assert(stats.multiplierStageCounts['3'] === 9500, 'stage-3 multiplier count changed');
for(const asset of stats.assets){
  const actual = fs.statSync(asset.file).size;
  assert(actual === asset.assetBytes, `${asset.file} byte size mismatch: stats=${asset.assetBytes}, actual=${actual}`);
  assert(script.includes(`expectedBytes:${asset.assetBytes}`), `${asset.file} expectedBytes not reflected in ries-script.js`);
}

console.log('PASS RIES v11.9 integral/sum packaging regression test');
