const fs = require('fs');
const {assert, loadRiesContext, b64F64, b64U8, runSuite} = require('./ries_test_utils');

function baseSettings(target, level=4){
  return {
    raw:String(target), normalizedRaw:String(target), target:Number(target), complexTarget:false, level, limit:20,
    modules:{hardDb:true,hypData:true,intsumDb:true},
    moduleLimits:{hardDb:20,hypData:20,intsumDb:20},
    hardDbOptions:{depth4:true,depth5:true,depth6:true,rational:true,power:true,exponential:true,logScale:true,rationalHeight:20,maxParamHeight:200},
    hypDataOptions:{depth1:true,depth2:true,depth3:true,multSimple:true},
    intsumDbOptions:{depth1:true,depth2:true,depth3:true,multSimple:true},
    stageBudgets:{hardDb4Ms:3000,hardDb5Ms:15000,hardDb6Ms:50000,hypData1Ms:3000,hypData2Ms:15000,hypData3Ms:50000,intsumDb1Ms:3000,intsumDb2Ms:15000,intsumDb3Ms:50000}
  };
}

function firstFiniteNonzero(arr){
  const a = Array.from(arr);
  return a.find(v => Number.isFinite(v) && Math.abs(v) > 1e-9);
}
function assertCleanRow(row, lhs){
  assert(row, `missing row for ${lhs}`);
  assert(String(row.latex).includes(lhs), `missing ${lhs} in row LaTeX: ${row.latex}`);
  assert(String(row.latex).includes('\\approx'), `closed form must use \\approx: ${row.latex}`);
  assert(!/\bapprox\b/.test(String(row.latex).replace(/\\approx/g,'')), `bare approx leaked: ${row.latex}`);
  assert(!/[\u0008\u0009\u000c\u000d]/.test(String(row.latex)), `control escape leaked: ${row.latex}`);
}

function pFqCountsFromMkBlob(blob){
  const counts = new Map();
  for(const line of String(blob||'').split('\n')){
    if(!line) continue;
    const p = line.split('|');
    const upper = p[2] ? p[2].split(',').filter(Boolean).length : 0;
    const lower = p[3] ? p[3].split(',').filter(Boolean).length : 0;
    const key = `${upper}F${lower}`;
    counts.set(key, (counts.get(key)||0)+1);
  }
  return counts;
}
function countOnlyWithin(counts, allowed){
  const allowedSet = new Set(allowed);
  return [...counts.keys()].every(k => allowedSet.has(k));
}

runSuite('RIES database modules', [
  ['asset stats and prune/merge metadata', () => {
    const hardStats = JSON.parse(fs.readFileSync('assets/ries-harddb-stats.json','utf8'));
    assert(hardStats.remainingRows === 23608 && hardStats.removedRows === 56324, 'harddb prune stats changed');
    const hypStats = JSON.parse(fs.readFileSync('assets/ries-hypdata-stats.json','utf8'));
    assert(hypStats.rows === 136170, 'hypergeom stats changed');
    assert(hypStats.mergeFilterStats.rowRationalExcluded === 593 && hypStats.mergeFilterStats.rowDuplicatesRemoved === 2297, 'hypergeom rational/dedupe report counts changed');
    assert(hypStats.scalarProjectionStats.scalarProjectionRationalExcluded === 5452 && hypStats.scalarProjectionStats.scalarProjectionDuplicatesRemoved === 3991, 'hypergeom projection rational/dedupe counts changed');
    const script = fs.readFileSync('ries-script.js','utf8');
    assert(script.includes("constantDbSource:'merged-hypdata'"), 'hypergeom result source marker missing');
    assert(script.includes("constantDbSource:'harddb-pruned'"), 'harddb result source marker missing');
  }],
  ['target-view transforms are shared by harddb/hypdata/intsumdb', () => {
    const {context} = loadRiesContext();
    const H=context.__RIES_HARDDB_TEST__, Y=context.__RIES_HYPDATA_TEST__, I=context.__RIES_INTSUMDB_TEST__;
    const base = baseSettings(2,4);
    assert(H.dbComparisonTargetViews(base).map(v=>v.id).join('|') === 'x|exp|logabs', 'target views should be x|exp|logabs');
    assert(!H.dbComparisonTargetViews({...base,target:11}).some(v=>v.id==='exp'), 'exp(x) view must be disabled when x > 10');
    assert(H.dbComparisonTargetViews({...base,target:0}).map(v=>v.id).join('|') === 'exp', 'x=0 should still permit exp(x) matching');
    assert(Y.dbComparisonTargetViews(base).map(v=>v.id).join('|') === 'x|exp|logabs', 'hypdata target views mismatch');
    assert(I.dbComparisonTargetViews(base).map(v=>v.id).join('|') === 'x|exp|logabs', 'intsum target views mismatch');
  }],
  ['harddb runtime, source marker, and formula sampling', async () => {
    const {context} = loadRiesContext();
    const H = context.__RIES_HARDDB_TEST__;
    assert(await H.ensureHardDbLoaded({stage:3,label:'harddb',phase:'test'}), 'harddb should load');
    assert(H.hardDbLoadedChunks(3).length === 1, 'harddb should use one pruned chunk at all depths');
    const asset = context.RIES_HARDDB_CHUNKS[0];
    assert(asset.rows === 23608, 'active harddb rows mismatch');
    const removed = new Set(['low-height hypergeometric pFq','Euler beta integral fast','incomplete beta integral fast','beta logarithmic integral fast','gamma log-laplace integral fast','rational Mellin integral fast']);
    for(let i=0;i<Math.min(300, asset.rows);i+=7){
      const meta = H.hardDbDecodeRowMeta(i, 3);
      assert(!removed.has(meta.category), `removed category decoded from active rows: ${meta.category}`);
      const latex = H.hardDbFormulaLatex(meta);
      assert(latex && !/[\u0008\u0009\u000c\u000d]/.test(latex), `harddb bad latex at row ${i}: ${latex}`);
      assert(!/[+-]{2}|\+\s*-|-\s*\+/.test(latex), `harddb sign run at row ${i}: ${latex}`);
    }
    const vals = b64F64(asset.valuesB64);
    const target = Math.abs(firstFiniteNonzero(vals));
    const rows = await H.hardDbRowsAsync(baseSettings(target,4));
    assert(rows.length > 0 && rows.length <= 20, 'harddb should return capped rows for direct target');
    assert(rows[0].constantDbSource === 'harddb-pruned', 'harddb source marker mismatch');
    assertCleanRow(rows[0], 'x');
  }],
  ['harddb transformed exp/log rows', async () => {
    const {context} = loadRiesContext();
    const H = context.__RIES_HARDDB_TEST__;
    await H.ensureHardDbLoaded({stage:1,label:'harddb',phase:'test'});
    const val = firstFiniteNonzero(b64F64(context.RIES_HARDDB_CHUNKS[0].valuesB64));
    const expRows = await H.hardDbRowsAsync(baseSettings(Math.log(Math.abs(val)),4));
    assertCleanRow(expRows.find(r=>String(r.candidate).includes('exp(x) ≈')), '\\exp(x)');
    const logRows = await H.hardDbRowsAsync(baseSettings(Math.exp(val),4));
    assertCleanRow(logRows.find(r=>String(r.candidate).includes('log|x| ≈')), '\\log\\left|x\\right|');
  }],
  ['hypergeom level chunks preserve requested pFq layering', async () => {
    const {context} = loadRiesContext();
    const Y = context.__RIES_HYPDATA_TEST__;
    assert(await Y.ensureHypDataLoaded({stage:3,label:'hypergeometric pFq database',phase:'test'}), 'all hypdata chunks should load');
    const chunks = context.RIES_HYPDATA_CHUNKS;
    assert(!chunks[0].mkBlob && !chunks[1].mkBlob && !chunks[2].mkBlob, 'hypdata index chunks should not eagerly include metadata');
    assert(await Y.ensureHypDataMetaLoaded({stage:3,label:'hypergeometric pFq database',phase:'test-meta'}), 'all hypdata metadata chunks should load');
    const c4 = pFqCountsFromMkBlob(chunks[0].mkBlob);
    const c5 = pFqCountsFromMkBlob(chunks[1].mkBlob);
    const c6 = pFqCountsFromMkBlob(chunks[2].mkBlob);
    assert(c4.get('2F1') > 0 && c4.get('3F2') > 0, 'level4 must contain 2F1 and 3F2 rows');
    assert(countOnlyWithin(c4, ['2F1','3F2']), 'level4 additional chunk must be exactly 2F1/3F2');
    assert(c5.get('4F3') > 0 && c5.get('5F4') > 0 && countOnlyWithin(c5, ['4F3','5F4']), 'level5 additional chunk must be exactly 4F3/5F4');
    assert(c6.get('6F5') > 0 && c6.get('7F6') > 0 && c6.get('8F7') > 0, 'level6 chunk must contain deep remaining families');
    const cumulative5 = new Set([...c4.keys(), ...c5.keys()]);
    assert(cumulative5.has('2F1') && cumulative5.has('3F2') && cumulative5.has('4F3') && cumulative5.has('5F4'), 'level5 cumulative load must cover 2F1,3F2,4F3,5F4');
    const total = chunks.reduce((a,ch)=>a+Number(ch.rows||0),0);
    assert(total === 136170, 'level6 cumulative row count should equal the full hypergeom database');
  }],
  ['hypergeom merge, Re/Im projections, and cumulative levels', async () => {
    const {context} = loadRiesContext();
    const Y = context.__RIES_HYPDATA_TEST__;
    assert(Y.RIES_HYPDATA_ASSET_LEVELS[0].url === 'assets/ries-hypdata-level4.js', 'hypergeom level4 asset URL should be versionless');
    assert(await Y.ensureHypDataLoaded({stage:1,label:'hypergeometric pFq database',phase:'test'}), 'level4 hypdata failed to load');
    const ch = context.RIES_HYPDATA_CHUNKS[0];
    assert(ch.version === '12.0.2' && ch.rows === 29618 && ch.realCompB64 && !ch.mkBlob, 'level4 search-index payload mismatch');
    assert(await Y.ensureHypDataMetaLoaded({stage:1,label:'hypergeometric pFq database',phase:'test-meta'}), 'level4 hypdata metadata failed to load');
    assert(ch.mkBlob.includes('P|0|1/12,1/12|-1/2|-1'), 'data.zip 2F1 grid row missing from level4 metadata');
    const comps = b64U8(ch.realCompB64), vals = b64F64(ch.realValuesB64);
    assert(comps.includes(1) && comps.includes(2), 'Re/Im scalar projection component codes missing');
    const k = comps.findIndex((c,i)=>c===1 && Number.isFinite(vals[i]) && Math.abs(vals[i])>1e-6);
    assert(k >= 0, 'no Re(H) projection available for smoke test');
    const target = vals[k];
    const out = await Y.hypDataRowsAsync(baseSettings(target,4));
    assert(out.length > 0, 'level4 projection target did not search');
    assert(out.some(r => String(r.latex||'').includes('\\operatorname{Re}') || String(r.valueHtml||'').includes('Re(H)')), 'Re(H) LaTeX/display not found in projection hit');
    assert(await Y.ensureHypDataLoaded({stage:3,label:'hypergeometric pFq database',phase:'test'}), 'level6 hypdata failed to load');
    assert(Y.hypDataLoadedChunks(3).length === 3, 'level6 should have all cumulative chunks');
    const out6 = await Y.hypDataRowsAsync(baseSettings(target,6));
    assert(out6.length > 0, 'level6 cumulative search missed level4 projection target');
  }],
  ['integral/sum runtime and cumulative levels', async () => {
    const {context} = loadRiesContext();
    const I = context.__RIES_INTSUMDB_TEST__;
    assert(await I.ensureIntsumDbLoaded({stage:1,label:'integral/sum database',phase:'test'}), 'intsum stage1 should load');
    assert(context.RIES_INTSUMDB_CHUNKS[0].rows === 6789, 'intsum stage1 row count mismatch');
    assert(context.RIES_INTSUMDB_CHUNKS[0].multiplierRows === 1200, 'intsum stage1 multiplier count mismatch');
    assert(!context.RIES_INTSUMDB_CHUNKS[0].plainBlob, 'intsum stage1 index should not eagerly include display metadata');
    const target = firstFiniteNonzero(b64F64(context.RIES_INTSUMDB_CHUNKS[0].valuesB64));
    const rows = await I.intsumDbRowsAsync(baseSettings(target,4));
    assert(rows.length > 0, 'expected at least one intsumdb row');
    assert(I.isIntsumDbMetaReady(1) && context.RIES_INTSUMDB_CHUNKS[0].plainBlob, 'intsum row formatting should lazy-load metadata after a hit');
    assert(String(rows[0].candidate).startsWith('integral/sum database:'), 'candidate should identify intsumdb');
    assert(String(rows[0].valueHtml).includes('\\(') && String(rows[0].valueHtml).includes('\\)'), 'valueHtml should keep LaTeX delimiters');
    assert(rows[0].constantDbSource === 'intsumdb', 'intsum source marker mismatch');
    assert(I.resultRowCategory(rows[0]) === 'intsumdb', 'intsum category should be visible to sorter/category logic');
    assert(await I.ensureIntsumDbLoaded({stage:2,label:'integral/sum database',phase:'test'}), 'intsum stage2 should load');
    assert(context.RIES_INTSUMDB_CHUNKS[1].rows === 29654 && context.RIES_INTSUMDB_CHUNKS[1].multiplierRows === 5300, 'intsum stage2 counts mismatch');
    assert(await I.ensureIntsumDbLoaded({stage:3,label:'integral/sum database',phase:'test'}), 'intsum stage3 should load');
    assert(context.RIES_INTSUMDB_CHUNKS[2].rows === 0 && context.RIES_INTSUMDB_CHUNKS[2].multiplierRows === 9500, 'intsum stage3 counts mismatch');
  }],
  ['integral/sum LaTeX display cleanup for generated zero/unit terms', () => {
    const {context} = loadRiesContext();
    const I = context.__RIES_INTSUMDB_TEST__;
    const latex = I.intsumDbCleanLatexFormula('\\int_0^1 x(1-x)e^{0-x+0x^2}(1+0x+0x^2)\\,dx');
    assert(latex === '\\int_0^1 x(1-x)e^{-x}\\,dx', `intsum zero/unit latex cleanup failed: ${latex}`);
    const plain = I.intsumDbCleanPlainFormula('int_0^1 x^1 (1-x)^1 exp(0+-1 x+0 x^2) (1+0 x+0 x^2)^1 dx');
    assert(plain === 'int_0^1 x (1-x) exp(-x) dx', `intsum zero/unit text cleanup failed: ${plain}`);
    assert(I.intsumDbMulLatex('-1', '\\Gamma(1/3)') === '-\\Gamma(1/3)', 'intsum -1 multiplier latex cleanup failed');
    assert(I.intsumDbMulLatex('1', '\\Gamma(1/3)') === '\\Gamma(1/3)', 'intsum +1 multiplier latex cleanup failed');
    const trigPlain = I.intsumDbCleanPlainFormula('int_0^pi cos(1 x) log(1+1 cos x)^2 / (1+1/2 cos x)^1 dx');
    assert(trigPlain === 'int_0^pi cos(x) log(1+cos x)^2 / (1+1/2 cos x) dx', `intsum trig unit cleanup failed: ${trigPlain}`);
  }],
  ['integral/sum transformed exp/log rows', async () => {
    const {context} = loadRiesContext();
    const I = context.__RIES_INTSUMDB_TEST__;
    await I.ensureIntsumDbLoaded({stage:1,label:'intsum',phase:'test'});
    const val = firstFiniteNonzero(b64F64(context.RIES_INTSUMDB_CHUNKS[0].valuesB64));
    const expRows = await I.intsumDbRowsAsync(baseSettings(Math.log(Math.abs(val)),4));
    assertCleanRow(expRows.find(r=>String(r.candidate).includes('exp(x) ≈')), '\\exp(x)');
    const logRows = await I.intsumDbRowsAsync(baseSettings(Math.exp(val),4));
    assertCleanRow(logRows.find(r=>String(r.candidate).includes('log|x| ≈')), '\\log\\left|x\\right|');
  }],
]).catch(err => { console.error(err); process.exit(1); });
