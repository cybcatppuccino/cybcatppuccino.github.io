const fs = require('fs');
const {assert, loadRiesContext, runSuite} = require('./ries_test_utils');

runSuite('RIES packaging/startup', [
  ['visible version, unversioned active assets, and lazy payload boundaries', () => {
    const html = fs.readFileSync('ries.html','utf8');
    assert(html.includes('<title>RIES v12.0.2 · cybcat</title>'), 'ries.html title should be v12.0.2');
    assert(html.includes('RIES <em>v12.0.2</em>'), 'visible navbar version should be v12.0.2');
    assert(html.includes('src="ries-script.js"'), 'ries-script should be loaded without a cache-buster');
    assert(html.includes('src="assets/lfunctions-l2l4.js"'), 'L-function asset should be loaded without a cache-buster');
    assert(html.includes('src="assets/constantdb300.js"'), 'constant DB asset should be loaded without a cache-buster');
    assert(!/\?v=/.test(html), 'active HTML must not use versioned cache-buster query strings');
    for(const asset of ['ries-harddb-level4.js','ries-hypdata-level4.js','ries-intsumdb-level4.js','shortform100k.js']){
      assert(!new RegExp(`<script[^>]+${asset.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}`, 'i').test(html), `${asset} must remain lazily loaded, not part of initial HTML`);
    }
    assert(/\.latex-render[^}]*overflow-x:auto/.test(html), 'LaTeX render overflow CSS missing');
  }],
  ['JS syntax, unversioned lazy asset URLs, and progress helpers', () => {
    const script = fs.readFileSync('ries-script.js','utf8');
    const inline = fs.readFileSync('ries_inline.js','utf8');
    for(const name of ['ries-harddb-level4.js','ries-hypdata-level4.js','ries-hypdata-level4-meta.js','ries-hypdata-level5.js','ries-hypdata-level5-meta.js','ries-hypdata-level6.js','ries-hypdata-level6-meta.js','ries-intsumdb-level4.js','ries-intsumdb-level4-meta.js','ries-intsumdb-level5.js','ries-intsumdb-level5-meta.js','ries-intsumdb-level6.js','ries-intsumdb-level6-meta.js','shortform100k.js']){
      assert(script.includes(name), `main script should reference ${name}`);
      assert(!script.includes(name+'?v='), `${name} must not use cache-buster query strings`);
      assert(inline.includes(name), `inline script should mirror ${name}`);
    }
    for(const oldName of ['RIES_HARDDB_V1173_CHUNKS','RIES_HYPDATA_V1192_CHUNKS','RIES_INTSUMDB_V117_CHUNKS','ries-harddb-v11_7_3-level4.js','ries-hypdata-v11_9_2-level4.js','ries-intsumdb-v11_7-level4.js']){
      assert(!script.includes(oldName), `active script still uses versioned name ${oldName}`);
      assert(!inline.includes(oldName), `inline script still uses versioned name ${oldName}`);
    }
    assert(script.includes('base + span*frac'), 'package loader should map chunk load progress exactly onto its configured span');
    assert(script.includes('.675 + Math.min(.045, frac*.045)'), 'L-function progress should stay in the L-function progress range');
    const bundle = loadRiesContext();
    assert(bundle.context.__RIES_EQUATION_TEST__ && bundle.context.__RIES_CONSTDB_TEST__, 'core test hooks missing after script load');
  }],
  ['test hook coverage matrix', () => {
    const {context} = loadRiesContext({loadNewforms:true});
    const expected = {
      __RIES_EQUATION_TEST__: ['generateConstants','generateLHS','equationSearch','exprToLatex','sanitizeLatexForDisplay','latexMulScalar','algebraicRowFromCoeff'],
      __RIES_INTEGER_TEST__: ['displayExprMatchesTarget','integerRowFormulaIsValid','integerShortformRowsAsync','staticShortformRows','simplifyIntegerExpressionDisplay'],
      __RIES_LOG_TEST__: ['logRelationRows','linearCombinationLatex','resetSearchFrameworkForInputChange'],
      __RIES_CONSTDB_TEST__: ['constantDbRecords','constantDbRows','constantDbRowsAsync','constDbFindLinearRelation','constDbPolyToLatex','constantDbBudgetMs'],
      __RIES_LFUNC_TEST__: ['lfuncEntries','lfuncFormulaLatex','lfuncCompareCandidates','lfuncGlobalBestRows','lfuncDbStage','lfuncDbMultiplierCatalog'],
      __RIES_HARDDB_TEST__: ['ensureHardDbLoaded','hardDbRowsAsync','hardDbFormulaLatex','hardDbDecodeRowMeta','hardDbMaxStage'],
      __RIES_HYPDATA_TEST__: ['ensureHypDataLoaded','ensureHypDataMetaLoaded','isHypDataMetaReady','hypDataRowsAsync','hypDataMkLatex','hypDataMaxStage','hypDataStageBudgetMs'],
      __RIES_INTSUMDB_TEST__: ['ensureIntsumDbLoaded','ensureIntsumDbMetaLoaded','isIntsumDbMetaReady','intsumDbRowsAsync','intsumDbMulLatex','intsumDbCleanLatexFormula','intsumDbCleanPlainFormula','intsumDbMaxStage','intsumDbStageBudgetMs'],
      __RIES_PRECISION_TEST__: ['parseDecimalComplex','typedInputPrecisionDigits','typedInputPrecisionForDouble','specialDecimalConstantRows','decimalToBaseString'],
    };
    for(const [hook, funcs] of Object.entries(expected)){
      assert(context[hook], `${hook} missing`);
      for(const fn of funcs) assert(typeof context[hook][fn] === 'function', `${hook}.${fn} missing`);
    }
    assert(Array.isArray(context.__RIES_LOG_TEST__.logConstants) && context.__RIES_LOG_TEST__.logConstants.length > 0, '__RIES_LOG_TEST__.logConstants missing');
    assert(Array.isArray(context.RIES_LFUNCTIONS_L1) && context.RIES_LFUNCTIONS_L1.length === 358, 'weight 1 L-function data missing');
    assert(Array.isArray(context.RIES_LFUNCTIONS_L3) && context.RIES_LFUNCTIONS_L3.length === 127, 'weight 3 L-function data missing');
    assert(Array.isArray(context.NEWFORMS) && context.NEWFORMS.some(f=>f.weight===1) && context.NEWFORMS.some(f=>f.weight===3), 'homepage newforms should include weights 1 and 3');
  }],
  ['lazy database package loading and readiness', async () => {
    const {context} = loadRiesContext();
    const H = context.__RIES_HARDDB_TEST__, Y = context.__RIES_HYPDATA_TEST__, I = context.__RIES_INTSUMDB_TEST__;
    assert(!H.isHardDbReady(1), 'harddb should start unloaded');
    assert(!Y.isHypDataReady(1), 'hypdata should start unloaded');
    assert(!Y.isHypDataMetaReady(1), 'hypdata metadata should start unloaded');
    assert(!I.isIntsumDbReady(1), 'intsumdb should start unloaded');
    assert(!I.isIntsumDbMetaReady(1), 'intsumdb metadata should start unloaded');
    assert(await H.ensureHardDbLoaded({stage:1,label:'harddb',phase:'test'}), 'harddb level package failed to load');
    assert(await Y.ensureHypDataLoaded({stage:1,label:'hypdata',phase:'test'}), 'hypdata level4 package failed to load');
    assert(await I.ensureIntsumDbLoaded({stage:1,label:'intsumdb',phase:'test'}), 'intsumdb level4 package failed to load');
    assert(H.isHardDbReady(1) && context.RIES_HARDDB_CHUNKS[0].rows === 23608, 'harddb active chunk mismatch');
    assert(Y.isHypDataReady(1) && context.RIES_HYPDATA_CHUNKS[0].rows === 29618, 'hypergeom level4 index chunk mismatch');
    assert(I.isIntsumDbReady(1) && context.RIES_INTSUMDB_CHUNKS[0].rows === 6789, 'intsumdb level4 index chunk mismatch');
    assert(!context.RIES_HYPDATA_CHUNKS[0].mkBlob && !context.RIES_INTSUMDB_CHUNKS[0].plainBlob, 'index chunks must not eagerly include display metadata');
    assert(await Y.ensureHypDataMetaLoaded({stage:1,label:'hypdata',phase:'test-meta'}), 'hypdata level4 metadata package failed to load');
    assert(await I.ensureIntsumDbMetaLoaded({stage:1,label:'intsumdb',phase:'test-meta'}), 'intsumdb level4 metadata package failed to load');
    assert(Y.isHypDataMetaReady(1) && context.RIES_HYPDATA_CHUNKS[0].mkBlob, 'hypergeom level4 metadata mismatch');
    assert(I.isIntsumDbMetaReady(1) && context.RIES_INTSUMDB_CHUNKS[0].plainBlob, 'intsumdb level4 metadata mismatch');
  }],
  ['stage budgets and level gating', () => {
    const {context} = loadRiesContext();
    const H=context.__RIES_HARDDB_TEST__, Y=context.__RIES_HYPDATA_TEST__, I=context.__RIES_INTSUMDB_TEST__, C=context.__RIES_CONSTDB_TEST__, L=context.__RIES_LFUNC_TEST__;
    const base={level:6, target:2, complexTarget:false, modules:{hardDb:true,hypData:true,intsumDb:true,lfunc:true}, hardDbOptions:{depth4:true,depth5:true,depth6:true}, hypDataOptions:{depth1:true,depth2:true,depth3:true}, intsumDbOptions:{depth1:true,depth2:true,depth3:true}, stageBudgets:{}};
    assert(H.hardDbMaxStage({...base, level:3}) === 0 && H.hardDbMaxStage({...base, level:4}) === 1 && H.hardDbMaxStage({...base, level:5}) === 2 && H.hardDbMaxStage({...base, level:6}) === 3, 'harddb level to stage mapping changed');
    assert(Y.hypDataMaxStage({...base, level:3}) === 0 && Y.hypDataMaxStage({...base, level:4}) === 1 && Y.hypDataMaxStage({...base, level:5}) === 2 && Y.hypDataMaxStage({...base, level:6}) === 3, 'hypdata level to stage mapping changed');
    assert(I.intsumDbMaxStage({...base, level:3}) === 0 && I.intsumDbMaxStage({...base, level:4}) === 1 && I.intsumDbMaxStage({...base, level:5}) === 2 && I.intsumDbMaxStage({...base, level:6}) === 3, 'intsumdb level to stage mapping changed');
    assert(C.constantDbBudgetMs(4,16) === 20000 && C.constantDbBudgetMs(5,16) === 45000 && C.constantDbBudgetMs(6,16) === 135000, 'constant DB default budgets changed');
    assert(H.hardDbBudgetMs({...base, level:4, stageBudgets:{}}) === 3000 && H.hardDbBudgetMs({...base, level:5, stageBudgets:{}}) === 15000, 'harddb stage budgets changed');
    assert(Y.hypDataStageBudgetMs({...base, stageBudgets:{}},1) === 3000 && Y.hypDataStageBudgetMs({...base, stageBudgets:{}},2) === 15000, 'hypdata stage budgets changed');
    assert(I.intsumDbStageBudgetMs({...base, stageBudgets:{}},1) === 3000 && I.intsumDbStageBudgetMs({...base, stageBudgets:{}},2) === 15000, 'intsumdb stage budgets changed');
    assert(L.lfuncDbStage({level:4}) === 1 && L.lfuncDbStage({level:5}) === 2 && L.lfuncDbStage({level:6}) === 3, 'L-function transformed DB stage gating changed');
  }],
]).catch(err => { console.error(err); process.exit(1); });
