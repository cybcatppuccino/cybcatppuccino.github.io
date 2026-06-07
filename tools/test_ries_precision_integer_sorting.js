const {assert, loadRiesContext, settingsFor, runSuite} = require('./ries_test_utils');

runSuite('RIES precision/integer/sorting', [
  ['typed precision helpers and high-precision number tools', () => {
    const {context} = loadRiesContext();
    const P = context.__RIES_PRECISION_TEST__;
    assert(P.typedInputPrecisionDigits('1.2300') === 5, 'typed trailing zeroes must count as user precision');
    assert(P.typedInputPrecisionDigits('0.0012300') === 5, 'leading zeroes should not count, trailing significant zeroes should');
    assert(P.typedInputPrecisionDigits('1.234567890123456789') === 19, 'high precision decimal digit count changed');
    assert(P.typedRelativeToleranceNumber(10,100,0,15) <= 1e-8, 'relative tolerance floor changed unexpectedly');
    assert(!P.numberToolsShouldAppear({raw:'1.2345', _hpEval:{error:false,z:{re:new context.Decimal('1.2345'), im:new context.Decimal(0), bi:null}}}), 'plain decimal input should hide number tools');
    assert(P.numberToolsShouldAppear({raw:'12345'}), 'integer input should show number tools');
    context.Decimal.set({precision:220});
    const computed = {raw:'sqrt(2)', _hpEval:{error:false,z:{re:new context.Decimal(2).sqrt(), im:new context.Decimal(0), bi:null}}};
    assert(P.numberToolsShouldAppear(computed), 'computed expression should show number tools');
    const dec = P.decimalToBaseString(new context.Decimal(2).sqrt(), 10, 120);
    const frac = (dec.split('.')[1]||'').replace(/…$/,'');
    assert(frac.length >= 100, `base 10 expansion should keep high precision; got ${dec}`);
  }],
  ['parseDecimalComplex and special numeric rows', () => {
    const {context} = loadRiesContext();
    const P = context.__RIES_PRECISION_TEST__;
    const z = P.parseDecimalComplex('3+4i');
    assert(z && P.rationalToNumber(z.re) === 3 && P.rationalToNumber(z.im) === 4, 'parseDecimalComplex should parse 3+4i');
    const parsed = P.parseDecimalComplex('3.62560990822190');
    const rows = P.specialDecimalConstantRows({raw:'3.62560990822190', parsedComplex:parsed, target:P.rationalToNumber(parsed.re), level:5, complexTarget:false, lfuncOptions:{specialConstants:true}, limit:5}, 1);
    assert(rows.some(r=>/Γ\(1\/4\)/.test(r.candidate || '') && /\\Gamma\(1\/4\)/.test(r.latex || '')), 'special constants should include Gamma(1/4) hit');
  }],
  ['integer display formula validation', () => {
    const {context} = loadRiesContext();
    const T = context.__RIES_INTEGER_TEST__;
    for(const bad of ['48^2','96^2','4^5·3','2^8·6']){
      assert(!T.displayExprMatchesTarget(bad, 768n), `bad integer formula validated: ${bad}`);
    }
    for(const good of ['4^4·3','2^8·3','8·96','32·4!','floor(((5!/7)^7)/7!)']){
      const target = good.startsWith('floor') ? 86328n : 768n;
      assert(T.displayExprMatchesTarget(good, target), `valid integer formula rejected: ${good}`);
    }
  }],
  ['integer shortform DB loads lazily and rows validate', async () => {
    const bundle = loadRiesContext({defaults:{target:'768', level:'4', limit:'20'}});
    const {context} = bundle;
    const T = context.__RIES_INTEGER_TEST__;
    assert(typeof context.ensureShortformDbLoaded === 'function', 'shortform lazy loader missing');
    await context.ensureShortformDbLoaded();
    assert(context.RIES_SHORTFORM_100K_PACKED && context.RIES_SHORTFORM_100K_PACKED.version === '10.8.1', 'packed shortform DB version mismatch');
    const settings = settingsFor(bundle, '768', 4, {elements:{limit:'20'}});
    const staticRows = T.staticShortformRows(settings);
    const shortRows = await T.integerShortformRowsAsync(settings);
    const all = [...staticRows, ...shortRows];
    assert(all.length > 0, 'no integer rows generated for 768');
    for(const r of all.slice(0,60)){
      assert(T.integerRowFormulaIsValid(r), `invalid displayed integer row: ${JSON.stringify(r)}`);
      assert(!/structured product:\s*(48\^2|96\^2|4\^5·3|2\^8·6)/.test(r.candidate), `known bad structured-product row survived: ${r.candidate}`);
      assert(!/undefined|null|NaN/.test(String(r.latex||'')), `bad integer LaTeX payload: ${r.latex}`);
    }
  }],
  ['confidence sorting keeps short explanations ahead of dense artefacts', () => {
    const {context} = loadRiesContext();
    const sorted1 = context.confidenceSortedRows([
      {candidate:'log|c| linear relation: x ≈ e^(-17/8) * 2^(43/16) * 3^(9/2) * 5^(19/8) * exp(π*(-1/8))', value:'terms 5; height 72', err:1e-16, height:72n, terms:5},
      {candidate:'log|c| linear relation: x ≈ π^(-1) * Γ(1/4)', value:'terms 2; height 2; direct sparse', err:3.41e-14, height:2n, terms:2},
    ], {raw:'1.154067477233', normalizedRaw:'1.154067477233', target:1.154067477233});
    assert(/π\^\(-1\) \* Γ\(1\/4\)/.test(sorted1[0].candidate), 'short Γ(1/4)/π log relation must outrank dense high-precision artefact');
    const sorted2 = context.confidenceSortedRows([
      {candidate:'RIES equation: (√(x))² = exp(π)', value:'x = 23.14069263278', err:2e-12},
      {candidate:'log|c| linear relation: x ≈ exp(π)', value:'terms 1; height 1; direct sparse', err:8e-11, height:1n, terms:1},
      {candidate:'log|c| linear relation: x ≈ 2^(11/7) * 3^(13/8) * 5^(-17/9) * exp(π)', value:'terms 4; height 79', err:2e-13, height:79n, terms:4},
    ], {raw:'23.1406926327', normalizedRaw:'23.1406926327', target:23.1406926327});
    assert(/exp\(π\)/.test(sorted2[0].candidate) && !/2\^/.test(sorted2[0].candidate), 'short exp(pi) explanation should be first');
    const sorted3 = context.confidenceSortedRows([
      {candidate:'L-rational #2: x = 2857*L(f,1)/(313*π)', value:'relative residual 1e-20', err:1e-20, lfuncCategory:'rational', modForm:{code:'1.2.1'}},
      {candidate:'L-rational #1: x = L(f,1)', value:'relative residual 5.397e-20', err:5.397e-20, lfuncCategory:'rational', modForm:{code:'2.2.1'}},
    ], {raw:'2.29848605816074', normalizedRaw:'2.29848605816074', target:2.29848605816074});
    assert(/x = L\(f,1\)$/.test(sorted3[0].candidate), 'x=L(f,1) must outrank longer small-residual L-rational formulas');
  }],
  ['state reset caches exist and are cleared', () => {
    const {context} = loadRiesContext();
    const LOG = context.__RIES_LOG_TEST__;
    LOG.solveRunCache.set('dummy', {rows:[1]});
    LOG.integerGlobalCache.set('dummy', {rows:[1]});
    LOG.lfuncProgressCache.set('dummy', {rows:[1]});
    LOG.resetSearchFrameworkForInputChange();
    assert(LOG.solveRunCache.size === 0, 'solve run cache should clear');
    assert(LOG.integerGlobalCache.size === 0, 'integer cache should clear');
    assert(LOG.lfuncProgressCache.size === 0, 'lfunc progress cache should clear');
  }],
]).catch(err => { console.error(err); process.exit(1); });
