const {assert, loadRiesContext, settingsFor, settingsForDecimal, runSuite} = require('./ries_test_utils');

function truncSigString(str,d){
  const s=String(str); const neg=s.startsWith('-'); const body=neg?s.slice(1):s; const [a,b='']=body.split('.');
  const intSig=a.replace(/^0+/,'').length;
  if(intSig>=d) return (neg?'-':'')+a.slice(0,d);
  return (neg?'-':'')+a+'.'+b.slice(0,Math.max(0,d-intSig));
}
function toSig(x,d){ return Number(x).toPrecision(d).replace(/(?:\.0+|(?<=\d)0+)$/,''); }

runSuite('RIES constdb/log/L-functions', [
  ['constant DB targeted rows and LaTeX escaping', () => {
    const bundle = loadRiesContext({realTimePerformance:true});
    const {context} = bundle;
    const C = context.__RIES_CONSTDB_TEST__;
    const s = settingsFor(bundle, '2.386110381167886', 4);
    const rows = C.constantDbRows(s);
    const logPiPoly = rows.find(r => /c = π/.test(r.valueHtml||'') && /(?:low-degree polynomial relation in transformed value|quadratic relation in b),1,c,c\^2/.test(r.constantDbCategory||''));
    assert(logPiPoly, 'missing constant DB log-polynomial hit');
    assert(/exp\(\(−12 \+ 4·c \+ c\^2\)\/12\)/.test(logPiPoly.candidate||''), `bad candidate text: ${logPiPoly.candidate}`);
    assert(/^x \\approx \\exp\\left\(\\frac\{-12 \+ 4\\,c \+ c\^\{2\}\}\{12\}\\right\)$/.test(logPiPoly.latex||''), `bad constant DB latex: ${JSON.stringify(logPiPoly.latex)}`);
    for(const row of rows.slice(0,20)){
      const latex = String(row.latex || '');
      assert(!/[\f\r\b]/.test(latex), `control escape in constant DB latex: ${JSON.stringify(latex)}`);
      assert(!/\bapprox\b/.test(latex.replace(/\\approx/g,'')), `bare approx in constant DB latex: ${latex}`);
    }
  }],
  ['constant DB precision and relation helper coverage', () => {
    const bundle = loadRiesContext();
    const C = bundle.context.__RIES_CONSTDB_TEST__;
    const s15 = settingsFor(bundle, '1.23456789012345', 4);
    const s8 = settingsFor(bundle, '1.2345678', 4);
    assert(C.typedInputPrecisionForDouble(s15) === 15, '15-digit input should remain 15 digits');
    assert(C.typedInputPrecisionForDouble(s8) === 8, 'sub-16 digit input precision should remain unchanged');
    const recs = C.constantDbRecords();
    assert(Array.isArray(recs) && recs.length >= 300, 'constant DB records should load');
    const piRec = recs.find(r => r.symbol === 'π' || r.name === 'pi' || /π|pi/i.test(String(r.label||r.name||'')));
    assert(piRec, 'constant DB should include a pi-like record');
    const poly = C.constDbPolyToLatex([1,0,-2], 'c');
    assert(poly.includes('c^{2}') && poly.includes('= 0'), `const DB polynomial LaTeX changed: ${poly}`);
  }],
  ['log relation precision policy and sparse products', () => {
    const bundle = loadRiesContext();
    const {context} = bundle;
    const P = context.__RIES_PRECISION_TEST__, LOG = context.__RIES_LOG_TEST__;
    assert(P.typedInputPrecisionDigits('1.2300') === 5, 'typed trailing zeroes must count as user precision');
    assert(P.typedInputPrecisionDigits('0.0012300') === 5, 'leading zeroes should not count, trailing significant zeroes should');
    assert(P.matchToleranceDigits(11,1,30) <= 10, 'match tolerance must not exceed typed precision bucket');
    for(const d of [6,10,15]){
      const raw = toSig(6*Math.PI**5, d);
      const st = settingsFor(bundle, raw, 4);
      st.normalizedRaw = '1836118108711395/1000000000000';
      assert(P.typedInputPrecision(st) === d, `typed precision from raw failed for ${raw}`);
      const rows = LOG.logRelationRows(st.target, st);
      assert(rows.some(r=>/2 \* 3 \* π\^\(5\)/.test(r.candidate)), `log|c| missed 6*pi^5 at ${d} significant digits (${raw})`);
    }
    const rel = {coeff:[3n,0n,-2n,0n,12n], rhs:(2/3)*Math.log(2)-4*Math.log(5), err:1e-16, height:12n};
    assert(LOG.logProductLatex(rel, LOG.logConstants) === '2^{2/3}\\,5^{-4}', 'bad log|c| product latex');
  }],
  ['log first-continue and special decimal constants', () => {
    const bundle = loadRiesContext();
    const {context} = bundle;
    const P = context.__RIES_PRECISION_TEST__, LOG = context.__RIES_LOG_TEST__;
    const gamma13 = 2.678938534707747633655692940974677644128689377957302;
    for(const d of [7,8]){
      const raw = toSig(gamma13/Math.log(Math.PI), d);
      const rows = LOG.logRelationRows(Number(raw), settingsFor(bundle, raw, 5));
      assert(rows.some(r=>/Γ\(1\/3\)/.test(r.candidate) && /log\(π\)\^\(-1\)/.test(r.candidate)), `log|c| missed Gamma(1/3)/log(pi) at ${d} digits (${raw})`);
    }
    for(const d of [6,10,15]){
      const raw = truncSigString('3.625609908221908311930685155867672002995167682880065467433377', d);
      const rows = P.specialDecimalConstantRows(settingsFor(bundle, raw, 5), 1);
      assert(rows.some(r => /Γ\(1\/4\)/.test(r.candidate) && /\\Gamma\(1\/4\)/.test(r.latex || '')), `Gamma(1/4) missed or bad LaTeX at ${d} typed digits (${raw})`);
    }
  }],
  ['L-function data integration and formula display', () => {
    const {context} = loadRiesContext();
    const L = context.__RIES_LFUNC_TEST__;
    const entries = L.lfuncEntries();
    assert(entries.some(e=>e.weight===1 && e.which==='1/2' && e.label==='L(f,1/2)'), 'RIES entries missing weight 1 L(f,1/2)');
    assert(entries.some(e=>e.weight===3 && e.which==='3/2' && e.label==='L(f,3/2)'), 'RIES entries missing weight 3 L(f,3/2)');
    assert(entries.some(e=>e.weight===2 && e.which==='1'), 'existing weight 2 entries missing');
    assert(entries.some(e=>e.weight===4 && e.which==='2'), 'existing weight 4 entries missing');
    assert(L.lfuncDbMultiplierCatalog(1).length < L.lfuncDbMultiplierCatalog(2).length && L.lfuncDbMultiplierCatalog(2).length < L.lfuncDbMultiplierCatalog(3).length, 'higher L-function transform stages should add multiplier products');
    const q = L.lfuncQExpansionLatex({coeffs:[0,1,-22,333,-4444,55555,-666666,7777777,-88888888,999999999,-1111111111,2222222222]});
    assert(q.includes('\\begin{aligned}') && q.includes('\\\\&\\quad'), 'long q-expansion should split into aligned display lines');
  }],
  ['L-function sorting and global cap policy', () => {
    const {context} = loadRiesContext();
    const T = context.__RIES_LFUNC_TEST__;
    const cmp = T.lfuncCompareCandidates(12);
    const exactButTall = {kind:'quadratic', i:2, j:3, height:50000, err:1e-20, formula:'complex', L:{entryKey:'a'}};
    const simpleWithin100x = {kind:'rational', i:1, j:0, height:2, err:8e-10, formula:'simple', L:{entryKey:'b'}};
    assert([exactButTall, simpleWithin100x].sort(cmp)[0] === simpleWithin100x, 'within 100× typed tolerance, simpler/low-height form should rank first');
    const tooLooseSimple = {kind:'rational', i:1, j:0, height:2, err:1e-6, formula:'loose', L:{entryKey:'c'}};
    assert([tooLooseSimple, exactButTall].sort(cmp)[0] === exactButTall, 'outside 100× tolerance, error should dominate');
    const best = T.lfuncGlobalBestRows([
      {kind:'log', formula:'g', L:{entryKey:'1'}, height:1, err:1e-20},
      {kind:'rational', i:1, j:0, formula:'r1', L:{entryKey:'2'}, height:4, err:9e-10},
      {kind:'rational', i:1, j:1, formula:'r2', L:{entryKey:'3'}, height:4, err:9e-10},
      {kind:'quadratic', i:1, j:0, formula:'q', L:{entryKey:'4'}, height:9, err:9e-10},
      {kind:'rational', i:-1, j:0, formula:'r3', L:{entryKey:'5'}, height:4, err:9e-10},
      {kind:'log', formula:'g2', L:{entryKey:'6'}, height:2, err:9e-10}
    ], 5, 12);
    assert(best.length === 5, 'L-function global result cap should keep only five rows');
    assert(best[0].formula === 'r1', 'global L-function cap should preserve simplicity-first order inside tolerance');
  }]
]).catch(err => { console.error(err); process.exit(1); });
