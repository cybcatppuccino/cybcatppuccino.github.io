const fs = require('fs');
const {assert, loadRiesContext, runSuite} = require('./ries_test_utils');

runSuite('RIES LaTeX/rendering', [
  ['core expression and sanitizer regressions', () => {
    const {context} = loadRiesContext();
    const E = context.__RIES_EQUATION_TEST__;
    assert(E.exprToLatex('1*x') === 'x', '1*x should display as x');
    assert(E.exprToLatex('x*1') === 'x', 'x*1 should display as x');
    assert(E.exprToLatex('5*(1/5)*sqrt(2)') === '\\sqrt{2}', 'scalar product should simplify to sqrt(2)');
    assert(E.exprToLatex('root(2+3,4)') === '\\sqrt[4]{2+3}', 'root(a,b) LaTeX changed');
    const logBase = E.exprToLatex('log_2(8)+log_pi(2)');
    assert(logBase.includes('\\log_{2}') && logBase.includes('\\log_{\\pi}'), `log_base display is wrong: ${logBase}`);
    const expr = E.exprToLatex('sqrt(2)+x^0+x^1+(x+1)^0+x^(1/2)+sin(pi/2)');
    assert(expr.includes('\\sqrt{2}') && expr.includes('\\sqrt{x}'), `sqrt/half-power display failed: ${expr}`);
    assert(!expr.includes('operatorname{sqrt}') && !/\^\{0\}|\^\{1\}|\^0(?!\d)|\^1(?!\d)/.test(expr.replace(/_0\^1/g,'')), `neutral powers leaked: ${expr}`);
    const normalized = E.sanitizeLatexForDisplay('1+1x-1\\sin(x)+x^{1-1}+\\log(1-\\sin(x))^{0}\\log(1+\\cos(x))^{1}+e^{--\\frac{1}{2}}');
    assert(!/[+\-(]1(?:x|\\sin|\\cos|\\log)/.test(normalized), `unit coefficients leaked: ${normalized}`);
    assert(!/\^\{0\}|\^\{1\}/.test(normalized), `neutral brace powers leaked: ${normalized}`);
    assert(!/[+-]{2}|\+\s*-|-\s*\+/.test(normalized), `sign run leaked: ${normalized}`);
    assert(normalized.includes('\\log(1+\\cos(x))') && normalized.includes('e^{\\frac{1}{2}}'), `structure changed unexpectedly: ${normalized}`);
    const neg = E.sanitizeLatexForDisplay('\\left(-x-1\\right)^{2}+\\left(-x+1\\right)^{4}');
    assert(neg.includes('x + 1') && neg.includes('x - 1') && !neg.includes('(-'), `negative even-power bases not normalized: ${neg}`);
    assert(E.sanitizeLatexForDisplay('√(1/(2+3))') === '\\sqrt{1/(2+3)}', 'nested radical range should be preserved');
  }],
  ['long formula, algebraic, and scalar multiplier display', () => {
    const {context} = loadRiesContext();
    const E = context.__RIES_EQUATION_TEST__, I = context.__RIES_INTSUMDB_TEST__, C = context.__RIES_CONSTDB_TEST__;
    const long = E.latexBreakLongFormulaForDisplay('x \\approx a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q', 20);
    assert(long.startsWith('\\begin{aligned}') && long.includes('\\\\&\\quad + h') && long.endsWith('\\end{aligned}'), `long formula should be aligned: ${long}`);
    const alg = E.algebraicRowFromCoeff([1n,-2n,1n], 'algebraic relation', {re:1, im:0});
    assert(alg && alg.latex && alg.latex.includes('x^{2}') && alg.latex.includes('= 0'), `algebraic relation should provide LaTeX: ${alg && alg.latex}`);
    assert(E.sanitizeLatexForDisplay('1\\,\\log 2') === '\\log 2', 'neutral 1 before log should be removed');
    assert(E.sanitizeLatexForDisplay('1\\,c') === 'c', 'neutral 1 before generic constant should be removed');
    assert(E.sanitizeLatexForDisplay('x \\approx 1\\,\\log 2 + 1\\,\\pi') === 'x \\approx \\log 2 + \\pi', 'neutral 1 should simplify in sums');
    assert(E.sanitizeLatexForDisplay('5\\,\\frac{1}{5}\\,S') === 'S', '5*1/5 scalar product should disappear');
    assert(E.sanitizeLatexForDisplay('\\frac{1}{2}\\,\\frac{2}{\\pi}') === '\\frac{1}{\\pi}', 'adjacent constant fractions should simplify over pi');
    const powFrac = E.sanitizeLatexForDisplay('x^{\\frac{1}{2}-1}+y^{1/2-1}+z^{2-1}');
    assert(powFrac === 'x^{-\\frac{1}{2}}+y^{-\\frac{1}{2}}+z', `fractional exponent arithmetic failed: ${powFrac}`);
    assert(I.intsumDbMulLatex('5','\\frac{1}{5}\\,\\int_0^1 x\\,dx') === '\\int_0^1 x\\,dx', 'intsum scalar 5*1/5 should simplify');
    assert(I.intsumDbMulLatex('-1','x+1') === '-\\left(x+1\\right)', 'intsum -1 multiplier grouping failed');
    assert(I.intsumDbMulLatex('\\frac{2}{3}','\\int_0^1 x\\,dx') === '\\frac{2}{3}\\,\\int_0^1 x\\,dx', 'intsum fractional multiplier failed');
    const alphaEq = C.constDbPolyToLatex([1,-2,1], '\\alpha');
    assert(alphaEq.includes('\\alpha^{2}') && alphaEq.includes('= 0'), `constant DB alpha equation should be LaTeX with equality: ${alphaEq}`);
  }],
  ['database formula LaTeX including pFq lower parameters', () => {
    const {context} = loadRiesContext();
    const H = context.__RIES_HARDDB_TEST__, Y = context.__RIES_HYPDATA_TEST__;
    const hyp = Y.hypDataMkLatex('P|0|1/6,1/2,5/6|1,1|1');
    assert(hyp.includes('1, 1\\end{array}'), `hypergeom lower parameter 1 should not be swallowed: ${hyp}`);
    const hard = H.hardDbFormulaLatex({category:'generalized hypergeometric value', cid:'ones', params:{a1:'1/6',a2:'1/2',a3:'5/6',b1:'1',b2:'1',z:'1'}});
    assert(hard.includes('1, 1\\end{array}'), `harddb 3F2 lower parameter 1 should not be swallowed: ${hard}`);
    const hyp3f2 = Y.hypDataMkLatex('P|0|1/2,2/3,3/4|4/5,5/6|-1/2');
    assert(hyp3f2.includes('{}_{3}F_{2}') && hyp3f2.includes('\\begin{array}{c}') && hyp3f2.includes('\\frac{4}{5}, \\frac{5}{6}'), `hypdata 3F2 lower parameters broken: ${hyp3f2}`);
    const pref = Y.hypDataMkLatex('P|1/2|1/2,-1/2|1|-1/2');
    assert(pref.startsWith('\\frac{1}{2}\\,'), `hypergeom prefactor should be scalar, not P-subscript: ${pref}`);
    assert(!pref.includes('P_{') && !/[\u0008\u0009\u000c\u000d]/.test(pref), `hypergeom LaTeX escape failure: ${pref}`);
    const beta = H.hardDbFormulaLatex({category:'Euler beta integral', params:{a:'1',b:'1'}, cid:'beta_unit'});
    assert(beta === '\\int_0^1 \\,dx', `harddb neutral powers not removed: ${beta}`);
    const gamma = H.hardDbFormulaLatex({category:'gamma log-laplace integral', params:{a:'1',q:'-1',logPower:'0'}, cid:'gamma_unit'});
    assert(gamma.includes('e^{x}') && !gamma.includes('e^{1x}') && !gamma.includes('--'), `harddb signed exponent/unit coefficient failed: ${gamma}`);
  }],
  ['L-function and log LaTeX escaping', () => {
    const {context} = loadRiesContext();
    const L = context.__RIES_LFUNC_TEST__, LOG = context.__RIES_LOG_TEST__;
    const ltex = L.lfuncFormulaLatex('1/2·2/π·L(f,1/2)','x');
    assert(ltex.includes('\\frac{1}{\\pi}\\cdot L(f,\\tfrac{1}{2})'), `Lfunc constants should simplify rational*pi factors: ${ltex}`);
    assert(context.lfuncFormulaLatex('L(f,1/2)').includes('\\tfrac{1}{2}'), 'L(f,1/2) LaTeX should use a fraction');
    assert(context.lfuncFormulaLatex('L(f,3/2)').includes('\\tfrac{3}{2}'), 'L(f,3/2) LaTeX should use a fraction');
    const loglog2 = LOG.logConstants.find(c=>c.id==='loglog2');
    const loglog3 = LOG.logConstants.find(c=>c.id==='loglog3');
    const tex = LOG.linearCombinationLatex({coeff:[1n,-1n,-1n], rhs:0, err:0, height:1n}, [loglog2, loglog3]);
    assert(tex.includes('\\log\\!\\left(\\log 2\\right)') && tex.includes('\\log\\!\\left(\\log 3\\right)'), `log(log ·) constants should remain log(log ·): ${tex}`);
    assert(!/(?:^|[^\\])\\log 2 \+ \\log 3/.test(tex), `linear-combination LaTeX collapsed log(log) constants: ${tex}`);
  }],
  ['LaTeX payloads avoid bare approx/control escapes', () => {
    const {context} = loadRiesContext();
    const E = context.__RIES_EQUATION_TEST__;
    const samples = [
      E.exprToLatex('(1+2)/3'),
      E.sanitizeLatexForDisplay('x \\approx \\log\\left(6 + c + c^{2}\\right)'),
      context.__RIES_HYPDATA_TEST__.hypDataMkLatex('P|0|1/2,2/3|4/5|-1/2'),
      context.__RIES_INTSUMDB_TEST__.intsumDbMulLatex('\\frac{2}{3}','\\int_0^1 x\\,dx'),
    ];
    for(const latex of samples){
      assert(!/[\f\r\b\x07]/.test(latex), `control escape in LaTeX: ${JSON.stringify(latex)}`);
      assert(!/\bapprox\b/.test(latex.replace(/\\approx/g,'')), `bare approx leaked: ${latex}`);
      assert(!/undefined|null|NaN/.test(latex), `bad token leaked in LaTeX: ${latex}`);
    }
  }],
]).catch(err => { console.error(err); process.exit(1); });
