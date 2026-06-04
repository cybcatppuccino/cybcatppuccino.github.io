// Node smoke/regression tests for the RIES v7.3 L-function matcher.
// Run from repository root: node tools/test_lfunc_v7_3.js
const fs = require('fs');
const vm = require('vm');

const sandbox = { console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout,
  requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame:()=>{}, MathJax:null };
sandbox.window = sandbox;
function fakeEl(id){ return { id, value:'', checked:true, hidden:false, disabled:false, dataset:{}, style:{}, className:'', textContent:'', innerHTML:'', open:false,
  addEventListener(){}, setAttribute(){}, appendChild(){}, querySelector(){return fakeEl('q')}, getContext(){return {}} }; }
const els = {};
for(const id of ['resultBody','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis']) els[id]=fakeEl(id);
els.digits.value='0123456789'; els.restrictMode.value='full'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='20'; els.doEq.checked=false; els.doAlg.checked=true; els.doLog.checked=true; els.algHeight.value='1000000000000'; els.algDegree.value='10'; els.algResidualPower.value='2'; els.logHeight.value='400'; els.logSlack.value='2';
sandbox.document = { getElementById:(id)=>els[id]||fakeEl(id), querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:(tag)=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true} };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);

const D = sandbox.Decimal;
D.set({precision:80});
const PI = new D('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899');
const L2 = new D('0.25384186085591068433775892335090946104');
const L41 = new D('0.16137033512412171675825905530648096927');
const L42 = new D('0.41186132838619917101546352034039865020');
const GAMMA13 = new D('2.6789385347077476336556929409746776441286893779573011009504283275904176101677');
function sig(x,n){ return x.toSignificantDigits(n).toString(); }
function rowsFor(input){ els.target.value=input; const settings=sandbox.readSettings(); return sandbox.lfuncRows(settings); }
function hasHit(rows, kind, label){ return rows.some(r => String(r.candidate).startsWith('L-'+kind) && String(r.value).includes(label)); }
const cases = [
  {name:'L2 rational low 8 digits: L/pi', input:sig(L2.div(PI),8), kind:'rational', label:'L(f2#1,1; N=11)'},
  {name:'L2 rational high 25 digits: 3/2*L*pi^2', input:sig(L2.mul(PI.pow(2)).mul(3).div(2),25), kind:'rational', label:'L(f2#1,1; N=11)'},
  {name:'L2 quadratic low 7 digits: sqrt(2)*L', input:sig(L2.mul(new D(2).sqrt()),7), kind:'quadratic', label:'L(f2#1,1; N=11)'},
  {name:'L2 log-extra high 25 digits: log(2)*L', input:sig(L2.mul(new D(2).ln()),25), kind:'log', label:'L(f2#1,1; N=11)'},
  {name:'L4,1 quadratic high 25 digits: sqrt(3)*L', input:sig(L41.mul(new D(3).sqrt()),25), kind:'quadratic', label:'L(f4#1,1; N=5)'},
  {name:'L4,2 log-extra high 25 digits: Gamma(1/3)*L', input:sig(L42.mul(GAMMA13),25), kind:'log', label:'L(f4#1,2; N=5)'}
];
let ok = 0;
const lines = ['# RIES v7.3 L-function matcher test results', '', '| case | input | expected | status | first matching row |', '|---|---:|---|---|---|'];
for(const tc of cases){
  const rows = rowsFor(tc.input);
  const pass = hasHit(rows, tc.kind, tc.label);
  if(pass) ok++;
  const first = rows.find(r => String(r.candidate).startsWith('L-'+tc.kind) && String(r.value).includes(tc.label));
  lines.push(`| ${tc.name} | \`${tc.input}\` | ${tc.kind} / ${tc.label} | ${pass?'PASS':'FAIL'} | ${first ? '`'+String(first.candidate).replace(/`/g,'')+'`' : '—'} |`);
  console.log(`${pass?'PASS':'FAIL'} ${tc.name}`);
  if(!pass){ console.log(rows.map(r=>r.candidate+' :: '+r.value).join('\n')); }
}
lines.push('', `${ok}/${cases.length} cases passed.`);
fs.writeFileSync('tools/lfunc_v7_3_test_results.md', lines.join('\n'));
if(ok !== cases.length) process.exit(1);
