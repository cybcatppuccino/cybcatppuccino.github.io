// RIES v10.6 regression: after all async modules complete, confidence order
// must collect complete module queues and display first-layer rows before any
// second algebraic row.  Short L/log heads must not be buried under algebraic.
const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){}},addEventListener(){},setAttribute(){},appendChild(){},prepend(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}}}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.target.value='0.418991077502'; els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='9'; els.shortEffort.value='3'; els.limit.value='10'; els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true;
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null}; sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
const settings={raw:'0.418991077502', target:0.418991077502};
const rows=[
  {candidate:'irreducible algebraic: 3x^7 + 7x^6 − 9x^5 − 9x^4 + 2x^3 + 5x^2 − 4x + 1 = 0', value:'x = 0.4189910775008; |P(input)| ≈ 2.553*10^-12', err:1.2e-12, degree:7, height:9n},
  {candidate:'irreducible algebraic: 2x^9 + 3x^8 + x^7 − 2x^6 + 8x^5 + 3x^4 − 7x^3 − 4x + 2 = 0', value:'x = 0.4189910775033; |P(input)| ≈ 7.131*10^-12', err:1.27e-12, degree:9, height:8n},
  {candidate:'irreducible algebraic: 3x^6 − 11x^5 + 33x^4 + 19x^3 + 29x^2 − 20x + 1 = 0', value:'x = 0.4189910775023; |P(input)| ≈ 5.921*10^-12', err:2.63e-13, degree:6, height:33n},
  {candidate:'irreducible algebraic: 5x^7 − 9x^6 − 8x^5 + 4x^4 + 21x^3 − 8x^2 − 17x + 7 = 0', value:'x = 0.4189910775023; |P(input)| ≈ 4.018*10^-12', err:3.04e-13, degree:7, height:21n},
  {candidate:'irreducible algebraic: x^9 + 4x^8 − 10x^7 − 7x^6 − 6x^5 + 2x^4 + 3x^3 + x^2 + 4x − 2 = 0', value:'x = 0.4189910775034; |P(input)| ≈ 7.579*10^-12', err:1.45e-12, degree:9, height:10n},
  {candidate:'L-rational #1: x = L(f,1)', value:'L(f,1) ≈ 0.41899107750220406755140953460657586030; x/L(f,1) ≈ 1; relative residual 0.000000000000487', err:4.87e-13, lfuncCategory:'rational', modForm:{code:'9.4.1',level:9,weight:4,index:1}},
  {candidate:'log|c| linear relation: x ≈ e^(9/8) * 2^(-1/2) * 3^(-3/8) * 5^(-3/8) * exp(π*(-3/8)) * π^(1/8) * log(π)^(3/4) * log(2)^(-1/2) * log(3)^(5/4)', value:'x = 0.418991077502; product = 0.4189910775021; terms 9; height 10', err:1.7e-13, terms:9, height:10n}
];
const sorted=sandbox.confidenceSortedRows(rows, settings);
const labels=sorted.map(r => r.lfuncCategory ? 'L' : (/^log\|c\|/.test(r.candidate)?'log':(/algebraic/.test(r.candidate)?'alg':'other')));
const firstL=labels.indexOf('L');
const firstLog=labels.indexOf('log');
const firstAlg=labels.indexOf('alg');
const secondAlg=labels.indexOf('alg', firstAlg+1);
if(!(firstL >= 0 && firstLog >= 0 && firstAlg >= 0 && secondAlg >= 0)) throw new Error('Expected L/log/algebraic rows. Got '+labels.join(','));
if(!(firstL < secondAlg && firstLog < secondAlg)) throw new Error('First L/log module heads must appear before second algebraic row. Got '+labels.join(','));
if(!(firstL < firstAlg)) throw new Error('Short L(f,1) head should precede longer algebraic head. Got '+labels.join(','));
const smallHeightScore=sandbox.resultLengthFirstScore({candidate:'irreducible algebraic: x − 9 = 0', degree:1, height:9n});
if(!(smallHeightScore > 0 && smallHeightScore < 80)) throw new Error('Small BigInt height log10 score regression: '+smallHeightScore);
if(!fs.readFileSync('ries.html','utf8').includes('RIES <em>v10.6.1</em>')) throw new Error('ries.html version should be v10.6.1.');
console.log('PASS RIES v10.6 confidence final module sorting test');
