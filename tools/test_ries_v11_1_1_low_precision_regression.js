// v11.1.1 regression: the double precision cap must not change <16-digit behavior.
const fs = require('fs');
const vm = require('vm');
const {performance} = require('perf_hooks');
const Decimal = require('../assets/decimal.js');
const elems = new Map();
function fakeEl(id){ return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){},toggle(){}},addEventListener(){},setAttribute(){},appendChild(child){return child},prepend(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}}; }
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
ids.forEach(id=>elems.set(id,fakeEl(id)));
Object.assign(elems.get('digits'), {value:'0123456789'});
Object.assign(elems.get('restrictMode'), {value:'none'});
Object.assign(elems.get('maxAbs'), {value:'1e9'});
Object.assign(elems.get('level'), {value:'4'});
Object.assign(elems.get('shortEffort'), {value:'3'});
Object.assign(elems.get('limit'), {value:'12'});
Object.assign(elems.get('logHeight'), {value:'600'});
Object.assign(elems.get('logPrecision'), {value:''});
Object.assign(elems.get('logSlack'), {value:'2'});
Object.assign(elems.get('algHeight'), {value:'1000000000000'});
Object.assign(elems.get('algDegree'), {value:'10'});
Object.assign(elems.get('algPrecision'), {value:''});
Object.assign(elems.get('algResidualPower'), {value:'3'});
['doEq','doAlg','doLog'].forEach(id=>elems.get(id).checked=true);
elems.get('allowExternalFactorization').checked=false;
const sandbox={console, performance, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(performance.now()),0), cancelAnimationFrame(){}, MathJax:null, Decimal};
sandbox.window=sandbox; sandbox.navigator={clipboard:null, scheduling:{isInputPending:()=>false}};
sandbox.__logChecked=[];
sandbox.document={getElementById:id=>elems.get(id)||null, querySelectorAll:sel=> sel==='[data-logconst]:checked' ? sandbox.__logChecked.map(id=>({dataset:{logconst:id}})) : [], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
const P=sandbox.__RIES_PRECISION_TEST__;
const L=sandbox.__RIES_LOG_TEST__;
sandbox.__logChecked=L.logConstants.filter(c=>c.default).map(c=>c.id);
function settings(raw, level=4){
  elems.get('target').value=raw; elems.get('level').value=String(level);
  const parsed=sandbox.parseDecimalComplex(raw);
  return {raw, normalizedRaw:raw, parsedComplex:parsed, complexTarget:false, target:sandbox.rationalToNumber(parsed.re), limit:12, level};
}
function toSig(x,d){ return Number(x).toPrecision(d).replace(/(?:\.0+|(?<=\d)0+)$/,''); }
if(P.typedInputPrecisionDigits('1.2300')!==5) throw new Error('typed trailing zeroes must count as user precision');
if(P.typedInputPrecisionForDouble('1.2345678')!==8) throw new Error('8-digit input should remain 8 under double cap');
if(P.typedInputPrecisionForDouble('1.234567890123456789')!==15) throw new Error('18-digit input should cap at 15 for Number modules');
if(P.matchToleranceDigits(11,1,30)>10) throw new Error('match tolerance must not exceed typed precision');
for(const d of [6,8,10,12,15]){
  const raw=toSig(6*Math.PI**5,d);
  const st=settings(raw,4);
  st.normalizedRaw='1836118108711395/1000000000000';
  if(P.typedInputPrecision(st)!==d) throw new Error(`typed precision from raw failed for ${raw}`);
  if(P.typedInputPrecisionForDouble(st)!==d) throw new Error(`double cap changed sub-16 precision for ${raw}`);
  const rows=L.logRelationRows(st.target, st);
  if(!rows.some(r=>/2 \* 3 \* π\^\(5\)/.test(r.candidate))){
    throw new Error(`log|c| missed 6*pi^5 at ${d} significant digits (${raw})`);
  }
}
// Keep this test focused on sub-16 digit behavior in the default log relation path;
// heavier Continue-level optional-basis checks are covered by older version-specific tests.
console.log('PASS RIES v11.1.1 low-precision regression test');
process.exit(0);
