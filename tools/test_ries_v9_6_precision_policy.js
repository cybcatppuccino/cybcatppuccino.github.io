// RIES v9.6 decimal precision policy smoke/stress test.
const fs = require('fs');
const vm = require('vm');
const {performance} = require('perf_hooks');
const Decimal = require('../assets/decimal.js');
const html = fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v9.6</em>')) throw new Error('ries.html was not updated to v9.6.');
for(const forbidden of ['id="tolerance"','id="algPrecision"','id="algResidualPower"','id="logPrecision"','id="logSlack"']){ if(html.includes(forbidden)) throw new Error('Removed precision parameter still visible: '+forbidden); }
if(html.includes('Please reload this v9.2 build')) throw new Error('old v9.2 reload guard text still present.');
const elems = new Map();
function fakeEl(id){ return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,
  classList:{contains(){return false},add(){},remove(){},toggle(){}}, addEventListener(){}, setAttribute(){}, appendChild(child){return child}, prepend(){},
  querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}, getAttribute(){return ''}}; }
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','algHeight','algDegree','defaultLogBasis','extraLogBasis'];
ids.forEach(id=>elems.set(id,fakeEl(id)));
Object.assign(elems.get('digits'), {value:'0123456789'});
Object.assign(elems.get('restrictMode'), {value:'none'});
Object.assign(elems.get('maxAbs'), {value:'1e9'});
Object.assign(elems.get('level'), {value:'4'});
Object.assign(elems.get('shortEffort'), {value:'3'});
Object.assign(elems.get('limit'), {value:'12'});
Object.assign(elems.get('logHeight'), {value:'600'});
Object.assign(elems.get('algHeight'), {value:'1000000000000'});
Object.assign(elems.get('algDegree'), {value:'10'});
['doEq','doAlg','doLog'].forEach(id=>elems.get(id).checked=true);
elems.get('allowExternalFactorization').checked=false;
const sandbox={console, performance, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(performance.now()),0), cancelAnimationFrame(){}, MathJax:null, Decimal};
sandbox.window=sandbox; sandbox.navigator={clipboard:null, scheduling:{isInputPending:()=>false}};
sandbox.__logChecked=[];
sandbox.document={
  getElementById:id=>elems.get(id)||null,
  querySelectorAll:sel=> sel==='[data-logconst]:checked' ? sandbox.__logChecked.map(id=>({dataset:{logconst:id}})) : [],
  querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
const test=sandbox.__RIES_PRECISION_TEST__;
const logTest=sandbox.__RIES_LOG_TEST__;
if(!test || !logTest) throw new Error('v9.6 precision/log test exports missing.');
sandbox.__logChecked=logTest.logConstants.filter(c=>c.default).map(c=>c.id);
function settings(raw, level=4){
  elems.get('target').value=raw; elems.get('level').value=String(level);
  const parsed=sandbox.parseDecimalComplex(raw);
  return {raw, normalizedRaw: parsed ? sandbox.canonicalComplexString?.(parsed) || raw : raw, parsedComplex:parsed, complexTarget:false, target:sandbox.rationalToNumber(parsed.re), limit:12, level};
}
function toSig(x,d){ return Number(x).toPrecision(d).replace(/(?:\.0+|(?<=\d)0+)$/,''); }
function truncSigString(str,d){ const s=String(str); const neg=s.startsWith('-'); const body=neg?s.slice(1):s; const [a,b='']=body.split('.'); const intSig=a.replace(/^0+/,'').length; if(intSig>=d) return (neg?'-':'')+a.slice(0,d); return (neg?'-':'')+a+'.'+b.slice(0,Math.max(0,d-intSig)); }
if(test.typedInputPrecisionDigits('1.2300')!==5) throw new Error('typed trailing zeroes must count as user precision.');
if(test.typedInputPrecisionDigits('0.0012300')!==5) throw new Error('leading zeroes should not count, typed trailing significant zeroes should.');
if(test.matchToleranceDigits(11,1,30)>10) throw new Error('match tolerance must not exceed typed precision.');
// Default log|c| should use raw typed precision, not the canonical rationalized normalizedRaw.
for(const d of [6,8,10,12,15]){
  const raw=toSig(6*Math.PI**5,d);
  const st=settings(raw,4);
  st.normalizedRaw='1836118108711395/1000000000000'; // would be falsely high/odd precision if used.
  if(test.typedInputPrecision(st)!==d) throw new Error(`typed precision from raw failed for ${raw}`);
  const rows=logTest.logRelationRows(st.target, st);
  if(!rows.some(r=>/2 \* 3 \* π\^\(5\)/.test(r.candidate))){
    throw new Error(`log|c| missed 6*pi^5 at ${d} significant digits (${raw})`);
  }
}
// First Continue should find Gamma(1/3)/log(pi) even at deliberately low typed precision.
const gamma13=2.678938534707747633655692940974677644128689377957302;
for(const d of [7,8,10,12,15]){
  const raw=toSig(gamma13/Math.log(Math.PI), d);
  const rows=logTest.logRelationRows(Number(raw), settings(raw,5));
  if(!rows.some(r=>/Γ\(1\/3\)/.test(r.candidate) && /log\(π\)\^\(-1\)/.test(r.candidate))){
    throw new Error(`log|c| first-continue missed Gamma(1/3)/log(pi) at ${d} digits (${raw})`);
  }
}
(async()=>{
  const entries=sandbox.lfuncEntries ? sandbox.lfuncEntries() : null;
  const lval=(entries || []).find(e=>String(e.value).startsWith('2.298486058160745'));
  if(!lval) throw new Error('expected L-function fixture not found');
  for(const d of [6,8,9,10,12,15,18,20]){
    const raw=truncSigString(lval.value, d);
    const rows=await test.lfuncRowsAsync(settings(raw,4),0,()=>{});
    if(!rows.some(r=>/^L-rational/.test(r.candidate) && /x = L\(f,1\)/.test(r.candidate))){
      throw new Error(`simple L-function positive missed under typed precision at ${d} sig digits (${raw})`);
    }
  }
  for(const d of [6,8,10,12,15,18,20]){
    const raw=truncSigString('3.625609908221908311930685155867672002995167682880065467433377',d);
    const rows=test.specialDecimalConstantRows(settings(raw,5),1);
    if(!rows.some(r=>/Γ\(1\/4\)/.test(r.candidate))) throw new Error(`Gamma(1/4) missed at ${d} typed digits (${raw})`);
  }
  console.log('PASS RIES v9.6 typed precision policy/log/L-function smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
