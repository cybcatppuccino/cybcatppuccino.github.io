const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){},toggle(){}},addEventListener(){},setAttribute(){},appendChild(child){return child},prepend(){},insertAdjacentElement(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}};}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='10'; els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true; els.logHeight.value='400';
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null, scheduling:{isInputPending:()=>false}};
sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
function settingsFor(v, level=4){ els.target.value=String(v); els.level.value=String(level); return sandbox.readSettings(); }
const mob=sandbox.__RIES_MOBIUS_TEST__;
const constants=mob.mobiusConstants.map(c=>c.id);
if(constants.indexOf('pi2')<0) throw new Error('π² missing from Möbius constant queue');
if(constants.indexOf('catalan')<0) throw new Error('Catalan G missing from Möbius constant queue');
if(constants.indexOf('catalan') > 8) throw new Error('Catalan G should be kept in the early Möbius queue');
const rows=mob.mobiusRelationRows(settingsFor('1.0748330721566944',4));
const hit=rows.find(r=>/π\^2/.test(r.candidate) && /8·G/.test(r.candidate) && /\/16/.test(r.candidate));
if(!hit) throw new Error('missing generalized Möbius hit for (8G+π²)/16: '+rows.map(r=>r.candidate).join(' | '));
if(!/\\frac\{.*\\pi\^2.*8\\,G.*\}\{16\}/.test(hit.latex) && !/\\frac\{.*8\\,G.*\\pi\^2.*\}\{16\}/.test(hit.latex)) throw new Error('bad generalized Möbius latex: '+hit.latex);
if(!fs.readFileSync('ries.html','utf8').includes('RIES <em>v11.4.3</em>')) throw new Error('ries.html version should be v11.4.3.');
console.log('PASS RIES v11.2.4 generalized Möbius denominator-constant regression');
