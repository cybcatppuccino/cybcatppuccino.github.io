/* DOM-free initialization smoke: detects immediate exceptions in app.js without browser policies. */
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const base=__dirname+'/../',ctxObj={};let timer=0;
const ctx=new Proxy({}, {get:(x,k)=>x[k]||function(){},set:(x,k,v)=>(x[k]=v,true)});
const controls={};
function el(key){if(!controls[key]){
 const obj={id:key,style:{},dataset:{},value:key==='#doc-name'?'未命名画板':'',innerHTML:'',textContent:'',checked:false,hidden:false,disabled:false,
 addEventListener(){},getBoundingClientRect(){return {width:730,height:620,top:0,left:0}},getContext(){return ctx},
 classList:{add(){},remove(){},toggle(){},contains(){return false}},click(){},focus(){},setPointerCapture(){},appendChild(){},remove(){},setAttribute(){}};
 controls[key]=obj;}return controls[key];}
const document={querySelector:el,querySelectorAll(){return []},body:el('body'),documentElement:{style:{setProperty(){}}},createElement:()=>el('created')};
const window={GeomKernel:require('../assets/kernel.js'),GeomPretty:require('../assets/pretty.js'),GeomToolGuides:require('../assets/tool-guides.js'),GeomRelationEngine:require('../assets/relations/detectors.js'),GeomRelationOverlay:require('../assets/relations/overlay.js'),addEventListener(){},devicePixelRatio:1,innerWidth:1440};
const localStorage={getItem(){return null},setItem(){}};
const globals={window,document,localStorage,console,Math,JSON,setTimeout(){return ++timer},clearTimeout(){},ResizeObserver:class{observe(){}},getComputedStyle(){return {getPropertyValue:()=>'#347ed3'}}};
vm.createContext(globals);
vm.runInContext(fs.readFileSync(base+'vendor/bs-relations.js','utf8'),globals);
vm.runInContext(fs.readFileSync(base+'assets/app.js','utf8'),globals);
assert(window.GeomTool,'public API absent');
assert.equal(window.GeomTool.kernel.list().length,6,'blank canvas must only contain six hidden reference objects');
assert(window.GeomTool.kernel.list().every(n=>n.style.system),'initial nodes must be internal references');
assert.equal(window.GeomTool.getProofPayload().computedFacts.length,0,'default graph contains no user constructions');
console.log('PASS — v2 static UI initialized blank, hidden references built, public adapter exposed.');
