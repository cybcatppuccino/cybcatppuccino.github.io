/* Nerdamer 2.0 exact sparse polynomial factor/GCD. Isolated from caller so
 * a costly GCD cannot use the time reserved for factorization. */
'use strict';
importScripts('rational-cas.js','nerdamer-sparse-v2.js');
let symbolKey=null;
self.onmessage=e=>{
 const {id,task,names,poly,other}=e.data;
 try{
  const next=names.join('\0');
  if(symbolKey!==next){self.GeomRationalCAS.reset();names.forEach(n=>self.GeomRationalCAS.symbol(n));symbolKey=next;}
  const q=self.GeomRationalCAS;
  const answer=task==='gcd'?q.nerdamerCommonFactor(poly,other):q.nerdamerFactorCandidates(poly);
  self.postMessage({id,answer});
 }catch(error){self.postMessage({id,error:String(error?.message||error)});}
};
