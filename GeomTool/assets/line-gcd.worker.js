/* v8 - exact projective common-coefficient normalization, isolated and bounded. */
'use strict';
importScripts('rational-cas.js','nerdamer-sparse-v2.js');
let symbolKey=null;
self.onmessage=async e=>{
 const {id,payload}=e.data;
 try{
  const next=payload.names.join('\0');
  if(symbolKey!==next){self.GeomRationalCAS.reset();payload.names.forEach(n=>self.GeomRationalCAS.symbol(n));symbolKey=next;}
  const Q=self.GeomRationalCAS,alias=name=>payload.aliases[name]||name;
  const normalized=Q.lineCommonCoefficients(payload.coefficients,alias);
  if(!normalized){self.postMessage({id,unchanged:true});return;}
  const coefficients={};
  for(const [i,k] of ['a','b','c'].entries())coefficients[k]=Q.string(normalized[i],alias);
  self.postMessage({id,coefficients});
  // Improve formatting, but only by exact factoring of normalized values.
  for(const [i,k] of ['a','b','c'].entries()){
   coefficients[k]=await Q.factor(normalized[i],alias);
   self.postMessage({id,coefficients:{...coefficients}});
  }
  self.postMessage({id,complete:true,coefficients:{...coefficients}});
 }catch(error){self.postMessage({id,error:String(error?.message||error)});}
};
