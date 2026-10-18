/* A hung Nerdamer factor() call is interrupted by terminating this worker. */
'use strict';
importScripts('nerdamer-factor.min.js');
self.onmessage=e=>{
 const {id,expr,ms}=e.data;
 try{nerdamer.getCore().Settings.TIMEOUT=ms;
  const output=nerdamer.factor(expr).toString();
  self.postMessage({id,output});
 }catch(error){self.postMessage({id,output:''});}
};
