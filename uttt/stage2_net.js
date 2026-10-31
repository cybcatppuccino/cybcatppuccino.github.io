// Full original Stage2 teacher (100000) with folded inference-time batch normalization.
// Weights are rounded only once to IEEE FP16: no low-capacity student/distillation.
// Built-in WASM SIMD accelerated inference; scalar WASM fallback; no CDN dependency.
const modelUrl = new URL('./stage2.fp16.gz',import.meta.url);
let singleton=null;
// The model sees the 9 mini-boards as one 9x9 grid. Precompute the
// board-major -> image-major layout once, not once per neural evaluation.
const SPATIAL=new Uint8Array(81);
for(let b=0;b<9;b++)for(let p=0;p<9;p++)
 SPATIAL[b*9+p]=(Math.floor(b/3)*3+Math.floor(p/3))*9+(b%3)*3+p%3;
export class Stage2Net {
 constructor(instance){
  this.e=instance.exports;
  this.input=new Float32Array(this.e.memory.buffer,this.e.stage2_input_ptr(),4*81);
  this.logits=new Float32Array(this.e.memory.buffer,this.e.stage2_logits_ptr(),81);
  this.mappedLogits=new Float32Array(81); // scratch; consume before next predict()
 }
 static async load(){
  if(!singleton)singleton=(async()=>{
   let wasm;
   const imports={env:{stage2_exp:Math.exp}};
   for(const file of ['stage2_simd.wasm','stage2_scalar.wasm']){
    try{
     const response=await fetch(new URL('./'+file,import.meta.url));
     if(!response.ok)throw Error(file+' HTTP '+response.status);
     const bytes=await response.arrayBuffer();
     wasm=(await WebAssembly.instantiate(bytes,imports)).instance;
     break;
    }catch(e){if(file==='stage2_scalar.wasm')throw e;}
   }
   const net=new Stage2Net(wasm);
   const response=await fetch(modelUrl);
   if(!response.ok)throw Error('Stage2 teacher weight HTTP '+response.status);
   if(typeof DecompressionStream!=='function')throw Error('浏览器不支持 DecompressionStream(gzip)');
   const source=response.body;
   const raw=await new Response(source.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
   const bytes=new Uint8Array(raw);
   const addr=net.e.stage2_packed_ptr();
   // Prevent loading a truncated or mismatched model into WASM memory.
   const expected=9988612;
   if(bytes.byteLength!==expected)throw Error('Invalid full Stage2 weight size: '+bytes.byteLength);
   new Uint8Array(net.e.memory.buffer,addr,expected).set(bytes);
   if(net.e.stage2_init()!==1)throw Error('Could not initialize full Stage2 network');
   return net;
  })().catch(e=>{singleton=null;throw e;});
  return singleton;
 }
 predict(game,legalMoves=game.legal()){
  const input=this.input;input.fill(0,0,162);
  const own=game.turn,sign=own===1?1:-1;
  input.fill(sign,162,243);input.fill(0,243,324);
  const cells=game.cells;
  for(let idx=0;idx<81;idx++){
   const c=cells[idx];if(c)input[(c===own?0:1)*81+SPATIAL[idx]]=1;
  }
  for(const move of legalMoves)input[243+SPATIAL[move]]=1;
  if(this.e.stage2_predict()!==1)throw Error('Stage2 inference failed');
  const mapped=this.mappedLogits,logits=this.logits;
  for(let idx=0;idx<81;idx++)mapped[idx]=logits[SPATIAL[idx]];
  return {value:this.e.stage2_value(),logits:mapped};
 }
}
