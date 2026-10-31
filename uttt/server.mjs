// Small static server for browsers that require an HTTP origin for modules/Workers.
import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {resolve,extname,sep} from 'node:path';
const root=fileURLToPath(new URL('.',import.meta.url));
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.wasm':'application/wasm','.json':'application/json; charset=utf-8'};
const port=Number(process.env.PORT||8080);
createServer(async(req,res)=>{
 try{
  const path=new URL(req.url,'http://localhost').pathname;
  const file=resolve(root,'.'+(path==='/'?'/index.html':decodeURIComponent(path)));
  if(file!==resolve(root,'index.html')&&!file.startsWith(root.endsWith(sep)?root:root+sep))throw new Error('bad path');
  const bytes=await readFile(file);
  res.writeHead(200,{'Content-Type':types[extname(file)]||'application/octet-stream','Cache-Control':'no-store'});res.end(bytes);
 }catch(e){res.writeHead(404,{'Content-Type':'text/plain'});res.end('404');}
}).listen(port,'0.0.0.0',()=>console.log(`http://localhost:${port}`));
