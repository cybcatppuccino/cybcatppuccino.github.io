/* Small, deterministic arithmetic grammar for geometric parameters. No eval or Function. */
(function(root,factory){const api=factory();root.GeomExpression=api;if(typeof module==='object'&&module.exports)module.exports=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
function parse(source){
 const text=String(source).replace(/[×·]/g,'*').replace(/÷/g,'/').replace(/[−–]/g,'-').replace(/π/g,'pi').replace(/√\s*\(/g,'sqrt(').replace(/√\s*(\d+(?:\.\d+)?)/g,'sqrt($1)').replace(/\bsqrt\s*(\d+(?:\.\d+)?)\b/gi,'sqrt($1)').trim();
 if(!text||text.length>160)throw Error('公式为空或过长');
 const tokens=text.match(/(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z]+|[()+*\/^\-]/g)||[];
 if(tokens.join('').toLowerCase()!==text.replace(/\s+/g,'').toLowerCase())throw Error('公式含不支持的字符');
 let i=0;const peek=()=>tokens[i],eat=t=>peek()===t?(i++,true):false;
 const primary=()=>{
  if(eat('(')){const v=sum();if(!eat(')'))throw Error('括号不匹配');return v;}
  const t=peek();if(!t)throw Error('公式不完整');i++;
  if(/^\d|^\./.test(t))return Number(t);
  if(t.toLowerCase()==='pi')return Math.PI;
  if(t.toLowerCase()==='e')return Math.E;
  if(t.toLowerCase()==='sqrt'){if(!eat('('))throw Error('sqrt 必须带括号');const n=sum();if(!eat(')')||n<0)throw Error('sqrt 参数无效');return Math.sqrt(n);}
  throw Error('未知符号：'+t);
 };
 const power=()=>{const a=primary();return eat('^')?Math.pow(a,unary()):a;};
 const unary=()=>eat('+')?unary():eat('-')?-unary():power();
 const product=()=>{let a=unary();while(true){const t=peek(),implicit=t==='('||/^(?:[a-z]+|(?:\d|\.))/i.test(t||'');if(t!=='*'&&t!=='/'&&!implicit)break;const op=implicit?'*':tokens[i++],b=unary();a=op==='*'?a*b:a/b;}return a;};
 const sum=()=>{let a=product();while(peek()==='+'||peek()==='-'){const op=tokens[i++],b=product();a=op==='+'?a+b:a-b;}return a;};
 const result=sum();if(i!==tokens.length)throw Error('公式语法错误：'+tokens[i]);if(!Number.isFinite(result)||Math.abs(result)>1e12)throw Error('公式计算结果无效或过大');return result;
}
function convertAngle(source,from,to){
 if(!['deg','rad'].includes(from)||!['deg','rad'].includes(to))throw Error('未知角度单位');
 const n=parse(source);if(from===to)return String(source);
 if(from==='deg'){
  const ratio=n/180;
  for(let d=1;d<=96;d++){const k=Math.round(ratio*d);if(Math.abs(ratio-k/d)<=1e-11*Math.max(1,Math.abs(ratio))){
    if(k===0)return '0';const prefix=k===1?'π':k===-1?'-π':String(k)+'π';return d===1?prefix:prefix+'/'+d;
  }}return '('+source+')*π/180';
 }
 const deg=n*180/Math.PI;
 for(const scale of [1,2,4,10,100,1000]){const round=Math.round(deg*scale)/scale;if(Math.abs(deg-round)<1e-10*Math.max(1,Math.abs(deg)))return String(round);}
 return '('+source+')*180/π';
}
return {parse,convertAngle};
});
