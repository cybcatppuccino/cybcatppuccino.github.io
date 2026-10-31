// WDL: only a genuinely proved/terminal draw ever has non-zero D.
// Ordinary evaluation is heuristic: stronger separation of X/O win rates.
const MATE=30000;
export function isMate(score){return Number.isFinite(score)&&Math.abs(score)>=MATE-128;}
export function fromStage2Value(valueX){
 if(!Number.isFinite(valueX))return null;
 const x=Math.max(0,Math.min(100,50+50*valueX));
 return {x,d:0,o:100-x};
}
export function wdlForX(score,turn,winner=0,solved=false){
 if(winner===3||solved&&score===0)return {x:0,d:100,o:0};
 if(winner===1)return {x:100,d:0,o:0};
 if(winner===2)return {x:0,d:0,o:100};
 if(!Number.isFinite(score))return null;
 const v=turn===1?score:-score;
 if(isMate(score)&&solved)return v>0?{x:100,d:0,o:0}:{x:0,d:0,o:100};
 if(isMate(score)&&!solved)return v>0?{x:97,d:0,o:3}:{x:3,d:0,o:97};
 // Previously a fabricated draw component diluted an advantage. Now its entire
 // amount is removed; 4x logit yields around 83%/17% for old ~66%/34% cases.
 const z=Math.max(-25,Math.min(25,4*v/448));
 const x=100/(1+Math.exp(-z));return {x,d:0,o:100-x};
}
export function scoreLabel(score,turn,winner=0,solved=false,proofKind=null){
 if(winner===3||solved&&score===0)return 'D';
 if(winner===1)return 'W+0';if(winner===2)return 'W-0';
 if(!Number.isFinite(score))return '—';
 const v=turn===1?score:-score;
 if(isMate(score)&&solved&&proofKind?.startsWith('WDL'))return v>0?'W+':'W-';
 if(isMate(score)&&solved)return 'W'+(v>0?'+':'-')+Math.max(1,MATE-Math.abs(score));
 return wdlForX(score,turn,winner,solved).x.toFixed(1)+'%';
}
