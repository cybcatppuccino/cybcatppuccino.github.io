import fs from 'node:fs';
import { DEFAULT_PARAMS, TARGETS, angleError, stepRK4, supportBounds, totalEnergy, energyAtTarget } from './js/physics.js';
import { PendulumController } from './js/controller.js';
import { adaptiveAI } from './js/ai_learning.js';
import { wasmRollout } from './js/wasm_rollout.js';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const wrap=a=>{a=(a+Math.PI)%(2*Math.PI);if(a<0)a+=2*Math.PI;return a-Math.PI;};
function rngMulberry(seed){let t=seed>>>0;return()=>{t+=0x6D2B79F5;let r=Math.imul(t^(t>>>15),1|t);r^=r+Math.imul(r^(r>>>7),61|r);return((r^(r>>>14))>>>0)/4294967296;};}
const uni=(r,a,b)=>a+(b-a)*r(); const tri=(r,a,b)=>(uni(r,a,b)+uni(r,a,b))/2;
function makeParams(r){return {...DEFAULT_PARAMS,maxAcc:Math.round(tri(r,14.1,25.9)*2)/2,g:Math.round(tri(r,7.8,10.2)*100)/100,windAmp:Math.round(tri(r,0,0.13)*200)/200,friction:Math.round(tri(r,0,0.13)*200)/200};}
function around(r,p,id,asp,osp,xsp=.5,vsp=.65){const b=supportBounds(p),t=TARGETS[id];return {mode:`low-state${id}`,state:{x:uni(r,b.center-xsp*b.half,b.center+xsp*b.half),vx:uni(r,-vsp,vsp),th1:wrap(t.angles[0]+uni(r,-asp,asp)),th2:wrap(t.angles[1]+uni(r,-asp,asp)),om1:uni(r,-osp,osp),om2:uni(r,-osp,osp)}};}
function scenario(r,p,mix){const b=supportBounds(p),q=r();
 if(q<.22)return around(r,p,0,.42,.85,.45,.5);
 if(q<.38)return around(r,p,1,.40,.90,.50,.6);
 if(q<.54)return around(r,p,2,.40,.90,.50,.6);
 if(q<.76)return {mode:'mid-random',state:{x:uni(r,b.center-.62*b.half,b.center+.62*b.half),vx:uni(r,-1.15,1.15),th1:uni(r,-Math.PI,Math.PI),th2:uni(r,-Math.PI,Math.PI),om1:uni(r,-2.25,2.25),om2:uni(r,-2.25,2.25)}};
 if(q<.90)return {mode:'near-state3',state:{x:uni(r,b.center-.58*b.half,b.center+.58*b.half),vx:uni(r,-1,1),th1:wrap(Math.PI+uni(r,-.82,.82)),th2:wrap(Math.PI+uni(r,-.82,.82)),om1:uni(r,-3,3),om2:uni(r,-3,3)}};
 return around(r,p,[0,1,2][Math.floor(r()*3)],.72,1.65,.70,1.1);
}
function nearness(s,t){const e1=angleError(s.th1,t.angles[0]),e2=angleError(s.th2,t.angles[1]);return {angleNorm:Math.hypot(e1,e2),speedNorm:Math.hypot(s.om1,s.om2)}}
function kineticProxy(s,p){return Math.max(0,totalEnergy(s,p)-energyAtTarget(TARGETS[3],p));}
function run(sc,p,seconds=15,dt=1/360){let s={...sc.state};const c=new PendulumController(p);c.setTarget(3,s);const tar=TARGETS[3],rail=supportBounds(p);let t=0,stableDwell=0,captureDwell=0,firstCapture=null,stableTime=null,bestAngle=Infinity,bestSpeed=Infinity,maxEdge=0,edgeRisk=false;for(let i=0;i<Math.floor(seconds/dt);i++){const n=nearness(s,tar);bestAngle=Math.min(bestAngle,n.angleNorm);bestSpeed=Math.min(bestSpeed,n.speedNorm);const edge=Math.abs(s.x-rail.center)/rail.half;maxEdge=Math.max(maxEdge,edge);if(edge>.965&&(s.x-rail.center)*s.vx>0)edgeRisk=true;const eScale=Math.max(1,p.g*((p.m1+p.m2)*p.l1+p.m2*p.l2));const stable=n.angleNorm<.34&&n.speedNorm<1.2&&edge<.90&&kineticProxy(s,p)/eScale<.16;const capture=n.angleNorm<.72&&n.speedNorm<3.6&&edge<.92;if(capture){captureDwell+=dt;if(captureDwell>.12&&firstCapture===null)firstCapture=t;}else captureDwell=Math.max(0,captureDwell-2*dt);if(stable){stableDwell+=dt;if(stableDwell>.36){stableTime=t;break;}}else stableDwell=Math.max(0,stableDwell-3*dt);const a=c.update(s,p,dt,t);s=stepRK4(s,a,p,t,dt,true);t+=dt;}let event=stableTime!==null?'stable':(firstCapture!==null?'capture-only':'timeout');if(edgeRisk)event=stableTime!==null?'stable-edge-risk':'edge-risk';return {mode:sc.mode,event,success:stableTime!==null,captured:firstCapture!==null||stableTime!==null,stableTime:stableTime??seconds,firstCapture:firstCapture??seconds,bestAngle,bestSpeed,maxEdge};}
const args=Object.fromEntries(process.argv.slice(2).map((x,i,a)=>x.startsWith('--')?[x.slice(2),a[i+1]&&!a[i+1].startsWith('--')?a[i+1]:true]:[String(i),x]));
adaptiveAI.importDatabase(JSON.parse(fs.readFileSync(args.db||'ai_data/training_db.json','utf-8')),'detail');try{await wasmRollout.ready;}catch{}
const r=rngMulberry(Number(args.seed||81131));const out=[];for(let i=0;i<Number(args.episodes||20);i++){const p=makeParams(r);const sc=scenario(r,p,'v7');const res=run(sc,p);out.push({i,params:{maxAcc:p.maxAcc,g:p.g,windAmp:p.windAmp,friction:p.friction},init:sc.state,...res});}
console.log(JSON.stringify(out,null,2));
