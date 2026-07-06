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
function nearness(s,t){const e1=angleError(s.th1,t.angles[0]),e2=angleError(s.th2,t.angles[1]);return {angleNorm:Math.hypot(e1,e2),speedNorm:Math.hypot(s.om1,s.om2)};}
function kineticProxy(s,p){return Math.max(0,totalEnergy(s,p)-energyAtTarget(TARGETS[3],p));}
function around(r,p,id,asp,osp,xsp=.5,vsp=.65){const b=supportBounds(p),t=TARGETS[id];return {mode:`low-state${id}`,state:{x:uni(r,b.center-xsp*b.half,b.center+xsp*b.half),vx:uni(r,-vsp,vsp),th1:wrap(t.angles[0]+uni(r,-asp,asp)),th2:wrap(t.angles[1]+uni(r,-asp,asp)),om1:uni(r,-osp,osp),om2:uni(r,-osp,osp)}};}
function scenario(r,p,mix){const b=supportBounds(p),q=r();
 if(mix==='near') return {mode:'near-state3',state:{x:uni(r,b.center-.58*b.half,b.center+.58*b.half),vx:uni(r,-1,1),th1:wrap(Math.PI+uni(r,-.82,.82)),th2:wrap(Math.PI+uni(r,-.82,.82)),om1:uni(r,-3,3),om2:uni(r,-3,3)}};
 if(mix==='legacy'){
  if(q<.28)return {mode:'legacy-broad',state:{x:uni(r,b.center-.72*b.half,b.center+.72*b.half),vx:uni(r,-1.45,1.45),th1:uni(r,-Math.PI,Math.PI),th2:uni(r,-Math.PI,Math.PI),om1:uni(r,-3.1,3.1),om2:uni(r,-3.1,3.1)}};
  if(q<.54)return scenario(r,p,'near');
  if(q<.76)return around(r,p,0,.65,2.2,.58,1.25);
  const side=r()<.5?-1:1;return {mode:'legacy-edge',state:{x:b.center+side*uni(r,.62*b.half,.91*b.half),vx:r()<.7?side*uni(r,.15,1.25):uni(r,-.85,.85),th1:r()<.55?wrap(Math.PI+uni(r,-.95,.95)):uni(r,-Math.PI,Math.PI),th2:r()<.55?wrap(Math.PI+uni(r,-.95,.95)):uni(r,-Math.PI,Math.PI),om1:uni(r,-2.9,2.9),om2:uni(r,-2.9,2.9)}};
 }
 if(q<.22)return around(r,p,0,.42,.85,.45,.5);
 if(q<.38)return around(r,p,1,.40,.90,.50,.6);
 if(q<.54)return around(r,p,2,.40,.90,.50,.6);
 if(q<.76)return {mode:'mid-random',state:{x:uni(r,b.center-.62*b.half,b.center+.62*b.half),vx:uni(r,-1.15,1.15),th1:uni(r,-Math.PI,Math.PI),th2:uni(r,-Math.PI,Math.PI),om1:uni(r,-2.25,2.25),om2:uni(r,-2.25,2.25)}};
 if(q<.90)return scenario(r,p,'near');
 return around(r,p,[0,1,2][Math.floor(r()*3)],.72,1.65,.70,1.1);
}
function runEpisode(r,seconds=15,dt=1/360,mix='v7'){const p=makeParams(r),sc=scenario(r,p,mix);let s=sc.state;const c=new PendulumController(p);c.setTarget(3,s);const tar=TARGETS[3],rail=supportBounds(p);let t=0,stableDwell=0,captureDwell=0,firstCapture=null,stableTime=null,bestAngle=Infinity,bestSpeed=Infinity,maxEdge=0,edgeRisk=false;for(let i=0;i<Math.floor(seconds/dt);i++){const n=nearness(s,tar);bestAngle=Math.min(bestAngle,n.angleNorm);bestSpeed=Math.min(bestSpeed,n.speedNorm);const edge=Math.abs(s.x-rail.center)/rail.half;maxEdge=Math.max(maxEdge,edge);if(edge>.965&&(s.x-rail.center)*s.vx>0)edgeRisk=true;const eScale=Math.max(1,p.g*((p.m1+p.m2)*p.l1+p.m2*p.l2));const stable=n.angleNorm<.34&&n.speedNorm<1.2&&edge<.90&&kineticProxy(s,p)/eScale<.16;const capture=n.angleNorm<.72&&n.speedNorm<3.6&&edge<.92;if(capture){captureDwell+=dt;if(captureDwell>.12&&firstCapture===null)firstCapture=t;}else captureDwell=Math.max(0,captureDwell-2*dt);if(stable){stableDwell+=dt;if(stableDwell>.36){stableTime=t;break;}}else stableDwell=Math.max(0,stableDwell-3*dt);const a=c.update(s,p,dt,t);s=stepRK4(s,a,p,t,dt,true);t+=dt;if(!Number.isFinite(s.x+s.vx+s.th1+s.th2+s.om1+s.om2))break;}let event=stableTime!==null?'stable':(firstCapture!==null?'capture-only':'timeout');if(edgeRisk)event=stableTime!==null?'stable-edge-risk':'edge-risk';return {mode:sc.mode,success:stableTime!==null,captured:firstCapture!==null||stableTime!==null,stableTime:stableTime??seconds,firstCapture:firstCapture??seconds,bestAngle,bestSpeed,maxEdge,event};}
function summarize(res){const n=res.length||1,s=res.filter(x=>x.success),cap=res.filter(x=>x.captured),mean=a=>a.reduce((p,c)=>p+c,0)/Math.max(1,a.length),events={},groups={};for(const x of res){events[x.event]=(events[x.event]||0)+1;(groups[x.mode]??=[]).push(x);}const byMode={};for(const [k,a] of Object.entries(groups)){const ss=a.filter(x=>x.success),cc=a.filter(x=>x.captured);byMode[k]={n:a.length,successRate:+(ss.length/a.length).toFixed(3),captureRate:+(cc.length/a.length).toFixed(3),meanStableTime:+mean(ss.map(x=>x.stableTime)).toFixed(3),meanFirstCaptureTime:+mean(cc.map(x=>x.firstCapture)).toFixed(3),edgeRiskRate:+(a.filter(x=>x.event.includes('edge-risk')).length/a.length).toFixed(3)};}return {episodes:res.length,successRate:+(s.length/n).toFixed(4),captureRate:+(cap.length/n).toFixed(4),meanStableTime:+mean(s.map(x=>x.stableTime)).toFixed(3),meanFirstCaptureTime:+mean(cap.map(x=>x.firstCapture)).toFixed(3),meanBestAngle:+mean(res.map(x=>x.bestAngle)).toFixed(3),edgeRiskRate:+(res.filter(x=>x.event.includes('edge-risk')).length/n).toFixed(4),events,byMode};}

const args=Object.fromEntries(process.argv.slice(2).map((x,i,a)=>x.startsWith('--')?[x.slice(2),a[i+1]&&!a[i+1].startsWith('--')?a[i+1]:true]:[String(i),x]));
adaptiveAI.importDatabase(JSON.parse(fs.readFileSync(args.db||'ai_data/training_db.json','utf-8')),'v8-multi-eval');try{await wasmRollout.ready;}catch{}
const seeds=String(args.seeds||'70421,81131,424242,110,221,13579').split(',').map(x=>Number(x.trim())).filter(Number.isFinite);
const episodes=Number(args.episodes||10); const seconds=Number(args.seconds||15); const mix=String(args.mix||'v7');
const all=[]; const perSeed=[];
for(const seed of seeds){const r=rngMulberry(seed); const res=[]; for(let i=0;i<episodes;i++)res.push(runEpisode(r,seconds,1/360,mix)); const sum=summarize(res); perSeed.push({seed,...sum}); all.push(...res);}
console.log(JSON.stringify({seeds,episodesPerSeed:episodes,total:summarize(all),perSeed},null,2));
