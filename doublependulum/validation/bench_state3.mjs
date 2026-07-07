import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = path.resolve(process.argv[2] || '.');
const controllerFile = process.argv[3] || 'js/controller.js';
const mode = process.argv[4] || 'default';
const { PendulumController } = await import(pathToFileURL(path.join(root, controllerFile)).href + `?v=${Date.now()}`);
const physics = await import(pathToFileURL(path.join(root, 'js/physics.js')).href);
const { adaptiveAI } = await import(pathToFileURL(path.join(root, 'js/ai_learning.js')).href);
const { wasmRollout } = await import(pathToFileURL(path.join(root, 'js/wasm_rollout.js')).href);
try {
  const db = JSON.parse(fs.readFileSync(path.join(root, 'ai_data/training_db.json'), 'utf8'));
  adaptiveAI.importDatabase(db, 'bench-static');
} catch (e) {}
await wasmRollout.ready;

const {
  DEFAULT_PARAMS, TARGETS, makeInitialState, stepRK4Into, angleError,
  totalEnergy, energyAtTarget, supportEdgeRatio
} = physics;

function nearTarget(s) {
  const e1 = angleError(s.th1, Math.PI);
  const e2 = angleError(s.th2, Math.PI);
  return { angle: Math.hypot(e1,e2), speed: Math.hypot(s.om1,s.om2), e1,e2 };
}

function isPractical(n) {
  return n.angle < 0.115 && n.speed < 0.42;
}
function isLooseCapture(n) {
  return n.angle < 0.66 && n.speed < 5.6;
}

function runCase(params, options={}) {
  let state = makeInitialState();
  let next = makeInitialState();
  const controller = new PendulumController(params);
  let t = 0;
  const dt = 1/360;
  const preDelay = options.preDelay || 0;
  // Run actual page state-0 controller before switching, when requested.
  if (preDelay > 0) {
    controller.setTarget(0, state);
    while (t < preDelay - 1e-12) {
      const a = controller.update(state, params, dt, t);
      stepRK4Into(state, a, params, t, dt, true, next, {}, {}, {}, {}, {}, {}, {});
      [state,next]=[next,state]; t += dt;
    }
  }
  controller.setTarget(3, state);
  const switchT = t;
  let stableHold=0, practicalTime=null, firstCapture=null, firstLocal=null;
  let rounds=0, wasCapture=false, wasLocal=false, maxEdge=0, minAngle=99, maxSpeedNear=0;
  let retries=0, prevRetry=0;
  let firstRoundExit=null;
  let captureEntries=[];
  let samples=[];
  const maxT = options.maxT || 45;
  const scratch1={},scratch2={},scratch3={},k1={},k2={},k3={},k4={};
  while (t-switchT < maxT) {
    const beforeLocal = controller.localCaptureActive;
    const beforeRetry = controller.retryTimer;
    const a = controller.update(state, params, dt, t);
    const n = nearTarget(state);
    const loose = isLooseCapture(n);
    if (loose && !wasCapture) {
      rounds++;
      captureEntries.push({t:t-switchT, angle:n.angle, speed:n.speed, x:state.x, vx:state.vx,
        dE:(totalEnergy(state,params)-energyAtTarget(TARGETS[3],params))/Math.max(1,params.g*3)});
      if (firstCapture===null) firstCapture=t-switchT;
    }
    if (controller.localCaptureActive && !wasLocal && firstLocal===null) firstLocal=t-switchT;
    if (!loose && wasCapture && firstRoundExit===null) firstRoundExit=t-switchT;
    if (controller.retryTimer > 0 && prevRetry <= 0) retries++;
    prevRetry = controller.retryTimer;
    wasCapture=loose; wasLocal=controller.localCaptureActive;
    maxEdge=Math.max(maxEdge,supportEdgeRatio(state,params));
    minAngle=Math.min(minAngle,n.angle);
    if (n.angle < 0.75) maxSpeedNear=Math.max(maxSpeedNear,n.speed);
    if (isPractical(n)) stableHold += dt; else stableHold=0;
    if (stableHold >= 0.65) { practicalTime=t-switchT-0.65; break; }
    if (options.trace && Math.floor((t-switchT)*100+1e-8)%5===0 && samples.length<2000) {
      samples.push({t:t-switchT,a,angle:n.angle,speed:n.speed,x:state.x,vx:state.vx,local:controller.localCaptureActive,retry:controller.retryTimer});
    }
    stepRK4Into(state, a, params, t, dt, true, next, scratch1,scratch2,scratch3,k1,k2,k3,k4);
    [state,next]=[next,state]; t+=dt;
  }
  const finalN=nearTarget(state);
  return {
    ok: practicalTime!==null, time: practicalTime ?? maxT, rounds, retries,
    firstCapture, firstLocal, firstRoundExit, maxEdge, minAngle, maxSpeedNear,
    finalAngle:finalN.angle, finalSpeed:finalN.speed, captures:captureEntries,
    samples: options.trace?samples:undefined
  };
}

function paramGrid(levels=3) {
  const vals = levels===3 ? [0,0.5,1] : Array.from({length:levels},(_,i)=>i/(levels-1));
  const ranges={maxAcc:[14.1,25.9],g:[7.8,10.2],windAmp:[0,0.13],friction:[0,0.13]};
  const arr=[];
  for (const ia of vals) for (const ig of vals) for (const iw of vals) for (const iff of vals) {
    arr.push({...DEFAULT_PARAMS,
      maxAcc:ranges.maxAcc[0]+ia*(ranges.maxAcc[1]-ranges.maxAcc[0]),
      g:ranges.g[0]+ig*(ranges.g[1]-ranges.g[0]),
      windAmp:ranges.windAmp[0]+iw*(ranges.windAmp[1]-ranges.windAmp[0]),
      friction:ranges.friction[0]+iff*(ranges.friction[1]-ranges.friction[0])});
  }
  return arr;
}
function rand(seed) { let x=seed>>>0; return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296};}
function randomCases(n,seed=12345){const r=rand(seed),a=[];for(let i=0;i<n;i++)a.push({...DEFAULT_PARAMS,maxAcc:14.1+11.8*r(),g:7.8+2.4*r(),windAmp:0.13*r(),friction:0.13*r()});return a;}
function summarize(results) {
  const ok=results.filter(x=>x.result.ok); const one=results.filter(x=>x.result.ok&&x.result.rounds<=1); const two=results.filter(x=>x.result.ok&&x.result.rounds<=2);
  const times=ok.map(x=>x.result.time).sort((a,b)=>a-b);
  const q=p=>times.length?times[Math.min(times.length-1,Math.floor(p*(times.length-1)))]:null;
  return {n:results.length,success:ok.length,successRate:ok.length/results.length,oneRound:one.length,oneRoundRate:one.length/results.length,twoRound:two.length,twoRoundRate:two.length/results.length,meanTime:times.length?times.reduce((a,b)=>a+b,0)/times.length:null,p50:q(.5),p90:q(.9),p95:q(.95),max:q(1),meanRounds:ok.length?ok.reduce((s,x)=>s+x.result.rounds,0)/ok.length:null,totalRetries:results.reduce((s,x)=>s+x.result.retries,0)};
}

let cases=[];
if (mode==='default') cases=[{...DEFAULT_PARAMS}];
else if (mode==='grid3') cases=paramGrid(3);
else if (mode==='randomfail') {
  const all=randomCases(20); cases=[3,8,9,17,19].map(i=>all[i]);
}
else if (mode.startsWith('random')) cases=randomCases(Number(mode.replace('random',''))||100);
else if (mode==='representative') cases=[
 {...DEFAULT_PARAMS},
 {...DEFAULT_PARAMS,maxAcc:14.1,g:7.8,windAmp:0,friction:0},
 {...DEFAULT_PARAMS,maxAcc:14.1,g:10.2,windAmp:0.13,friction:0.13},
 {...DEFAULT_PARAMS,maxAcc:25.9,g:7.8,windAmp:0.13,friction:0},
 {...DEFAULT_PARAMS,maxAcc:25.9,g:10.2,windAmp:0,friction:0.13},
 {...DEFAULT_PARAMS,maxAcc:17,g:9.8,windAmp:0.09,friction:0.08},
 {...DEFAULT_PARAMS,maxAcc:23,g:8.2,windAmp:0.08,friction:0.02}
];
else if (mode==='hard') cases=[
 {...DEFAULT_PARAMS,maxAcc:14.1,g:10.2,windAmp:0.13,friction:0.13},
 {...DEFAULT_PARAMS,maxAcc:17,g:9.8,windAmp:0.09,friction:0.08}
];
else if (mode==='caseenv') {
  const spec = JSON.parse(process.env.CASE_PARAMS || '{}');
  cases=[{...DEFAULT_PARAMS,...spec}];
}
else if (mode.startsWith('delay')) {
  const delays=mode==='delayset'?[0,0.35,0.7,1.05,1.4,1.75,2.1,2.45,2.8,3.15]:[Number(mode.slice(5))||0];
  const results=delays.map(d=>({params:{...DEFAULT_PARAMS},delay:d,result:runCase({...DEFAULT_PARAMS},{preDelay:d,maxT:45})}));
  console.log(JSON.stringify({summary:summarize(results),results},null,2)); process.exit(0);
} else cases=[{...DEFAULT_PARAMS}];

const trace = process.argv.includes('--trace');
const benchMaxT = Number(process.env.MAXT) || 45;
const results=cases.map((params,i)=>({i,params,result:runCase(params,{maxT:benchMaxT,trace})}));
console.log(JSON.stringify({controllerFile,mode,summary:summarize(results),worst:[...results].sort((a,b)=>(b.result.ok?b.result.time+10*b.result.rounds:999)-(a.result.ok?a.result.time+10*a.result.rounds:999)).slice(0,12),results:mode==='default'||trace?results:undefined},null,2));
