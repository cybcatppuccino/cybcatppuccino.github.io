import {Game,cellName} from './game.js';
import {scoreLabel,wdlForX,fromStage2Value} from './score.js';
import {replayMoves} from './persistence.js';
const $=id=>document.getElementById(id);
const KEY='UTTT_APEX_V6_GAME';
const game=new Game();
let mode='ha',human=1,paused=false,worker=null,job=0;
let latest=null,thinking=false,moveTimer=0,configTimer=0;
let startClock=0,iteration=0,iterationBudget=0,liveNodes=0,searchTime=0;
const modeBtns=[...document.querySelectorAll('[data-mode]')];
const settings=()=>({student:$('studentSwitch').checked,time:+$('timeRange').value,backend:'wasm'});
function aiTurn(){return !game.winner&&!paused&&(mode==='aa'||mode==='ha'&&game.turn!==human);}
function save(){try{localStorage.setItem(KEY,JSON.stringify({v:9,moves:game.moves,mode,human,paused,time:+$('timeRange').value,student:$('studentSwitch').checked,heat:$('heatSwitch').checked}));}catch{}}
function restore(){try{
 const raw=localStorage.getItem(KEY);if(!raw)return;
 const s=JSON.parse(raw);if(!s||!Array.isArray(s.moves)||s.moves.length>81)return;
 const g=replayMoves(s.moves);if(!g)return;
 game.reset();for(const m of g.moves)game.play(m);
 if(['hh','ha','aa'].includes(s.mode))mode=s.mode;
 human=s.human===2?2:1;paused=!!s.paused;
 if(Number.isFinite(s.time))$('timeRange').value=String(Math.min(20000,Math.max(500,s.time)));
 // Upgrade older saved games to the new default-on setting once; v8 choices persist.
 if(s.v>=8&&typeof s.student==='boolean')$('studentSwitch').checked=s.student;
 if(typeof s.heat==='boolean')$('heatSwitch').checked=s.heat;
 }catch(e){console.warn('Save restore:',e);}}
function numberText(n=0){if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'k';return String(n|0);}
function formatTime(ms){return(ms/1000).toFixed(1)+' s';}
function resultText(score,solved=false,valueX=null,proofKind=null){
 if(game.winner||Number.isFinite(score))return scoreLabel(score,game.turn,game.winner,solved,proofKind);
 const p=fromStage2Value(valueX);return p?p.x.toFixed(1)+'%':'—';
}
function colorOfLabel(text){return text==='D'?'draw':(text.startsWith('W+')||text.startsWith('M+'))?'mate-x':(text.startsWith('W-')||text.startsWith('M-'))?'mate-o':'';}
function decorateValue(el,label){el.classList.remove('mate-x','mate-o','draw');const cl=colorOfLabel(label);if(cl)el.classList.add(cl);}
function wdl(score,solved=false,valueX=null){
 if(game.winner||Number.isFinite(score))return wdlForX(score,game.turn,game.winner,solved);
 return fromStage2Value(valueX);
}
// The ordinary and Stage2 probability displays never fabricate a draw.
function expectedScoreX(score,solved=false,valueX=null,proofKind=null){
 if(game.winner||Number.isFinite(score))return resultText(score,solved,valueX,proofKind);
 const p=fromStage2Value(valueX);return p?p.x.toFixed(1)+'%':'—';
}
function marker(side){const ns='http://www.w3.org/2000/svg';const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 100 100');svg.setAttribute('class','piece-svg '+(side===1?'mark-x':'mark-o'));svg.setAttribute('aria-hidden','true');
 const shape=document.createElementNS(ns,side===1?'path':'circle');
 if(side===1){shape.setAttribute('d','M14 14 L86 86 M86 14 L14 86');shape.setAttribute('stroke-linecap','round');}
 else {shape.setAttribute('cx','50');shape.setAttribute('cy','50');shape.setAttribute('r','34');}
 shape.setAttribute('fill','none');shape.setAttribute('stroke','currentColor');shape.setAttribute('stroke-width','12');svg.append(shape);return svg;}
function renderBoard(){
 const board=$('board');board.replaceChildren();board.classList.toggle('finished',!!game.winner);const legal=new Set(game.legal());
 for(let b=0;b<9;b++){
  const mini=document.createElement('div');mini.className='mini';
  if(game.boards[b])mini.classList.add('closed',game.boards[b]===1?'x-win':game.boards[b]===2?'o-win':'drawn');
  if(game.target===b&&!game.boards[b])mini.classList.add('active');
  // All nine cells remain in the DOM after a local board closes.
  // A translucent overlay is attached *after* cells so the final move
  // outline can be placed above it with a higher stacking context.
  for(let p=0;p<9;p++){
   const move=b*9+p,cell=document.createElement('button');cell.type='button';cell.className='cell';cell.setAttribute('aria-label',cellName(move));
   const v=game.cells[move];if(v){cell.append(marker(v));cell.classList.add(v===1?'mark-x':'mark-o');}
   if(move===game.last)cell.classList.add('last');
   if(legal.has(move)&&!aiTurn())cell.classList.add('can-play');
   if(legal.has(move)&&latest&&(latest.depth>0||latest.model||latest.solved)&&latest.best===move)cell.classList.add('selected-hint');
   cell.disabled=!legal.has(move)||aiTurn();
   cell.addEventListener('click',()=>{if(game.winner||aiTurn()||!game.isLegal(move))return;play(move);});mini.append(cell);
  }
  if(game.boards[b]){
   const w=document.createElement('div');w.className='winner-mark';w.setAttribute('aria-hidden','true');
   if(game.boards[b]===3)w.textContent='—';else w.append(marker(game.boards[b]));mini.append(w);
  }
  board.append(mini);
 }
 const t=$('turnIndicator');t.className='turn-indicator '+(game.winner===3?'draw':game.turn===1?'x':'o');
 $('turnSymbol').textContent=game.winner===3?'D':game.winner&&game.winner<3?(game.winner===1?'X':'O'):(game.turn===1?'X':'O');
 $('turnText').textContent=game.winner?(game.winner===3?'和棋':'胜利'):(aiTurn()?'AI':'执棋');
 $('boardState').textContent=String(game.moves.length).padStart(2,'0')+' / 81';
 $('moveTrail').replaceChildren(...game.moves.slice(-7).map((m,i,all)=>{const el=document.createElement(i===all.length-1?'b':'span');el.textContent=cellName(m);return el;}));
 $('sidePick').classList.toggle('hidden',mode!=='ha');$('sidePick').classList.toggle('o',human===2);$('sidePick').classList.toggle('x',human===1);$('humanSymbol').textContent=human===1?'X':'O';
 $('pauseBtn').textContent=paused?'▶':'Ⅱ';modeBtns.forEach(b=>b.classList.toggle('selected',b.dataset.mode===mode));
 if(latest)renderHeat(latest);
}
function highlightBest(){document.querySelectorAll('.cell.selected-hint').forEach(e=>e.classList.remove('selected-hint'));
 if(!latest||(!latest.depth&&!latest.model&&!latest.solved)||!game.isLegal(latest.best))return;
 const b=latest.best/9|0,p=latest.best%9,cell=$('board').children[b]?.children[p];
 if(cell?.classList.contains('cell'))cell.classList.add('selected-hint');
}
function clearHeat(){document.querySelectorAll('.eval-tag').forEach(n=>n.remove());document.querySelectorAll('.cell.eval-positive,.cell.eval-negative').forEach(n=>n.classList.remove('eval-positive','eval-negative'));}
function renderHeat(a){clearHeat();if(!$('heatSwitch').checked||!a?.candidates)return;
 const valid=a.candidates.filter(c=>!c.bound&&game.isLegal(c.move));
 const ranked=valid.map(c=>({c,rate:wdl(c.score,!!c.solved,c.valueX)?.x??-1}));
 const best=ranked.length?Math.max(...ranked.map(q=>q.rate)):-Infinity;
 for(const {c,rate} of ranked){const b=c.move/9|0,p=c.move%9,cell=$('board').children[b]?.children[p];if(!cell?.classList.contains('cell'))continue;
  const label=document.createElement('span');label.className='eval-tag';label.textContent=resultText(c.score,!!c.solved,c.valueX,c.proofKind);decorateValue(label,label.textContent);cell.append(label);
  if(best>=0&&rate===best)cell.classList.add('eval-positive');else if(best>=0&&rate>=0&&rate<best-8)cell.classList.add('eval-negative');
 }
}
function renderCandidates(a){
 const list=$('candidateList');list.replaceChildren();
 const ml=!!a?.model&&!Number.isFinite(a.score);
 const candidates=(a?.candidates||[]).filter(c=>!c.bound&&game.isLegal(c.move))
  .sort((x,y)=>ml?((y.visits||0)-(x.visits||0)||((y.prior||0)-(x.prior||0))):
   (Number(y.move===a.best)-Number(x.move===a.best))||
   (Number(y.solved)-Number(x.solved))||
   ((Number.isFinite(y.score)?y.score:-32000)-(Number.isFinite(x.score)?x.score:-32000)))
  .slice(0,5);
 if(!candidates.length){list.textContent='—';return;}
 candidates.forEach((c,i)=>{
  const btn=document.createElement('div');btn.className='candidate'+(c.move===a.best?' active':'');
  const rank=document.createElement('span');rank.className='idx';rank.textContent=String(i+1).padStart(2,'0');
  const name=document.createElement('span');name.className='name';name.textContent=cellName(c.move);
  const bar=document.createElement('span');bar.className='bar';const values=wdl(c.score,!!c.solved,c.valueX);
  for(const [cls,amount] of values?[['wdl-x',values.x],['wdl-d',values.d],['wdl-o',values.o]]:[]){
   if(amount<=0)continue; // No D segment at all in ordinary / neural evaluation.
   const seg=document.createElement('i');seg.className=cls;seg.style.width=amount+'%';
   seg.title=cls==='wdl-x'?'X 胜 '+amount.toFixed(1)+'%':cls==='wdl-d'?'和 '+amount.toFixed(1)+'%':'O 胜 '+amount.toFixed(1)+'%';
   if(amount>=1){const pct=document.createElement('span');pct.textContent=amount.toFixed(1)+'%';seg.append(pct);}
   bar.append(seg);
  }
  const val=document.createElement('span');val.className='value';val.textContent=resultText(c.score,!!c.solved,c.valueX,c.proofKind);decorateValue(val,val.textContent);
  btn.append(rank,name,bar,val);btn.title=values?`X 胜 ${values.x.toFixed(1)}%${values.d?' · 和 '+values.d.toFixed(1)+'%':''} · O 胜 ${values.o.toFixed(1)}%`:'此候选尚未完成严格证明';
  list.append(btn);
 });
}
function clearPanel(){latest=null;iteration=0;liveNodes=0;startClock=0;searchTime=0;
 $('metricDepth').innerHTML='0<span class="unit"> ply</span>';$('metricNodes').textContent='0';$('metricScore').textContent=expectedScoreX(null);$('metricSpeed').innerHTML='0<span class="unit"> k/s</span>';
 $('searchProgress').style.width='0%';$('suggestedMove').textContent='—';$('candidateList').textContent='—';
 $('puctMetrics').textContent='';clearHeat();}

function renderPanel(){
 const elapsed=thinking?Math.max(0,performance.now()-startClock):searchTime;
 const depth=latest?.depth||0;
 $('metricDepth').innerHTML=depth+'<span class="unit"> ply</span>';
 $('metricNodes').textContent=numberText(Math.max(latest?.totalNodes||0,liveNodes));
 const verdict=expectedScoreX(latest?.score,!!latest?.solved,latest?.valueX,latest?.proofKind);
 $('metricScore').textContent=verdict;decorateValue($('metricScore'),verdict);
 const nps=Math.max(latest?.totalNodes||0,liveNodes)/Math.max(elapsed,1);
 $('metricSpeed').innerHTML=nps.toFixed(nps>100?0:1)+'<span class="unit"> k/s</span>';
 $('searchProgress').style.width=mode==='hh'?Math.min(96,5+Math.log10(1+Math.max(0,liveNodes))*12)+'%':Math.min(100,elapsed/(+$('timeRange').value)*100)+'%';
 $('suggestedMove').textContent=(latest?.depth>0||latest?.model||latest?.solved)?cellName(latest.best):'—';
 if(latest?.model)$('puctMetrics').textContent='Stage2 原网络 · PUCT '+numberText(latest.visits||0)+' 次 · '+(latest.model.evaluations||0)+' 次推理';
 else $('puctMetrics').textContent='';
 $('engineLight').classList.toggle('active',thinking);
 if(thinking)$('engineStatus').textContent=(mode==='hh'?'持续分析':'搜索中')+' · '+formatTime(elapsed)+(iteration?' · '+iteration+' ply':'');
}
function stopSearch(){clearTimeout(moveTimer);job++;thinking=false;
 worker?.postMessage({type:'cancel'});
 renderPanel();}
function search(){stopSearch();if(game.winner){$('engineStatus').textContent='终局';return;}
 if(paused){$('engineStatus').textContent='已停止';return;}
 const id=job;latest=null;iteration=0;liveNodes=0;startClock=performance.now();searchTime=0;thinking=true;
 const m={type:'search',id,cells:Array.from(game.cells),turn:game.turn,target:game.target,endless:mode==='hh',...settings()};
 try{
  if(!worker){worker=new Worker('./worker.js',{type:'module'});worker.onmessage=e=>handleEngine(e.data);worker.onerror=e=>{$('engineStatus').textContent='引擎错误';console.error(e);};}
  worker.postMessage(m);

 }catch(e){thinking=false;$('engineStatus').textContent='无法启动';console.error(e);}
 renderPanel();}
function handleEngine(a){if(a.id!==job)return;
 if(a.type==='ready'){$('engineType').textContent=a.backend.toUpperCase();return;}
 if(a.type==='notice'){$('engineStatus').textContent=a.text;return;}
 if(a.type==='thinking'){iteration=a.depth;iterationBudget=a.budget;liveNodes=Math.max(liveNodes,a.totalNodes||0);return;}
 if(a.type==='progress'){liveNodes=Math.max(liveNodes,(latest?.totalNodes||0)+(a.nodes||0));return;}
 if(a.type==='model'){$('puctMetrics').textContent='Stage2 原网络 · FP16 WASM';return;}
 if(a.type==='info'){latest=a;liveNodes=Math.max(liveNodes,a.totalNodes||0);renderCandidates(a);renderHeat(a);highlightBest();renderPanel();return;}
 if(a.type==='done'){
  thinking=false;latest=a;searchTime=a.elapsed||performance.now()-startClock;liveNodes=Math.max(liveNodes,a.totalNodes||0);
  renderCandidates(a);renderHeat(a);highlightBest();renderPanel();
  $('engineStatus').textContent='完成 · '+formatTime(searchTime);

  if(aiTurn()&&game.isLegal(a.best)){const old=job;moveTimer=setTimeout(()=>{if(job===old&&aiTurn()&&game.isLegal(a.best))play(a.best);},130);}
 }
}
function play(move){if(!game.isLegal(move))return;stopSearch();game.play(move);clearPanel();save();renderBoard();search();}
function reset(){stopSearch();game.reset();paused=false;clearPanel();save();renderBoard();search();}
function changeConfig(){renderSettings();save();clearTimeout(configTimer);configTimer=setTimeout(()=>{clearPanel();search();},190);}
function renderSettings(){$('timeValue').textContent=formatTime(+$('timeRange').value);}
modeBtns.forEach(b=>b.addEventListener('click',()=>{if(mode===b.dataset.mode)return;mode=b.dataset.mode;clearPanel();save();renderBoard();search();}));
$('sidePick').addEventListener('click',()=>{human=3-human;clearPanel();save();renderBoard();search();});
$('resetBtn').addEventListener('click',reset);
$('undoBtn').addEventListener('click',()=>{if(!game.moves.length)return;stopSearch();
 // Rewind to the previous human-to-move position when possible.
 let n=1;if(mode==='ha'&&game.moves.length>=2){const copy=game.clone();copy.undo(1);if(copy.turn!==human)n=2;}
 game.undo(n);clearPanel();save();renderBoard();search();});
$('pauseBtn').addEventListener('click',()=>{paused=!paused;save();if(paused){stopSearch();$('engineStatus').textContent='已停止';}else search();renderBoard();});
$('restartSearch').addEventListener('click',()=>{paused=false;save();search();renderBoard();});
$('stopAnalysis').addEventListener('click',()=>{paused=true;stopSearch();save();$('engineStatus').textContent='已停止';renderBoard();});
$('heatSwitch').addEventListener('change',()=>{save();latest?renderHeat(latest):clearHeat();});
for(const key of ['timeRange','studentSwitch'])$(key).addEventListener('input',changeConfig);
restore();renderSettings();renderBoard();clearPanel();search();
// 500ms UI heartbeat; candidate percentages come EXCLUSIVELY from completed ply results.
setInterval(()=>{
 highlightBest();renderPanel();
},500);
