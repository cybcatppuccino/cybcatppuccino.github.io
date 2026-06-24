'use strict';
function formValues(form){let out=Object.fromEntries(new FormData(form).entries()); for(const k of Object.keys(out))if(typeof out[k]==='string')out[k]=out[k].trim(); return out}
const SEARCH_VALUE_KEYS=['id','src','vol','V','B','I','N','area','codeg','degree','h1','h2','hMax','hSum','ehrN','hasInterior','palHstar','unimodalHstar','logHstar','lw','box','content','primitiveVol','primitiveB','prim','nonPrim','maxEdge','minEdge','maxMult','minMult','sing','edge','mult','smooth','allPrimEdges','reflexiveCandidate','kCartier','gorFan','clTorsion','antiK2','fanoIndex','smoothCones','rho','b2','h11','euler','c2','c1sq','signature','projDim','codim','sectionalGenus','canonicalDim','adjDepth','canonicalVertices','centSym','autOrder','autPos','autNeg','nontrivAut','hasReflection','autType'];
function hasActiveSearchCondition(F){return (hasVal(F.ehrN)&&hasVal(F.ehrK))||SEARCH_VALUE_KEYS.some(k=>k!=='ehrN'&&hasVal(F[k]))}
function filterRow(r,F){
  if(F.id){let q=parseInt(F.id.replace(/\D/g,'')); if(r[MIDX.id]!==q)return false}
  if(F.src&&String(r[MIDX.src])!==F.src)return false;
  let h1=hStar1FromRow(r), h2=r[MIDX.I], hMax=Math.max(1,h1,h2), hSum=1+h1+h2, cd=rowCodegree(r), deg=rowDegree(r), mult=rowMultArray(r), maxMult=mult.length?Math.max(...mult):0, minMult=mult.length?Math.min(...mult):0, sing=mult.filter(x=>x>1).length;
  if(!cmp(r[MIDX.Vol],F.volOp,F.vol))return false;
  if(!cmp(r[MIDX.V],F.vOp,F.V))return false;
  if(!cmp(r[MIDX.B],F.bOp,F.B))return false;
  if(!cmp(r[MIDX.I],F.iOp,F.I))return false;
  if(!cmp(r[MIDX.N],F.nOp,F.N))return false;
  if(!cmp(r[MIDX.Vol]/2,F.areaOp,F.area))return false;
  if(!cmp(h1,F.h1Op,F.h1))return false;
  if(!cmp(h2,F.h2Op,F.h2))return false;
  if(!cmp(hMax,F.hMaxOp,F.hMax))return false;
  if(!cmp(hSum,F.hSumOp,F.hSum))return false;
  let k=parseIntegerInput(F.ehrK); if(k!==null&&hasVal(F.ehrN)&&!cmp(ehrhartValueFromRow(r,k),F.ehrNOp,F.ehrN))return false;
  if(!cmp(cd,F.codegOp,F.codeg))return false;
  if(!cmp(deg,F.degreeOp,F.degree))return false;
  if(!cmp(r[MIDX.lw],F.lwOp,F.lw))return false;
  if(!cmp(r[MIDX.content],F.contentOp,F.content))return false;
  if(!cmp(r[MIDX.prim],F.primOp,F.prim))return false;
  if(!cmp(r[MIDX.V]-r[MIDX.prim],F.nonPrimOp,F.nonPrim))return false;
  if(!cmp(r[MIDX.maxE],F.maxEdgeOp,F.maxEdge))return false;
  if(!cmp(r[MIDX.minE],F.minEdgeOp,F.minEdge))return false;
  if(!cmp(maxMult,F.maxMultOp,F.maxMult))return false;
  if(!cmp(minMult,F.minMultOp,F.minMult))return false;
  if(!cmp(sing,F.singOp,F.sing))return false;
  if(!cmp(r[MIDX.V]-2,F.rhoOp,F.rho))return false;
  if(!cmp(r[MIDX.V]-2,F.b2Op,F.b2))return false;
  if(!cmp(r[MIDX.V]-2,F.h11Op,F.h11))return false;
  if(!cmp(r[MIDX.V],F.eulerOp,F.euler))return false;
  if(!cmp(r[MIDX.V],F.c2Op,F.c2))return false;
  if(!cmp(12-r[MIDX.V],F.c1sqOp,F.c1sq))return false;
  if(!cmp(3-r[MIDX.V],F.signatureOp,F.signature))return false;
  if(!cmp(r[MIDX.N]-1,F.projDimOp,F.projDim))return false;
  if(!cmp(r[MIDX.N]-3,F.codimOp,F.codim))return false;
  if(!cmp(r[MIDX.I],F.sectionalGenusOp,F.sectionalGenus))return false;
  if(!cmp(r[MIDX.I]>0?r[MIDX.I]-1:-1,F.canonicalDimOp,F.canonicalDim))return false;
  if(!cycleMatches(r[MIDX.edge],F.edge))return false;
  if(!cycleMatches(r[MIDX.mult],F.mult))return false;
  if(!boolFilter(r[MIDX.smooth],F.smooth))return false;
  if(!boolFilter(r[MIDX.cent],F.centSym))return false;
  if(!boolFilter(h2===1?1:0,F.palHstar))return false;
  if(!boolFilter(h1>=h2?1:0,F.unimodalHstar))return false;
  if(!boolFilter(h1*h1>=h2?1:0,F.logHstar))return false;
  if(!boolFilter(r[MIDX.prim]===r[MIDX.V]?1:0,F.allPrimEdges))return false;
  if(!boolFilter(r[MIDX.I]>0?1:0,F.hasInterior))return false;
  if(!boolFilter(h2===1?1:0,F.reflexiveCandidate))return false;
  return true
}
function exactSearchRequested(F){return ['box','primitiveVol','primitiveB','kCartier','clTorsion','antiK2','fanoIndex','smoothCones','adjDepth','canonicalVertices','autOrder','autPos','autNeg'].some(k=>hasVal(F[k]))||hasVal(F.gorFan)||hasVal(F.nontrivAut)||hasVal(F.hasReflection)||hasVal(F.autType)}
function symmetryTypeValue(A){if(A.order===1)return 'trivial'; if(A.order===2)return 'C2'; if(A.detNeg===0)return 'cyclic'; return 'dihedral'}
async function exactFilterRow(r,F){
  let rec=await fetchRecord(r), v=optDisplay(rec.v).v, s=stats(v);
  if(!cmp(s.boxSq,F.boxOp,F.box))return false;
  let pd=primitivePolygonData(v), ps=stats(pd.poly);
  if(!cmp(ps.Vol,F.primitiveVolOp,F.primitiveVol))return false;
  if(!cmp(ps.B,F.primitiveBOp,F.primitiveB))return false;
  if(hasVal(F.kCartier)||hasVal(F.gorFan)||hasVal(F.clTorsion)||hasVal(F.antiK2)||hasVal(F.fanoIndex)||hasVal(F.smoothCones)){
    let Kidx=s.Ns.map((n,i)=>localGorensteinIndex(s.Ns[(i-1+s.V)%s.V],n)).reduce((a,b)=>lcm(a,b),1);
    let tors=torsionOrder(s.Ns);
    let anti=anticanonicalSquare(s.Ns), antiVal=anti.n/anti.d;
    let fidx=fanoIndexData(s.Ns).freePartIndex;
    let smoothCones=s.mm.filter(x=>x===1).length;
    if(!cmp(Kidx,F.kCartierOp,F.kCartier))return false;
    if(!boolFilter(Kidx===1?1:0,F.gorFan))return false;
    if(!cmp(tors,F.clTorsionOp,F.clTorsion))return false;
    if(!cmp(antiVal,F.antiK2Op,F.antiK2))return false;
    if(!cmp(fidx,F.fanoIndexOp,F.fanoIndex))return false;
    if(!cmp(smoothCones,F.smoothConesOp,F.smoothCones))return false;
  }
  if(hasVal(F.adjDepth)||hasVal(F.canonicalVertices)){
    let tower=adjointTower(v), depth=Math.max(0,tower.length-1), intPoly=s.I?interiorPolygon(v):[];
    let canVerts=(intPoly.length>=3&&area2(intPoly)>0)?hull(intPoly).length:0;
    if(!cmp(depth,F.adjDepthOp,F.adjDepth))return false;
    if(!cmp(canVerts,F.canonicalVerticesOp,F.canonicalVertices))return false;
  }
  if(hasVal(F.autOrder)||hasVal(F.autPos)||hasVal(F.autNeg)||hasVal(F.nontrivAut)||hasVal(F.hasReflection)||hasVal(F.autType)){
    let A=automorphisms(v), typ=symmetryTypeValue(A);
    if(!cmp(A.order,F.autOrderOp,F.autOrder))return false;
    if(!cmp(A.detPos,F.autPosOp,F.autPos))return false;
    if(!cmp(A.detNeg,F.autNegOp,F.autNeg))return false;
    if(!boolFilter(A.order>1?1:0,F.nontrivAut))return false;
    if(!boolFilter(A.detNeg>0?1:0,F.hasReflection))return false;
    if(hasVal(F.autType)&&F.autType!==typ)return false;
  }
  return true
}
async function search(ev){
  ev&&ev.preventDefault();
  renderStop=false;
  let F=formValues($('#search-form'));
  if(!hasActiveSearchCondition(F)){
    $('#results').innerHTML='<p class="empty">Please enter at least one search condition before searching. The Reset state is intentionally not searched.</p>';
    $('#result-count').textContent='0';
    $('#search-status').textContent='No search was run: enter at least one condition such as Record ID, volume, vertex count, source, toric data, or symmetry data.';
    return;
  }
  let lim=Math.max(1,parseIntegerInput(F.limit)||DEFAULT_DISPLAY_LIMIT), needsExact=exactSearchRequested(F);
  let hits=[], limitHit=false;
  let t0=performance.now(), strong=0, checked=0;
  for(const r of META){
    if(renderStop)break;
    if(!filterRow(r,F))continue;
    strong++;
    if(needsExact){
      checked++;
      if(!await exactFilterRow(r,F))continue;
      if(checked%150===0){$('#search-status').textContent=`Strong metadata pass: ${strong.toLocaleString()} candidate row(s); exact/weak checks: ${checked.toLocaleString()}...`; await new Promise(x=>setTimeout(x,0))}
    }
    hits.push(r);
    if(hits.length>=lim){limitHit=true; break}
  }
  $('#search-status').textContent=`Strong metadata pass kept ${strong.toLocaleString()} row(s)${needsExact?`; exact post-filter checked ${checked.toLocaleString()}`:''} in ${(performance.now()-t0).toFixed(0)} ms. Merging AGL2Z-equivalent records...`;
  $('#results').innerHTML='';
  let grouped=await groupRowsByPolygon(hits);
  $('#result-count').textContent=grouped.length.toLocaleString();
  await renderResults(grouped,$('#results'),{alreadyGrouped:true,autoOpen:true});
  if(limitHit){
    $('#results').insertAdjacentHTML('afterbegin',`<p class="small-note limit-warning"><strong>Display limit ${lim.toLocaleString()} reached.</strong> There may be more matching records; only the first ${lim.toLocaleString()} raw match(es) were rendered. Increase Display limit for a broader search.</p>`);
  }
  $('#search-status').textContent=`${grouped.length.toLocaleString()} merged result${grouped.length===1?'':'s'} rendered from ${hits.length.toLocaleString()} matching record${hits.length===1?'':'s'}.${limitHit?` Display limit ${lim.toLocaleString()} was reached; more records may match.`:''}`;
}
async function groupRowsByPolygon(rows){let groups=[], byKey=new Map(), bySig=new Map(), groupPolys=[]; for(let i=0;i<rows.length&&!renderStop;i++){let r=rows[i], rec=await fetchRecord(r), st=stats(rec.v), sig=signatureFromStats(st), key=optKey(rec.v), g=byKey.get(key); if(!g){let idx=null; for(const gi of bySig.get(sig)||[]){if(aglEquivalent(rec.v,groupPolys[gi])){idx=gi;break}} if(idx===null){r.equivRows=[]; r.equivKey=key; idx=groups.length; groups.push(r); groupPolys.push(rec.v); if(!bySig.has(sig))bySig.set(sig,[]); bySig.get(sig).push(idx)} g=groups[idx]; byKey.set(key,g)} g.equivRows.push(r); if(i&&i%250===0){$('#search-status')&&($('#search-status').textContent=`Merging equivalent polygons with exact AGL2Z fallback: ${i.toLocaleString()}/${rows.length.toLocaleString()} records...`); await pause()}} return groups}
function recordLabel(r){return r&&r.__virtual?(r.__virtualTitle||'Drawn polygon'):id(r[MIDX.id])}
function groupLabel(r){if(r&&r.__virtual)return recordLabel(r); let g=r.equivRows||[r]; return g.length>1?`${id(r[MIDX.id])} + ${g.length-1} equivalent record(s)`:id(r[MIDX.id])}
function groupSources(r){if(r&&r.__virtual)return 'drawn'; let g=r.equivRows||[r], labs=[...new Set(g.map(x=>srcName(x[MIDX.src])))]; return labs.join(', ')}
async function renderResults(rows,container,opts={}){container.innerHTML=''; let grouped=opts.alreadyGrouped?rows:await groupRowsByPolygon(rows), batch=300; for(let i=0;i<grouped.length&&!renderStop;i+=batch){let frag=document.createDocumentFragment(); for(const r of grouped.slice(i,i+batch))frag.appendChild(resultElement(r)); container.appendChild(frag); await pause()} if(!grouped.length)container.innerHTML='<p class="empty">No matching records.</p>'; if(opts.autoOpen&&grouped.length===1){let btn=container.querySelector('[data-act="open"]'); if(btn)btn.click()}}
function resultElement(r,originHint=null){let el=document.createElement('div'); el.className='result'; let eq=r.equivRows?.length>1?`<span class="chip">merged ${r.equivRows.length}</span>`:''; el.innerHTML=`<div class="result-main"><div><div class="result-title">${groupLabel(r)} <span class="chip">${groupSources(r)}</span>${eq}</div><div class="chips"><span class="chip">Vol ${r[MIDX.Vol]}</span><span class="chip">V ${r[MIDX.V]}</span><span class="chip">B ${r[MIDX.B]}</span><span class="chip">I ${r[MIDX.I]}</span><span class="chip">N ${r[MIDX.N]}</span><span class="chip">lw ${r[MIDX.lw]}</span><span class="chip">edge [${r[MIDX.edge]}]</span></div></div><div class="result-actions"><button type="button" data-act="open" class="open-record-btn">Open</button><button type="button" data-act="download">Download</button></div></div><div class="detail-slot" hidden></div>`; let slot=el.querySelector('.detail-slot'), openBtn=el.querySelector('[data-act="open"]'); openBtn.onclick=async()=>{if(!slot.hidden){slot.hidden=true;slot.innerHTML='';openBtn.textContent='Open';openBtn.classList.remove('opened');return} slot.hidden=false; openBtn.textContent='Close'; openBtn.classList.add('opened'); slot.innerHTML='<p class="empty">Loading record...</p>'; await openRecord(r,slot,originHint)}; el.querySelector('[data-act="download"]').onclick=()=>downloadRecordData(r); return el}
function transformDisplay(v){let base=optDisplay(v).v; let b=bbox(base), W=b.maxX-b.minX, H=b.maxY-b.minY; let P=base.map(p=>[p[0]-b.minX,p[1]-b.minY]); for(let k=0;k<((viewState.rot%4)+4)%4;k++){P=P.map(p=>[p[1],W-p[0]]); [W,H]=[H,W]} if(viewState.fx)P=P.map(p=>[W-p[0],p[1]]); if(viewState.fy)P=P.map(p=>[p[0],H-p[1]]); return cyc(P)}
async function openRecord(row,slot,originHint=null){try{let rec=await fetchRecord(row); currentOpen={row,rec,slot}; selectedOrigin=originHint; viewState={rot:0,fx:false,fy:false}; renderRecord()}catch(e){console.error('openRecord failed',e); if(slot)slot.innerHTML=`<div class="card error-card"><h3>Could not render record</h3><p class="empty">${e&&e.message?e.message:e}</p><pre class="code-box">${String(e&&e.stack?e.stack:e).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}</pre></div>`}}
function recordCardKey(details){let h=details.querySelector(':scope > summary > h3, :scope > summary > h5'); return h?(h.textContent||'').trim():details.className}
function syncRecordCollapseHint(details){let hint=details.querySelector(':scope > summary > .collapse-hint'); if(hint)hint.textContent=details.open?'expanded':'collapsed'}
function bindRecordCollapsibleState(details,defaultOpen=true){if(!details||details.dataset.recordCollapseBound)return; let key=recordCardKey(details); if(RECORD_CARD_OPEN_STATE.has(key))details.open=RECORD_CARD_OPEN_STATE.get(key); else details.open=defaultOpen; syncRecordCollapseHint(details); details.dataset.recordCollapseBound='1'; details.addEventListener('toggle',()=>{RECORD_CARD_OPEN_STATE.set(key,details.open); syncRecordCollapseHint(details)})}
function makeRecordCardsCollapsible(root){if(!root)return; let cards=Array.from(root.querySelectorAll('.record-grid > div > .card, .record-grid > .wide-figures > .card')); cards.forEach(card=>{if(card.tagName==='DETAILS'||card.dataset.collapsibleWrapped)return; let title=card.querySelector(':scope > h3'); if(!title)return; let details=document.createElement('details'); details.className=(card.className+' collapsed-card record-collapsible-card').trim(); let summary=document.createElement('summary'); summary.appendChild(title); let hint=document.createElement('span'); hint.className='collapse-hint'; summary.appendChild(hint); details.appendChild(summary); while(card.firstChild)details.appendChild(card.firstChild); card.replaceWith(details); bindRecordCollapsibleState(details,true)}); root.querySelectorAll('.record-grid > div > details.card, .record-grid > .wide-figures > details.card').forEach(d=>{let title=recordCardKey(d); bindRecordCollapsibleState(d,!/^Symmetry group and point orbits$/.test(title))})}
function renderRecord(){let {row,rec,slot}=currentOpen; let v=transformDisplay(rec.v), s=stats(v), P=latticePoints(v); if(!selectedOrigin||!P.inn.some(p=>p[0]===selectedOrigin[0]&&p[1]===selectedOrigin[1]))selectedOrigin=P.inn[0]||null; let verts=new Set(v.map(p=>p.join(','))), boundaryNonVerts=P.bd.filter(p=>!verts.has(p.join(','))); let G=[{P:boundaryNonVerts,c:'boundary',r:3,label:'boundary lattice point'},{P:P.inn,c:'interior',r:4,label:'interior lattice point'},{P:v,c:'vertex',r:5,label:'vertex'}]; if(selectedOrigin)G.push({P:[selectedOrigin],c:'origin',r:7,label:'selected origin'}); let cent=centroidData(v,P); let CG=[{P:[cent.area.point],c:'centA',r:6,label:'area centroid'},{P:[cent.all.point],c:'centL',r:6,label:'lattice centroid'},{P:[cent.boundary.point],c:'centB',r:6,label:'boundary centroid'}]; if(cent.interior)CG.push({P:[cent.interior.point],c:'centI',r:6,label:'interior centroid'}); slot.innerHTML=`<div class="record-grid"><div class="wide-figures"><div class="card"><h3>${recordLabel(row)}: minimum-square display</h3><div class="toggle-row"><button id="rotL">↺ 90°</button><button id="rotR">↻ 90°</button><button id="flipX">left/right flip</button><button id="flipY">up/down flip</button>${copyButton(v)}</div><div class="figure-box" id="poly-figure">${svgPolygon(v,G,{large:true})}</div><p class="subtle">Vertices are red, boundary lattice points blue, interior lattice points green, and the selected origin is black. All invariants depending on a pointed origin use the black interior point selected here. If no external choice is provided, the first interior lattice point is selected automatically.</p><div class="point-buttons">${P.inn.map(p=>`<button data-p="${p.join(',')}" class="${selectedOrigin&&selectedOrigin[0]===p[0]&&selectedOrigin[1]===p[1]?'active':''}">(${p[0]},${p[1]})</button>`).join('')||'<span class="subtle">No interior lattice points.</span>'}</div></div><div class="card"><h3>Centroids</h3><div class="toggle-row">${copyButton(v)}</div><div class="figure-box">${svgPolygon(v,CG,{large:false})}</div><div class="coords">${centroidText(cent)}</div></div></div><div>${basicSection(s,row,rec,P)}${symmetrySection(s,P)}${ehrhartSection(s)}${toricSection(s,v,P)}${algebraicSection(s)}${minkowskiSection(s,P)}</div><div>${fanSection(s,v)}${pointedSection(s,v)}${primitivePolygonSection(s,v)}${dualPolygonSection(s,v)}${neighborSection()}</div></div>`; makeRecordCardsCollapsible(slot); slot.querySelector('#rotL').onclick=()=>{viewState.rot--;selectedOrigin=null;renderRecord()}; slot.querySelector('#rotR').onclick=()=>{viewState.rot++;selectedOrigin=null;renderRecord()}; slot.querySelector('#flipX').onclick=()=>{viewState.fx=!viewState.fx;selectedOrigin=null;renderRecord()}; slot.querySelector('#flipY').onclick=()=>{viewState.fy=!viewState.fy;selectedOrigin=null;renderRecord()}; slot.querySelectorAll('#poly-figure .interior,.point-buttons button[data-p]').forEach(el=>el.onclick=()=>{selectedOrigin=el.dataset.p.split(',').map(Number);renderRecord()}); let eh=slot.querySelector('#ehrhart-order'); if(eh)eh.onchange=()=>updateEhrhartSeries(s,slot); let nb=slot.querySelector('#compute-neighbors'); if(nb)nb.onclick=()=>computeNeighbors(v,slot); attachBlowupControls(v,slot); let bu=slot.querySelector('#compute-blowup'); if(bu)bu.onclick=()=>computeBlowup(v,slot); let bd=slot.querySelector('#compute-blowdown'); if(bd)bd.onclick=()=>computeBlowdownDAG(v,slot); let per=slot.querySelector('#compute-period'); if(per)per.onclick=()=>computeMirrorPeriod(v,slot); let pf=slot.querySelector('#compute-pf'); if(pf)pf.onclick=()=>computePicardFuchsPackage(v,slot); let stopPf=slot.querySelector('#stop-pf'); if(stopPf)stopPf.onclick=()=>{if(PF_CANCEL)PF_CANCEL.cancelled=true}; let mk=slot.querySelector('#compute-minkowski'); if(mk)mk.onclick=()=>computeMinkowskiDecompositions(v,slot); attachCopyButtons(slot); renderMath(slot); setTimeout(()=>{if(currentOpen&&currentOpen.slot===slot){computeNeighbors(v,slot); computeAdjointMatches(v,slot); computePrimitiveSearch(v,slot); computeDualSearch(v,slot); computeMinkowskiDecompositions(v,slot)}},0)}
