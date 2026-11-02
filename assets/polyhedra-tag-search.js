/* Polyhedra v13 tag/alias search companion. Complements the site's built-in search. */
(() => {
 'use strict';
 function ready() {
  const input=document.getElementById('solidSearch');
  const db=window.POLYHEDRA_LIBRARY;
  if(!input||!db||!Array.isArray(db.objects))return;
  const root=input.parentElement;
  root.style.position='relative';
  const pop=document.createElement('div');pop.id='polyTagSuggestions';
  Object.assign(pop.style,{position:'absolute',zIndex:'99',left:'0',right:'0',top:'100%',maxHeight:'260px',overflowY:'auto',display:'none',border:'1px solid #9997',borderRadius:'9px',boxShadow:'0 8px 20px #0002',background:'var(--surface,#fff)',color:'var(--text,#242424)',padding:'4px'});
  root.appendChild(pop);
  const txt=o=>(o.searchText||[o.name,o.id,o.source,...(o.aliases||[]),...(o.tags||[])].join(' ')).toLocaleLowerCase();
  function select(o){
   pop.style.display='none';input.value=o.name;
   input.dispatchEvent(new Event('input',{bubbles:true}));
   input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',code:'Enter',bubbles:true}));
   input.dispatchEvent(new KeyboardEvent('keyup',{key:'Enter',code:'Enter',bubbles:true}));
  }
  function update(){
   const q=input.value.trim().toLocaleLowerCase();pop.replaceChildren();
   if(q.length<2){pop.style.display='none';return;}
   const found=db.objects.filter(o=>{
    const tags=[...(o.aliases||[]),...(o.tags||[])].join(' ').toLocaleLowerCase();
    return tags.includes(q) || (o.id&&o.id.toLocaleLowerCase().includes(q));
   }).slice(0,12);
   if(!found.length){pop.style.display='none';return;}
   for(const o of found){
    const b=document.createElement('button');b.type='button';
    b.textContent=o.name+'  ·  '+o.source;
    Object.assign(b.style,{display:'block',width:'100%',border:'0',textAlign:'left',padding:'7px 9px',cursor:'pointer',background:'transparent',color:'inherit',font:'inherit'});
    b.addEventListener('pointerdown',e=>e.preventDefault());
    b.addEventListener('click',()=>select(o));pop.appendChild(b);
   }
   pop.style.display='block';
  }
  input.addEventListener('input',update);
  input.addEventListener('focus',update);
  input.addEventListener('keydown',e=>{if(e.key==='Escape')pop.style.display='none'; if(e.key==='Enter'&&pop.style.display!=='none'&&pop.firstChild){e.stopImmediatePropagation();e.preventDefault();pop.firstChild.click();}});
  document.addEventListener('pointerdown',e=>{if(!root.contains(e.target))pop.style.display='none';});
 }
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',ready);else ready();
})();
