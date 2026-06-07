    function createRIESCommandPreviewFallback(){
      return null;
    }
    const resultBody = document.getElementById('resultBody');
    const resultTools = document.getElementById('resultTools');
    const resultToolsMeta = document.getElementById('resultToolsMeta');
    const sortConfidenceBtn = document.getElementById('sortConfidenceBtn');
    const sortDiscoveryBtn = document.getElementById('sortDiscoveryBtn');
    const hpPanel = document.getElementById('hpPanel');
    const hpContent = document.getElementById('hpContent');
    const numberTools = document.getElementById('numberTools');
    const numberToolsContent = document.getElementById('numberToolsContent');
    const statusEl = document.getElementById('status');
    const paramToggle = document.getElementById('paramToggle');
    const parametersPanel = document.getElementById('parametersPanel');
    const stopBtn = document.getElementById('stopBtn');
    const continueBtn = document.getElementById('continueBtn');
    const runBtn = document.getElementById('runBtn');
    const targetInput = document.getElementById('target');
    const DEFAULT_SHORT_EFFORT = '3';
    const DEFAULT_RIES_LEVEL = '4';
    let activeShortformRun = null;
    let lastSolvedRaw = '';
    let pendingContinueSolve = false;
    let lastRenderedRows = [];
    let activeSolveRun = null;
    let inputStateEpoch = 0;
    let lastInputSnapshot = '';
    let currentResultAllRows = [];
    let currentResultDiscoveryRows = [];
    let currentResultSettings = null;
    let currentResultSorted = false;
    let shortformDbLoadPromise = null;
    const packageLoadPromises = new Map();
    const SHORTFORM_DB_ASSET_URL = 'assets/shortform100k.js?v=11.5.1';
    function isShortformDbReady(){ return !!(window.RIES_SHORTFORM_100K_PACKED || window.RIES_SHORTFORM_100K || window.RIES_SHORTFORM_100K_MULTI); }
    function packageStatusText(label, loaded, total, stage, expectedBytes){
      const loadedMb = loaded ? (loaded/1048576).toFixed(2)+' MB' : '0 MB';
      const totalMb = total ? ' / '+(total/1048576).toFixed(2)+' MB' : '';
      const exp = Number(expectedBytes||0);
      const expText = exp>0 && (!total || Math.abs(exp-total)>Math.max(524288,total*.20)) ? ` · JS ≈ ${(exp/1048576).toFixed(2)} MB` : '';
      return `${stage || 'Loading'} ${label} package… ${loadedMb}${totalMb}${expText}`;
    }
    function updatePackageLoadStatus(label, loaded, total, baseProgress, spanProgress, phase, stage, expectedBytes){
      if(typeof setSearchStatus !== 'function') return;
      const exp = Number(expectedBytes||0);
      const effectiveTotal = (Number.isFinite(total) && total>0) ? total : (exp>0 ? exp : 0);
      const known = Number.isFinite(effectiveTotal) && effectiveTotal>0;
      const frac = known ? Math.min(1, Math.max(0, loaded/effectiveTotal)) : Math.min(.90, Math.max(.04, loaded/2500000));
      setSearchStatus(packageStatusText(label, loaded, total, stage, exp), Math.min(.995, baseProgress + spanProgress*frac), phase || 'loading package');
    }
    function appendScriptPackage(url, isReady, label, baseProgress, spanProgress, phase, expectedBytes){
      return new Promise(resolve=>{
        if(typeof document === 'undefined' || !document.createElement){ resolve(false); return; }
        const base=url.split('?')[0].replace(/^\.\//,'');
        const existing=[...document.querySelectorAll('script[src]')].find(s=>String(s.getAttribute('src')||s.src||'').includes(base));
        if(existing){
          if(isReady()){ resolve(true); return; }
          let settled=false;
          const finish=ok=>{ if(!settled){ settled=true; resolve(!!ok); } };
          if(existing.addEventListener){
            existing.addEventListener('load', ()=>finish(isReady()), {once:true});
            existing.addEventListener('error', ()=>finish(false), {once:true});
          }else{
            const poll=()=>{ if(isReady()) finish(true); else setTimeout(poll, 100); };
            poll();
          }
          return;
        }
        updatePackageLoadStatus(label, 0, 0, baseProgress, spanProgress, phase, 'Loading', expectedBytes);
        const script=document.createElement('script');
        script.src=url;
        script.async=true;
        script.onload=()=>{ updatePackageLoadStatus(label, expectedBytes||1, expectedBytes||1, baseProgress, spanProgress, phase, 'Loaded', expectedBytes); resolve(!!isReady()); };
        script.onerror=()=>{ console.warn(`RIES package failed to load: ${url}`); resolve(false); };
        (document.head || document.body || document.documentElement).appendChild(script);
      });
    }
    async function loadScriptPackageWithProgress(url, isReady, opts={}){
      if(isReady()) return true;
      const label=opts.label || url.split('/').pop().split('?')[0];
      const phase=opts.phase || 'loading package';
      const baseProgress=Number.isFinite(opts.baseProgress) ? opts.baseProgress : .10;
      const spanProgress=Number.isFinite(opts.spanProgress) ? opts.spanProgress : .12;
      const expectedBytes=Number(opts.expectedBytes||0);
      const key=url.split('?')[0];
      if(packageLoadPromises.has(key)) return packageLoadPromises.get(key);
      const promise=(async()=>{
        const canFetch = typeof fetch==='function' && typeof ReadableStream!=='undefined' && typeof TextDecoder!=='undefined' && !(typeof location!=='undefined' && location.protocol==='file:');
        if(canFetch){
          try{
            updatePackageLoadStatus(label, 0, 0, baseProgress, spanProgress, phase, 'Loading', expectedBytes);
            const res=await fetch(url, {cache:'force-cache'});
            if(!res.ok) throw new Error(`HTTP ${res.status}`);
            const total=Number(res.headers.get('content-length') || 0);
            if(res.body && res.body.getReader){
              const reader=res.body.getReader();
              const chunks=[]; let loaded=0;
              while(true){
                const {done,value}=await reader.read();
                if(done) break;
                chunks.push(value); loaded += value.byteLength || value.length || 0;
                updatePackageLoadStatus(label, loaded, total, baseProgress, spanProgress, phase, 'Loading', expectedBytes)
              }
              const bytes=new Uint8Array(loaded); let offset=0;
              for(const chunk of chunks){ bytes.set(chunk, offset); offset += chunk.byteLength || chunk.length || 0; }
              const code=new TextDecoder('utf-8').decode(bytes);
              const script=document.createElement('script');
              script.text=code+'\n//# sourceURL='+url.split('?')[0];
              (document.head || document.body || document.documentElement).appendChild(script);
              updatePackageLoadStatus(label, expectedBytes||total||loaded||1, total||loaded||expectedBytes||1, baseProgress, spanProgress, phase, 'Loaded', expectedBytes);
              return !!isReady();
            }
            const code=await res.text();
            const script=document.createElement('script');
            script.text=code+'\n//# sourceURL='+url.split('?')[0];
            (document.head || document.body || document.documentElement).appendChild(script);
            updatePackageLoadStatus(label, expectedBytes||code.length, code.length, baseProgress, spanProgress, phase, 'Loaded', expectedBytes);
            return !!isReady();
          }catch(e){
            console.warn(`Progressive package load failed for ${url}; falling back to script tag.`, e);
          }
        }
        return appendScriptPackage(url, isReady, label, baseProgress, spanProgress, phase, expectedBytes);
      })();
      packageLoadPromises.set(key, promise);
      return promise;
    }
    function ensureShortformDbLoaded(opts={}){
      if(isShortformDbReady()) return Promise.resolve(true);
      if(shortformDbLoadPromise) return shortformDbLoadPromise;
      shortformDbLoadPromise = loadScriptPackageWithProgress(SHORTFORM_DB_ASSET_URL, isShortformDbReady, {
        label: opts.label || 'precomputed shortform database',
        phase: opts.phase || 'integer database',
        baseProgress: Number.isFinite(opts.baseProgress) ? opts.baseProgress : .20,
        spanProgress: Number.isFinite(opts.spanProgress) ? opts.spanProgress : .08
      }).then(ok=>{
        if(!ok) console.warn('RIES precomputed shortform database is unavailable; structured and exact searches will still run.');
        return ok;
      });
      return shortformDbLoadPromise;
    }
    const idle = () => new Promise(resolve => setTimeout(resolve, 0));
    const nextPaint = () => new Promise(resolve => {
      let done=false;
      const finish=()=>{ if(!done){ done=true; resolve(); } };
      if(typeof requestAnimationFrame === 'function') requestAnimationFrame(()=>requestAnimationFrame(finish));
      setTimeout(finish, 80);
    });
    async function yieldToUI(){
      // Yield all the way to a paint opportunity, not just a macrotask.  This
      // keeps the progress bar and the SO(4) tesseract animation alive while
      // integer database / shortform work is split into small slices.
      await new Promise(resolve=>{
        let done=false;
        const finish=()=>{ if(!done){ done=true; resolve(); } };
        if(typeof requestAnimationFrame === 'function') requestAnimationFrame(()=>setTimeout(finish,0));
        else setTimeout(finish,0);
        setTimeout(finish,50);
      });
    }
    function resetContinueState(){
      continueBtn.dataset.mode='';
      continueBtn.disabled=true;
      continueBtn.textContent='Continue higher';
    }
    function setContinueState(mode,nextLabel,disabled=false){
      continueBtn.dataset.mode=mode || '';
      continueBtn.disabled=!!disabled;
      continueBtn.textContent=nextLabel || 'Continue higher';
    }
    function stopActiveSolve(message='Stopping now; the current slice will finish and all rows already found stay on screen.'){
      if(activeSolveRun) activeSolveRun.stopped=true;
      if(activeShortformRun) activeShortformRun.stopped=true;
      if(stopBtn){ stopBtn.disabled=true; }
      if(statusEl){ statusEl.className='notice status-line'; statusEl.textContent=message; }
    }
    function abortIfStaleOrStopped(run){
      if(!run) return;
      if(run.epoch !== inputStateEpoch) throw new Error('RIES_STALE_INPUT');
      if(run.stopped) throw new Error('RIES_STOPPED');
    }
    async function yieldAndCheck(run){
      await yieldToUI();
      abortIfStaleOrStopped(run);
    }
    function resetSearchFrameworkForInputChange(){
      inputStateEpoch++;
      if(activeSolveRun) activeSolveRun.stopped=true;
      if(activeShortformRun) activeShortformRun.stopped=true;
      pendingContinueSolve=false;
      lastSolvedRaw='';
      lastRenderedRows=[];
      solveRunCache.clear();
      integerGlobalCache.clear();
      lfuncProgressCache.clear();
      resetContinueState();
      clearResultTools();
      if(statusEl){ statusEl.dataset.progress='0'; statusEl.className='notice status-line'; statusEl.textContent='Input changed. Search state and per-input caches were reset.'; }
      if(stopBtn) stopBtn.disabled=true;
      if(runBtn) runBtn.disabled=false;
      if(hpPanel){ hpPanel.hidden=true; hpContent.innerHTML=''; }
      if(numberTools){ numberTools.hidden=true; numberTools.open=false; numberToolsContent.innerHTML=''; }
    }
    function shortAbort(deadline){ return performance.now()>deadline || !!activeShortformRun?.stopped || !!activeSolveRun?.stopped; }
    function timeSliceDeadline(ms, hardDeadline=null){
      const now=performance.now();
      const soft=now+Math.max(20, Number(ms)||20);
      return hardDeadline===null ? soft : Math.min(soft, hardDeadline);
    }
    function isUserInputPending(){
      try{
        return !!(navigator && navigator.scheduling && navigator.scheduling.isInputPending && navigator.scheduling.isInputPending({includeContinuous:true}));
      }catch(e){ return false; }
    }
    function escapeHtml(s){ return String(s ?? '').replace(/[&<>"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch])); }

    // v11.7.2: central LaTeX cleanup helpers used by RIES, harddb, hypdata,
    // intsumdb, L-function, and result rendering paths.  The helpers are
    // intentionally conservative: they remove only algebraically neutral ^0/^1
    // powers from simple/generated expressions, normalize adjacent signs created
    // by templating negative parameters, and preserve existing commands such as
    // \frac, \sqrt, \Gamma, \left/\right, and \operatorname.
    function escapeLatex(s){
      const map={'\\':'\\backslash{}','{':'\\{','}':'\\}','$':'\\$','&':'\\&','%':'\\%','#':'\\#','_':'\\_','^':'\\^{}','~':'\\sim{}'};
      return String(s ?? '').replace(/[\\{}$&%#_^~]/g, ch=>map[ch]||ch);
    }
    function latexTrimOuterSpaces(s){ return String(s ?? '').replace(/\s+/g,' ').trim(); }
    function latexIsZero(s){ return /^[-+]?0(?:\.0+)?$/.test(latexTrimOuterSpaces(s)); }
    function latexIsOne(s){ return /^\+?1(?:\.0+)?$/.test(latexTrimOuterSpaces(s)); }
    function latexIsMinusOne(s){ return /^-1(?:\.0+)?$/.test(latexTrimOuterSpaces(s)); }
    function latexNegateScalarLatex(s){
      s=latexTrimOuterSpaces(s);
      if(latexIsZero(s)) return '0';
      if(s.startsWith('-')) return s.slice(1).trim();
      return '-'+s;
    }
    function latexHasTopLevelAddSub(s){
      s=String(s ?? ''); let brace=0, par=0, bracket=0, escaped=false;
      for(let i=0;i<s.length;i++){
        const ch=s[i];
        if(escaped){ escaped=false; continue; }
        if(ch==='\\'){ escaped=true; continue; }
        if(ch==='{') brace++; else if(ch==='}') brace=Math.max(0,brace-1);
        else if(ch==='(') par++; else if(ch===')') par=Math.max(0,par-1);
        else if(ch==='[') bracket++; else if(ch===']') bracket=Math.max(0,bracket-1);
        else if((ch==='+' || ch==='-') && i>0 && brace===0 && par===0 && bracket===0) return true;
      }
      return false;
    }
    function latexGroupIfNeeded(s){
      s=latexTrimOuterSpaces(s);
      if(!s) return s;
      if(latexHasTopLevelAddSub(s) || /^-/.test(s)) return `\\left(${s}\\right)`;
      return s;
    }
    function latexPow(base, exp){
      base=latexTrimOuterSpaces(base); exp=latexNormalizeSigns(exp);
      if(!base) return '';
      if(latexIsZero(exp)) return '1';
      if(latexIsOne(exp)) return base;
      const b=/^(?:[A-Za-z0-9]|\\[A-Za-z]+|\\pi|\\theta|\\varphi)$/.test(base) ? base : latexGroupIfNeeded(base);
      return `${b}^{${exp}}`;
    }
    function latexFactorPow(base, exp){
      const p=latexPow(base, exp);
      return p==='1' ? '' : p;
    }
    function latexProductFactors(factors, sep=''){
      const kept=[];
      for(const f of factors){ const t=latexTrimOuterSpaces(f); if(t && t!=='1') kept.push(t); }
      return kept.length ? kept.join(sep) : '1';
    }
    function latexMulScalar(coeff, expr){
      coeff=latexNormalizeSigns(coeff); expr=latexTrimOuterSpaces(expr);
      if(!expr || latexIsZero(coeff)) return '0';
      if(!coeff || latexIsOne(coeff)) return expr;
      const e=latexGroupIfNeeded(expr);
      if(latexIsMinusOne(coeff)) return '-'+e;
      return `${coeff}\\,${e}`;
    }
    function latexNormalizeSigns(s){
      s=String(s ?? '');
      let out=s.replace(/−/g,'-');
      for(let k=0;k<8;k++){
        const before=out;
        out=out
          .replace(/\+\s*-\s*/g,' - ')
          .replace(/-\s*\+\s*/g,' - ')
          .replace(/\+\s*\+\s*/g,' + ')
          .replace(/-\s*-\s*/g,' + ')
          .replace(/\{\s*\+\s*/g,'{')
          .replace(/\(\s*\+\s*/g,'(')
          .replace(/\[\s*\+\s*/g,'[')
          .replace(/\^\{\s*\+\s*/g,'^{')
          .replace(/_\{\s*\+\s*/g,'_{')
          .replace(/\s+,/g,',')
          .replace(/;\s*\+\s*/g,';')
          .replace(/\s{2,}/g,' ');
        if(out===before) break;
      }
      return out.trim();
    }

    function latexFindMatchingParen(s, openIndex){
      let depth=0;
      for(let i=openIndex;i<s.length;i++){
        const c=s[i];
        if(c==='(') depth++;
        else if(c===')'){
          depth--;
          if(depth===0) return i;
        }
      }
      return -1;
    }
    function latexNeutralPowerAt(s, index){
      const rest=s.slice(index);
      const m=rest.match(/^\^\{\s*([+-]?\d+)\s*\}|^\^([+-]?\d+)(?!\d)/);
      if(!m) return null;
      const n=Number(m[1]!==undefined ? m[1] : m[2]);
      if(n===0 || n===1) return {value:n, len:m[0].length};
      return null;
    }
    function latexSimplifyFunctionCallPowers(s){
      const names=new Set(['sin','cos','tan','cot','sec','csc','asin','acos','atan','arcsin','arccos','arctan','sinh','cosh','tanh','arsinh','arcosh','artanh','log','ln','exp']);
      let out=String(s ?? '');
      for(let pass=0; pass<10; pass++){
        let changed=false;
        let res='';
        for(let i=0;i<out.length;){
          if(out[i]==='\\'){
            const m=out.slice(i+1).match(/^[A-Za-z]+/);
            if(m && names.has(m[0])){
              let j=i+1+m[0].length;
              while(out[j]===' ') j++;
              if(out[j]==='('){
                const close=latexFindMatchingParen(out,j);
                if(close>j){
                  const pow=latexNeutralPowerAt(out, close+1);
                  if(pow){
                    res += pow.value===1 ? out.slice(i, close+1) : '1';
                    i=close+1+pow.len;
                    changed=true;
                    continue;
                  }
                }
              }
            }
          }
          res += out[i++];
        }
        out=res;
        if(!changed) break;
      }
      return out;
    }
    function latexSimplifyBalancedGroupPowers(s){
      let out=String(s ?? '');
      for(let pass=0; pass<10; pass++){
        let changed=false;
        let res='';
        for(let i=0;i<out.length;){
          if(out[i]==='('){
            const close=latexFindMatchingParen(out,i);
            if(close>i){
              const pow=latexNeutralPowerAt(out, close+1);
              if(pow){
                res += pow.value===1 ? out.slice(i, close+1) : '1';
                i=close+1+pow.len;
                changed=true;
                continue;
              }
            }
          }
          res += out[i++];
        }
        out=res;
        if(!changed) break;
      }
      return out;
    }
    function latexSimplifyNeutralPowers(s){
      s=String(s ?? '');
      s=s.replace(/\^\{\s*([-+]?\d+)\s*([+-])\s*([-+]?\d+)\s*\}/g,(m,a,op,b)=>`^{${Number(a)+(op==='+'?Number(b):-Number(b))}}`);
      s=s.replace(/\^\{\s*([-+]?\d+)\s*\}/g,(m,a)=>`^{${Number(a)}}`);
      const zeroReplace=(m, offset, full)=> full.trim()===m.trim() ? '1' : '';
      s=latexSimplifyFunctionCallPowers(s);
      const fnCall=/\\(?:sin|cos|tan|log|exp|arctan|arsinh|sinh|cosh)\s*\((?:[^()]|\([^()]*\))*\)/g;
      for(let pass=0; pass<4; pass++){
        const before=s;
        s=s.replace(new RegExp('('+fnCall.source+')\\^\\{1\\}','g'),'$1');
        s=s.replace(new RegExp('('+fnCall.source+')\\^\\{0\\}','g'),(m,atom,offset,full)=>zeroReplace(m,offset,full));
        s=s.replace(new RegExp('('+fnCall.source+')\\^1(?!\\d)','g'),'$1');
        s=s.replace(new RegExp('('+fnCall.source+')\\^0(?!\\d)','g'),(m,atom,offset,full)=>zeroReplace(m,offset,full));
        if(s===before) break;
      }
      s=s.replace(/(\\log\s*x|\\log\s*t|\\log\s*\([^()]*\)|[A-Za-z]|\\[A-Za-z]+|\\pi|\\theta|\\varphi)\^\{1\}/g,'$1');
      s=s.replace(/(\\log\s*x|\\log\s*t|\\log\s*\([^()]*\)|[A-Za-z]|\\[A-Za-z]+|\\pi|\\theta|\\varphi)\^\{0\}/g,(m,atom,offset,full)=>zeroReplace(m,offset,full));
      s=s.replace(/(\\log\s*x|\\log\s*t|\\log\s*\([^()]*\)|[A-Za-z]|\\[A-Za-z]+|\\pi|\\theta|\\varphi)\^1/g,'$1');
      s=s.replace(/(\\log\s*x|\\log\s*t|\\log\s*\([^()]*\)|[A-Za-z]|\\[A-Za-z]+|\\pi|\\theta|\\varphi)\^0/g,(m,atom,offset,full)=>zeroReplace(m,offset,full));
      s=s.replace(/(\([^()]*\)|\\left\([^{}]*?\\right\)|\{[^{}]*?\})\^\{1\}/g,'$1');
      s=s.replace(/(\([^()]*\)|\\left\([^{}]*?\\right\)|\{[^{}]*?\})\^\{0\}/g,(m,atom,offset,full)=>zeroReplace(m,offset,full));
      s=s.replace(/(\([^()]*\)|\\left\([^{}]*?\\right\)|\{[^{}]*?\})\^1/g,'$1');
      s=s.replace(/(\([^()]*\)|\\left\([^{}]*?\\right\)|\{[^{}]*?\})\^0/g,(m,atom,offset,full)=>zeroReplace(m,offset,full));
      s=s.replace(/\(\\log\s*x\)\^\{1\}/g,'\\log x').replace(/\(\\log\s*x\)\^\{0\}/g,(m,offset,full)=>zeroReplace(m,offset,full));
      s=s.replace(/\(\\log\s*t\)\^\{1\}/g,'\\log t').replace(/\(\\log\s*t\)\^\{0\}/g,(m,offset,full)=>zeroReplace(m,offset,full));
      s=latexSimplifyBalancedGroupPowers(s);
      s=s.replace(/([\{+\-(])\s*1(?=(?:[A-Za-z]|\\(?!right\b)[A-Za-z]+|\\left|\())/g,'$1');
      s=s.replace(/(?:^|(?<=\s))1(?=(?:[A-Za-z]|\\(?!right\b)[A-Za-z]+|\\left|\())/g,'');
      if(!s.trim()) s='1';
      return s;
    }
    function latexNormalizeSqrt(s){
      s=String(s ?? '');
      // Convert common RIES text remnants such as √(a/b) if any survive to the
      // MathJax \sqrt form.  Do not touch already-correct \sqrt{...} commands.
      return s.replace(/√\(([^()]*)\)/g,(m,body)=>`\\sqrt{${body}}`).replace(/√\{([^{}]*)\}/g,(m,body)=>`\\sqrt{${body}}`);
    }
    function sanitizeLatexForDisplay(s){
      let out=String(s ?? '');
      out=out.replace(/\\operatorname\{sqrt\}\s*\\?!?\\left\((.*?)\\right\)/g,'\\sqrt{$1}');
      out=out.replace(/\\operatorname\{sqrt\}\s*\\?!?\((.*?)\)/g,'\\sqrt{$1}');
      out=latexNormalizeSqrt(out);
      out=latexNormalizeSigns(out);
      out=latexSimplifyNeutralPowers(out);
      out=latexNormalizeSigns(out);
      out=out.replace(/\\left\(([^(){}+\-]*?)\\right\)/g,'$1');
      out=out.replace(/(?<![0-9])1+\\,d/g,'\\,d');
      out=out.replace(/\s+,/g,',').replace(/\s+;/g,';').replace(/\s{2,}/g,' ').trim();
      return out;
    }
    const fmtValue = x => Number.isFinite(x) ? Number(x).toPrecision(13).replace(/(?:\.0+|(?<=\d)0+)$/,'') : String(x);
    function fmtErr(x){
      if(!Number.isFinite(x)) return String(x);
      x = Math.abs(x); if(x===0) return '0';
      const e = Math.floor(Math.log10(x));
      let m = x / Math.pow(10,e);
      let s = m.toPrecision(3).replace(/\.0+$/,'').replace(/(\.\d*?)0+$/,'$1');
      if(s === '10'){ s='1'; return `${s}*10^${e+1}`; }
      return `${s}*10^${e}`;
    }
    const quant = x => Number.isFinite(x) ? x.toPrecision(12) : null;
    function parseTarget(s){
      const raw = String(s || '').trim(); if(!raw) return NaN;
      const expr = raw
        .replace(/[−–—]/g,'-')
        .replace(/[×·]/g,'*')
        .replace(/[÷]/g,'/')
        .replaceAll('π','pi')
        .replace(/\bC\s*\(/gi,'binom(')
        .replace(/\bA\s*\(/gi,'perm(')
        .replaceAll('^','**');
      if(/^[0-9a-zA-Z_+\-*/().,!\s*]+$/.test(expr)){
        try{
          const fact=n=>{ n=Number(n); if(!Number.isInteger(n)||n<0||n>170) throw Error('bad factorial'); let r=1; for(let k=2;k<=n;k++) r*=k; return r; };
          const binom=(n,k)=>{ n=Number(n); k=Number(k); if(!Number.isInteger(n)||!Number.isInteger(k)||n<0||k<0||k>n) throw Error('bad binom'); k=Math.min(k,n-k); let r=1; for(let i=1;i<=k;i++) r=r*(n-k+i)/i; return r; };
          const perm=(n,k)=>{ n=Number(n); k=Number(k); if(!Number.isInteger(n)||!Number.isInteger(k)||n<0||k<0||k>n) throw Error('bad perm'); let r=1; for(let i=0;i<k;i++) r*=n-i; return r; };
          const gcd=(a,b)=>{ a=Math.abs(Math.trunc(a)); b=Math.abs(Math.trunc(b)); while(b){ const t=a%b; a=b; b=t; } return a; };
          const lcm=(a,b)=>{ a=Math.trunc(a); b=Math.trunc(b); return Math.abs(a/gcd(a,b)*b); };
          const fib=n=>{ n=Number(n); if(!Number.isInteger(n)||n<0||n>1476) throw Error('bad fib'); let a=0,b=1; for(let i=0;i<n;i++){ const t=a+b; a=b; b=t; } return a; };
          const catalan=n=>binom(2*n,n)/(n+1);
          const jsExpr = expr.replace(/([0-9A-Za-z_).]+)!/g,'fact($1)');
          const v = Function('"use strict"; const pi=Math.PI, e=Math.E, phi=(1+Math.sqrt(5))/2, sqrt=Math.sqrt, sin=Math.sin, cos=Math.cos, tan=Math.tan, log=Math.log, ln=Math.log, exp=Math.exp, abs=Math.abs, pow=Math.pow, fact=arguments[0], factorial=arguments[0], binom=arguments[1], choose=arguments[1], ncr=arguments[1], c=arguments[1], perm=arguments[2], npr=arguments[2], a=arguments[2], gcd=arguments[3], lcm=arguments[4], fib=arguments[5], fibonacci=arguments[5], catalan=arguments[6]; return ('+jsExpr.toLowerCase()+');')(fact,binom,perm,gcd,lcm,fib,catalan);
          if(Number.isFinite(v)) return v;
        }catch(e){}
      }
      return Number(raw);
    }
    function canonicalTargetString(raw, value){
      if(!Number.isFinite(value)) return String(raw || '').trim();
      const s = Number(value).toPrecision(17);
      return s.replace(/(?:\.0+|(\.\d*?)0+)(e[+\-]?\d+)?$/i, (m, dec, exp)=> (dec || '') + (exp || ''));
    }
    function normalizeDecimalString(raw){
      return String(raw || '').trim().replace(/[，]/g,'.').replace(/[−–—]/g,'-').replace(/\s+/g,'');
    }
    function parseDecimalRational(raw){
      let s=normalizeDecimalString(raw).toLowerCase();
      if(!/^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?$/.test(s)) return null;
      let sign=1n;
      if(s[0]==='-'){ sign=-1n; s=s.slice(1); }
      else if(s[0]==='+') s=s.slice(1);
      let [mant, expstr] = s.split('e');
      let exp = expstr ? Number(expstr) : 0;
      if(!Number.isFinite(exp) || Math.abs(exp)>1000000) return null;
      let frac = 0;
      if(mant.includes('.')){ frac = mant.length - mant.indexOf('.') - 1; mant = mant.replace('.',''); }
      const digitMantForSig = mant;
      const firstSig = digitMantForSig.search(/[1-9]/);
      // v9.3 precision policy: count exactly the precision the user typed.
      // Do not strip typed trailing zeroes (e.g. 1.2300 has five significant
      // digits), but still ignore leading zeroes before the first non-zero.
      const sigDigits = Math.max(1, firstSig >= 0 ? digitMantForSig.length - firstSig : digitMantForSig.replace(/^0+/, '').length || 1);
      mant = mant.replace(/^0+(?=\d)/,'') || '0';
      let num = BigInt(mant) * sign;
      let den = 10n ** BigInt(frac);
      if(exp>0) num *= 10n ** BigInt(exp);
      if(exp<0) den *= 10n ** BigInt(-exp);
      const g=gcdBig(num,den); if(g>1n || g<-1n){ num/=g; den/=g; }
      if(den<0n){ den=-den; num=-num; }
      return {num, den, sigDigits, raw:raw};
    }
    function rationalToNumber(q){
      if(!q) return NaN;
      return Number(q.num) / Number(q.den);
    }
    function rationalIsZero(q){ return !q || q.num===0n; }
    function signedRationalString(q){
      if(!q) return '0';
      return q.den===1n ? q.num.toString() : `${q.num.toString()}/${q.den.toString()}`;
    }
    function parseImagCoeff(s){
      if(s==='' || s==='+') return '1';
      if(s==='-') return '-1';
      return s;
    }
    function findLastAddSubOutsideExponent(s){
      for(let i=s.length-1;i>0;i--){
        const ch=s[i];
        if((ch==='+' || ch==='-') && s[i-1] !== 'e' && s[i-1] !== 'E') return i;
      }
      return -1;
    }
    function parseDecimalComplex(raw){
      let s=normalizeDecimalString(raw).replace(/[ijIＩ]/g,'i').replace(/\*i/g,'i').replace(/i\*/g,'i');
      s=s.replace(/\+\-/g,'-').replace(/\-\+/g,'-').replace(/\+\+/g,'+');
      while(s.includes('--')) s=s.replace(/--/g,'+');
      if(!s) return null;
      const icount=(s.match(/i/g)||[]).length;
      if(icount===0){
        const re=parseDecimalRational(s); if(!re) return null;
        return {re, im:{num:0n, den:1n, sigDigits:0}, isComplex:false, precisionDigits:re.sigDigits};
      }
      if(icount!==1 || !s.endsWith('i')) return null;
      const body=s.slice(0,-1);
      let reStr='0', imStr='1';
      if(body==='' || body==='+' || body==='-') imStr=parseImagCoeff(body);
      else {
        const idx=findLastAddSubOutsideExponent(body);
        if(idx>0){ reStr=body.slice(0,idx); imStr=parseImagCoeff(body.slice(idx)); }
        else { imStr=parseImagCoeff(body); }
      }
      const re=parseDecimalRational(reStr), im=parseDecimalRational(imStr);
      if(!re || !im) return null;
      return {re, im, isComplex:im.num!==0n, precisionDigits:Math.max(re.sigDigits, im.sigDigits)};
    }
    function canonicalComplexString(parsed){
      if(!parsed) return '';
      if(rationalIsZero(parsed.im)) return signedRationalString(parsed.re);
      const re=signedRationalString(parsed.re), im=signedRationalString(parsed.im);
      return `${re}${parsed.im.num<0n?'':'+'}${im}i`;
    }
    function decimalPrecision(raw){
      const parsed=parseDecimalComplex(raw);
      if(parsed) return Math.max(0, Math.min(120, parsed.precisionDigits));
      let s=String(raw).trim(); if(!/^[+\-]?\d*(\.\d*)?([eE][+\-]?\d+)?$/.test(s) || !/[0-9]/.test(s)) return 12;
      let [mant, expstr] = s.toLowerCase().split('e'); const exp = expstr ? Number(expstr) : 0;
      const frac = mant.includes('.') ? mant.length - mant.indexOf('.') - 1 : 0;
      return Math.max(0, Math.min(120, frac - exp));
    }
    function readSettings(){
      const byId=id=>document.getElementById(id);
      const checkedId=(id, fallback=true)=>{ const el=byId(id); return el ? !!el.checked : !!fallback; };
      const numId=(id, fallback, lo=-Infinity, hi=Infinity)=>{ const raw=String(byId(id)?.value ?? '').trim(); const v=Number(raw); const x=(!raw || !Number.isFinite(v)) ? Number(fallback) : v; return Math.max(lo, Math.min(hi, x)); };
      const budgetMsId=(id, fallback, lo=0, hi=120000)=>{ const raw=String(byId(id)?.value ?? '').trim(); const v=Number(raw); const x=(!raw || !Number.isFinite(v)) ? Number(fallback) : v; if(x<=0) return 0; return Math.max(lo, Math.min(hi, x)); };
      const only = new Set((byId('onlySyms')?.value || '').trim().split(''));
      const never = new Set((byId('neverSyms')?.value || '').trim().split(''));
      const checked = new Set([...document.querySelectorAll('[data-sym]:checked')].map(x=>x.dataset.sym));
      const digits = new Set((byId('digits')?.value || '0123456789').replace(/[^0-9]/g,'').split(''));
      const restrict = byId('restrictMode')?.value || 'none';
      const maxAbs = parseTarget(byId('maxAbs')?.value || '1e9') || 1e9;
      const maxRelErrorRaw=String(byId('maxRelError')?.value || 'Infinity').trim();
      const maxRelErrorNum=Number(maxRelErrorRaw);
      const maxRelError = /^inf(inity)?$/i.test(maxRelErrorRaw) ? Infinity : (Number.isFinite(maxRelErrorNum) ? Math.max(0, maxRelErrorNum) : Infinity);
      function allowed(sym){
        if(!checkedId('moduleRiesEq', true)) return false;
        if(/[0-9]/.test(sym) && !digits.has(sym)) return false;
        if(only.size && !only.has(sym)) return false;
        if(never.has(sym)) return false;
        if(/[pefnrsqlESTC+\-*/^vL]/.test(sym) && !checked.has(sym) && !/[0-9]/.test(sym)) return false;
        if(restrict === 'rational' && 'pefqslESTC^vL'.includes(sym)) return false;
        if(restrict === 'constructible' && 'pelESTC^vL'.includes(sym)) return false;
        return true;
      }
      const raw = byId('target')?.value.trim() || '';
      const parsedComplex = parseDecimalComplex(raw);
      const complexTarget = !!(parsedComplex && parsedComplex.isComplex);
      const target = parsedComplex && !complexTarget ? rationalToNumber(parsedComplex.re) : parseTarget(raw);
      const normalizedRaw = parsedComplex ? canonicalComplexString(parsedComplex) : canonicalTargetString(raw, target);
      const modules={
        riesEq:checkedId('moduleRiesEq', true),
        algebraic:checkedId('moduleAlgebraic', true),
        log:checkedId('moduleLog', true),
        linearCombo:checkedId('moduleLinearCombo', true),
        mobius:checkedId('moduleMobius', true),
        constantDb:checkedId('moduleConstantDb', true),
        hardDb:checkedId('moduleHardDb', true),
        hypData:checkedId('moduleHypData', true),
        intsumDb:checkedId('moduleIntsumDb', true),
        lfunc:checkedId('moduleLfunc', true),
        integer:checkedId('moduleInteger', true)
      };
      const constDbTransforms={
        pow1:checkedId('cdbTransX', true),
        exp:checkedId('cdbTransExp', true),
        log:checkedId('cdbTransLog', true),
        powm1:checkedId('cdbTransInv', true),
        pow2:checkedId('cdbTransSquare', true)
      };
      const constDbPasses={
        rational:checkedId('cdbPassRational', true),
        affine:checkedId('cdbPassAffine', true),
        quadratic:checkedId('cdbPassQuadratic', true),
        mobius:checkedId('cdbPassMobius', true),
        algebraic:checkedId('cdbPassAlgebraic', true),
        log:checkedId('cdbPassLog', true)
      };
      const hardDbOptions={
        depth4:checkedId('hardDbDepth4', true), depth5:checkedId('hardDbDepth5', true),
        rational:checkedId('hardDbPassRational', true), power:checkedId('hardDbPassPower', true), exponential:checkedId('hardDbPassExponential', true), logScale:checkedId('hardDbPassLogScale', true),
        rationalHeight:numId('hardDbRationalHeight',10,1,80), maxParamHeight:numId('hardDbMaxParamHeight',15,1,200)
      };
      const hypDataOptions={
        depth1:checkedId('hypDepth1', true), depth2:checkedId('hypDepth2', true), depth3:checkedId('hypDepth3', true),
        multSimple:checkedId('hypMultSimple', true), multGamma:checkedId('hypMultGamma', true), multDeep:checkedId('hypMultDeep', true)
      };
      const intsumDbOptions={
        depth1:checkedId('intsumDepth1', true), depth2:checkedId('intsumDepth2', true), depth3:checkedId('intsumDepth3', true),
        multSimple:checkedId('intsumMultSimple', true), multGamma:checkedId('intsumMultGamma', true), multDeep:checkedId('intsumMultDeep', true)
      };
      const logOptions={targetLogAbs:checkedId('logTargetLogAbs', true), targetRaw:checkedId('logTargetRaw', true), targetLogLogAbs:checkedId('logTargetLogLogAbs', true)};
      const mobiusOptions={direct:checkedId('mobiusDirect', true), logabs:checkedId('mobiusLogAbs', true), exp:checkedId('mobiusExp', true), triple:checkedId('mobiusTriple', true)};
      const linearComboOptions={one:checkedId('linearCombo1Term', true), two:checkedId('linearCombo2Term', true), three:checkedId('linearCombo3Term', true)};
      const lfuncOptions={rational:checkedId('lfuncRational', true), quadratic:checkedId('lfuncQuadratic', true), log:checkedId('lfuncLog', true), specialConstants:checkedId('specialConstants', true)};
      const integerAllowExternal=checkedId('integerAllowExternalFactorization', checkedId('allowExternalFactorization', false));
      const integerOptions={factor:checkedId('integerFactor', true), db:checkedId('integerDb', true), shortform:checkedId('integerShortform', true), allowExternal:integerAllowExternal};
      const moduleLimits={
        riesEq:numId('riesLimit',5,1,50), algebraic:numId('algLimit',5,1,50), log:numId('logLimit',5,1,50), linearCombo:numId('linearComboLimit',5,1,50), mobius:numId('mobiusLimit',5,1,50), constantDb:numId('constantDbLimit',5,1,50), hardDb:numId('hardDbLimit',5,1,50), hypData:numId('hypDataLimit',5,1,50), intsumDb:numId('intsumDbLimit',5,1,50), lfunc:numId('lfuncLimit',5,1,50), integer:numId('integerLimit',5,1,50)
      };
      const depthBudgetDefault=riesLevelDefaultModuleBudgetMs(Number(byId('level')?.value || DEFAULT_RIES_LEVEL));
      const stageBudgets={
        riesMs:budgetMsId('riesBudgetMs', depthBudgetDefault, 0, 120000),
        algebraicMs:budgetMsId('algBudgetMs', 3600, 0, 120000), logMs:budgetMsId('logBudgetMs', depthBudgetDefault, 0, 120000),
        linearComboMs:budgetMsId('linearComboBudgetMs', 3000, 0, 120000), mobiusMs:budgetMsId('mobiusBudgetMs', depthBudgetDefault, 0, 120000),
        constantDb4Ms:budgetMsId('constantDb4BudgetMs', 20000, 0, 300000), constantDb5Ms:budgetMsId('constantDb5BudgetMs', 45000, 0, 300000), constantDb6Ms:budgetMsId('constantDb6BudgetMs', 135000, 0, 600000),
        hardDb4Ms:budgetMsId('hardDb4BudgetMs', 1000, 0, 120000), hardDb5Ms:budgetMsId('hardDb5BudgetMs', 1000, 0, 120000),
        hypData1Ms:budgetMsId('hypData1BudgetMs', 1000, 0, 120000), hypData2Ms:budgetMsId('hypData2BudgetMs', 5000, 0, 120000), hypData3Ms:budgetMsId('hypData3BudgetMs', 50000, 0, 300000),
        intsumDb1Ms:budgetMsId('intsumDb1BudgetMs', 1000, 0, 120000), intsumDb2Ms:budgetMsId('intsumDb2BudgetMs', 5000, 0, 120000), intsumDb3Ms:budgetMsId('intsumDb3BudgetMs', 50000, 0, 300000),
        lfuncMs:budgetMsId('lfuncBudgetMs', depthBudgetDefault, 0, 120000), integerFactorMs:budgetMsId('integerFactorBudgetMs', 0, 0, 300000)
      };
      return { raw, normalizedRaw, parsedComplex, complexTarget, target, level: Number(byId('level')?.value || DEFAULT_RIES_LEVEL), shortEffort: Number(byId('shortEffort')?.value || 0), limit: Math.max(1, Math.min(50, Number(byId('limit')?.value)||5)), restrict, allowed, tol: Infinity, maxAbs, maxRelError, only: [...only].join(''), never: [...never].join(''), doEq:modules.riesEq, doExpr:false, doAlg:modules.algebraic, doLog:modules.log, modules, constDbTransforms, constDbPasses, hardDbOptions, hypDataOptions, intsumDbOptions, logOptions, mobiusOptions, linearComboOptions, lfuncOptions, integerOptions, moduleLimits, stageBudgets, allowExternalFactorization: !!byId('allowExternalFactorization')?.checked || integerAllowExternal };
    }
    function settingsForModule(settings, key){
      const lim=Number(settings?.moduleLimits?.[key]);
      return Number.isFinite(lim) ? {...settings, limit:Math.max(1, Math.min(50, lim))} : settings;
    }
    function pushExpr(store, arr, byC, expr, maxAbs){
      if(!Number.isFinite(expr.v) || Math.abs(expr.v) > maxAbs) return false;
      const k = quant(expr.v); if(!k) return false;
      const old = store.get(k);
      if(old && (old.c < expr.c || (old.c === expr.c && old.s.length <= expr.s.length))) return false;
      store.set(k, expr); arr.push(expr); if(byC){ if(!byC.has(expr.c)) byC.set(expr.c, []); byC.get(expr.c).push(expr); }
      return true;
    }
    function safePow(a,b){ if(!Number.isFinite(a)||!Number.isFinite(b)) return NaN; if(a<0 && Math.abs(b-Math.round(b))>1e-10) return NaN; if(Math.abs(b)>8 || Math.abs(Math.log(Math.max(Math.abs(a),1e-12))*b)>24) return NaN; return Math.pow(a,b); }
    function constOps(settings){
      const un=[];
      if(settings.allowed('n')) un.push({sym:'n', w:1, f:a=>-a.v, s:a=>`-(${a.s})`});
      if(settings.allowed('r')) un.push({sym:'r', w:2, f:a=>Math.abs(a.v)>1e-12?1/a.v:NaN, s:a=>`1/(${a.s})`});
      if(settings.allowed('s')) un.push({sym:'s', w:2, f:a=>a.v*a.v, s:a=>`(${a.s})²`});
      if(settings.allowed('q')) un.push({sym:'q', w:2, f:a=>a.v>=0?Math.sqrt(a.v):NaN, s:a=>`√(${a.s})`});
      if(settings.allowed('l')) un.push({sym:'l', w:3, f:a=>a.v>0?Math.log(a.v):NaN, s:a=>`ln(${a.s})`});
      if(settings.allowed('E')) un.push({sym:'E', w:3, f:a=>Math.abs(a.v)<16?Math.exp(a.v):NaN, s:a=>`exp(${a.s})`});
      if(settings.allowed('S')) un.push({sym:'S', w:4, f:a=>Math.sin(Math.PI*a.v), s:a=>`sinπ(${a.s})`});
      if(settings.allowed('C')) un.push({sym:'C', w:4, f:a=>Math.cos(Math.PI*a.v), s:a=>`cosπ(${a.s})`});
      if(settings.allowed('T')) un.push({sym:'T', w:5, f:a=>Math.abs(Math.cos(Math.PI*a.v))>1e-8?Math.tan(Math.PI*a.v):NaN, s:a=>`tanπ(${a.s})`});
      const bin=[];
      if(settings.allowed('+')) bin.push({sym:'+', w:1, comm:true, f:(a,b)=>a.v+b.v, s:(a,b)=>`(${a.s}+${b.s})`});
      if(settings.allowed('-')) bin.push({sym:'-', w:1, comm:false, f:(a,b)=>a.v-b.v, s:(a,b)=>`(${a.s}-${b.s})`});
      if(settings.allowed('*')) bin.push({sym:'*', w:1, comm:true, f:(a,b)=>a.v*b.v, s:(a,b)=>`(${a.s}·${b.s})`});
      if(settings.allowed('/')) bin.push({sym:'/', w:2, comm:false, f:(a,b)=>Math.abs(b.v)>1e-12?a.v/b.v:NaN, s:(a,b)=>`(${a.s}/${b.s})`});
      if(settings.allowed('^')) bin.push({sym:'^', w:3, comm:false, f:(a,b)=>safePow(a.v,b.v), s:(a,b)=>`(${a.s}^${b.s})`});
      if(settings.allowed('v')) bin.push({sym:'v', w:3, comm:false, f:(a,b)=>Math.abs(b.v)>1e-12?safePow(a.v,1/b.v):NaN, s:(a,b)=>`root(${a.s},${b.s})`});
      if(settings.allowed('L')) bin.push({sym:'L', w:4, comm:false, f:(a,b)=>a.v>0&&b.v>0&&Math.abs(b.v-1)>1e-9?Math.log(a.v)/Math.log(b.v):NaN, s:(a,b)=>`log_${b.s}(${a.s})`});
      return {un,bin};
    }
    function generateConstants(settings){
      const maxExpr = {1:1000,2:2600,3:6200,4:11500,5:19000,6:30000,7:45000,8:65000,9:90000}[settings.level];
      const maxC = {1:7,2:9,3:11,4:13,5:15,6:17,7:19,8:21,9:23}[settings.level];
      const cap = {1:120,2:220,3:330,4:420,5:520,6:650,7:800,8:950,9:1100}[settings.level];
      const store=new Map(), all=[], byC=new Map();
      function add(e){ return pushExpr(store, all, byC, e, settings.maxAbs); }
      for(const d of '0123456789') if(settings.allowed(d)) add({s:d,v:Number(d),c:1});
      if(settings.allowed('p')) add({s:'π',v:Math.PI,c:2});
      if(settings.allowed('e')) add({s:'e',v:Math.E,c:2});
      if(settings.allowed('f')) add({s:'φ',v:(1+Math.sqrt(5))/2,c:2});
      const {un,bin}=constOps(settings);
      for(let c=2; c<=maxC && all.length < maxExpr; c++){
        const snap = all.slice(0, maxExpr);
        for(const a of snap){ for(const op of un){ if(a.c+op.w!==c) continue; add({s:op.s(a), v:op.f(a), c}); if(all.length>=maxExpr) break; } if(all.length>=maxExpr) break; }
        for(let ca=1; ca<c; ca++){
          for(const op of bin){ const cb = c-ca-op.w; if(cb<1) continue; const A=(byC.get(ca)||[]).slice(0,cap), B=(byC.get(cb)||[]).slice(0,cap); for(const a of A){ for(const b of B){ if(op.comm && a.s>b.s) continue; add({s:op.s(a,b), v:op.f(a,b), c}); if(all.length>=maxExpr) break; } if(all.length>=maxExpr) break; } if(all.length>=maxExpr) break; }
          if(all.length>=maxExpr) break;
        }
      }
      return all;
    }
    function lhsUnaryOps(settings){ return constOps(settings).un.map(op=>({w:op.w, s:op.s, f:fn=>x=>op.f({v:fn(x)})})); }
    function generateLHS(constants, settings){
      const maxC = {1:6,2:8,3:10,4:12,5:14,6:16,7:18,8:20,9:22}[settings.level];
      const maxForms = {1:700,2:1800,3:4200,4:7800,5:12500,6:19000,7:28000,8:40000,9:55000}[settings.level];
      const capConst = {1:60,2:110,3:170,4:230,5:280,6:340,7:420,8:520,9:650}[settings.level];
      const store=new Map(), all=[];
      function valAt(f){ try{ const v=f(settings.target); return Number.isFinite(v) ? v : NaN; }catch(e){ return NaN; } }
      function add(e){ const v=valAt(e.fn); if(!Number.isFinite(v)||Math.abs(v)>settings.maxAbs) return; const k=`${e.c}|${v.toPrecision(10)}|${e.s}`; if(store.has(k)) return; store.set(k,e); all.push(e); }
      add({s:'x', c:1, fn:x=>x});
      const un = lhsUnaryOps(settings);
      const bin = constOps(settings).bin.filter(op=>op.sym!=='L' || settings.level>=3);
      for(let c=2; c<=maxC && all.length<maxForms; c++){
        const snap=all.slice();
        for(const a of snap){ for(const op of un){ if(a.c+op.w!==c) continue; add({s:op.s({s:a.s}), c, fn:op.f(a.fn)}); if(all.length>=maxForms) break; } if(all.length>=maxForms) break; }
        for(const a of snap){
          for(const op of bin){
            for(const b of constants.filter(k=>k.c+a.c+op.w===c).slice(0,capConst)){
              add({s:op.s({s:a.s},{s:b.s}), c, fn:x=>op.f({v:a.fn(x)},{v:b.v})});
              if(!op.comm) add({s:op.s({s:b.s},{s:a.s}), c, fn:x=>op.f({v:b.v},{v:a.fn(x)})});
              if(all.length>=maxForms) break;
            }
            if(all.length>=maxForms) break;
          }
          if(all.length>=maxForms) break;
        }
      }
      return all;
    }
    function numericDerivative(fn, x){ const h=1e-6*Math.max(1,Math.abs(x)); try{ const d=(fn(x+h)-fn(x-h))/(2*h); return Number.isFinite(d) ? d : NaN; }catch(e){ return NaN; } }
    function refineRoot(fn, rhs, target){
      let x=target;
      for(let i=0;i<8;i++){
        let y; try{ y=fn(x)-rhs; }catch(e){ return NaN; }
        if(!Number.isFinite(y)) return NaN;
        const h=1e-6*Math.max(1,Math.abs(x));
        let dy; try{ dy=(fn(x+h)-fn(x-h))/(2*h); }catch(e){ return NaN; }
        if(!Number.isFinite(dy)||Math.abs(dy)<1e-11) break;
        const nx=x-y/dy; if(!Number.isFinite(nx)||Math.abs(nx)>1e12) return NaN;
        if(Math.abs(nx-x)<1e-13*Math.max(1,Math.abs(x))) { x=nx; break; }
        x=nx;
      }
      return x;
    }
    function equationSearch(constants, settings){
      const lhs = generateLHS(constants, settings);
      const sorted = constants.slice().sort((a,b)=>a.v-b.v);
      const rows=[]; const seen=new Set();
      function lowerBound(arr, v){ let lo=0,hi=arr.length; while(lo<hi){ const mid=(lo+hi)>>1; if(arr[mid].v<v) lo=mid+1; else hi=mid; } return lo; }
      for(const f of lhs){
        let fv; try{ fv=f.fn(settings.target); }catch(e){ continue; }
        if(!Number.isFinite(fv)) continue;
        const derivAtTarget = numericDerivative(f.fn, settings.target);
        if(!Number.isFinite(derivAtTarget) || Math.abs(derivAtTarget) < 1e-8) continue;
        const pos = lowerBound(sorted, fv);
        for(let j=Math.max(0,pos-8); j<Math.min(sorted.length,pos+9); j++){
          const rhs=sorted[j]; const root=refineRoot(f.fn, rhs.v, settings.target);
          if(!Number.isFinite(root)) continue;
          const err=Math.abs(root-settings.target);
          let residual; try{ residual = Math.abs(f.fn(root)-rhs.v); }catch(e){ residual = Infinity; }
          if(err>settings.tol || !Number.isFinite(residual) || residual > 1e-7*Math.max(1,Math.abs(rhs.v))) continue;
          const candidate=`RIES equation: ${f.s} = ${rhs.s}`; const key=candidate+'|'+root.toPrecision(10); if(seen.has(key)) continue; seen.add(key);
          const lhsLatex=exprToLatex(f.s), rhsLatex=exprToLatex(rhs.s);
          const eqLatex=`${lhsLatex} = ${rhsLatex}`;
          rows.push({candidate, latex:eqLatex, copyLatex:eqLatex, value:`x = ${fmtValue(root)}`, err, c:f.c+rhs.c});
        }
      }
      return rows.sort((a,b)=>a.err-b.err||a.c-b.c||a.candidate.length-b.candidate.length).slice(0,settings.limit);
    }
    function absBig(x){ return x < 0n ? -x : x; }
    function gcdBig(a,b){ a=absBig(a); b=absBig(b); while(b){ const t=a%b; a=b; b=t; } return a; }
    function floorDiv(n,d){ if(d<0n){ n=-n; d=-d; } if(n>=0n) return n/d; return -((-n+d-1n)/d); }
    function ceilDiv(n,d){ if(d<0n){ n=-n; d=-d; } if(n>=0n) return (n+d-1n)/d; return -((-n)/d); }
    function roundDiv(n,d){ if(d<0n){ n=-n; d=-d; } return n>=0n ? (2n*n + d)/(2n*d) : -((2n*(-n)+d)/(2n*d)); }
    function roundModeForDiv(n,d,target){
      // v7: never leave exact integer ratio candidates as "round".
      // Any rational value that rounds to an integer target is also either
      // floor(...) or ceil(...), except invalid zero-denominator cases.
      if(d<0n){ n=-n; d=-d; }
      if(d===0n) return 'invalid';
      if(n%d===0n && n/d===target) return 'exact';
      if(floorDiv(n,d)===target) return 'floor';
      if(ceilDiv(n,d)===target) return 'ceil';
      const rd=roundDiv(n,d);
      if(rd===target){
        const cmp=n-target*d;
        if(cmp>=0n) return 'floor';
        return 'ceil';
      }
      return 'invalid';
    }
    function decimalScaledPowers(raw, deg, prec){
      const q=parseDecimalRational(raw);
      if(!q) return null;
      const scale = 10n ** BigInt(Math.max(0, Math.min(120, prec)));
      const scaled=[]; let pn=1n, dn=1n;
      for(let i=0;i<=deg;i++){ scaled.push(roundDiv(scale*pn,dn)); pn*=q.num; dn*=q.den; }
      return {scaled, scaledRe:scaled, scaledIm:Array(deg+1).fill(0n), scale, source:'decimal', complex:false};
    }
    function lcmBig(a,b){ a=absBig(a); b=absBig(b); if(a===0n || b===0n) return 0n; return a/gcdBig(a,b)*b; }
    function complexScaledPowers(parsed, deg, prec){
      if(!parsed) return null;
      const scale = 10n ** BigInt(Math.max(0, Math.min(120, prec)));
      const L = lcmBig(parsed.re.den, parsed.im.den) || 1n;
      const ar = parsed.re.num * (L / parsed.re.den);
      const ai = parsed.im.num * (L / parsed.im.den);
      const scaledRe=[], scaledIm=[];
      let pr=1n, pi=0n, den=1n;
      for(let k=0;k<=deg;k++){
        scaledRe.push(roundDiv(scale*pr,den));
        scaledIm.push(roundDiv(scale*pi,den));
        const nr=pr*ar - pi*ai;
        const ni=pr*ai + pi*ar;
        pr=nr; pi=ni; den*=L;
      }
      return {scaled:scaledRe, scaledRe, scaledIm, scale, source:'decimal-complex', complex:scaledIm.some(x=>x!==0n)};
    }
    function doubleScaledPowers(value, deg, prec){
      if(!Number.isFinite(value)) return null; const p = Math.min(Math.max(0,prec), 17); const scale = 10n ** BigInt(p); const scaled=[];
      for(let i=0;i<=deg;i++){ const v = Math.pow(value,i); if(!Number.isFinite(v) || Math.abs(v)>1e120) return null; scaled.push(BigInt(Math.round(Number(scale)*v))); }
      return {scaled, scaledRe:scaled, scaledIm:Array(deg+1).fill(0n), scale, source:'double', complex:false};
    }
    function scaledPowersForAlgebraic(settings, parsed, val, deg, prec, cache=null){
      // v11.2.3: shared exact scaled-power cache for algebraic PSLQ/LLL passes.
      // For real finite decimal literals, prefer decimalScaledPowers() instead of
      // routing through complexScaledPowers(); complexScaledPowers() is reserved
      // for genuinely complex decimal inputs.
      const raw=(settings && (settings.normalizedRaw || settings.raw)) || '';
      const mode=parsed ? (parsed.isComplex ? 'complex' : 'real-decimal') : 'numeric';
      const key=`${raw}|${mode}|${deg}|${prec}`;
      if(cache && cache.has(key)) return cache.get(key);
      let data=null;
      if(parsed){
        data = parsed.isComplex
          ? complexScaledPowers(parsed, deg, prec)
          : (decimalScaledPowers(raw, deg, prec) || complexScaledPowers(parsed, deg, prec));
      }else{
        data = decimalScaledPowers(raw, deg, prec) || doubleScaledPowers(val, deg, prec);
      }
      if(cache) cache.set(key,data);
      return data;
    }
    function bigToFloat(x){
      if(x===0n) return 0;
      const neg=x<0n; let s=(neg?-x:x).toString();
      if(s.length<285){ const v=Number(neg?'-'+s:s); return Number.isFinite(v)?v:(neg?-Number.MAX_VALUE/16:Number.MAX_VALUE/16); }
      const lead=Number(s.slice(0,18));
      const exp=Math.min(280, s.length-18);
      const v=lead*Math.pow(10,exp);
      return neg ? -v : v;
    }
    function dotNum(a,b){ let s=0; for(let i=0;i<a.length;i++) s += bigToFloat(a[i]) * Number(b[i]); return s; }
    function lllReduce(rows, delta=0.75){
      rows = rows.map(r=>r.slice()); const n=rows.length; let B=[], mu=[], bstar=[];
      function recompute(){ B=[]; mu=Array.from({length:n},()=>Array(n).fill(0)); bstar=[]; for(let i=0;i<n;i++){ let v=rows[i].map(bigToFloat); for(let j=0;j<i;j++){ const m=B[j] ? dotNum(rows[i], bstar[j]) / B[j] : 0; mu[i][j]=Number.isFinite(m)?m:0; for(let k=0;k<v.length;k++) v[k] -= mu[i][j]*bstar[j][k]; } bstar[i]=v; B[i]=v.reduce((s,x)=>s+x*x,0); if(!Number.isFinite(B[i])) B[i]=Number.MAX_VALUE/16; } }
      recompute(); let k=1, iter=0;
      while(k<n && iter++<18000){
        for(let j=k-1;j>=0;j--){ const q=Math.round(mu[k][j]); if(q!==0 && Number.isFinite(q)){ const Q=BigInt(q); for(let c=0;c<rows[k].length;c++) rows[k][c] -= Q*rows[j][c]; recompute(); } }
        if(B[k] >= (delta - mu[k][k-1]*mu[k][k-1]) * B[k-1]) k++;
        else { const tmp=rows[k]; rows[k]=rows[k-1]; rows[k-1]=tmp; recompute(); k=Math.max(k-1,1); }
      }
      return rows;
    }

    // v11.2 constant-database-only floating LLL.  The global lllReduce()
    // remains unchanged for the other modules.  This reducer keeps the same
    // integer row operations but avoids rebuilding the entire Gram-Schmidt
    // table after every size-reduction step; only the current/local rows are
    // recomputed lazily, and all candidates are still validated by the caller.
    function constDbBigIntDigits(x){ x=x<0n?-x:x; return x.toString().length; }
    function constDbRowTooLarge(row, maxDigits=50){
      for(const x of row||[]) if(typeof x==='bigint' && constDbBigIntDigits(x)>maxDigits) return true;
      return false;
    }
    const CONSTDB_RELATION_COEFF_BOUND = 100;
    function constDbBoundedMaxHeight(maxHeight){
      return Math.max(1, Math.min(CONSTDB_RELATION_COEFF_BOUND, Number(maxHeight)||CONSTDB_RELATION_COEFF_BOUND));
    }
    function constDbPslqBits(sig, dim){
      sig=typedInputPrecisionForDouble(sig);
      dim=Math.max(2, Number(dim)||4);
      // Enough guard bits for <=100 coefficient relations, but much smaller than
      // the generic algebraic PSLQ ladder.  The residual/root checks remain the
      // acceptance gate, so this is only a fast existence probe.
      return Math.max(58, Math.min(98, Math.ceil(sig*3.322)+34+Math.max(0,dim-4)*4));
    }
    function constDbFixedVectorFromValues(vals, bits){
      if(!Array.isArray(vals) || vals.some(v=>!Number.isFinite(v) || Math.abs(v)>1e100)) return null;
      const scale=Math.pow(2,bits);
      if(!Number.isFinite(scale)) return null;
      const out=[null];
      for(const v of vals){
        const z=v*scale;
        if(!Number.isFinite(z) || Math.abs(z)>1e290) return null;
        out.push(BigInt(Math.round(z)));
      }
      return out;
    }
    function constDbFastLLLReduce(rows, delta=0.84, maxIter=6000, deadline=0, maxQuotient=1e8){
      // v11.2.2: constant-DB bounded float LLL.  Keep exact integer row
      // operations for output, but do Gram-Schmidt entirely on Float64 rows and
      // recompute the tiny basis after each operation.  This is intentionally
      // closer to the global lllReduce() semantics than the older lazy version,
      // but avoids repeated BigInt->Number conversion inside every dot product.
      rows=rows.map(r=>r.slice());
      const n=rows.length; if(n<2) return rows;
      const frows=rows.map(r=>r.map(bigToFloat));
      let B=[], mu=[], bstar=[];
      function dotF(a,b){ let s=0; for(let i=0;i<a.length;i++) s += a[i]*b[i]; return s; }
      function recompute(){
        B=[]; mu=Array.from({length:n},()=>Array(n).fill(0)); bstar=[];
        for(let i=0;i<n;i++){
          let v=frows[i].slice();
          for(let j=0;j<i;j++){
            const denom=B[j];
            const m=denom ? dotF(frows[i], bstar[j]) / denom : 0;
            mu[i][j]=Number.isFinite(m)?m:0;
            const mj=mu[i][j];
            for(let c=0;c<v.length;c++) v[c] -= mj*bstar[j][c];
          }
          bstar[i]=v;
          let norm=0; for(const x of v) norm += x*x;
          B[i]=Number.isFinite(norm) && norm>0 ? norm : Number.MAX_VALUE/16;
        }
      }
      recompute();
      let k=1, iter=0;
      while(k<n && iter++<maxIter){
        if(deadline && performance.now()>deadline) break;
        for(let j=k-1;j>=0;j--){
          const q=Math.round(mu[k][j]);
          if(!Number.isFinite(q) || Math.abs(q)>maxQuotient) return rows;
          if(q!==0){
            const Q=BigInt(q);
            for(let c=0;c<rows[k].length;c++){
              rows[k][c] -= Q*rows[j][c];
              frows[k][c] -= q*frows[j][c];
            }
            recompute();
          }
          if(deadline && performance.now()>deadline) break;
        }
        const muk=mu[k][k-1]||0;
        if(B[k] >= (delta - muk*muk) * B[k-1]) k++;
        else{
          let tmp=rows[k]; rows[k]=rows[k-1]; rows[k-1]=tmp;
          tmp=frows[k]; frows[k]=frows[k-1]; frows[k-1]=tmp;
          recompute(); k=Math.max(k-1,1);
        }
      }
      return rows;
    }


    // v6.4 exact BigInt LLL fallback.  The floating Gram-Schmidt LLL above is
    // fast, but at 10^28+ lattice scales it can miss the genuinely short vector
    // and return large low-degree rational approximants.  This version recomputes
    // Gram-Schmidt data as exact rationals, which is slower but much more reliable
    // for the small algebraic-relation lattices used here.
    function ratMake(n,d=1n){
      if(d<0n){ n=-n; d=-d; }
      if(n===0n) return {n:0n,d:1n};
      const g=gcdBig(n,d); return {n:n/g,d:d/g};
    }
    function ratSub(a,b){ return ratMake(a.n*b.d-b.n*a.d, a.d*b.d); }
    function ratMul(a,b){ return ratMake(a.n*b.n, a.d*b.d); }
    function ratDiv(a,b){ return ratMake(a.n*b.d, a.d*b.n); }
    function ratCmp(a,b){ const v=a.n*b.d-b.n*a.d; return v<0n?-1:v>0n?1:0; }
    function ratRound(a){
      const n=a.n, d=a.d;
      if(n>=0n) return (2n*n+d)/(2n*d);
      return -((2n*(-n)+d)/(2n*d));
    }
    function dotBig(a,b){ let s=0n; for(let i=0;i<a.length;i++) s += a[i]*b[i]; return s; }
    function exactLLLState(rows){
      const n=rows.length;
      const G=Array.from({length:n},()=>Array(n).fill(0n));
      for(let i=0;i<n;i++) for(let j=0;j<=i;j++){ G[i][j]=G[j][i]=dotBig(rows[i],rows[j]); }
      const mu=Array.from({length:n},()=>Array(n).fill(null));
      const d=Array(n).fill(null);
      for(let i=0;i<n;i++){
        for(let j=0;j<i;j++){
          let acc=ratMake(G[i][j]);
          for(let k=0;k<j;k++) if(mu[i][k] && mu[j][k]) acc=ratSub(acc, ratMul(ratMul(mu[i][k],mu[j][k]),d[k]));
          mu[i][j]=ratDiv(acc,d[j]);
        }
        let norm=ratMake(G[i][i]);
        for(let k=0;k<i;k++) if(mu[i][k]) norm=ratSub(norm, ratMul(ratMul(mu[i][k],mu[i][k]),d[k]));
        d[i]=norm;
      }
      return {mu,d};
    }
    function exactLLLReduce(rows, deltaNum=99n, deltaDen=100n, deadline=0){
      rows=rows.map(r=>r.slice());
      const n=rows.length;
      if(n<2) return rows;
      let st=exactLLLState(rows), k=1, iter=0;
      while(k<n && iter++<30000){
        if(deadline && performance.now()>deadline) break;
        for(let j=k-1;j>=0;j--){
          const q=ratRound(st.mu[k][j] || ratMake(0n));
          if(q!==0n){
            for(let c=0;c<rows[k].length;c++) rows[k][c] -= q*rows[j][c];
            st=exactLLLState(rows);
          }
          if(deadline && performance.now()>deadline) break;
        }
        const muk=st.mu[k][k-1] || ratMake(0n);
        const rhs=ratMul(ratSub(ratMake(deltaNum,deltaDen), ratMul(muk,muk)), st.d[k-1]);
        if(ratCmp(st.d[k],rhs)>=0) k++;
        else { const tmp=rows[k]; rows[k]=rows[k-1]; rows[k-1]=tmp; st=exactLLLState(rows); k=Math.max(k-1,1); }
      }
      return rows;
    }
    function relationLatticeReduce(basis, usePrec, deg){
      const out=[]; const seen=new Set();
      function addRows(rs){
        for(const r of rs){ const k=r.join(','); if(!seen.has(k)){ seen.add(k); out.push(r); } }
      }
      // Exact LLL follows the uploaded Fraction/Gram-Schmidt reduction idea,
      // but runs in-browser with BigInt rationals. Use it on the precision rungs
      // where high-degree algebraic relations are usually recovered; then add the
      // faster floating reducer as a broad fallback.
      if(deg<=10 && [20,28,36,50].includes(usePrec)){
        try{ addRows(exactLLLReduce(basis,99n,100n,performance.now()+(deg>=8?420:220))); }catch(e){}
      }
      if(deg<=12 && usePrec===28){
        try{ addRows(exactLLLReduce(basis,97n,100n,performance.now()+260)); }catch(e){}
      }
      try{ addRows(lllReduce(basis,0.84)); }catch(e){}
      return out;
    }
    function rationalAbsScientific(num, den, sig=4){
      num=absBig(num); den=absBig(den);
      if(num===0n) return '0';
      sig=Math.max(2,Math.min(8,sig||4));
      let e=num.toString().length-den.toString().length;
      if(e>=0){ if(num < den*(10n**BigInt(e))) e--; }
      else { if(num*(10n**BigInt(-e)) < den) e--; }
      const k=sig-1-e;
      let m = k>=0 ? (num*(10n**BigInt(k)))/den : num/(den*(10n**BigInt(-k)));
      const top=10n**BigInt(sig);
      if(m>=top){ m/=10n; e++; }
      let ms=m.toString().padStart(sig,'0').slice(0,sig);
      let mant=ms[0]+(sig>1?'.'+ms.slice(1):'');
      mant=mant.replace(/\.0+$/,'').replace(/(\.\d*?)0+$/,'$1');
      return e===0 ? mant : `${mant}*10^${e}`;
    }
    function realPolynomialResidualText(coeff, parsed){
      if(!parsed || parsed.isComplex) return null;
      const q=parsed.re; if(!q) return null;
      let rn=0n, rd=1n;
      for(let i=coeff.length-1;i>=0;i--){
        rn = rn*q.num + BigInt(coeff[i])*rd*q.den;
        rd = rd*q.den;
        const g=gcdBig(rn,rd); if(g>1n){ rn/=g; rd/=g; }
      }
      return rationalAbsScientific(rn,rd,4);
    }
    function shiftRightFloor(x,bits){ const d=1n<<BigInt(bits); return x>=0n ? x/d : -(((-x)+d-1n)/d); }
    function roundFixedMultiple(x,bits){ const half=1n<<BigInt(bits-1); const d=1n<<BigInt(bits); return shiftRightFloor(x+half,bits)*d; }
    function roundFixedInt(x,bits){ return shiftRightFloor(roundFixedMultiple(x,bits),bits); }
    function sqrtFixed(x,bits){ if(x<=0n) return 0n; return sqrtBigInt(x << BigInt(bits)); }
    function fixedFromRational(num,den,bits){
      if(den<0n){ num=-num; den=-den; }
      const scale=1n<<BigInt(bits);
      return num>=0n ? (num*scale + den/2n)/den : -(((-num)*scale + den/2n)/den);
    }
    function fixedPowersForParsedReal(parsed, deg, bits){
      if(!parsed || parsed.isComplex) return null;
      const q=parsed.re;
      const out=[null];
      let pn=1n, pd=1n;
      for(let i=0;i<=deg;i++){ out.push(fixedFromRational(pn,pd,bits)); pn*=q.num; pd*=q.den; const g=gcdBig(pn,pd); if(g>1n){ pn/=g; pd/=g; } }
      return out;
    }
    function pslqFixed(fixedX, bits, maxCoeff, maxSteps=9000, deadline=0){
      const n=fixedX.length-1; if(n<2) return null;
      const scale=1n<<BigInt(bits);
      const minx=fixedX.slice(1).reduce((m,x)=>absBig(x)<m?absBig(x):m, absBig(fixedX[1]));
      if(minx===0n) return null;
      const targetBits=Math.floor(bits*0.75);
      const tol=1n<<BigInt(Math.max(0,bits-targetBits));
      const g=sqrtFixed((4n<<BigInt(bits))/3n,bits);
      const A=Array.from({length:n+1},()=>Array(n+1).fill(0n));
      const B=Array.from({length:n+1},()=>Array(n+1).fill(0n));
      const H=Array.from({length:n+1},()=>Array(n+1).fill(0n));
      for(let i=1;i<=n;i++) for(let j=1;j<=n;j++){ A[i][j]=B[i][j]=(i===j?scale:0n); }
      const s=[null];
      for(let k=1;k<=n;k++){ let t=0n; for(let j=k;j<=n;j++) t += shiftRightFloor(fixedX[j]*fixedX[j],bits); s[k]=sqrtFixed(t,bits); }
      const t0=s[1]; if(!t0) return null;
      const y=fixedX.slice();
      for(let k=1;k<=n;k++){ y[k]=(fixedX[k]<<BigInt(bits))/t0; s[k]=(s[k]<<BigInt(bits))/t0; }
      for(let i=1;i<=n;i++){
        for(let j=i+1;j<n;j++) H[i][j]=0n;
        if(i<=n-1) H[i][i]=s[i] ? (s[i+1]<<BigInt(bits))/s[i] : 0n;
        for(let j=1;j<i;j++){ const sj=s[j]*s[j+1]; H[i][j]=sj ? (-y[i]*y[j]<<BigInt(bits))/sj : 0n; }
      }
      for(let i=2;i<=n;i++){
        for(let j=i-1;j>=1;j--){
          if(!H[j][j]) continue;
          const t=roundFixedMultiple((H[i][j]<<BigInt(bits))/H[j][j],bits);
          y[j] = y[j] + shiftRightFloor(t*y[i],bits);
          for(let k=1;k<=j;k++) H[i][k] = H[i][k] - shiftRightFloor(t*H[j][k],bits);
          for(let k=1;k<=n;k++){ A[i][k] = A[i][k] - shiftRightFloor(t*A[j][k],bits); B[k][j] = B[k][j] + shiftRightFloor(t*B[k][i],bits); }
        }
      }
      for(let rep=0; rep<maxSteps; rep++){
        if(deadline && performance.now()>deadline) return null;
        let m=1, szmax=-1n;
        for(let i=1;i<n;i++){ const h=H[i][i]; let gp=1n; for(let k=0;k<i;k++) gp=shiftRightFloor(gp*g,bits); const sz=shiftRightFloor(gp*absBig(h), bits*Math.max(0,i-1)); if(sz>szmax){ szmax=sz; m=i; } }
        [y[m],y[m+1]]=[y[m+1],y[m]];
        for(let i=1;i<=n;i++){ [H[m][i],H[m+1][i]]=[H[m+1][i],H[m][i]]; [A[m][i],A[m+1][i]]=[A[m+1][i],A[m][i]]; [B[i][m],B[i][m+1]]=[B[i][m+1],B[i][m]]; }
        if(m<=n-2){
          const tt=sqrtFixed(shiftRightFloor(H[m][m]*H[m][m] + H[m][m+1]*H[m][m+1],bits),bits);
          if(!tt) break;
          const t1=(H[m][m]<<BigInt(bits))/tt, t2=(H[m][m+1]<<BigInt(bits))/tt;
          for(let i=m;i<=n;i++){ const t3=H[i][m], t4=H[i][m+1]; H[i][m]=shiftRightFloor(t1*t3+t2*t4,bits); H[i][m+1]=shiftRightFloor(-t2*t3+t1*t4,bits); }
        }
        for(let i=m+1;i<=n;i++){
          for(let j=Math.min(i-1,m+1);j>=1;j--){
            if(!H[j][j]) break;
            const t=roundFixedMultiple((H[i][j]<<BigInt(bits))/H[j][j],bits);
            y[j] = y[j] + shiftRightFloor(t*y[i],bits);
            for(let k=1;k<=j;k++) H[i][k] = H[i][k] - shiftRightFloor(t*H[j][k],bits);
            for(let k=1;k<=n;k++){ A[i][k] = A[i][k] - shiftRightFloor(t*A[j][k],bits); B[k][j] = B[k][j] + shiftRightFloor(t*B[k][i],bits); }
          }
        }
        let bestErr=maxCoeff*scale;
        for(let i=1;i<=n;i++){
          const err=absBig(y[i]);
          if(err<tol){
            const vec=[]; let mh=0n;
            for(let j=1;j<=n;j++){ const c=roundFixedInt(B[j][i],bits); vec.push(c); if(absBig(c)>mh) mh=absBig(c); }
            if(mh>0n && mh<maxCoeff) return vec;
          }
          if(err<bestErr) bestErr=err;
        }
        let recnorm=0n; for(let i=1;i<=n;i++) for(let j=1;j<=n;j++) if(absBig(H[i][j])>recnorm) recnorm=absBig(H[i][j]);
        if(recnorm){ const norm=shiftRightFloor((1n<<BigInt(2*bits))/recnorm,bits)/100n; if(norm>=maxCoeff) break; }
      }
      return null;
    }
    function pslqAlgebraicRows(settings, maxDegree, maxHeight, limit){
      const parsed=settings.parsedComplex || parseDecimalComplex(settings.normalizedRaw);
      if(!parsed || parsed.isComplex) return [];
      const sig=parsed.precisionDigits || significantDigitCount(settings.raw);
      if(sig<10) return [];
      const bits=Math.max(90, Math.min(420, Math.ceil(sig*3.322)+56));
      const rows=[]; const seen=new Set(); const val=rationalToNumber(parsed.re);
      const algBudget=stageBudgetValueToMs(settings?.stageBudgets?.algebraicMs, 3600);
      const deadline=performance.now()+(algBudget===Infinity ? Infinity : Math.min(algBudget, 520+maxDegree*260));
      for(let deg=1; deg<=maxDegree && rows.length<limit && performance.now()<deadline; deg++){
        const fixed=fixedPowersForParsedReal(parsed,deg,bits);
        if(!fixed) continue;
        const rel=pslqFixed(fixed,bits,maxHeight+1n,5000,deadline);
        if(!rel) continue;
        const coeff=normalizeCoeffs(rel); const pd=polyDegree(coeff);
        if(pd<1 || pd>deg) continue;
        const h=coeffHeight(coeff); if(h===0n || h>maxHeight) continue;
        const key=coeff.join(','); if(seen.has(key)) continue; seen.add(key);
        const hpResText=realPolynomialResidualText(coeff, parsed);
        const root=refinePolyRoot(coeff, val); const err=Math.abs(root-val);
        const logH=Math.log10(Math.max(1,Number(h)));
        rows.push({candidate:`algebraic relation: ${polyString(coeff)}`, value:`x = ${fmtValue(root)}${hpResText ? `; |P(input)| ≈ ${hpResText}` : ''}`, err, errText:hpResText ? `|P(input)|≈${hpResText}; root≈${fmtErr(err)}` : fmtErr(err), degree:pd, height:h, residual:0n, score:logH*10000 + pd*2500 - sig*180});
      }
      return rows.sort((a,b)=>a.score-b.score || a.degree-b.degree || Number(a.height-b.height)).slice(0,limit);
    }
    function normalizeCoeffs(coeff){
      let c=coeff.slice(); while(c.length>1 && c[c.length-1]===0n) c.pop(); let g=0n; for(const x of c) g=gcdBig(g,x); if(g>1n) c=c.map(x=>x/g); if(c[c.length-1]<0n) c=c.map(x=>-x); return c;
    }
    function coeffHeight(c){ return c.reduce((m,x)=>absBig(x)>m?absBig(x):m,0n); }
    function polyDegree(c){ for(let i=c.length-1;i>=0;i--) if(c[i]!==0n) return i; return 0; }
    function polyString(c){
      const parts=[]; for(let i=c.length-1;i>=0;i--){ const a=c[i]; if(a===0n) continue; const neg=a<0n, mag=absBig(a); let term=''; if(parts.length) term += neg ? ' − ' : ' + '; else if(neg) term += '−'; if(i===0) term += mag.toString(); else { if(mag!==1n) term += mag.toString(); term += i===1 ? 'x' : `x^${i}`; } parts.push(term); }
      return (parts.join('') || '0') + ' = 0';
    }
    function polyEvalNum(c,x){ let y=0; for(let i=c.length-1;i>=0;i--) y = y*x + Number(c[i]); return y; }
    function polyDerivNum(c,x){ let y=0; for(let i=c.length-1;i>=1;i--) y = y*x + i*Number(c[i]); return y; }
    function refinePolyRoot(c,target){ let x=target; for(let i=0;i<16;i++){ const y=polyEvalNum(c,x), dy=polyDerivNum(c,x); if(!Number.isFinite(y)||!Number.isFinite(dy)||Math.abs(dy)<1e-14) break; const nx=x-y/dy; if(!Number.isFinite(nx)||Math.abs(nx)>1e100) break; if(Math.abs(nx-x)<1e-13*Math.max(1,Math.abs(x))) return nx; x=nx; } return x; }
    function cAdd(a,b){ return {re:a.re+b.re, im:a.im+b.im}; }
    function cMul(a,b){ return {re:a.re*b.re-a.im*b.im, im:a.re*b.im+a.im*b.re}; }
    function cDiv(a,b){ const d=b.re*b.re+b.im*b.im; return {re:(a.re*b.re+a.im*b.im)/d, im:(a.im*b.re-a.re*b.im)/d}; }
    function cAbs(a){ return Math.hypot(a.re,a.im); }
    function polyEvalComplexNum(c,z){ let y={re:0,im:0}; for(let i=c.length-1;i>=0;i--) y=cAdd(cMul(y,z),{re:Number(c[i]),im:0}); return y; }
    function polyDerivComplexNum(c,z){ let y={re:0,im:0}; for(let i=c.length-1;i>=1;i--) y=cAdd(cMul(y,z),{re:i*Number(c[i]),im:0}); return y; }
    function refinePolyRootComplex(c,target){
      let z={re:target.re, im:target.im};
      for(let i=0;i<24;i++){
        const y=polyEvalComplexNum(c,z), dy=polyDerivComplexNum(c,z);
        if(!Number.isFinite(y.re)||!Number.isFinite(y.im)||!Number.isFinite(dy.re)||!Number.isFinite(dy.im)||cAbs(dy)<1e-14) break;
        const step=cDiv(y,dy); const nz={re:z.re-step.re, im:z.im-step.im};
        if(!Number.isFinite(nz.re)||!Number.isFinite(nz.im)||cAbs(nz)>1e100) break;
        if(cAbs({re:nz.re-z.re, im:nz.im-z.im})<1e-13*Math.max(1,cAbs(z))) return nz;
        z=nz;
      }
      return z;
    }
    function fmtComplexApprox(z){
      if(!z || !Number.isFinite(z.re) || !Number.isFinite(z.im)) return 'complex root';
      const imSign=z.im<0 ? '−' : '+';
      return `${fmtValue(z.re)} ${imSign} ${fmtValue(Math.abs(z.im))}i`;
    }
    function modInvSmall(a,p){ a=((a%p)+p)%p; for(let x=1;x<p;x++) if((a*x)%p===1) return x; return 0; }
    function polyModNormalize(coeff,p){
      const d0=polyDegree(coeff);
      if(d0<=0) return null;
      if(Number(((coeff[d0]%BigInt(p))+BigInt(p))%BigInt(p))===0) return null;
      let f=coeff.map(x=>Number(((x%BigInt(p))+BigInt(p))%BigInt(p)));
      while(f.length>1 && f[f.length-1]===0) f.pop();
      if(f.length-1!==d0) return null;
      const inv=modInvSmall(f[f.length-1],p); if(!inv) return null;
      return f.map(x=>(x*inv)%p);
    }
    function polyDivisibleMod(f,g,p){
      let r=f.slice(); const gd=g.length-1; if(gd<1) return false;
      const inv=modInvSmall(g[gd],p); if(!inv) return false;
      for(let i=r.length-1;i>=gd;i--){
        const coef=(r[i]*inv)%p;
        if(coef){
          for(let j=0;j<=gd;j++) r[i-gd+j]=((r[i-gd+j]-coef*g[j])%p+p)%p;
        }
      }
      for(let i=0;i<gd;i++) if(r[i]%p!==0) return false;
      return true;
    }
    function hasMonicFactorMod(f,p,d){
      const total=Math.pow(p,d);
      if(total>350000) return null;
      for(let mask=0; mask<total; mask++){
        let x=mask; const g=[];
        for(let i=0;i<d;i++){ g.push(x%p); x=Math.floor(x/p); }
        g.push(1);
        if(polyDivisibleMod(f,g,p)) return true;
      }
      return false;
    }
    function irreducibleModPrime(coeff,p){
      const f=polyModNormalize(coeff,p); if(!f) return null;
      const n=f.length-1; if(n<=1) return true;
      for(let d=1; d<=Math.floor(n/2); d++){
        const has=hasMonicFactorMod(f,p,d);
        if(has===true) return false;
        if(has===null) return null;
      }
      return true;
    }
    function sqrtBigInt(n){ if(n<0n) return null; if(n<2n) return n; let x=1n << BigInt(Math.ceil(n.toString(2).length/2)); while(true){ const y=(x+n/x)>>1n; if(y>=x) return x; x=y; } }
    function isSquareBig(n){ const r=sqrtBigInt(n); return r!==null && r*r===n; }
    function divisorsBigSmall(n, limit=1000000n){
      n=absBig(n); if(n===0n || n>limit) return null;
      const out=[]; const N=Number(n);
      for(let i=1;i*i<=N;i++) if(N%i===0){ out.push(BigInt(i)); if(i*i!==N) out.push(BigInt(N/i)); }
      return out;
    }
    function evalHomogeneousAtRational(coeff,p,q){
      const n=polyDegree(coeff); let s=0n;
      for(let i=0;i<=n;i++){
        const pi=powBigCapped(absBig(p), BigInt(i), null) * (p<0n && i%2 ? -1n : 1n);
        const qi=powBigCapped(absBig(q), BigInt(n-i), null) * (q<0n && (n-i)%2 ? -1n : 1n);
        s += coeff[i]*pi*qi;
      }
      return s;
    }
    function hasRationalRoot(coeff){
      const d=polyDegree(coeff); if(d<=0) return false;
      if(coeff[0]===0n) return true;
      const pDivs=divisorsBigSmall(coeff[0]);
      const qDivs=divisorsBigSmall(coeff[d]);
      if(!pDivs || !qDivs) return false;
      const tested=new Set();
      for(const p0 of pDivs){
        for(const q0 of qDivs){
          for(const sg of [1n,-1n]){
            const p=sg*p0, q=q0;
            const key=p.toString()+'/'+q.toString(); if(tested.has(key)) continue; tested.add(key);
            if(evalHomogeneousAtRational(coeff,p,q)===0n) return true;
          }
        }
      }
      return false;
    }
    function signedDivisorsSmall(n){
      const d=divisorsBigSmall(n); if(!d) return null;
      const out=[]; for(const x of d){ out.push(x); out.push(-x); }
      return out;
    }
    function isQuarticReducible(coeff){
      if(polyDegree(coeff)!==4) return false;
      const a0=coeff[0], a1=coeff[1], a2=coeff[2], a3=coeff[3], a4=coeff[4];
      const A=signedDivisorsSmall(a4), C=signedDivisorsSmall(a0);
      if(!A || !C) return false;
      for(const a of A){
        if(a===0n || a4%a!==0n) continue;
        const d=a4/a;
        for(const c of C){
          if(c===0n || a0%c!==0n) continue;
          const f=a0/c;
          const det=d*c-a*f;
          if(det!==0n){
            const bn=a3*c-a*a1;
            const en=d*a1-f*a3;
            if(bn%det!==0n || en%det!==0n) continue;
            const b=bn/det, e=en/det;
            if(b*e + c*d + a*f === a2) return true;
          }else{
            // Rare singular linear system. Try a modest exact range so cyclotomic-like examples are still handled safely.
            for(let bi=-2000; bi<=2000; bi++){
              const b=BigInt(bi);
              const rem=a3-d*b;
              if(a===0n || rem%a!==0n) continue;
              const e=rem/a;
              if(b*f+c*e===a1 && b*e+c*d+a*f===a2) return true;
            }
          }
        }
      }
      return false;
    }
    function isIrreducibleIntegerPoly(coeff){
      const d=polyDegree(coeff); if(d<=1) return true;
      coeff=normalizeCoeffs(coeff);
      if(hasRationalRoot(coeff)) return false;
      if(d===2){ const disc=coeff[1]*coeff[1]-4n*coeff[2]*coeff[0]; return !isSquareBig(disc); }
      if(d===3) return true;
      if(d===4) return !isQuarticReducible(coeff);
      for(const p of [2,3,5,7,11,13,17,19,23,29,31]){
        const cert=irreducibleModPrime(coeff,p);
        if(cert===true) return true;
      }
      return false;
    }
    function significantDigitCount(raw){
      const parsed=parseDecimalComplex(raw);
      if(parsed) return Math.max(1, Math.min(120, parsed.precisionDigits));
      let t=String(raw||'').replace(/[+\-\.eE]/g,'').replace(/^0+/,'');
      return Math.max(1, Math.min(120, t.length || 1));
    }
    function typedInputPrecisionDigits(raw){
      // v9.3: precision belongs to the literal text the user entered.  We use
      // it as an upper bound for PSLQ/LLL scaling and as the default acceptance
      // tolerance for decimal/log/L-function comparisons.  In particular, never
      // verify a short decimal against invisible extra trailing zeroes.
      const parsed=parseDecimalComplex(raw);
      if(parsed) return Math.max(1, Math.min(120, parsed.precisionDigits));
      let s=normalizeDecimalString(raw).toLowerCase();
      if(!s) return 12;
      const m=s.match(/^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?$/);
      if(!m) return significantDigitCount(raw);
      let mant=s.replace(/^[+\-]/,'').split('e')[0].replace('.','');
      const first=mant.search(/[1-9]/);
      const n=first>=0 ? mant.length-first : 1;
      return Math.max(1, Math.min(120,n));
    }
    function typedInputPrecision(settingsOrRaw){
      const raw = typeof settingsOrRaw === 'string' ? settingsOrRaw : (settingsOrRaw?.raw || settingsOrRaw?.normalizedRaw || '');
      return typedInputPrecisionDigits(raw);
    }
    const DOUBLE_EFFECTIVE_PRECISION_DIGITS = 15;
    function typedInputPrecisionForDouble(settingsOrRaw){
      // v11.1: modules that convert the decimal input to JavaScript Number must
      // not treat a 16–20 digit literal as if those extra digits survived the
      // binary64 conversion. Exact BigInt/Decimal paths keep using the full
      // typed precision through typedInputPrecision().
      const sig=typedInputPrecision(settingsOrRaw);
      return Math.max(1, Math.min(DOUBLE_EFFECTIVE_PRECISION_DIGITS, sig));
    }
    function typedDecimalScaleDigitsForDouble(settingsOrRaw){
      return Math.max(0, Math.min(DOUBLE_EFFECTIVE_PRECISION_DIGITS, typedDecimalScaleDigits(settingsOrRaw)));
    }
    function matchToleranceDigits(sig, slack=1, maxDigits=28){
      // Returns the number of decimal digits allowed in an acceptance threshold.
      // The result is intentionally never larger than the user-supplied precision
      // minus a small slack; this prevents low-precision inputs from being treated
      // as if the omitted tail were zero.
      sig=Math.max(1, Math.min(120, Number(sig)||1));
      return Math.max(0, Math.min(maxDigits, Math.floor(sig - slack)));
    }
    function typedRelativeToleranceNumber(settingsOrRaw, multiplier=8, slack=1, maxDigits=28){
      const sig=typeof settingsOrRaw === 'number' ? settingsOrRaw : typedInputPrecision(settingsOrRaw);
      return Math.pow(10, -matchToleranceDigits(sig, slack, maxDigits)) * Math.max(1, Number(multiplier)||1);
    }
    function stageBudgetValueToMs(v, fallback, lo=0, hi=120000){
      v=Number(v);
      if(!Number.isFinite(v)) v=Number(fallback);
      if(!Number.isFinite(v)) v=0;
      if(v<=0) return Infinity;
      return Math.max(lo, Math.min(hi, v));
    }
    function riesLevelDefaultModuleBudgetMs(settingsOrLevel){
      const lvl=Number(typeof settingsOrLevel==='number' ? settingsOrLevel : (settingsOrLevel?.level || document.getElementById('level')?.value || DEFAULT_RIES_LEVEL));
      const level=Math.max(1, Math.floor(Number.isFinite(lvl) ? lvl : DEFAULT_RIES_LEVEL));
      if(level<=4) return 5000;
      if(level===5) return 10000;
      if(level===6) return 30000;
      return Math.min(45000, 30000 + (level-6)*5000);
    }
    function riesLevelModuleBudgetMs(settingsOrLevel){
      if(typeof settingsOrLevel==='object' && Object.prototype.hasOwnProperty.call(settingsOrLevel?.stageBudgets||{}, 'riesMs')) return stageBudgetValueToMs(settingsOrLevel.stageBudgets.riesMs, riesLevelDefaultModuleBudgetMs(settingsOrLevel));
      return riesLevelDefaultModuleBudgetMs(settingsOrLevel);
    }
    function typedDecimalScaleDigits(settingsOrRaw){
      const raw=typeof settingsOrRaw === 'string' ? settingsOrRaw : (settingsOrRaw?.raw || settingsOrRaw?.normalizedRaw || '');
      const s=normalizeDecimalString(raw).toLowerCase();
      const m=s.match(/^[+\-]?(?:(\d+)(?:\.(\d*))?|\.(\d+))(?:e([+\-]?\d+))?$/);
      if(!m) return typedInputPrecision(settingsOrRaw);
      const frac=(m[2]!==undefined ? m[2] : (m[3]||''));
      const exp=Number(m[4]||0);
      return Math.max(0, Math.min(120, frac.length - exp));
    }
    function isDirectDecimalInput(raw){
      // A plain decimal / finite decimal complex input may carry meaningful
      // typed precision.  Computed expressions such as sqrt(2)+1 are evaluated
      // for display/RIES target purposes, but v7 does not automatically run
      // high-precision algebraic reconstruction on them.
      return !!parseDecimalComplex(raw);
    }
    function shouldRunHighPrecisionAlgebraic(settings){
      const sig=typedInputPrecision(settings);
      return !!settings.doAlg && settings.modules?.algebraic!==false && isDirectDecimalInput(settings.raw) && sig>=20;
    }
    function shouldRunLowPrecisionAlgebraic(settings){
      const sig=typedInputPrecision(settings);
      return !!settings.doAlg && settings.modules?.algebraic!==false && isDirectDecimalInput(settings.raw) && sig>1 && sig<20;
    }

    function coeffsForExactParsedInput(parsed){
      if(!parsed) return null;
      if(rationalIsZero(parsed.im)){
        const c=[-parsed.re.num, parsed.re.den];
        return normalizeCoeffs(c);
      }
      const a=parsed.re, b=parsed.im;
      const L=lcmBig(lcmBig(a.den*a.den, b.den*b.den), a.den);
      const c0=L*(a.num*a.num)/(a.den*a.den) + L*(b.num*b.num)/(b.den*b.den);
      const c1=-(2n*a.num*(L/a.den));
      const c2=L;
      return normalizeCoeffs([c0,c1,c2]);
    }
    function algebraicRowFromCoeff(coeff, label, targetComplex){
      if(!coeff || polyDegree(coeff)<1) return null;
      let value, err;
      if(targetComplex && Math.abs(targetComplex.im)>0){
        const root=refinePolyRootComplex(coeff,targetComplex);
        err=Math.hypot(root.re-targetComplex.re, root.im-targetComplex.im);
        value=`z = ${fmtComplexApprox(root)}; |P(z)| ≈ ${fmtErr(cAbs(polyEvalComplexNum(coeff,targetComplex)))}`;
      }else{
        const x=targetComplex ? targetComplex.re : 0;
        const root=refinePolyRoot(coeff,x); err=Math.abs(root-x);
        const hpRes=realPolynomialResidualText(coeff, parseDecimalComplex(document.getElementById('target')?.value || ''));
        value=Number.isFinite(root)?`x = ${fmtValue(root)}${hpRes ? `; |P(input)| ≈ ${hpRes}` : ''}`:`P(x) = ${fmtValue(Math.abs(polyEvalNum(coeff,x)))}`;
      }
      const h=coeffHeight(coeff);
      return {candidate:`${label}: ${polyString(coeff)}`, value, err, errText:(typeof hpRes!=='undefined' && hpRes ? `|P(input)|≈${hpRes}; root≈${Number.isFinite(err)?fmtErr(err):String(err)}` : (Number.isFinite(err)?fmtErr(err):String(err))), degree:polyDegree(coeff), height:h, residual:0n, score:Math.log10(Number(h)+1)*300 + polyDegree(coeff)*20};
    }
    function exactInputAlgebraicRows(settings, maxHeight, limit){
      const parsed=settings.parsedComplex || parseDecimalComplex(settings.normalizedRaw);
      if(!parsed) return [];
      const coeff=coeffsForExactParsedInput(parsed);
      if(!coeff) return [];
      const h=coeffHeight(coeff);
      const deg=polyDegree(coeff);
      const visibleLimit=maxHeight*1000n;
      // v6.3: do not let a huge linear polynomial that merely encodes the
      // finite decimal rational crowd out genuine low-degree algebraic hits.
      if(deg===1 && h>maxHeight) return [];
      if(h>visibleLimit && deg>1) return [];
      const tc={re:rationalToNumber(parsed.re), im:rationalToNumber(parsed.im)};
      const row=algebraicRowFromCoeff(coeff, 'exact decimal input algebraic', tc);
      return row ? [row].slice(0,limit) : [];
    }

    function relationCandidates(settings, maxDegree, prec, maxHeight, limit, slack){
      const rows=[]; const seen=new Set(); const powerCache=new Map();
      const parsed=settings.parsedComplex || parseDecimalComplex(settings.normalizedRaw);
      const val=Number.isFinite(settings.target) ? settings.target : (parsed ? rationalToNumber(parsed.re) : NaN);
      const complexTarget=parsed && parsed.isComplex;
      const targetComplex=parsed ? {re:rationalToNumber(parsed.re), im:rationalToNumber(parsed.im)} : {re:val, im:0};
      const sigDigits=typedInputPrecision(settings);
      const adaptiveHeight=10n ** BigInt(Math.min(18, Math.max(2, Math.ceil(sigDigits*0.80)+2)));
      const requestedPrec=Math.max(0, Math.min(sigDigits, 120, Number(prec)||sigDigits));
      for(const r of pslqAlgebraicRows(settings, maxDegree, maxHeight, Math.max(limit,8))){
        const key=r.candidate; if(!seen.has(key)){ seen.add(key); rows.push(r); }
      }
      // Numerical LLL is far more stable at medium scales, while verification still
      // uses the exact BigInt-scaled residual for each tried precision.  Try a
      // small PSLQ-style ladder first, then the requested precision, so high
      // precision input does not accidentally hide a low-height exact relation.
      const lowSchedule = requestedPrec<20
        ? (requestedPrec<=9
          ? [Math.max(2,Math.floor(requestedPrec*0.65)), Math.max(2,requestedPrec-2), Math.max(2,requestedPrec-1), requestedPrec]
          : [6,8,10,12,14,16,18,19].filter(p=>p<=requestedPrec).concat([requestedPrec]))
        : [12,20,28,36,50,64].filter(p=>p<=Math.max(20,requestedPrec));
      // v7.2: for <20 significant digits, never test a precision larger than
      // the user actually supplied.  Earlier builds started at 8 digits, which
      // effectively padded short decimals with invisible trailing zeros.
      const precSchedule=lowSchedule.filter(p=>p>=0 && (requestedPrec>=20 || p<=requestedPrec));
      const uniquePrec=[...new Set(precSchedule)].sort((a,b)=>a-b);
      for(const usePrec of uniquePrec){
      for(let deg=1; deg<=maxDegree; deg++){
        const data = scaledPowersForAlgebraic(settings, parsed, val, deg, usePrec, powerCache);
        if(!data) continue;
        const useComplex = !!data.complex;
        const basis=[];
        for(let i=0;i<=deg;i++){
          const row=Array(deg + (useComplex?3:2)).fill(0n);
          row[i]=1n; row[deg+1]=data.scaledRe[i] || 0n;
          if(useComplex) row[deg+2]=data.scaledIm[i] || 0n;
          basis.push(row);
        }
        const red=relationLatticeReduce(basis, usePrec, deg);
        const maxResidual = 10n ** BigInt(Math.max(0, Math.ceil(usePrec/2) + slack));
        for(const r of red){
          const coeff=normalizeCoeffs(r.slice(0,deg+1)); const pd=polyDegree(coeff); if(pd===0 || pd>deg) continue;
          const h=coeffHeight(coeff); if(h===0n || h>maxHeight || h>adaptiveHeight) continue;
            let residualRe=0n, residualIm=0n;
          for(let i=0;i<coeff.length;i++){ residualRe += coeff[i]*(data.scaledRe[i] || 0n); if(useComplex) residualIm += coeff[i]*(data.scaledIm[i] || 0n); }
          let residual=useComplex ? (absBig(residualRe)>absBig(residualIm)?absBig(residualRe):absBig(residualIm)) : absBig(residualRe);
          if(residual > maxResidual) continue;
          // Re-verify every relation at the highest requested precision.  This
          // removes low-precision overfits while preserving small-height true
          // cubics/quintics from long decimal inputs.
          if(requestedPrec>usePrec){
            const verifyData = scaledPowersForAlgebraic(settings, parsed, val, pd, requestedPrec, powerCache);
            if(!verifyData) continue;
            let vRe=0n, vIm=0n;
            for(let i=0;i<coeff.length;i++){ vRe += coeff[i]*(verifyData.scaledRe[i] || 0n); if(verifyData.complex) vIm += coeff[i]*(verifyData.scaledIm[i] || 0n); }
            const verified = verifyData.complex ? (absBig(vRe)>absBig(vIm)?absBig(vRe):absBig(vIm)) : absBig(vRe);
            const requiredDigits = requestedPrec>30 ? Math.max(24, Math.ceil(requestedPrec*0.78)) : Math.max(10, Math.ceil(requestedPrec*0.56));
            const maxVerified = 10n ** BigInt(Math.max(0, requestedPrec - requiredDigits + Math.min(4,slack)));
            if(verified>maxVerified) continue;
            residual=verified;
          }
          const key=coeff.join(','); if(seen.has(key)) continue; seen.add(key);
          let value, err, fallback, hpResText=null;
          if(useComplex || complexTarget){
            const root=refinePolyRootComplex(coeff,targetComplex);
            err=Math.hypot(root.re-targetComplex.re, root.im-targetComplex.im);
            fallback=cAbs(polyEvalComplexNum(coeff,targetComplex));
            value=Number.isFinite(err) ? `z = ${fmtComplexApprox(root)}; |P(z)| ≈ ${fmtErr(fallback)}` : `|P(z)| ≈ ${fmtErr(fallback)}`;
          }else{
            const root=refinePolyRoot(coeff, val); err=Math.abs(root-val); fallback=Math.abs(polyEvalNum(coeff,val));
            const hpRes=realPolynomialResidualText(coeff, parsed); hpResText=hpRes;
            value=Number.isFinite(root)?`x = ${fmtValue(root)}${hpRes ? `; |P(input)| ≈ ${hpRes}` : ''}`:`P(x) = ${fmtValue(fallback)}`;
          }
          const hNum=Number(h);
          // v6.3 rank: first favor genuinely low algebraic degree, then small
          // coefficient height and exact scaled residual.  Previous versions
          // accidentally rewarded high degree, which made degree-7/8 overfits
          // outrank clean cubic/quintic relations.
          const logH=Math.log10(Math.max(1,hNum));
          const logR=Math.log10(1+Number(residual));
          const logE=Number.isFinite(err) ? Math.max(-30, Math.log10(err+1e-30)) : 0;
          const score=logH*10000 + pd*2500 + logR*6500 + logE*120;
          rows.push({candidate:`algebraic relation: ${polyString(coeff)}`, value, err:Number.isFinite(err)?err:fallback, errText:(hpResText ? `|P(input)|≈${hpResText}; root≈${Number.isFinite(err)?fmtErr(err):fmtErr(fallback)}` : (Number.isFinite(err)?fmtErr(err):fmtErr(fallback))), degree:pd, height:h, residual, score});
        }
      }
      }
      return rows.sort((a,b)=>a.score-b.score || a.degree-b.degree || Number(a.height-b.height) || Number(a.residual-b.residual) || a.err-b.err).slice(0,limit);
    }



    async function pslqAlgebraicRowsAsync(settings, maxDegree, maxHeight, limit, progressCb=null, shouldAbort=null){
      const parsed=settings.parsedComplex || parseDecimalComplex(settings.normalizedRaw);
      if(!parsed || parsed.isComplex) return [];
      const sig=parsed.precisionDigits || significantDigitCount(settings.raw);
      if(sig<10) return [];
      const bits=Math.max(90, Math.min(420, Math.ceil(sig*3.322)+56));
      const rows=[]; const seen=new Set(); const val=rationalToNumber(parsed.re);
      const algBudget=stageBudgetValueToMs(settings?.stageBudgets?.algebraicMs, 3600);
      const deadline=performance.now()+(algBudget===Infinity ? Infinity : Math.min(algBudget, 520+maxDegree*260));
      let lastYield=0;
      async function maybeYield(deg){
        const now=performance.now();
        if(now-lastYield<45 && now<deadline) return;
        lastYield=now;
        if(progressCb) progressCb({phase:'pslq', degree:deg, maxDegree, rows:rows.slice()});
        await yieldToUI();
        if(shouldAbort) shouldAbort();
      }
      for(let deg=1; deg<=maxDegree && rows.length<limit && performance.now()<deadline; deg++){
        await maybeYield(deg);
        const fixed=fixedPowersForParsedReal(parsed,deg,bits);
        if(!fixed) continue;
        const rel=pslqFixed(fixed,bits,maxHeight+1n,5000,deadline);
        if(!rel) continue;
        const coeff=normalizeCoeffs(rel); const pd=polyDegree(coeff);
        if(pd<1 || pd>deg) continue;
        const h=coeffHeight(coeff); if(h===0n || h>maxHeight) continue;
        const key=coeff.join(','); if(seen.has(key)) continue; seen.add(key);
        const hpResText=realPolynomialResidualText(coeff, parsed);
        const root=refinePolyRoot(coeff, val); const err=Math.abs(root-val);
        const logH=Math.log10(Math.max(1,Number(h)));
        rows.push({candidate:`algebraic relation: ${polyString(coeff)}`, value:`x = ${fmtValue(root)}${hpResText ? `; |P(input)| ≈ ${hpResText}` : ''}`, err, errText:hpResText ? `|P(input)|≈${hpResText}; root≈${fmtErr(err)}` : fmtErr(err), degree:pd, height:h, residual:0n, score:logH*10000 + pd*2500 - sig*180});
        await maybeYield(deg);
      }
      return rows.sort((a,b)=>a.score-b.score || a.degree-b.degree || Number(a.height-b.height)).slice(0,limit);
    }

    async function relationCandidatesAsync(settings, maxDegree, prec, maxHeight, limit, slack, progressCb=null, shouldAbort=null){
      // v11.2.3 cooperative version of relationCandidates() for the low-
      // precision algebraic pass. It skips irreducibility filtering, caches exact
      // scaled powers, and yields between PSLQ/LLL batches so progress and the SO(4) tesseract stay responsive.
      const rows=[]; const seen=new Set(); const powerCache=new Map();
      const parsed=settings.parsedComplex || parseDecimalComplex(settings.normalizedRaw);
      const val=Number.isFinite(settings.target) ? settings.target : (parsed ? rationalToNumber(parsed.re) : NaN);
      const complexTarget=parsed && parsed.isComplex;
      const targetComplex=parsed ? {re:rationalToNumber(parsed.re), im:rationalToNumber(parsed.im)} : {re:val, im:0};
      const sigDigits=typedInputPrecision(settings);
      const adaptiveHeight=10n ** BigInt(Math.min(18, Math.max(2, Math.ceil(sigDigits*0.80)+2)));
      const requestedPrec=Math.max(0, Math.min(sigDigits, 120, Number(prec)||sigDigits));
      const algOverallBudget=stageBudgetValueToMs(settings?.stageBudgets?.algebraicMs, 3600);
      const algOverallDeadline=performance.now()+algOverallBudget;
      let lastYield=0;
      async function maybeYield(phase, frac){
        const now=performance.now();
        if(now-lastYield<45) return;
        lastYield=now;
        if(progressCb) progressCb({phase, frac:Math.max(0,Math.min(1,frac||0)), rows:rows.slice()});
        await yieldToUI();
        if(shouldAbort) shouldAbort();
      }
      await maybeYield('pslq start', .02);
      const pslqRows=await pslqAlgebraicRowsAsync(settings, maxDegree, maxHeight, Math.max(limit,8), info=>{
        if(progressCb) progressCb({phase:'pslq', frac:0.02 + 0.18*((info.degree||0)/Math.max(1,maxDegree)), rows:rows.concat(info.rows||[])});
      }, shouldAbort);
      for(const r of pslqRows){ const key=r.candidate; if(!seen.has(key)){ seen.add(key); rows.push(r); } }
      await maybeYield('lll start', .22);
      // v11.4.2: test only the current typed input precision.
      // Older versions swept several lower/guard precisions; that helped recall
      // but made the final low-precision algebraic pass much slower.
      const uniquePrec=[Math.max(1, requestedPrec)];
      const totalBatches=Math.max(1, uniquePrec.length*maxDegree);
      let batch=0;
      for(let pi=0; pi<uniquePrec.length && performance.now()<=algOverallDeadline; pi++){
        const usePrec=uniquePrec[pi];
        for(let deg=1; deg<=maxDegree && performance.now()<=algOverallDeadline; deg++){
          batch++;
          await maybeYield(`LLL p=${usePrec}, d=${deg}`, .22 + .76*(batch-1)/totalBatches);
          const data = scaledPowersForAlgebraic(settings, parsed, val, deg, usePrec, powerCache);
          if(!data) continue;
          const useComplex = !!data.complex;
          const basis=[];
          for(let i=0;i<=deg;i++){
            const row=Array(deg + (useComplex?3:2)).fill(0n);
            row[i]=1n; row[deg+1]=data.scaledRe[i] || 0n;
            if(useComplex) row[deg+2]=data.scaledIm[i] || 0n;
            basis.push(row);
          }
          const red=relationLatticeReduce(basis, usePrec, deg);
          const maxResidual = 10n ** BigInt(Math.max(0, Math.ceil(usePrec/2) + slack));
          for(const r of red){
            const coeff=normalizeCoeffs(r.slice(0,deg+1)); const pd=polyDegree(coeff); if(pd===0 || pd>deg) continue;
            const h=coeffHeight(coeff); if(h===0n || h>maxHeight || h>adaptiveHeight) continue;
                let residualRe=0n, residualIm=0n;
            for(let i=0;i<coeff.length;i++){ residualRe += coeff[i]*(data.scaledRe[i] || 0n); if(useComplex) residualIm += coeff[i]*(data.scaledIm[i] || 0n); }
            let residual=useComplex ? (absBig(residualRe)>absBig(residualIm)?absBig(residualRe):absBig(residualIm)) : absBig(residualRe);
            if(residual > maxResidual) continue;
            if(requestedPrec>usePrec){
              const verifyData = scaledPowersForAlgebraic(settings, parsed, val, pd, requestedPrec, powerCache);
              if(!verifyData) continue;
              let vRe=0n, vIm=0n;
              for(let i=0;i<coeff.length;i++){ vRe += coeff[i]*(verifyData.scaledRe[i] || 0n); if(verifyData.complex) vIm += coeff[i]*(verifyData.scaledIm[i] || 0n); }
              const verified = verifyData.complex ? (absBig(vRe)>absBig(vIm)?absBig(vRe):absBig(vIm)) : absBig(vRe);
              const requiredDigits = requestedPrec>30 ? Math.max(24, Math.ceil(requestedPrec*0.78)) : Math.max(10, Math.ceil(requestedPrec*0.56));
              const maxVerified = 10n ** BigInt(Math.max(0, requestedPrec - requiredDigits + Math.min(4,slack)));
              if(verified>maxVerified) continue;
              residual=verified;
            }
            const key=coeff.join(','); if(seen.has(key)) continue; seen.add(key);
            let value, err, fallback, hpResText=null;
            if(useComplex || complexTarget){
              const root=refinePolyRootComplex(coeff,targetComplex);
              err=Math.hypot(root.re-targetComplex.re, root.im-targetComplex.im);
              fallback=cAbs(polyEvalComplexNum(coeff,targetComplex));
              value=Number.isFinite(err) ? `z = ${fmtComplexApprox(root)}; |P(z)| ≈ ${fmtErr(fallback)}` : `|P(z)| ≈ ${fmtErr(fallback)}`;
            }else{
              const root=refinePolyRoot(coeff, val); err=Math.abs(root-val); fallback=Math.abs(polyEvalNum(coeff,val));
              const hpRes=realPolynomialResidualText(coeff, parsed); hpResText=hpRes;
              value=Number.isFinite(root)?`x = ${fmtValue(root)}${hpRes ? `; |P(input)| ≈ ${hpRes}` : ''}`:`P(x) = ${fmtValue(fallback)}`;
            }
            const hNum=Number(h);
            const logH=Math.log10(Math.max(1,hNum));
            const logR=Math.log10(1+Number(residual));
            const logE=Number.isFinite(err) ? Math.max(-30, Math.log10(err+1e-30)) : 0;
            const score=logH*10000 + pd*2500 + logR*6500 + logE*120;
            rows.push({candidate:`algebraic relation: ${polyString(coeff)}`, value, err:Number.isFinite(err)?err:fallback, errText:(hpResText ? `|P(input)|≈${hpResText}; root≈${Number.isFinite(err)?fmtErr(err):fmtErr(fallback)}` : (Number.isFinite(err)?fmtErr(err):fmtErr(fallback))), degree:pd, height:h, residual, score});
            if(rows.length>=Math.max(limit,10)*2) await maybeYield('candidate verification', .22 + .76*batch/totalBatches);
          }
        }
      }
      await maybeYield('final algebraic ranking', .99);
      return rows.sort((a,b)=>a.score-b.score || a.degree-b.degree || Number(a.height-b.height) || Number(a.residual-b.residual) || a.err-b.err).slice(0,limit);
    }

    const logConstants = [
      {id:'one', label:'1', value:1, default:true, kind:'one', product:'e'},
      {id:'log2', label:'log(2)', value:Math.log(2), default:true, kind:'log', product:'2'},
      {id:'log3', label:'log(3)', value:Math.log(3), default:true, kind:'log', product:'3'},
      {id:'log5', label:'log(5)', value:Math.log(5), default:true, kind:'log', product:'5'},
      {id:'pi', label:'π', value:Math.PI, default:true, kind:'raw', product:'π'},
      {id:'logpi', label:'log(π)', value:Math.log(Math.PI), default:true, kind:'log', product:'π'},
      {id:'loglogpi', label:'log(log π)', value:Math.log(Math.log(Math.PI)), default:true, kind:'log', product:'log(π)'},
      {id:'loglog2', label:'log(log 2)', value:Math.log(Math.log(2)), default:true, kind:'log', product:'log(2)'},
      {id:'loglog3', label:'log(log 3)', value:Math.log(Math.log(3)), default:true, kind:'log', product:'log(3)'},
      {id:'loggamma16', label:'log Γ(1/6)', value:1.7167334350782405, default:false, kind:'log', product:'Γ(1/6)'},
      {id:'log7', label:'log(7)', value:Math.log(7), default:false, kind:'log', product:'7'},
      {id:'log11', label:'log(11)', value:Math.log(11), default:false, kind:'log', product:'11'},
      {id:'e', label:'e', value:Math.E, default:false, kind:'raw', product:'e'},
      {id:'loggamma13', label:'log Γ(1/3)', value:0.9854206469277671, default:false, kind:'log', product:'Γ(1/3)'},
      {id:'loggamma14', label:'log Γ(1/4)', value:1.2880225246980775, default:false, kind:'log', product:'Γ(1/4)'},
      {id:'eulergamma', label:'Euler γ', value:0.5772156649015329, default:false, kind:'raw', product:'γ'},
      {id:'logeulergamma', label:'log(γ)', value:Math.log(0.5772156649015329), default:false, kind:'log', product:'γ'},
      {id:'logG', label:'log(G)', value:Math.log(0.91596559417721901505), default:false, kind:'log', product:'G'},
      {id:'logzeta3', label:'log ζ(3)', value:Math.log(1.2020569031595942854), default:false, kind:'log', product:'ζ(3)'},
      {id:'logzeta5', label:'log ζ(5)', value:Math.log(1.0369277551433699263), default:false, kind:'log', product:'ζ(5)'},
      {id:'logphi', label:'log(φ)', value:Math.log((1+Math.sqrt(5))/2), default:false, kind:'log', product:'φ'},
      {id:'logA', label:'log(A)', value:Math.log(1.28242712910062263687), default:false, kind:'log', product:'A'}
    ];
    function fillLogBasis(){
      const def=document.getElementById('defaultLogBasis'), extra=document.getElementById('extraLogBasis'); def.innerHTML=''; extra.innerHTML='';
      for(const c of logConstants){ const lab=document.createElement('label'); lab.className='option-chip'; lab.innerHTML=`<input type="checkbox" data-logconst="${c.id}" ${c.default?'checked':''}> ${c.label}`; (c.default?def:extra).appendChild(lab); }
    }
    function selectedLogConstants(){ const ids=new Set([...document.querySelectorAll('[data-logconst]:checked')].map(x=>x.dataset.logconst)); return logConstants.filter(c=>ids.has(c.id)); }
    function normalizeVector(v){ let c=v.slice(); let g=0n; for(const x of c) g=gcdBig(g,x); if(g>1n) c=c.map(x=>x/g); if(c[0]<0n) c=c.map(x=>-x); return c; }
    function linearRelations(values, labels, prec, maxHeight, limit, slack, deadlineMs=160){
      const effectivePrec = Math.max(1, Math.min(17, Number(prec)||1));
      const scaleNum = Math.pow(10, effectivePrec); const scale = BigInt(Math.round(scaleNum));
      const basis=[]; for(let i=0;i<values.length;i++){ const row=Array(values.length+1).fill(0n); row[i]=1n; row[values.length]=BigInt(Math.round(values[i]*scaleNum)); basis.push(row); }
      const red=[]; const seenRows=new Set();
      function addReduced(rs){ for(const r of rs){ const k=r.join(','); if(!seenRows.has(k)){ seenRows.add(k); red.push(r); } } }
      try{ addReduced(exactLLLReduce(basis,99n,100n,performance.now()+Math.max(12, Number(deadlineMs)||160))); }catch(e){}
      try{ addReduced(lllReduce(basis,0.82)); }catch(e){}
      const rows=[]; const seen=new Set(); const maxResidual=10n ** BigInt(Math.max(0, slack));
      for(const r of red){ const coeff=normalizeVector(r.slice(0,values.length)); if(coeff[0]===0n || !coeff.slice(1).some(x=>x!==0n)) continue; const h=coeffHeight(coeff); if(h===0n || h>maxHeight) continue; let residual=0n; for(let i=0;i<coeff.length;i++) residual += coeff[i]*BigInt(Math.round(values[i]*scaleNum)); residual=absBig(residual); if(residual>maxResidual) continue; const key=coeff.join(','); if(seen.has(key)) continue; seen.add(key); const a0=Number(coeff[0]); let rhs=0; for(let i=1;i<coeff.length;i++) rhs -= Number(coeff[i])*values[i]/a0; rows.push({coeff, rhs, err:Math.abs(values[0]-rhs), height:h, residual}); }
      return rows.sort((a,b)=>a.err-b.err || Number(a.height-b.height)).slice(0,limit);
    }
    function rationalString(num, den){
      if(num===0n) return '0';
      const neg = (num<0n) !== (den<0n);
      let a=absBig(num), d=absBig(den); const g=gcdBig(a,d); a/=g; d/=g;
      const core = d===1n ? a.toString() : `${a}/${d}`;
      return neg ? `-${core}` : core;
    }
    function logFactor(meta, num, den){
      if(num===0n) return '';
      const exp = rationalString(num, den);
      if(meta.kind === 'log') return exp === '1' ? meta.product : `${meta.product}^(${exp})`;
      if(meta.kind === 'one') return exp === '1' ? 'e' : `e^(${exp})`;
      if(exp === '1') return `exp(${meta.product})`;
      if(exp === '-1') return `exp(-${meta.product})`;
      return `exp(${meta.product}*(${exp}))`;
    }
    function logProductBaseLatex(meta){
      const id=String(meta?.id||'');
      const map={
        log2:'2', log3:'3', log5:'5', pi:String.raw`\pi`, logpi:String.raw`\pi`,
        loglogpi:String.raw`\log\pi`, loglog2:String.raw`\log 2`, loglog3:String.raw`\log 3`,
        loggamma16:String.raw`\Gamma(1/6)`, log7:'7', log11:'11', e:'e',
        loggamma13:String.raw`\Gamma(1/3)`, loggamma14:String.raw`\Gamma(1/4)`,
        eulergamma:String.raw`\gamma`, logeulergamma:String.raw`\gamma`, logG:'G',
        logzeta3:String.raw`\zeta(3)`, logzeta5:String.raw`\zeta(5)`,
        logphi:String.raw`\varphi`, logA:'A', one:'e'
      };
      if(map[id]) return map[id];
      return exprToLatex(String(meta?.product || meta?.label || id || 'c'));
    }
    function logLatexNeedsGroupedPower(base){
      return /[+\-\s]|\\log|\\Gamma|\\zeta/.test(String(base||''));
    }
    function logFactorLatex(meta, num, den){
      if(num===0n) return '';
      const exp=rationalString(num, den);
      const base=logProductBaseLatex(meta);
      if(meta.kind === 'log' || meta.kind === 'one'){
        if(exp==='1') return base;
        const b=logLatexNeedsGroupedPower(base) ? `\\left(${base}\\right)` : base;
        return `${b}^{${exp}}`;
      }
      if(exp==='1') return `\\exp\\left(${base}\\right)`;
      if(exp==='-1') return `\\exp\\left(-${base}\\right)`;
      const coeff = exp.includes('/') ? `\\frac{${exp.split('/')[0]}}{${exp.split('/')[1]}}` : exp;
      return `\\exp\\left(${coeff}\\,${base}\\right)`;
    }
    function logProductLatex(rel, consts){
      const den=rel.coeff[0]; const parts=[];
      for(let i=1;i<rel.coeff.length;i++){
        const factor=logFactorLatex(consts[i-1], -rel.coeff[i], den);
        if(factor) parts.push(factor);
      }
      return parts.join('\\,') || '1';
    }
    function logProductString(rel, consts){
      const den=rel.coeff[0]; const parts=[];
      for(let i=1;i<rel.coeff.length;i++){
        const factor = logFactor(consts[i-1], -rel.coeff[i], den);
        if(factor) parts.push(factor);
      }
      return parts.join(' * ') || '1';
    }

    function directSparseLogRows(target, consts, settings, basisNote=''){
      if(!Number.isFinite(target) || target===0 || !Array.isArray(consts) || !consts.length) return [];
      const y=Math.log(Math.abs(target));
      const sig=typedInputPrecisionForDouble(settings || String(target));
      const tol=typedRelativeToleranceNumber(sig, 10, 1, 13) * Math.max(1, Math.abs(y));
      const maxCoeff=Math.max(6, Math.min(14, 8 + Math.max(0, logContinueEffort(settings))*2));
      const rows=[]; const seen=new Set();
      const terms=[];
      for(let i=0;i<consts.length;i++){
        for(let c=-maxCoeff;c<=maxCoeff;c++){
          if(c===0) continue;
          terms.push({idx:i, c, sum:c*consts[i].value, h:Math.abs(c)});
        }
      }
      terms.sort((a,b)=>a.sum-b.sum || a.idx-b.idx || a.c-b.c);
      function lowerBound(arr, value){ let lo=0, hi=arr.length; while(lo<hi){ const mid=(lo+hi)>>1; if(arr[mid].sum<value) lo=mid+1; else hi=mid; } return lo; }
      function tryTerms(list){
        const used=new Set(); let sum=0, h=0;
        for(const t of list){ if(used.has(t.idx)) return; used.add(t.idx); sum+=t.sum; h+=Math.abs(t.c); }
        const err=Math.abs(y-sum); if(err>tol) return;
        const vec=Array(consts.length+1).fill(0n); vec[0]=1n;
        for(const t of list) vec[t.idx+1]=BigInt(-t.c);
        const key=vec.join(','); if(seen.has(key)) return; seen.add(key);
        rows.push(buildLogRelationRow(target,{coeff:vec,rhs:sum,err,height:BigInt(h),residual:0n}, consts, basisNote ? `direct sparse; ${basisNote}` : 'direct sparse'));
      }
      for(const a of terms) tryTerms([a]);
      for(let i=0;i<terms.length;i++){
        const a=terms[i];
        for(let j=i+1;j<terms.length;j++){
          const b=terms[j]; if(a.idx===b.idx) continue;
          const sum2=a.sum+b.sum;
          if(Math.abs(y-sum2)<=tol) tryTerms([a,b]);
          const want=y-sum2;
          let k=lowerBound(terms, want-tol)-2;
          const kend=Math.min(terms.length, lowerBound(terms, want+tol)+3);
          for(; k<kend; k++){
            if(k<0 || k===i || k===j) continue;
            const c=terms[k]; if(c.idx===a.idx || c.idx===b.idx) continue;
            tryTerms([a,b,c]);
          }
        }
      }
      return rows.sort((a,b)=>(a.score??1e9)-(b.score??1e9) || (a.err||0)-(b.err||0)).slice(0, Math.max(4, Math.min(12, Number(settings?.limit||5))));
    }
    function logContinueEffort(settings){
      const lvl=Number(settings?.level || document.getElementById('level')?.value || DEFAULT_RIES_LEVEL);
      return Math.max(0, Math.min(5, Math.floor(lvl - Number(DEFAULT_RIES_LEVEL))));
    }
    function defaultLogBasisForContinuation(){
      const checked=new Set([...document.querySelectorAll('[data-logconst]:checked')].map(x=>x.dataset.logconst));
      const selectedDefaults=logConstants.filter(c=>c.default && checked.has(c.id));
      return selectedDefaults.length ? selectedDefaults : logConstants.filter(c=>c.default);
    }
    const logContinuationRemovalOrder=['loglog3','loglogpi','log5','loglog2'];
    function* logCombinations(items, k, deadline, maxCount=Infinity){
      if(k<=0){ yield []; return; }
      const n=items.length;
      if(k>n) return;
      const combo=[];
      let count=0;
      function* rec(start, left){
        if(performance.now()>deadline || count>=maxCount) return;
        if(left===0){ count++; yield combo.slice(); return; }
        for(let i=start;i<=n-left;i++){
          combo.push(items[i]);
          yield* rec(i+1,left-1);
          combo.pop();
          if(performance.now()>deadline || count>=maxCount) return;
        }
      }
      yield* rec(0,k);
    }
    function logRelationPrettyScore(rel, product){
      const nonZero=rel.coeff.slice(1).filter(x=>x!==0n).length;
      const height=Number(rel.height || coeffHeight(rel.coeff));
      const productLen=String(product||'').length;
      const err=Number.isFinite(rel.err) ? Math.abs(rel.err) : 1;
      // v9.2: log|c| matches are most useful when they are sparse and low
      // height. Once the residual is well inside typed precision, do not let a
      // many-term LLL artefact outrank a simple product such as 6·π^5.
      const accurate = err < 1e-9;
      return (accurate ? 0 : Math.max(0, -Math.log10(err+1e-30))*-35) + nonZero*420 + height*7 + productLen*.45 + Math.log10(err+1e-30)*18;
    }
    function linearCombinationString(rel, consts){
      const den=rel.coeff[0]; const parts=[];
      for(let i=1;i<rel.coeff.length;i++){
        const num=-rel.coeff[i]; if(num===0n) continue;
        const coeff=rationalString(num, den);
        const label=consts[i-1]?.label || `c${i}`;
        if(coeff==='1') parts.push(label);
        else if(coeff==='-1') parts.push(`-${label}`);
        else parts.push(`${coeff}*${label}`);
      }
      return parts.join(' + ').replace(/\+ -/g,'- ') || '0';
    }
    function logLinearConstantLatex(meta){
      const id=String(meta?.id||'');
      const map={
        one:'1', log2:String.raw`\log 2`, log3:String.raw`\log 3`, log5:String.raw`\log 5`, pi:String.raw`\pi`, logpi:String.raw`\log \pi`,
        loglogpi:String.raw`\log\!\left(\log \pi\right)`, loglog2:String.raw`\log\!\left(\log 2\right)`, loglog3:String.raw`\log\!\left(\log 3\right)`,
        loggamma16:String.raw`\log\Gamma(1/6)`, log7:String.raw`\log 7`, log11:String.raw`\log 11`, e:'e',
        loggamma13:String.raw`\log\Gamma(1/3)`, loggamma14:String.raw`\log\Gamma(1/4)`,
        eulergamma:String.raw`\gamma`, logeulergamma:String.raw`\log\gamma`, logG:String.raw`\log G`,
        logzeta3:String.raw`\log\zeta(3)`, logzeta5:String.raw`\log\zeta(5)`,
        logphi:String.raw`\log\varphi`, logA:String.raw`\log A`
      };
      if(map[id]) return map[id];
      return exprToLatex(String(meta?.label || meta?.product || id || 'c'));
    }
    function linearCombinationLatex(rel, consts){
      const den=rel.coeff[0]; const parts=[];
      for(let i=1;i<rel.coeff.length;i++){
        const num=-rel.coeff[i]; if(num===0n) continue;
        const coeff=rationalString(num, den);
        const base=logLinearConstantLatex(consts[i-1] || {});
        const texCoeff=coeff.includes('/') ? `\\frac{${coeff.split('/')[0]}}{${coeff.split('/')[1]}}` : coeff;
        if(coeff==='1') parts.push(base);
        else if(coeff==='-1') parts.push(`-${base}`);
        else parts.push(`${texCoeff}\\,${base}`);
      }
      return parts.join(' + ').replace(/\+ -/g,'- ') || '0';
    }
    function buildLinearRelationRow(targetValue, rel, consts, variant, basisNote=''){
      const rhs=linearCombinationString(rel,consts);
      const rhsLatex=linearCombinationLatex(rel,consts);
      const h=rel.height || coeffHeight(rel.coeff);
      const nonZero=rel.coeff.slice(1).filter(x=>x!==0n).length;
      const lhs=variant?.label || 'x';
      const lhsLatex=variant?.latex || 'x';
      const note=basisNote ? `; ${basisNote}` : '';
      const latex=`${lhsLatex} \\approx ${rhsLatex}`;
      return {candidate:`log-combination relation: ${lhs} ≈ ${rhs}`, latex, copyLatex:latex, value:`${lhs} = ${fmtValue(targetValue)}; rhs = ${fmtValue(rel.rhs)}; terms ${nonZero}; height ${h.toString()}${note}`, err:rel.err, height:h, terms:nonZero, score:logRelationPrettyScore(rel, rhs)+20};
    }
    function buildLogRelationRow(target, rel, consts, basisNote=''){
      const left = target < 0 ? '−x' : 'x';
      const leftLatex = target < 0 ? '-x' : 'x';
      const product=logProductString(rel, consts);
      const productLatex=logProductLatex(rel, consts);
      const productValue=Math.exp(rel.rhs);
      const nonZero=rel.coeff.slice(1).filter(x=>x!==0n).length;
      const h=rel.height || coeffHeight(rel.coeff);
      const note=basisNote ? `; ${basisNote}` : '';
      const latex=`${leftLatex} \\approx ${productLatex}`;
      return {
        candidate:`log|c| linear relation: ${left} ≈ ${product}`,
        latex,
        copyLatex:latex,
        value:`${left} = ${fmtValue(Math.abs(target))}; product = ${fmtValue(productValue)}; terms ${nonZero}; height ${h.toString()}${note}`,
        err:rel.err,
        height:h,
        terms:nonZero,
        score:logRelationPrettyScore(rel, product)
      };
    }
    function logBasisSignature(consts){ return consts.map(c=>c.id).join('|'); }
    function logContinuationBasisRows(target, settings, prec, maxH, slack){
      const effort=logContinueEffort(settings);
      if(effort<=0) return [];
      const baseDefaults=defaultLogBasisForContinuation();
      const optionalPriority=['loggamma13','loggamma14','loggamma16','log11','e','log7','logzeta3','logG','logA','logphi','logeulergamma','eulergamma','logzeta5'];
      const priorityIndex=id=>{ const i=optionalPriority.indexOf(id); return i<0 ? 999 : i; };
      const optional=logConstants.filter(c=>!c.default).sort((a,b)=>priorityIndex(a.id)-priorityIndex(b.id) || a.id.localeCompare(b.id));
      const optTake=Math.max(1, Math.min(effort, optional.length));
      const moduleBudget=riesLevelModuleBudgetMs(settings);
      const perCallMs=Math.max([0,48,42,34,28,24][Math.min(5,optTake)] || 24, Math.min(900, Math.floor(moduleBudget/32)));
      const totalBudget=Math.max([0,900,1250,1650,2050,2450][Math.min(5,optTake)] || 2450, moduleBudget);
      const maxCombos=Math.max([0,80,160,220,280,340][Math.min(5,optTake)] || 340, optTake===1 ? optional.length : Math.min(900, optional.length*optional.length));
      const started=performance.now();
      const deadline=started+totalBudget;
      const rows=[];
      const seenBasis=new Set();
      let combosSeen=0;
      for(const optCombo of logCombinations(optional, optTake, deadline, maxCombos)){
        combosSeen++;
        for(let drop=0; drop<=logContinuationRemovalOrder.length; drop++){
          if(performance.now()>deadline) break;
          const dropIds=new Set(logContinuationRemovalOrder.slice(0,drop));
          const consts=baseDefaults.filter(c=>!dropIds.has(c.id)).concat(optCombo);
          const sig=logBasisSignature(consts);
          if(seenBasis.has(sig)) continue;
          seenBasis.add(sig);
          const added=optCombo.map(c=>c.label).join(', ');
          const removed=logContinuationRemovalOrder.slice(0,drop).map(id=>logConstants.find(c=>c.id===id)?.label || id).join(', ');
          const basisNote=`auto basis +${added || 'none'}${removed ? `; removed ${removed}` : ''}`;
          rows.push(...directSparseLogRows(target, consts, settings, basisNote));
          const values=[Math.log(Math.abs(target)), ...consts.map(c=>c.value)];
          const labels=['log|x|', ...consts.map(c=>c.label)];
          const rels=linearRelations(values, labels, prec, maxH, 3, slack, perCallMs);
          if(!rels.length) continue;
          for(const rel of rels){
            const nonZero=rel.coeff.slice(1).filter(x=>x!==0n).length;
            const h=Number(rel.height || coeffHeight(rel.coeff));
            const pretty = nonZero<=4 || h<=48 || rel.residual<=10n;
            if(pretty || rows.length<Math.max(6, Number(settings.limit)||5)) rows.push(buildLogRelationRow(target, rel, consts, basisNote));
          }
        }
        if(performance.now()>deadline) break;
      }
      const map=new Map();
      for(const r of rows){
        const key=normalizeResultTextKey(r.candidate);
        if(!map.has(key) || (r.score??1e9)<(map.get(key).score??1e9)) map.set(key,r);
      }
      const limit=Math.max(6, Math.min(30, (Number(settings.limit)||5)*3));
      return [...map.values()].sort((a,b)=>(a.score??1e9)-(b.score??1e9) || (a.err||1)-(b.err||1)).slice(0,limit).map(r=>{
        r.value += `; enumerated ${combosSeen} optional combination${combosSeen===1?'':'s'}`;
        return r;
      });
    }
    function logRelationRows(target, settings){
      if(!Number.isFinite(target) || target===0) return [];
      let maxH; try{ maxH=BigInt(document.getElementById('logHeight').value.trim() || '400'); }catch(e){ maxH=400n; }
      const autoPrec=typedInputPrecision(settings);
      const prec=Math.max(1, Math.min(17, autoPrec));
      const slack=Math.max(1, Math.min(4, autoPrec<=10 ? 1 : 2));
      const consts=selectedLogConstants(); if(!consts.length) return [];
      const opts=settings?.logOptions || {targetLogAbs:true,targetRaw:true,targetLogLogAbs:true};
      const rows=[];
      const budget=stageBudgetValueToMs(settings?.stageBudgets?.logMs, riesLevelDefaultModuleBudgetMs(settings));
      const perVariantBudget=Math.max(80, Math.floor(budget / Math.max(1, [opts.targetLogAbs, opts.targetRaw, opts.targetLogLogAbs].filter(x=>x!==false).length)));
      if(opts.targetLogAbs!==false){
        const y=Math.log(Math.abs(target));
        const values=[y, ...consts.map(c=>c.value)]; const labels=['log|x|', ...consts.map(c=>c.label)];
        const rels=linearRelations(values, labels, prec, maxH, Math.min(2, settings.limit || 2), slack, perVariantBudget);
        rows.push(...directSparseLogRows(target, consts, settings), ...rels.map(rel=>buildLogRelationRow(target, rel, consts, 'target log|x|')));
        if(logContinueEffort(settings)>0) rows.push(...logContinuationBasisRows(target, settings, prec, maxH, slack));
      }
      if(opts.targetRaw!==false){
        const values=[target, ...consts.map(c=>c.value)]; const labels=['x', ...consts.map(c=>c.label)];
        const rels=linearRelations(values, labels, prec, maxH, Math.min(2, settings.limit || 2), slack, perVariantBudget);
        rows.push(...rels.map(rel=>buildLinearRelationRow(target, rel, consts, {label:'x',latex:'x'}, 'target x')));
      }
      if(opts.targetLogLogAbs!==false && Math.abs(target)>0 && Math.log(Math.abs(target))!==0){
        const z=Math.log(Math.abs(Math.log(Math.abs(target))));
        if(Number.isFinite(z)){
          const values=[z, ...consts.map(c=>c.value)]; const labels=['log|log|x||', ...consts.map(c=>c.label)];
          const rels=linearRelations(values, labels, prec, maxH, Math.min(2, settings.limit || 2), slack, perVariantBudget);
          rows.push(...rels.map(rel=>buildLinearRelationRow(z, rel, consts, {label:'log|log|x||',latex:'\\log\\!\\left|\\log|x|\\right|'}, 'target log|log|x||')));
        }
      }
      const map=new Map();
      for(const r of rows){ const k=normalizeResultTextKey(r.candidate); if(!map.has(k) || (r.score??1e9)<(map.get(k).score??1e9)) map.set(k,r); }
      const lim=Math.max(6, Math.min(30, Number(settings?.moduleLimits?.log || settings.limit || 5)*3));
      return [...map.values()].sort((a,b)=>(a.score??1e9)-(b.score??1e9) || (a.err||0)-(b.err||0)).slice(0, lim);
    }





    // v11.2.4 generalized Möbius / fractional-linear matcher for decimal inputs.
    // For a small typed decimal x, try x, exp(x), and log(|x|) against
    //     (a A + b B (+ c C)) / (d A + e B (+ f C) + g)
    // over the constant catalogue below.  The LLL lattice is the linear
    // relation P - y(Q + g) = 0, i.e. columns
    // [A,B,C,-yA,-yB,-yC,-y].  Verification is done against the visible
    // formula value before a row is returned.
    const mobiusConstants = [
      {id:'one', name:'1', latex:'1', value:1},
      {id:'pi', name:'π', latex:'\\pi', value:Math.PI},
      {id:'e', name:'e', latex:'e', value:Math.E},
      {id:'log2', name:'log(2)', latex:'\\log(2)', value:Math.log(2)},
      {id:'log3', name:'log(3)', latex:'\\log(3)', value:Math.log(3)},
      {id:'logpi', name:'log(π)', latex:'\\log(\\pi)', value:Math.log(Math.PI)},
      {id:'pi2', name:'π^2', latex:'\\pi^2', value:Math.PI*Math.PI},
      {id:'catalan', name:'G', latex:'G', value:0.91596559417721901505},
      {id:'e2', name:'e^2', latex:'e^2', value:Math.E*Math.E},
      {id:'pie', name:'π·e', latex:'\\pi e', value:Math.PI*Math.E},
      {id:'epi', name:'e^π', latex:'e^{\\pi}', value:Math.pow(Math.E, Math.PI)},
      {id:'zeta3', name:'ζ(3)', latex:'\\zeta(3)', value:1.2020569031595942854},
      {id:'sqrt2', name:'√2', latex:'\\sqrt{2}', value:Math.SQRT2},
      {id:'sqrt3', name:'√3', latex:'\\sqrt{3}', value:Math.sqrt(3)},
      {id:'sqrtpi', name:'√π', latex:'\\sqrt{\\pi}', value:Math.sqrt(Math.PI)},
      {id:'phi', name:'φ', latex:'\\varphi', value:(1+Math.sqrt(5))/2},
      {id:'eulergamma', name:'γ', latex:'\\gamma', value:0.5772156649015328606},
      {id:'sinpi5', name:'sin(π/5)', latex:'\\sin(\\pi/5)', value:Math.sin(Math.PI/5)},
      {id:'sinpi7', name:'sin(π/7)', latex:'\\sin(\\pi/7)', value:Math.sin(Math.PI/7)},
      {id:'sinpi8', name:'sin(π/8)', latex:'\\sin(\\pi/8)', value:Math.sin(Math.PI/8)},
      {id:'cospi5', name:'cos(π/5)', latex:'\\cos(\\pi/5)', value:Math.cos(Math.PI/5)},
      {id:'cospi7', name:'cos(π/7)', latex:'\\cos(\\pi/7)', value:Math.cos(Math.PI/7)},
      {id:'cospi8', name:'cos(π/8)', latex:'\\cos(\\pi/8)', value:Math.cos(Math.PI/8)}
    ];
    function mobiusEffort(settings){
      const lvl=Number(settings?.level || document.getElementById('level')?.value || DEFAULT_RIES_LEVEL);
      return Math.max(0, Math.min(5, Math.floor(lvl - Number(DEFAULT_RIES_LEVEL))));
    }
    function shouldRunMobiusRows(settings){
      if(!settings || settings.modules?.mobius===false || settings.complexTarget || !Number.isFinite(settings.target)) return false;
      if(!isDirectDecimalInput(settings.raw)) return false;
      const sig=typedInputPrecision(settings);
      if(sig<2 || sig>20) return false;
      // v10.7.1: Möbius is an independent decimal module.  Do not hide it
      // behind Log/Algebraic checkboxes; simple cases such as gamma+1 and
      // exp(pi/sqrt(3)) should be visible whenever decimal search runs.
      return true;
    }
    function mobiusFormatLinear(coeff, basis, latex=false){
      const parts=[];
      for(let i=0;i<coeff.length;i++){
        let c=Number(coeff[i]||0n);
        if(!c) continue;
        const neg=c<0; c=Math.abs(c);
        const sym=latex ? basis[i].latex : basis[i].name;
        let body;
        if(c===1) body=sym;
        else if(sym==='1') body=String(c);
        else body=latex ? `${c}\\,${sym}` : `${c}·${sym}`;
        if(!parts.length) parts.push((neg?'-':'')+body);
        else parts.push((neg?(latex?' - ':' − '):(latex?' + ':' + '))+body);
      }
      return parts.join('') || '0';
    }
    function mobiusNeedParens(s){ return /[+−-]/.test(String(s).replace(/^-/,'')); }
    function mobiusMergeDenominatorCoeffs(q, e, basis){
      const coeff=[];
      const denBasis=[];
      let constant=BigInt(e||0n);
      for(let i=0;i<basis.length;i++){
        const c=BigInt(q[i]||0n);
        if(basis[i]?.id==='one') constant += c;
        else { coeff.push(c); denBasis.push(basis[i]); }
      }
      if(constant!==0n || !coeff.length){ coeff.push(constant); denBasis.push({id:'one', name:'1', latex:'1', value:1}); }
      return {coeff, basis:denBasis};
    }
    function mobiusRatioStrings(p, q, e, basis){
      const num=mobiusFormatLinear(p,basis,false), nlatex=mobiusFormatLinear(p,basis,true);
      const denData=mobiusMergeDenominatorCoeffs(q,e,basis);
      const den=mobiusFormatLinear(denData.coeff,denData.basis,false), dlatex=mobiusFormatLinear(denData.coeff,denData.basis,true);
      let text, latex;
      if(den==='1'){ text=num; latex=nlatex; }
      else{
        text=`${mobiusNeedParens(num)?`(${num})`:num}/${mobiusNeedParens(den)?`(${den})`:den}`;
        latex=`\\frac{${nlatex}}{${dlatex}}`;
      }
      return {text, latex, num, den, nlatex, dlatex};
    }
    function mobiusTransformStrings(kind, target, ratio){
      if(kind==='direct') return {text:ratio.text, latex:ratio.latex, label:'x', lhsValue:'x'};
      if(kind==='exp') return {text:`log(${ratio.text})`, latex:`\\log\\left(${ratio.latex}\\right)`, label:'e^x', lhsValue:'e^x'};
      const neg=target<0;
      return {text:`${neg?'−':''}exp(${ratio.text})`, latex:`${neg?'-':''}\\exp\\left(${ratio.latex}\\right)`, label:'log|x|', lhsValue:'log|x|'};
    }
    function mobiusRelationPrettyScore(row){
      const len=formulaVisibleLength(row.copyLatex || row.latex || row.candidate);
      const terms=Number(row.terms||1), h=Number(row.height||1);
      return len + Math.max(0,terms-2)*3 + Math.log10(1+h)*5 + (row.variant==='direct'?0:8);
    }
    function mobiusBasisCombos(basisSize, deadline){
      const consts=mobiusConstants.filter(c=>Number.isFinite(c.value));
      const byId=new Map(consts.map(c=>[c.id,c]));
      const combos=[]; const seen=new Set();
      function addCombo(ids){
        if(ids.length!==basisSize) return;
        const combo=[];
        for(const id of ids){ const c=byId.get(id); if(!c) return; combo.push(c); }
        const key=combo.map(c=>c.id).slice().sort().join(',');
        if(seen.has(key)) return; seen.add(key); combos.push(combo);
      }
      // v11.2.4: put Catalan G and pi^2 near the front so the generalized
      // denominator-constant scan reaches identities such as (8G+pi^2)/16
      // before the level-4 wall-clock budget is consumed.
      if(basisSize===2){
        addCombo(['pi2','catalan']);
        addCombo(['one','catalan']);
        addCombo(['one','pi2']);
        addCombo(['pi','catalan']);
        addCombo(['pi','pi2']);
        addCombo(['e','catalan']);
        addCombo(['e','pi2']);
      } else if(basisSize===3){
        addCombo(['one','pi2','catalan']);
        addCombo(['pi','pi2','catalan']);
        addCombo(['e','pi2','catalan']);
      }
      function rec(start, left, cur){
        if(performance.now()>deadline) return;
        if(left===0){
          const key=cur.map(c=>c.id).slice().sort().join(',');
          if(!seen.has(key)){ seen.add(key); combos.push(cur.slice()); }
          return;
        }
        for(let i=start;i<=consts.length-left;i++){
          cur.push(consts[i]); rec(i+1,left-1,cur); cur.pop();
          if(performance.now()>deadline) return;
        }
      }
      rec(0,basisSize,[]);
      return combos;
    }
    function mobiusLinearFormsForBasis(basis, coeffLimit, maxForms){
      const forms=[];
      const n=basis.length;
      const coeff=Array(n).fill(0n);
      function rec(i){
        if(i===n){
          if(!coeff.some(x=>x!==0n)) return;
          let val=0, terms=0, height=0;
          for(let j=0;j<n;j++){
            const c=Number(coeff[j]||0n);
            if(c){ terms++; height=Math.max(height, Math.abs(c)); val += c*basis[j].value; }
          }
          if(!Number.isFinite(val) || Math.abs(val)<1e-15) return;
          const cpy=coeff.slice();
          forms.push({coeff:cpy, value:val, terms, height, score:terms*10+height+formulaVisibleLength(mobiusFormatLinear(cpy,basis,false))*.03});
          return;
        }
        for(let c=-coeffLimit;c<=coeffLimit;c++){ coeff[i]=BigInt(c); rec(i+1); }
      }
      rec(0);
      forms.sort((a,b)=>a.score-b.score || Math.abs(a.value)-Math.abs(b.value));
      const simple=forms.slice(0, Math.max(12, maxForms));
      for(const f of forms){
        if(f.terms<=2 && f.height<=2 && !simple.some(g=>g.coeff.join(',')===f.coeff.join(','))) simple.push(f);
      }
      simple.sort((a,b)=>a.value-b.value);
      return simple;
    }
    function mobiusDenominatorFormsForBasis(basis, coeffLimit, maxForms){
      const forms=[];
      const n=basis.length;
      const q=Array(n).fill(0n);
      function rec(i){
        if(i===n+1){
          const e=q[n];
          if(!q.some(x=>x!==0n)) return;
          let val=Number(e), terms=e!==0n?1:0, height=Math.abs(Number(e||0n));
          for(let j=0;j<n;j++){
            const c=Number(q[j]||0n);
            if(c){ terms++; height=Math.max(height, Math.abs(c)); val += c*basis[j].value; }
          }
          if(!Number.isFinite(val) || Math.abs(val)<1e-15) return;
          const coeff=q.slice(0,n), denData=mobiusMergeDenominatorCoeffs(coeff,e,basis);
          forms.push({coeff, e, value:val, terms, height, score:terms*10+height+formulaVisibleLength(mobiusFormatLinear(denData.coeff,denData.basis,false))*.03});
          return;
        }
        for(let c=-coeffLimit;c<=coeffLimit;c++){ q[i]=BigInt(c); rec(i+1); }
      }
      rec(0);
      forms.sort((a,b)=>a.score-b.score || Math.abs(a.value)-Math.abs(b.value));
      const simple=forms.slice(0, Math.max(12, maxForms));
      for(const f of forms){
        const key=f.coeff.join(',')+'|'+String(f.e);
        if(f.terms<=2 && f.height<=2 && !simple.some(g=>g.coeff.join(',')+'|'+String(g.e)===key)) simple.push(f);
      }
      simple.sort((a,b)=>a.value-b.value);
      return simple;
    }
    function mobiusBuildRow(settings, variant, basis, pIn, qIn, eIn, y, yyIgnored){
      let p=pIn.slice(), q=qIn.slice(), e=BigInt(eIn||0n);
      let g=0n; for(const x of p.concat(q,[e])) g=gcdBig(g, x<0n ? -x : x);
      if(g>1n){ p=p.map(x=>x/g); q=q.map(x=>x/g); e=e/g; }
      let num=0, den=Number(e);
      for(let i=0;i<basis.length;i++){ num += Number(p[i])*basis[i].value; den += Number(q[i])*basis[i].value; }
      if(!Number.isFinite(num) || !Number.isFinite(den) || Math.abs(den)<1e-14) return null;
      let yy=num/den;
      if(!Number.isFinite(yy)) return null;
      if(den<0){ for(let i=0;i<p.length;i++) p[i]=-p[i]; for(let i=0;i<q.length;i++) q[i]=-q[i]; e=-e; num=-num; den=-den; yy=num/den; }
      const h=coeffHeight(p.concat(q,[e]));
      const ratio=mobiusRatioStrings(p,q,e,basis);
      const out=mobiusTransformStrings(variant.kind, settings.target, ratio);
      const predicted = variant.kind==='direct' ? yy : (variant.kind==='exp' ? Math.log(yy) : (settings.target<0 ? -Math.exp(yy) : Math.exp(yy)));
      if(!Number.isFinite(predicted)) return null;
      const sig=typedInputPrecisionForDouble(settings);
      const err=Math.abs(predicted-settings.target);
      const relErr=err/Math.max(1,Math.abs(settings.target));
      const accept=typedRelativeToleranceNumber(sig, 24, 1, 17);
      if(relErr>accept) return null;
      const coeff=p.concat(q,[e]);
      const nonZero=coeff.filter(x=>x!==0n).length;
      const row={
        candidate:`Möbius relation: x ≈ ${out.text}`,
        latex:`x \\approx ${out.latex}`,
        copyLatex:`x \\approx ${out.latex}`,
        value:`${out.lhsValue} ≈ ${fmtValue(y)}; ratio = ${fmtValue(yy)}; basis ${basis.map(b=>b.name).join(', ')}; height ${h.toString()}`,
        copyValue:`${out.lhsValue} ≈ ${ratio.text}`,
        err:relErr,
        errText:fmtErr(relErr),
        height:h,
        terms:nonZero,
        mobiusCategory:variant.kind,
        variant:variant.kind,
        score:0
      };
      row.score=mobiusRelationPrettyScore(row);
      return row;
    }
    function mobiusSparseRowsForVariant(settings, variant, basisSize, deadline){
      const y=Number(variant.y);
      if(!Number.isFinite(y) || Math.abs(y)>1e80) return [];
      const sig=typedInputPrecisionForDouble(settings);
      const tol=typedRelativeToleranceNumber(sig, 10, 1, 17) * Math.max(1, Math.abs(y));
      const rows=[];
      const combos=mobiusBasisCombos(basisSize, deadline);
      const coeffLimit=basisSize===2 ? 5 : 3;
      const maxForms=basisSize===2 ? 90 : 120;
      for(const basis of combos){
        if(performance.now()>deadline) break;
        const numForms=mobiusLinearFormsForBasis(basis, coeffLimit, maxForms);
        const denForms=mobiusDenominatorFormsForBasis(basis, coeffLimit, maxForms);
        if(!numForms.length || !denForms.length) continue;
        function nearForms(v){
          let lo=0, hi=numForms.length;
          while(lo<hi){ const mid=(lo+hi)>>1; if(numForms[mid].value<v) lo=mid+1; else hi=mid; }
          const out=[];
          for(let k=Math.max(0,lo-5); k<Math.min(numForms.length,lo+6); k++) out.push(numForms[k]);
          return out;
        }
        for(const q of denForms){
          if(performance.now()>deadline) break;
          const need=y*q.value;
          for(const pf of nearForms(need)){
            const yy=pf.value/q.value;
            if(!Number.isFinite(yy) || Math.abs(yy-y)>tol) continue;
            const row=mobiusBuildRow(settings, variant, basis, pf.coeff, q.coeff, q.e, y, yy);
            if(row) rows.push(row);
          }
        }
      }
      return rows;
    }
    function mobiusRowsForVariant(settings, variant, basisSize, deadline){
      const y=Number(variant.y);
      if(!Number.isFinite(y) || Math.abs(y)>1e80) return [];
      const sig=typedInputPrecisionForDouble(settings);
      const effectivePrec=Math.max(6, Math.min(DOUBLE_EFFECTIVE_PRECISION_DIGITS, sig));
      const scaleNum=Math.pow(10,effectivePrec);
      const tol=typedRelativeToleranceNumber(sig, 10, 1, 17) * Math.max(1, Math.abs(y));
      let maxH; try{ maxH=BigInt(document.getElementById('logHeight')?.value.trim() || '400'); }catch(e){ maxH=400n; }
      if(maxH<20n) maxH=20n;
      if(maxH>1200n) maxH=1200n;
      const rows=[]; const seen=new Set();
      try{ rows.push(...mobiusSparseRowsForVariant(settings, variant, basisSize, Math.min(deadline, performance.now()+180))); }catch(e){}
      const combos=mobiusBasisCombos(basisSize, deadline);
      for(const basis of combos){
        if(performance.now()>deadline) break;
        const vals=[];
        for(const b of basis) vals.push(b.value);
        for(const b of basis) vals.push(-y*b.value);
        vals.push(-y);
        if(vals.some(v=>!Number.isFinite(v))) continue;
        const dim=vals.length;
        const lattice=[];
        for(let i=0;i<dim;i++){
          const row=Array(dim+1).fill(0n);
          row[i]=1n;
          row[dim]=BigInt(Math.round(vals[i]*scaleNum));
          lattice.push(row);
        }
        const reduced=[];
        // v11.2.4: generalized tri-basis Möbius lattices are larger; avoid
        // repeated exact-rational Gram-Schmidt there and use the bounded fast
        // reducer so the level budget remains a real wall-clock limit.
        if(basisSize<=2){
          try{ reduced.push(...exactLLLReduce(lattice.map(r=>r.slice()),99n,100n,performance.now()+18)); }catch(e){}
          try{ reduced.push(...lllReduce(lattice.map(r=>r.slice()),0.82)); }catch(e){}
        } else {
          try{ reduced.push(...constDbFastLLLReduce(lattice.map(r=>r.slice()),0.82,1600,Math.min(deadline, performance.now()+18),1e7)); }catch(e){}
        }
        for(const rr of reduced){
          const coeff=normalizeVector(rr.slice(0,dim));
          const p=coeff.slice(0,basisSize), q=coeff.slice(basisSize, basisSize*2), e=coeff[basisSize*2]||0n;
          if(!p.some(x=>x!==0n) || (!q.some(x=>x!==0n) && e===0n)) continue;
          const h=coeffHeight(coeff); if(h===0n || h>maxH) continue;
          let num=0, den=Number(e);
          for(let i=0;i<basisSize;i++){ num += Number(p[i])*basis[i].value; den += Number(q[i])*basis[i].value; }
          if(!Number.isFinite(num) || !Number.isFinite(den) || Math.abs(den)<1e-14) continue;
          let yy=num/den;
          if(!Number.isFinite(yy) || Math.abs(yy-y)>tol) continue;
          let pp=p.slice(), qq=q.slice(), ee=e;
          if(den<0){ for(let i=0;i<pp.length;i++) pp[i]=-pp[i]; for(let i=0;i<qq.length;i++) qq[i]=-qq[i]; ee=-ee; num=-num; den=-den; yy=num/den; }
          const basisKey=basis.map(b=>b.id).join(',');
          const key=`${variant.kind}|${basisKey}|${pp.join(',')}|${qq.join(',')}|${String(ee)}`;
          if(seen.has(key)) continue; seen.add(key);
          const row=mobiusBuildRow(settings, variant, basis, pp, qq, ee, y, yy);
          if(row) rows.push(row);
        }
      }
      return rows;
    }
    function mobiusRelationRows(settings){
      if(!shouldRunMobiusRows(settings)) return [];
      const effort=mobiusEffort(settings);
      const opt=settings?.mobiusOptions || {direct:true,logabs:true,exp:true,triple:true};
      const variants=[];
      if(opt.direct!==false) variants.push({kind:'direct', y:settings.target});
      if(opt.logabs!==false && settings.target!==0){ const y=Math.log(Math.abs(settings.target)); if(Number.isFinite(y)) variants.push({kind:'logabs', y}); }
      if(opt.exp!==false && settings.target<=10){ const y=Math.exp(settings.target); if(Number.isFinite(y)) variants.push({kind:'exp', y}); }
      if(!variants.length) return [];
      const rows=[];
      const moduleBudget=riesLevelModuleBudgetMs(settings);
      const moduleStart=performance.now();
      const pairDeadline=moduleStart + (effort>0 ? Math.max(5000, Math.floor(moduleBudget*0.42)) : moduleBudget);
      const pairSlice=Math.max(650, Math.floor((pairDeadline-moduleStart)/Math.max(1, variants.length)));
      for(const v of variants){
        if(performance.now()>pairDeadline) break;
        rows.push(...mobiusRowsForVariant(settings,v,2,Math.min(pairDeadline, performance.now()+pairSlice)));
      }
      if(effort>0 && opt.triple!==false){
        const triDeadline=moduleStart + moduleBudget;
        const triSlice=Math.max(650, Math.floor((triDeadline-performance.now())/Math.max(1, variants.length)));
        for(const v of variants){
          if(performance.now()>triDeadline) break;
          rows.push(...mobiusRowsForVariant(settings,v,3,Math.min(triDeadline, performance.now()+triSlice)));
        }
      }
      const map=new Map();
      for(const r of rows){
        const k=normalizeResultTextKey(r.candidate);
        if(!map.has(k) || (r.score??1e9)<(map.get(k).score??1e9)) map.set(k,r);
      }
      return [...map.values()].sort((a,b)=>(a.score??1e9)-(b.score??1e9) || (a.err||1)-(b.err||1)).slice(0,5);
    }



    // v11 low-precision constant database matcher.
    // The database is stored in assets/constantdb300.js as 190 uploaded named
    // constants plus 110 generated elementary constants.  For each typed decimal
    // x (<=20 significant digits) we test b in {x^-2,x^-1,x^-1/2,x^1/2,x,x^2, exp(x), log|x|}
    // against each database constant c by (1) degree <= 2 algebraic ratios b/c
    // by base tests plus Continue-only higher algebraic-ratio and log-linear LLL passes.
    function constantDbRaw(){ return Array.isArray(window.RIES_CONSTANT_DB_300) ? window.RIES_CONSTANT_DB_300 : []; }
    let constantDbCache=null;
    function constantDbRecords(){
      if(constantDbCache) return constantDbCache;
      constantDbCache=constantDbRaw().map((c,idx)=>{
        const value=Number(c.value);
        return {id:String(c.id||('db'+idx)), label:String(c.label||c.id||('constant '+idx)), latex:String(c.latex||''), description:String(c.description||''), source:String(c.source||''), value};
      }).filter(c=>Number.isFinite(c.value) && c.value!==0);
      return constantDbCache;
    }
    function shouldRunConstantDbRows(settings){
      if(!settings || settings.modules?.constantDb===false || settings.complexTarget || !Number.isFinite(settings.target)) return false;
      if(!isDirectDecimalInput(settings.raw)) return false;
      const sig=typedInputPrecision(settings);
      return sig>=2 && sig<=20;
    }
    function constDbSign(x){ return x<0 ? '−' : ''; }
    function constDbParenText(s){ s=String(s); return /[+−-]/.test(s.replace(/^[-−]/,'')) ? `(${s})` : s; }
    function constDbParenLatex(s){ s=String(s); return /\\frac|[+\-]/.test(s.replace(/^-/,'')) ? `\\left(${s}\\right)` : s; }
    function constDbConstLatex(c){ return 'c'; }
    function constDbConstExpr(c){ return {text:'c', latex:'c'}; }
    function constDbDisplayNotation(c){
      const latex=String(c?.latex||'').trim();
      const label=String(c?.label||c?.id||'constant').trim();
      return latex ? `${label} (${latex})` : label;
    }
    function constDbRationalApprox(x, maxDen, maxNum, relTol){
      if(!Number.isFinite(x)) return null;
      const neg=x<0; let a=Math.abs(x);
      if(a===0) return {p:0n,q:1n,err:0};
      let h1=1,h0=0,k1=0,k0=1, b=Math.floor(a), xcur=a;
      let best=null;
      for(let iter=0;iter<32;iter++){
        const h=b*h1+h0, k=b*k1+k0;
        if(k>maxDen || Math.abs(h)>maxNum) break;
        const val=h/k, err=Math.abs(val-a)/Math.max(1,a);
        best={p:BigInt(neg?-h:h), q:BigInt(k), err};
        if(err<=relTol) break;
        const frac=xcur-b; if(Math.abs(frac)<1e-18) break;
        xcur=1/frac; b=Math.floor(xcur);
        h0=h1; h1=h; k0=k1; k1=k;
      }
      if(best && best.err<=relTol) return best;
      // Also try the nearest denominator in the allowed range; useful for small typed precision.
      for(let q=1;q<=maxDen;q++){
        const h=Math.round(a*q); if(Math.abs(h)>maxNum) continue;
        const val=h/q, err=Math.abs(val-a)/Math.max(1,a);
        if(!best || err<best.err) best={p:BigInt(neg?-h:h), q:BigInt(q), err};
      }
      return best && best.err<=relTol ? best : null;
    }
    function constDbRatText(p,q){ return rationalString(BigInt(p), BigInt(q)); }
    function constDbMulConstExpr(alpha, c){
      const p=BigInt(alpha.p), q=BigInt(alpha.q); const ce=constDbConstExpr(c);
      if(p===0n) return {text:'0', latex:'0'};
      if(p===q) return ce;
      if(p===-q) return {text:`−${ce.text}`, latex:`-${ce.latex}`};
      const neg=p<0n; const ap=neg?-p:p;
      let text, latex;
      if(q===1n){
        text=`${neg?'−':''}${ap.toString()}·${ce.text}`;
        latex=`${neg?'-':''}${ap.toString()}\\,${ce.latex}`;
      }else if(ap===1n){
        text=`${neg?'−':''}${ce.text}/${q.toString()}`;
        latex=`${neg?'-':''}\\frac{${ce.latex}}{${q.toString()}}`;
      }else{
        text=`${neg?'−':''}${ap.toString()}·${ce.text}/${q.toString()}`;
        latex=`${neg?'-':''}\\frac{${ap.toString()}\\,${ce.latex}}{${q.toString()}}`;
      }
      return {text, latex};
    }
    function constDbSqrtMulConstExpr(rat, sign, c){
      const p=BigInt(rat.p), q=BigInt(rat.q); if(p<0n || q<=0n) return null;
      const ce=constDbConstExpr(c);
      const neg=sign<0;
      if(p===q) return neg ? {text:`−${ce.text}`, latex:`-${ce.latex}`} : ce;
      const rtxt = q===1n ? `√${p.toString()}` : `√(${p.toString()}/${q.toString()})`;
      const rlatex = q===1n ? `\\sqrt{${p.toString()}}` : `\\sqrt{\\frac{${p.toString()}}{${q.toString()}}}`;
      return {text:`${neg?'−':''}${rtxt}·${ce.text}`, latex:`${neg?'-':''}${rlatex}\\,${ce.latex}`};
    }
    function constDbQuadraticExpr(coeff, ratio, c){
      let [a0,a1,a2]=coeff.map(Number);
      a0=Number(a0||0); a1=Number(a1||0); a2=Number(a2||0);
      if(a2===0){
        if(a1===0) return null;
        const p=BigInt(-a0), q=BigInt(a1); return constDbMulConstExpr({p,q}, c);
      }
      const D=a1*a1-4*a2*a0; if(D<0) return null;
      const sqrtD=Math.sqrt(D); const roots=[(-a1+sqrtD)/(2*a2),(-a1-sqrtD)/(2*a2)];
      const branch=Math.abs(roots[0]-ratio)<=Math.abs(roots[1]-ratio) ? 1 : -1;
      const den=2*a2;
      // Perfect square discriminants collapse to a rational multiple.
      const sd=Math.round(sqrtD);
      if(sd*sd===D){
        const num=-a1 + branch*sd;
        return constDbMulConstExpr({p:BigInt(num), q:BigInt(den)}, c);
      }
      const ce=constDbConstExpr(c);
      const mid=-a1;
      const sign=branch>0 ? '+' : '−';
      const text=`(${mid}${sign}√${D})/${den}·${ce.text}`;
      const latex=`\\frac{${mid}${branch>0?'+':'-'}\\sqrt{${D}}}{${den}}\\,${ce.latex}`;
      return {text, latex};
    }
    function constDbCoeffTermCount(coeff){
      return (coeff||[]).reduce((n,a)=>n+(Number(a||0)!==0?1:0),0);
    }
    function constDbRootInfoNearRatio(coeff, ratio, sig){
      if(!Array.isArray(coeff) || !Number.isFinite(ratio)) return null;
      const c=normalizeCoeffs(coeff.map(x=>BigInt(Math.trunc(Number(x)||0))));
      const d=polyDegree(c);
      if(d<1 || constDbCoeffTermCount(c)<2) return null;
      let root=NaN;
      if(d===1){
        const a0=Number(c[0]||0n), a1=Number(c[1]||0n);
        if(a1===0) return null;
        root=-a0/a1;
      }else{
        root=refinePolyRoot(c, ratio);
      }
      if(!Number.isFinite(root)) return null;
      const rootRel=Math.abs(root-ratio)/Math.max(1,Math.abs(ratio));
      const rootTol=Math.max(1e-12, typedRelativeToleranceNumber(typedInputPrecisionForDouble(sig), 40, 1, 13));
      // v11.1 guard: residual-only tests falsely accept monomials such as
      // α^3=0 when b/c is merely small. The displayed α must be an actual
      // nearby root, not just a small residual at the input ratio.
      if(rootRel>rootTol) return null;
      return {root, rootRel, coeff:c};
    }

    function constDbExactFallbackAllowed(deadline, minRemainingMs=250){
      // Exact BigInt-rational LLL is reliable but a single Gram-Schmidt rebuild
      // can overrun a short UI timeslice.  Constant-DB v11.2+ uses the fast
      // floating reducer first and only enters exact fallback when there is a
      // genuinely roomy local budget; calls without an explicit deadline keep
      // the old behavior for offline/manual tests.
      return !deadline || (performance.now()+minRemainingMs < deadline);
    }
    function constDbFindPolynomialRatio(ratio, degree, sig, deadline=0){
      if(!Number.isFinite(ratio) || degree<1) return null;
      sig=typedInputPrecisionForDouble(sig);
      degree=Math.max(1, Math.min(3, Number(degree)||3));
      const relTol=typedRelativeToleranceNumber(sig, 18, 1, 14);
      const H=Math.max(5, Math.min(sig<=8 ? 9 : 16, sig<=6 ? 6 : (sig<=12 ? 10 : 14)));
      const powers=[1];
      for(let i=1;i<=degree;i++) powers.push(powers[powers.length-1]*ratio);
      let best=null;
      const coeff=Array(degree+1).fill(0);
      function scoreCandidate(){
        const a0=Math.round(-coeff.slice(1).reduce((sum,a,i)=>sum+a*powers[i+1],0));
        if(Math.abs(a0)>H) return;
        coeff[0]=a0;
        let val=0, norm=1, nz=0, maxc=0;
        for(let i=0;i<=degree;i++){
          const ai=coeff[i];
          if(ai){ nz++; maxc=Math.max(maxc,Math.abs(ai)); }
          val += ai*powers[i];
          norm=Math.max(norm, Math.abs(ai*powers[i]));
        }
        if(nz<2 || maxc===0) return;
        // Trim apparent degree; a cubic search may discover a simpler relation.
        let d=degree; while(d>0 && coeff[d]===0) d--;
        if(d<1) return;
        const rel=Math.abs(val)/norm;
        if(rel>relTol) return;
        const rootInfo=constDbRootInfoNearRatio(coeff.slice(0,d+1), ratio, sig);
        if(!rootInfo) return;
        const score=d*18 + nz*3 + maxc + rel*1e6 + rootInfo.rootRel*1e8;
        if(!best || score<best.score) best={coeff:coeff.slice(0,d+1), err:rel, height:BigInt(maxc), degree:d, score, root:rootInfo.root, rootRel:rootInfo.rootRel};
      }
      function considerPolyCoeff(raw){
        const cc=normalizeCoeffs(raw.slice(0,degree+1).map(x=>BigInt(x)));
        const d=polyDegree(cc); if(d<1 || d>degree) return;
        const h=coeffHeight(cc); if(h===0n || h>BigInt(CONSTDB_RELATION_COEFF_BOUND)) return;
        let val=0, norm=1, nz=0;
        for(let i=0;i<cc.length;i++){ const ai=Number(cc[i]); if(ai) nz++; val += ai*powers[i]; norm=Math.max(norm,Math.abs(ai*powers[i])); }
        if(nz<2) return;
        const rel=Math.abs(val)/norm; if(rel>relTol) return;
        const rootInfo=constDbRootInfoNearRatio(cc, ratio, sig); if(!rootInfo) return;
        const score=d*18+nz*3+Number(h)+rel*1e6+rootInfo.rootRel*1e8;
        if(!best || score<best.score) best={coeff:cc.map(Number), err:rel, height:h, degree:d, score, root:rootInfo.root, rootRel:rootInfo.rootRel};
      }
      // v11.2.2: in wall-clocked constant-DB scans, use bounded PSLQ/LLL as
      // the primary degree<=3 ratio probe.  The older exhaustive coefficient
      // recursion is retained for no-deadline/offline calls.
      if(deadline){
        const bits=constDbPslqBits(sig, degree+1);
        const fixed=constDbFixedVectorFromValues(powers.slice(0,degree+1), bits);
        if(fixed){ try{ const pr=pslqFixed(fixed,bits,BigInt(CONSTDB_RELATION_COEFF_BOUND+1),Math.max(260,(degree+1)*160),Math.min(deadline,performance.now()+5)); if(pr) considerPolyCoeff(pr); }catch(e){} }
        if(!best && performance.now()<deadline){
          try{
            const prec=Math.max(7, Math.min(15, sig)); const scaleNum=Math.pow(10,prec); const lattice=[];
            for(let i=0;i<=degree;i++){ const row=Array(degree+2).fill(0n); row[i]=1n; row[degree+1]=BigInt(Math.round(powers[i]*scaleNum)); lattice.push(row); }
            for(const rr of constDbFastLLLReduce(lattice,0.84,1600,Math.min(deadline,performance.now()+12),1e16)) considerPolyCoeff(rr.slice(0,degree+1));
          }catch(e){}
        }
        return best;
      }
      function rec(pos){
        if(deadline && performance.now()>deadline) return;
        if(pos>degree){ scoreCandidate(); return; }
        for(let a=-H;a<=H;a++){
          coeff[pos]=a;
          rec(pos+1);
          if(deadline && performance.now()>deadline) return;
        }
      }
      rec(1);
      // A tiny LLL fallback catches low-height cubic relations that the bounded
      // scan can miss when the typed precision is just high enough to be picky.
      // In the UI constant-DB path this function is called with very short
      // local deadlines inside the deep scan; running even the fast LLL fallback
      // after the exhaustive degree<=3 scan can overrun those slices.  Keep the
      // fallback for offline/no-deadline calls or genuinely roomy local budgets,
      // but do not let it monopolize level-4/5/6 constant-DB scans.
      if((!best || best.degree<degree) && constDbExactFallbackAllowed(deadline,120) && (!deadline || performance.now()<deadline)){
        try{
          const prec=Math.max(8, Math.min(16, sig));
          const scaleNum=Math.pow(10,prec);
          const lattice=[];
          for(let i=0;i<=degree;i++){
            const row=Array(degree+2).fill(0n);
            row[i]=1n;
            row[degree+1]=BigInt(Math.round(powers[i]*scaleNum));
            lattice.push(row);
          }
          const reduced=[];
          try{ reduced.push(...constDbFastLLLReduce(lattice.map(r=>r.slice()),0.82,300,deadline,1e4)); }catch(e){}
          if(!reduced.length && constDbExactFallbackAllowed(deadline,250) && (!deadline || performance.now()<deadline)){ try{ reduced.push(...exactLLLReduce(lattice.map(r=>r.slice()),99n,100n,performance.now()+Math.min(10, deadline?Math.max(2,deadline-performance.now()):10))); }catch(e){} }
          for(const rr of reduced){
            const cc=normalizeCoeffs(rr.slice(0,degree+1));
            const d=polyDegree(cc); if(d<1 || d>degree) continue;
            const h=coeffHeight(cc); if(h===0n || h>BigInt(H*2)) continue;
            let val=0, norm=1, nz=0;
            for(let i=0;i<cc.length;i++){ const ai=Number(cc[i]); if(ai) nz++; val += ai*powers[i]; norm=Math.max(norm,Math.abs(ai*powers[i])); }
            if(nz<2) continue;
            const rel=Math.abs(val)/norm;
            if(rel>relTol) continue;
            const rootInfo=constDbRootInfoNearRatio(cc, ratio, sig);
            if(!rootInfo) continue;
            const score=d*18+nz*3+Number(h)+rel*1e6+rootInfo.rootRel*1e8;
            if(!best || score<best.score) best={coeff:cc.map(Number), err:rel, height:h, degree:d, score, root:rootInfo.root, rootRel:rootInfo.rootRel};
          }
        }catch(e){}
      }
      return best;
    }
    function constDbFindQuadraticRatio(ratio, sig){ return constDbFindPolynomialRatio(ratio, 2, sig); }
    function constDbFindAlgebraicRatioLLL(ratio, maxDegree, sig, maxHeight, deadline=0, relTol=null){
      if(!Number.isFinite(ratio) || Math.abs(ratio)>1e100) return null;
      sig=typedInputPrecisionForDouble(sig);
      maxDegree=Math.max(1, Math.min(8, Math.floor(maxDegree||1)));
      relTol = relTol || typedRelativeToleranceNumber(sig, 18, 1, 14);
      const powers=[1];
      for(let i=1;i<=maxDegree;i++) powers.push(powers[i-1]*ratio);
      if(powers.some(v=>!Number.isFinite(v) || Math.abs(v)>1e100)) return null;
      const prec=Math.max(7, Math.min(16, sig));
      const scaleNum=Math.pow(10,prec);
      const lattice=[];
      for(let i=0;i<=maxDegree;i++){
        const row=Array(maxDegree+2).fill(0n);
        row[i]=1n;
        row[maxDegree+1]=BigInt(Math.round(powers[i]*scaleNum));
        lattice.push(row);
      }
      const localDeadline=deadline || performance.now()+18;
      let best=null;
      const maxH=BigInt(Math.max(8, Number(maxHeight)||18));
      const seenRows=new Set();
      function considerReduced(rs){
        for(const rr of rs||[]){
          if(deadline && performance.now()>deadline) break;
          const key=rr.join(','); if(seenRows.has(key)) continue; seenRows.add(key);
          const coeff=normalizeCoeffs(rr.slice(0,maxDegree+1));
          const d=polyDegree(coeff); if(d<1 || d>maxDegree) continue;
          const h=coeffHeight(coeff); if(h===0n || h>maxH) continue;
          let val=0, norm=1, terms=0;
          for(let i=0;i<coeff.length;i++){
            const ai=Number(coeff[i]);
            if(ai) terms++;
            val += ai*powers[i];
            norm=Math.max(norm, Math.abs(ai*powers[i]));
          }
          if(terms<2) continue;
          const rel=Math.abs(val)/norm;
          if(rel>relTol) continue;
          const rootInfo=constDbRootInfoNearRatio(coeff, ratio, sig);
          if(!rootInfo) continue;
          const score=d*18+terms*3+Number(h)+rel*1e7+rootInfo.rootRel*1e8;
          if(!best || score<best.score) best={coeff, err:rel, height:h, degree:d, terms, score, root:rootInfo.root, rootRel:rootInfo.rootRel};
        }
      }
      const pslqFixedVals=constDbFixedVectorFromValues(powers.slice(0,maxDegree+1), constDbPslqBits(sig, maxDegree+1));
      if(pslqFixedVals){
        try{
          const pr=pslqFixed(pslqFixedVals, constDbPslqBits(sig, maxDegree+1), BigInt(Number(maxH)+1), Math.max(320, (maxDegree+1)*180), Math.min(localDeadline, performance.now()+Math.max(8, maxDegree*3)));
          if(pr) considerReduced([pr.concat([0n])]);
        }catch(e){}
      }
      // v11.2: constant database uses the fast floating reducer first, then
      // falls back to the older exact BigInt rational LLL only when the fast
      // pass produces no validated relation.  Relation forms and acceptance
      // tests are unchanged.
      try{ considerReduced(constDbFastLLLReduce(lattice.map(r=>r.slice()),0.82,6000,Math.min(localDeadline, performance.now()+35),1e12)); }catch(e){}
      if(!best && performance.now()<localDeadline && constDbExactFallbackAllowed(localDeadline,250)){
        try{ considerReduced(exactLLLReduce(lattice.map(r=>r.slice()),99n,100n,performance.now()+Math.min(12, Math.max(2, localDeadline-performance.now())))); }catch(e){}
      }
      if(!best && maxDegree<=5 && (!deadline || performance.now()<deadline)){
        const H=Math.min(3, Number(maxH));
        const coeff=Array(maxDegree+1).fill(0);
        const seen=new Set();
        function scoreSmall(){
          if(deadline && performance.now()>deadline) return;
          const cc=normalizeCoeffs(coeff.map(x=>BigInt(x)));
          const key=cc.join(','); if(seen.has(key)) return; seen.add(key);
          const d=polyDegree(cc); if(d<1 || d>maxDegree) return;
          const h=coeffHeight(cc); if(h===0n || h>maxH) return;
          let val=0, norm=1, terms=0;
          for(let i=0;i<cc.length;i++){
            const ai=Number(cc[i]); if(ai) terms++;
            val += ai*powers[i]; norm=Math.max(norm, Math.abs(ai*powers[i]));
          }
          if(terms<2) return;
          const rel=Math.abs(val)/norm; if(rel>relTol) return;
          const rootInfo=constDbRootInfoNearRatio(cc, ratio, sig);
          if(!rootInfo) return;
          const score=d*18+terms*3+Number(h)+rel*1e7+rootInfo.rootRel*1e8;
          if(!best || score<best.score) best={coeff:cc, err:rel, height:h, degree:d, terms, score, root:rootInfo.root, rootRel:rootInfo.rootRel};
        }
        function rec(pos){
          if(deadline && performance.now()>deadline) return;
          if(pos>maxDegree){ scoreSmall(); return; }
          for(let a=-H;a<=H;a++){ coeff[pos]=a; rec(pos+1); if(deadline && performance.now()>deadline) return; }
        }
        rec(0);
      }
      return best;
    }
    function constDbPolyToInline(coeff, variable='α'){
      const parts=[];
      for(let i=coeff.length-1;i>=0;i--){
        const a=Number(coeff[i]||0); if(!a) continue;
        const neg=a<0, mag=Math.abs(a);
        let term='';
        if(parts.length) term += neg ? ' − ' : ' + '; else if(neg) term += '−';
        if(i===0) term += String(mag);
        else { if(mag!==1) term += String(mag); term += i===1 ? variable : `${variable}^${i}`; }
        parts.push(term);
      }
      return (parts.join('')||'0')+' = 0';
    }
    function constDbLinearForms(c, H){
      const forms=[];
      for(let a=-H;a<=H;a++) for(let b=-H;b<=H;b++){
        if(a===0 && b===0) continue;
        const value=a+b*c.value;
        if(!Number.isFinite(value) || Math.abs(value)<1e-15) continue;
        forms.push({a,b,value,height:Math.max(Math.abs(a),Math.abs(b)), terms:(a?1:0)+(b?1:0)});
      }
      forms.sort((x,y)=>x.value-y.value);
      return forms;
    }
    function constDbLinearExpr(f, c){
      const ce=constDbConstExpr(c);
      const parts=[], lparts=[];
      if(f.a){ parts.push(String(f.a)); lparts.push(String(f.a)); }
      if(f.b){
        const neg=f.b<0, ab=Math.abs(f.b);
        const body=ab===1 ? ce.text : `${ab}·${ce.text}`;
        const lbody=ab===1 ? ce.latex : `${ab}\\,${ce.latex}`;
        if(!parts.length){ parts.push((neg?'−':'')+body); lparts.push((neg?'-':'')+lbody); }
        else { parts.push((neg?' − ':' + ')+body); lparts.push((neg?' - ':' + ')+lbody); }
      }
      return {text:parts.join('')||'0', latex:lparts.join('')||'0'};
    }
    function constDbMobiusExpr(numF, denF, c){
      const ne=constDbLinearExpr(numF,c), de=constDbLinearExpr(denF,c);
      if(de.text==='1') return ne;
      return {text:`${constDbParenText(ne.text)}/${constDbParenText(de.text)}`, latex:`\\frac{${ne.latex}}{${de.latex}}`};
    }
    function constDbTransformRows(settings){
      const x=settings.target; const arr=[];
      const enabled=settings?.constDbTransforms || {pow1:true,exp:true,log:true,powm1:true,pow2:true};
      const add=(kind,y,label)=>{ if(enabled[kind]!==false && Number.isFinite(y)) arr.push({kind,y,label}); };
      // v11.6: the parameter UI can disable individual transform families.
      // Defaults keep the v11.5.1 order: x, exp(x), log(x), 1/x, x^2.
      add('pow1', x, 'x');
      add('exp', Math.exp(x), 'exp(x)');
      if(x>0) add('log', Math.log(x), 'log(x)');
      if(x!==0) add('powm1', 1/x, 'x^{-1}');
      add('pow2', x*x, 'x^2');
      return arr;
    }
    function constDbApplyInverse(settings, tr, expr){
      const x=settings.target; const et=expr.text, el=expr.latex;
      if(tr.kind==='pow1') return {text:et, latex:el};
      if(tr.kind==='pow2') return {text:`${x<0?'−':''}√(${et})`, latex:`${x<0?'-':''}\\sqrt{${el}}`};
      if(tr.kind==='powhalf') return {text:`(${et})^2`, latex:`${constDbParenLatex(el)}^2`};
      if(tr.kind==='powm1') return {text:`1/${constDbParenText(et)}`, latex:`\\frac{1}{${el}}`};
      if(tr.kind==='powm2') return {text:`${x<0?'−':''}1/√(${et})`, latex:`${x<0?'-':''}\\frac{1}{\\sqrt{${el}}}`};
      if(tr.kind==='powmhalf') return {text:`1/(${et})^2`, latex:`\\frac{1}{${constDbParenLatex(el)}^2}`};
      if(tr.kind==='exp') return {text:`log(${et})`, latex:`\\log\\left(${el}\\right)`};
      if(tr.kind==='log' || tr.kind==='logabs') return {text:`${x<0?'−':''}exp(${et})`, latex:`${x<0?'-':''}\\exp\\left(${el}\\right)`};
      return {text:et, latex:el};
    }
    function constDbPredictedFromB(settings, tr, b){
      const x=settings.target;
      if(tr.kind==='pow1') return b;
      if(tr.kind==='pow2') return (x<0?-1:1)*Math.sqrt(b);
      if(tr.kind==='powhalf') return b*b;
      if(tr.kind==='powm1') return 1/b;
      if(tr.kind==='powm2') return (x<0?-1:1)/Math.sqrt(b);
      if(tr.kind==='powmhalf') return 1/(b*b);
      if(tr.kind==='exp') return Math.log(b);
      if(tr.kind==='log' || tr.kind==='logabs') return (x<0?-1:1)*Math.exp(b);
      return NaN;
    }
    function constDbMaxRelativeError(settings){
      const sig=typedInputPrecisionForDouble(settings);
      const scaleDigits=typedDecimalScaleDigitsForDouble(settings);
      const bySig=Math.pow(10, -Math.max(0, sig-3));
      const byScale=Math.pow(10, -Math.max(0, scaleDigits-3));
      return Math.min(0.25, Math.max(bySig, byScale));
    }
    function constDbBuildRow(settings, tr, c, expr, bPred, method, err, extra={}){
      const out=constDbApplyInverse(settings,tr,expr);
      const predicted=constDbPredictedFromB(settings,tr,bPred);
      if(!Number.isFinite(predicted)) return null;
      const rel=Math.abs(predicted-settings.target)/Math.max(1,Math.abs(settings.target));
      const sig=typedInputPrecision(settings);
      if(rel>constDbMaxRelativeError(settings)) return null;
      const sourceNote=c.source==='uploaded190' ? 'uploaded 190-constant database' : 'generated basic constant';
      const desc=c.description ? escapeHtml(c.description) : '';
      const notation=constDbDisplayNotation(c);
      const valueHtml=`<div><b>c = ${escapeHtml(notation)}</b> <span class="muted">(${escapeHtml(sourceNote)})</span></div>${desc?`<div class="muted">${desc}</div>`:''}<div>${escapeHtml(tr.label)} ≈ ${escapeHtml(fmtValue(bPred))}; ${escapeHtml(method)}</div>`;
      const row={
        candidate:`constant database: x ≈ ${out.text}`,
        latex:`x \\approx ${out.latex}`,
        copyLatex:`x \\approx ${out.latex}`,
        valueHtml,
        copyValue:`c = ${constDbDisplayNotation(c)}: ${c.description || ''}`,
        err:rel,
        errText:fmtErr(rel),
        constantDbCategory:method,
        constantDbSource:c.source,
        constantDbId:c.id,
        terms:Number(extra.terms||1),
        height:BigInt(extra.height||1),
        score:formulaVisibleLength(out.text)+Math.log10(1+Number(extra.height||1))*6+(extra.degree?extra.degree*4:0)+(tr.kind==='pow1'?0:8)
      };
      return row;
    }

    function constDbPriorityRecords(consts){
      const pri=new Map([
        ['basic_const',0], ['basic_e',1], ['basic_log',2], ['basic_const_2',3], ['basic_const_3',4],
        ['basic_sqrt_2',5], ['basic_sqrt_3',6], ['basic_phi',7], ['basic_gamma',8], ['basic_catalan',9],
        ['basic_log2',10], ['basic_log3',11], ['basic_log5',12], ['basic_loglogpi',13], ['basic_loglog2',14], ['basic_loglog3',15]
      ]);
      return (consts||[]).slice().sort((a,b)=>{
        const pa=pri.has(a.id)?pri.get(a.id):(a.source==='generated110'?40:80);
        const pb=pri.has(b.id)?pri.get(b.id):(b.source==='generated110'?40:80);
        return pa-pb || Math.abs(a.value)-Math.abs(b.value);
      });
    }
    function constDbIsPriorityNoiseConstant(c){
      const id=String(c?.id||'');
      const label=String(c?.label||'');
      const desc=String(c?.description||'');
      // v11.2.1: do not spend the prioritized deep-relation budget on exact
      // trigonometric algebraic values such as tan(pi/8).  They often generate
      // identities in c alone, e.g. b*(2+c-1/c)=0, which are mathematically true
      // for every input b and can crowd out meaningful constant-DB rows.  The
      // full-catalog direct pass still sees these constants; they are only
      // removed from the expensive prioritized deep/LLL scans.
      if(/^basic_(sin|cos|tan|arctan)_/.test(id)) return true;
      if(String(c?.source||'')==='generated110' && /\b(?:sin|cos|tan|arctan)\s*\(/i.test(label+' '+desc)) return true;
      return false;
    }
    function constDbPriorityRelationRecords(consts){
      return constDbPriorityRecords(consts).filter(c=>!constDbIsPriorityNoiseConstant(c));
    }
    function constDbAlgebraicRatioRow(settings,tr,c,b,ratio,found,methodPrefix){
      if(!found) return null;
      let coeff=(found.coeff||[]).map(x=>BigInt(Math.trunc(Number(x)||0)));
      if(coeff.length) coeff=normalizeCoeffs(coeff);
      const degree=polyDegree(coeff);
      const height=coeff.length ? coeffHeight(coeff) : BigInt(found.height||1n);
      const coeffNum=coeff.map(Number);
      const rootInfo=Number.isFinite(found.root) ? {root:Number(found.root), rootRel:Number(found.rootRel||0)} : constDbRootInfoNearRatio(coeffNum, ratio, typedInputPrecisionForDouble(settings));
      if(!rootInfo) return null;
      let expr=null;
      if(degree<=2) expr=constDbQuadraticExpr(coeffNum, ratio, c);
      else expr={text:'α·c', latex:String.raw`\alpha c`};
      if(!expr) return null;
      const method = degree===1 ? 'degree-1 ratio b/c' : (
        degree===2 ? 'degree-2 ratio b/c' : `${methodPrefix || ('degree-'+degree+' algebraic ratio b/c')}; α root of ${constDbPolyToInline(coeffNum,'α')}`
      );
      return constDbBuildRow(settings,tr,c,expr,rootInfo.root*c.value,method,found.err,{height:Number(height||1n),degree,terms:found.terms||degree+1});
    }
    function constDbNearestRationalApprox(x, maxDen=24, maxNum=96){
      if(!Number.isFinite(x)) return null;
      const neg=x<0; const ax=Math.abs(x);
      let best=null;
      for(let q=1;q<=maxDen;q++){
        const p=Math.round(ax*q);
        if(p===0 || Math.abs(p)>maxNum) continue;
        const val=p/q;
        const err=Math.abs(val-ax)/Math.max(1,ax);
        if(!best || err<best.err) best={p:BigInt(neg?-p:p), q:BigInt(q), err};
      }
      return best;
    }
    function constDbBuildApproxRow(settings, tr, c, expr, bPred, method, extra={}){
      const out=constDbApplyInverse(settings,tr,expr);
      const predicted=constDbPredictedFromB(settings,tr,bPred);
      if(!Number.isFinite(predicted)) return null;
      const rel=Math.abs(predicted-settings.target)/Math.max(1,Math.abs(settings.target));
      if(!Number.isFinite(rel)) return null;
      if(rel>constDbMaxRelativeError(settings)) return null;
      const sourceNote=c.source==='uploaded190' ? 'uploaded 190-constant database' : 'generated basic constant';
      const desc=c.description ? escapeHtml(c.description) : '';
      const notation=constDbDisplayNotation(c);
      const valueHtml=`<div><b>c = ${escapeHtml(notation)}</b> <span class="muted">(${escapeHtml(sourceNote)})</span></div>${desc?`<div class="muted">${desc}</div>`:''}<div>${escapeHtml(tr.label)} ≈ ${escapeHtml(fmtValue(bPred))}; ${escapeHtml(method)}</div><div class="muted">fallback row: approximate nearest database relation, not an exact certificate.</div>`;
      return {
        candidate:`constant database: x ≈ ${out.text}`,
        latex:`x \\approx ${out.latex}`,
        copyLatex:`x \\approx ${out.latex}`,
        valueHtml,
        copyValue:`c = ${constDbDisplayNotation(c)}: ${c.description || ''}`,
        err:rel,
        errText:fmtErr(rel),
        constantDbCategory:method,
        constantDbSource:c.source,
        constantDbId:c.id,
        terms:Number(extra.terms||1),
        height:BigInt(extra.height||1),
        score:1000 + rel*1e4 + formulaVisibleLength(out.text)+Math.log10(1+Number(extra.height||1))*6+(tr.kind==='pow1'?0:8)
      };
    }
    function constDbApproxFallbackRows(settings,consts,variants,count){
      const out=[]; const seen=new Set();
      const ordered=constDbPriorityRecords(consts);
      for(const tr of variants){
        if(tr.kind!=='pow1') continue;
        const b=tr.y; if(!Number.isFinite(b) || Math.abs(b)>1e100) continue;
        for(const c of ordered){
          const cv=c.value; if(!Number.isFinite(cv) || cv===0) continue;
          const rr=constDbNearestRationalApprox(b/cv, 24, 96);
          if(!rr) continue;
          const expr=constDbMulConstExpr(rr,c);
          const bPred=Number(rr.p)/Number(rr.q)*cv;
          const row=constDbBuildApproxRow(settings,tr,c,expr,bPred,'nearest rational multiple of database constant', {height:Number(absBig(rr.p)>absBig(rr.q)?absBig(rr.p):absBig(rr.q)),degree:1,terms:2});
          if(!row) continue;
          const k=normalizeResultTextKey(row.candidate)+'|'+c.id+'|'+tr.kind;
          if(seen.has(k)) continue; seen.add(k); out.push(row);
        }
      }
      out.sort((a,b)=>(a.err||1e9)-(b.err||1e9) || (a.score||1e9)-(b.score||1e9));
      return out.slice(0,Math.max(0,count||8));
    }
    function constDbRelationSearchHeight(sig){ return Math.max(5, Math.min(sig<=8 ? 8 : 14, sig<=6 ? 6 : (sig<=12 ? 10 : 12))); }
    function constDbValidateLinearRelation(vals, coeff, maxHeight, relTol, required=[]){
      if(!Array.isArray(vals) || !Array.isArray(coeff) || coeff.length<vals.length) return null;
      const maxH=BigInt(constDbBoundedMaxHeight(maxHeight));
      const cc=normalizeVector(coeff.slice(0,vals.length).map(x=>BigInt(x)));
      const h=coeffHeight(cc); if(h===0n || h>maxH) return null;
      if(Array.isArray(required) && required.length && !required.every(i=>cc[i]!==0n)) return null;
      let val=0, norm=1, terms=0;
      for(let i=0;i<vals.length;i++){
        const ci=Number(cc[i]);
        if(ci) terms++;
        val += ci*vals[i];
        norm=Math.max(norm, Math.abs(ci*vals[i]));
      }
      if(terms<2) return null;
      const rel=Math.abs(val)/norm; if(!Number.isFinite(rel) || rel>relTol) return null;
      const score=terms*6+Number(h)+rel*1e7;
      return {coeff:cc, err:rel, height:h, terms, score};
    }
    function constDbPslqLinearRelation(vals, sig, maxHeight, deadline=0, relTol=null, required=[]){
      if(!Array.isArray(vals) || vals.length<3 || vals.some(v=>!Number.isFinite(v) || Math.abs(v)>1e100)) return null;
      relTol = relTol || typedRelativeToleranceNumber(sig, 18, 1, 14);
      const dim=vals.length, maxH=constDbBoundedMaxHeight(maxHeight);
      const bits=constDbPslqBits(sig, dim);
      const fixed=constDbFixedVectorFromValues(vals, bits);
      if(!fixed) return null;
      const localDeadline=Math.min(deadline||Infinity, performance.now()+Math.max(3, Math.min(6, 2+dim)));
      try{
        const rel=pslqFixed(fixed, bits, BigInt(maxH+1), Math.max(280, dim*180), localDeadline);
        if(!rel) return null;
        return constDbValidateLinearRelation(vals, rel, maxH, relTol, required);
      }catch(e){ return null; }
    }
    function constDbFindLinearRelation(vals, sig, maxHeight, deadline=0, relTol=null, required=[]){
      if(!Array.isArray(vals) || vals.length<3 || vals.some(v=>!Number.isFinite(v) || Math.abs(v)>1e100)) return null;
      relTol = relTol || typedRelativeToleranceNumber(sig, 18, 1, 14);
      const dim=vals.length;
      const maxH=constDbBoundedMaxHeight(maxHeight);
      let best=null;
      const pslq=constDbPslqLinearRelation(vals, sig, maxH, deadline, relTol, required);
      if(pslq) best=pslq;
      const prec=Math.max(7, Math.min(15, typedInputPrecisionForDouble(sig)));
      const scaleNum=Math.pow(10,prec);
      const lattice=[];
      for(let i=0;i<dim;i++){
        const row=Array(dim+1).fill(0n);
        row[i]=1n;
        row[dim]=BigInt(Math.round(vals[i]*scaleNum));
        lattice.push(row);
      }
      const seenRows=new Set();
      function considerReduced(rs){
        for(const rr of rs||[]){
          if(deadline && performance.now()>deadline) break;
          const key=rr.join(','); if(seenRows.has(key)) continue; seenRows.add(key);
          const got=constDbValidateLinearRelation(vals, rr.slice(0,dim), maxH, relTol, required);
          if(!got) continue;
          if(!best || got.score<best.score) best=got;
        }
      }
      const localDeadline=deadline || performance.now()+14;
      try{ considerReduced(constDbFastLLLReduce(lattice.map(r=>r.slice()),0.84,2200,Math.min(localDeadline, performance.now()+18),1e16)); }catch(e){}
      if(!best && performance.now()<localDeadline && constDbExactFallbackAllowed(localDeadline,250)){
        try{ considerReduced(exactLLLReduce(lattice.map(r=>r.slice()),99n,100n,performance.now()+Math.min(10, Math.max(2,localDeadline-performance.now())))); }catch(e){}
      }
      return best;
    }
    function constDbPolyTermExpr(power, c){
      const ce=constDbConstExpr(c);
      if(power===0) return {text:'1', latex:'1'};
      if(power===1) return ce;
      return {text:`${ce.text}^${power}`, latex:`${ce.latex}^{${power}}`};
    }
    function constDbRecipConstExpr(c){
      const ce=constDbConstExpr(c);
      return {text:`1/${ce.text}`, latex:`\\frac{1}{${ce.latex}}`};
    }
    function constDbFormatTermCoeff(coeff, term, first){
      coeff=BigInt(coeff); if(coeff===0n) return {text:'', latex:''};
      const neg=coeff<0n, mag=neg?-coeff:coeff;
      const signText=first ? (neg?'−':'') : (neg?' − ':' + ');
      const signLatex=first ? (neg?'-':'') : (neg?' - ':' + ');
      const isOne=term.text==='1';
      let bodyText=isOne ? mag.toString() : (mag===1n ? term.text : `${mag.toString()}·${term.text}`);
      let bodyLatex=isOne ? mag.toString() : (mag===1n ? term.latex : `${mag.toString()}\\,${term.latex}`);
      return {text:signText+bodyText, latex:signLatex+bodyLatex};
    }
    function constDbSumExpr(items){
      const parts=[], lparts=[];
      for(const it of items){
        const f=constDbFormatTermCoeff(it.coeff, it.term, parts.length===0);
        if(f.text){ parts.push(f.text); lparts.push(f.latex); }
      }
      return {text:parts.join('')||'0', latex:lparts.join('')||'0'};
    }
    function constDbDivideExpr(num, den){
      den=BigInt(den);
      if(den<0n){ den=-den; num.items=num.items.map(it=>({coeff:-BigInt(it.coeff), term:it.term})); }
      const ne=constDbSumExpr(num.items);
      if(den===1n) return ne;
      return {text:`${constDbParenText(ne.text)}/${den.toString()}`, latex:`\\frac{${ne.latex}}{${den.toString()}}`};
    }
    function constDbPolynomialInCExpr(coeffB, coeffs, c){
      // coeffB*b + coeffs[0] + coeffs[1]*c + coeffs[2]*c^2 + coeffs[-1]/c = 0.
      const items=[];
      for(const [key,val] of Object.entries(coeffs)){
        const cc=BigInt(val); if(cc===0n) continue;
        let term;
        if(key==='0') term={text:'1', latex:'1'};
        else if(key==='1') term=constDbPolyTermExpr(1,c);
        else if(key==='2') term=constDbPolyTermExpr(2,c);
        else if(key==='3') term=constDbPolyTermExpr(3,c);
        else if(key==='-1') term=constDbRecipConstExpr(c);
        else continue;
        items.push({coeff:-cc, term});
      }
      if(!items.length) return null;
      return constDbDivideExpr({items}, coeffB);
    }
    function constDbRelationFromCoeff(c, coeff, keys, method){
      if(!coeff || coeff.length<2 || coeff[0]===0n) return null;
      const coeffs={};
      let rhs=0;
      for(let i=1;i<coeff.length && i<=keys.length;i++){
        const key=keys[i-1], ci=BigInt(coeff[i]);
        coeffs[key]=ci;
        if(key==='0') rhs += Number(ci);
        else if(key==='1') rhs += Number(ci)*c.value;
        else if(key==='2') rhs += Number(ci)*c.value*c.value;
        else if(key==='3') rhs += Number(ci)*c.value*c.value*c.value;
        else if(key==='-1') rhs += Number(ci)/c.value;
      }
      const expr=constDbPolynomialInCExpr(BigInt(coeff[0]), coeffs, c);
      if(!expr) return null;
      return {expr, yy:-rhs/Number(coeff[0]), method};
    }
    function constDbTryRelation_b_1_c_c2(settings,tr,c,b,sig,deadline,relTol){
      const vals=[b,1,c.value,c.value*c.value];
      const maxH=CONSTDB_RELATION_COEFF_BOUND;
      const rel=constDbFindLinearRelation(vals, sig, maxH, Math.min(deadline||Infinity, performance.now()+24), relTol, [0]);
      if(rel && rel.coeff[0]!==0n){
        const out=constDbRelationFromCoeff(c, rel.coeff, ['0','1','2'], 'quadratic relation in b,1,c,c^2');
        if(out) return {...out, err:rel.err, height:Number(rel.height), terms:rel.terms, score:rel.score};
      }
      if(deadline) return null;
      const H=constDbRelationSearchHeight(sig);
      let best=null;
      for(let kb=-H;kb<=H;kb++){
        if(kb===0) continue;
        for(let k1=-H;k1<=H;k1++) for(let kc=-H;kc<=H;kc++) for(let kc2=-H;kc2<=H;kc2++){
          if(k1===0 && kc===0 && kc2===0) continue;
          const val=kb*b+k1+kc*c.value+kc2*c.value*c.value;
          const norm=Math.max(1,Math.abs(kb*b),Math.abs(k1),Math.abs(kc*c.value),Math.abs(kc2*c.value*c.value));
          const er=Math.abs(val)/norm; if(er>relTol) continue;
          const h=Math.max(Math.abs(kb),Math.abs(k1),Math.abs(kc),Math.abs(kc2));
          const terms=(kb?1:0)+(k1?1:0)+(kc?1:0)+(kc2?1:0);
          const score=terms*6+h+er*1e7;
          if(!best || score<best.score){
            const expr=constDbPolynomialInCExpr(BigInt(kb), {'0':BigInt(k1),'1':BigInt(kc),'2':BigInt(kc2)}, c);
            if(expr) best={expr, yy:-(k1+kc*c.value+kc2*c.value*c.value)/kb, err:er, height:h, terms, score, method:'quadratic relation in b,1,c,c^2'};
          }
        }
      }
      return best;
    }
    function constDbTryRelation_b_1_c_c2_c3(settings,tr,c,b,sig,deadline,relTol){
      const c2=c.value*c.value, c3=c2*c.value;
      if(!Number.isFinite(c3) || Math.abs(c3)>1e100) return null;
      const vals=[b,1,c.value,c2,c3];
      const rel=constDbFindLinearRelation(vals, sig, CONSTDB_RELATION_COEFF_BOUND, Math.min(deadline||Infinity, performance.now()+28), relTol, [0]);
      if(!rel || rel.coeff[0]===0n) return null;
      const out=constDbRelationFromCoeff(c, rel.coeff, ['0','1','2','3'], 'cubic relation in b,1,c,c^2,c^3');
      if(!out) return null;
      return {...out, err:rel.err, height:Number(rel.height), terms:rel.terms, score:rel.score};
    }
    function constDbTryRelation_b_1_c_invc(settings,tr,c,b,sig,deadline,relTol){
      const invc=1/c.value;
      if(!Number.isFinite(invc)) return null;
      const vals=[b,1,c.value,invc];
      const rel=constDbFindLinearRelation(vals, sig, CONSTDB_RELATION_COEFF_BOUND, Math.min(deadline||Infinity, performance.now()+24), relTol, [0]);
      if(rel && rel.coeff[0]!==0n){
        const out=constDbRelationFromCoeff(c, rel.coeff, ['0','1','-1'], 'reciprocal relation in b,1,c,1/c');
        if(out) return {...out, err:rel.err, height:Number(rel.height), terms:rel.terms, score:rel.score};
      }
      if(deadline) return null;
      const H=constDbRelationSearchHeight(sig);
      let best=null;
      for(let kb=-H;kb<=H;kb++){
        if(kb===0) continue;
        for(let k1=-H;k1<=H;k1++) for(let kc=-H;kc<=H;kc++) for(let ki=-H;ki<=H;ki++){
          if(k1===0 && kc===0 && ki===0) continue;
          const val=kb*b+k1+kc*c.value+ki*invc;
          const norm=Math.max(1,Math.abs(kb*b),Math.abs(k1),Math.abs(kc*c.value),Math.abs(ki*invc));
          const er=Math.abs(val)/norm; if(er>relTol) continue;
          const h=Math.max(Math.abs(kb),Math.abs(k1),Math.abs(kc),Math.abs(ki));
          const terms=(kb?1:0)+(k1?1:0)+(kc?1:0)+(ki?1:0);
          const score=terms*6+h+er*1e7;
          if(!best || score<best.score){
            const expr=constDbPolynomialInCExpr(BigInt(kb), {'0':BigInt(k1),'1':BigInt(kc),'-1':BigInt(ki)}, c);
            if(expr) best={expr, yy:-(k1+kc*c.value+ki*invc)/kb, err:er, height:h, terms, score, method:'reciprocal relation in b,1,c,1/c'};
          }
        }
      }
      return best;
    }
    function constDbTryRelation_b_1_c_c2_c3_invc(settings,tr,c,b,sig,deadline,relTol){
      const cv=c.value, c2=cv*cv, c3=c2*cv, invc=1/cv;
      if(!Number.isFinite(c3) || !Number.isFinite(invc) || Math.abs(c3)>1e100) return null;
      const vals=[b,1,cv,c2,c3,invc];
      const rel=constDbFindLinearRelation(vals, sig, CONSTDB_RELATION_COEFF_BOUND, Math.min(deadline||Infinity, performance.now()+26), relTol, [0]);
      if(!rel || rel.coeff[0]===0n) return null;
      const hasC2=rel.coeff[3]!==0n, hasC3=rel.coeff[4]!==0n, hasInv=rel.coeff[5]!==0n;
      let keys=null, method='', degree=1;
      if(hasC3 && hasInv) return null; // avoid broad mixed fits that are not an intended relation family
      if(hasC3){ if(rel.terms>3) return null; keys=['0','1','2','3']; method='cubic relation in b,1,c,c^2,c^3'; degree=3; }
      else if(hasInv && !hasC2){ keys=['0','1','-1']; method='reciprocal relation in b,1,c,1/c'; degree=2; }
      else if(!hasInv){ keys=['0','1','2']; method='quadratic relation in b,1,c,c^2'; degree=hasC2?2:1; }
      else return null;
      const coeff=rel.coeff.slice(0,1).concat(keys.map((key)=>{ const idx={'0':1,'1':2,'2':3,'3':4,'-1':5}[key]; return rel.coeff[idx]||0n; }));
      const out=constDbRelationFromCoeff(c, coeff, keys, method);
      if(!out) return null;
      return {...out, err:rel.err, height:Number(rel.height), terms:rel.terms, score:rel.score, degree};
    }
    function constDbTryPolynomialInCOverSmallDen(settings,tr,c,b,sig,deadline,relTol){
      // v11.1.1: fast targeted pass for transformed values such as log|x| that
      // are low-degree polynomials in a database constant with a small rational
      // denominator, e.g. log(x)=(-12+4π+π²)/12.  The older general scan could
      // find this internally, but log|x| was late enough in the transform order
      // that the wall-clock budget was often exhausted first.
      const H=Math.max(6, constDbRelationSearchHeight(sig));
      const maxDen=Math.max(12, Math.min(36, H*3));
      const cv=c.value; if(!Number.isFinite(cv)) return null;
      let best=null;
      for(let den=1; den<=maxDen; den++){
        for(let a1=-H; a1<=H; a1++) for(let a2=-H; a2<=H; a2++){
          if(deadline && performance.now()>deadline) return best;
          const a0=Math.round(den*b - a1*cv - a2*cv*cv);
          if(Math.abs(a0)>maxDen) continue;
          if(a0===0 && a1===0 && a2===0) continue;
          const yy=(a0 + a1*cv + a2*cv*cv)/den;
          if(!Number.isFinite(yy)) continue;
          const er=Math.abs(yy-b)/Math.max(1,Math.abs(b)); if(er>relTol) continue;
          const h=Math.max(Math.abs(a0),Math.abs(a1),Math.abs(a2),den);
          const terms=(a0?1:0)+(a1?1:0)+(a2?1:0)+1;
          const score=terms*5+h+er*1e7;
          if(!best || score<best.score){
            const expr=constDbPolynomialInCExpr(BigInt(den), {'0':BigInt(-a0),'1':BigInt(-a1),'2':BigInt(-a2)}, c);
            if(expr) best={expr, yy, err:er, height:h, terms, score, method:'low-degree polynomial relation in transformed value,1,c,c^2'};
          }
        }
      }
      return best;
    }
    function constDbPriorityTransformedPolynomialRows(settings,consts,variants,sig,relTol,deadline){
      const rows=[];
      const priority=constDbPriorityRecords(consts).slice(0,48);
      const trs=variants.filter(tr=>tr && (tr.kind==='log' || tr.kind==='logabs' || tr.kind==='exp')).sort((a,b)=>((a.kind==='log'||a.kind==='logabs')?0:1)-((b.kind==='log'||b.kind==='logabs')?0:1));
      for(const tr of trs){
        const b=tr.y; if(!Number.isFinite(b) || Math.abs(b)>1e100) continue;
        for(const c of priority){
          if(deadline && performance.now()>deadline) return rows;
          if(!Number.isFinite(c.value) || c.value===0) continue;
          const rel=constDbTryPolynomialInCOverSmallDen(settings,tr,c,b,sig,Math.min(deadline||Infinity, performance.now()+30),relTol);
          if(!rel) continue;
          rows.push(constDbBuildRow(settings,tr,c,rel.expr,rel.yy,rel.method,rel.err,{height:rel.height,terms:rel.terms,degree:2}));
          if(rows.length>=4) return rows;
        }
      }
      return rows;
    }
    function constDbTermForPool(id, b, c){
      const ce=constDbConstExpr(c);
      if(id==='one') return {id, value:1, text:'1', latex:'1'};
      if(id==='b') return {id, value:b, text:'b', latex:'b'};
      if(id==='c') return {id, value:c.value, text:'c', latex:'c'};
      if(id==='bc') return {id, value:b*c.value, text:'b·c', latex:'bc'};
      if(id==='b2') return {id, value:b*b, text:'b^2', latex:'b^2'};
      if(id==='c2') return {id, value:c.value*c.value, text:'c^2', latex:'c^2'};
      if(id==='invb') return {id, value:1/b, text:'1/b', latex:'\\frac{1}{b}'};
      if(id==='invc') return {id, value:1/c.value, text:'1/c', latex:'\\frac{1}{c}'};
      if(id==='bdivc') return {id, value:b/c.value, text:'b/c', latex:'\\frac{b}{c}'};
      if(id==='cdivb') return {id, value:c.value/b, text:'c/b', latex:'\\frac{c}{b}'};
      return null;
    }
    function constDbTermBPower(id){
      if(id==='one' || id==='c' || id==='c2' || id==='invc') return 0;
      if(id==='b' || id==='bc' || id==='bdivc') return 1;
      if(id==='b2') return 2;
      if(id==='invb' || id==='cdivb') return -1;
      return null;
    }
    function constDbRelationUsesTargetNontrivially(terms, coeff){
      const powers=new Set();
      for(let i=0;i<terms.length;i++){
        if(BigInt(coeff[i]||0)===0n) continue;
        const p=constDbTermBPower(terms[i]?.id);
        if(p===null) return true;
        powers.add(p);
      }
      // Relations with only one b-power are just f(c)=0, b*f(c)=0,
      // b^2*f(c)=0, or f(c)/b=0.  They do not constrain the input value.
      return powers.size>=2;
    }
    function constDbTermWithTransformLabel(term, tr){
      const bl=constDbTransformLabelLatex(tr);
      if(term.id==='b') return {text:tr.label, latex:bl};
      if(term.id==='bc') return {text:`${tr.label}·c`, latex:`${bl}c`};
      if(term.id==='b2') return {text:`${tr.label}^2`, latex:`${bl}^2`};
      if(term.id==='invb') return {text:`1/${tr.label}`, latex:`\\frac{1}{${bl}}`};
      if(term.id==='bdivc') return {text:`${tr.label}/c`, latex:`\\frac{${bl}}{c}`};
      if(term.id==='cdivb') return {text:`c/${tr.label}`, latex:`\\frac{c}{${bl}}`};
      return {text:term.text, latex:term.latex};
    }
    function constDbRelationText(terms, coeff, tr){
      const items=[];
      for(let i=0;i<terms.length;i++) if(coeff[i]!==0n) items.push({coeff:coeff[i], term:constDbTermWithTransformLabel(terms[i],tr)});
      const e=constDbSumExpr(items);
      return {text:`${e.text} = 0`, latex:`${e.latex} = 0`};
    }
    function constDbBuildImplicitRelationRow(settings,tr,c,terms,coeff,method,err,extra={}){
      if(!constDbRelationUsesTargetNontrivially(terms, coeff)) return null;
      const relExpr=constDbRelationText(terms, coeff, tr);
      const sourceNote=c.source==='uploaded190' ? 'uploaded 190-constant database' : 'generated basic constant';
      const desc=c.description ? escapeHtml(c.description) : '';
      const notation=constDbDisplayNotation(c);
      const valueHtml=`<div><b>c = ${escapeHtml(notation)}</b> <span class="muted">(${escapeHtml(sourceNote)})</span></div>${desc?`<div class="muted">${desc}</div>`:''}<div>${escapeHtml(method)}</div>`;
      return {
        candidate:`constant database: ${relExpr.text}`,
        latex:relExpr.latex,
        copyLatex:relExpr.latex,
        valueHtml,
        copyValue:`c = ${constDbDisplayNotation(c)}: ${c.description || ''}`,
        err:err,
        errText:fmtErr(err),
        constantDbCategory:method,
        constantDbSource:c.source,
        constantDbId:c.id,
        terms:Number(extra.terms||coeff.filter(x=>x!==0n).length||1),
        height:BigInt(extra.height||coeffHeight(coeff)||1n),
        score:formulaVisibleLength(relExpr.text)+Math.log10(1+Number(extra.height||1))*6+18
      };
    }
    function constDbTransformLabelLatex(tr){
      const kind=String(tr?.kind||'');
      if(kind==='pow1') return 'x';
      if(kind==='powm1') return 'x^{-1}';
      if(kind==='powm2') return 'x^{-2}';
      if(kind==='powhalf') return 'x^{1/2}';
      if(kind==='powmhalf') return 'x^{-1/2}';
      if(kind==='pow2') return 'x^2';
      if(kind==='exp') return 'e^x';
      if(kind==='log') return String.raw`\log x`;
      if(kind==='logabs') return String.raw`\log|x|`;
      return String(tr?.label||'b').replace(/log\(x\)/g,String.raw`\log x`).replace(/log\|x\|/g,String.raw`\log|x|`);
    }
    function logConstLatex(c){
      const id=String(c?.id||'');
      if(id==='one') return '1';
      if(id==='log2') return String.raw`\log 2`;
      if(id==='log3') return String.raw`\log 3`;
      if(id==='log5') return String.raw`\log 5`;
      if(id==='pi') return String.raw`\pi`;
      if(id==='logpi') return String.raw`\log \pi`;
      if(id==='loglogpi') return String.raw`\log(\log \pi)`;
      if(id==='loglog2') return String.raw`\log(\log 2)`;
      if(id==='loglog3') return String.raw`\log(\log 3)`;
      if(id==='loggamma16') return String.raw`\log\Gamma(1/6)`;
      if(id==='log7') return String.raw`\log 7`;
      if(id==='log11') return String.raw`\log 11`;
      if(id==='e') return 'e';
      if(id==='loggamma13') return String.raw`\log\Gamma(1/3)`;
      if(id==='loggamma14') return String.raw`\log\Gamma(1/4)`;
      if(id==='eulergamma') return String.raw`\gamma`;
      if(id==='logeulergamma') return String.raw`\log\gamma`;
      if(id==='logG') return String.raw`\log G`;
      if(id==='logzeta3') return String.raw`\log\zeta(3)`;
      if(id==='logzeta5') return String.raw`\log\zeta(5)`;
      if(id==='logphi') return String.raw`\log\varphi`;
      if(id==='logA') return String.raw`\log A`;
      return escapeLatex(String(c?.label||id||'u'));
    }
    function constDbLogOptionalTerms(){
      const order=['pi','log5','loglogpi','loglog2','loglog3','loggamma16','log7','log11','e','loggamma13','loggamma14','eulergamma','logeulergamma','logG','logzeta3','logzeta5','logphi','logA'];
      const byId=new Map(logConstants.map(c=>[c.id,c]));
      return order.map(id=>byId.get(id)).filter(Boolean).map(c=>({id:c.id, value:c.value, term:{text:c.label, latex:logConstLatex(c)}}));
    }
    function constDbLogBaseTerms(tr,c,b){
      if(!(b>0) || !(c.value>0)) return null;
      const bl=constDbTransformLabelLatex(tr);
      return [
        {id:'logb', value:Math.log(b), term:{text:`log(${tr.label})`, latex:String.raw`\log\left(${bl}\right)`}},
        {id:'logc', value:Math.log(c.value), term:{text:'log(c)', latex:String.raw`\log(c)`}},
        {id:'logpi', value:Math.log(Math.PI), term:{text:'log(π)', latex:String.raw`\log\pi`}},
        {id:'one', value:1, term:{text:'1', latex:'1'}},
        {id:'log2', value:Math.log(2), term:{text:'log(2)', latex:String.raw`\log 2`}},
        {id:'log3', value:Math.log(3), term:{text:'log(3)', latex:String.raw`\log 3`}}
      ];
    }
    function constDbLogLinearRelationText(terms, coeff){
      const items=[];
      for(let i=0;i<terms.length;i++) if(coeff[i]!==0n) items.push({coeff:coeff[i], term:terms[i].term});
      const e=constDbSumExpr(items);
      return {text:`${e.text} = 0`, latex:`${e.latex} = 0`};
    }
    function constDbBuildLogLinearRow(settings,tr,c,terms,coeff,method,err,extra={}){
      const relExpr=constDbLogLinearRelationText(terms, coeff);
      const sourceNote=c.source==='uploaded190' ? 'uploaded 190-constant database' : 'generated basic constant';
      const desc=c.description ? escapeHtml(c.description) : '';
      const notation=constDbDisplayNotation(c);
      const added=extra.added ? `<div class="muted">added: ${escapeHtml(extra.added)}</div>` : '';
      const valueHtml=`<div><b>c = ${escapeHtml(notation)}</b> <span class="muted">(${escapeHtml(sourceNote)})</span></div>${desc?`<div class="muted">${desc}</div>`:''}<div>${escapeHtml(method)}</div>${added}`;
      return {
        candidate:`constant database: ${relExpr.text}`,
        latex:relExpr.latex,
        copyLatex:relExpr.latex,
        valueHtml,
        copyValue:`c = ${constDbDisplayNotation(c)}: ${c.description || ''}`,
        err:err,
        errText:fmtErr(err),
        constantDbCategory:method,
        constantDbSource:c.source,
        constantDbId:c.id,
        terms:Number(extra.terms||coeff.filter(x=>x!==0n).length||1),
        height:BigInt(extra.height||coeffHeight(coeff)||1n),
        score:formulaVisibleLength(relExpr.text)+Math.log10(1+Number(extra.height||1))*6+22
      };
    }
    function constDbFindLinearRelationSmall(vals, maxHeight, deadline=0, relTol=1e-12, required=[]){
      if(!Array.isArray(vals) || vals.length<2 || vals.some(v=>!Number.isFinite(v))) return null;
      const dim=vals.length;
      const H=Math.min(3, Math.max(1, Number(maxHeight)||3));
      const coeff=Array(dim).fill(0);
      let best=null; const seen=new Set();
      function score(){
        if(deadline && performance.now()>deadline) return;
        const cc=normalizeVector(coeff.map(x=>BigInt(x)));
        const key=cc.join(','); if(seen.has(key)) return; seen.add(key);
        const h=coeffHeight(cc); if(h===0n || h>BigInt(maxHeight)) return;
        if(Array.isArray(required) && required.length && !required.every(i=>cc[i]!==0n)) return;
        let val=0, norm=1, terms=0;
        for(let i=0;i<dim;i++){
          const ai=Number(cc[i]); if(ai) terms++;
          val += ai*vals[i]; norm=Math.max(norm, Math.abs(ai*vals[i]));
        }
        if(terms<2) return;
        const rel=Math.abs(val)/norm; if(rel>relTol) return;
        const score=terms*6+Number(h)+rel*1e7;
        if(!best || score<best.score) best={coeff:cc, err:rel, height:h, terms, score};
      }
      function rec(pos){
        if(deadline && performance.now()>deadline) return;
        if(pos>=dim){ score(); return; }
        for(let a=-H;a<=H;a++){ coeff[pos]=a; rec(pos+1); if(deadline && performance.now()>deadline) return; }
      }
      rec(0);
      return best;
    }
    function constDbLogLinearRows(settings,tr,c,b,sig,deadline,relTol){
      const level=Math.max(4, Number(settings.level||4));
      if(level<5) return [];
      const base=constDbLogBaseTerms(tr,c,b); if(!base) return [];
      const optCount=Math.max(0, Math.min(2, Math.floor(level)-5));
      const optionals=constDbLogOptionalTerms().filter(t=>Number.isFinite(t.value));
      const combos=optCount===0 ? [[]] : constDbCombinations(optionals, optCount, deadline, optCount===1 ? optionals.length : 120);
      const rows=[];
      const maxH=constDbRelationSearchHeight(sig)+8;
      for(const combo of combos){
        if(deadline && performance.now()>deadline) break;
        const terms=base.concat(combo);
        let rel=constDbFindLinearRelation(terms.map(t=>t.value), sig, maxH, Math.min(deadline||Infinity, performance.now()+10), relTol);
        if(!rel || rel.coeff[0]===0n || rel.coeff[1]===0n) rel=constDbFindLinearRelationSmall(terms.map(t=>t.value), maxH, Math.min(deadline||Infinity, performance.now()+10), relTol, [0,1]);
        if(!rel) continue;
        if(rel.coeff[0]===0n || rel.coeff[1]===0n) continue;
        const added=combo.map(t=>t.term.text).join(', ');
        const method=optCount===0 ? 'log-linear LLL relation in log b, log c, log π, 1, log 2, log 3' : `log-linear LLL relation with ${optCount} added candidate${optCount===1?'':'s'}`;
        rows.push(constDbBuildLogLinearRow(settings,tr,c,terms,rel.coeff,method,rel.err,{height:Number(rel.height),terms:rel.terms,added}));
        if(rows.length>=2) break;
      }
      return rows;
    }
    function constDbCombinations(arr, k, deadline, limit=400){
      const out=[];
      function rec(start, cur){
        if(deadline && performance.now()>deadline) return;
        if(out.length>=limit) return;
        if(cur.length===k){ out.push(cur.slice()); return; }
        for(let i=start;i<=arr.length-(k-cur.length);i++){ cur.push(arr[i]); rec(i+1,cur); cur.pop(); if(out.length>=limit) return; }
      }
      rec(0,[]); return out;
    }
    function constDbExtraSubsetRows(settings,tr,c,b,sig,deadline,relTol){
      const level=Number(settings.level||4);
      const maxK=Math.max(4, Math.min(10, 4 + Math.max(0, Math.floor(level)-4)));
      if(maxK<5) return [];
      const ids=['one','b','c','bc','b2','c2','invb','invc','bdivc','cdivb'];
      const base=ids.map(id=>constDbTermForPool(id,b,c)).filter(t=>t && Number.isFinite(t.value) && Math.abs(t.value)<1e100);
      const rows=[]; const maxH=constDbRelationSearchHeight(sig)+3;
      for(let k=5;k<=maxK;k++){
        const combos=constDbCombinations(base,k,deadline,260);
        for(const terms of combos){
          if(deadline && performance.now()>deadline) return rows;
          const rel=constDbFindLinearRelation(terms.map(t=>t.value), sig, maxH, deadline, relTol);
          if(!rel) continue;
          // Prefer subsets that genuinely use b and c information.
          const used=terms.filter((t,i)=>rel.coeff[i]!==0n).map(t=>t.id);
          if(!used.some(id=>/^b|invb|bdivc|cdivb/.test(id)) || !used.some(id=>/c/.test(id))) continue;
          if(!constDbRelationUsesTargetNontrivially(terms, rel.coeff)) continue;
          rows.push(constDbBuildImplicitRelationRow(settings,tr,c,terms,rel.coeff,`${k}-term LLL subset relation`,rel.err,{height:Number(rel.height),terms:rel.terms}));
          if(rows.length>=3) return rows;
        }
      }
      return rows;
    }
    function constantDbBudgetMs(level, sig, settings=null){
      // v11.2: with the constant-DB-only fast floating LLL path, use explicit
      // search budgets for the main levels while preserving deeper/manual
      // behavior for levels above 6.
      const lv=Math.max(4, Number(level||4));
      if(lv===4) return stageBudgetValueToMs(settings?.stageBudgets?.constantDb4Ms, 20000, 0, 300000);
      if(lv===5) return stageBudgetValueToMs(settings?.stageBudgets?.constantDb5Ms, 45000, 0, 300000);
      if(lv===6) return stageBudgetValueToMs(settings?.stageBudgets?.constantDb6Ms, 135000, 0, 600000);
      return Math.ceil(riesLevelDefaultModuleBudgetMs(lv) * 1.2);
    }
    function constantDbRows(settings){
      if(!shouldRunConstantDbRows(settings)) return [];
      const consts=constantDbRecords(); if(!consts.length) return [];
      const sig=typedInputPrecisionForDouble(settings);
      const variants=constDbTransformRows(settings);
      const passes=settings?.constDbPasses || {rational:true,affine:true,quadratic:true,mobius:true,algebraic:true,log:true};
      const relTol=typedRelativeToleranceNumber(sig, 20, 1, 14);
      const rows=[]; const seen=new Set();
      const level=Math.max(4, Number(settings.level||4));
      const startTime=performance.now();
      const budgetMs=constantDbBudgetMs(level, sig, settings);
      const deadline=startTime+budgetMs;
      const localPolyMs = level>=6 ? 180 : (level>=5 ? 90 : 60);
      const localAlgMs = level>=6 ? 260 : (level>=5 ? 120 : 70);
      const localFormMs = level>=6 ? 120 : (level>=5 ? 70 : 40);
      const localLogMs = level>=6 ? 320 : (level>=5 ? 160 : 80);
      if(settings) settings._constantDbBudgetMs=budgetMs;
      function add(row){
        if(!row) return;
        const k=normalizeResultTextKey(row.candidate)+'|'+(row.constantDbId||'')+'|'+(row.constantDbCategory||'');
        if(seen.has(k)) return; seen.add(k); rows.push(row);
      }
      // Cheap full-catalog pass first.  This guarantees direct database hits and
      // tiny affine shifts such as x = 1 + c are found before the heavier LLL
      // scans spend their time budget on early uploaded constants.
      for(const tr of variants){
        if(performance.now()>deadline) break;
        const b=tr.y; if(!Number.isFinite(b) || Math.abs(b)>1e100) continue;
        for(const c of consts){
          const cv=c.value; if(!Number.isFinite(cv) || cv===0) continue;
          const ratio=b/cv;
          if(passes.rational!==false && Number.isFinite(ratio)){
            const rr=constDbRationalApprox(ratio, 18, 60, relTol);
            if(rr){
              const expr=constDbMulConstExpr(rr,c);
              add(constDbBuildRow(settings,tr,c,expr,Number(rr.p)/Number(rr.q)*cv,'degree-1 ratio b/c',rr.err,{height:Number(absBig(rr.p)>absBig(rr.q)?absBig(rr.p):absBig(rr.q)),degree:1,terms:2}));
            }
          }
          if(passes.affine!==false) for(let m=-6;m<=6;m++){
            if(m===0) continue;
            const a=Math.round(b-m*cv);
            if(Math.abs(a)>12) continue;
            const yy=a+m*cv;
            const er=Math.abs(yy-b)/Math.max(1,Math.abs(b));
            if(er>relTol) continue;
            const expr=constDbPolynomialInCExpr(1n, {'0':BigInt(-a),'1':BigInt(-m)}, c);
            if(expr) add(constDbBuildRow(settings,tr,c,expr,yy,'affine relation in b,1,c',er,{height:Math.max(Math.abs(a),Math.abs(m)),terms:2,degree:1}));
          }
          // Cheap low-height quadratic and reciprocal passes across the whole
          // catalog, so simple identities like b=1+c+c^2 or b=c+1/c are not
          // missed just because the heavier per-constant scans timed out.
          if(passes.quadratic!==false) for(let m1=-3;m1<=3;m1++) for(let m2=-3;m2<=3;m2++){
            if(m1===0 && m2===0) continue;
            const a=Math.round(b-m1*cv-m2*cv*cv);
            if(Math.abs(a)>12) continue;
            const yy=a+m1*cv+m2*cv*cv;
            const er=Math.abs(yy-b)/Math.max(1,Math.abs(b));
            if(er>relTol) continue;
            const expr=constDbPolynomialInCExpr(1n, {'0':BigInt(-a),'1':BigInt(-m1),'2':BigInt(-m2)}, c);
            if(expr) add(constDbBuildRow(settings,tr,c,expr,yy,'quadratic relation in b,1,c,c^2',er,{height:Math.max(Math.abs(a),Math.abs(m1),Math.abs(m2)),terms:3,degree:2}));
          }
          const invc=1/cv;
          if(passes.quadratic!==false && Number.isFinite(invc)){
            for(let m1=-3;m1<=3;m1++) for(let mi=-3;mi<=3;mi++){
              if(m1===0 && mi===0) continue;
              const a=Math.round(b-m1*cv-mi*invc);
              if(Math.abs(a)>12) continue;
              const yy=a+m1*cv+mi*invc;
              const er=Math.abs(yy-b)/Math.max(1,Math.abs(b));
              if(er>relTol) continue;
              const expr=constDbPolynomialInCExpr(1n, {'0':BigInt(-a),'1':BigInt(-m1),'-1':BigInt(-mi)}, c);
              if(expr) add(constDbBuildRow(settings,tr,c,expr,yy,'reciprocal relation in b,1,c,1/c',er,{height:Math.max(Math.abs(a),Math.abs(m1),Math.abs(mi)),terms:3,degree:2}));
            }
          }
        }
      }
      const priorityConsts=constDbPriorityRecords(consts);
      const relationPriorityConsts=constDbPriorityRelationRecords(consts);
      const deepConsts=relationPriorityConsts.slice(0, level<=4 ? 96 : 140);
      let deepDone=0; const deepTotal=deepConsts.length*variants.length;
      // v11.1.2: iterate the prioritized constants outside and transforms
      // inside.  This keeps π/e/log constants early while honoring the transform
      // order x, log|x|, e^x, 1/x, x^2 before the lower-value variants.
      for(const c of deepConsts){
        if(performance.now()>deadline) break;
        const cv=c.value; if(!Number.isFinite(cv) || cv===0) continue;
        const H=sig<=8 ? 5 : 8;
        const forms=constDbLinearForms(c,H);
        function nearForms(v){
          let lo=0, hi=forms.length;
          while(lo<hi){ const mid=(lo+hi)>>1; if(forms[mid].value<v) lo=mid+1; else hi=mid; }
          const out=[]; for(let k=Math.max(0,lo-3); k<Math.min(forms.length,lo+4); k++) out.push(forms[k]); return out;
        }
        for(const tr of variants){
          if(performance.now()>deadline) break;
          const b=tr.y; if(!Number.isFinite(b) || Math.abs(b)>1e100) continue;
          const ratio=b/cv;
          if((passes.quadratic!==false || passes.algebraic!==false) && Number.isFinite(ratio)){
            if(passes.quadratic!==false){
              const qr=constDbFindPolynomialRatio(ratio, 3, sig, Math.min(deadline, performance.now()+localPolyMs));
              const qrow=constDbAlgebraicRatioRow(settings,tr,c,b,ratio,qr,qr && qr.degree===3 ? 'degree-3 ratio b/c' : null);
              if(qrow) add(qrow);
            }
            if(passes.algebraic!==false && level>=5 && rows.length<22){
              const maxAlgDegree=Math.max(4, Math.min(8, Math.floor(level)-1));
              const alg=constDbFindAlgebraicRatioLLL(ratio, maxAlgDegree, sig, constDbRelationSearchHeight(sig)+8, Math.min(deadline, performance.now()+localAlgMs), relTol);
              if(alg && alg.degree>=4){
                const expr={text:'α·c', latex:String.raw`\alpha c`};
                const method=`degree-${alg.degree} algebraic ratio b/c; α root of ${constDbPolyToInline(alg.coeff.map(Number),'α')}`;
                add(constDbBuildRow(settings,tr,c,expr,ratio*cv,method,alg.err,{height:Number(alg.height||1n),degree:alg.degree,terms:alg.terms||alg.degree+1}));
              }
            }
          }
          if(passes.mobius!==false) for(const den of forms){
            if(performance.now()>deadline) break;
            const need=b*den.value;
            for(const num of nearForms(need)){
              const yy=num.value/den.value;
              if(!Number.isFinite(yy)) continue;
              const er=Math.abs(yy-b)/Math.max(1,Math.abs(b));
              if(er>relTol) continue;
              if(den.b===0 && den.a===1 && num.a===0) continue;
              const expr=constDbMobiusExpr(num,den,c);
              const h=Math.max(num.height,den.height);
              add(constDbBuildRow(settings,tr,c,expr,yy,'Möbius relation in 1,b,c,bc',er,{height:h,terms:num.terms+den.terms,degree:1}));
            }
          }
          if(passes.quadratic!==false){
            const lrel=constDbTryRelation_b_1_c_c2_c3_invc(settings,tr,c,b,sig,Math.min(deadline, performance.now()+localFormMs),relTol);
            if(lrel) add(constDbBuildRow(settings,tr,c,lrel.expr,lrel.yy,lrel.method,lrel.err,{height:lrel.height,terms:lrel.terms,degree:lrel.degree||2}));
          }
          if(passes.log!==false && level>=5 && rows.length<24){
            for(const rr of constDbLogLinearRows(settings,tr,c,b,sig,Math.min(deadline, performance.now()+localLogMs),relTol)) add(rr);
          }
          if(passes.log!==false && level>=5 && rows.length<18){
            for(const rr of constDbExtraSubsetRows(settings,tr,c,b,sig,Math.min(deadline, performance.now()+localLogMs),relTol)) add(rr);
          }
          deepDone++;
        }
      }
      if(settings){ settings._constantDbDeepDone=deepDone; settings._constantDbDeepTotal=deepTotal; settings._constantDbDeepFrac=deepTotal ? deepDone/deepTotal : 1; }
      // Priority algebraic pass runs after the cheap and deep relation sweeps in
      // v11.1.2, so the reordered transform list gets the first chance to find
      // direct, log|x|, exp(x), reciprocal, and square-polynomial matches.
      const priorityAlgebraicConsts=relationPriorityConsts.slice(0,72);
      for(const tr of variants){
        if(performance.now()>deadline) break;
        const b=tr.y; if(!Number.isFinite(b) || Math.abs(b)>1e100) continue;
        for(const c of priorityAlgebraicConsts){
          if(performance.now()>deadline) break;
          const cv=c.value; if(!Number.isFinite(cv) || cv===0) continue;
          const ratio=b/cv; if(!Number.isFinite(ratio)) continue;
          if(passes.quadratic!==false){
            const qr=constDbFindPolynomialRatio(ratio, 3, sig, Math.min(deadline, performance.now()+localPolyMs));
            const qrow=constDbAlgebraicRatioRow(settings,tr,c,b,ratio,qr,qr && qr.degree===3 ? 'degree-3 ratio b/c' : null);
            if(qrow) add(qrow);
          }
          if(passes.algebraic!==false && level>=5 && rows.length<24){
            const maxAlgDegree=Math.max(4, Math.min(8, Math.floor(level)-1));
            const alg=constDbFindAlgebraicRatioLLL(ratio, maxAlgDegree, sig, constDbRelationSearchHeight(sig)+8, Math.min(deadline, performance.now()+localAlgMs), relTol);
            if(alg && alg.degree>=4){
              const arow=constDbAlgebraicRatioRow(settings,tr,c,b,ratio,alg,`degree-${alg.degree} algebraic ratio b/c`);
              if(arow) add(arow);
            }
          }
          if(performance.now()>deadline) break;
        }
      }

      let map=new Map();
      for(const r of rows){
        const k=normalizeResultTextKey(r.candidate);
        if(!map.has(k) || (r.score??1e9)<(map.get(k).score??1e9)) map.set(k,r);
      }
      let finalRows=[...map.values()].sort((a,b)=>(a.score??1e9)-(b.score??1e9) || (a.err||1)-(b.err||1));
      // v11: do not pad to eight rows with weak nearest-rational fallback
      // matches.  Returning fewer than eight rows is better than showing
      // approximate database coincidences whose error is far outside the typed
      // precision envelope.
      if(settings) settings._constantDbMs=Math.round(performance.now()-startTime);
      return finalRows.slice(0,8);
    }


    async function constantDbRowsAsync(settings, progressCb=null){
      // v11.1 cooperative UI version of constantDbRows(): same passes, but yields
      // between small batches so progress and the SO(4) animation keep painting.
      if(!shouldRunConstantDbRows(settings)) return [];
      const consts=constantDbRecords(); if(!consts.length) return [];
      const sig=typedInputPrecisionForDouble(settings);
      const variants=constDbTransformRows(settings);
      const passes=settings?.constDbPasses || {rational:true,affine:true,quadratic:true,mobius:true,algebraic:true,log:true};
      const relTol=typedRelativeToleranceNumber(sig, 20, 1, 14);
      const rows=[]; const seen=new Set();
      const level=Math.max(4, Number(settings.level||4));
      const startTime=performance.now();
      const budgetMs=constantDbBudgetMs(level, sig, settings);
      const deadline=startTime+budgetMs;
      // Keep individual synchronous inner probes short in the async solve path.
      // v11.4.2 has explicit 20/45/135 s level 4/5/6 budgets, but no single PSLQ/LLL/log
      // probe should monopolize the main thread long enough to freeze animation.
      const localPolyMs = level>=5 ? 45 : 35;
      const localAlgMs = level>=5 ? 55 : 40;
      const localFormMs = level>=5 ? 38 : 28;
      const localLogMs = level>=5 ? 55 : 40;
      if(settings) settings._constantDbBudgetMs=budgetMs;
      let lastYield=0;
      function add(row){ if(!row) return; const k=normalizeResultTextKey(row.candidate)+'|'+(row.constantDbId||'')+'|'+(row.constantDbCategory||''); if(seen.has(k)) return; seen.add(k); rows.push(row); }
      async function maybeYield(phase, frac, force=false){ const now=performance.now(); if(!force && now-lastYield<35 && now<deadline && !isUserInputPending()) return; lastYield=now; if(progressCb) progressCb({phase, frac:Math.max(0,Math.min(1,frac||0)), rows:rows.slice(), elapsed:Math.round(now-startTime), budgetMs}); await yieldToUI(); }
      await maybeYield('constant database start', .015, true);
      for(let ti=0; ti<variants.length; ti++){
        if(performance.now()>deadline) break;
        const tr=variants[ti], b=tr.y; if(!Number.isFinite(b) || Math.abs(b)>1e100) continue;
        for(let ci=0; ci<consts.length; ci++){
          const c=consts[ci], cv=c.value; if(!Number.isFinite(cv) || cv===0) continue;
          const ratio=b/cv;
          if(Number.isFinite(ratio)){ const rr=constDbRationalApprox(ratio, 18, 60, relTol); if(rr){ const expr=constDbMulConstExpr(rr,c); add(constDbBuildRow(settings,tr,c,expr,Number(rr.p)/Number(rr.q)*cv,'degree-1 ratio b/c',rr.err,{height:Number(absBig(rr.p)>absBig(rr.q)?absBig(rr.p):absBig(rr.q)),degree:1,terms:2})); } }
          for(let m=-6;m<=6;m++){ if(m===0) continue; const a=Math.round(b-m*cv); if(Math.abs(a)>12) continue; const yy=a+m*cv, er=Math.abs(yy-b)/Math.max(1,Math.abs(b)); if(er>relTol) continue; const expr=constDbPolynomialInCExpr(1n, {'0':BigInt(-a),'1':BigInt(-m)}, c); if(expr) add(constDbBuildRow(settings,tr,c,expr,yy,'affine relation in b,1,c',er,{height:Math.max(Math.abs(a),Math.abs(m)),terms:2,degree:1})); }
          for(let m1=-3;m1<=3;m1++) for(let m2=-3;m2<=3;m2++){ if(m1===0 && m2===0) continue; const a=Math.round(b-m1*cv-m2*cv*cv); if(Math.abs(a)>12) continue; const yy=a+m1*cv+m2*cv*cv, er=Math.abs(yy-b)/Math.max(1,Math.abs(b)); if(er>relTol) continue; const expr=constDbPolynomialInCExpr(1n, {'0':BigInt(-a),'1':BigInt(-m1),'2':BigInt(-m2)}, c); if(expr) add(constDbBuildRow(settings,tr,c,expr,yy,'quadratic relation in b,1,c,c^2',er,{height:Math.max(Math.abs(a),Math.abs(m1),Math.abs(m2)),terms:3,degree:2})); }
          const invc=1/cv;
          if(Number.isFinite(invc)){ for(let m1=-3;m1<=3;m1++) for(let mi=-3;mi<=3;mi++){ if(m1===0 && mi===0) continue; const a=Math.round(b-m1*cv-mi*invc); if(Math.abs(a)>12) continue; const yy=a+m1*cv+mi*invc, er=Math.abs(yy-b)/Math.max(1,Math.abs(b)); if(er>relTol) continue; const expr=constDbPolynomialInCExpr(1n, {'0':BigInt(-a),'1':BigInt(-m1),'-1':BigInt(-mi)}, c); if(expr) add(constDbBuildRow(settings,tr,c,expr,yy,'reciprocal relation in b,1,c,1/c',er,{height:Math.max(Math.abs(a),Math.abs(m1),Math.abs(mi)),terms:3,degree:2})); } }
          if((ci&7)===7) await maybeYield('full catalog scan', .26 + (ti/Math.max(1,variants.length))*0.22 + (ci/Math.max(1,consts.length))*0.22);
        }
      }
      const priorityConsts=constDbPriorityRecords(consts), relationPriorityConsts=constDbPriorityRelationRecords(consts), deepConsts=relationPriorityConsts.slice(0, level<=4 ? 96 : 140);
      let deepDone=0; const deepTotal=deepConsts.length*variants.length;
      // v11.2.1: prioritized constants outside, reordered transforms inside;
      // low-value trig algebraic constants are excluded from the expensive deep scans.
      for(let ci=0; ci<deepConsts.length; ci++){
        if(performance.now()>deadline) break;
        const c=deepConsts[ci], cv=c.value; if(!Number.isFinite(cv) || cv===0) continue;
        const H=sig<=8 ? 5 : 8, forms=constDbLinearForms(c,H);
        function nearForms(v){ let lo=0, hi=forms.length; while(lo<hi){ const mid=(lo+hi)>>1; if(forms[mid].value<v) lo=mid+1; else hi=mid; } const out=[]; for(let k=Math.max(0,lo-3); k<Math.min(forms.length,lo+4); k++) out.push(forms[k]); return out; }
        for(let ti=0; ti<variants.length; ti++){
          if(performance.now()>deadline) break;
          const tr=variants[ti], b=tr.y; if(!Number.isFinite(b) || Math.abs(b)>1e100) continue;
          const ratio=b/cv;
          if(Number.isFinite(ratio)){ const qr=constDbFindPolynomialRatio(ratio, 3, sig, Math.min(deadline, performance.now()+localPolyMs)); const qrow=constDbAlgebraicRatioRow(settings,tr,c,b,ratio,qr,qr && qr.degree===3 ? 'degree-3 ratio b/c' : null); if(qrow) add(qrow); if(level>=5 && rows.length<22){ const maxAlgDegree=Math.max(4, Math.min(8, Math.floor(level)-1)); const alg=constDbFindAlgebraicRatioLLL(ratio, maxAlgDegree, sig, constDbRelationSearchHeight(sig)+8, Math.min(deadline, performance.now()+localAlgMs), relTol); if(alg && alg.degree>=4){ const arow=constDbAlgebraicRatioRow(settings,tr,c,b,ratio,alg,`degree-${alg.degree} algebraic ratio b/c`); if(arow) add(arow); } } }
          for(const den of forms){ if(performance.now()>deadline) break; const need=b*den.value; for(const num of nearForms(need)){ const yy=num.value/den.value; if(!Number.isFinite(yy)) continue; const er=Math.abs(yy-b)/Math.max(1,Math.abs(b)); if(er>relTol) continue; if(den.b===0 && den.a===1 && num.a===0) continue; const expr=constDbMobiusExpr(num,den,c), h=Math.max(num.height,den.height); add(constDbBuildRow(settings,tr,c,expr,yy,'Möbius relation in 1,b,c,bc',er,{height:h,terms:num.terms+den.terms,degree:1})); } }
          const lrel=constDbTryRelation_b_1_c_c2_c3_invc(settings,tr,c,b,sig,Math.min(deadline, performance.now()+localFormMs),relTol); if(lrel) add(constDbBuildRow(settings,tr,c,lrel.expr,lrel.yy,lrel.method,lrel.err,{height:lrel.height,terms:lrel.terms,degree:lrel.degree||2}));
          if(level>=5 && rows.length<24){ for(const rr of constDbLogLinearRows(settings,tr,c,b,sig,Math.min(deadline, performance.now()+localLogMs),relTol)) add(rr); }
          if(level>=5 && rows.length<18){ for(const rr of constDbExtraSubsetRows(settings,tr,c,b,sig,Math.min(deadline, performance.now()+localLogMs),relTol)) add(rr); }
          deepDone++;
        }
        if(settings){ settings._constantDbDeepDone=deepDone; settings._constantDbDeepTotal=deepTotal; settings._constantDbDeepFrac=deepTotal ? deepDone/deepTotal : 1; }
        await maybeYield('deep relation scan', .55 + (ci/Math.max(1,deepConsts.length))*0.40);
      }
      const priorityAlgebraicConsts=relationPriorityConsts.slice(0,72);
      // v11.1.2: run the priority algebraic sweep after the cheap and deep
      // relation passes, preserving responsiveness while honoring
      // the reordered transform priority.
      for(let ti=0; ti<variants.length; ti++){
        if(performance.now()>deadline) break;
        const tr=variants[ti], b=tr.y; if(!Number.isFinite(b) || Math.abs(b)>1e100) continue;
        for(let ci=0; ci<priorityAlgebraicConsts.length; ci++){
          if(performance.now()>deadline) break;
          const c=priorityAlgebraicConsts[ci], cv=c.value; if(!Number.isFinite(cv) || cv===0) continue;
          const ratio=b/cv; if(!Number.isFinite(ratio)) continue;
          const qr=constDbFindPolynomialRatio(ratio, 3, sig, Math.min(deadline, performance.now()+localPolyMs));
          const qrow=constDbAlgebraicRatioRow(settings,tr,c,b,ratio,qr,qr && qr.degree===3 ? 'degree-3 ratio b/c' : null); if(qrow) add(qrow);
          if(level>=5 && rows.length<24){ const maxAlgDegree=Math.max(4, Math.min(8, Math.floor(level)-1)); const alg=constDbFindAlgebraicRatioLLL(ratio, maxAlgDegree, sig, constDbRelationSearchHeight(sig)+8, Math.min(deadline, performance.now()+localAlgMs), relTol); if(alg && alg.degree>=4){ const arow=constDbAlgebraicRatioRow(settings,tr,c,b,ratio,alg,`degree-${alg.degree} algebraic ratio b/c`); if(arow) add(arow); } }
          if((ci&1)===1) await maybeYield('priority algebraic scan', .02 + (ti/Math.max(1,variants.length))*0.18 + (ci/Math.max(1,priorityAlgebraicConsts.length))*0.18);
          if(performance.now()>deadline) break;
        }
      }
      const map=new Map(); for(const r of rows){ const k=normalizeResultTextKey(r.candidate); if(!map.has(k) || (r.score??1e9)<(map.get(k).score??1e9)) map.set(k,r); }
      const finalRows=[...map.values()].sort((a,b)=>(a.score??1e9)-(b.score??1e9) || (a.err||1)-(b.err||1));
      if(settings){ settings._constantDbMs=Math.round(performance.now()-startTime); if(settings._constantDbDeepTotal===undefined){ settings._constantDbDeepDone=deepDone||0; settings._constantDbDeepTotal=deepTotal||0; settings._constantDbDeepFrac=deepTotal ? deepDone/deepTotal : 1; } }
      if(progressCb) progressCb({phase:'finalizing', frac:1, rows:finalRows.slice(0,8), elapsed:Math.round(performance.now()-startTime), budgetMs, deepDone:settings?._constantDbDeepDone, deepTotal:settings?._constantDbDeepTotal});
      return finalRows.slice(0,8);
    }

    // v11.3.1 low-precision sparse linear-combination matcher.
    // It tries to write the input as a rationally scaled sum of at most three
    // constants from a Q-basis-pruned catalogue, with coefficient height <= 36.
    const RIES_LOWPREC_LINEAR_COMBO_HEIGHT = 36;
    const RIES_LOWPREC_LINEAR_COMBO_LIMIT = 5;
    const RIES_LOWPREC_LINEAR_COMBO_BUDGET_MS = 3000;
    const RIES_LOWPREC_LINEAR_COMBO_RAW = [{"id":"lc_0","sourceIndex":0,"label":"1","latex":"1","value":1.0,"description":"The unit constant; allows affine rational offsets."},{"id":"lc_1","sourceIndex":1,"label":"π","latex":"\\pi","value":3.141592653589793,"description":"Archimedes’ constant, the circle constant."},{"id":"lc_2","sourceIndex":2,"label":"2π","latex":"2\\pi","value":6.283185307179586,"description":"Full turn angle; rational multiple of pi."},{"id":"lc_3","sourceIndex":3,"label":"π/2","latex":"\\pi/2","value":1.5707963267948966,"description":"Half pi; rational multiple of pi."},{"id":"lc_4","sourceIndex":4,"label":"π²","latex":"\\pi^2","value":9.869604401089358,"description":"Square of pi; proportional to zeta(2)."},{"id":"lc_5","sourceIndex":5,"label":"π³","latex":"\\pi^3","value":31.00627668029982,"description":"Cube of pi; appears in odd special-value formulas."},{"id":"lc_6","sourceIndex":6,"label":"π⁴","latex":"\\pi^4","value":97.40909103400244,"description":"Fourth power of pi; proportional to zeta(4)."},{"id":"lc_7","sourceIndex":7,"label":"γ","latex":"\\gamma","value":0.5772156649015329,"description":"Euler–Mascheroni constant, the limiting harmonic-log difference."},{"id":"lc_8","sourceIndex":8,"label":"log 2","latex":"\\log 2","value":0.6931471805599453,"description":"Natural logarithm of 2."},{"id":"lc_9","sourceIndex":9,"label":"log 3","latex":"\\log 3","value":1.0986122886681098,"description":"Natural logarithm of 3."},{"id":"lc_10","sourceIndex":10,"label":"log 5","latex":"\\log 5","value":1.6094379124341005,"description":"Natural logarithm of 5."},{"id":"lc_11","sourceIndex":11,"label":"log π","latex":"\\log \\pi","value":1.1447298858494002,"description":"Natural logarithm of pi."},{"id":"lc_12","sourceIndex":12,"label":"√π","latex":"\\sqrt{\\pi}","value":1.772453850905516,"description":"Square root of pi, equal to Gamma(1/2)."},{"id":"lc_13","sourceIndex":13,"label":"Catalan G","latex":"G","value":0.915965594177219,"description":"Catalan’s constant, also Dirichlet beta beta(2)."},{"id":"lc_14","sourceIndex":14,"label":"Cl₂(π/3)","latex":"\\operatorname{Cl}_2(\\pi/3)","value":1.0149416064096537,"description":"Clausen function value at pi/3; the Gieseking constant."},{"id":"lc_15","sourceIndex":15,"label":"Cl₂(π/4)","latex":"\\operatorname{Cl}_2(\\pi/4)","value":0.9818721510502034,"description":"Clausen function value at pi/4."},{"id":"lc_16","sourceIndex":16,"label":"Cl₂(π/6)","latex":"\\operatorname{Cl}_2(\\pi/6)","value":0.8643791310538927,"description":"Clausen function value at pi/6."},{"id":"lc_17","sourceIndex":17,"label":"Cl₂(π/5)","latex":"\\operatorname{Cl}_2(\\pi/5)","value":0.9237551681005353,"description":"Clausen function value at pi/5."},{"id":"lc_18","sourceIndex":18,"label":"Cl₂(2π/5)","latex":"\\operatorname{Cl}_2(2\\pi/5)","value":0.9973546913984148,"description":"Clausen function value at 2pi/5."},{"id":"lc_19","sourceIndex":19,"label":"Cl₂(π/8)","latex":"\\operatorname{Cl}_2(\\pi/8)","value":0.760601239358469,"description":"Clausen function value at pi/8."},{"id":"lc_20","sourceIndex":20,"label":"πG","latex":"\\pi G","value":2.8775907816081614,"description":"Product of pi and Catalan’s constant."},{"id":"lc_21","sourceIndex":21,"label":"π²G","latex":"\\pi^2 G","value":9.04021805953791,"description":"Product of pi squared and Catalan’s constant."},{"id":"lc_22","sourceIndex":22,"label":"G²","latex":"G^2","value":0.8389929697164259,"description":"Square of Catalan’s constant."},{"id":"lc_23","sourceIndex":23,"label":"β(3)","latex":"\\beta(3)","value":0.9689461462593694,"description":"Dirichlet beta value at 3; a rational multiple of pi cubed."},{"id":"lc_24","sourceIndex":24,"label":"β(4)","latex":"\\beta(4)","value":0.9889445517411053,"description":"Dirichlet beta value at 4."},{"id":"lc_25","sourceIndex":25,"label":"β(5)","latex":"\\beta(5)","value":0.9961578280770881,"description":"Dirichlet beta value at 5."},{"id":"lc_26","sourceIndex":26,"label":"β(6)","latex":"\\beta(6)","value":0.9986852222184381,"description":"Dirichlet beta value at 6."},{"id":"lc_27","sourceIndex":27,"label":"ζ(2)","latex":"\\zeta(2)","value":1.6449340668482264,"description":"Basel constant, equal to pi squared divided by 6."},{"id":"lc_28","sourceIndex":28,"label":"ζ(3)","latex":"\\zeta(3)","value":1.2020569031595942,"description":"Apéry’s constant, the zeta value at 3."},{"id":"lc_29","sourceIndex":29,"label":"ζ(4)","latex":"\\zeta(4)","value":1.0823232337111381,"description":"Zeta value at 4, equal to pi fourth divided by 90."},{"id":"lc_30","sourceIndex":30,"label":"ζ(5)","latex":"\\zeta(5)","value":1.03692775514337,"description":"Riemann zeta value at 5."},{"id":"lc_31","sourceIndex":31,"label":"PolyGamma[2, 7/8]","latex":"\\operatorname{PolyGamma}(2,7/8)","value":-3.458947233666973,"description":"Polygamma function value of order 2 at 7/8."},{"id":"lc_32","sourceIndex":32,"label":"Li₂(1/2)","latex":"\\operatorname{Li}_2(1/2)","value":0.5822405264650125,"description":"Dilogarithm at one half."},{"id":"lc_33","sourceIndex":33,"label":"Li₂(-1/2)","latex":"\\operatorname{Li}_2(-1/2)","value":-0.4484142069236462,"description":"Dilogarithm at negative one half."},{"id":"lc_34","sourceIndex":34,"label":"Li₃(1/2)","latex":"\\operatorname{Li}_3(1/2)","value":0.5372131936080402,"description":"Trilogarithm at one half."},{"id":"lc_35","sourceIndex":35,"label":"Li₄(1/2)","latex":"\\operatorname{Li}_4(1/2)","value":0.5174790616738995,"description":"Polylogarithm Li_4 at one half."},{"id":"lc_36","sourceIndex":36,"label":"Li₅(1/2)","latex":"\\operatorname{Li}_5(1/2)","value":0.5084005792422687,"description":"Polylogarithm Li_5 at one half."},{"id":"lc_37","sourceIndex":37,"label":"ζ(3)log2","latex":"\\zeta(3)\\log 2","value":0.833202353297692,"description":"Product of Apéry’s constant and log 2."},{"id":"lc_38","sourceIndex":38,"label":"π²log2","latex":"\\pi^2\\log 2","value":6.841088463857116,"description":"Product of pi squared and log 2."},{"id":"lc_39","sourceIndex":39,"label":"π²log²2","latex":"\\pi^2(\\log 2)^2","value":4.741881180683728,"description":"Product of pi squared and log squared 2."},{"id":"lc_40","sourceIndex":40,"label":"log³2","latex":"(\\log 2)^3","value":0.3330246519889295,"description":"Cube of log 2."},{"id":"lc_41","sourceIndex":41,"label":"log⁴2","latex":"(\\log 2)^4","value":0.2308350985830835,"description":"Fourth power of log 2."},{"id":"lc_42","sourceIndex":42,"label":"Ti₂(1/2)","latex":"\\operatorname{Ti}_2(1/2)","value":0.4872223582945224,"description":"Inverse tangent integral at one half."},{"id":"lc_43","sourceIndex":43,"label":"Ti₂(√2−1)","latex":"\\operatorname{Ti}_2(\\sqrt2-1)","value":0.4067661542498135,"description":"Inverse tangent integral at sqrt(2)-1."},{"id":"lc_44","sourceIndex":44,"label":"A","latex":"A","value":1.2824271291006226,"description":"Glaisher–Kinkelin constant."},{"id":"lc_45","sourceIndex":45,"label":"log A","latex":"\\log A","value":0.2487544770337843,"description":"Natural logarithm of the Glaisher–Kinkelin constant."},{"id":"lc_46","sourceIndex":46,"label":"Khinchin K","latex":"K","value":2.6854520010653062,"description":"Khinchin’s constant from continued fractions."},{"id":"lc_47","sourceIndex":47,"label":"log Khinchin K","latex":"\\log K","value":0.9878490568338107,"description":"Natural logarithm of Khinchin’s constant."},{"id":"lc_48","sourceIndex":48,"label":"PolyGamma[1, 1/3]","latex":"\\operatorname{PolyGamma}(1,1/3)","value":10.095597125427094,"description":"Polygamma function value of order 1 at 1/3."},{"id":"lc_49","sourceIndex":49,"label":"PolyGamma[1, 2/3]","latex":"\\operatorname{PolyGamma}(1,2/3)","value":3.0638754093587175,"description":"Polygamma function value of order 1 at 2/3."},{"id":"lc_50","sourceIndex":50,"label":"PolyGamma[1, 1/6]","latex":"\\operatorname{PolyGamma}(1,1/6)","value":37.31851309234966,"description":"Polygamma function value of order 1 at 1/6."},{"id":"lc_51","sourceIndex":51,"label":"PolyGamma[1, 5/6]","latex":"\\operatorname{PolyGamma}(1,5/6)","value":2.159904512007776,"description":"Polygamma function value of order 1 at 5/6."},{"id":"lc_54","sourceIndex":54,"label":"log Γ(1/3)","latex":"\\log\\Gamma(1/3)","value":0.9854206469277671,"description":"Natural logarithm of Gamma(1/3)."},{"id":"lc_55","sourceIndex":55,"label":"log Γ(1/4)","latex":"\\log\\Gamma(1/4)","value":1.2880225246980774,"description":"Natural logarithm of Gamma(1/4)."},{"id":"lc_60","sourceIndex":60,"label":"Γ(1/4)²/√π","latex":"\\Gamma(1/4)^2/\\sqrt{\\pi}","value":7.4162987092054875,"description":"Lemniscatic gamma combination."},{"id":"lc_61","sourceIndex":61,"label":"Γ(1/3)³/(2π)","latex":"\\Gamma(1/3)^3/(2\\pi)","value":3.059908074114386,"description":"Equianharmonic gamma combination."},{"id":"lc_62","sourceIndex":62,"label":"lemniscate constant","latex":"\\varpi","value":2.6220575542921196,"description":"The lemniscate constant, a gamma-based elliptic period."},{"id":"lc_63","sourceIndex":63,"label":"Gauss constant","latex":"G_{\\rm Gauss}","value":0.8346268416740732,"description":"Gauss’s constant, reciprocal of agm(1,sqrt(2))."},{"id":"lc_64","sourceIndex":64,"label":"agm(1,√2)","latex":"\\operatorname{agm}(1,\\sqrt2)","value":1.1981402347355923,"description":"Arithmetic-geometric mean of 1 and sqrt(2)."},{"id":"lc_65","sourceIndex":65,"label":"K(1/√2)","latex":"K(1/\\sqrt2)","value":1.8540746773013719,"description":"Complete elliptic integral of the first kind at modulus 1/sqrt(2)."},{"id":"lc_66","sourceIndex":66,"label":"E(1/√2)","latex":"E(1/\\sqrt2)","value":1.3506438810476755,"description":"Complete elliptic integral of the second kind at modulus 1/sqrt(2)."},{"id":"lc_67","sourceIndex":67,"label":"√2","latex":"\\sqrt2","value":1.414213562373095,"description":"Square root of 2."},{"id":"lc_68","sourceIndex":68,"label":"√3","latex":"\\sqrt3","value":1.7320508075688772,"description":"Square root of 3."},{"id":"lc_69","sourceIndex":69,"label":"√5","latex":"\\sqrt5","value":2.23606797749979,"description":"Square root of 5."},{"id":"lc_70","sourceIndex":70,"label":"√6","latex":"\\sqrt6","value":2.449489742783178,"description":"Square root of 6."},{"id":"lc_72","sourceIndex":72,"label":"∛2","latex":"\\sqrt[3]{2}","value":1.2599210498948732,"description":"Cube root of 2."},{"id":"lc_73","sourceIndex":73,"label":"∛3","latex":"\\sqrt[3]{3}","value":1.4422495703074083,"description":"Cube root of 3."},{"id":"lc_79","sourceIndex":79,"label":"log(log(2))","latex":"\\log(\\log 2)","value":-0.36651292058166435,"description":"Nested natural logarithm log(log(2))."},{"id":"lc_80","sourceIndex":80,"label":"log(log(3))","latex":"\\log(\\log 3)","value":0.09404782761669903,"description":"Nested natural logarithm log(log(3))."},{"id":"lc_81","sourceIndex":81,"label":"π(log2)^2","latex":"\\pi(\\log 2)^2","value":1.5093876589204962,"description":"Product of pi and squared log 2."},{"id":"lc_82","sourceIndex":82,"label":"π log3","latex":"\\pi\\log 3","value":3.4513922952232026,"description":"Product of pi and log 3."},{"id":"lc_83","sourceIndex":83,"label":"e","latex":"e","value":2.718281828459045,"description":"Euler’s number, the natural exponential base."},{"id":"lc_84","sourceIndex":84,"label":"√e","latex":"\\sqrt e","value":1.6487212707001282,"description":"Square root of Euler’s number."},{"id":"lc_85","sourceIndex":85,"label":"e²","latex":"e^2","value":7.38905609893065,"description":"Euler’s number squared."},{"id":"lc_86","sourceIndex":86,"label":"e⁻¹","latex":"e^{-1}","value":0.3678794411714423,"description":"Reciprocal of Euler’s number."},{"id":"lc_87","sourceIndex":87,"label":"e⁻²","latex":"e^{-2}","value":0.1353352832366127,"description":"Second reciprocal power of Euler’s number."},{"id":"lc_88","sourceIndex":88,"label":"e^π","latex":"e^\\pi","value":23.14069263277927,"description":"Gelfond’s constant."},{"id":"lc_89","sourceIndex":89,"label":"e^−π","latex":"e^{-\\pi}","value":0.0432139182637722,"description":"Exponential nome-like constant e^{-pi}."},{"id":"lc_90","sourceIndex":90,"label":"PolyGamma[2, 1/5]","latex":"\\operatorname{PolyGamma}(2,1/5)","value":-251.47803611443592,"description":"Polygamma function value of order 2 at 1/5."},{"id":"lc_91","sourceIndex":91,"label":"PolyGamma[2, 2/5]","latex":"\\operatorname{PolyGamma}(2,2/5)","value":-32.23912862357836,"description":"Polygamma function value of order 2 at 2/5."},{"id":"lc_92","sourceIndex":92,"label":"PolyGamma[2, 3/5]","latex":"\\operatorname{PolyGamma}(2,3/5)","value":-9.962831537143458,"description":"Polygamma function value of order 2 at 3/5."},{"id":"lc_93","sourceIndex":93,"label":"PolyGamma[2, 4/5]","latex":"\\operatorname{PolyGamma}(2,4/5)","value":-4.430115708421635,"description":"Polygamma function value of order 2 at 4/5."},{"id":"lc_94","sourceIndex":94,"label":"PolyGamma[2, 1/8]","latex":"\\operatorname{PolyGamma}(2,1/8)","value":-1025.7533381181356,"description":"Polygamma function value of order 2 at 1/8."},{"id":"lc_95","sourceIndex":95,"label":"PolyGamma[2, 3/8]","latex":"\\operatorname{PolyGamma}(2,3/8)","value":-38.96211849703415,"description":"Polygamma function value of order 2 at 3/8."},{"id":"lc_96","sourceIndex":96,"label":"e^γ","latex":"e^\\gamma","value":1.781072417990198,"description":"Exponential of the Euler–Mascheroni constant."},{"id":"lc_97","sourceIndex":97,"label":"PolyGamma[2, 5/8]","latex":"\\operatorname{PolyGamma}(2,5/8)","value":-8.86858138215968,"description":"Polygamma function value of order 2 at 5/8."},{"id":"lc_98","sourceIndex":98,"label":"Ei(1)","latex":"\\operatorname{Ei}(1)","value":1.8951178163559368,"description":"Exponential integral Ei at 1."},{"id":"lc_99","sourceIndex":99,"label":"Ei(−1)","latex":"\\operatorname{Ei}(-1)","value":-0.2193839343955203,"description":"Exponential integral Ei at -1."},{"id":"lc_100","sourceIndex":100,"label":"E₁(1)","latex":"E_1(1)","value":0.2193839343955203,"description":"Exponential integral E1 at 1, equal to -Ei(-1)."},{"id":"lc_101","sourceIndex":101,"label":"li(2)=Ei(log2)","latex":"\\operatorname{li}(2)","value":1.045163780117493,"description":"Logarithmic integral value li(2), equal to Ei(log 2)."},{"id":"lc_102","sourceIndex":102,"label":"Si(1)","latex":"\\operatorname{Si}(1)","value":0.946083070367183,"description":"Sine integral at 1."},{"id":"lc_103","sourceIndex":103,"label":"Si(π)","latex":"\\operatorname{Si}(\\pi)","value":1.8519370519824663,"description":"Sine integral at pi, also Gibbs’ constant."},{"id":"lc_104","sourceIndex":104,"label":"Ci(1)","latex":"\\operatorname{Ci}(1)","value":0.3374039229009681,"description":"Cosine integral at 1."},{"id":"lc_105","sourceIndex":105,"label":"Ci(π)","latex":"\\operatorname{Ci}(\\pi)","value":0.0736679120464255,"description":"Cosine integral at pi."},{"id":"lc_106","sourceIndex":106,"label":"π log2","latex":"\\pi\\log 2","value":2.177586090303602,"description":"Product of pi and log 2."},{"id":"lc_107","sourceIndex":107,"label":"(log2)^2","latex":"(\\log 2)^2","value":0.4804530139182014,"description":"Square of log 2."},{"id":"lc_108","sourceIndex":108,"label":"sinc(1)","latex":"\\operatorname{sinc}(1)","value":0.8414709848078965,"description":"Unnormalized sinc value sin(1)/1."},{"id":"lc_109","sourceIndex":109,"label":"sinc(π/2)","latex":"\\operatorname{sinc}(\\pi/2)","value":0.6366197723675813,"description":"Unnormalized sinc value at pi/2, equal to 2/pi."},{"id":"lc_110","sourceIndex":110,"label":"sinc(π/3)","latex":"\\operatorname{sinc}(\\pi/3)","value":0.8269933431326881,"description":"Unnormalized sinc value at pi/3."},{"id":"lc_111","sourceIndex":111,"label":"erf(1)","latex":"\\operatorname{erf}(1)","value":0.8427007929497149,"description":"Error function value at 1."},{"id":"lc_112","sourceIndex":112,"label":"erfc(1)","latex":"\\operatorname{erfc}(1)","value":0.1572992070502851,"description":"Complementary error function at 1, equal to 1-erf(1)."},{"id":"lc_113","sourceIndex":113,"label":"erfi(1)","latex":"\\operatorname{erfi}(1)","value":1.6504257587975428,"description":"Imaginary error function value at 1."},{"id":"lc_114","sourceIndex":114,"label":"Dawson F(1)","latex":"F(1)","value":0.5380795069127684,"description":"Dawson integral at 1."},{"id":"lc_115","sourceIndex":115,"label":"Fresnel S(1)","latex":"S(1)","value":0.4382591473903548,"description":"Fresnel sine integral at 1."},{"id":"lc_116","sourceIndex":116,"label":"Fresnel C(1)","latex":"C(1)","value":0.7798934003768228,"description":"Fresnel cosine integral at 1."},{"id":"lc_117","sourceIndex":117,"label":"J₀(1)","latex":"J_0(1)","value":0.7651976865579666,"description":"Bessel function J_0 at 1."},{"id":"lc_118","sourceIndex":118,"label":"J₁(1)","latex":"J_1(1)","value":0.4400505857449335,"description":"Bessel function J_1 at 1."},{"id":"lc_119","sourceIndex":119,"label":"I₀(1)","latex":"I_0(1)","value":1.2660658777520084,"description":"Modified Bessel function I_0 at 1."},{"id":"lc_120","sourceIndex":120,"label":"I₁(1)","latex":"I_1(1)","value":0.565159103992485,"description":"Modified Bessel function I_1 at 1."},{"id":"lc_121","sourceIndex":121,"label":"K₀(1)","latex":"K_0(1)","value":0.4210244382407083,"description":"Modified Bessel function K_0 at 1."},{"id":"lc_122","sourceIndex":122,"label":"K₁(1)","latex":"K_1(1)","value":0.6019072301972346,"description":"Modified Bessel function K_1 at 1."},{"id":"lc_123","sourceIndex":123,"label":"Ai(0)","latex":"\\operatorname{Ai}(0)","value":0.3550280538878172,"description":"Airy Ai function at 0."},{"id":"lc_124","sourceIndex":124,"label":"Ai′(0)","latex":"\\operatorname{Ai}'(0)","value":-0.2588194037928068,"description":"Derivative of the Airy Ai function at 0."},{"id":"lc_125","sourceIndex":125,"label":"Bi(0)","latex":"\\operatorname{Bi}(0)","value":0.6149266274460007,"description":"Airy Bi function at 0."},{"id":"lc_126","sourceIndex":126,"label":"Bi′(0)","latex":"\\operatorname{Bi}'(0)","value":0.4482883573538264,"description":"Derivative of the Airy Bi function at 0."},{"id":"lc_127","sourceIndex":127,"label":"π√2","latex":"\\pi\\sqrt{2}","value":4.442882938158366,"description":"Product of pi and sqrt 2. Added for level 5 linear-combination matching.","minLevel":5},{"id":"lc_128","sourceIndex":128,"label":"π√3","latex":"\\pi\\sqrt{3}","value":5.441398092702653,"description":"Product of pi and sqrt 3. Added for level 5 linear-combination matching.","minLevel":5},{"id":"lc_129","sourceIndex":129,"label":"Σ 1/(n²+1) / Im PolyGamma[0, exp(2πi/4)]","latex":"\\sum_{n=0}^{\\infty}\\frac{1}{n^2+1}=\\Im\\,\\operatorname{PolyGamma}(0,e^{2\\pi i/4})","value":2.076674047468581,"description":"Series value Sum[1/(n^2+1), {n,0,Infinity}], numerically equal to Im PolyGamma[0, Exp[2 Pi I/4]]. Combined as one level 5 linear-combination constant with two display labels.","minLevel":5},{"id":"lc_130","sourceIndex":130,"label":"Σ 1/(n³+1)","latex":"\\sum_{n=0}^{\\infty}\\frac{1}{n^3+1}","value":1.686503342338624,"description":"Series value Sum[1/(n^3+1), {n,0,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5},{"id":"lc_131","sourceIndex":131,"label":"Σ (-1)^n/(n²+1)","latex":"\\sum_{n=0}^{\\infty}\\frac{(-1)^n}{n^2+1}","value":0.6360145274910666,"description":"Alternating series value Sum[(-1)^n/(n^2+1), {n,0,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5},{"id":"lc_132","sourceIndex":132,"label":"Σ (-1)^n/C(2n,n)","latex":"\\sum_{n=0}^{\\infty}\\frac{(-1)^n}{\\binom{2n}{n}}","value":0.6278364236143984,"description":"Alternating inverse central-binomial series Sum[(-1)^n/Binomial[2 n,n], {n,0,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5},{"id":"lc_133","sourceIndex":133,"label":"Σ 1/C(3n,n)","latex":"\\sum_{n=0}^{\\infty}\\frac{1}{\\binom{3n}{n}}","value":1.4143220443218203,"description":"Inverse binomial series Sum[1/Binomial[3 n,n], {n,0,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5},{"id":"lc_134","sourceIndex":134,"label":"Σ 1/(n²+n+1)","latex":"\\sum_{n=0}^{\\infty}\\frac{1}{n^2+n+1}","value":1.7981472805626901,"description":"Series value Sum[1/(n^2+n+1), {n,0,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5},{"id":"lc_135","sourceIndex":135,"label":"Σ (-1)^n/(n²+n+1)","latex":"\\sum_{n=0}^{\\infty}\\frac{(-1)^n}{n^2+n+1}","value":0.7613102040011035,"description":"Alternating series value Sum[(-1)^n/(n^2+n+1), {n,0,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5},{"id":"lc_136","sourceIndex":136,"label":"Σ 1/(1+2^n)","latex":"\\sum_{n=0}^{\\infty}\\frac{1}{1+2^n}","value":1.2644997803484441,"description":"Series value Sum[1/(1+2^n), {n,0,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5},{"id":"lc_137","sourceIndex":137,"label":"Re PolyGamma[0, exp(2πi/8)]","latex":"\\Re\\,\\operatorname{PolyGamma}(0,e^{2\\pi i/8})","value":-0.36034322975434907,"description":"Real part of PolyGamma[0, Exp[2 Pi I/8]]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_138","sourceIndex":138,"label":"Re PolyGamma[0, exp(2πi/6)]","latex":"\\Re\\,\\operatorname{PolyGamma}(0,e^{2\\pi i/6})","value":-0.2149265587296965,"description":"Real part of PolyGamma[0, Exp[2 Pi I/6]]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_139","sourceIndex":139,"label":"Re PolyGamma[0, exp(2πi/4)]","latex":"\\Re\\,\\operatorname{PolyGamma}(0,e^{2\\pi i/4})","value":0.09465032062247698,"description":"Real part of PolyGamma[0, Exp[2 Pi I/4]]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_140","sourceIndex":140,"label":"Re PolyGamma[0, exp(2πi/3)]","latex":"\\Re\\,\\operatorname{PolyGamma}(0,e^{2\\pi i/3})","value":0.28507344127030354,"description":"Real part of PolyGamma[0, Exp[2 Pi I/3]]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_141","sourceIndex":141,"label":"Im PolyGamma[0, exp(2πi/8)]","latex":"\\Im\\,\\operatorname{PolyGamma}(0,e^{2\\pi i/8})","value":1.2199689267137874,"description":"Imaginary part of PolyGamma[0, Exp[2 Pi I/8]]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_142","sourceIndex":142,"label":"Im PolyGamma[0, exp(2πi/6)]","latex":"\\Im\\,\\operatorname{PolyGamma}(0,e^{2\\pi i/6})","value":1.5572412247131941,"description":"Imaginary part of PolyGamma[0, Exp[2 Pi I/6]]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_144","sourceIndex":144,"label":"Im PolyGamma[0, exp(2πi/3)]","latex":"\\Im\\,\\operatorname{PolyGamma}(0,e^{2\\pi i/3})","value":2.4232666284976325,"description":"Imaginary part of PolyGamma[0, Exp[2 Pi I/3]]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_145","sourceIndex":145,"label":"Σ (-1)^n/(n log n), n≥2","latex":"\\sum_{n=2}^{\\infty}\\frac{(-1)^n}{n\\log n}","value":0.52641224653331,"description":"Alternating logarithmic series NSum[(-1)^n/(n Log[n]), {n,2,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_146","sourceIndex":146,"label":"Σ (-1)^n/((n−1)log n), n≥2","latex":"\\sum_{n=2}^{\\infty}\\frac{(-1)^n}{(n-1)\\log n}","value":1.136448654977018,"description":"Alternating logarithmic series NSum[(-1)^n/((n-1) Log[n]), {n,2,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_147","sourceIndex":147,"label":"Σ (-1)^n/log n, n≥2","latex":"\\sum_{n=2}^{\\infty}\\frac{(-1)^n}{\\log n}","value":0.92429989722294,"description":"Alternating logarithmic series NSum[(-1)^n/Log[n], {n,2,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_148","sourceIndex":148,"label":"Σ (-1)^n/(log n)², n≥2","latex":"\\sum_{n=2}^{\\infty}\\frac{(-1)^n}{(\\log n)^2}","value":1.55701835019512,"description":"Alternating logarithmic series NSum[(-1)^n/Log[n]^2, {n,2,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_149","sourceIndex":149,"label":"Σ (-1)^n log n/n, n≥1","latex":"\\sum_{n=1}^{\\infty}\\frac{(-1)^n\\log n}{n}","value":0.15986890374243098,"description":"Alternating logarithmic series Sum[(-1)^n Log[n]/n, {n,1,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_150","sourceIndex":150,"label":"Σ (-1)^n log n/n², n≥1","latex":"\\sum_{n=1}^{\\infty}\\frac{(-1)^n\\log n}{n^2}","value":0.1013165781635045,"description":"Alternating logarithmic series Sum[(-1)^n Log[n]/n^2, {n,1,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_151","sourceIndex":151,"label":"Σ log n/n², n≥1","latex":"\\sum_{n=1}^{\\infty}\\frac{\\log n}{n^2}","value":0.9375482543158438,"description":"Logarithmic zeta-derivative series Sum[Log[n]/n^2, {n,1,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true},{"id":"lc_152","sourceIndex":152,"label":"Σ log n/n³, n≥1","latex":"\\sum_{n=1}^{\\infty}\\frac{\\log n}{n^3}","value":0.19812624288563685,"description":"Logarithmic zeta-derivative series Sum[Log[n]/n^3, {n,1,Infinity}]. Added for level 5 linear-combination matching.","minLevel":5,"keepDependent":true}];
    const lowPrecisionLinearComboBasisCache = new Map();
    function lowPrecisionLinearComboGcdInt(a,b){ a=Math.abs(Math.trunc(a)||0); b=Math.abs(Math.trunc(b)||0); while(b){ const t=a%b; a=b; b=t; } return a; }
    function lowPrecisionLinearComboRationalApprox(x, maxDen=1000, maxNum=1000, relTol=2e-13){
      if(!Number.isFinite(x)) return null;
      const neg=x<0; let a=Math.abs(x);
      if(a===0) return {p:0,q:1,err:0};
      let h1=1,h0=0,k1=0,k0=1,b=Math.floor(a),xcur=a,best=null;
      for(let iter=0;iter<28;iter++){
        const h=b*h1+h0, k=b*k1+k0;
        if(k>maxDen || Math.abs(h)>maxNum) break;
        const val=h/k, err=Math.abs(val-a)/Math.max(1,a);
        best={p:neg?-h:h,q:k,err};
        if(err<=relTol) break;
        const frac=xcur-b; if(Math.abs(frac)<1e-18) break;
        xcur=1/frac; b=Math.floor(xcur);
        h0=h1; h1=h; k0=k1; k1=k;
      }
      return best && best.err<=relTol ? best : null;
    }
    function lowPrecisionLinearComboBasisConstants(settingsOrLevel=null){
      const fallbackLevel=Number(DEFAULT_RIES_LEVEL)||4;
      const rawLevel=typeof settingsOrLevel==='number'
        ? settingsOrLevel
        : (settingsOrLevel && typeof settingsOrLevel==='object'
          ? settingsOrLevel.level
          : (typeof document!=='undefined' ? document.getElementById('level')?.value : fallbackLevel));
      const level=Math.max(1, Math.floor(Number(rawLevel || fallbackLevel) || fallbackLevel));
      const cacheKey=String(level);
      if(lowPrecisionLinearComboBasisCache.has(cacheKey)) return lowPrecisionLinearComboBasisCache.get(cacheKey);
      const kept=[];
      const affineDropById=new Map([
        ['lc_112','1 - erf(1)'],
      ]);
      for(const raw of RIES_LOWPREC_LINEAR_COMBO_RAW){
        if(Number(raw.minLevel||1)>level) continue;
        const value=Number(raw.value);
        if(!Number.isFinite(value) || Math.abs(value)<1e-300) continue;
        if(affineDropById.has(raw.id)) continue;
        let dependent=false;
        if(raw.keepDependent!==true){
          for(const k of kept){
            const ratio=value/k.value;
            const rat=lowPrecisionLinearComboRationalApprox(ratio, 1000, 1000, 2.5e-13);
            if(rat){ dependent=true; break; }
          }
        }
        if(!dependent) kept.push({...raw, value, basisIndex:kept.length});
      }
      lowPrecisionLinearComboBasisCache.set(cacheKey,kept);
      return kept;
    }
    function shouldRunLowPrecisionLinearComboRows(settings){
      if(!settings || settings.modules?.linearCombo===false || settings.complexTarget || !Number.isFinite(settings.target)) return false;
      if(Math.abs(settings.target)>1e8) return false;
      const sig=typedInputPrecisionForDouble(settings);
      return sig>=2 && sig<=DOUBLE_EFFECTIVE_PRECISION_DIGITS;
    }
    function lowPrecisionLinearComboRelTol(settings){
      // v11.3.1: acceptance is tied to the visible decimal precision rather than
      // to a fixed request for five rows.  A 10-digit input therefore accepts
      // roughly 1e-8 relative residuals (about 100x the input precision), while
      // binary64-only paths keep a small floor to avoid losing real hits to
      // floating-point round-off.
      const sig=typedInputPrecisionForDouble(settings);
      const scale=typedDecimalScaleDigitsForDouble(settings);
      const bySig=120*Math.pow(10, -Math.max(0, sig));
      const byScale=scale>0 ? 120*Math.pow(10, -Math.max(0, scale)) : bySig;
      return Math.min(0.20, Math.max(2e-12, Math.min(bySig, byScale)));
    }
    function lowPrecisionLinearComboCoeffOrders(H){
      const out=[]; for(let a=1;a<=H;a++){ out.push(a,-a); } return out;
    }
    function lowPrecisionLinearComboTermText(coeff, c){
      const a=Math.abs(coeff), core=a===1 ? c.label : `${a}·${c.label}`;
      return {sign:coeff<0?'−':'+', body:core};
    }
    function lowPrecisionLinearComboTermLatex(coeff, c){
      const a=Math.abs(coeff), core=a===1 ? c.latex : `${a}\\,${c.latex}`;
      return {sign:coeff<0?'-':'+', body:core};
    }
    function lowPrecisionLinearComboFormat(den, terms, consts){
      const textParts=[], latexParts=[];
      for(const t of terms){
        const c=consts[t.idx];
        const tt=lowPrecisionLinearComboTermText(t.coeff,c);
        const lt=lowPrecisionLinearComboTermLatex(t.coeff,c);
        if(!textParts.length) textParts.push((tt.sign==='−'?'−':'')+tt.body);
        else textParts.push(` ${tt.sign==='−'?'−':'+'} ${tt.body}`);
        if(!latexParts.length) latexParts.push((lt.sign==='-'?'-':'')+lt.body);
        else latexParts.push(` ${lt.sign==='-'?'-':'+'} ${lt.body}`);
      }
      const nt=textParts.join('') || '0', nl=latexParts.join('') || '0';
      if(den===1) return {text:nt, latex:nl};
      return {text:`(${nt})/${den}`, latex:`\\frac{${nl}}{${den}}`};
    }
    function lowPrecisionLinearComboAdd(rows, seen, consts, settings, den, rawTerms, method){
      den=Math.trunc(Number(den)||0); if(den===0) return;
      const map=new Map();
      for(const t of rawTerms||[]){
        const idx=Math.trunc(Number(t.idx)); const coeff=Math.trunc(Number(t.coeff)||0);
        if(!Number.isInteger(idx) || idx<0 || idx>=consts.length || coeff===0) continue;
        map.set(idx,(map.get(idx)||0)+coeff);
      }
      let terms=[...map.entries()].map(([idx,coeff])=>({idx, coeff})).filter(t=>t.coeff!==0);
      if(!terms.length || terms.length>3) return;
      if(den<0){ den=-den; terms=terms.map(t=>({idx:t.idx, coeff:-t.coeff})); }
      let g=den;
      for(const t of terms) g=lowPrecisionLinearComboGcdInt(g,t.coeff);
      if(g>1){ den/=g; terms=terms.map(t=>({idx:t.idx, coeff:t.coeff/g})).filter(t=>t.coeff!==0); }
      if(!terms.length || terms.length>3) return;
      const H=RIES_LOWPREC_LINEAR_COMBO_HEIGHT;
      if(den<1 || den>H || terms.some(t=>Math.abs(t.coeff)>H)) return;
      terms.sort((a,b)=>a.idx-b.idx);
      const key=den+'|'+terms.map(t=>`${t.idx}:${t.coeff}`).join(',');
      if(seen.has(key)) return;
      let pred=0, maxCoeff=den;
      for(const t of terms){ pred += t.coeff*consts[t.idx].value; maxCoeff=Math.max(maxCoeff,Math.abs(t.coeff)); }
      pred/=den;
      const rel=Math.abs(pred-settings.target)/Math.max(1,Math.abs(settings.target));
      if(rel>lowPrecisionLinearComboRelTol(settings)) return;
      seen.add(key);
      const f=lowPrecisionLinearComboFormat(den,terms,consts);
      const details=terms.map(t=>{
        const c=consts[t.idx];
        return `<div><b>${escapeHtml(c.label)}</b> ≈ ${escapeHtml(fmtValue(c.value))} — ${escapeHtml(c.description||'special constant')}</div>`;
      }).join('');
      const relationLeft=den===1 ? 'x' : `${den}·x`;
      const rhs=terms.map(t=>`${t.coeff<0?'−':'+'}${Math.abs(t.coeff)===1?'':Math.abs(t.coeff)+'·'}${consts[t.idx].label}`).join(' ').replace(/^\+/, '').trim();
      const valueHtml=`<div>predicted x ≈ ${escapeHtml(fmtValue(pred))}; relative residual ${escapeHtml(fmtErr(rel))}</div><div class="muted">primitive relation: ${escapeHtml(relationLeft)} ≈ ${escapeHtml(rhs)}; terms ${terms.length}; height ${maxCoeff}; ${escapeHtml(method)}</div>${details}`;
      rows.push({
        candidate:`low-precision linear combo: x ≈ ${f.text}`,
        latex:`x \\approx ${f.latex}`,
        copyLatex:`x \\approx ${f.latex}`,
        valueHtml,
        copyValue:`x ≈ ${f.text}; constants: ${terms.map(t=>`${consts[t.idx].label}≈${fmtValue(consts[t.idx].value)} (${consts[t.idx].description||''})`).join('; ')}`,
        err:rel,
        errText:fmtErr(rel),
        lowPrecisionLinearCombo:true,
        linearComboCategory:method,
        terms:terms.length,
        height:BigInt(maxCoeff),
        denominator:den,
        score:terms.length*900 + (den-1)*10 + maxCoeff*4 + formulaVisibleLength(f.text)*2 + Math.log10(rel+1e-30)*10
      });
    }
    function lowPrecisionLinearComboConstFamily(c){
      const label=String(c?.label||''), latex=String(c?.latex||'');
      if(label==='1') return 'unit';
      if(/[π]|\\pi/.test(label) || /\\pi/.test(latex)) return 'pi';
      if(/^log|log /.test(label) || /\\log/.test(latex)) return 'log';
      if(/[ζβ]|zeta|beta/.test(label) || /\\zeta|\\beta/.test(latex)) return 'zeta';
      if(/Li|Cl|Ti/.test(label) || /Li|Cl|Ti/.test(latex)) return 'polylog';
      if(/Γ|Gamma|agm|lemniscate|Gauss|K\(/.test(label) || /Gamma|agm|varpi/.test(latex)) return 'gamma';
      if(/^√|∛|cos|φ|sqrt/.test(label) || /sqrt|cos|varphi/.test(latex)) return 'alg';
      if(/^e|Ei|E₁|li|Shi|Chi/.test(label) || /operatorname\{Ei\}|operatorname\{li\}/.test(latex)) return 'expint';
      if(/Si|Ci|sinc|erf|Dawson|Fresnel|J₀|J₁|I₀|I₁|K₀|K₁|Ai|Bi/.test(label)) return 'special';
      return 'misc';
    }
    function lowPrecisionLinearComboConstPriority(c, idx){
      const label=String(c?.label||'');
      if(label==='1') return 0;
      if(['π','π²','π³','π⁴','log 2','log 3','log 5','γ','ζ(3)','Catalan G','√2','√3','√5','e'].includes(label)) return 1 + idx*.010;
      const fam=lowPrecisionLinearComboConstFamily(c);
      const base={pi:2,log:2.2,zeta:2.7,alg:3.0,polylog:3.7,gamma:4.0,expint:5.3,special:6.0,misc:4.5,unit:0}[fam] ?? 5;
      return base + idx*.012;
    }
    function lowPrecisionLinearComboPairPriority(a,b,ia,ib){
      const fa=lowPrecisionLinearComboConstFamily(a), fb=lowPrecisionLinearComboConstFamily(b);
      const pa=lowPrecisionLinearComboConstPriority(a,ia), pb=lowPrecisionLinearComboConstPriority(b,ib);
      let penalty=0;
      if(fa!==fb){
        const key=[fa,fb].sort().join('+');
        if(key==='log+pi' || key==='alg+pi' || key==='alg+log' || key==='log+zeta' || key==='pi+zeta') penalty=.15;
        else if(key==='gamma+pi' || key==='gamma+log' || key==='polylog+zeta' || key==='alg+zeta') penalty=.55;
        else penalty=1.4;
      }
      return pa+pb+penalty+Math.abs(ia-ib)*.001;
    }
    function lowPrecisionLinearComboPairTasks(consts){
      const pairs=[];
      for(let i=0;i<consts.length;i++) for(let j=i+1;j<consts.length;j++) pairs.push({i,j,priority:lowPrecisionLinearComboPairPriority(consts[i],consts[j],i,j)});
      pairs.sort((a,b)=>a.priority-b.priority || a.i-b.i || a.j-b.j);
      return pairs;
    }
    function lowPrecisionLinearComboRows(settings, progressCb=null){
      if(!shouldRunLowPrecisionLinearComboRows(settings)) return [];
      const H=RIES_LOWPREC_LINEAR_COMBO_HEIGHT;
      const consts=lowPrecisionLinearComboBasisConstants(settings);
      const x=Number(settings.target);
      const rows=[], seen=new Set();
      const coeffOrder=lowPrecisionLinearComboCoeffOrders(H);
      const level=Math.max(4, Number(settings.level||DEFAULT_RIES_LEVEL));
      const start=performance.now();
      const budget=stageBudgetValueToMs(settings?.stageBudgets?.linearComboMs, RIES_LOWPREC_LINEAR_COMBO_BUDGET_MS);
      const deadline=start+budget;
      const absTol=lowPrecisionLinearComboRelTol(settings)*Math.max(1,Math.abs(x));
      const terms=[];
      for(let idx=0; idx<consts.length; idx++){
        const c=consts[idx], cp=lowPrecisionLinearComboConstPriority(c,idx);
        for(const coeff of coeffOrder){
          const value=coeff*c.value;
          if(Number.isFinite(value)) terms.push({idx, coeff, value, priority:Math.abs(coeff)*2.4 + cp*6 + idx*.01});
        }
      }
      terms.sort((a,b)=>a.priority-b.priority || Math.abs(a.value)-Math.abs(b.value) || a.idx-b.idx || a.coeff-b.coeff);

      // Bucketed individual terms.  The key is deliberately based on the widest
      // denominator-scaled tolerance, and every collision is verified by the
      // primitive formula checker before display.
      const bucketWidth=Math.max(absTol*H*1.0000001, 1e-12);
      const termBuckets=new Map();
      function bucketKey(v){ return Math.floor(v/bucketWidth); }
      function bucketAdd(t){ const k=bucketKey(t.value); let a=termBuckets.get(k); if(!a){ a=[]; termBuckets.set(k,a); } a.push(t); }
      for(const t of terms) bucketAdd(t);
      function nearbyTerms(value){
        const k=bucketKey(value), out=[];
        for(let d=-2; d<=2; d++){ const a=termBuckets.get(k+d); if(a) for(const t of a) out.push(t); }
        return out;
      }

      // 1-term rational coefficient scan: x = a*A/d.
      if(settings?.linearComboOptions?.one !== false) for(let den=1; den<=H; den++){
        const target=den*x;
        for(let idx=0; idx<consts.length; idx++){
          const c=consts[idx];
          const a=Math.round(target/c.value);
          if(a!==0 && Math.abs(a)<=H) lowPrecisionLinearComboAdd(rows,seen,consts,settings,den,[{idx,coeff:a}],'1-term rational coefficient scan');
        }
      }

      // 2-term scan is now exhaustive over the height<=36 signed term list.  This
      // is cheap enough with the term buckets and fixes the old prefix truncation.
      if(settings?.linearComboOptions?.two !== false) for(let den=1; den<=H; den++){
        if(performance.now()>deadline) break;
        const target=den*x, denTol=absTol*den*1.0000001 + 1e-15;
        for(let ti=0; ti<terms.length; ti++){
          const a=terms[ti], want=target-a.value;
          for(const b of nearbyTerms(want)){
            if(b.idx===a.idx) continue;
            if(Math.abs(a.value+b.value-target)<=denTol) lowPrecisionLinearComboAdd(rows,seen,consts,settings,den,[a,b],'2-term exhaustive bucket scan');
          }
          if((ti&255)===255 && performance.now()>deadline) break;
        }
      }
      const twoTermDone=performance.now();
      if(progressCb) progressCb({phase:'1–2 term exhaustive scan', frac:Math.min(1,(twoTermDone-start)/budget), rows:rows.slice(0, Math.max(1, Math.min(50, Number(settings?.moduleLimits?.linearCombo || RIES_LOWPREC_LINEAR_COMBO_LIMIT))))});

      // 3-term v11.3.1 tiered meet-in-the-middle scan.  It searches the most
      // plausible integral/sum pairs first (π/log/zeta/radicals/gamma periods),
      // then spends the remaining budget on broader pairs.  Each attempted pair
      // still uses the full height<=36 coefficient range and the verifier above,
      // so the rows that appear are genuine primitive formulas, not rounded hash
      // artifacts.
      const pairTasks=lowPrecisionLinearComboPairTasks(consts);
      const tier1=Math.max(1, Math.ceil(pairTasks.length*.05));
      const tier2=Math.max(tier1, Math.ceil(pairTasks.length*.25));
      let pairDone=0, exhaustiveComplete=false;
      function scanPairTaskRange(from,to,method,frac0,frac1){
        const nCoeff=coeffOrder.length;
        for(let pp=from; pp<to; pp++){
          const task=pairTasks[pp], i=task.i, j=task.j, vi=consts[i].value, vj=consts[j].value;
          for(let den=1; den<=H; den++){
            const target=den*x, denTol=absTol*den*1.0000001 + 1e-15;
            for(let aiPos=0; aiPos<nCoeff; aiPos++){
              const ai=coeffOrder[aiPos], av=ai*vi;
              for(let bjPos=0; bjPos<nCoeff; bjPos++){
                const bj=coeffOrder[bjPos], pairValue=av + bj*vj;
                const want=target-pairValue;
                for(const c of nearbyTerms(want)){
                  if(c.idx===i || c.idx===j) continue;
                  if(Math.abs(pairValue+c.value-target)<=denTol) lowPrecisionLinearComboAdd(rows,seen,consts,settings,den,[{idx:i,coeff:ai},{idx:j,coeff:bj},c],method);
                }
              }
            }
          }
          pairDone++;
          if((pairDone&3)===0){
            const now=performance.now();
            if(progressCb){
              const local=(pp-from+1)/Math.max(1,to-from);
              progressCb({phase:method, frac:frac0+(frac1-frac0)*local, rows:rows.slice(0, Math.max(1, Math.min(50, Number(settings?.moduleLimits?.linearCombo || RIES_LOWPREC_LINEAR_COMBO_LIMIT)))), pairDone, pairTotal:pairTasks.length, elapsed:Math.round(now-start)});
            }
            if(now>deadline) return false;
          }
        }
        return true;
      }
      if(settings?.linearComboOptions?.three !== false && performance.now()<deadline) scanPairTaskRange(0,tier1,'3-term prioritized full-coefficient scan tier 1/3',.30,.54);
      if(settings?.linearComboOptions?.three !== false && pairDone>=tier1 && performance.now()<deadline) scanPairTaskRange(tier1,tier2,'3-term prioritized full-coefficient scan tier 2/3',.54,.78);
      if(settings?.linearComboOptions?.three !== false && pairDone>=tier2 && performance.now()<deadline) exhaustiveComplete=scanPairTaskRange(tier2,pairTasks.length,'3-term prioritized full-coefficient scan tier 3/3',.78,.96);

      const map=new Map();
      for(const r of rows){
        const k=normalizeResultTextKey(r.candidate);
        if(!map.has(k) || (r.score??1e99)<(map.get(k).score??1e99)) map.set(k,r);
      }
      const finalRows=[...map.values()].sort((a,b)=>(a.score??1e99)-(b.score??1e99) || (a.err||1)-(b.err||1)).slice(0, Math.max(1, Math.min(50, Number(settings?.moduleLimits?.linearCombo || RIES_LOWPREC_LINEAR_COMBO_LIMIT))));
      if(settings){
        settings._linearComboBasisSize=consts.length;
        settings._linearComboMs=Math.round(performance.now()-start);
        settings._linearComboPairDone=pairDone;
        settings._linearComboPairTotal=pairTasks.length;
        settings._linearComboExhaustiveComplete=exhaustiveComplete && pairDone>=pairTasks.length;
        settings._linearComboTolerance=lowPrecisionLinearComboRelTol(settings);
      }
      if(progressCb) progressCb({phase:(settings?._linearComboExhaustiveComplete?'finalizing':'finalizing under 3s budget'), frac:1, rows:finalRows.slice(0,RIES_LOWPREC_LINEAR_COMBO_LIMIT), elapsed:Math.round(performance.now()-start), basisSize:consts.length, pairDone, pairTotal:pairTasks.length});
      return finalRows;
    }





    // v11.6 lazy-loading split filtered hard-constant database matcher.
    // Level 4 loads a representative low-height 20% slice covering all categories;
    // level 5 loads the complementary rows and scans the cumulative table.
    const RIES_HARDDB_ROWS = 79932;
    const RIES_HARDDB_LIMIT = 5;
    const RIES_HARDDB_MIN_REL_TOL = 1e-12;
    const RIES_HARDDB_SPECIALS = [-2,-1,-0.5,0.5,1,2];
    const RIES_HARDDB_ASSET_LEVELS = [
      {stage:1, level:4, url:'assets/ries-harddb-v11_6-level4.js?v=11.6', label:'harddb level 4 low-height representative chunk', expectedBytes:989486},
      {stage:2, level:5, url:'assets/ries-harddb-v11_6-level5.js?v=11.6', label:'harddb level 5 remaining filtered chunk', expectedBytes:1948679}
    ];
    const RIES_HARDDB_LEGACY_ASSET_URL = 'assets/ries-harddb-v11_4_1-filtered.js?v=11.6';
    let hardDbValuesCache = null;
    let hardDbValuesCacheStage = 0;
    let hardDbRowMapCache = null;
    let hardDbRowMapCacheStage = 0;
    let hardDbDictCache = null;
    let hardDbRationalsCache = new Map();
    let hardDbOrigRowsCache = null;
    let hardDbOrigRowsCacheStage = 0;

    function hardDbChunksRaw(){ return (typeof window!=='undefined' && Array.isArray(window.RIES_HARDDB_V116_CHUNKS)) ? window.RIES_HARDDB_V116_CHUNKS : []; }
    function hardDbData(){
      const chunks=hardDbChunksRaw();
      return chunks[0] || chunks[1] || ((typeof window!=='undefined' && window.RIES_HARDDB_V114_DIRECT) ? window.RIES_HARDDB_V114_DIRECT : null);
    }
    function hardDbMaxStage(settings){
      const lvl=Math.max(1, Math.floor(Number(settings?.level || document.getElementById('level')?.value || DEFAULT_RIES_LEVEL) || DEFAULT_RIES_LEVEL));
      const opt=settings?.hardDbOptions || {depth4:true, depth5:true};
      if(lvl<4) return 0;
      if(lvl===4) return opt.depth4===false ? 0 : 1;
      if(opt.depth4===false) return 0;
      return opt.depth5===false ? 1 : 2;
    }
    function hardDbLoadedChunks(stage=2){
      const chunks=hardDbChunksRaw();
      const upto=Math.max(1, Math.min(2, Number(stage||2)));
      const out=[];
      for(let i=0;i<upto;i++) if(chunks[i]) out.push(chunks[i]);
      return out;
    }
    function hardDbLevelEnabled(settings){ return hardDbMaxStage(settings)>0; }
    function hardDbPotentiallyRunnable(settings){
      return !!settings && settings.modules?.hardDb!==false && hardDbLevelEnabled(settings) && Number.isFinite(settings.target) && !settings.complexTarget && settings.target!==0;
    }
    function isHardDbReady(stage=1){
      const chunks=hardDbChunksRaw();
      const upto=Math.max(1, Math.min(2, Number(stage||1)));
      for(let i=0;i<upto;i++) if(!chunks[i]) return !!window.RIES_HARDDB_V114_DIRECT;
      return true;
    }
    async function ensureHardDbLoaded(opts={}){
      const stage=Math.max(1, Math.min(2, Number(opts.stage || hardDbMaxStage(opts.settings||{}) || 1)));
      const base=Number.isFinite(opts.baseProgress) ? opts.baseProgress : .40;
      const span=Number.isFinite(opts.spanProgress) ? opts.spanProgress : .04;
      for(let i=0;i<stage;i++){
        const spec=RIES_HARDDB_ASSET_LEVELS[i];
        const loaded=await loadScriptPackageWithProgress(spec.url, ()=>isHardDbReady(spec.stage), {
          label: opts.label ? `${opts.label} ${spec.level}` : spec.label,
          phase: opts.phase || 'hard-constant database',
          baseProgress: base + span*(i/stage),
          spanProgress: span/stage,
          expectedBytes: spec.expectedBytes
        });
        if(!loaded) return false;
      }
      return isHardDbReady(stage);
    }
    function hardDbShouldRun(settings){
      return hardDbPotentiallyRunnable(settings) && isHardDbReady(hardDbMaxStage(settings));
    }
    function hardDbRelTol(settings){
      // User request: about 100 times the typed input precision error, e.g.
      // 10 typed significant digits -> around 1e-8.  Values are stored as a direct
      // Float64 table, so the floor can be much tighter than the old q-log grid.
      const sig=typedInputPrecision(settings);
      return Math.max(RIES_HARDDB_MIN_REL_TOL, typedRelativeToleranceNumber(sig,100,0,15));
    }
    function hardDbGcd(a,b){ a=Math.abs(a|0); b=Math.abs(b|0); while(b){ const t=a%b; a=b; b=t; } return a||1; }
    function hardDbRationalsHeight(maxHeight=10){
      const H=Math.max(1, Math.min(80, Math.floor(Number(maxHeight)||10)));
      if(hardDbRationalsCache.has(H)) return hardDbRationalsCache.get(H);
      const out=[]; const seen=new Set();
      for(let p=1;p<=H;p++) for(let q=1;q<=H;q++){
        const g=hardDbGcd(p,q), n=p/g, d=q/g, key=n+'/'+d;
        if(seen.has(key) || Math.max(Math.abs(n),Math.abs(d))>H) continue;
        seen.add(key); out.push({n,d,value:n/d,text:d===1?String(n):`${n}/${d}`,latex:d===1?String(n):`\frac{${n}}{${d}}`});
      }
      out.sort((a,b)=>a.value-b.value || a.n-b.n || a.d-b.d);
      hardDbRationalsCache.set(H,out); return out;
    }
    function hardDbRationalsHeight10(){ return hardDbRationalsHeight(10); }
    function hardDbAtob(s){
      if(typeof atob==='function') return atob(s);
      if(typeof Buffer!=='undefined') return Buffer.from(s,'base64').toString('binary');
      throw new Error('No base64 decoder is available for the direct hard DB asset.');
    }
    function hardDbB64ToBytes(b64){
      const pad=b64.endsWith('==')?2:(b64.endsWith('=')?1:0);
      const outLen=Math.floor(b64.length*3/4)-pad;
      const out=new Uint8Array(outLen);
      const chunkChars=262144; // multiple of four
      let pos=0;
      for(let i=0;i<b64.length;i+=chunkChars){
        const bin=hardDbAtob(b64.slice(i, Math.min(b64.length, i+chunkChars)));
        for(let j=0;j<bin.length;j++) out[pos++]=bin.charCodeAt(j)&255;
      }
      return out;
    }
    function hardDbB64ReadBytes(b64, byteOffset, byteLength){
      const aligned=byteOffset - (byteOffset % 3);
      const end=byteOffset + byteLength;
      const charStart=Math.floor(aligned/3)*4;
      const charEnd=Math.ceil(end/3)*4;
      const bin=hardDbAtob(b64.slice(charStart,charEnd));
      const skip=byteOffset-aligned;
      const out=new Uint8Array(byteLength);
      for(let i=0;i<byteLength;i++) out[i]=bin.charCodeAt(skip+i)&255;
      return out;
    }
    function hardDbChunkValues(ch){
      if(!ch) return new Float64Array(0);
      if(!ch._values){ const bytes=hardDbB64ToBytes(ch.valuesB64||''); ch._values=new Float64Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength/8)); }
      return ch._values;
    }
    function hardDbChunkRowMap(ch){
      if(!ch) return new Uint8Array(0);
      if(!ch._rowMap) ch._rowMap=hardDbB64ToBytes(ch.rowMapB64||'');
      return ch._rowMap;
    }
    function hardDbChunkOrigRows(ch){
      if(!ch) return new Uint32Array(0);
      if(!ch._origRows){ const bytes=hardDbB64ToBytes(ch.origRowsB64||''); ch._origRows=new Uint32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength/4)); }
      return ch._origRows;
    }
    function hardDbMergeChunksTyped(chunks, kind){
      if(kind==='values'){
        const arrays=chunks.map(hardDbChunkValues); const n=arrays.reduce((a,b)=>a+b.length,0); const out=new Float64Array(n); let off=0; for(const a of arrays){ out.set(a,off); off+=a.length; } return out;
      }
      if(kind==='rowmap'){
        const arrays=chunks.map(hardDbChunkRowMap); const n=arrays.reduce((a,b)=>a+b.length,0); const out=new Uint8Array(n); let off=0; for(const a of arrays){ out.set(a,off); off+=a.length; } return out;
      }
      const arrays=chunks.map(hardDbChunkOrigRows); const n=arrays.reduce((a,b)=>a+b.length,0); const out=new Uint32Array(n); let off=0; for(const a of arrays){ out.set(a,off); off+=a.length; } return out;
    }
    function hardDbValues(stage=null){
      const st=Math.max(1, Math.min(2, Number(stage || hardDbMaxStage(readSettings?.()) || 1)));
      const chunks=hardDbLoadedChunks(st);
      if(chunks.length){
        if(hardDbValuesCache && hardDbValuesCacheStage===st) return hardDbValuesCache;
        hardDbValuesCache=hardDbMergeChunksTyped(chunks,'values'); hardDbValuesCacheStage=st;
        return hardDbValuesCache;
      }
      if(hardDbValuesCache) return hardDbValuesCache;
      const d=hardDbData(); if(!d || !d.valuesB64) return new Float64Array(0);
      const bytes=hardDbB64ToBytes(d.valuesB64);
      hardDbValuesCache=new Float64Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength/8));
      hardDbValuesCacheStage=st;
      return hardDbValuesCache;
    }
    function hardDbRowMapBytes(stage=null){
      const st=Math.max(1, Math.min(2, Number(stage || hardDbMaxStage(readSettings?.()) || 1)));
      const chunks=hardDbLoadedChunks(st);
      if(chunks.length){
        if(hardDbRowMapCache && hardDbRowMapCacheStage===st) return hardDbRowMapCache;
        hardDbRowMapCache=hardDbMergeChunksTyped(chunks,'rowmap'); hardDbRowMapCacheStage=st;
        return hardDbRowMapCache;
      }
      if(hardDbRowMapCache) return hardDbRowMapCache;
      const d=hardDbData(); if(!d || !d.rowMapB64) return new Uint8Array(0);
      hardDbRowMapCache=hardDbB64ToBytes(d.rowMapB64); hardDbRowMapCacheStage=st;
      return hardDbRowMapCache;
    }
    function hardDbDictionary(){
      if(hardDbDictCache) return hardDbDictCache;
      const d=hardDbData();
      hardDbDictCache=String(d?.dict||'').split('	');
      return hardDbDictCache;
    }
    function hardDbOriginalRows(stage=null){
      const st=Math.max(1, Math.min(2, Number(stage || hardDbMaxStage(readSettings?.()) || 1)));
      const chunks=hardDbLoadedChunks(st);
      if(chunks.length){
        if(hardDbOrigRowsCache && hardDbOrigRowsCacheStage===st) return hardDbOrigRowsCache;
        hardDbOrigRowsCache=hardDbMergeChunksTyped(chunks,'orig'); hardDbOrigRowsCacheStage=st;
        return hardDbOrigRowsCache;
      }
      if(hardDbOrigRowsCache) return hardDbOrigRowsCache;
      const d=hardDbData();
      if(!d || !d.origRowsB64){ hardDbOrigRowsCache=new Uint32Array(0); return hardDbOrigRowsCache; }
      const bytes=hardDbB64ToBytes(d.origRowsB64);
      hardDbOrigRowsCache=new Uint32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength/4)); hardDbOrigRowsCacheStage=st;
      return hardDbOrigRowsCache;
    }
    function hardDbOriginalRow(rowIndex, stage=null){
      const rows=hardDbOriginalRows(stage);
      return (rowIndex>=0 && rowIndex<rows.length) ? rows[rowIndex] : rowIndex+1;
    }
    function hardDbDecodeRowMeta(rowIndex, stage=null){
      const d=hardDbData(); const map=hardDbRowMapBytes(stage);
      const off=rowIndex*3;
      if(!d || off+2>=map.length) return {cid:-1, category:'uploaded hard constant', params:{}};
      const cid=map[off];
      const ordinal=map[off+1] | (map[off+2]<<8);
      const keys=d.paramKeys?.[cid] || [];
      const paramIndex=(Number(d.catParamOffsets?.[cid]||0) + ordinal*keys.length);
      const bytes=hardDbB64ReadBytes(d.paramB64, paramIndex*2, keys.length*2);
      const dict=hardDbDictionary(); const p={};
      for(let i=0;i<keys.length;i++){
        const code=bytes[i*2] | (bytes[i*2+1]<<8);
        p[keys[i]]=dict[code] || '';
      }
      return {cid, category:(d.categories?.[cid] || 'uploaded hard constant'), params:p};
    }
    function hardDbUnquote(s){ return String(s||'').replace(/^"|"$/g,''); }
    function hardDbRatLatex(s){
      s=hardDbUnquote(s).trim();
      if(!s) return '';
      const m=s.match(/^(-?\d+)\/(\d+)$/);
      if(m){ const a=Number(m[1]), b=Number(m[2]); return (a<0?'-':'')+`\\frac{${Math.abs(a)}}{${b}}`; }
      return s.replace(/\*/g,'');
    }
    function hardDbRatText(s){ return hardDbUnquote(s).trim(); }
    function hardDbRatMinusOneLatex(s){
      s=hardDbUnquote(s).trim();
      let n=0,d=1; const m=s.match(/^(-?\d+)\/(\d+)$/);
      if(m){ n=Number(m[1]); d=Number(m[2]); } else if(/^[-+]?\d+$/.test(s)){ n=Number(s); d=1; } else return `${hardDbRatLatex(s)}-1`;
      n-=d; const g=hardDbGcd(n,d); n/=g; d/=g;
      if(n===0) return '0';
      if(d===1) return String(n);
      return (n<0?'-':'')+`\\frac{${Math.abs(n)}}{${d}}`;
    }
    function hardDbListItems(s){
      s=hardDbUnquote(s).trim();
      if(s==='{}' || !s) return [];
      if(s[0]==='{' && s[s.length-1]==='}') s=s.slice(1,-1);
      if(!s.trim()) return [];
      return s.split(',').map(x=>x.trim()).filter(Boolean);
    }
    function hardDbListLatex(s){
      return hardDbListItems(s).map(x=>hardDbRatLatex(x)).join(', ');
    }
    function hardDbListText(s){
      return hardDbListItems(s).map(x=>hardDbRatText(x)).join(', ');
    }
    function hardDbComboLatex(template, params){
      const caseNo=Math.max(1, Math.min(8, Number(hardDbUnquote(template)||1)));
      const items=hardDbListItems(params);
      const P=i=>hardDbRatLatex(items[i] || '0');
      const a=P(0), b=P(1), c=P(2), d=P(3);
      switch(caseNo){
        case 1: return `\\log\\left(1+e^{-${a}}+\\sin^2(${b}\\pi)+${c}^2\\right)`;
        case 2: return `e^{-${a}}\\cos(${b}\\pi)+\\log\\left(1+${c}+${d}^2\\right)`;
        case 3: return `\\log\\left(1+${a}e^{-${b}}+\\tan^2(${c}\\pi/4)\\right)+\\arctan(${d})`;
        case 4: return `e^{-${a}}\\left(\\sin(${b}\\pi)+\\cos^2(${c}\\pi)\\right)+\\log(1+${d})`;
        case 5: return `\\log\\left(1+e^{-${a}}\\log^2(1+${b})+\\sin^2(${c}\\pi)\\right)`;
        case 6: return `\\operatorname{arsinh}\\left(${a}+\\sin(${b}\\pi)\\right)+\\log(1+e^{-${c}})`;
        case 7: return `\\log\\left(1+e^{-${a}}\\cos^2(${b}\\pi)\\right)+e^{-${c}}\\sin(${d}\\pi)`;
        default: return `\\log\\left(1+\\sqrt{${a}}+e^{-${b}}+\\cos^2(${c}\\pi/2)\\right)-\\arctan(${d})`;
      }
    }
    function hardDbSpecialLatexValue(x){
      if(x===-2) return '-2'; if(x===-1) return '-1'; if(x===-0.5) return '-\\frac{1}{2}';
      if(x===0.5) return '\\frac{1}{2}'; if(x===1) return '1'; if(x===2) return '2';
      return String(x);
    }
    function hardDbSpecialTextValue(x){ return x===-0.5 ? '-1/2' : (x===0.5 ? '1/2' : String(x)); }
    function hardDbFormulaLatexRaw(meta){
      const c=meta.category, p=meta.params, L=hardDbRatLatex, E=hardDbRatMinusOneLatex, list=hardDbListLatex;
      const cat=String(c||'');
      if(cat.includes('Euler beta integral')) return `\\int_0^1 x^{${E(p.a)}}(1-x)^{${E(p.b)}}\\,dx`;
      if(cat.includes('incomplete beta integral')) return `\\int_0^{${L(p.x)}} t^{${E(p.a)}}(1-t)^{${E(p.b)}}\\,dt`;
      if(cat.includes('beta logarithmic integral')) return `\\int_0^1 x^{${E(p.a)}}(1-x)^{${E(p.b)}}(\\log x)^{${L(p.logPower)}}\\,dx`;
      if(cat.includes('gamma log-laplace integral')) return `\\int_0^\\infty x^{${E(p.a)}} e^{-${L(p.q)}x}(\\log x)^{${L(p.logPower)}}\\,dx`;
      if(cat.includes('Bessel J Mellin')) return `\\int_0^\\infty x^{${E(p.s)}}J_{${L(p.nu)}}(x)\\,dx`;
      if(cat.includes('Bessel K Mellin')) return `\\int_0^\\infty x^{${E(p.s)}}K_{${L(p.nu)}}(${L(p.q)}x)\\,dx`;
      if(cat==='Bessel Mellin integral fast'){
        const kind=hardDbUnquote(p.kind);
        if(kind==='J') return `\\int_0^\\infty x^{${E(p.s)}}J_{${L(p.nu)}}(x)\\,dx`;
        return `\\int_0^\\infty x^{${E(p.s)}}K_{${L(p.nu)}}(${L(p.q)}x)\\,dx`;
      }
      if(cat.includes('trigonometric Mellin')){ const k=hardDbUnquote(p.kind)||'cos'; return `\\int_0^\\infty x^{${E(p.mu)}}\\${k}(${L(p.q)}x^{${L(p.power)}})\\,dx`; }
      if(cat.includes('exponential rational Laplace')) return `\\int_0^\\infty \\frac{x^{${E(p.a)}}e^{-${L(p.q)}x}}{(1+x)^{${L(p.v)}}}\\,dx`;
      if(cat.includes('log-one-minus integral')) return `\\int_0^1 x^{${E(p.a)}}\\log(1-${L(p.z)}x)\\,dx`;
      if(cat==='elliptic K integral') return `\\int_0^{\\pi/2}\\frac{d\\theta}{\\sqrt{1-${L(p.m)}\\sin^2\\theta}}`;
      if(cat==='elliptic E integral') return `\\int_0^{\\pi/2}\\sqrt{1-${L(p.m)}\\sin^2\\theta}\\,d\\theta`;
      if(cat==='elliptic Pi integral') return `\\int_0^{\\pi/2}\\frac{d\\theta}{(1-${L(p.n)}\\sin^2\\theta)\\sqrt{1-${L(p.m)}\\sin^2\\theta}}`;
      if(cat==='polylog root-of-unity sum') return `\\operatorname{${hardDbUnquote(p.part)}}\\operatorname{Li}_{${L(p.s)}}\\!\\left(e^{2\\pi i ${L(p.r)}}\\right)`;
      if(cat==='Lerch transcendent sum') return `\\sum_{n=0}^{\\infty}\\frac{(${L(p.z)})^n}{(n+${L(p.a)})^{${L(p.s)}}}`;
      if(cat==='Hurwitz zeta sum') return `\\sum_{n=0}^{\\infty}\\frac{1}{(n+${L(p.a)})^{${L(p.s)}}}`;
      if(cat==='Gauss hypergeometric value') return `{}_{2}F_{1}\\!\\left(\\begin{matrix}${L(p.a)}, ${L(p.b)}\\\\${L(p.c)}\\end{matrix};${L(p.z)}\\right)`;
      if(cat==='generalized hypergeometric value') return `{}_{3}F_{2}\\!\\left(\\begin{matrix}${L(p.a1)}, ${L(p.a2)}, ${L(p.a3)}\\\\${L(p.b1)}, ${L(p.b2)}\\end{matrix};${L(p.z)}\\right)`;
      if(cat==='low-height hypergeometric pFq') return `{}_{${L(p.p)}}F_{${L(p.q)}}\\!\\left(\\begin{matrix}${list(p.a)}\\\\${list(p.b)}\\end{matrix};${L(p.z)}\\right)`;
      if(cat==='incomplete gamma integral') return `\\int_{${L(p.x)}}^{\\infty} t^{${E(p.a)}}e^{-t}\\,dt`;
      if(cat==='rational Mellin integral' || cat==='rational Mellin integral fast') return `\\int_0^\\infty \\frac{x^{${E(p.a)}}}{1+x^{${L(p.b)}}}\\,dx`;
      if(cat==='Bessel/Airy special values' || cat==='Bessel-Airy value fallback'){
        const fn=hardDbUnquote(p.function);
        if(fn==='AiryAi' || fn==='AiryBi') return `\\operatorname{${fn.replace('Airy','')}}\\!\\left(${L(p.z)}\\right)`;
        const short=fn.replace('Bessel','');
        const zLatex = cat==='Bessel-Airy value fallback' ? `\\left|${L(p.z)}\\right|+\\frac{1}{7}` : L(p.z);
        return `${short}_{${L(p.nu)}}\\!\\left(${zLatex}\\right)`;
      }
      if(cat==='special-function zeros'){
        const fn=hardDbUnquote(p.function);
        if(fn==='BesselJZero') return `j_{${L(p.nu)},${L(p.k)}}`;
        if(fn==='BesselYZero') return `y_{${L(p.nu)},${L(p.k)}}`;
        if(fn==='AiryAiZero') return `a_{\\operatorname{Ai},${L(p.k)}}`;
        if(fn==='AiryBiZero') return `a_{\\operatorname{Bi},${L(p.k)}}`;
      }
      if(cat==='Barnes/Gamma products'){
        const fn=hardDbUnquote(p.function);
        if(fn==='LogBarnesG') return `\\log G\\!\\left(${L(p.a)}\\right)`;
        if(fn==='LogBeta') return `\\log B\\!\\left(${L(p.a)},${L(p.b)}\\right)`;
        if(fn==='GammaRatio') return `\\log\\frac{\\Gamma(${L(p.a)}+${L(p.b)})}{\\Gamma(${L(p.a)})\\Gamma(${L(p.b)})}`;
      }
      if(cat==='Barnes-Gamma fallback'){
        const kind=Number(hardDbUnquote(p.kind)||0);
        if(kind===1) return `\\log B\\!\\left(${L(p.a)},${L(p.b)}\\right)`;
        if(kind===2) return `\\log\\frac{\\Gamma(${L(p.a)}+${L(p.b)})}{\\Gamma(${L(p.a)})\\Gamma(${L(p.b)})}`;
        if(kind===3) return `\\log\\frac{G(${L(p.a)}+${L(p.b)})}{G(${L(p.a)})}`;
        return `\\log\\Gamma(${L(p.a)})+\\sin(${L(p.b)}\\pi)\\log(1+${L(p.a)})`;
      }
      if(cat==='common Log-Exp-Trig composition') return hardDbComboLatex(p.template, p.params);

      return `A_{${meta.cid}}(${Object.keys(p).map(k=>`${k}=${L(p[k])}`).join(',')})`;
    }
    function hardDbFormulaLatex(meta){ return sanitizeLatexForDisplay(hardDbFormulaLatexRaw(meta)); }
    function hardDbValueString(A){
      if(!Number.isFinite(A)) return String(A);
      return A.toPrecision(20).replace(/e\+/,'e');
    }
    function hardDbMakeTargetSpecs(settings){
      const x=settings.target, ax=Math.abs(x), logx=Math.log(ax), signX=x<0?-1:1;
      const specs=[];
      function add(sp){
        if(Number.isFinite(sp.targetAbs) && sp.targetAbs>0) specs.push(sp);
      }
      const opt=settings?.hardDbOptions || {};
      const rh=Number(opt.rationalHeight || 10);
      if(opt.rational!==false){
        for(const r of hardDbRationalsHeight(rh)){
          add({type:'rat', targetAbs:ax/r.value, rational:r, signX, label:`|x/A|=${r.text}`});
        }
      }
      for(const s of RIES_HARDDB_SPECIALS){
        if(opt.power!==false) add({type:'loglog', targetAbs:Math.exp(logx/s), s, signX, label:`log|x|/log|A|=${hardDbSpecialTextValue(s)}`});
        const a=logx/s;
        if(opt.exponential!==false && Number.isFinite(a) && a!==0) add({type:'logovera', targetAbs:Math.abs(a), expectedNeg:a<0, s, signX, label:`log|x|/A=${hardDbSpecialTextValue(s)}`});
        if(opt.logScale!==false) add({type:'xoverlog', targetAbs:Math.exp(x/s), s, label:`x/log|A|=${hardDbSpecialTextValue(s)}`});
      }
      specs.sort((a,b)=>a.targetAbs-b.targetAbs || String(a.label).localeCompare(String(b.label)));
      return specs;
    }
    function hardDbLowerBoundSpecs(specs, value){
      let lo=0, hi=specs.length;
      while(lo<hi){ const mid=(lo+hi)>>1; if(specs[mid].targetAbs<value) lo=mid+1; else hi=mid; }
      return lo;
    }
    function hardDbFormulaForSpec(spec, A, aLatex){
      if(spec.type==='rat'){
        const prefix=spec.signX<0?'-':'';
        const r=spec.rational;
        const texCoeff=r.d===1 ? (r.n===1?'':String(r.n)) : `\\frac{${r.n}}{${r.d}}`;
        const textCoeff=r.d===1 ? (r.n===1?'':String(r.n)) : `${r.n}/${r.d}`;
        return {text:`${prefix}${textCoeff}${textCoeff?'·':''}|A|`, latex:`x \\approx ${prefix}${texCoeff}${texCoeff?'\\,':''}\\left|${aLatex}\\right|`};
      }
      const sT=hardDbSpecialTextValue(spec.s), sL=hardDbSpecialLatexValue(spec.s);
      const prefix=spec.signX<0?'-':'';
      if(spec.type==='loglog') return {text:`sign(x)·|A|^(${sT})`, latex:sanitizeLatexForDisplay(`x \\approx ${prefix}${latexPow(`\\left|${aLatex}\\right|`, sL)}`)};
      if(spec.type==='logovera') return {text:`sign(x)·exp(${sT}·A)`, latex:sanitizeLatexForDisplay(`x \\approx ${prefix}\\exp\\left(${latexMulScalar(sL, aLatex)}\\right)`)};
      if(spec.type==='xoverlog') return {text:`${sT}·log|A|`, latex:sanitizeLatexForDisplay(`x \\approx ${latexMulScalar(sL, `\\log\\left|${aLatex}\\right|`)}`)};
      return {text:'A relation', latex:sanitizeLatexForDisplay(`x \\approx ${aLatex}`)};
    }
    function hardDbParamLabel(key, meta){
      const cat=String(meta?.category||'');
      const base={
        a:'rational parameter a', b:'rational parameter b', c:'rational parameter c', z:'evaluation point z', x:'upper/lower limit x',
        a1:'first upper hypergeometric parameter', a2:'second upper hypergeometric parameter', a3:'third upper hypergeometric parameter',
        b1:'first lower hypergeometric parameter', b2:'second lower hypergeometric parameter',
        logPower:'integer log-power m', mu:'Mellin exponent μ', nu:'Bessel order ν', s:'series/integral exponent s',
        q:'scale parameter q', v:'denominator exponent v', power:'power p in x^p', r:'root-of-unity angle r in e^{2πir}', part:'real/imaginary part selector',
        kind:'subfamily selector', m:'elliptic parameter m', n:'elliptic Π characteristic n', function:'special function name', k:'zero index k',
        template:'Log/Exp/Trig template number', params:'template rational parameter list', p:'number of upper hypergeometric parameters p'
      };
      if(key==='q' && cat.includes('hypergeometric')) return 'number of lower hypergeometric parameters q';
      if(key==='p' && cat.includes('hypergeometric')) return 'number of upper hypergeometric parameters p';
      if(key==='a' && cat.includes('hypergeometric')) return 'upper hypergeometric parameter list a';
      if(key==='b' && cat.includes('hypergeometric')) return 'lower hypergeometric parameter list b';
      if(key==='a' && cat.includes('Hurwitz')) return 'Hurwitz shift a';
      if(key==='a' && cat.includes('beta')) return 'beta shape parameter a';
      if(key==='b' && cat.includes('beta')) return 'beta shape parameter b';
      if(key==='q' && /Laplace|Mellin|Bessel/.test(cat)) return 'positive scale q';
      return base[key] || `parameter ${key}`;
    }
    function hardDbParamValueText(key, val){
      if(key==='params') return hardDbListText(val);
      if(/^[-+]?\d+(?:\/\d+)?$/.test(hardDbUnquote(val).trim())) return hardDbRatText(val);
      return hardDbUnquote(val);
    }
    function hardDbParamDefinitionsHtml(meta){
      const p=meta?.params || {};
      const parts=[];
      for(const key of Object.keys(p)){
        const v=hardDbParamValueText(key, p[key]);
        if(v!==undefined && String(v).length) parts.push(`<code>${escapeHtml(key)}</code> = ${escapeHtml(v)} <span class="muted">(${escapeHtml(hardDbParamLabel(key, meta))})</span>`);
      }
      return parts.join('; ');
    }
    function hardDbFunctionDefinitions(meta){
      const cat=String(meta?.category||'');
      const defs=[];
      if(/Gamma|gamma|Barnes|beta/i.test(cat)) defs.push('Γ is the gamma function, B is the beta function, and G is the Barnes G-function when shown.');
      if(/Bessel/i.test(cat)) defs.push('Jν, Iν, and Kν denote Bessel / modified Bessel functions of order ν.');
      if(/Airy/i.test(cat)) defs.push('Ai and Bi denote Airy functions.');
      if(/polylog|Lerch/i.test(cat)) defs.push('Li_s is the polylogarithm; e^{2πir} is a root of unity.');
      if(/Hurwitz|zeta|Dirichlet/i.test(cat)) defs.push('ζ(s,a) / the displayed sum is the Hurwitz-zeta type series.');
      if(/hypergeometric/i.test(cat)) defs.push('{}_pF_q is the generalized hypergeometric function; p and q are the counts of upper and lower parameters.');
      if(/elliptic/i.test(cat)) defs.push('K, E, and Π are complete elliptic integrals in their displayed integral forms.');
      if(/Log-Exp-Trig/.test(cat)) defs.push('log is the natural logarithm; exp/e, sin, cos, tan, arctan, and arsinh have their standard meanings.');
      return defs.join(' ');
    }
    function hardDbLocalVariableDefinitions(meta){
      const cat=String(meta?.category||'');
      const defs=[];
      if(/integral|Mellin|Laplace|beta|gamma/i.test(cat)) defs.push('the x/t/θ appearing inside an integral is a local integration variable, not a second target');
      if(/sum|zeta|Lerch|Dirichlet/i.test(cat)) defs.push('n in a displayed infinite sum is the summation index');
      return defs.join('; ');
    }
    function hardDbMatchRuleText(spec){
      if(spec.type==='rat') return `target x is compared with a small rational multiple of |A|; here |x/A| = ${spec.rational.text}`;
      if(spec.type==='loglog') return `target x is compared through a power law, log|x| / log|A| = ${hardDbSpecialTextValue(spec.s)}`;
      if(spec.type==='logovera') return `target x is compared through an exponential law, log|x| / A = ${hardDbSpecialTextValue(spec.s)}`;
      if(spec.type==='xoverlog') return `target x is compared with ${hardDbSpecialTextValue(spec.s)}·log|A|`;
      return 'target x is compared with the database value A by the displayed relation';
    }
    function hardDbExplanationHtml(meta, spec, rel, settings, relTol){
      const paramHtml=hardDbParamDefinitionsHtml(meta);
      const localDefs=hardDbLocalVariableDefinitions(meta);
      const fnDefs=hardDbFunctionDefinitions(meta);
      const lines=[];
      lines.push(`<div><b>Definitions.</b> <code>x</code> is the user target; <code>A</code> is the hard-database constant defined by the formula above; ${escapeHtml(hardDbMatchRuleText(spec))}.</div>`);
      if(paramHtml) lines.push(`<div><b>Parameters.</b> ${paramHtml}.</div>`);
      if(localDefs || fnDefs) lines.push(`<div><b>Notation.</b> ${escapeHtml([localDefs, fnDefs].filter(Boolean).join('. '))}</div>`);
      lines.push(`<div><b>Acceptance.</b> Relative tolerance is about ${escapeHtml(relTol.toExponential(2))}, derived from the typed precision; displayed error is absolute error in x.</div>`);
      return `<div class="harddb-explain">${lines.join('')}</div>`;
    }

    function hardDbPredictionAndError(spec, A, x){
      const absA=Math.abs(A); const lnA=Math.log(absA); let pred=NaN;
      if(spec.type==='rat') pred=spec.signX*spec.rational.value*absA;
      else if(spec.type==='loglog') pred=spec.signX*Math.exp(spec.s*lnA);
      else if(spec.type==='logovera') pred=spec.signX*Math.exp(spec.s*A);
      else if(spec.type==='xoverlog') pred=spec.s*lnA;
      const errAbs=Math.abs(pred-x);
      return {pred, errAbs, rel:errAbs/Math.max(1,Math.abs(x))};
    }

    function hardDbLimit(settings){
      return Math.max(1, Math.min(50, Number(settings?.moduleLimits?.hardDb || RIES_HARDDB_LIMIT) || RIES_HARDDB_LIMIT));
    }
    function hardDbBudgetMs(settings){
      const opt=settings?.stageBudgets || {};
      const stage=hardDbMaxStage(settings);
      const key=stage<=1 ? 'hardDb4Ms' : 'hardDb5Ms';
      return stageBudgetValueToMs(Object.prototype.hasOwnProperty.call(opt,key) ? opt[key] : opt.hardDbMs, 1000, 0, 120000);
    }
    function hardDbMetaHeight(meta){
      let h=1;
      const params=meta?.params || {};
      for(const val of Object.values(params)){
        const txt=String(val||'').replace(/^"|"$/g,'');
        const re=/-?\d+\/\d+|-?\d+/g;
        let m;
        while((m=re.exec(txt))){
          const parts=m[0].split('/');
          for(const part of parts){ const n=Math.abs(Number(part)); if(Number.isFinite(n)) h=Math.max(h,n); }
        }
      }
      return h;
    }
    function hardDbCandidateInsert(best, cand, maxKeep=80){
      const key=`${cand.rowIndex}|${cand.spec.type}|${cand.spec.label}`;
      const old=best.get(key);
      if(!old || cand.errAbs<old.errAbs) best.set(key,cand);
      if(best.size>maxKeep*3){
        const arr=[...best.values()].sort((a,b)=>a.errAbs-b.errAbs || a.complexity-b.complexity).slice(0,maxKeep);
        best.clear(); for(const x of arr) best.set(`${x.rowIndex}|${x.spec.type}|${x.spec.label}`,x);
      }
    }
    function hardDbSearchValues(values, settings, progressCb=null){
      const x=settings.target; const relTol=hardDbRelTol(settings);
      const specs=hardDbMakeTargetSpecs(settings);
      const best=new Map(); const deadline=performance.now()+hardDbBudgetMs(settings);
      const wide=Math.max(relTol*1.5, 4e-16);
      for(let i=0;i<values.length;i++){
        const A=values[i]; const absA=Math.abs(A);
        if(!(absA>0) || !Number.isFinite(absA)) continue;
        const lo=absA*(1-wide), hi=absA*(1+wide);
        for(let j=hardDbLowerBoundSpecs(specs, lo); j<specs.length && specs[j].targetAbs<=hi; j++){
          const sp=specs[j];
          if(sp.type==='logovera' && (A<0)!==!!sp.expectedNeg) continue;
          const pe=hardDbPredictionAndError(sp, A, x);
          if(!Number.isFinite(pe.errAbs) || pe.rel>relTol*1.25) continue;
          const complexity=String(sp.label).length + (sp.type==='rat' ? 0 : 12);
          hardDbCandidateInsert(best,{rowIndex:i, A, spec:sp, errAbs:pe.errAbs, rel:pe.rel, pred:pe.pred, complexity}, Math.max(80, hardDbLimit(settings)*16));
        }
        if((i&65535)===0 && progressCb){
          progressCb({phase:'direct Float64 scan', done:i, total:values.length, rows:[...best.values()].sort((a,b)=>a.errAbs-b.errAbs).slice(0,hardDbLimit(settings))});
          if(performance.now()>deadline && best.size>=hardDbLimit(settings)) break;
        }
      }
      return [...best.values()].sort((a,b)=>a.errAbs-b.errAbs || a.complexity-b.complexity).slice(0,hardDbLimit(settings));
    }
    async function hardDbRowsAsync(settings, progressCb=null){
      if(!hardDbPotentiallyRunnable(settings)) return [];
      if(!isHardDbReady()){
        const loaded=await ensureHardDbLoaded({settings, stage:hardDbMaxStage(settings), label:'filtered hard-constant database', phase:'hard-constant database', baseProgress:.40, spanProgress:.04});
        if(!loaded) return [];
      }
      if(!hardDbShouldRun(settings)) return [];
      const t0=performance.now();
      const hdStage=hardDbMaxStage(settings);
      if(progressCb) progressCb({phase:'decoding direct Float64 table', done:0, total:1, rows:[]});
      let values=null;
      try{ values=hardDbValues(hdStage); }catch(e){ console.warn(e); return []; }
      if(!values.length) return [];
      if(progressCb) progressCb({phase:`scanning ${values.length} filtered direct constants`, done:0, total:values.length, rows:[]});
      const hits=hardDbSearchValues(values, settings, progressCb);
      if(!hits.length) return [];
      if(progressCb) progressCb({phase:'reading compact direct formula metadata', done:1, total:1, rows:[]});
      const rows=[];
      for(const h of hits){
        const meta=hardDbDecodeRowMeta(h.rowIndex, hdStage);
        const metaHeight=hardDbMetaHeight(meta);
        const maxMetaHeight=Number(settings?.hardDbOptions?.maxParamHeight || 15);
        if(Number.isFinite(maxMetaHeight) && metaHeight>maxMetaHeight) continue;
        const aLatex=hardDbFormulaLatex(meta);
        const rel=hardDbFormulaForSpec(h.spec, h.A, aLatex);
        const cat=meta.category || 'uploaded hard constant';
        const aval=hardDbValueString(h.A);
        const originalRow=hardDbOriginalRow(h.rowIndex, hdStage);
        const totalSource=Number(hardDbData()?.sourceRows||420000);
        const desc=`${cat}; harddb depth ${RIES_HARDDB_ASSET_LEVELS[Math.max(0,hdStage-1)]?.level || 5}; filtered row ${h.rowIndex+1} of ${values.length}; original source row ${originalRow} of ${totalSource}; parameter height ${metaHeight}`;
        const relTol=hardDbRelTol(settings);
        const explainHtml=hardDbExplanationHtml(meta, h.spec, rel, settings, relTol);
        const valueHtml=`<div><b>A = <span class="latex-render">\\(${escapeHtml(aLatex)}\\)</span></b></div><div class="muted">${escapeHtml(desc)}</div><div>A ≈ ${escapeHtml(aval)} <span class="muted">(direct 64-bit table)</span></div><div>${escapeHtml(h.spec.label)}; predicted x ≈ ${escapeHtml(fmtValue(h.pred))}; module ${Math.round(performance.now()-t0)} ms</div>${explainHtml}`;
        rows.push({
          candidate:`hard constant database: x ≈ ${rel.text}`,
          latex:rel.latex,
          copyLatex:rel.latex,
          valueHtml,
          copyValue:`A ≈ ${aval}; ${desc}`,
          err:h.errAbs,
          errText:fmtErr(h.errAbs),
          hardDbCategory:h.spec.type,
          constantDbCategory:'uploaded hard-constant database',
          constantDbSource:'uploaded420k-filtered80k-direct',
          constantDbId:`rhc_${String(originalRow).padStart(7,'0')}`,
          terms:h.spec.type==='rat'?2:3,
          height: h.spec.type==='rat' ? BigInt(Math.max(h.spec.rational.n,h.spec.rational.d)) : 2n,
          score: formulaVisibleLength(rel.text) + (h.spec.type==='rat'?0:18) + h.rel*1e8
        });
      }
      return rows.sort((a,b)=>(a.score??1e99)-(b.score??1e99) || (a.err||1)-(b.err||1)).slice(0,hardDbLimit(settings));
    }


    // v11.6 incremental lazy hypergeometric pFq database matcher.
    // The v11.5 monolithic asset is split into level4/5/6 chunks.  Higher levels
    // load all lower chunks and compare all loaded H rows against all loaded
    // multiplier families, so low-tier 2F1/3F2 values are not missed when a
    // level-5/6 search enables gamma or deeper pFq coefficients.
    const RIES_HYPDATA_LIMIT = 5;
    const RIES_HYPDATA_MIN_REL_TOL = 1e-12;
    const RIES_HYPDATA_TOTAL_ROWS = 109738;
    const RIES_HYPDATA_ASSET_LEVELS = [
      {stage:1, level:4, url:'assets/ries-hypdata-v11_5_2-level4.js?v=11.6', label:'pFq level 4 2F1/3F2 chunk', expectedBytes:438798},
      {stage:2, level:5, url:'assets/ries-hypdata-v11_5_2-level5.js?v=11.6', label:'pFq level 5 4F3/5F4 chunk', expectedBytes:5232588},
      {stage:3, level:6, url:'assets/ries-hypdata-v11_5_2-level6.js?v=11.6', label:'pFq level 6 full/deep chunk', expectedBytes:12161028}
    ];

    function hypDataLimit(settings){
      return Math.max(1, Math.min(50, Number(settings?.moduleLimits?.hypData || RIES_HYPDATA_LIMIT) || RIES_HYPDATA_LIMIT));
    }
    function hypDataChunksRaw(){
      return (typeof window!=='undefined' && Array.isArray(window.RIES_HYPDATA_V1152_CHUNKS)) ? window.RIES_HYPDATA_V1152_CHUNKS : [];
    }
    function hypDataMaxStage(settings){
      const lvl=Math.max(1, Math.floor(Number(settings?.level || document.getElementById('level')?.value || DEFAULT_RIES_LEVEL) || DEFAULT_RIES_LEVEL));
      if(lvl<4) return 0;
      const base = lvl===4 ? 1 : (lvl===5 ? 2 : 3);
      const opt=settings?.hypDataOptions || {depth1:true,depth2:true,depth3:true};
      let max=0;
      if(base>=1 && opt.depth1!==false) max=1; else return 0;
      if(base>=2 && opt.depth2!==false) max=2; else return max;
      if(base>=3 && opt.depth3!==false) max=3;
      return max;
    }
    function hypDataStageLabel(stage){
      return stage<=1 ? 'level 4 fast/common 2F1–3F2 layer' : (stage===2 ? 'level 5 classical 4F3/5F4 + gamma layer' : 'level 6 full deep pFq layer');
    }
    function isHypDataReady(stage=1){
      const chunks=hypDataChunksRaw();
      const upto=Math.max(1, Math.min(3, Number(stage||1)));
      for(let i=0;i<upto;i++) if(!chunks[i]) return false;
      return true;
    }
    function hypDataLoadedChunks(stage=3){
      const chunks=hypDataChunksRaw();
      const upto=Math.max(1, Math.min(3, Number(stage||3)));
      const out=[];
      for(let i=0;i<upto;i++) if(chunks[i]) out.push(chunks[i]);
      return out;
    }
    function hypDataPotentiallyRunnable(settings){
      if(!settings || settings.modules?.hypData===false || hypDataMaxStage(settings)<1) return false;
      if(settings.complexTarget) return !!settings.parsedComplex;
      return Number.isFinite(settings.target) && settings.target!==0;
    }
    async function ensureHypDataLoaded(opts={}){
      const stage=Math.max(1, Math.min(3, Number(opts.stage || hypDataMaxStage(opts.settings||{}) || 1)));
      const base=Number.isFinite(opts.baseProgress) ? opts.baseProgress : .49;
      const span=Number.isFinite(opts.spanProgress) ? opts.spanProgress : .05;
      for(let i=0;i<stage;i++){
        const spec=RIES_HYPDATA_ASSET_LEVELS[i];
        const loaded=await loadScriptPackageWithProgress(spec.url, ()=>isHypDataReady(spec.stage), {
          label: opts.label ? `${opts.label} ${spec.level}` : spec.label,
          phase: opts.phase || 'hypergeometric database',
          baseProgress: base + span*(i/stage),
          spanProgress: span/stage,
          expectedBytes: spec.expectedBytes
        });
        if(!loaded) return false;
      }
      return isHypDataReady(stage);
    }
    function hypDataB64Bytes(b64){ return hardDbB64ToBytes(b64 || ''); }
    function hypDataFloat64(b64){ const bytes=hypDataB64Bytes(b64); return new Float64Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength/8)); }
    function hypDataFloat32(b64){ const bytes=hypDataB64Bytes(b64); return new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength/4)); }
    function hypDataUint32(b64){ const bytes=hypDataB64Bytes(b64); return new Uint32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength/4)); }
    function hypDataUint16(b64){ const bytes=hypDataB64Bytes(b64); return new Uint16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength/2)); }
    function hypDataUint8(b64){ return hypDataB64Bytes(b64); }
    function hypDataChunkMkLines(ch){ if(!ch._mkLines) ch._mkLines=String(ch.mkBlob||'').split('\n'); return ch._mkLines; }
    function hypDataChunkValue20Lines(ch){ if(!ch._value20Lines) ch._value20Lines=String(ch.value20Blob||'').split('\n'); return ch._value20Lines; }
    function hypDataChunkP(ch){ if(!ch._p) ch._p=hypDataUint8(ch.pB64); return ch._p; }
    function hypDataChunkQ(ch){ if(!ch._q) ch._q=hypDataUint8(ch.qB64); return ch._q; }
    function hypDataChunkSource(ch){ if(!ch._source) ch._source=hypDataUint8(ch.sourceB64); return ch._source; }
    function hypDataChunkComplexity(ch){ if(!ch._complexity) ch._complexity=hypDataUint16(ch.complexityB64); return ch._complexity; }
    function hypDataChunkRealValues(ch){ if(!ch._realValues) ch._realValues=hypDataFloat64(ch.realValuesB64); return ch._realValues; }
    function hypDataChunkRealRows(ch){ if(!ch._realRows) ch._realRows=hypDataUint32(ch.realRowB64); return ch._realRows; }
    function hypDataChunkComplexRe(ch){ if(!ch._complexRe) ch._complexRe=hypDataFloat64(ch.complexReB64); return ch._complexRe; }
    function hypDataChunkComplexIm(ch){ if(!ch._complexIm) ch._complexIm=hypDataFloat64(ch.complexImB64); return ch._complexIm; }
    function hypDataChunkComplexRows(ch){ if(!ch._complexRows) ch._complexRows=hypDataUint32(ch.complexRowB64); return ch._complexRows; }
    function hypDataChunkMultValues(ch){ if(!ch._multValues) ch._multValues=hypDataFloat64(ch.multValuesB64); return ch._multValues; }
    function hypDataChunkMultComplexity(ch){ if(!ch._multComplexity) ch._multComplexity=hypDataFloat32(ch.multComplexityB64); return ch._multComplexity; }
    function hypDataChunkMultTextLines(ch){ if(!ch._multTextLines) ch._multTextLines=String(ch.multTextBlob||'').split('\n'); return ch._multTextLines; }
    function hypDataChunkMultLatexLines(ch){ if(!ch._multLatexLines) ch._multLatexLines=String(ch.multLatexBlob||'').split('\n'); return ch._multLatexLines; }
    function hypDataChunkMultFamilyCodes(ch){ if(!ch._multFamilyCodes) ch._multFamilyCodes=hypDataUint8(ch.multFamilyB64); return ch._multFamilyCodes; }
    function hypDataChunkMultFamily(ch, i){
      const dict=Array.isArray(ch.multFamilyDict) ? ch.multFamilyDict : [];
      const codes=hypDataChunkMultFamilyCodes(ch);
      return dict[Number(codes[i]||0)] || 'multiplier';
    }
    function hypDataLowerBound(arr, x){ let lo=0, hi=arr.length; while(lo<hi){ const mid=(lo+hi)>>1; if(arr[mid]<x) lo=mid+1; else hi=mid; } return lo; }
    function hypDataRelTol(settings, stage){
      const sig=typedInputPrecision(settings);
      const slack = stage<=1 ? 130 : (stage===2 ? 170 : 240);
      return Math.max(RIES_HYPDATA_MIN_REL_TOL, typedRelativeToleranceNumber(sig, slack, 0, 15));
    }
    function hypDataStageBudgetMs(settings, stage){
      // Level 4/5/6 are designed as progressive web-facing tiers.  The budget is
      // intentionally about search time after the required chunk(s) have loaded.
      const opt=settings?.stageBudgets || {};
      if(stage<=1) return stageBudgetValueToMs(opt.hypData1Ms, 1000, 0, 120000);
      if(stage===2) return stageBudgetValueToMs(opt.hypData2Ms, 5000, 0, 120000);
      return stageBudgetValueToMs(opt.hypData3Ms, 50000, 0, 300000);
    }
    function hypDataTargetComplex(settings){
      if(settings?.complexTarget && settings.parsedComplex){
        return {re:rationalToNumber(settings.parsedComplex.re), im:rationalToNumber(settings.parsedComplex.im)};
      }
      return {re:Number(settings?.target), im:0};
    }
    function hypDataSourceText(code){
      code=Number(code||0);
      if(code===3) return 'data1 + data2';
      if(code===2) return 'data2';
      if(code===1) return 'data1';
      return 'merged source';
    }
    function hypDataMkParts(mk){
      const p=String(mk||'').split('|');
      return {pref:p[1]||'0', upper:(p[2]||'').split(',').filter(Boolean), lower:(p[3]||'').split(',').filter(Boolean), z:p[4]||''};
    }
    function hypDataParamText(a){ return a.length ? a.join(',') : '-'; }
    function hypDataParamLatex(a){ return a.length ? a.map(x=>sanitizeLatexForDisplay(hardDbRatLatex(x))).join(', ') : ''; }
    function hypDataMkText(mk){
      const p=hypDataMkParts(mk), pf=`_${p.upper.length}F${p.lower.length}`;
      const pref = p.pref && p.pref!=='0' ? `pref=${p.pref}; ` : '';
      return `${pf}(${hypDataParamText(p.upper)}; ${hypDataParamText(p.lower)}; ${p.z})${pref?` [${pref.slice(0,-2)}]`:''}`;
    }
    function hypDataMkLatex(mk){
      const p=hypDataMkParts(mk);
      const zLatex=sanitizeLatexForDisplay(hardDbRatLatex(p.z));
      const core=`{}_{${p.upper.length}}F_{${p.lower.length}}\\!\\left(\\begin{matrix}${hypDataParamLatex(p.upper)}\\\\${hypDataParamLatex(p.lower)}\\end{matrix};${zLatex}\\right)`;
      if(!p.pref || p.pref==='0' || p.pref==='1') return sanitizeLatexForDisplay(core);
      return sanitizeLatexForDisplay(latexMulScalar(hardDbRatLatex(p.pref), core));
    }
    function hypDataMulText(m,h){ return (!m || m==='1') ? h : (m==='-1' ? `-${h}` : `${m}·${h}`); }
    function hypDataMulLatex(m,h){ return sanitizeLatexForDisplay(latexMulScalar(m || '1', h)); }
    function hypDataValue20(ch, rowIndex){
      const line=hypDataChunkValue20Lines(ch)[rowIndex] || '0.00000000000000000000|0.00000000000000000000';
      const parts=line.split('|');
      return {re:parts[0]||'0.00000000000000000000', im:parts[1]||'0.00000000000000000000'};
    }
    function hypDataCandidateInsert(best, cand, maxKeep=100){
      const key=`${cand.chunkStage}|${cand.rowIndex}|${cand.multStage}|${cand.multIndex}`;
      const old=best.get(key);
      if(!old || cand.score<old.score || (cand.score===old.score && cand.errAbs<old.errAbs)) best.set(key,cand);
      if(best.size>maxKeep*3){
        const arr=[...best.values()].sort((a,b)=>a.score-b.score || a.errAbs-b.errAbs).slice(0,maxKeep);
        best.clear(); for(const x of arr) best.set(`${x.chunkStage}|${x.rowIndex}|${x.multStage}|${x.multIndex}`,x);
      }
    }
    function hypDataRowCountForStage(stage){ return hypDataLoadedChunks(stage).reduce((a,ch)=>a+Number(ch.rows||0),0); }
    function hypDataMultiplierChunks(settings, stage){
      const opt=settings?.hypDataOptions || {multSimple:true,multGamma:true,multDeep:true};
      return hypDataLoadedChunks(stage).filter(ch=>{
        if(!Number(ch.multiplierRows||0)) return false;
        const st=Number(ch.stage||1);
        if(st<=1) return opt.multSimple!==false;
        if(st===2) return opt.multGamma!==false;
        return opt.multDeep!==false;
      });
    }
    function hypDataMultiplierCountForStage(stage, settings=null){ return hypDataMultiplierChunks(settings, stage).reduce((a,ch)=>a+Number(ch.multiplierRows||0),0); }
    function hypDataSearch(settings, progressCb=null){
      const stage=hypDataMaxStage(settings);
      if(stage<1 || !isHypDataReady(stage)) return [];
      const relTol=hypDataRelTol(settings, stage);
      const deadline=performance.now()+hypDataStageBudgetMs(settings, stage);
      const target=hypDataTargetComplex(settings);
      const isComplex=!!(settings?.complexTarget || Math.abs(target.im)>0);
      const chunks=hypDataLoadedChunks(stage);
      const multChunks=hypDataMultiplierChunks(settings, stage);
      const hCount=Math.max(1,hypDataRowCountForStage(stage));
      const mCount=Math.max(1,hypDataMultiplierCountForStage(stage, settings));
      const volumePenalty=Math.log10(Math.max(10, hCount*mCount));
      const best=new Map();
      let doneMult=0;
      if(!isComplex){
        const x=Number(target.re); if(!Number.isFinite(x) || x===0) return [];
        outer: for(const mch of multChunks){
          const mVals=hypDataChunkMultValues(mch), mComp=hypDataChunkMultComplexity(mch);
          for(let mi=0; mi<mVals.length; mi++,doneMult++){
            const m=mVals[mi]; if(!(m!==0 && Number.isFinite(m))) continue;
            const y=x/m;
            const eps=Math.max(1, Math.abs(y))*relTol;
            for(const hch of chunks){
              const values=hypDataChunkRealValues(hch), rows=hypDataChunkRealRows(hch), hComp=hypDataChunkComplexity(hch);
              let pos=hypDataLowerBound(values, y-eps)-2; if(pos<0) pos=0;
              const end=Math.min(values.length, hypDataLowerBound(values, y+eps)+3);
              for(let k=pos;k<end;k++){
                const rowIndex=rows[k];
                const h=values[k]; const pred=m*h;
                const errAbs=Math.abs(pred-x); const rel=errAbs/Math.max(1,Math.abs(x));
                if(!Number.isFinite(rel) || rel>relTol*1.35) continue;
                const matched=-Math.log10(Math.max(rel,1e-300));
                const effStage=Math.max(Number(hch.stage||1), Number(mch.stage||1));
                const complexity=Number(mComp[mi]||0)+Number(hComp[rowIndex]||0)/10;
                const score=(effStage-1)*120 + complexity*5 + volumePenalty*25 - matched*80;
                hypDataCandidateInsert(best,{chunk:hch, chunkStage:Number(hch.stage||1), rowIndex, multChunk:mch, multStage:Number(mch.stage||1), multIndex:mi, hRe:h, hIm:0, predRe:pred, predIm:0, errAbs, rel, matched, complexity, score, stage:effStage, volumePenalty});
              }
            }
            if((doneMult&255)===0 && progressCb){
              progressCb({phase:`level ${RIES_HYPDATA_ASSET_LEVELS[stage-1].level} cumulative real multiplier scan`, done:doneMult, total:mCount, rows:[...best.values()].sort((a,b)=>a.score-b.score).slice(0,hypDataLimit(settings))});
              if(performance.now()>deadline && best.size>=hypDataLimit(settings)) break outer;
            }
          }
        }
      }else{
        const xre=Number(target.re), xim=Number(target.im);
        if(!Number.isFinite(xre) || !Number.isFinite(xim) || (xre===0 && xim===0)) return [];
        const xAbs=Math.max(1, Math.hypot(xre,xim));
        outerC: for(const mch of multChunks){
          const mVals=hypDataChunkMultValues(mch), mComp=hypDataChunkMultComplexity(mch);
          for(let mi=0; mi<mVals.length; mi++,doneMult++){
            const m=mVals[mi]; if(!(m!==0 && Number.isFinite(m))) continue;
            const yre=xre/m, yim=xim/m;
            const eps=Math.max(1, Math.hypot(yre,yim))*relTol;
            for(const hch of chunks){
              const reArr=hypDataChunkComplexRe(hch), imArr=hypDataChunkComplexIm(hch), rowArr=hypDataChunkComplexRows(hch), hComp=hypDataChunkComplexity(hch);
              let pos=hypDataLowerBound(reArr, yre-eps)-2; if(pos<0) pos=0;
              const end=Math.min(reArr.length, hypDataLowerBound(reArr, yre+eps)+3);
              for(let k=pos;k<end;k++){
                const rowIndex=rowArr[k], hre=reArr[k], him=imArr[k];
                if(Math.abs(him-yim)>eps) continue;
                const predRe=m*hre, predIm=m*him;
                const errAbs=Math.hypot(predRe-xre,predIm-xim); const rel=errAbs/xAbs;
                if(!Number.isFinite(rel) || rel>relTol*1.35) continue;
                const matched=-Math.log10(Math.max(rel,1e-300));
                const effStage=Math.max(Number(hch.stage||1), Number(mch.stage||1));
                const complexity=Number(mComp[mi]||0)+Number(hComp[rowIndex]||0)/10+6;
                const score=(effStage-1)*120 + complexity*5 + volumePenalty*25 - matched*80;
                hypDataCandidateInsert(best,{chunk:hch, chunkStage:Number(hch.stage||1), rowIndex, multChunk:mch, multStage:Number(mch.stage||1), multIndex:mi, hRe:hre, hIm:him, predRe, predIm, errAbs, rel, matched, complexity, score, stage:effStage, volumePenalty});
              }
            }
            if((doneMult&255)===0 && progressCb){
              progressCb({phase:`level ${RIES_HYPDATA_ASSET_LEVELS[stage-1].level} cumulative complex multiplier scan`, done:doneMult, total:mCount, rows:[...best.values()].sort((a,b)=>a.score-b.score).slice(0,hypDataLimit(settings))});
              if(performance.now()>deadline && best.size>=hypDataLimit(settings)) break outerC;
            }
          }
        }
      }
      return [...best.values()].sort((a,b)=>a.score-b.score || a.errAbs-b.errAbs).slice(0,hypDataLimit(settings));
    }
    function hypDataRowsFromHits(hits, settings, t0){
      return hits.map(h=>{
        const hch=h.chunk, mch=h.multChunk;
        const mk=hypDataChunkMkLines(hch)[h.rowIndex] || '';
        const hText=hypDataMkText(mk), hLatex=hypDataMkLatex(mk);
        const mt=hypDataChunkMultTextLines(mch), ml=hypDataChunkMultLatexLines(mch);
        const mText=mt[h.multIndex] || '1', mLatex=sanitizeLatexForDisplay(ml[h.multIndex] || '1'), family=hypDataChunkMultFamily(mch,h.multIndex);
        const formulaText=hypDataMulText(mText,hText), formulaLatex=hypDataMulLatex(mLatex,hLatex);
        const val20=hypDataValue20(hch,h.rowIndex);
        const rowKind=(Math.abs(Number(val20.im)||0)<=5e-21) ? 'real' : 'complex';
        const p=Number(hypDataChunkP(hch)[h.rowIndex]||0), q=Number(hypDataChunkQ(hch)[h.rowIndex]||0);
        const globalRow=Number(hch.rowOffset||0)+Number(h.rowIndex||0);
        const predicted = Math.abs(h.predIm||0)>1e-15 ? `${fmtValue(h.predRe)} ${h.predIm<0?'−':'+'} ${fmtValue(Math.abs(h.predIm))}i` : fmtValue(h.predRe);
        const stageLabel=hypDataStageLabel(h.stage);
        const desc=`merged row ${globalRow+1} of ${RIES_HYPDATA_TOTAL_ROWS}; ${p}F${q}; ${hypDataSourceText(hypDataChunkSource(hch)[h.rowIndex])}; ${family}; H chunk level ${hch.level}; multiplier chunk level ${mch.level}; ${stageLabel}`;
        const explain=`<div class="harddb-explain"><div><b>Definitions.</b> <code>H</code> is the stored hypergeometric value from the merged pFq database; the result tests <code>x ≈ M·H</code>.</div><div><b>Database value.</b> H ≈ ${escapeHtml(val20.re)}${rowKind==='complex' ? ' + '+escapeHtml(val20.im)+'i' : ''} <span class="muted">(20 decimal places stored for display; Float64 mirror used for matching)</span>.</div><div><b>Acceptance.</b> matched digits ≈ ${h.matched.toFixed(2)}; search-volume penalty ≈ ${h.volumePenalty.toFixed(2)}; relative error ≈ ${h.rel.toExponential(2)}.</div></div>`;
        const valueHtml=`<div><b>H = <span class="latex-render">\\(${escapeHtml(hLatex)}\\)</span></b></div><div class="muted">${escapeHtml(desc)}</div><div>Multiplier M = <span class="latex-render">\\(${escapeHtml(mLatex)}\\)</span></div><div>predicted x ≈ ${escapeHtml(predicted)}; module ${Math.round(performance.now()-t0)} ms</div>${explain}`;
        return {
          candidate:`hypergeometric database: x ≈ ${formulaText}`,
          latex:`x \\approx ${formulaLatex}`,
          copyLatex:`x \\approx ${formulaLatex}`,
          valueHtml,
          copyValue:`H≈${val20.re}${rowKind==='complex'?'+('+val20.im+')i':''}; ${desc}`,
          err:h.errAbs,
          errText:fmtErr(h.errAbs),
          hypDataCategory:stageLabel,
          constantDbCategory:'hypergeometric pFq database',
          constantDbSource:'merged-hypdata-v11.6',
          constantDbId:`hyp_${String(globalRow+1).padStart(6,'0')}`,
          terms: 2 + Math.max(0, String(mText).split('·').length-1),
          height: BigInt(Math.max(1, Math.round(h.complexity||1))),
          score:h.score
        };
      });
    }
    async function hypDataRowsAsync(settings, progressCb=null){
      if(!hypDataPotentiallyRunnable(settings)) return [];
      const stage=hypDataMaxStage(settings);
      if(!isHypDataReady(stage)){
        const loaded=await ensureHypDataLoaded({settings, stage, label:'hypergeometric pFq database', phase:'hypergeometric database', baseProgress:.49, spanProgress:.05});
        if(!loaded) return [];
      }
      const t0=performance.now();
      if(progressCb) progressCb({phase:`decoding cumulative pFq chunks through level ${RIES_HYPDATA_ASSET_LEVELS[stage-1].level}`, done:0, total:1, rows:[]});
      try{
        for(const ch of hypDataLoadedChunks(stage)){ hypDataChunkRealValues(ch); hypDataChunkMultValues(ch); }
      }catch(e){ console.warn(e); return []; }
      const hits=hypDataSearch(settings, progressCb);
      if(!hits.length) return [];
      if(progressCb) progressCb({phase:'formatting hypergeometric pFq hits', done:1, total:1, rows:hits});
      return hypDataRowsFromHits(hits, settings, t0);
    }


    // v11.7 lazy integral/sum candidate database matcher.
    // The database is independent from harddb and hypdata.  Level 4 loads the
    // low-height row subset (about one fifth of the data); level 5 loads the
    // remaining rows so the cumulative S-table is complete; level 6 keeps the
    // same full row table but enables deeper multiplier constants.
    const RIES_INTSUMDB_LIMIT = 5;
    const RIES_INTSUMDB_MIN_REL_TOL = 1e-12;
    const RIES_INTSUMDB_TOTAL_ROWS = 36685;
    const RIES_INTSUMDB_ASSET_LEVELS = [
      {stage:1, level:4, url:'assets/ries-intsumdb-v11_7-level4.js?v=11.7.2', label:'integral/sum level 4 simple low-height chunk', expectedBytes:2309645},
      {stage:2, level:5, url:'assets/ries-intsumdb-v11_7-level5.js?v=11.7.2', label:'integral/sum level 5 full-data chunk', expectedBytes:10766900},
      {stage:3, level:6, url:'assets/ries-intsumdb-v11_7-level6.js?v=11.7.2', label:'integral/sum level 6 deep multiplier chunk', expectedBytes:608585}
    ];
    function intsumDbLimit(settings){
      return Math.max(1, Math.min(50, Number(settings?.moduleLimits?.intsumDb || RIES_INTSUMDB_LIMIT) || RIES_INTSUMDB_LIMIT));
    }
    function intsumDbChunksRaw(){
      return (typeof window!=='undefined' && Array.isArray(window.RIES_INTSUMDB_V117_CHUNKS)) ? window.RIES_INTSUMDB_V117_CHUNKS : [];
    }
    function intsumDbMaxStage(settings){
      const lvl=Math.max(1, Math.floor(Number(settings?.level || document.getElementById('level')?.value || DEFAULT_RIES_LEVEL) || DEFAULT_RIES_LEVEL));
      if(lvl<4) return 0;
      const base = lvl===4 ? 1 : (lvl===5 ? 2 : 3);
      const opt=settings?.intsumDbOptions || {depth1:true,depth2:true,depth3:true};
      let max=0;
      if(base>=1 && opt.depth1!==false) max=1; else return 0;
      if(base>=2 && opt.depth2!==false) max=2; else return max;
      if(base>=3 && opt.depth3!==false) max=3;
      return max;
    }
    function intsumDbStageLabel(stage){
      return stage<=1 ? 'level 4 simple low-height integral/sum layer' : (stage===2 ? 'level 5 full integral/sum data + Gamma layer' : 'level 6 full data + deep multiplier layer');
    }
    function isIntsumDbReady(stage=1){
      const chunks=intsumDbChunksRaw();
      const upto=Math.max(1, Math.min(3, Number(stage||1)));
      for(let i=0;i<upto;i++) if(!chunks[i]) return false;
      return true;
    }
    function intsumDbLoadedChunks(stage=3){
      const chunks=intsumDbChunksRaw();
      const upto=Math.max(1, Math.min(3, Number(stage||3)));
      const out=[];
      for(let i=0;i<upto;i++) if(chunks[i]) out.push(chunks[i]);
      return out;
    }
    function intsumDbLoadedRowChunks(stage=3){ return intsumDbLoadedChunks(stage).filter(ch=>Number(ch.rows||0)>0); }
    function intsumDbPotentiallyRunnable(settings){
      return !!settings && settings.modules?.intsumDb!==false && intsumDbMaxStage(settings)>=1 && Number.isFinite(settings.target) && !settings.complexTarget && settings.target!==0;
    }
    async function ensureIntsumDbLoaded(opts={}){
      const stage=Math.max(1, Math.min(3, Number(opts.stage || intsumDbMaxStage(opts.settings||{}) || 1)));
      const base=Number.isFinite(opts.baseProgress) ? opts.baseProgress : .585;
      const span=Number.isFinite(opts.spanProgress) ? opts.spanProgress : .045;
      for(let i=0;i<stage;i++){
        const spec=RIES_INTSUMDB_ASSET_LEVELS[i];
        const loaded=await loadScriptPackageWithProgress(spec.url, ()=>isIntsumDbReady(spec.stage), {
          label: opts.label ? `${opts.label} ${spec.level}` : spec.label,
          phase: opts.phase || 'integral/sum database',
          baseProgress: base + span*(i/stage),
          spanProgress: span/stage,
          expectedBytes: spec.expectedBytes
        });
        if(!loaded) return false;
      }
      return isIntsumDbReady(stage);
    }
    function intsumDbFloat64(b64){ const bytes=hardDbB64ToBytes(b64||''); return new Float64Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength/8)); }
    function intsumDbFloat32(b64){ const bytes=hardDbB64ToBytes(b64||''); return new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength/4)); }
    function intsumDbUint32(b64){ const bytes=hardDbB64ToBytes(b64||''); return new Uint32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength/4)); }
    function intsumDbUint16(b64){ const bytes=hardDbB64ToBytes(b64||''); return new Uint16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength/2)); }
    function intsumDbUint8(b64){ return hardDbB64ToBytes(b64||''); }
    function intsumDbLines(ch, prop){ const key='_'+prop+'Lines'; if(!ch[key]) ch[key]=String(ch[prop]||'').split('\n'); return ch[key]; }
    function intsumDbChunkValues(ch){ if(!ch._values) ch._values=intsumDbFloat64(ch.valuesB64); return ch._values; }
    function intsumDbChunkRows(ch){ if(!ch._rows) ch._rows=intsumDbUint32(ch.rowB64); return ch._rows; }
    function intsumDbChunkComplexity(ch){ if(!ch._complexity) ch._complexity=intsumDbUint16(ch.complexityB64); return ch._complexity; }
    function intsumDbChunkVerifiedDigits(ch){ if(!ch._verifiedDigits) ch._verifiedDigits=intsumDbUint16(ch.verifiedDigitsB64); return ch._verifiedDigits; }
    function intsumDbChunkStatusCodes(ch){ if(!ch._statusCodes) ch._statusCodes=intsumDbUint8(ch.statusB64); return ch._statusCodes; }
    function intsumDbChunkDatasetCodes(ch){ if(!ch._datasetCodes) ch._datasetCodes=intsumDbUint8(ch.datasetB64); return ch._datasetCodes; }
    function intsumDbChunkMultValues(ch){ if(!ch._multValues) ch._multValues=intsumDbFloat64(ch.multValuesB64); return ch._multValues; }
    function intsumDbChunkMultComplexity(ch){ if(!ch._multComplexity) ch._multComplexity=intsumDbFloat32(ch.multComplexityB64); return ch._multComplexity; }
    function intsumDbChunkMultFamilyCodes(ch){ if(!ch._multFamilyCodes) ch._multFamilyCodes=intsumDbUint8(ch.multFamilyB64); return ch._multFamilyCodes; }
    function intsumDbChunkMultFamily(ch, i){
      const dict=Array.isArray(ch.multFamilyDict) ? ch.multFamilyDict : [];
      const codes=intsumDbChunkMultFamilyCodes(ch);
      return dict[Number(codes[i]||0)] || 'multiplier';
    }
    function intsumDbRowStatus(ch, i){
      const dict=Array.isArray(ch.statusDict) ? ch.statusDict : [];
      const codes=intsumDbChunkStatusCodes(ch);
      return dict[Number(codes[i]||0)] || 'verified';
    }
    function intsumDbRowDataset(ch, i){
      const dict=Array.isArray(ch.datasetDict) ? ch.datasetDict : [];
      const codes=intsumDbChunkDatasetCodes(ch);
      return dict[Number(codes[i]||0)] || '';
    }
    function intsumDbLowerBound(arr, x){ let lo=0, hi=arr.length; while(lo<hi){ const mid=(lo+hi)>>1; if(arr[mid]<x) lo=mid+1; else hi=mid; } return lo; }
    function intsumDbRelTol(settings, stage){
      const sig=typedInputPrecision(settings);
      const slack = stage<=1 ? 130 : (stage===2 ? 170 : 240);
      return Math.max(RIES_INTSUMDB_MIN_REL_TOL, typedRelativeToleranceNumber(sig, slack, 0, 15));
    }
    function intsumDbStageBudgetMs(settings, stage){
      const opt=settings?.stageBudgets || {};
      if(stage<=1) return stageBudgetValueToMs(opt.intsumDb1Ms, 1000, 0, 120000);
      if(stage===2) return stageBudgetValueToMs(opt.intsumDb2Ms, 5000, 0, 120000);
      return stageBudgetValueToMs(opt.intsumDb3Ms, 50000, 0, 300000);
    }
    function intsumDbMultiplierChunks(settings, stage){
      const opt=settings?.intsumDbOptions || {multSimple:true,multGamma:true,multDeep:true};
      return intsumDbLoadedChunks(stage).filter(ch=>{
        if(!Number(ch.multiplierRows||0)) return false;
        const st=Number(ch.stage||1);
        if(st<=1) return opt.multSimple!==false;
        if(st===2) return opt.multGamma!==false;
        return opt.multDeep!==false;
      });
    }
    function intsumDbRowCountForStage(stage){ return intsumDbLoadedRowChunks(stage).reduce((a,ch)=>a+Number(ch.rows||0),0); }
    function intsumDbMultiplierCountForStage(stage, settings=null){ return intsumDbMultiplierChunks(settings, stage).reduce((a,ch)=>a+Number(ch.multiplierRows||0),0); }
    function intsumDbCandidateInsert(best, cand, maxKeep=100){
      const key=`${cand.chunkStage}|${cand.rowIndex}|${cand.multStage}|${cand.multIndex}`;
      const old=best.get(key);
      if(!old || cand.score<old.score || (cand.score===old.score && cand.errAbs<old.errAbs)) best.set(key,cand);
      if(best.size>maxKeep*3){
        const arr=[...best.values()].sort((a,b)=>a.score-b.score || a.errAbs-b.errAbs).slice(0,maxKeep);
        best.clear(); for(const x of arr) best.set(`${x.chunkStage}|${x.rowIndex}|${x.multStage}|${x.multIndex}`,x);
      }
    }
    function intsumDbSearch(settings, progressCb=null){
      const stage=intsumDbMaxStage(settings);
      if(stage<1 || !isIntsumDbReady(stage)) return [];
      const relTol=intsumDbRelTol(settings, stage);
      const deadline=performance.now()+intsumDbStageBudgetMs(settings, stage);
      const x=Number(settings?.target);
      if(!Number.isFinite(x) || x===0) return [];
      const chunks=intsumDbLoadedRowChunks(stage);
      const multChunks=intsumDbMultiplierChunks(settings, stage);
      const sCount=Math.max(1,intsumDbRowCountForStage(stage));
      const mCount=Math.max(1,intsumDbMultiplierCountForStage(stage, settings));
      const volumePenalty=Math.log10(Math.max(10, sCount*mCount));
      const best=new Map();
      let doneMult=0;
      outer: for(const mch of multChunks){
        const mVals=intsumDbChunkMultValues(mch), mComp=intsumDbChunkMultComplexity(mch);
        for(let mi=0; mi<mVals.length; mi++,doneMult++){
          const m=mVals[mi]; if(!(m!==0 && Number.isFinite(m))) continue;
          const y=x/m;
          const eps=Math.max(1, Math.abs(y))*relTol;
          for(const sch of chunks){
            const values=intsumDbChunkValues(sch), rows=intsumDbChunkRows(sch), sComp=intsumDbChunkComplexity(sch);
            let pos=intsumDbLowerBound(values, y-eps)-2; if(pos<0) pos=0;
            const end=Math.min(values.length, intsumDbLowerBound(values, y+eps)+3);
            for(let k=pos;k<end;k++){
              const rowIndex=rows[k];
              const sval=values[k]; const pred=m*sval;
              const errAbs=Math.abs(pred-x); const rel=errAbs/Math.max(1,Math.abs(x));
              if(!Number.isFinite(rel) || rel>relTol*1.35) continue;
              const matched=-Math.log10(Math.max(rel,1e-300));
              const effStage=Math.max(Number(sch.stage||1), Number(mch.stage||1));
              const complexity=Number(mComp[mi]||0)+Number(sComp[rowIndex]||0)/3;
              const score=(effStage-1)*115 + complexity*5.5 + volumePenalty*24 - matched*80;
              intsumDbCandidateInsert(best,{chunk:sch, chunkStage:Number(sch.stage||1), rowIndex, multChunk:mch, multStage:Number(mch.stage||1), multIndex:mi, sValue:sval, pred, errAbs, rel, matched, complexity, score, stage:effStage, volumePenalty}, Math.max(100, intsumDbLimit(settings)*20));
            }
          }
          if((doneMult&255)===0 && progressCb){
            progressCb({phase:`level ${RIES_INTSUMDB_ASSET_LEVELS[stage-1].level} cumulative real multiplier scan`, done:doneMult, total:mCount, rows:[...best.values()].sort((a,b)=>a.score-b.score).slice(0,intsumDbLimit(settings))});
            if(performance.now()>deadline && best.size>=intsumDbLimit(settings)) break outer;
          }
        }
      }
      return [...best.values()].sort((a,b)=>a.score-b.score || a.errAbs-b.errAbs).slice(0,intsumDbLimit(settings));
    }
    function intsumDbMulText(m,s){ return (!m || m==='1') ? s : (m==='-1' ? `-${s}` : `${m}·${s}`); }
    function intsumDbMulLatex(m,s){ return sanitizeLatexForDisplay(latexMulScalar(m || '1', s)); }
    function intsumDbFamilyDescription(family, sub){
      const f=String(family||''), s=String(sub||'');
      if(f==='FINITE_EXP_POLY') return 'finite-interval exponential-polynomial integral family';
      if(f==='RATIONAL_LOG_BETA') return 'rational beta/log integral family';
      if(f==='BETA_LOG_PLUS_MINUS') return 'beta-type logarithmic integral with plus/minus algebraic factor';
      if(f==='LAPLACE_GAUSS_LOG') return 'Laplace/Gaussian logarithmic integral family';
      if(f==='RATIONAL_TAIL_SUM') return 'rational tail infinite-sum family';
      if(f==='HYPERGEOM_EULER_INTEGRAL') return 'Euler-type hypergeometric integral family';
      if(f==='TRIG_BETA_LOG') return 'trigonometric beta/log integral family';
      if(f==='TRIG_RATIONAL_FOURIER') return 'trigonometric rational Fourier sum/integral family';
      if(f==='BINOMIAL_INVBINOM_SUM') return 'inverse-binomial or central-binomial infinite-sum family';
      if(f==='POLYLOG_LERCH_SUM') return 'polylogarithm/Lerch-type sum family';
      return [f,s].filter(Boolean).join(' / ') || 'integral/sum candidate family';
    }
    function intsumDbRowsFromHits(hits, settings, t0){
      return hits.map(h=>{
        const sch=h.chunk, mch=h.multChunk, i=h.rowIndex;
        const ids=intsumDbLines(sch,'idBlob'), plains=intsumDbLines(sch,'plainBlob'), latexes=intsumDbLines(sch,'latexBlob'), values=intsumDbLines(sch,'valueBlob'), values16=intsumDbLines(sch,'value16Blob'), families=intsumDbLines(sch,'familyBlob'), subs=intsumDbLines(sch,'subBlob');
        const mt=intsumDbLines(mch,'multTextBlob'), ml=intsumDbLines(mch,'multLatexBlob');
        const sText=plains[i] || ids[i] || 'integral/sum candidate';
        const sLatex=sanitizeLatexForDisplay(latexes[i] || sText);
        const mText=mt[h.multIndex] || '1', mLatex=sanitizeLatexForDisplay(ml[h.multIndex] || '1'), family=intsumDbChunkMultFamily(mch,h.multIndex);
        const formulaText=intsumDbMulText(mText,sText), formulaLatex=intsumDbMulLatex(mLatex,sLatex);
        const rowFamily=families[i] || 'INTSUM', rowSub=subs[i] || '';
        const id=ids[i] || `intsum_${String(i+1).padStart(6,'0')}`;
        const valueText=values[i] || values16[i] || h.sValue.toPrecision(20);
        const verifiedDigits=Number(intsumDbChunkVerifiedDigits(sch)[i]||0);
        const rowStatus=intsumDbRowStatus(sch,i), rowDataset=intsumDbRowDataset(sch,i);
        const predicted=fmtValue(h.pred);
        const stageLabel=intsumDbStageLabel(h.stage);
        const desc=`${rowFamily}${rowSub?'/'+rowSub:''}; ${intsumDbFamilyDescription(rowFamily,rowSub)}; row ${id}; source ${rowDataset || 'candidate package'}; status ${rowStatus}; S chunk level ${sch.level}; multiplier chunk level ${mch.level}; ${family}; ${stageLabel}`;
        const explain=`<div class="harddb-explain"><div><b>Definitions.</b> <code>S</code> is the stored integral/sum value; the result tests <code>x ≈ M·S</code> with a stage-gated constant multiplier <code>M</code>.</div><div><b>Database value.</b> S ≈ ${escapeHtml(valueText)} <span class="muted">(high-precision text kept for display; Float64 mirror used for matching)</span>.</div><div><b>Acceptance.</b> matched digits ≈ ${h.matched.toFixed(2)}; search-volume penalty ≈ ${h.volumePenalty.toFixed(2)}; relative error ≈ ${h.rel.toExponential(2)}; verified digits ${verifiedDigits || 'n/a'}.</div></div>`;
        const valueHtml=`<div><b>S = <span class="latex-render">\\(${escapeHtml(sLatex)}\\)</span></b></div><div class="muted">${escapeHtml(desc)}</div><div>Multiplier M = <span class="latex-render">\\(${escapeHtml(mLatex)}\\)</span></div><div>predicted x ≈ ${escapeHtml(predicted)}; module ${Math.round(performance.now()-t0)} ms</div>${explain}`;
        return {
          candidate:`integral/sum database: x ≈ ${formulaText}`,
          latex:`x \\approx ${formulaLatex}`,
          copyLatex:`x \\approx ${formulaLatex}`,
          valueHtml,
          copyValue:`S≈${valueText}; ${desc}`,
          err:h.errAbs,
          errText:fmtErr(h.errAbs),
          intsumDbCategory:stageLabel,
          constantDbCategory:'integral/sum candidate database',
          constantDbSource:'intsumdb-v11.7.2',
          constantDbId:id,
          terms:2 + Math.max(0, String(mText).split('·').length-1),
          height: BigInt(Math.max(1, Math.round(h.complexity||1))),
          score:h.score
        };
      });
    }
    async function intsumDbRowsAsync(settings, progressCb=null){
      if(!intsumDbPotentiallyRunnable(settings)) return [];
      const stage=intsumDbMaxStage(settings);
      if(!isIntsumDbReady(stage)){
        const loaded=await ensureIntsumDbLoaded({settings, stage, label:'integral/sum database', phase:'integral/sum database', baseProgress:.585, spanProgress:.045});
        if(!loaded) return [];
      }
      const t0=performance.now();
      if(progressCb) progressCb({phase:`decoding cumulative integral/sum chunks through level ${RIES_INTSUMDB_ASSET_LEVELS[stage-1].level}`, done:0, total:1, rows:[]});
      try{
        for(const ch of intsumDbLoadedChunks(stage)){ if(Number(ch.rows||0)>0) intsumDbChunkValues(ch); if(Number(ch.multiplierRows||0)>0) intsumDbChunkMultValues(ch); }
      }catch(e){ console.warn(e); return []; }
      const hits=intsumDbSearch(settings, progressCb);
      if(!hits.length) return [];
      if(progressCb) progressCb({phase:'formatting integral/sum database hits', done:1, total:1, rows:hits});
      return intsumDbRowsFromHits(hits, settings, t0);
    }


    // v8.2 L-function decimal matcher.  This browser-native implementation keeps
    // the v7.3 alltest-style rational/log/quadratic comparisons, but runs them
    // incrementally with per-input caches so Continue never repeats completed
    // L-function work.  Formulas deliberately write the modular form as f; the
    // separate result columns show N.k.# and the q-expansion.
    const RIES_LFUNC_PI = '3.14159265358979323846264338327950288419716939937510582097494459230781640628620899';
    let lfuncEntryCache = null;
    let lfuncQuadraticCatalogCache = null;
    const lfuncLogCatalogCache = new Map();
    const lfuncProgressCache = new Map();
    function lfuncDataAvailable(){ return Array.isArray(window.RIES_LFUNCTIONS_L2) && Array.isArray(window.RIES_LFUNCTIONS_L4); }
    function lfuncEntries(){
      if(lfuncEntryCache) return lfuncEntryCache;
      const rows=[];
      function addEntry(row, weight, which, valueIndex){
        if(!Array.isArray(row)) return;
        const hasCoeffs=Array.isArray(row[2]);
        const n=row[0], idx=row[1], coeffs=hasCoeffs ? row[2].slice() : [];
        const value=String(row[valueIndex]);
        const num=Number(value);
        if(!Number.isFinite(num) || Math.abs(num)<=1e-36) return;
        rows.push({family:`f${weight}`, weight, which:String(which), n, index:idx, coeffs, value,
          label:`L(f,${which})`, shortId:`${n}.${weight}.${idx}`, entryKey:`${n}.${weight}.${idx}:L${which}`});
      }
      if(Array.isArray(window.RIES_LFUNCTIONS_L2)){
        for(const r of window.RIES_LFUNCTIONS_L2){ addEntry(r, 2, 1, Array.isArray(r[2]) ? 3 : 2); }
      }
      if(Array.isArray(window.RIES_LFUNCTIONS_L4)){
        for(const r of window.RIES_LFUNCTIONS_L4){
          const off=Array.isArray(r[2]) ? 0 : -1;
          addEntry(r, 4, 1, 3+off);
          addEntry(r, 4, 2, 4+off);
        }
      }
      rows.sort((a,b)=>(a.n-b.n)||(a.weight-b.weight)||(a.index-b.index)||Number(a.which)-Number(b.which));
      lfuncEntryCache = rows;
      return lfuncEntryCache;
    }
    function lfuncShouldRun(settings){
      return lfuncDataAvailable() && settings && settings.modules?.lfunc!==false && settings.parsedComplex && !settings.complexTarget && !rationalIsZero(settings.parsedComplex.re) && Number.isFinite(settings.target) && /[.eE]/.test(String(settings.raw||''));
    }
    function lfuncCacheKey(settings){
      const q=settings?.parsedComplex?.re;
      if(!q) return '';
      return `${q.num.toString()}/${q.den.toString()}|sig=${typedInputPrecision(settings)}`;
    }
    function lfuncDecimalFromRational(D, q){ return new D(q.num.toString()).div(q.den.toString()); }
    function lfuncDecimalPowInt(D, x, k){
      if(k===0) return new D(1);
      if(k>0) return x.pow(k);
      return new D(1).div(x.pow(-k));
    }
    function lfuncPiPow(D, j){
      if(j===0) return new D(1);
      const pi=new D(RIES_LFUNC_PI);
      return j>0 ? pi.pow(j) : new D(1).div(pi.pow(-j));
    }
    function lfuncTolDigits(sig, mode){ return matchToleranceDigits(sig, mode==='log' ? 1 : 1, 28); }
    function lfuncTolerance(D, sig, mode){ return new D(10).pow(-lfuncTolDigits(sig, mode)); }
    function lfuncDecimalAbsMax1(D, x){ const a=x.abs(); return a.gt(1) ? a : new D(1); }
    function lfuncRationalApproxNumber(x, bound, sig){
      if(!Number.isFinite(x) || x===0) return null;
      const sign=x<0?-1n:1n;
      const orig=Math.abs(x);
      let y=orig;
      let p0=0n,q0=1n,p1=1n,q1=0n;
      const relBase=Math.max(1, orig);
      // This is only a double-precision prefilter; keep a 1e-12 floor so
      // high-precision true positives are not rejected by Number roundoff before
      // the Decimal verifier applies the real typed-precision tolerance.
      const tol=Math.max(1e-12, typedRelativeToleranceNumber(sig, 8, 1, 12));
      for(let depth=0; depth<24; depth++){
        const aNum=Math.floor(y); if(!Number.isFinite(aNum) || aNum>1e9) return null;
        const a=BigInt(aNum);
        const p=a*p1+p0, q=a*q1+q0;
        if(absBig(p)+absBig(q)>BigInt(bound)) break;
        const approx=Number(p)/Number(q);
        if(Math.abs(orig-approx)/relBase <= tol){
          let num=p*sign, den=q; const g=gcdBig(num,den); if(g>1n){ num/=g; den/=g; }
          return {num, den};
        }
        const frac=y-aNum; if(Math.abs(frac)<1e-15) break;
        y=1/frac;
        p0=p1; q0=q1; p1=p; q1=q;
      }
      return null;
    }
    function lfuncVerifyRationalDecimal(D, ratio, num, den, sig){
      const tol=lfuncTolerance(D,sig,'rational');
      const approx=new D(num.toString()).div(den.toString());
      const err=ratio.minus(approx).abs().div(lfuncDecimalAbsMax1(D,ratio));
      return err.lte(tol.mul(5)) ? {num, den, err} : null;
    }
    function lfuncSuperscript(n){ const map={'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'}; return String(n).split('').map(ch=>map[ch]||ch).join(''); }
    function lfuncPiFactor(j){ return j===0 ? '' : (j===1 ? 'π' : `π${lfuncSuperscript(j)}`); }
    function lfuncMul(parts){ return parts.filter(Boolean).join('·') || '1'; }
    function lfuncDiv(numer, denom){ const n=lfuncMul(numer), d=lfuncMul(denom); return d==='1' ? n : `${n}/${d.includes('·') ? '('+d+')' : d}`; }
    function lfuncRationalAbsString(num, den){ return rationalString(absBig(num), absBig(den)); }
    function lfuncFormulaFromScale(scaleExpr, i, j, L, targetNegative){
      const piPos = j>0 ? lfuncPiFactor(j) : '';
      const piNeg = j<0 ? lfuncPiFactor(-j) : '';
      const scaleParts = scaleExpr==='1' ? [] : [scaleExpr];
      if(i===1) return lfuncDiv([...scaleParts,L,piNeg],[piPos]);
      if(i===-1) return lfuncDiv([piPos],[...scaleParts,L,piNeg]);
      const inside = i===2 ? lfuncDiv([...scaleParts,L,piNeg],[piPos]) : lfuncDiv([piPos],[...scaleParts,L,piNeg]);
      return `${targetNegative?'−':''}√(${inside})`;
    }
    function lfuncRationalPart(n){ n=BigInt(n); return n===1n ? '' : n.toString(); }
    function lfuncSignedCore(negative, core){ return negative ? `−${core}` : core; }
    function lfuncSqrtCore(numer, denom, targetNegative){
      const inside=lfuncDiv(numer, denom);
      return `${targetNegative?'−':''}√(${inside})`;
    }
    function lfuncRationalFormula(num, den, i, j, L, targetNegative){
      // Keep rational scale factors mathematically simplified.  In v8.3 the
      // reciprocal cases could display forms such as π/(2/3·L), which are
      // correct but visually noisy.  Build the numerator/denominator directly
      // from the reduced rational so denominators never contain fractions.
      const negative = (num<0n) !== (den<0n);
      let n=absBig(num), d=absBig(den);
      const g=gcdBig(n,d); if(g>1n){ n/=g; d/=g; }
      const nPart=lfuncRationalPart(n), dPart=lfuncRationalPart(d);
      const piPos = j>0 ? lfuncPiFactor(j) : '';
      const piNeg = j<0 ? lfuncPiFactor(-j) : '';
      if(i===1) return lfuncSignedCore(negative, lfuncDiv([nPart,L,piNeg],[dPart,piPos]));
      if(i===-1) return lfuncSignedCore(negative, lfuncDiv([dPart,piPos],[nPart,L,piNeg]));
      if(i===2) return lfuncSqrtCore([nPart,L,piNeg],[dPart,piPos],targetNegative);
      if(i===-2) return lfuncSqrtCore([dPart,piPos],[nPart,L,piNeg],targetNegative);
      return lfuncFormulaFromScale(rationalString(num,den), i, j, L, targetNegative);
    }
    function lfuncTestMonomialString(i,j,L){
      const xpart = i===1 ? 'x' : (i===2 ? 'x²' : (i===-1 ? '1/x' : '1/x²'));
      const ppart = j===0 ? '' : `·${lfuncPiFactor(j)}`;
      return `${xpart}${ppart}/${L}`;
    }
    function lfuncFormatDecimal(D, x, sig=8){ try{ return x.toSignificantDigits(sig).toString(); }catch(e){ return fmtValue(Number(x)); } }
    function lfuncQExpansionLatex(L){
      const coeffs=Array.isArray(L.coeffs) ? L.coeffs : [];
      const terms=[];
      for(let k=1;k<coeffs.length;k++){
        const a=Number(coeffs[k]||0); if(!a) continue;
        const neg=a<0, mag=Math.abs(a);
        const qpow=k===1 ? 'q' : `q^{${k}}`;
        const body=(mag===1 ? '' : String(mag)) + qpow;
        if(!terms.length) terms.push((neg?'-':'')+body);
        else terms.push((neg?' - ':' + ')+body);
      }
      return `f(q)=${terms.join('') || '0'}+O(q^{${Math.max(1, coeffs.length)}})`;
    }
    function lfuncFormulaLatex(formula){
      const supMap={'⁻':'-','⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9'};
      const unsup=t=>String(t).split('').map(ch=>supMap[ch]||ch).join('');
      const cleanExp=exp=>{
        exp=String(exp||'').trim();
        if(/^[-+]?\d+$/.test(exp) && !/^[-+]/.test(exp) && exp.length===1) return '^'+exp;
        return `^{${exp}}`;
      };
      let s=String(formula||'');
      // Convert both Unicode superscripts and plain-text ^(...) powers into
      // simple MathJax exponents: 2^{-2}, 3^2, 5^{5/3}.  This specifically
      // fixes log/L-function products such as 2^(-2)·3^(2)·5^(5/3)/L(f,1).
      s=s.replace(/([A-Za-z0-9π\)\}]+)([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)/g,(m,b,e)=>`${b}${cleanExp(unsup(e))}`);
      s=s.replace(/\^\(([-+]?\d+(?:\/[-+]?\d+)?)\)/g,(m,e)=>cleanExp(e));
      s=s.replace(/\^\{([-+]?\d+(?:\/[-+]?\d+)?)\}/g,(m,e)=>cleanExp(e));
      s=s.replace(/−/g,'-').replace(/π/g,'\\pi').replace(/Γ/g,'\\Gamma').replace(/·/g,'\\,');
      s=s.replace(/√\(([^()]+)\)/g,'\\sqrt{$1}');
      return `x \approx ${sanitizeLatexForDisplay(s)}`;
    }
    function lfuncCandidateRow(kind, rank, formula, l0, detail, err, score){
      const valueText=`${l0.label} ≈ ${l0.value}; ${detail}`;
      return {candidate:`L-${kind} #${rank}: x ≈ ${formula}`, latex:lfuncFormulaLatex(formula), copyLatex:lfuncFormulaLatex(formula), value:valueText, err, score, lfuncCategory:kind,
        lfuncFormula:formula, lfuncEntryKey:l0.entryKey, lfuncLabel:l0.label,
        modForm:{level:l0.n, weight:l0.weight, index:l0.index, code:l0.shortId}, qLatex:lfuncQExpansionLatex(l0), copyValue:valueText};
    }
    function lfuncBuildQuadraticCatalog(bound=40){
      if(lfuncQuadraticCatalogCache && lfuncQuadraticCatalogCache.bound===bound) return lfuncQuadraticCatalogCache.items;
      const items=[], seen=new Set();
      function gcd3(a,b,c){ return Number(gcdBig(gcdBig(BigInt(Math.abs(a)),BigInt(Math.abs(b))),BigInt(Math.abs(c)))); }
      function isSquare(n){ const s=Math.floor(Math.sqrt(n)); return s*s===n || (s+1)*(s+1)===n; }
      for(let a=-bound; a<=bound; a++){
        if(a===0) continue;
        for(let b=-bound; b<=bound; b++){
          for(let c=-bound; c<=bound; c++){
            const g=gcd3(a,b,c); if(g!==1) continue;
            let aa=a, bb=b, cc=c;
            if(aa<0){ aa=-aa; bb=-bb; cc=-cc; }
            const key=`${aa},${bb},${cc}`; if(seen.has(key)) continue; seen.add(key);
            const D=bb*bb-4*aa*cc; if(D<=0 || isSquare(D)) continue;
            const sd=Math.sqrt(D), den=2*aa;
            const r1=(-bb+sd)/den, r2=(-bb-sd)/den;
            if(Number.isFinite(r1) && Math.abs(r1)>1e-14 && Math.abs(r1)<1e8) items.push({v:r1, coeff:[cc,bb,aa], rootSign:1});
            if(Number.isFinite(r2) && Math.abs(r2)>1e-14 && Math.abs(r2)<1e8) items.push({v:r2, coeff:[cc,bb,aa], rootSign:-1});
          }
        }
      }
      items.sort((a,b)=>a.v-b.v);
      lfuncQuadraticCatalogCache={bound, items};
      return items;
    }
    function lfuncQuadraticSearchNumber(rn, sig, bound=40){
      if(!Number.isFinite(rn) || Math.abs(rn)>1e8 || Math.abs(rn)<1e-14) return null;
      // Fast path for the common and important case α² ∈ Q, e.g. √2, √3,
      // √n/q.  This avoids spending the first low-effort slice building the
      // full quadratic catalog before simple surd matches have a chance to show.
      const sqRat=lfuncRationalApproxNumber(rn*rn, Math.max(2000, bound*bound*8), sig);
      if(sqRat && sqRat.num>0n && sqRat.den>0n){
        let c0=-sqRat.num, c1=0n, c2=sqRat.den;
        const g=gcdBig(c0,c2); if(g>1n){ c0/=g; c2/=g; }
        const complexity=Number(absBig(c0)+absBig(c2));
        if(complexity<=Math.max(4000,bound*bound*8)) return {coeff:[Number(c0),0,Number(c2)], rootSign:rn>=0?1:-1, complexity, value:rn, numErr:0};
      }
      const cat=lfuncBuildQuadraticCatalog(bound);
      let lo=0, hi=cat.length;
      while(lo<hi){ const mid=(lo+hi)>>1; if(cat[mid].v<rn) lo=mid+1; else hi=mid; }
      const tolNum=Math.max(1e-12, typedRelativeToleranceNumber(sig, 8, 1, 12)) * Math.max(1,Math.abs(rn));
      let best=null;
      for(let k=Math.max(0,lo-18); k<Math.min(cat.length,lo+19); k++){
        const item=cat[k]; if(Math.abs(item.v-rn)>tolNum) continue;
        const [c0,c1,c2]=item.coeff;
        const complexity=Math.abs(c0)+Math.abs(c1)+Math.abs(c2);
        const cur={coeff:item.coeff, rootSign:item.rootSign, complexity, value:item.v, numErr:Math.abs(item.v-rn)};
        if(!best || cur.numErr<best.numErr || (cur.numErr===best.numErr && complexity<best.complexity)) best=cur;
      }
      return best;
    }
    function lfuncVerifyQuadraticDecimal(D, r, item, sig){
      if(!item) return null;
      const [c0,c1,c2]=item.coeff;
      const rr=r.mul(r);
      const residual=new D(c0).plus(new D(c1).mul(r)).plus(new D(c2).mul(rr)).abs();
      const denom=new D(Math.abs(c0)).plus(new D(Math.abs(c1)).mul(r.abs())).plus(new D(Math.abs(c2)).mul(rr.abs())).plus(1);
      const rel=residual.div(denom);
      const tol=lfuncTolerance(D,sig,'quadratic').mul(25);
      return rel.lte(tol) ? {...item, err:rel} : null;
    }
    function lfuncQuadraticAlphaString(q){
      const [c,b,a]=q.coeff;
      const D=b*b-4*a*c;
      const den=2*a;
      const left=-b;
      const sign=q.rootSign>=0 ? '+' : '−';
      const rad=`√${D}`;
      let num;
      if(left===0) num = q.rootSign>=0 ? rad : `−${rad}`;
      else num = `${left}${sign}${rad}`;
      return den===1 ? `(${num})` : `(${num})/${den}`;
    }
    function lfuncLogConstants(highPrecision, effort=0){
      effort=Math.max(0, Math.min(7, Number(effort)||0));
      // Low RIES levels should try only very small log products.  Larger
      // exponents and special constants are unlocked gradually by Continue, so
      // a low-level decimal run first surfaces simple L, π/L, and small-product
      // explanations instead of spending time in a broad log catalog.
      const coreMax=highPrecision ? (effort>=5 ? 3 : 2) : [1,2,3,4,5,6,6,7][effort];
      const base=[
        {name:'2', log:Math.log(2), max:coreMax},
        {name:'3', log:Math.log(3), max:coreMax},
        {name:'5', log:Math.log(5), max:coreMax},
        {name:'π', log:Math.log(Math.PI), max:coreMax}
      ];
      if(highPrecision){
        const extraMax=effort>=5 ? 2 : 1;
        base.push({name:'log(2)', log:Math.log(Math.log(2)), max:extraMax});
        base.push({name:'log(3)', log:Math.log(Math.log(3)), max:extraMax});
        base.push({name:'Γ(1/3)', log:0.9854206469277671, max:extraMax});
        base.push({name:'Γ(1/4)', log:1.2880225246980775, max:extraMax});
      }
      return base;
    }
    function lfuncGenerateLogAll(constants){
      const out=[];
      function rec(i, sum, coeff, complexity){
        if(i===constants.length){ out.push({sum, coeff:coeff.slice(), complexity}); return; }
        const c=constants[i];
        for(let k=-c.max;k<=c.max;k++){ coeff.push(k); rec(i+1, sum+k*c.log, coeff, complexity+Math.abs(k)); coeff.pop(); }
      }
      rec(0,0,[],0);
      out.sort((a,b)=>a.sum-b.sum);
      return out;
    }
    function lfuncLogCatalog(highPrecision, effort=0){
      effort=Math.max(0, Math.min(7, Number(effort)||0));
      const key=(highPrecision?'hi':'lo')+':e'+effort;
      if(lfuncLogCatalogCache.has(key)) return lfuncLogCatalogCache.get(key);
      const constants=lfuncLogConstants(highPrecision, effort);
      const combos=lfuncGenerateLogAll(constants);
      const cat={constants, combos, effort, highPrecision};
      lfuncLogCatalogCache.set(key, cat);
      return cat;
    }
    function lfuncNearestLogCombo(cat, target, sig){
      const combos=cat.combos;
      const tol=Math.max(1e-12, typedRelativeToleranceNumber(sig, 6, 1, 12));
      let lo=0, hi=combos.length;
      while(lo<hi){ const mid=(lo+hi)>>1; if(combos[mid].sum<target) lo=mid+1; else hi=mid; }
      let best=null;
      for(let k=Math.max(0,lo-4); k<Math.min(combos.length,lo+5); k++){
        const C=combos[k];
        const err=Math.abs(C.sum-target);
        if(err>tol) continue;
        if(!best || err<best.err || (err===best.err && C.complexity<best.complexity)) best={coeff:C.coeff, err, complexity:C.complexity};
      }
      return best;
    }
    function lfuncProductFromLogCombo(constants, coeff, denom){
      const parts=[];
      for(let i=0;i<constants.length;i++){
        const k=coeff[i]||0; if(k===0) continue;
        const exp=rationalString(BigInt(-k), BigInt(denom));
        parts.push(exp==='1' ? constants[i].name : `${constants[i].name}^(${exp})`);
      }
      return parts.join('·') || '1';
    }
    function lfuncSignedFormula(signNeg, core){ return signNeg ? `−${core}` : core; }
    function lfuncLogFormula(mode, signNeg, L, product){
      if(mode==='direct'){
        const core=product==='1' ? L : `${L}·${product}`;
        return lfuncSignedFormula(signNeg, core);
      }
      const core=product==='1' ? `1/${L}` : `${product}/${L}`;
      return lfuncSignedFormula(signNeg, core);
    }
    function lfuncStateFor(settings){
      const key=lfuncCacheKey(settings);
      let state=lfuncProgressCache.get(key);
      if(!state){
        state={key, simpleRatPointer:0, ratPointer:0, rqPointer:0, logLoPointer:0, logHiPointer:0, rational:[], quadratic:[], log:[], seenRat:new Set(), seenQuad:new Set(), seenLog:new Set(), lastEffort:-1};
        lfuncProgressCache.set(key,state);
      }
      return state;
    }
    function lfuncEffortConfig(effort, totalTasks, entryCount, sig, levelOrSettings=DEFAULT_RIES_LEVEL){
      effort=Math.max(0, Math.min(7, Number(effort)||0));
      const rqCaps=[4200,12000,26000,52000,90000,125000,180000,totalTasks];
      const logCaps=[60,160,420,900,1800,3200,entryCount,entryCount];
      const moduleMs=(typeof levelOrSettings==='object') ? stageBudgetValueToMs(levelOrSettings?.stageBudgets?.lfuncMs, riesLevelDefaultModuleBudgetMs(levelOrSettings)) : riesLevelDefaultModuleBudgetMs(levelOrSettings);
      return {
        effort,
        moduleMs,
        rqTaskCap: Math.min(totalTasks, rqCaps[effort]),
        logEntryCap: Math.min(entryCount, logCaps[effort]),
        ratBound: effort>=2 || sig>=18 ? 6000 : 2200,
        quadBound: effort<=0 ? 12 : (effort===1 ? 18 : (effort<=3 ? 32 : (effort>=5 ? 60 : 40))),
        rationalRankMax: [12,18,28,40,60,80,99,999][effort],
        quadRankMax: [0,4,12,24,40,60,99,999][effort],
        maxLogComplexity: [2,3,5,7,10,14,20,40][effort],
        simpleMs: Math.max([1600,1800,2100,2400,2700,3000,3600,5200][effort], Math.floor(moduleMs*0.34)),
        ratioMs: Math.max([650,850,1100,1450,1900,2300,4200,30000][effort], Math.floor(moduleMs*0.46)),
        logMs: Math.max([240,420,700,1000,1350,1700,3000,15000][effort], Math.floor(moduleMs*0.20)),
        highLog: sig>=18 && effort>=3
      };
    }
    function lfuncMonomialRank(i,j){
      const aj=Math.abs(j);
      if(i===1 && j===0) return 0;       // x / L
      if(i===1 && j===1) return 1;       // x·π / L
      if(i===1 && j===-1) return 2;      // x / (π·L)
      if(i===-1 && j===0) return 3;      // 1 / (x·L)
      if(i===1 && aj===2) return 8+(j<0?1:0);
      if(i===-1 && aj===1) return 12+(j<0?1:0);
      if(i===1 && aj===3) return 20+(j<0?1:0);
      if(i===-1 && aj===2) return 28+(j<0?1:0);
      if(i===2 && j===0) return 36;
      if(i===-2 && j===0) return 40;
      if(Math.abs(i)===2 && aj===1) return 48+(i<0?4:0)+(j<0?1:0);
      if(i===-1 && aj===3) return 64+(j<0?1:0);
      return 80+(Math.abs(i)-1)*10+aj*2+(j<0?1:0);
    }
    const LFUNC_MONOMIALS = (()=>{
      const arr=[];
      for(const i of [1,-1,2,-2]) for(let j=-3;j<=3;j++) arr.push({i,j,rank:lfuncMonomialRank(i,j)});
      arr.sort((a,b)=>a.rank-b.rank || a.i-b.i || a.j-b.j);
      return arr;
    })();
    const LFUNC_PRIORITY_RATIONAL_MONOMIALS = [
      {i:1,j:0,rank:0}, {i:1,j:-1,rank:1}, {i:1,j:1,rank:2},
      {i:-1,j:0,rank:3}, {i:-1,j:-1,rank:4}, {i:-1,j:1,rank:5}
    ];
    function lfuncRationalTaskAllowed(mono, cfg){ return (mono?.rank ?? 999) <= cfg.rationalRankMax; }
    function lfuncQuadraticTaskAllowed(mono, cfg){ return (mono?.rank ?? 999) <= cfg.quadRankMax; }
    function lfuncTryRationalCandidate(D, state, val, valPows, valPowsNum, piPows, piPowsNum, L, mono, cfg, sig){
      if(!L || !mono) return false;
      const i=mono.i, j=mono.j;
      const lnum=Number(L.value); if(!Number.isFinite(lnum) || Math.abs(lnum)<1e-35) return false;
      const rn=valPowsNum[i]*piPowsNum[j]/lnum;
      if(!Number.isFinite(rn) || Math.abs(rn)>1e12 || Math.abs(rn)<1e-12) return false;
      const taskKey=`${L.entryKey}|${i}|${j}`;
      if(state.seenRat.has(taskKey)) return false;
      const ratN=lfuncRationalApproxNumber(rn, cfg.ratBound, sig);
      if(!ratN) return false;
      let l0=null; try{ l0=new D(L.value); }catch(e){ l0=null; }
      if(!l0 || !l0.isFinite() || l0.abs().lt('1e-35')) return false;
      const ratio=valPows[i].mul(piPows[j]).div(l0);
      if(!ratio.isFinite() || ratio.abs().gt('1e12') || ratio.abs().lt('1e-12')) return false;
      const rat=lfuncVerifyRationalDecimal(D, ratio, ratN.num, ratN.den, sig);
      if(!rat) return false;
      state.seenRat.add(taskKey);
      const formula=lfuncRationalFormula(rat.num, rat.den, i, j, L.label, val.isNeg());
      const qstr=rationalString(rat.num,rat.den);
      const cplx=Number(rat.err.toString()) + Math.log10(Number(absBig(rat.num)+rat.den)+1)*1e-6 + (Math.abs(i)-1)*4e-6 + Math.abs(j)*2e-7;
      state.rational.push({formula, L, i, j, qstr, err:rat.err, score:cplx, detail:`${lfuncTestMonomialString(i,j,L.label)} ≈ ${qstr}; relative residual ${lfuncFormatDecimal(D,rat.err,4)}`});
      return true;
    }
    function lfuncBestRows(arr, n=3){
      const seen=new Set();
      return arr.sort((a,b)=>a.score-b.score).filter(r=>{ const k=r.formula+'|'+r.L.entryKey; if(seen.has(k)) return false; seen.add(k); return true; }).slice(0,n);
    }
    async function lfuncRowsAsync(settings, effort=0, onProgress=null){
      if(!lfuncShouldRun(settings) || !window.Decimal) return [];
      const D=window.Decimal;
      const prev={precision:D.precision, rounding:D.rounding, toExpNeg:D.toExpNeg, toExpPos:D.toExpPos};
      const sig=typedInputPrecision(settings);
      const workPrec=Math.max(24, Math.min(80, sig+12));
      D.set({precision:workPrec, toExpNeg:-80, toExpPos:80});
      const entries=lfuncEntries();
      const totalTasks=entries.length*LFUNC_MONOMIALS.length;
      const cfg=lfuncEffortConfig(effort,totalTasks,entries.length,sig, settings);
      const state=lfuncStateFor(settings);
      state.lastEffort=Math.max(state.lastEffort,cfg.effort);
      try{
        const val=lfuncDecimalFromRational(D, settings.parsedComplex.re);
        if(val.isZero()) return [];
        const valPows={}; for(const i of [-2,-1,1,2]) valPows[i]=lfuncDecimalPowInt(D,val,i);
        const piPows={}; for(let j=-3;j<=3;j++) piPows[j]=lfuncPiPow(D,j);
        const valNum=Number(val.toString());
        const valPowsNum={}; for(const i of [-2,-1,1,2]) valPowsNum[i]=Math.pow(valNum,i);
        const piPowsNum={}; for(let j=-3;j<=3;j++) piPowsNum[j]=Math.pow(Math.PI,j);
        if(typeof state.simpleRatPointer!=='number') state.simpleRatPointer=0;
        // v9.1: before any broad L-function work, sweep every modular form with
        // the mathematically simplest shapes x = c·L0·π^a and x = c·π^a/L0
        // (a=-1,0,1; small rational c).  v9 sorted simple monomials first, but
        // still iterated entry-major; a late database entry with x/L0=1 could
        // therefore require several Continue clicks.  This monomial-major pass
        // avoids excluding exact positives at low RIES level.
        const simpleTotal=entries.length*LFUNC_PRIORITY_RATIONAL_MONOMIALS.length;
        const simpleDeadline=performance.now()+cfg.simpleMs;
        while(state.simpleRatPointer<simpleTotal && performance.now()<simpleDeadline){
          const mono=LFUNC_PRIORITY_RATIONAL_MONOMIALS[Math.floor(state.simpleRatPointer / entries.length)];
          const L=entries[state.simpleRatPointer % entries.length];
          state.simpleRatPointer++;
          lfuncTryRationalCandidate(D,state,val,valPows,valPowsNum,piPows,piPowsNum,L,mono,cfg,sig);
          if((state.simpleRatPointer&255)===0){ if(onProgress) onProgress({phase:'simple-rational', done:state.simpleRatPointer, total:simpleTotal, effort:cfg.effort}); await yieldToUI(); }
        }
        if(typeof state.ratPointer!=='number') state.ratPointer=0;
        const rationalDeadline=performance.now()+Math.min(1150, Math.max(450, cfg.ratioMs*0.50));
        while(state.ratPointer<cfg.rqTaskCap && performance.now()<rationalDeadline){
          const L=entries[Math.floor(state.ratPointer / LFUNC_MONOMIALS.length)];
          const mono=LFUNC_MONOMIALS[state.ratPointer % LFUNC_MONOMIALS.length];
          state.ratPointer++;
          if(!L) continue;
          if(!lfuncRationalTaskAllowed(mono,cfg)) continue;
          lfuncTryRationalCandidate(D,state,val,valPows,valPowsNum,piPows,piPowsNum,L,mono,cfg,sig);
          if((state.ratPointer&255)===0){ if(onProgress) onProgress({phase:'rational', done:state.ratPointer, total:totalTasks, effort:cfg.effort}); await yieldToUI(); }
        }
        const ratioDeadline=performance.now()+Math.max(0, cfg.ratioMs-(performance.now()-rationalDeadline+Math.min(1150, Math.max(450, cfg.ratioMs*0.50))));
        while(state.rqPointer<cfg.rqTaskCap && performance.now()<ratioDeadline){
          const L=entries[Math.floor(state.rqPointer / LFUNC_MONOMIALS.length)];
          const mono=LFUNC_MONOMIALS[state.rqPointer % LFUNC_MONOMIALS.length];
          state.rqPointer++;
          if(!L) continue;
          const allowRat=lfuncRationalTaskAllowed(mono,cfg);
          const allowQuad=lfuncQuadraticTaskAllowed(mono,cfg);
          if(!allowRat && !allowQuad) continue;
          const i=mono.i, j=mono.j;
          const lnum=Number(L.value); if(!Number.isFinite(lnum) || Math.abs(lnum)<1e-35) continue;
          const rn=valPowsNum[i]*piPowsNum[j]/lnum;
          if(!Number.isFinite(rn) || Math.abs(rn)>1e12 || Math.abs(rn)<1e-12) continue;
          const taskKey=`${L.entryKey}|${i}|${j}`;
          let l0=null;
          const getRatio=()=>{
            if(!l0){ try{ l0=new D(L.value); }catch(e){ l0=null; } }
            if(!l0 || !l0.isFinite() || l0.abs().lt('1e-35')) return null;
            const ratio=valPows[i].mul(piPows[j]).div(l0);
            if(!ratio.isFinite() || ratio.abs().gt('1e12') || ratio.abs().lt('1e-12')) return null;
            return ratio;
          };
          const ratN=allowRat ? lfuncRationalApproxNumber(rn, cfg.ratBound, sig) : null;
          if(ratN && !state.seenRat.has(taskKey)){
            const ratio=getRatio();
            if(ratio){
              const rat=lfuncVerifyRationalDecimal(D, ratio, ratN.num, ratN.den, sig);
              if(rat){
                state.seenRat.add(taskKey);
                const formula=lfuncRationalFormula(rat.num, rat.den, i, j, L.label, val.isNeg());
                const qstr=rationalString(rat.num,rat.den);
                const cplx=Number(rat.err.toString()) + Math.log10(Number(absBig(rat.num)+rat.den)+1)*1e-6 + (Math.abs(i)-1)*4e-6 + Math.abs(j)*2e-7;
                state.rational.push({formula, L, i, j, qstr, err:rat.err, score:cplx, detail:`${lfuncTestMonomialString(i,j,L.label)} ≈ ${qstr}; relative residual ${lfuncFormatDecimal(D,rat.err,4)}`});
              }
            }
          }
          const quadN=allowQuad ? lfuncQuadraticSearchNumber(rn, sig, cfg.quadBound) : null;
          if(quadN && !state.seenQuad.has(taskKey)){
            const ratio=getRatio();
            if(ratio){
              const quad=lfuncVerifyQuadraticDecimal(D, ratio, quadN, sig);
              if(quad){
                state.seenQuad.add(taskKey);
                const alpha=lfuncQuadraticAlphaString(quad);
                const formula=lfuncFormulaFromScale(alpha, i, j, L.label, val.isNeg());
                const cplx=Number(quad.err.toString()) + quad.complexity*1e-9 + (Math.abs(i)-1)*3e-9 + Math.abs(j)*1e-9;
                state.quadratic.push({formula, L, i, j, alpha, coeff:quad.coeff, err:quad.err, score:cplx, detail:`${lfuncTestMonomialString(i,j,L.label)} ≈ α, α=${alpha}; polynomial ${polyString(quad.coeff.map(BigInt))}; relative residual ${lfuncFormatDecimal(D,quad.err,4)}`});
              }
            }
          }
          if((state.rqPointer&127)===0){ if(onProgress) onProgress({phase:'ratio', done:state.rqPointer, total:totalTasks, effort:cfg.effort}); await yieldToUI(); }
        }
        const logDeadline=performance.now()+cfg.logMs;
        const runLogPass=async (highPrecision)=>{
          const pointerName=highPrecision?'logHiPointer':'logLoPointer';
          const cat=lfuncLogCatalog(highPrecision, cfg.effort);
          const logAbsVal=Math.log(Math.abs(Number(val.toString())));
          const maxDen=highPrecision ? (cfg.effort>=5 ? 5 : 3) : (cfg.effort<=1 ? 1 : (cfg.effort<=3 ? 2 : 3));
          while(state[pointerName]<cfg.logEntryCap && performance.now()<logDeadline){
            const idx=state[pointerName]++;
            const L=entries[idx]; if(!L) continue;
            const lnum=Number(L.value); if(!Number.isFinite(lnum) || lnum===0) continue;
            const logAbsL=Math.log(Math.abs(lnum));
            for(const mode of ['direct','inverse']){
              const base = mode==='direct' ? (logAbsVal-logAbsL) : (logAbsVal+logAbsL);
              for(let den=1; den<=maxDen; den++){
                const combo=lfuncNearestLogCombo(cat, -den*base, sig);
                if(!combo || combo.complexity>cfg.maxLogComplexity) continue;
                const product=lfuncProductFromLogCombo(cat.constants, combo.coeff, den);
                const signNeg = (Number(val.toString())<0) !== (lnum<0);
                const formula=lfuncLogFormula(mode, signNeg, L.label, product);
                const key=`${highPrecision?'hi':'lo'}|${L.entryKey}|${mode}|${den}|${product}`;
                if(state.seenLog.has(key)) continue;
                state.seenLog.add(key);
                const score=combo.err*1e6 + combo.complexity + den*2 + (mode==='inverse'?3:0) + (highPrecision?0.25:0);
                state.log.push({formula, L, product, mode, den, err:combo.err, score, detail:`log relation ${mode==='direct'?'x/L0':'x·L0'} with denominator ${den}; product ${product}`});
              }
            }
            if((idx&63)===0){ if(onProgress) onProgress({phase:highPrecision?'high-log':'log', done:idx, total:cfg.logEntryCap, effort:cfg.effort}); await yieldToUI(); }
          }
        };
        await runLogPass(false);
        if(cfg.highLog) await runLogPass(true);
        const out=[];
        const ltake=Math.max(12, Math.min(96, Number(settings.limit||5)*8));
        const lopt=settings?.lfuncOptions || {rational:true,log:true,quadratic:true};
        if(lopt.rational!==false) lfuncBestRows(state.rational, ltake).forEach((r,idx)=>out.push(lfuncCandidateRow('rational',idx+1,r.formula,r.L,r.detail,Number(r.err.toString()),r.score)));
        if(lopt.log!==false) lfuncBestRows(state.log, ltake).forEach((r,idx)=>out.push(lfuncCandidateRow('log',idx+1,r.formula,r.L,r.detail,r.err,r.score)));
        if(lopt.quadratic!==false) lfuncBestRows(state.quadratic, ltake).forEach((r,idx)=>out.push(lfuncCandidateRow('quadratic',idx+1,r.formula,r.L,r.detail,Number(r.err.toString()),r.score)));
        out._lfuncProgress={effort:cfg.effort, simpleDone:state.simpleRatPointer, simpleTotal, ratioDone:state.rqPointer, ratioTotal:totalTasks, logDone:Math.max(state.logLoPointer,state.logHiPointer), logTotal:entries.length};
        return out;
      }catch(e){ console.warn('L-function matcher failed', e); return []; }
      finally{ try{ D.set(prev); }catch(e){} }
    }

    const RIES_SPECIAL_DECIMAL_CONSTANTS = [
      {name:'Γ(1/4)', latex:'\\Gamma(1/4)', value:'3.625609908221908311930685155867672002995167682880065467433377'},
      {name:'Γ(3/4)', latex:'\\Gamma(3/4)', value:'1.225416702465177645129098303362890526851239248108078520'},
      {name:'Γ(1/3)', latex:'\\Gamma(1/3)', value:'2.678938534707747633655692940974677644128689377957302'},
      {name:'Γ(2/3)', latex:'\\Gamma(2/3)', value:'1.3541179394264004169452880281545137855193272660568'},
      {name:'ζ(3)', latex:'\\zeta(3)', value:'1.202056903159594285399738161511449990764986292340499'},
      {name:'Catalan G', latex:'G', value:'0.91596559417721901505460351493238411077414937428167'},
      {name:'Glaisher A', latex:'A', value:'1.2824271291006226368753425688697917277676889273250'}
    ];
    function specialDecimalConstantRows(settings, effort=0){
      if(!settings || settings.lfuncOptions?.specialConstants===false || settings.complexTarget || !Number.isFinite(settings.target) || effort<1 || !window.Decimal) return [];
      const sig=typedInputPrecision(settings);
      const D=window.Decimal;
      const prev={precision:D.precision, rounding:D.rounding, toExpNeg:D.toExpNeg, toExpPos:D.toExpPos};
      const rows=[];
      try{
        D.set({precision:Math.max(24, Math.min(90, sig+12)), toExpNeg:-100, toExpPos:100});
        const val=lfuncDecimalFromRational(D, settings.parsedComplex.re);
        const tol=new D(10).pow(-matchToleranceDigits(sig,1,30)).mul(5);
        for(const c of RIES_SPECIAL_DECIMAL_CONSTANTS){
          const cv=new D(c.value);
          const err=val.minus(cv).abs().div(lfuncDecimalAbsMax1(D,cv));
          if(err.lte(tol)){
            const e=Number(err.toString());
            rows.push({candidate:`constant match: ${c.name}`, latex:`x \approx ${sanitizeLatexForDisplay(c.latex)}`, copyLatex:`x \approx ${sanitizeLatexForDisplay(c.latex)}`, value:`x = ${c.name} ≈ ${cv.toSignificantDigits(Math.min(24, Math.max(12,sig+4))).toString()}; relative residual ${lfuncFormatDecimal(D,err,4)}`, copyValue:`${c.name} ≈ ${c.value}`, err:e, specialConstant:true, score:e + c.name.length*1e-12});
          }
        }
      }catch(e){}
      finally{ try{ D.set(prev); }catch(e){} }
      return rows.sort((a,b)=>(a.score??1e99)-(b.score??1e99)).slice(0,Math.max(3, Math.min(10, Number(settings.limit||5))));
    }

    function integerInputBig(raw){
      const s=String(raw||'').trim();
      if(!/^[+-]?\d+$/.test(s)) return null;
      try { return BigInt(s); } catch(e){ return null; }
    }
    function normalizeIntegerExprInput(raw){
      return String(raw||'').trim()
        .replace(/[−–—]/g,'-')
        .replace(/[×*]/g,'·')
        .replace(/÷/g,'/')
        .replace(/π/gi,'pi')
        .replace(/\bchoose\s*\(/gi,'binom(')
        .replace(/\bncr\s*\(/gi,'binom(')
        .replace(/\bc\s*\(/gi,'binom(')
        .replace(/\bcomb\s*\(/gi,'binom(')
        .replace(/\bnpr\s*\(/gi,'perm(')
        .replace(/\ba\s*\(/gi,'perm(')
        .replace(/\bpermutation\s*\(/gi,'perm(')
        .replace(/\bfibonacci\s*\(/gi,'fib(')
        .replace(/\bfactorial\s*\(/gi,'fact(')
        .replace(/\s+/g,'')
        .toLowerCase();
    }
    function splitArgsTopLevel(s){
      const out=[]; let depth=0, start=0;
      for(let i=0;i<s.length;i++){
        const ch=s[i];
        if(ch==='(') depth++;
        else if(ch===')') depth--;
        else if(ch===',' && depth===0){ out.push(s.slice(start,i)); start=i+1; }
      }
      out.push(s.slice(start));
      return out;
    }
    function parseExactIntegerExpression(raw){
      const src=normalizeIntegerExprInput(raw);
      if(!src || src.length>500 || /[^0-9a-z_+\-·/^(),!]/.test(src)) return null;
      let pos=0;
      const capAbs=10n**250n;
      function peek(){ return src[pos]; }
      function eat(ch){ if(src[pos]===ch){ pos++; return true; } return false; }
      function fail(){ throw new Error('bad integer expr'); }
      function checked(v){ v=BigInt(v); if(v>capAbs || v<-capAbs) fail(); return v; }
      function parseExpr(){ return parseAddSub(); }
      function parseAddSub(){
        let v=parseMulDiv();
        while(true){
          if(eat('+')) v=checked(v+parseMulDiv());
          else if(eat('-')) v=checked(v-parseMulDiv());
          else break;
        }
        return v;
      }
      function parseMulDiv(){
        let v=parsePow();
        while(true){
          if(eat('·')) v=checked(v*parsePow());
          else if(eat('/')){ const b=parsePow(); if(b===0n || v%b!==0n) fail(); v=checked(v/b); }
          else break;
        }
        return v;
      }
      function parsePow(){
        let v=parseUnary();
        if(eat('^')){
          const e=parsePow();
          if(e<0n || e>10000n) fail();
          v=powBigCapped(v,e,capAbs);
          if(v===null) fail();
        }
        return checked(v);
      }
      function parseUnary(){
        if(eat('+')) return parseUnary();
        if(eat('-')) return checked(-parseUnary());
        return parsePostfix();
      }
      function parsePostfix(){
        let v=parsePrimary();
        while(eat('!')){ if(v<0n || v>400n) fail(); v=powFactorialBig(v,capAbs); if(v===null) fail(); }
        return checked(v);
      }
      function parsePrimary(){
        if(eat('(')){ const v=parseExpr(); if(!eat(')')) fail(); return v; }
        if(/[0-9]/.test(peek()||'')){ let j=pos; while(/[0-9]/.test(src[j]||'')) j++; const v=BigInt(src.slice(pos,j)); pos=j; return checked(v); }
        if(/[a-z_]/.test(peek()||'')){
          let j=pos; while(/[a-z0-9_]/.test(src[j]||'')) j++; const name=src.slice(pos,j); pos=j;
          if(!eat('(')) fail();
          const args=[]; if(!eat(')')){ do{ args.push(parseExpr()); }while(eat(',')); if(!eat(')')) fail(); }
          return evalFn(name,args);
        }
        fail();
      }
      function evalFn(name,args){
        if((name==='fact' || name==='factorial') && args.length===1){ const n=args[0]; if(n<0n || n>400n) fail(); const r=powFactorialBig(n,capAbs); if(r===null) fail(); return r; }
        if((name==='binom' || name==='choose' || name==='ncr' || name==='c') && args.length===2){ const [n,k]=args; if(n<0n || k<0n || k>n || n>5000n) fail(); const r=binomBigCapped(n,k,capAbs); if(r===null) fail(); return r; }
        if((name==='perm' || name==='npr' || name==='a') && args.length===2){ const [n,k]=args; if(n<0n || k<0n || k>n || n>5000n) fail(); let r=1n; for(let i=0n;i<k;i++){ r*=n-i; if(absBig(r)>capAbs) fail(); } return r; }
        if(name==='gcd' && args.length>=2){ let g=0n; for(const a of args) g=gcdBig(g,a); return absBig(g); }
        if(name==='lcm' && args.length>=2){ let r=1n; for(const a of args){ if(a===0n) return 0n; r=absBig(r/gcdBig(r,a)*a); if(r>capAbs) fail(); } return r; }
        if((name==='fib' || name==='fibonacci') && args.length===1){ const n=args[0]; if(n<0n || n>2000n) fail(); let a=0n,b=1n; for(let i=0n;i<n;i++){ const t=a+b; a=b; b=t; if(b>capAbs) fail(); } return a; }
        if(name==='catalan' && args.length===1){ const n=args[0]; if(n<0n || n>1000n) fail(); const c=binomBigCapped(2n*n,n,capAbs*(n+1n)); if(c===null || c%(n+1n)!==0n) fail(); return c/(n+1n); }
        fail();
      }
      try{ const v=parseExpr(); if(pos!==src.length) return null; return checked(v); }catch(e){ return null; }
    }
    function resolvedIntegerBig(settings){
      const direct=integerInputBig(settings && settings.raw);
      if(direct!==null) return direct;
      const exprInt=parseExactIntegerExpression(settings && settings.raw);
      if(exprInt!==null) return exprInt;
      const hpInt=settings && settings._hpIntegerString;
      if(typeof hpInt==='string' && /^[+-]?\d+$/.test(hpInt)){
        try{ return BigInt(hpInt); }catch(e){}
      }
      const norm=integerInputBig(settings && settings.normalizedRaw);
      return norm;
    }
    function modPowBig(a,e,m){ let r=1n; a%=m; while(e>0n){ if(e&1n) r=(r*a)%m; a=(a*a)%m; e>>=1n; } return r; }
    function isProbablePrime(n){
      if(n<2n) return false;
      const small=[2n,3n,5n,7n,11n,13n,17n,19n,23n,29n,31n,37n];
      for(const p of small){ if(n===p) return true; if(n%p===0n) return false; }
      let d=n-1n, s=0; while((d&1n)===0n){ d>>=1n; s++; }
      const bases=[2n,3n,5n,7n,11n,13n,17n,19n,23n,29n,31n,37n];
      for(const a0 of bases){
        const a=a0%(n-2n)+2n;
        let x=modPowBig(a,d,n);
        if(x===1n || x===n-1n) continue;
        let ok=false;
        for(let r=1;r<s;r++){ x=(x*x)%n; if(x===n-1n){ ok=true; break; } }
        if(!ok) return false;
      }
      return true;
    }
    function pollardRho(n, deadline){
      if(n%2n===0n) return 2n;
      if(n%3n===0n) return 3n;
      let seed=1n;
      while(performance.now()<deadline){
        let c=(seed++ % (n-1n)) + 1n;
        let x=(2n + seed) % n, y=x, d=1n;
        const f=v=>(v*v+c)%n;
        let iter=0;
        while(d===1n && performance.now()<deadline){
          x=f(x); y=f(f(y)); d=gcdBig(absBig(x-y), n);
          if(++iter>12000) break;
        }
        if(d>1n && d<n) return d;
      }
      return 0n;
    }
    async function pollardRhoAsync(n, deadline, onProgress){
      if(n%2n===0n) return 2n;
      if(n%3n===0n) return 3n;
      let seed=1n;
      let lastYield=performance.now();
      while(performance.now()<deadline){
        let c=(seed++ % (n-1n)) + 1n;
        let x=(2n + seed) % n, y=x, d=1n;
        const f=v=>(v*v+c)%n;
        let iter=0;
        while(d===1n && performance.now()<deadline){
          x=f(x); y=f(f(y)); d=gcdBig(absBig(x-y), n);
          iter++;
          const now=performance.now();
          if(iter>12000) break;
          if(now-lastYield>32){
            lastYield=now;
            if(onProgress) onProgress(now);
            await yieldToUI();
            if(activeShortformRun?.stopped) return 0n;
          }
        }
        if(d>1n && d<n) return d;
        const now=performance.now();
        if(now-lastYield>32){
          lastYield=now;
          if(onProgress) onProgress(now);
          await yieldToUI();
          if(activeShortformRun?.stopped) return 0n;
        }
      }
      return 0n;
    }
    let SMALL_PRIMES_10000=null;
    function smallPrimes10000(){
      if(SMALL_PRIMES_10000) return SMALL_PRIMES_10000;
      const lim=10000, sieve=Array(lim+1).fill(true); sieve[0]=sieve[1]=false;
      for(let p=2;p*p<=lim;p++) if(sieve[p]) for(let q=p*p;q<=lim;q+=p) sieve[q]=false;
      SMALL_PRIMES_10000=[]; for(let p=2;p<=lim;p++) if(sieve[p]) SMALL_PRIMES_10000.push(BigInt(p));
      return SMALL_PRIMES_10000;
    }
    function factorTimeLimitMs(settings,n){
      const override=Number(settings?.stageBudgets?.integerFactorMs || 0);
      if(Number.isFinite(override) && override>0) return override;
      const effort=Math.max(0,Math.min(7,Number(settings.shortEffort)||0));
      const digits=decimalDigitCountBig(absBig(n));
      const external=!!settings.allowExternalFactorization;
      if(!external){
        // v8.7: do not let medium-size integers spend most of Continue on
        // factoring before the database/shortform phases can show good rows.
        // Pollard/trial factoring still runs, but hard cofactors hand off sooner.
        if(digits<=18) return Math.min(2600, 850 + effort*260);
        if(digits<=27) return Math.min(4800, 1500 + effort*420);
        if(digits>=40) return Math.min(9000, 2400 + effort*650);
        if(digits>=28) return Math.min(7000, 2600 + effort*550);
        return Math.min(6200, 2200 + effort*500);
      }
      if(digits>=40 && digits<=55 && effort>=5) return Math.min(55000, 16000 + effort*5200);
      if(digits>=32 && effort>=4) return Math.min(38000, 12000 + effort*3600);
      return Math.min(24000, 9000 + effort*1800);
    }
    function factorBigIntWithin(n, ms=10000){
      const start=performance.now();
      const deadline=start+ms;
      const factors=[];
      let sign='';
      if(n<0n){ sign='-1'; n=-n; }
      function rec(x){
        if(shortAbort(deadline)) return false;
        if(x===1n) return true;
        for(const p of smallPrimes10000()){
          if(x===p){ factors.push(p); return true; }
          if(x%p===0n){ factors.push(p); return rec(x/p); }
          if(p*p>x && x<100000000n) break;
          if(performance.now()>deadline) return false;
        }
        if(isProbablePrime(x)){ factors.push(x); return true; }
        const d=pollardRho(x, deadline);
        if(!d) return false;
        return rec(d) && rec(x/d);
      }
      const complete=rec(n);
      factors.sort((a,b)=>a<b?-1:a>b?1:0);
      return {complete, sign, factors, ms:Math.round(performance.now()-start), limitMs:ms};
    }
    async function factorBigIntWithinAsync(n, ms=10000, onProgress=null){
      const start=performance.now();
      const deadline=start+ms;
      const factors=[];
      let sign='';
      if(n<0n){ sign='-1'; n=-n; }
      let lastYield=start;
      const report=()=>{ if(onProgress) onProgress({elapsed:Math.round(performance.now()-start), limitMs:ms, factors:factors.slice()}); };
      async function maybeYield(force=false){
        const now=performance.now();
        if(force || now-lastYield>34){
          lastYield=now;
          report();
          await yieldToUI();
        }
      }
      async function rec(x){
        if(performance.now()>deadline || activeShortformRun?.stopped) return false;
        if(x===1n) return true;
        const primes=smallPrimes10000();
        for(let i=0;i<primes.length;i++){
          const p=primes[i];
          if(x===p){ factors.push(p); report(); return true; }
          if(x%p===0n){ factors.push(p); report(); await maybeYield(true); return rec(x/p); }
          if(p*p>x && x<100000000n) break;
          if(performance.now()>deadline) return false;
          if((i&63)===0) await maybeYield(false);
        }
        await maybeYield(true);
        if(isProbablePrime(x)){ factors.push(x); report(); return true; }
        await maybeYield(true);
        const d=await pollardRhoAsync(x, deadline, ()=>report());
        if(!d) return false;
        await maybeYield(true);
        const a=await rec(d);
        if(!a) return false;
        return rec(x/d);
      }
      const complete=await rec(n);
      factors.sort((a,b)=>a<b?-1:a>b?1:0);
      return {complete, sign, factors, ms:Math.round(performance.now()-start), limitMs:ms};
    }
    function factorOutToRow(n,out, sourceLabel='integer factorization'){
      if(!out.factors.length) return {candidate:sourceLabel, value:`No nontrivial factor found within ${(out.limitMs/1000).toFixed(1)} s for ${n.toString()}.`, copyValue:'', err:0, hideError:true, noCandidateCopy:true};
      const counts=new Map();
      for(const f of out.factors) counts.set(f.toString(), (counts.get(f.toString())||0)+1);
      const parts=[]; if(out.sign) parts.push(out.sign);
      for(const [p,e] of counts) parts.push(e===1?p:`${p}^${e}`);
      const math=`${n.toString()} = ${parts.join(' × ')}${out.complete?'': ' × …'}`;
      return {candidate: out.complete ? sourceLabel : `${sourceLabel} (partial, ${(out.limitMs/1000).toFixed(1)} s cutoff)`, value:`${math} (${out.ms} ms)`, copyValue:math, err:0, hideError:true, noCandidateCopy:true};
    }
    function unresolvedCompositeRemainder(original, out){
      if(out.complete) return 1n;
      let rem=absBig(original);
      for(const f of out.factors){ if(f>1n && rem%f===0n) rem/=f; }
      return rem;
    }
    function alpertronLinkRow(n){
      const valueHtml=`For hard 40+ digit composites, open Alpertron ECM/SIQS and paste <code>${escapeHtml(n.toString())}</code>. `+
        `<a target="_blank" rel="noopener" href="https://www.alpertron.com.ar/ECM.HTM">Open external ECM/SIQS calculator</a>`;
      return {candidate:'external factorization handoff', valueHtml, value:`Open Alpertron ECM/SIQS for ${n.toString()}`, copyValue:'', err:0, hideError:true, noCandidateCopy:true};
    }
    function tryExternalQuadraticSieveFactor(n, ms=16000){
      // v6.8: optional web-worker handoff to Yaffle's browser BigInt quadratic
      // sieve package via jsDelivr.  It is isolated and killed on timeout so it
      // cannot freeze the UI; if the CDN is unavailable, the row simply reports
      // the normal local result.
      if(typeof Worker==='undefined' || typeof Blob==='undefined') return Promise.resolve(null);
      const code=`
        const urls=[
          'https://cdn.jsdelivr.net/npm/quadraticsievefactorization/QuadraticSieveFactorization.js',
          'https://cdn.jsdelivr.net/gh/Yaffle/QuadraticSieveFactorization/QuadraticSieveFactorization.js'
        ];
        self.onmessage=async ev=>{
          const s=ev.data;
          for(const url of urls){
            try{
              const mod=await import(url);
              const factorize=mod.default || mod.factorize || mod;
              if(typeof factorize!=='function') continue;
              const f=factorize(BigInt(s));
              const g=Array.isArray(f)?f[0]:f;
              if(typeof g==='bigint' && g>1n && g<BigInt(s)) { self.postMessage({factor:g.toString(), url}); return; }
            }catch(e){}
          }
          self.postMessage({factor:null});
        };`;
      return new Promise(resolve=>{
        let worker=null, done=false;
        const finish=v=>{ if(done) return; done=true; try{worker&&worker.terminate();}catch(e){} resolve(v); };
        try{
          const blob=new Blob([code], {type:'text/javascript'});
          worker=new Worker(URL.createObjectURL(blob), {type:'module'});
          worker.onmessage=e=>finish(e.data && e.data.factor ? {factor:BigInt(e.data.factor), source:e.data.url||'quadratic sieve'} : null);
          worker.onerror=()=>finish(null);
          worker.postMessage(n.toString());
          setTimeout(()=>finish(null), ms);
        }catch(e){ finish(null); }
      });
    }
    async function factorRowsAsync(settings, onProgress=null){
      const n=resolvedIntegerBig(settings);
      if(n===null) return [];
      if(n===0n) return [{candidate:'integer factorization', value:'0 has no finite prime factorization.', err:0}];
      if(absBig(n)===1n) return [{candidate:'integer factorization', value:`${n.toString()} is a unit.`, err:0}];
      const limitMs=factorTimeLimitMs(settings,n);
      const out=await factorBigIntWithinAsync(n, limitMs, onProgress);
      const rows=[factorOutToRow(n,out)];
      const rem=unresolvedCompositeRemainder(n,out);
      const remDigits=decimalDigitCountBig(rem);
      const effort=Math.max(0,Math.min(7,Number(settings.shortEffort)||0));
      if(rem>1n && remDigits>=40){
        const remPrime=isProbablePrime(rem);
        if(!remPrime && settings.allowExternalFactorization){
          if(effort>=5 && remDigits<=58){
            const qs=await tryExternalQuadraticSieveFactor(rem, Math.min(36000, 9000+effort*4200));
            if(qs && qs.factor && rem%qs.factor===0n){
              const f=qs.factor, g=rem/f;
              const math=`${rem.toString()} = ${f.toString()} × ${g.toString()}`;
              rows.push({candidate:'external quadratic sieve factor', value:`${math} (${qs.source})`, copyValue:math, err:0, hideError:true, noCandidateCopy:true});
            }else{
              rows.push(alpertronLinkRow(rem));
            }
          }else{
            rows.push(alpertronLinkRow(rem));
          }
        }else if(!remPrime){
          rows.push({candidate:'remaining composite status', value:`Remaining ${remDigits}-digit cofactor is composite; external QS/ECM is disabled, so local factoring stops here.`, copyValue:'', err:0, hideError:true, noCandidateCopy:true});
        }
      }
      return rows;
    }
    function decimalDigitCountBig(n){ return absBig(n).toString().length; }
    function shortPrettyValue(v){ const s=v.toString(); return s.length > 34 ? s.slice(0,18)+'…'+s.slice(-12) : s; }
    function rawSignedTargetForCopy(target){ return String(target); }
    function digitCountExpr(s){ const m=String(s).match(/\d/g); return m ? m.length : 0; }
    function exprFeature(s){
      s=String(s||'');
      if(s.includes('round(')) return 'round';
      if(s.includes('floor(') || s.includes('ceil(')) return 'floor-ceil';
      if(s.includes('binom(')) return 'binom';
      if(s.includes('!')) return 'factorial';
      if(s.includes('^')) return 'power';
      if(s.includes('/')) return 'division';
      if(s.includes('·')) return 'product';
      if(s.includes('+') || s.includes('-')) return 'offset';
      return 'simple';
    }
    function normalizeShortDisplay(s){
      return String(s)
        .replaceAll('+-','-')
        .replaceAll('--','+')
        .replace(/\b1!/g,'1')
        .replace(/\b2!/g,'2')
        .replace(/\b3!/g,'6')
        .replace(/\(([^()]+)\)/g, (m,x)=>/^[\w!^]+$/.test(x)?x:m);
    }
    function shortRank(e){
      const s=e.s || '';
      const digits=e.digits ?? digitCountExpr(s);
      const ops=e.ops||0;
      const depth=e.depth||0;
      const nums=(s.match(/\d+/g)||[]).map(x=>Number(x)).filter(Number.isFinite);
      let r=digits*1000000 + ops*9000 + depth*900 + s.length*18;
      const maxNum=nums.length ? Math.max(...nums) : 0;
      r += nums.reduce((a,b)=>a+Math.min(220,b),0)*2.0;
      r += Math.min(9000, maxNum>0 ? Math.log10(maxNum+1)*420 : 0);
      if(nums.some(x=>x>=1000)) r += 3600;
      if(/binom\(/.test(s)) r -= 9000;
      if(/round\(/.test(s)) r -= 12000;
      if(/floor\(|ceil\(/.test(s)) r -= 11000;
      if(/!/.test(s)) r -= 9000;
      if(/\^/.test(s)) r -= 12000;
      if(/\+1$/.test(s)||/-1$/.test(s)) r -= 3800;
      if(/[+\-][2-9]$/.test(s)) r -= 1600;
      if(/\d{4,}/.test(s)) r += 6000;
      if(/\^binom\(|\^\([^)]*binom\(/.test(s)) r += 18000;
      if((s.match(/binom\(/g)||[]).length>1) r += 10000;
      if((s.match(/round\(|floor\(|ceil\(/g)||[]).length>1) r += 12000;
      return r;
    }
    function makeDExpr(v,s,kind='atom',ops=0,depth=0){
      const out={v:BigInt(v), s:normalizeShortDisplay(String(s)), kind, ops, depth};
      out.digits=digitCountExpr(out.s);
      out.rank=shortRank(out);
      return out;
    }
    function cmpExpr(a,b){ return (a.digits-b.digits) || (a.rank-b.rank) || (a.s.length-b.s.length) || (a.s<b.s?-1:a.s>b.s?1:0); }
    function shortOperandD(e, op){
      const s=e.s;
      if(op==='^') return /[+\-·/]/.test(s) ? `(${s})` : s;
      if(op==='*') return /[+\-]/.test(s) ? `(${s})` : s;
      if(op==='/') return /[+\-·/]/.test(s) ? `(${s})` : s;
      if(op==='-') return /[+\-]/.test(s) ? `(${s})` : s;
      return s;
    }
    function combineD(a,op,b,v,kind='compound'){
      if(op==='+' && b.v===0n) return makeDExpr(v, a.s, kind, a.ops||0, a.depth||0);
      if(op==='+' && a.v===0n) return makeDExpr(v, b.s, kind, b.ops||0, b.depth||0);
      if(op==='-' && b.v===0n) return makeDExpr(v, a.s, kind, a.ops||0, a.depth||0);
      if(op==='*' && b.v===1n) return makeDExpr(v, a.s, kind, a.ops||0, a.depth||0);
      if(op==='*' && a.v===1n) return makeDExpr(v, b.s, kind, b.ops||0, b.depth||0);
      if(op==='/' && b.v===1n) return makeDExpr(v, a.s, kind, a.ops||0, a.depth||0);
      if((op==='+' || op==='*') && a.v!==undefined && b.v!==undefined && a.v<b.v){ const t=a; a=b; b=t; }
      let s;
      if(op==='+') s=`${shortOperandD(a,'+')}+${shortOperandD(b,'+')}`;
      else if(op==='-') s=`${shortOperandD(a,'-')}-${shortOperandD(b,'-')}`;
      else if(op==='*') s=`${shortOperandD(a,'*')}·${shortOperandD(b,'*')}`;
      else if(op==='/') s=`${shortOperandD(a,'/')}/${shortOperandD(b,'/')}`;
      else s=`${op}(${a.s},${b.s})`;
      return makeDExpr(v, s, kind, (a.ops||0)+(b.ops||0)+1, Math.max(a.depth||0,b.depth||0)+1);
    }
    function splitTopLevel(s, ops){
      let depth=0;
      for(let i=s.length-1;i>=0;i--){
        const ch=s[i];
        if(ch===')') depth++;
        else if(ch==='(') depth--;
        else if(depth===0 && ops.includes(ch)){
          if((ch==='+' || ch==='-') && i===0) continue;
          return [s.slice(0,i), ch, s.slice(i+1)];
        }
      }
      return null;
    }
    function stripOuterParens(s){
      s=String(s||'').trim();
      while(s.startsWith('(') && s.endsWith(')')){
        let d=0, ok=true;
        for(let i=0;i<s.length;i++){
          if(s[i]==='(') d++;
          else if(s[i]===')') d--;
          if(d===0 && i<s.length-1){ ok=false; break; }
        }
        if(!ok) break;
        s=s.slice(1,-1).trim();
      }
      return s;
    }
    function latexArgsInside(s, fn){
      const open=fn.length+1;
      let depth=0, parts=[''];
      for(let i=open;i<s.length-1;i++){
        const ch=s[i];
        if(ch==='(') depth++;
        else if(ch===')') depth--;
        if(ch===',' && depth===0) parts.push('');
        else parts[parts.length-1]+=ch;
      }
      return parts;
    }
    function latexTokenize(expr){
      const s=String(expr||'').replaceAll('×','·').replaceAll('*','·').replaceAll('−','-');
      const tokens=[];
      for(let i=0;i<s.length;){
        const ch=s[i];
        if(/\s/.test(ch)){ i++; continue; }
        if(/[0-9.]/.test(ch)){
          let j=i+1;
          while(j<s.length && /[0-9.]/.test(s[j])) j++;
          tokens.push({type:'num', value:s.slice(i,j)}); i=j; continue;
        }
        if(/[A-Za-zπφ]/.test(ch)){
          let j=i+1;
          while(j<s.length && /[A-Za-z0-9_πφ]/.test(s[j])) j++;
          tokens.push({type:'id', value:s.slice(i,j)}); i=j; continue;
        }
        if('()+-/^!,·'.includes(ch)){ tokens.push({type:ch, value:ch}); i++; continue; }
        tokens.push({type:'id', value:ch}); i++;
      }
      return tokens;
    }
    function parseLatexExprFromString(expr){
      const toks=latexTokenize(expr); let pos=0;
      const peek=()=>toks[pos];
      const eat=t=>{ if(peek() && peek().type===t){ pos++; return true; } return false; };
      const expect=t=>{ if(!eat(t)) throw new Error('expected '+t); };
      function parseExpression(){ return parseAddSub(); }
      function parseAddSub(){
        let node=parseMulDiv();
        while(peek() && (peek().type==='+' || peek().type==='-')){
          const op=peek().type; pos++;
          const right=parseMulDiv();
          node={type:'binary', op, left:node, right};
        }
        return node;
      }
      function parseMulDiv(){
        let node=parsePower();
        while(peek() && (peek().type==='·' || peek().type==='/')){
          const op=peek().type; pos++;
          const right=parsePower();
          node={type:'binary', op:op==='·'?'*':'/', left:node, right};
        }
        return node;
      }
      function parsePower(){
        let node=parseUnary();
        if(eat('^')) node={type:'binary', op:'^', left:node, right:parsePower()};
        return node;
      }
      function parseUnary(){
        if(eat('+')) return parseUnary();
        if(eat('-')) return {type:'unary', op:'-', arg:parseUnary()};
        return parsePostfix();
      }
      function parsePostfix(){
        let node=parsePrimary();
        while(eat('!')) node={type:'factorial', arg:node};
        return node;
      }
      function parsePrimary(){
        const t=peek(); if(!t) throw new Error('unexpected end');
        if(eat('(')){ const node=parseExpression(); expect(')'); return node; }
        if(t.type==='num'){ pos++; return {type:'num', value:t.value}; }
        if(t.type==='id'){
          pos++; const name=t.value;
          if(eat('(')){
            const args=[];
            if(!eat(')')){
              do { args.push(parseExpression()); } while(eat(','));
              expect(')');
            }
            return {type:'call', name, args};
          }
          return {type:'id', value:name};
        }
        throw new Error('unexpected token '+t.type);
      }
      const ast=parseExpression();
      if(pos!==toks.length) throw new Error('trailing tokens');
      return ast;
    }
    function astPrecedence(n){
      if(!n) return 99;
      if(n.type==='binary') return n.op==='+'||n.op==='-' ? 1 : (n.op==='*'||n.op==='/' ? 2 : 4);
      if(n.type==='unary') return 3;
      if(n.type==='factorial' || n.type==='call') return 5;
      return 6;
    }
    function latexName(name){
      const m={pi:'\\pi', π:'\\pi', phi:'\\varphi', φ:'\\varphi', theta:'\\theta', Θ:'\\Theta', alpha:'\\alpha', beta:'\\beta', gamma:'\\gamma', Gamma:'\\Gamma', zeta:'\\zeta'};
      return m[name] || escapeLatex(String(name));
    }
    function astSmallRational(node){
      if(!node) return null;
      if(node.type==='num' && /^\d+$/.test(String(node.value))) return {n:BigInt(node.value), d:1n};
      if(node.type==='unary' && node.op==='-'){ const r=astSmallRational(node.arg); return r ? {n:-r.n,d:r.d} : null; }
      if(node.type==='unary' && node.op==='+') return astSmallRational(node.arg);
      if(node.type==='binary' && node.op==='/'){
        const a=astSmallRational(node.left), b=astSmallRational(node.right);
        if(!a || !b || b.n===0n) return null;
        let n=a.n*b.d, d=a.d*b.n; if(d<0n){ n=-n; d=-d; }
        const g=gcdBig(n<0n?-n:n,d); return {n:n/g,d:d/g};
      }
      return null;
    }
    function astRationalEquals(node,n,d=1n){ const r=astSmallRational(node); return !!r && r.n===BigInt(n) && r.d===BigInt(d); }
    function needsParensForChild(child, parentOp, side){
      if(!child || child.type!=='binary') return false;
      const cp=astPrecedence(child);
      const pp=parentOp==='+'||parentOp==='-' ? 1 : (parentOp==='*'||parentOp==='/' ? 2 : 4);
      if(cp<pp) return true;
      if(parentOp==='-' && side==='right' && cp===1) return true;
      if(parentOp==='/' && side==='right' && cp<=2) return true;
      if(parentOp==='^' && side==='left') return true;
      return false;
    }
    function astToLatex(node, parentOp=null, side=null){
      if(!node) return '';
      let out='';
      if(node.type==='num') out=node.value;
      else if(node.type==='id') out=latexName(node.value);
      else if(node.type==='unary'){
        const inner=astToLatex(node.arg, 'unary', 'right');
        out='-'+(astPrecedence(node.arg)<=1 ? `\\left(${inner}\\right)` : inner);
      }else if(node.type==='factorial'){
        const inner=astToLatex(node.arg, '!', 'left');
        out=(astPrecedence(node.arg)<5 ? `\\left(${inner}\\right)` : inner)+'!';
      }else if(node.type==='call'){
        const name=String(node.name).toLowerCase();
        if(name==='floor' && node.args.length===1) out=`\\left\\lfloor ${astToLatex(node.args[0])} \\right\\rfloor`;
        else if(name==='ceil' && node.args.length===1) out=`\\left\\lceil ${astToLatex(node.args[0])} \\right\\rceil`;
        else if((name==='sqrt' || name==='surd') && node.args.length===1) out=`\\sqrt{${astToLatex(node.args[0])}}`;
        else if(name==='cbrt' && node.args.length===1) out=`\\sqrt[3]{${astToLatex(node.args[0])}}`;
        else if((name==='abs' || name==='fabs') && node.args.length===1) out=`\\left|${astToLatex(node.args[0])}\\right|`;
        else if(name==='round' && node.args.length===1) out=`\\operatorname{round}\\!\\left(${astToLatex(node.args[0])}\\right)`;
        else if(name==='binom' && node.args.length===2) out=`\\binom{${astToLatex(node.args[0])}}{${astToLatex(node.args[1])}}`;
        else if(['sin','cos','tan','arctan','asin','acos','atan','sinh','cosh','tanh','log','ln','exp'].includes(name) && node.args.length===1){
          const cmd=name==='ln' ? 'log' : name;
          out=`\\${cmd}\\!\\left(${astToLatex(node.args[0])}\\right)`;
        }
        else if(['gamma','zeta'].includes(name) && node.args.length===1){
          const cmd=name==='gamma' ? 'Gamma' : name;
          out=`\\${cmd}\\!\\left(${astToLatex(node.args[0])}\\right)`;
        }
        else out=`\\operatorname{${escapeLatex(node.name)}}\\!\\left(${node.args.map(a=>astToLatex(a)).join(',')}\\right)`;
      }else if(node.type==='binary'){
        if(node.op==='/'){
          out=`\\frac{${astToLatex(node.left)}}{${astToLatex(node.right)}}`;
        }else if(node.op==='^'){
          const base=astToLatex(node.left);
          const exp=astToLatex(node.right);
          const baseOut=astPrecedence(node.left)<5 ? `\\left(${base}\\right)` : base;
          if(astRationalEquals(node.right,0n)) out='1';
          else if(astRationalEquals(node.right,1n)) out=baseOut;
          else if(astRationalEquals(node.right,1n,2n)) out=`\\sqrt{${baseOut}}`;
          else if(astRationalEquals(node.right,-1n,2n)) out=`\\frac{1}{\\sqrt{${baseOut}}}`;
          else out=`{${baseOut}}^{${exp}}`;
        }else{
          let L=astToLatex(node.left);
          let R=astToLatex(node.right);
          if(needsParensForChild(node.left,node.op,'left')) L=`\\left(${L}\\right)`;
          if(needsParensForChild(node.right,node.op,'right')) R=`\\left(${R}\\right)`;
          const op = node.op==='*' ? '\\cdot ' : node.op;
          out=`${L}${op}${R}`;
        }
      }
      if(parentOp && node.type==='binary' && needsParensForChild(node,parentOp,side)) return `\\left(${out}\\right)`;
      return out;
    }
    function exprToLatex(expr){
      try{
        const ast=parseLatexExprFromString(String(expr||''));
        return sanitizeLatexForDisplay(astToLatex(ast));
      }catch(e){
        let s=stripOuterParens(String(expr||''));
        if(!s) return '';
        return sanitizeLatexForDisplay(s.replaceAll('·','\\cdot ').replaceAll('*','\\cdot ').replace(/\bpi\b/g,'\\pi'));
      }
    }

    // v10.3: exact display-expression validator for integer-mode rows.
    // It evaluates the *shown* expression text (after any pretty simplification)
    // as a rational BigInt expression, so a row is never displayed as
    // "exact = target" unless the visible formula really equals the target.
    function ratNorm(num, den=1n){
      num=BigInt(num); den=BigInt(den);
      if(den===0n) return null;
      if(den<0n){ num=-num; den=-den; }
      const g=gcdBig(num,den);
      return {num:num/g, den:den/g};
    }
    function ratAdd(a,b){ return a&&b ? ratNorm(a.num*b.den + b.num*a.den, a.den*b.den) : null; }
    function ratSub(a,b){ return a&&b ? ratNorm(a.num*b.den - b.num*a.den, a.den*b.den) : null; }
    function ratMul(a,b){ return a&&b ? ratNorm(a.num*b.num, a.den*b.den) : null; }
    function ratDiv(a,b){ return a&&b&&b.num!==0n ? ratNorm(a.num*b.den, a.den*b.num) : null; }
    function ratAbsTooLarge(r, capAbs){
      if(!r || capAbs===null || capAbs===undefined) return false;
      const cap=BigInt(capAbs);
      return absBig(r.num) > cap * r.den;
    }
    function powBigCappedAbs(base, exp, capAbs){
      base=BigInt(base); exp=BigInt(exp);
      if(exp<0n) return null;
      let r=1n, b=base;
      const cap=capAbs===null||capAbs===undefined ? null : BigInt(capAbs);
      while(exp>0n){
        if(exp&1n){ r*=b; if(cap!==null && absBig(r)>cap) return null; }
        exp >>= 1n;
        if(exp){ b*=b; if(cap!==null && absBig(b)>cap) return null; }
      }
      return r;
    }
    function ratPowInt(a, e, capAbs){
      if(!a || a.den===0n || !e || e.den!==1n) return null;
      const exp=e.num;
      if(exp<0n || exp>10000n) return null;
      const n=powBigCappedAbs(a.num, exp, capAbs);
      const d=powBigCappedAbs(a.den, exp, capAbs);
      if(n===null || d===null) return null;
      return ratNorm(n,d);
    }
    function factorialCappedRat(a, capAbs){
      if(!a || a.den!==1n || a.num<0n || a.num>5000n) return null;
      let r=1n;
      const cap=capAbs===null||capAbs===undefined ? null : BigInt(capAbs);
      for(let i=2n;i<=a.num;i++){ r*=i; if(cap!==null && r>cap) return null; }
      return ratNorm(r,1n);
    }
    function binomCappedRat(nr, kr, capAbs){
      if(!nr || !kr || nr.den!==1n || kr.den!==1n) return null;
      let n=nr.num, k=kr.num;
      if(n<0n || k<0n || k>n || n>10000n) return null;
      if(k>n-k) k=n-k;
      let r=1n;
      const cap=capAbs===null||capAbs===undefined ? null : BigInt(capAbs);
      for(let i=1n;i<=k;i++){
        r=(r*(n-k+i))/i;
        if(cap!==null && r>cap) return null;
      }
      return ratNorm(r,1n);
    }
    function floorRatToBig(r){ if(!r) return null; return floorDiv(r.num,r.den); }
    function ceilRatToBig(r){ if(!r) return null; return ceilDiv(r.num,r.den); }
    function roundRatToBig(r){ if(!r) return null; return roundDiv(r.num,r.den); }
    function evalDisplayAstRat(node, capAbs){
      if(!node) return null;
      let out=null;
      if(node.type==='num'){
        if(!/^\d+$/.test(String(node.value))) return null;
        out=ratNorm(BigInt(node.value),1n);
      }else if(node.type==='id'){
        return null;
      }else if(node.type==='unary'){
        const a=evalDisplayAstRat(node.arg, capAbs); out=a ? ratNorm(-a.num,a.den) : null;
      }else if(node.type==='factorial'){
        out=factorialCappedRat(evalDisplayAstRat(node.arg, capAbs), capAbs);
      }else if(node.type==='call'){
        const name=String(node.name).toLowerCase();
        if((name==='floor' || name==='ceil' || name==='round') && node.args.length===1){
          const a=evalDisplayAstRat(node.args[0], capAbs);
          const v=name==='floor' ? floorRatToBig(a) : (name==='ceil' ? ceilRatToBig(a) : roundRatToBig(a));
          out=(v===null) ? null : ratNorm(v,1n);
        }else if(name==='binom' && node.args.length===2){
          out=binomCappedRat(evalDisplayAstRat(node.args[0], capAbs), evalDisplayAstRat(node.args[1], capAbs), capAbs);
        }else{
          return null;
        }
      }else if(node.type==='binary'){
        const L=evalDisplayAstRat(node.left, capAbs);
        const R=evalDisplayAstRat(node.right, capAbs);
        if(node.op==='+') out=ratAdd(L,R);
        else if(node.op==='-') out=ratSub(L,R);
        else if(node.op==='*') out=ratMul(L,R);
        else if(node.op==='/') out=ratDiv(L,R);
        else if(node.op==='^') out=ratPowInt(L,R,capAbs);
        else return null;
      }
      if(ratAbsTooLarge(out, capAbs)) return null;
      return out;
    }
    function integerDisplayValidationCap(target){
      const t=absBig(BigInt(target));
      const base=t>0n ? t : 1n;
      // Allow rational templates whose numerator is much larger than the final
      // integer (for example ((5!/7)^7)/7!), but still cap far above the local
      // search envelope to avoid accidental runaway exponentiation.
      return base*1000000000000000000000000n + 1000000000000000000000000000000000000n;
    }
    function exactIntegerValueFromDisplay(expr, capAbs=null){
      try{
        const ast=parseLatexExprFromString(String(expr||''));
        const r=evalDisplayAstRat(ast, capAbs);
        if(!r || r.den!==1n) return null;
        return r.num;
      }catch(e){ return null; }
    }
    function displayExprMatchesTarget(expr, target){
      const cap=integerDisplayValidationCap(target);
      const v=exactIntegerValueFromDisplay(expr, cap);
      return v!==null && v===BigInt(target);
    }
    function validateDExprForTarget(expr, target){
      if(!expr || expr.v!==BigInt(target)) return null;
      const original={...expr, s:normalizeShortDisplay(expr.s)};
      const simplified=simplifyDExprIfBetter(original);
      const candidates=[];
      if(simplified && simplified.s) candidates.push(simplified);
      candidates.push(original);
      const seenLocal=new Set();
      for(const cand of candidates){
        cand.s=normalizeShortDisplay(cand.s);
        if(seenLocal.has(cand.s)) continue;
        seenLocal.add(cand.s);
        if(displayExprMatchesTarget(cand.s, target)){
          cand.digits=digitCountExpr(cand.s);
          cand.rank=shortRank(cand);
          return cand;
        }
      }
      return null;
    }
    function powBigCapped(a,e,cap=null){
      a=BigInt(a); e=BigInt(e); if(e<0n) return null;
      let r=1n;
      while(e>0n){
        if(e&1n){ r*=a; if(cap!==null && r>cap) return null; }
        e >>= 1n;
        if(e){ a*=a; if(cap!==null && a>cap) a=cap+1n; }
      }
      return r;
    }
    function binomBigCapped(n,k,cap=null){
      n=BigInt(n); k=BigInt(k); if(n<0n || k<0n || k>n) return null;
      if(k>n-k) k=n-k;
      let r=1n;
      for(let i=1n;i<=k;i++){
        r=(r*(n-k+i))/i;
        if(cap!==null && r>cap) return null;
      }
      return r;
    }
    function digitSearchConfig(effort, target){
      effort=Math.max(0, Math.min(7, Number(effort)||0));
      const td=decimalDigitCountBig(target);
      const ceiling=[6,7,8,10,12,14,16,19][effort];
      const minInteresting = td<=3 ? td : Math.min(td-1, Math.ceil(td*0.62));
      const maxDigits = Math.max(2, Math.min(td, Math.max(ceiling, minInteresting)));
      return {
        effort,
        timeMs: Math.min(128000, 1000*Math.pow(2, effort)),
        maxDigits,
        literalCap: [420,900,1800,5200,12000,26000,56000,105000][effort],
        argCap: [140,220,360,620,1000,1650,2700,4300][effort],
        maxFactN: [36,48,66,92,125,170,230,310][effort],
        maxPowBase: [65,95,150,240,400,700,1100,1700][effort],
        maxPowExp: [90,150,250,420,680,1050,1600,2400][effort],
        maxBinomN: [140,210,340,560,850,1250,1800,2600][effort],
        maxBinomK: [24,34,48,68,92,125,165,220][effort],
        pairProbe: [2600,5200,9800,19000,34000,58000,92000,145000][effort],
        dbSoftLimit: [16000,32000,68000,135000,240000,390000,620000,920000][effort],
        denomProbe: [900,1700,3400,7200,12500,20500,33000,52000][effort],
        residualProbe: [70,130,240,430,720,1100,1700,2500][effort],
        reverseProbe: [500,1000,2100,4300,7800,12500,19000,29000][effort],
        boundDigits: Math.min(150, td + [16,20,28,38,52,70,92,118][effort])
      };
    }
    function tuneIntegerConfigForDigitBudget(cfg, target){
      // v9.4: low digit budgets should only test genuinely compact forms.
      // Huge denominator/reverse pools at budget 2-4 were the main cause of
      // apparent stalls such as 9169995354 effort 4: they spend seconds proving
      // that large fractions cannot possibly beat a tiny exact expression.  The
      // families remain enabled, but each pool is capped to values that can still
      // fit the requested digit budget.
      const b=Number(cfg.maxDigits||0);
      if(b<=3){
        cfg.literalCap=Math.min(cfg.literalCap, 999);
        cfg.argCap=Math.min(cfg.argCap, 260);
        cfg.pairProbe=Math.min(cfg.pairProbe, 900);
        cfg.dbSoftLimit=Math.min(cfg.dbSoftLimit, 5500);
        cfg.denomProbe=Math.min(cfg.denomProbe, 180);
        cfg.residualProbe=Math.min(cfg.residualProbe, 28);
        cfg.reverseProbe=Math.min(cfg.reverseProbe, 220);
        cfg.maxPowBase=Math.min(cfg.maxPowBase, 180);
        cfg.maxPowExp=Math.min(cfg.maxPowExp, 260);
        cfg.maxBinomN=Math.min(cfg.maxBinomN, 220);
        cfg.maxBinomK=Math.min(cfg.maxBinomK, 36);
      }else if(b<=4){
        cfg.literalCap=Math.min(cfg.literalCap, 2200);
        cfg.argCap=Math.min(cfg.argCap, 420);
        cfg.pairProbe=Math.min(cfg.pairProbe, 2600);
        cfg.dbSoftLimit=Math.min(cfg.dbSoftLimit, 13000);
        cfg.denomProbe=Math.min(cfg.denomProbe, 520);
        cfg.residualProbe=Math.min(cfg.residualProbe, 56);
        cfg.reverseProbe=Math.min(cfg.reverseProbe, 680);
      }
      return cfg;
    }
    function uniqueExprsByValue(list, maxCount=Infinity){
      const map=new Map();
      for(const e of list||[]){
        if(!e) continue;
        const k=e.v.toString();
        const old=map.get(k);
        if(!old || cmpExpr(e,old)<0) map.set(k,e);
      }
      return [...map.values()].sort(cmpExpr).slice(0, maxCount);
    }
    function addBest(map, e, cfg){
      if(!e || e.v<0n || e.digits>cfg.maxDigits) return false;
      const k=e.v.toString();
      const old=map.get(k);
      if(!old || cmpExpr(e,old)<0){ map.set(k,e); return true; }
      return false;
    }
    function buildDigitSearchDB(target, settings, cfg, deadline){
      const bound=(10n**BigInt(cfg.boundDigits))-1n;
      const byValue=new Map();
      const argMap=new Map();
      function add(e){ if(shortAbort(deadline) || !e || e.v>bound) return false; return addBest(byValue,e,cfg); }
      function addArg(e){ if(!e || e.v<0n || e.v>BigInt(cfg.argCap) || e.digits>cfg.maxDigits) return; const k=e.v.toString(); const old=argMap.get(k); if(!old || cmpExpr(e,old)<0) argMap.set(k,e); }
      for(let i=0;i<=cfg.literalCap && performance.now()<deadline;i++){
        const e=makeDExpr(BigInt(i), String(i), 'literal');
        add(e); addArg(e);
      }
      let changed=true;
      for(let pass=0; pass<3 && changed && performance.now()<deadline; pass++){
        changed=false;
        const args=[...argMap.values()].sort(cmpExpr).slice(0, Math.min(argMap.size, cfg.reverseProbe));
        for(const a of args){
          if(shortAbort(deadline)) break;
          if(a.v>=2n && a.v<=BigInt(cfg.maxFactN)){
            const v=powFactorialBig(a.v, bound);
            if(v!==null){ const e=makeDExpr(v, `${shortOperandD(a,'^')}!`, 'factorial', (a.ops||0)+1, (a.depth||0)+1); if(add(e)) changed=true; addArg(e); }
          }
        }
        const nowArgs=[...argMap.values()].sort(cmpExpr).slice(0, Math.min(argMap.size, cfg.reverseProbe));
        const baseSources=nowArgs.filter(e=>e.v>=2n && e.v<=BigInt(cfg.maxPowBase));
        const expSources=nowArgs.filter(e=>e.v>=2n && e.v<=BigInt(cfg.maxPowExp));
        for(const a of baseSources){
          if(shortAbort(deadline)) break;
          for(const b of expSources){
            if(shortAbort(deadline)) break;
            if(a.digits+b.digits>cfg.maxDigits) continue;
            if(b.v>1500n || (a.v>100n && b.v>260n) || (a.v>30n && b.v>700n)) continue;
            const v=powBigCapped(a.v,b.v,bound);
            if(v!==null){ const e=makeDExpr(v, `${shortOperandD(a,'^')}^${shortOperandD(b,'^')}`, 'power', (a.ops||0)+(b.ops||0)+1, Math.max(a.depth||0,b.depth||0)+1); if(add(e)) changed=true; addArg(e); }
          }
        }
        const nSources=nowArgs.filter(e=>e.v>=2n && e.v<=BigInt(cfg.maxBinomN));
        const kSources=nowArgs.filter(e=>e.v>=0n && e.v<=BigInt(cfg.maxBinomK));
        for(const n of nSources){
          if(shortAbort(deadline)) break;
          for(const k of kSources){
            if(shortAbort(deadline)) break;
            if(k.v>n.v/2n || n.digits+k.digits>cfg.maxDigits) continue;
            const v=binomBigCapped(n.v,k.v,bound);
            if(v!==null){ const e=makeDExpr(v, `binom(${n.s},${k.s})`, 'binomial', (n.ops||0)+(k.ops||0)+1, Math.max(n.depth||0,k.depth||0)+1); if(add(e)) changed=true; addArg(e); }
          }
        }
      }
      // Combine promising expressions. This is intentionally normalized and budgeted: it borrows the Countdown solver idea of
      // generating expressions once, avoiding commutative duplicates, and pruning identities like x+0, x*1.
      for(let pass=0; pass<2 && performance.now()<deadline; pass++){
        const pool=[...byValue.values()].sort(cmpExpr).slice(0, Math.min(cfg.pairProbe, byValue.size));
        let made=0;
        for(let i=0;i<pool.length && performance.now()<deadline;i++){
          const a=pool[i];
          for(let j=i;j<pool.length && performance.now()<deadline;j++){
            const b=pool[j];
            if(a.digits+b.digits>cfg.maxDigits) continue;
            if(a.v===0n && (b.v===0n || b.v===1n)) continue;
            if(b.v===0n && a.v===0n) continue;
            if(a.v+b.v<=bound){ if(add(combineD(a,'+',b,a.v+b.v,'sum'))) made++; }
            if(a.v>=b.v && b.v!==0n && a.v-b.v<=bound){ if(add(combineD(a,'-',b,a.v-b.v,'difference'))) made++; }
            if(a.v!==0n && b.v!==0n && a.v!==1n && b.v!==1n){
              const prod=a.v*b.v; if(prod<=bound){ if(add(combineD(a,'*',b,prod,'product'))) made++; }
            }
            if(b.v!==0n && b.v!==1n && a.v%b.v===0n){ if(add(combineD(a,'/',b,a.v/b.v,'quotient'))) made++; }
            if(a.v!==b.v && a.v!==0n && a.v!==1n && b.v!==0n && a.v!==1n && a.v<=bound && a.v!==b.v && a.v%b.v===0n){ /* handled by order if encountered */ }
            if(byValue.size>cfg.dbSoftLimit && made>cfg.dbSoftLimit*0.25) break;
          }
        }
      }
      const atoms=[...byValue.values()].filter(e=>e.v>=0n && e.v<=bound).sort(cmpExpr);
      const atomsByValue=[...byValue.values()].filter(e=>e.v>=0n && e.v<=bound).sort((a,b)=>a.v<b.v?-1:a.v>b.v?1:cmpExpr(a,b));
      return {byValue, atoms, atomsByValue, argSources:[...argMap.values()].sort(cmpExpr), bound, cfg};
    }
    function powFactorialBig(n, cap=null){
      n=BigInt(n); if(n<0n) return null;
      let r=1n;
      for(let i=2n;i<=n;i++){ r*=i; if(cap!==null && r>cap) return null; }
      return r;
    }
    let tinyPrettyDBCache = null;
    function cloneDExpr(e){
      const out=makeDExpr(e.v, e.s, e.kind || 'tiny-pretty', e.ops || 0, e.depth || 0);
      out.rank=e.rank; out.digits=e.digits; return out;
    }
    function getTinyPrettyDB(deadline=performance.now()+180){
      if(tinyPrettyDBCache) return tinyPrettyDBCache;
      const cap=100000n;
      const byValue=new Map();
      function add(e){
        if(!e || e.v<0n || e.v>cap) return;
        e.s=normalizeShortDisplay(e.s); e.digits=digitCountExpr(e.s); e.rank=shortRank(e);
        if(e.digits>5) return;
        const k=e.v.toString(), old=byValue.get(k);
        if(!old || cmpExpr(e,old)<0) byValue.set(k,e);
      }
      for(let i=0;i<=5000 && performance.now()<deadline;i++) add(makeDExpr(BigInt(i), String(i), 'tiny-literal'));
      for(let a=2;a<=99 && performance.now()<deadline;a++){
        let v=BigInt(a);
        for(let b=1;b<=99;b++){
          if(v>cap) break;
          if(String(a).length+String(b).length<=5) add(makeDExpr(v, b===1?String(a):`${a}^${b}`, b===1?'tiny-literal':'tiny-power', b===1?0:1, b===1?0:1));
          v*=BigInt(a);
        }
      }
      let f=1n;
      for(let n=1;n<=12 && performance.now()<deadline;n++){
        f*=BigInt(n);
        if(String(n).length<=2) add(makeDExpr(f, `${n}!`, 'tiny-factorial', 1, 1));
      }
      for(let n=2;n<=99 && performance.now()<deadline;n++){
        for(let k=1;k<=Math.floor(n/2);k++){
          if(String(n).length+String(k).length>5) continue;
          const v=binomBigCapped(BigInt(n), BigInt(k), cap); if(v===null) break;
          add(makeDExpr(v, `binom(${n},${k})`, 'tiny-binom', 1, 1));
        }
      }
      for(let pass=0;pass<2 && performance.now()<deadline;pass++){
        const pool=[...byValue.values()].filter(e=>e.digits<=4).sort(cmpExpr).slice(0,900);
        for(let i=0;i<pool.length && performance.now()<deadline;i++){
          const a=pool[i];
          for(let j=i;j<pool.length && performance.now()<deadline;j++){
            const b=pool[j];
            if(a.digits+b.digits>5) continue;
            add(combineD(a,'+',b,a.v+b.v,'tiny-sum'));
            if(a.v>=b.v) add(combineD(a,'-',b,a.v-b.v,'tiny-diff'));
            if(a.v>1n && b.v>1n && a.v*b.v<=cap) add(combineD(a,'*',b,a.v*b.v,'tiny-product'));
            if(b.v>1n && a.v%b.v===0n) add(combineD(a,'/',b,a.v/b.v,'tiny-quotient'));
          }
        }
      }
      tinyPrettyDBCache=byValue;
      return byValue;
    }
    function compactLiteralD(n, opts={}){
      n=BigInt(n);
      const absn=absBig(n);
      const denominator=!!opts.denominator;
      const literalDigits=decimalDigitCountBig(absn);
      function allowed(e){
        if(!e) return false;
        // Denominators should stay clean: no division and no rounding/floor/ceil.
        // Offsets/constants are allowed to use the full precomputed database.
        if(denominator && /round\(|floor\(|ceil\(|\//.test(e.s)) return false;
        return true;
      }
      if(absn<=100000n){
        let best=null;
        function consider(e){
          if(!allowed(e)) return;
          const expressive=/[!^+\-·/]|binom\(|floor\(|ceil\(/.test(e.s);
          if(e.digits<=5 && (e.digits<literalDigits || (literalDigits>=3 && expressive && e.digits<=literalDigits))){
            if(!best || cmpExpr(e,best)<0) best=cloneDExpr(e);
          }
        }
        const key=absn.toString();
        if(window.RIES_SHORTFORM_100K_PACKED && typeof window.RIES_SHORTFORM_100K_PACKED.get==='function'){
          for(const expr of window.RIES_SHORTFORM_100K_PACKED.get(key).slice(0,8)) consider(makeDExpr(absn, expr, 'precomputed-literal', 1, 1));
        }
        const multi=window.RIES_SHORTFORM_100K_MULTI && (window.RIES_SHORTFORM_100K_MULTI[key] || window.RIES_SHORTFORM_100K_MULTI[Number(absn)]);
        if(Array.isArray(multi)){ for(const expr of multi.slice(0,8)) consider(makeDExpr(absn, expr, 'precomputed-literal', 1, 1)); }
        const single=window.RIES_SHORTFORM_100K && (window.RIES_SHORTFORM_100K[key] || window.RIES_SHORTFORM_100K[Number(absn)]);
        if(single) consider(makeDExpr(absn, single, 'precomputed-literal', 1, 1));
        consider(getTinyPrettyDB().get(key));
        if(best) return best;
        // Avoid recursive medium-offset searches for very small constants and
        // denominators. They are rarely prettier than the literal, and recursive
        // calls here can explode when many fallback denominators are tested.
        if(denominator || literalDigits<=3) return makeDExpr(absn, absn.toString(), 'literal');
      }
      // v6.8: slightly recursive prettification for six-digit constants used in
      // fallback ratios, without allowing ugly decimal-split A*10^B+C output.
      if(absn<=999999n){
        let best=null;
        const deadline=performance.now()+7;
        const candidates=[];
        function push(e){ if(!allowed(e) || e.v!==absn) return; if(!best || cmpExpr(e,best)<0) best=e; }
        for(let a=2;a<=999 && performance.now()<deadline;a++){
          let v=BigInt(a);
          for(let b=1;b<=8;b++){
            if(v>absn+100000n) break;
            const pe=makeDExpr(v, b===1?String(a):`${a}^${b}`, b===1?'literal':'medium-power', b===1?0:1, b===1?0:1);
            candidates.push(pe); if(v===absn) push(pe);
            v*=BigInt(a);
          }
        }
        let f=1n;
        for(let a=1;a<=12 && performance.now()<deadline;a++){ f*=BigInt(a); const fe=makeDExpr(f, `${a}!`, 'medium-factorial', 1, 1); candidates.push(fe); if(f===absn) push(fe); }
        const offCap=100000n;
        for(const c of candidates.sort(cmpExpr).slice(0,1600)){
          if(performance.now()>deadline) break;
          const off=absn-c.v;
          if(off!==0n && absBig(off)<=offCap){
            const oe=compactLiteralD(absBig(off), {denominator:false});
            let e=off>0n ? combineD(c,'+',oe,absn,'medium-offset') : combineD(c,'-',oe,absn,'medium-offset');
            if(e.digits<=literalDigits && (!denominator || !/[\/]|round\(|floor\(|ceil\(/.test(e.s))) push(e);
          }
        }
        if(best && best.digits<=literalDigits) return cloneDExpr(best);
      }
      return makeDExpr(absn, absn.toString(), 'literal');
    }

    function lowerBoundExprByValue(arr, v){ let lo=0, hi=arr.length; while(lo<hi){ const mid=(lo+hi)>>1; if(arr[mid].v<v) lo=mid+1; else hi=mid; } return lo; }
    function bestValueExpr(db, v){ return db.byValue.get(BigInt(v).toString()) || null; }
    function nearValueExprs(db, lo, hi, maxCount=12){
      const arr=db.atomsByValue; const out=[];
      let idx=lowerBoundExprByValue(arr,lo);
      while(idx<arr.length && arr[idx].v<=hi && out.length<maxCount){ out.push(arr[idx]); idx++; }
      return out.sort(cmpExpr);
    }
    function residualPool(db,cfg,target){
      const out=[]; const seen=new Set();
      function push(e){ if(!e || e.v<=0n || e.digits>cfg.maxDigits) return; const k=e.v.toString(); if(seen.has(k)) return; seen.add(k); out.push(e); }
      for(let i=1;i<=80;i++) push(makeDExpr(BigInt(i), String(i), 'literal'));
      for(const e of db.atoms){
        if(out.length>=cfg.residualProbe) break;
        if(e.v>0n && e.v<=target*20n && e.kind!=='literal') push(e);
      }
      return out.sort(cmpExpr).slice(0,cfg.residualProbe);
    }
    function addDigitCandidate(rows, seen, expr, target, cfg, label='shortform'){
      if(!expr || expr.v!==target || expr.ops<1) return;
      expr=validateDExprForTarget(expr, target);
      if(!expr) return;
      const targetDigits=decimalDigitCountBig(target);
      if(expr.digits>cfg.maxDigits) return;
      const special=/round\(|floor\(|ceil\(|binom\(|!|\^|·|\//.test(expr.s);
      if(!(expr.digits<targetDigits || (targetDigits<=3 && expr.digits<=targetDigits && special))) return;
      const row={candidate:`${label}: ${expr.s}`, latex:exprToLatex(expr.s), value:`exact = ${shortPrettyValue(target)}`, copyValue:target.toString(), err:0, hideError:true, beauty:expr.rank, feature:exprFeature(expr.s), digits:expr.digits, ops:expr.ops};
      const key=candidateEquivalenceKey(row);
      if(seen.has(key)) return;
      seen.add(key);
      rows.push(row);
    }
    function pairTargetSearch(rows,seen,target,db,cfg,deadline){
      const pool=db.atoms.filter(e=>e.v>=0n && e.digits<=cfg.maxDigits).sort(cmpExpr).slice(0, Math.min(cfg.pairProbe, db.atoms.length));
      const localSeen=new Set();
      for(const a of pool){
        if(shortAbort(deadline)) return;
        // target = a + b, searched directly instead of relying on the database to have built the pair.
        if(target>=a.v){
          const b=bestValueExpr(db,target-a.v);
          if(b && a.digits+b.digits<=cfg.maxDigits){
            const key=`+|${a.v< b.v ? a.v+','+b.v : b.v+','+a.v}`;
            if(!localSeen.has(key)){ localSeen.add(key); addDigitCandidate(rows,seen,combineD(a,'+',b,target,'sum'),target,cfg,'pair exact'); }
          }
        }
        // target = a - b.
        if(a.v>=target){
          const b=bestValueExpr(db,a.v-target);
          if(b && a.digits+b.digits<=cfg.maxDigits){
            const key=`-|${a.v},${b.v}`;
            if(!localSeen.has(key)){ localSeen.add(key); addDigitCandidate(rows,seen,combineD(a,'-',b,target,'difference'),target,cfg,'pair exact'); }
          }
        }
        // target = a * b.
        if(a.v>1n && target%a.v===0n){
          const b=bestValueExpr(db,target/a.v);
          if(b && a.digits+b.digits<=cfg.maxDigits){
            const key=`*|${a.v< b.v ? a.v+','+b.v : b.v+','+a.v}`;
            if(!localSeen.has(key)){ localSeen.add(key); addDigitCandidate(rows,seen,combineD(a,'*',b,target,'product'),target,cfg,'pair exact'); }
          }
        }
        // target = a / b.
        if(a.v>=target && target!==0n && a.v%target===0n){
          const b=bestValueExpr(db,a.v/target);
          if(b && b.v>1n && a.digits+b.digits<=cfg.maxDigits){
            const key=`/|${a.v},${b.v}`;
            if(!localSeen.has(key)){ localSeen.add(key); addDigitCandidate(rows,seen,combineD(a,'/',b,target,'quotient'),target,cfg,'pair exact'); }
          }
        }
      }
    }
    function directAndReverseSearch(rows,seen,target,db,cfg,deadline){
      const direct=bestValueExpr(db,target);
      if(direct) addDigitCandidate(rows,seen,direct,target,cfg,'direct exact');
      pairTargetSearch(rows,seen,target,db,cfg,Math.min(deadline, performance.now()+Math.max(90, cfg.effort*95+160)));
      const residualCap=cfg.maxDigits<=3 ? 28 : (cfg.maxDigits<=4 ? 56 : cfg.residualProbe);
      const residuals=residualPool(db,cfg,target).slice(0,residualCap);
      for(const r of residuals){
        if(shortAbort(deadline)) return;
        if(target>=r.v){ const a=bestValueExpr(db,target-r.v); if(a) addDigitCandidate(rows,seen,combineD(a,'+',r,target,'sum'),target,cfg,'offset exact'); }
        const a2=bestValueExpr(db,target+r.v); if(a2) addDigitCandidate(rows,seen,combineD(a2,'-',r,target,'offset exact'),target,cfg,'offset exact');
      }
      const divisorCap=cfg.maxDigits<=3 ? 180 : (cfg.maxDigits<=4 ? 520 : cfg.reverseProbe);
      const divisors=uniqueExprsByValue(db.atoms.filter(e=>e.v>1n && e.digits<=cfg.maxDigits-1), divisorCap);
      for(const d of divisors){
        if(shortAbort(deadline) || isUserInputPending()) return;
        if(target%d.v===0n){ const q=bestValueExpr(db,target/d.v); if(q) addDigitCandidate(rows,seen,combineD(q,'*',d,target,'product'),target,cfg,'product exact'); }
        // reverse with small offsets: target = q*d +/- r
        for(const r of residuals.slice(0, Math.min(cfg.maxDigits<=3?18:90,cfg.residualProbe))){
          if(shortAbort(deadline) || isUserInputPending()) return;
          if(target>=r.v && (target-r.v)%d.v===0n){ const q=bestValueExpr(db,(target-r.v)/d.v); if(q) addDigitCandidate(rows,seen,combineD(combineD(q,'*',d,target-r.v,'product'),'+',r,target,'compound'),target,cfg,'reverse exact'); }
          if((target+r.v)%d.v===0n){ const q=bestValueExpr(db,(target+r.v)/d.v); if(q) addDigitCandidate(rows,seen,combineD(combineD(q,'*',d,target+r.v,'product'),'-',r,target,'compound'),target,cfg,'reverse exact'); }
        }
      }
    }
    function pairExprsInRange(db, lo, hi, digitBudget, cfg, deadline, maxCount=10){
      if(hi<lo || digitBudget<2) return [];
      const out=[]; const seen=new Set();
      const basePool=db.atoms.filter(e=>e.v>=0n && e.digits<digitBudget);
      const poolMap=new Map();
      for(const e of basePool.sort(cmpExpr).slice(0, Math.min(basePool.length, Math.max(500, cfg.reverseProbe)))) poolMap.set(e.v.toString()+e.s,e);
      // Also keep large, low-digit structured values. They are often the missing numerator side in floor/ceil/round(A/B).
      for(const e of basePool.filter(x=>x.kind!=='literal' || x.v>10000n).sort((a,b)=>a.v>b.v?-1:a.v<b.v?1:cmpExpr(a,b)).slice(0, Math.min(basePool.length, Math.max(500, cfg.reverseProbe)))) poolMap.set(e.v.toString()+e.s,e);
      const rawPool=[...poolMap.values()];
      const pool=cfg.maxDigits<=3 ? rawPool.sort(cmpExpr).slice(0,420) : rawPool;
      function push(e){
        if(!e || e.v<lo || e.v>hi || e.digits>digitBudget) return;
        const key=e.v.toString()+'|'+canonicalExpressionKey(e.s);
        if(seen.has(key)) return;
        seen.add(key); out.push(e);
      }
      for(const a of pool){
        if(shortAbort(deadline) || out.length>=maxCount) break;
        const rem=digitBudget-a.digits;
        if(rem<=0) continue;
        // a + b in [lo, hi]
        const lb=lo-a.v, hb=hi-a.v;
        if(hb>=0n){
          for(const b of nearValueExprs(db, lb<0n?0n:lb, hb, 3)){
            if(b.digits<=rem) push(combineD(a,'+',b,a.v+b.v,'range-sum'));
          }
        }
        // a - b in [lo, hi]
        if(a.v>=lo){
          const lb2=a.v-hi, hb2=a.v-lo;
          if(hb2>=0n){
            for(const b of nearValueExprs(db, lb2<0n?0n:lb2, hb2, 3)){
              if(b.digits<=rem) push(combineD(a,'-',b,a.v-b.v,'range-diff'));
            }
          }
        }
        // b - a in [lo, hi]
        for(const b of nearValueExprs(db, lo+a.v, hi+a.v, 3)){
          if(b.digits<=rem && b.v>=a.v) push(combineD(b,'-',a,b.v-a.v,'range-diff'));
        }
      }
      return out.sort(cmpExpr).slice(0,maxCount);
    }
    function ratioSearch(rows,seen,target,db,cfg,deadline){
      const denomCap=cfg.maxDigits<=3 ? 160 : (cfg.maxDigits<=4 ? 460 : cfg.denomProbe);
      const residualCap=cfg.maxDigits<=3 ? 24 : (cfg.maxDigits<=4 ? 48 : Math.min(160,cfg.residualProbe));
      const denoms=uniqueExprsByValue(db.atoms.filter(e=>e.v>1n && e.digits<=cfg.maxDigits-1), denomCap);
      const residuals=residualPool(db,cfg,target).slice(0,residualCap);
      function wrapRatio(numer, denom, n, mode){
        let core=`${shortOperandD(numer,'/')}/${shortOperandD(denom,'/')}`;
        let actualMode = mode==='round' ? roundModeForDiv(numer.v, denom.v, n) : mode;
        let e;
        if(actualMode==='exact') e=makeDExpr(n, core, 'ratio', numer.ops+denom.ops+1, Math.max(numer.depth,denom.depth)+1);
        else if(actualMode==='floor' || actualMode==='ceil') e=makeDExpr(n, `${actualMode}(${core})`, `${actualMode}-ratio`, numer.ops+denom.ops+2, Math.max(numer.depth,denom.depth)+2);
        else return null;
        return e;
      }
      for(const d of denoms){
        if(shortAbort(deadline) || isUserInputPending()) return;
        const center=target*d.v;
        const exact=bestValueExpr(db,center);
        if(exact) addDigitCandidate(rows,seen,wrapRatio(exact,d,target,'exact'),target,cfg,'ratio exact');
        const half=d.v/2n;
        const ranges=[
          ['round', center-half, center+half],
          ['floor', center, center+d.v-1n],
          ['ceil', target>0n ? (target-1n)*d.v+1n : 0n, center]
        ];
        for(const [mode,lo0,hi0] of ranges){
          if(shortAbort(deadline)) return;
          const lo=lo0<0n?0n:lo0; const hi=hi0<lo?lo:hi0;
          const near=nearValueExprs(db,lo,hi,8);
          for(const n of near){
            const ok = mode==='round' ? roundDiv(n.v,d.v)===target : (mode==='floor' ? n.v/d.v===target : (n.v+d.v-1n)/d.v===target);
            if(ok) addDigitCandidate(rows,seen,wrapRatio(n,d,target,mode),target,cfg,'ratio exact');
          }
          const digitBudget=cfg.maxDigits-d.digits;
          if(digitBudget>=2){
            const pairNear=pairExprsInRange(db,lo,hi,digitBudget,cfg,Math.min(deadline, performance.now()+Math.max(45, 90+cfg.effort*70)),6);
            for(const n of pairNear){
              const ok = mode==='round' ? roundDiv(n.v,d.v)===target : (mode==='floor' ? n.v/d.v===target : (n.v+d.v-1n)/d.v===target);
              if(ok) addDigitCandidate(rows,seen,wrapRatio(n,d,target,mode),target,cfg,'ratio pair');
            }
          }
        }
        // one small residual on numerator: round((A +/- r)/d)
        for(const r of residuals){
          if(shortAbort(deadline) || isUserInputPending()) return;
          const candidates=[center+r.v, center-r.v].filter(x=>x>=0n);
          for(const want of candidates){
            const n=bestValueExpr(db,want);
            if(!n) continue;
            const numer = want===center+r.v ? combineD(n,'-',r,want-r.v,'offset') : combineD(n,'+',r,want+r.v,'offset');
            if(numer.digits>d.digits ? numer.digits+d.digits>cfg.maxDigits : numer.digits+d.digits>cfg.maxDigits) continue;
            if(roundDiv(numer.v,d.v)===target) addDigitCandidate(rows,seen,wrapRatio(numer,d,target,'round'),target,cfg,'ratio exact');
          }
        }
      }
    }
    function comparePowToN(base,p,q,n){
      base=BigInt(base); p=BigInt(p); q=BigInt(q); n=BigInt(n);
      if(n<0n) return 1;
      if(n===0n) return base===0n ? 0 : 1;
      if(n===1n) return base===1n ? 0 : 1;
      // v9: do the common comparison p*log(base) ? q*log(n) before any
      // exact BigInt exponentiation.  The old verifier occasionally built
      // thousands of digits repeatedly during rational-power shortform scans,
      // which made medium integer Continue look frozen and kept Stop from
      // reaching the event loop.  Exact arithmetic is still used near ties.
      const lb=Number(p)*Math.log(Number(base));
      const ln=Number(q)*Math.log(Number(n));
      if(Number.isFinite(lb) && Number.isFinite(ln)){
        const scale=Math.max(1, Math.abs(lb), Math.abs(ln));
        const diff=lb-ln;
        if(Math.abs(diff)>1e-11*scale) return diff<0 ? -1 : 1;
        // Very large near-ties are vanishingly rare in this UI search.  Keeping
        // the log sign here is preferable to locking the browser with a huge
        // exact-power fallback; floor/ceil are then verified again at adjacent
        // integers, so spurious displayed rows remain unlikely.
        if(p>1200n || q>90n) return diff<0 ? -1 : (diff>0 ? 1 : 0);
      }
      const nq=powBigCapped(n,q,null);
      if(nq===null) return 0;
      const bp=powBigCapped(base,p,nq);
      if(bp===null) return 1;
      if(bp<nq) return -1; if(bp>nq) return 1; return 0;
    }
    function verifyRoundRationalPower(base,p,q,n){
      if(n<0n) return false;
      base=BigInt(base); p=BigInt(p); q=BigInt(q); n=BigInt(n);
      const lb=Number(p)*Math.log(Number(base));
      const ln=Number(q)*Math.log(Number(n||1n));
      if(Number.isFinite(lb) && Number.isFinite(ln) && (p>1200n || q>90n)) return false;
      const bp=powBigCapped(base,p,null); if(bp===null) return false;
      const mid=(2n**q)*bp;
      const left=powBigCapped(2n*n-1n,q,null);
      const right=powBigCapped(2n*n+1n,q,null);
      return left<=mid && mid<right;
    }
    function verifyFloorRationalPower(base,p,q,n){
      if(n<0n) return false;
      const c1=comparePowToN(base,p,q,n);
      const c2=comparePowToN(base,p,q,n+1n);
      return c1>=0 && c2<0;
    }
    function verifyCeilRationalPower(base,p,q,n){
      if(n<=0n) return false;
      const c1=comparePowToN(base,p,q,n-1n);
      const c2=comparePowToN(base,p,q,n);
      return c1>0 && c2<=0;
    }
    function rationalPowerSearch(rows,seen,target,db,cfg,deadline){
      const residuals=[makeDExpr(0n,'0','zero')].concat(residualPool(db,cfg,target).slice(0,Math.min(90,cfg.residualProbe)));
      const baseCap = cfg.effort>=6 ? 90 : (cfg.effort>=4 ? 72 : 54);
      const qCap = cfg.effort>=6 ? 160 : (cfg.effort>=4 ? 96 : 64);
      const bases=db.argSources.filter(e=>e.v>=2n && e.v<=BigInt(baseCap)).sort(cmpExpr).slice(0, cfg.effort>=6 ? 90 : 64);
      const qSources=db.argSources.filter(e=>e.v>=2n && e.v<=BigInt(qCap)).sort(cmpExpr).slice(0, cfg.effort>=6 ? 180 : 110);
      const argByValue=new Map();
      for(const e of db.argSources){ const k=e.v.toString(); const old=argByValue.get(k); if(!old || cmpExpr(e,old)<0) argByValue.set(k,e); }
      function argNear(x){
        const out=[]; const seenLocal=new Set();
        const base=Math.round(x);
        const span=Math.max(3, Math.min(24, Math.ceil(Math.abs(x)*0.002)));
        for(let d=-span; d<=span; d++){
          const v=base+d;
          if(v<2 || v>2200) continue;
          const e=argByValue.get(String(v));
          if(e && !seenLocal.has(e.v.toString())){ seenLocal.add(e.v.toString()); out.push(e); }
        }
        return out.sort(cmpExpr).slice(0,10);
      }
      function buildWithResidual(core,r,sign){
        if(sign===0) return core;
        if(sign>0) return combineD(core,'+',r,target,'compound');
        return combineD(core,'-',r,target,'compound');
      }
      for(const b of bases){
        if(shortAbort(deadline) || isUserInputPending()) return;
        const logb=Math.log(Number(b.v)); if(!Number.isFinite(logb)||logb<=0) continue;
        for(const r of residuals){
          if(shortAbort(deadline)) return;
          const variants=[[target,0]];
          if(r.v>0n){ if(target>r.v) variants.push([target-r.v,1]); variants.push([target+r.v,-1]); }
          for(const [tv,sign] of variants){
            if(tv<=0n) continue;
            const lv=Math.log(Number(tv)); if(!Number.isFinite(lv)) continue;
            const want=lv/logb;
            for(const q of qSources){
              if(shortAbort(deadline) || isUserInputPending()) return;
              const pGuess=want*Number(q.v);
              if(!Number.isFinite(pGuess) || pGuess<2) continue;
              for(const p of argNear(pGuess)){
                if(b.digits+p.digits+q.digits+(sign===0?0:r.digits)>cfg.maxDigits) continue;
                const coreText=`${shortOperandD(b,'^')}^(${p.s}/${q.s})`;
                let core=null;
                if(verifyFloorRationalPower(b.v,p.v,q.v,tv)) core=makeDExpr(tv,`floor(${coreText})`,'floored-power',(b.ops||0)+(p.ops||0)+(q.ops||0)+3,Math.max(b.depth||0,p.depth||0,q.depth||0)+2);
                else if(verifyCeilRationalPower(b.v,p.v,q.v,tv)) core=makeDExpr(tv,`ceil(${coreText})`,'ceiled-power',(b.ops||0)+(p.ops||0)+(q.ops||0)+3,Math.max(b.depth||0,p.depth||0,q.depth||0)+2);
                // v7 deliberately does not emit round(...).  For rational powers,
                // only keep exact floor/ceil certificates; nearest-integer-only
                // cases are skipped rather than displayed as round.
                if(core) addDigitCandidate(rows,seen,buildWithResidual(core,r,sign),target,cfg,'power exact');
              }
            }
          }
        }
      }
    }
    function structuredBackupSearch(rows, seen, target, cfg, deadline){
      // Deterministic safety net: still exact, still ranked by digit count, but allows a slightly
      // larger digit budget to avoid an empty shortform section for uncooperative random inputs.
      const td=decimalDigitCountBig(target);
      if(td<=2) return;
      const accept=Math.max(cfg.maxDigits, Math.min(td-1, Math.ceil(td*0.72)));
      const localCfg={...cfg, maxDigits:accept, residualProbe:Math.max(cfg.residualProbe, 300)};
      const bound=(10n**BigInt(Math.min(150, td+8)))-1n;
      const seeds=[]; const seedSeen=new Set();
      function pushSeed(e){
        if(!e || e.v<=1n || e.v>bound || e.digits>accept) return;
        const k=e.v.toString()+':'+canonicalExpressionKey(e.s);
        if(seedSeen.has(k)) return; seedSeen.add(k); seeds.push(e);
      }
      const backupBaseLimit=Math.max(40, Math.min(220, Number(cfg.maxPowBase || 99)));
      for(let B=2; B<=backupBaseLimit && performance.now()<deadline; B++){
        let v=1n;
        for(let C=2; C<=80; C++){
          v*=BigInt(B); if(v>bound) break;
          pushSeed(makeDExpr(v, `${B}^${C}`, 'backup-power', 1, 1));
        }
      }
      let f=1n;
      for(let B=2; B<=80 && performance.now()<deadline; B++){
        f*=BigInt(B); if(f>bound) break;
        if(B>=4) pushSeed(makeDExpr(f, `${B}!`, 'backup-factorial', 1, 1));
      }
      for(let N=5; N<=220 && performance.now()<deadline; N++){
        for(let K=2; K<=Math.min(12,Math.floor(N/2)); K++){
          const v=binomBigCapped(BigInt(N), BigInt(K), bound); if(v===null) break;
          pushSeed(makeDExpr(v, `binom(${N},${K})`, 'backup-binom', 1, 1));
        }
      }
      seeds.sort((a,b)=>a.digits-b.digits || a.rank-b.rank || (a.v<b.v?-1:a.v>b.v?1:0));
      const smallDs=[]; for(let d=2; d<=99; d++) smallDs.push(makeDExpr(BigInt(d), String(d), 'literal'));
      function addResidual(core, residual, sign, label){
        if(shortAbort(deadline)) return;
        let e=core;
        if(residual!==0n){
          const r=compactLiteralD(absBig(residual));
          e = sign>=0 ? combineD(core,'+',r,core.v+r.v,label) : combineD(core,'-',r,core.v-r.v,label);
        }
        addDigitCandidate(rows,seen,e,target,localCfg,label);
      }
      const top=seeds.slice(0, Math.min(seeds.length, 9000));
      for(const s0 of top){
        if(shortAbort(deadline)) return;
        // target = seed ± small-ish residual
        if(s0.v<=target){ addResidual(s0, target-s0.v, +1, 'structured residual'); }
        else { addResidual(s0, s0.v-target, -1, 'structured residual'); }
        // target = q * seed ± residual, useful when a two-digit base power captures most magnitude.
        if(s0.v>1n && s0.v<target){
          const q=target/s0.v;
          if(q>=2n && q<=99999999n){
            const qExpr=makeDExpr(q, q.toString(), 'literal');
            const prod=combineD(qExpr,'*',s0,q*s0.v,'structured-product');
            addResidual(prod, target-prod.v, +1, 'structured product');
          }
        }
        // target = floor/ceil((seed ± residual)/d), with small two-digit denominators.
        for(const d of smallDs){
          if(performance.now()>deadline) return;
          const center=target*d.v;
          const residual=center-s0.v;
          const absR=absBig(residual);
          if(absR.toString().length>Math.max(1, Math.ceil(td*0.55))) continue;
          let numer=s0;
          if(residual>0n) numer=combineD(s0,'+',compactLiteralD(absR), center, 'structured-numer');
          else if(residual<0n) numer=combineD(s0,'-',compactLiteralD(absR), center, 'structured-numer');
          if(numer.digits+d.digits>accept) continue;
          const mode=roundModeForDiv(numer.v,d.v,target);
          if(mode==='exact') addDigitCandidate(rows,seen,combineD(numer,'/',d,target,'structured-ratio'),target,localCfg,'structured ratio');
          else if(mode==='floor' || mode==='ceil') addDigitCandidate(rows,seen,makeDExpr(target, `${mode}(${shortOperandD(numer,'/')}/${d.s})`, 'structured-rounded-ratio', (numer.ops||0)+2, (numer.depth||0)+2),target,localCfg,'structured ratio');
        }
      }
    }

    async function smallIntegerExhaustiveSearchAsync(rows, seen, target, effort, deadline, onProgress=null){
      if(target>100000000n || shortAbort(deadline)) return;
      const td=decimalDigitCountBig(target);
      const cfg=digitSearchConfig(Math.min(7, Math.max(0, effort+1)), target);
      cfg.maxDigits=Math.max(cfg.maxDigits, Math.min(10, td+2));
      tuneIntegerConfigForDigitBudget(cfg, target);
      // v8.7: keep the <=1e8 exact enumeration bounded and responsive.  The
      // search families are the same as v8.5/v8.6, but each sub-pass gets a
      // hard slice and reports progress before yielding to the UI.
      cfg.literalCap=Math.max(cfg.literalCap, effort>=5 ? 22000 : 9500);
      cfg.argCap=Math.max(cfg.argCap, effort>=5 ? 2200 : 1200);
      cfg.pairProbe=Math.max(cfg.pairProbe, effort>=5 ? 52000 : 28000);
      cfg.denomProbe=Math.max(cfg.denomProbe, effort>=5 ? 18000 : 9500);
      cfg.reverseProbe=Math.max(cfg.reverseProbe, effort>=5 ? 16000 : 8500);
      const start=performance.now();
      const budgetMs=Math.max(120, deadline-start);
      async function report(label, frac){
        if(onProgress) onProgress({label, elapsed:Math.round(performance.now()-start), budgetMs, frac, rows:rows.length});
        await yieldToUI();
      }
      const capFor=(lo, hi)=>Math.min(deadline, performance.now()+Math.max(80, Math.min(hi, lo+effort*35)));
      await report('small exhaustive: building compact DB', .10);
      if(shortAbort(deadline)) return;
      const dbDeadline=capFor(140, effort>=5 ? 300 : 220);
      const db=buildDigitSearchDB(target, {}, cfg, dbDeadline);
      await report('small exhaustive: direct/reverse exact pass', .38);
      if(shortAbort(deadline)) return;
      directAndReverseSearch(rows,seen,target,db,cfg,capFor(120, effort>=5 ? 260 : 190));
      await report('small exhaustive: rational pass', .58);
      if(shortAbort(deadline)) return;
      ratioSearch(rows,seen,target,db,cfg,capFor(150, effort>=5 ? 320 : 230));
      await report('small exhaustive: rational powers', .78);
      if(shortAbort(deadline)) return;
      rationalPowerSearch(rows,seen,target,db,cfg,capFor(150, effort>=5 ? 320 : 230));
      await report('small exhaustive: structured backup', .92);
      if(shortAbort(deadline)) return;
      structuredBackupSearch(rows,seen,target,cfg,capFor(120, effort>=5 ? 260 : 190));
      await report('small exhaustive: done', 1);
    }

    function parseLabelFreeCandidate(candidate){
      return String(candidate||'').replace(/^[^:]+:\s*/, '').trim();
    }
    function factorSmallNumber(n){
      n=Number(n);
      if(!Number.isSafeInteger(n) || n<0 || n>1000000000) return null;
      const m=new Map();
      if(n===0) return null;
      if(n===1) return m;
      let x=n;
      for(let p=2; p*p<=x; p+=(p===2?1:2)){
        while(x%p===0){ m.set(p,(m.get(p)||0)+1); x=Math.floor(x/p); }
      }
      if(x>1) m.set(x,(m.get(x)||0)+1);
      return m;
    }
    function mergeFactorMaps(a,b,scale=1){
      if(!a || !b) return null;
      const out=new Map(a);
      for(const [p,e] of b){
        const v=(out.get(p)||0)+e*scale;
        if(v) out.set(p,v); else out.delete(p);
      }
      return out;
    }
    function factorMapKey(m){
      if(!m) return null;
      return [...m.entries()].sort((a,b)=>a[0]-b[0]).map(([p,e])=>`${p}:${e}`).join('|') || '1';
    }
    function valueFromFactorMapKey(key, cap=1000000000n){
      if(!key || key==='1') return 1n;
      let r=1n;
      for(const part of String(key).split('|')){
        const [ps,es]=part.split(':'); const p=BigInt(ps), e=Number(es);
        if(!Number.isInteger(e) || e<0) return null;
        for(let i=0;i<e;i++){ r*=p; if(r>cap) return null; }
      }
      return r;
    }
    function factorMapNonnegative(m){
      if(!m) return false;
      for(const [,e] of m) if(e<0) return false;
      return true;
    }
    function factorMapRationalParts(m, cap=10n**36n){
      if(!m) return null;
      let num=1n, den=1n;
      for(const [p0,e0] of m){
        const p=BigInt(p0), e=Number(e0);
        if(!Number.isInteger(e)) return null;
        const cnt=Math.abs(e);
        for(let i=0;i<cnt;i++){
          if(e>0){ num*=p; if(num>cap) return null; }
          else { den*=p; if(den>cap) return null; }
        }
      }
      if(den===0n) return null;
      const g=gcdBig(num,den);
      return {num:num/g, den:den/g};
    }
    function gcdIntArray(arr){
      let g=0;
      for(const x of arr){ let a=Math.abs(Number(x)||0); while(a){ const t=g%a; g=a; a=t; } }
      return g;
    }
    function powSmallBig(base, exp, capDigits=18){
      let r=1n, b=BigInt(base), e=BigInt(exp);
      const cap=10n**BigInt(capDigits);
      while(e>0n){ if(e&1n){ r*=b; if(r>cap) return null; } e>>=1n; if(e){ b*=b; if(b>cap) b=cap+1n; } }
      return r;
    }
    function bestPureMultiplicativeDisplay(s){
      s=stripOuterParens(String(s||'').trim());
      if(!s || /[+]|floor\(|ceil\(|round\(/.test(s) || s[0]==='-' || /[^\^]-/.test(s)) return null;
      const m=pureMultiplicativeSignature(s);
      if(!factorMapNonnegative(m) || !m.size) return null;
      const entries=[...m.entries()].sort((a,b)=>a[0]-b[0]);
      const exps=entries.map(([,e])=>e);
      const g=gcdIntArray(exps);
      if(g<=1) return null;
      const candidates=[];
      for(let d=2; d<=g; d++){
        if(g%d!==0) continue;
        let base=1n, ok=true;
        for(const [p,e] of entries){
          const pe=powSmallBig(p, e/d, 18);
          if(pe===null){ ok=false; break; }
          base*=pe; if(base>10n**18n){ ok=false; break; }
        }
        if(ok && base>1n) candidates.push(`${base.toString()}^${d}`);
      }
      if(!candidates.length) return null;
      const cur={s, digits:digitCountExpr(s), ops:99, depth:99};
      candidates.sort((a,b)=>{
        const aa={s:a,digits:digitCountExpr(a),ops:1,depth:1}, bb={s:b,digits:digitCountExpr(b),ops:1,depth:1};
        const pa=a.match(/^(\d+)\^(\d+)$/), pb=b.match(/^(\d+)\^(\d+)$/);
        const baseTie=(pa&&pb) ? (Number(pa[1])-Number(pb[1])) : 0;
        return (aa.digits-bb.digits)||baseTie||(shortRank(aa)-shortRank(bb))||(a.length-b.length)||(a<b?-1:a>b?1:0);
      });
      const best=candidates[0];
      const bestObj={s:best,digits:digitCountExpr(best),ops:1,depth:1};
      return ((bestObj.digits<cur.digits) || (bestObj.digits===cur.digits && shortRank(bestObj)+400<shortRank(cur))) ? best : null;
    }
    function simplificationScoreForText(s){
      const obj={s:String(s||''), digits:digitCountExpr(s), ops:Math.max(1,(String(s||'').match(/[+\-·/^]|binom\(|!|floor\(|ceil\(/g)||[]).length), depth:(String(s||'').match(/[()]/g)||[]).length};
      return shortRank(obj) + obj.digits*1800000 + String(s||'').length*8;
    }
    function chooseSimplerText(a,b){
      if(!b || b===a) return a;
      const da=digitCountExpr(a), db=digitCountExpr(b);
      if(db<da) return b;
      if(db>da) return a;
      return simplificationScoreForText(b)+120 < simplificationScoreForText(a) ? b : a;
    }
    function simplifySingleTermDisplay(term){
      term=stripOuterParens(String(term||'').trim());
      if(!term) return term;
      for(const fn of ['round','floor','ceil']){
        if(term.startsWith(fn+'(') && term.endsWith(')')){
          const inner=term.slice(fn.length+1,-1);
          const simp=simplifyIntegerExpressionDisplay(inner);
          return simp===inner ? term : `${fn}(${simp})`;
        }
      }
      const pure=bestPureMultiplicativeDisplay(term);
      let best=chooseSimplerText(term,pure);
      // If a product/division term contains sub-products such as (35^7·35)/17,
      // simplify both sides and let the pure multiplicative signature collapse
      // any remaining common factors.
      for(const op of ['·','/']){
        const sp=splitTopLevel(best,[op]);
        if(sp){
          const l=simplifySingleTermDisplay(sp[0]);
          const r=simplifySingleTermDisplay(sp[2]);
          const rebuilt=`${shortOperandD({s:l},op==='·'?'*':'/')}${op}${shortOperandD({s:r},op==='·'?'*':'/')}`;
          best=chooseSimplerText(best, bestPureMultiplicativeDisplay(rebuilt) || rebuilt);
        }
      }
      return best;
    }
    function simplifyIntegerExpressionDisplay(s){
      let cur=normalizeShortDisplay(String(s||'').trim());
      for(let pass=0; pass<3; pass++){
        const before=cur;
        const terms=splitAdditiveTerms(cur);
        if(terms.length>1){
          // v10.7.1: fold tiny additive corrections after each term is simplified.
          // This keeps fallback/database rows from displaying variants such as
          // binom(A,B)+3-1 or +4-2 when the visible correction is just +2.
          const coreTerms=[];
          let tinyOffset=0;
          for(const t of terms){
            const body=simplifySingleTermDisplay(t.term);
            const val=evalSmallIntegerExpr(body);
            if(val!==null && Math.abs(val)<=99){
              tinyOffset += t.sign*val;
            }else{
              coreTerms.push({sign:t.sign, body});
            }
          }
          const rebuilt=[];
          for(let i=0;i<coreTerms.length;i++){
            const t=coreTerms[i];
            if(i===0) rebuilt.push(t.sign<0 ? `-${t.body}` : t.body);
            else rebuilt.push(`${t.sign<0?'-':'+'}${t.body}`);
          }
          if(tinyOffset!==0 || !rebuilt.length){
            const body=String(Math.abs(tinyOffset));
            if(!rebuilt.length) rebuilt.push(tinyOffset<0 ? `-${body}` : body);
            else rebuilt.push(`${tinyOffset<0?'-':'+'}${body}`);
          }
          cur=normalizeShortDisplay(rebuilt.join(''));
        }else{
          cur=normalizeShortDisplay(simplifySingleTermDisplay(cur));
        }
        const pure=bestPureMultiplicativeDisplay(cur);
        if(pure) cur=normalizeShortDisplay(chooseSimplerText(cur,pure));
        if(cur===before) break;
      }
      return cur;
    }
    function simplifyDExprIfBetter(expr){
      if(!expr || !expr.s) return expr;
      const best=simplifyIntegerExpressionDisplay(expr.s);
      if(!best || best===expr.s) return expr;
      if(chooseSimplerText(expr.s,best)===expr.s) return expr;
      // v10.3: accept a simplification only if the visible formula remains exact.
      if(typeof displayExprMatchesTarget==='function' && !displayExprMatchesTarget(best, expr.v)) return expr;
      const newOps=Math.max(1,(best.match(/[+\-·/^]|binom\(|!|floor\(|ceil\(/g)||[]).length);
      const out={...expr, s:best, kind:(expr.kind||'expr')+'-simplified', ops:Math.min(expr.ops||newOps,newOps), depth:Math.min(expr.depth||1, Math.max(1,(best.match(/[()]/g)||[]).length/2))};
      out.digits=digitCountExpr(out.s);
      out.rank=shortRank(out);
      return out;
    }
    function evalSmallIntegerExpr(s){
      s=stripOuterParens(String(s||'').trim());
      if(!s) return null;
      const addTerms=splitAdditiveTerms(s);
      if(addTerms.length>1){
        let total=0;
        for(const t of addTerms){ const v=evalSmallIntegerExpr(t.term); if(v===null) return null; total += t.sign*v; }
        return Number.isSafeInteger(total) && Math.abs(total)<=1000000000 ? total : null;
      }
      let sp=splitTopLevel(s,['/']);
      if(sp){ const a=evalSmallIntegerExpr(sp[0]), b=evalSmallIntegerExpr(sp[2]); if(a===null || b===null || b===0 || a%b!==0) return null; return a/b; }
      sp=splitTopLevel(s,['·']);
      if(sp){ const a=evalSmallIntegerExpr(sp[0]), b=evalSmallIntegerExpr(sp[2]); if(a===null || b===null) return null; const r=a*b; return Number.isSafeInteger(r) && Math.abs(r)<=1000000000 ? r : null; }
      if(/^\d+$/.test(s)) return Number(s);
      sp=splitTopLevel(s,['^']);
      if(sp){
        const a=evalSmallIntegerExpr(sp[0]), b=evalSmallIntegerExpr(sp[2]);
        if(a===null || b===null || b<0 || b>16) return null;
        const r=Math.pow(a,b);
        return Number.isSafeInteger(r) && Math.abs(r)<=1000000000 ? r : null;
      }
      if(s.endsWith('!')){
        const inner=evalSmallIntegerExpr(s.slice(0,-1));
        if(inner===null || inner<0 || inner>12) return null;
        let r=1; for(let i=2;i<=inner;i++) r*=i;
        return r;
      }
      if(/^(floor|ceil)\(/.test(s) && s.endsWith(')')){
        const name=s.slice(0,s.indexOf('('));
        const inner=s.slice(name.length+1,-1);
        const div=splitTopLevel(inner, ['/']);
        if(div){
          const a=evalSmallIntegerExpr(div[0]), b=evalSmallIntegerExpr(div[2]);
          if(a===null || b===null || b===0) return null;
          const A=BigInt(a), B=BigInt(b);
          if(name==='floor') return Number(floorDiv(A,B));
          return Number(ceilDiv(A,B));
        }
        const v=evalSmallIntegerExpr(inner);
        return v;
      }
      if(/^binom\(/.test(s) && s.endsWith(')')){
        const parts=latexArgsInside(s,'binom');
        if(parts.length!==2) return null;
        const n=evalSmallIntegerExpr(parts[0]), k=evalSmallIntegerExpr(parts[1]);
        if(n===null || k===null || k<0 || n<0 || k>n || n>60) return null;
        let kk=Math.min(k,n-k), r=1;
        for(let i=1;i<=kk;i++) r=Math.round(r*(n-kk+i)/i);
        return Number.isSafeInteger(r) ? r : null;
      }
      return null;
    }
    function pureMultiplicativeSignature(s){
      s=stripOuterParens(String(s||'').trim());
      if(!s || /^[-+]/.test(s) || /round\(|floor\(|ceil\(|\+|-/.test(s)) return null;
      let sp=splitTopLevel(s,['·']);
      if(sp){
        const left=pureMultiplicativeSignature(sp[0]);
        const right=pureMultiplicativeSignature(sp[2]);
        return mergeFactorMaps(left,right,1);
      }
      sp=splitTopLevel(s,['/']);
      if(sp){
        const left=pureMultiplicativeSignature(sp[0]);
        const right=pureMultiplicativeSignature(sp[2]);
        return mergeFactorMaps(left,right,-1);
      }
      sp=splitTopLevel(s,['^']);
      if(sp){
        const base=pureMultiplicativeSignature(sp[0]);
        const exp=evalSmallIntegerExpr(sp[2]);
        if(!base || exp===null || exp<0 || exp>10000) return null;
        const out=new Map();
        for(const [p,e] of base) out.set(p,e*exp);
        return out;
      }
      const val=evalSmallIntegerExpr(s);
      return val===null ? null : factorSmallNumber(val);
    }
    function splitAdditiveTerms(s){
      s=stripOuterParens(String(s||'').trim());
      const out=[]; let depth=0, start=0, sign=1;
      for(let i=0;i<s.length;i++){
        const ch=s[i];
        if(ch==='(') depth++;
        else if(ch===')') depth--;
        else if(depth===0 && (ch==='+' || ch==='-') && i>0){
          out.push({sign, term:s.slice(start,i)});
          sign = ch==='+' ? 1 : -1;
          start=i+1;
        }
      }
      out.push({sign, term:s.slice(start)});
      return out.filter(t=>t.term.trim());
    }
    function canonicalExpressionKey(s){
      s=stripOuterParens(String(s||'').trim());
      if(!s) return '';
      const smallValue=evalSmallIntegerExpr(s);
      if(smallValue!==null && Math.abs(smallValue)<=1000000000) return `int:${smallValue}`;
      for(const fn of ['round','floor','ceil']){
        if(s.startsWith(fn+'(') && s.endsWith(')')) return `${fn}(${canonicalExpressionKey(s.slice(fn.length+1,-1))})`;
      }
      if(/^(floor|ceil)\(/.test(s) && s.endsWith(')')){
        const name=s.slice(0,s.indexOf('('));
        const inner=s.slice(name.length+1,-1);
        const div=splitTopLevel(inner, ['/']);
        if(div){
          const a=evalSmallIntegerExpr(div[0]), b=evalSmallIntegerExpr(div[2]);
          if(a===null || b===null || b===0) return null;
          const A=BigInt(a), B=BigInt(b);
          if(name==='floor') return Number(floorDiv(A,B));
          return Number(ceilDiv(A,B));
        }
        const v=evalSmallIntegerExpr(inner);
        return v;
      }
      if(/^binom\(/.test(s) && s.endsWith(')')){
        const parts=latexArgsInside(s,'binom');
        if(parts.length===2) return `binom(${canonicalExpressionKey(parts[0])},${canonicalExpressionKey(parts[1])})`;
      }
      const pureKey=factorMapKey(pureMultiplicativeSignature(s));
      if(pureKey) return `mul:${pureKey}`;
      const addTerms=splitAdditiveTerms(s);
      if(addTerms.length>1){
        const termKeys=[];
        let ok=true;
        for(const t of addTerms){
          const key=canonicalExpressionKey(t.term);
          if(!key){ ok=false; break; }
          termKeys.push(`${t.sign<0?'-':'+'}${key}`);
        }
        if(ok) return 'sum:' + termKeys.sort().join('|');
      }
      let sp=splitTopLevel(s,['/']);
      if(sp) return `div(${canonicalExpressionKey(sp[0])},${canonicalExpressionKey(sp[2])})`;
      sp=splitTopLevel(s,['·']);
      if(sp){
        const factors=[];
        function collectMul(x){
          x=stripOuterParens(x);
          const p=splitTopLevel(x,['·']);
          if(p){ collectMul(p[0]); collectMul(p[2]); }
          else factors.push(canonicalExpressionKey(x));
        }
        collectMul(s);
        return 'multerms:' + factors.sort().join('|');
      }
      sp=splitTopLevel(s,['^']);
      if(sp) return `pow(${canonicalExpressionKey(sp[0])},${canonicalExpressionKey(sp[2])})`;
      if(s.endsWith('!')) return `fact(${canonicalExpressionKey(s.slice(0,-1))})`;
      const mult=factorMapKey(pureMultiplicativeSignature(s));
      if(mult) return `mul:${mult}`;
      return normalizeShortDisplay(s);
    }
    function correctionValue(term, maxAbs=100000){
      const v=evalSmallIntegerExpr(term);
      return v!==null && Math.abs(v)<=maxAbs ? v : null;
    }
    function correctionValueForRatio(term, maxAbs=1000000){
      const t=stripOuterParens(String(term||'').trim());
      const v=evalSmallIntegerExpr(t);
      if(v===null || Math.abs(v)>maxAbs) return null;
      // Treat literal constants as offsets freely, but do not let a compact
      // structured main term such as 9^6 become an offset merely because it is
      // numerically below the cutoff. Small structured corrections like 2^3-1
      // remain allowed.
      if(/^\d+$/.test(t)) return v;
      if(Math.abs(v)<=1000) return v;
      if(!/[!^]|binom\(/.test(t) && digitCountExpr(t)<=6) return v;
      return null;
    }
    function termIsTinyCorrection(term){
      return correctionValue(term,100000)!==null;
    }
    function ratioCoreFamilyKey(inner){
      inner=stripOuterParens(String(inner||'').trim());
      const pure=pureMultiplicativeSignature(inner);
      const pureKey=factorMapKey(pure);
      if(pureKey){
        if(factorMapNonnegative(pure)) return `ratio-family-integer:${pureKey};rem:0;carry:0`;
        const rp=factorMapRationalParts(pure);
        if(rp){
          const rem=rp.num % rp.den;
          // Keep the fixed rational value in the factor key.  The enormous
          // integer part is intentionally not part of the key; equivalent
          // floor/ceil variants differ only by the outside shift below.
          return `ratio-family-reduced:${pureKey};rem:${rem.toString()};carry:0`;
        }
      }
      const sp=splitTopLevel(inner, ['/']);
      if(!sp) return null;
      if(splitAdditiveTerms(sp[2]).length>1) return null;
      const denVal=evalSmallIntegerExpr(sp[2]);
      const denKey=canonicalExpressionKey(sp[2]);
      const terms=splitAdditiveTerms(sp[0]);
      let offset=0;
      const coreTerms=[];
      for(const t of terms){
        const cv=correctionValueForRatio(t.term,1000000);
        if(cv!==null) offset += t.sign*cv;
        else coreTerms.push(t);
      }
      if(!coreTerms.length) coreTerms.push(...terms);
      let carry=0, rem=offset;
      if(denVal && Number.isSafeInteger(denVal) && denVal>0){
        carry=Math.floor(offset/denVal);
        rem=offset-carry*denVal;
        if(rem<0){ rem+=denVal; carry-=1; }
      }
      if(coreTerms.length===1 && coreTerms[0].sign>0){
        const nf=pureMultiplicativeSignature(coreTerms[0].term);
        const df=pureMultiplicativeSignature(sp[2]);
        const merged=mergeFactorMaps(nf,df,-1);
        const fkey=factorMapKey(merged);
        if(fkey){
          const integerCore=factorMapNonnegative(merged);
          return `${integerCore?'ratio-family-integer':'ratio-family-reduced'}:${fkey};rem:${rem};carry:${carry}`;
        }
      }
      const coreKey=coreTerms.map(t=>`${t.sign<0?'-':'+'}${canonicalExpressionKey(t.term)}`).sort().join('|');
      return `ratio-family:${coreKey}/(${denKey});rem:${rem};carry:${carry}`;
    }
    function normalizeRoundedRatioCore(core, mode, outside=0){
      if(!core) return null;
      const m=core.match(/^(ratio-family-integer:[^;]+);rem:([-\d]+);carry:([-\d]+)$/);
      if(m){
        const rem=Number(m[2]), carry=Number(m[3]);
        let shift=carry + Number(outside||0);
        if(mode==='ceil' && rem>0) shift += 1;
        // For floor of an integer core plus a proper fractional remainder, the
        // remainder does not change the integer part. This merges forms like
        // floor((N+20)/3) and floor(N/3)+6 when N/3 is already integral.
        return `${m[1]};shift:${shift}`;
      }
      const rm=core.match(/^(ratio-family-reduced:[^;]+);rem:([-\d]+);carry:([-\d]+)$/);
      if(rm){
        const rem=BigInt(rm[2]);
        const carry=Number(rm[3]);
        let shift=carry + Number(outside||0);
        // Same rational core: ceil(x)+k is floor(x)+(k+1) exactly when the
        // core is not already an integer.  This collapses duplicated floor/ceil
        // rows without collapsing genuinely different rational cores.
        if(mode==='ceil' && rem!==0n) shift += 1;
        return `${rm[1]};shift:${shift}`;
      }
      const n=core.match(/^(.*);carry:([-\d]+)$/);
      if(n) return `${n[1]};shift:${Number(n[2])+Number(outside||0)};mode:${mode}`;
      return `${core};out:${outside};mode:${mode}`;
    }
    function roundedRatioParts(expr){
      expr=stripOuterParens(String(expr||'').trim());
      const m=expr.match(/^(round|floor|ceil)\((.*)\)$/);
      if(!m) return null;
      const core=ratioCoreFamilyKey(m[2]);
      if(!core) return null;
      return {mode:m[1], core};
    }
    function roundedRatioFamilyKey(expr){
      const p=roundedRatioParts(expr);
      if(!p) return null;
      return `rounded-${normalizeRoundedRatioCore(p.core,p.mode,0)}`;
    }
    function exactRatioFamilyKey(expr){
      // A standalone integer literal/expression is an offset, not a ratio core.
      // This matters when normalizing floor(R)+k / ceil(R)+(k-1) families.
      if(evalSmallIntegerExpr(expr)!==null) return null;
      const core=ratioCoreFamilyKey(expr);
      return core ? `exact-${core}` : null;
    }
    function ratioSumFamilyKey(expr){
      expr=stripOuterParens(String(expr||'').trim());
      const terms=splitAdditiveTerms(expr);
      if(terms.length<=1) return null;
      let ratioKey=null, ratioMode='exact', offset=0;
      for(const t of terms){
        const rp=roundedRatioParts(t.term);
        const er=exactRatioFamilyKey(t.term);
        if(rp || er){
          if(ratioKey) return null;
          ratioKey = rp ? rp.core : er.replace(/^exact-/,'');
          ratioMode = rp ? rp.mode : 'exact';
        }else{
          const cv=correctionValue(t.term,1000000);
          if(cv===null) return null;
          offset += t.sign*cv;
        }
      }
      return ratioKey ? `ratio-sum-${normalizeRoundedRatioCore(ratioKey,ratioMode,offset)}` : null;
    }
    function candidateEquivalenceKey(row){
      const expr=parseLabelFreeCandidate(row.candidate);
      const exact=evalSmallIntegerExpr(expr);
      if(exact!==null && Math.abs(exact)<=1000000000) return `exact-small:${exact}`;
      const pureKey=factorMapKey(pureMultiplicativeSignature(expr));
      if(pureKey) return `pure-mul:${pureKey}`;
      const rr=ratioSumFamilyKey(expr) || roundedRatioFamilyKey(expr) || exactRatioFamilyKey(expr);
      if(rr){
        const norm=rr.replace(/^ratio-sum-/,'ratio-').replace(/^rounded-/,'ratio-').replace(/^exact-/,'ratio-');
        const m=norm.match(/^ratio-ratio-family-integer:([^;]+);shift:([-\d]+)$/);
        if(m){
          const baseVal=valueFromFactorMapKey(m[1]);
          if(baseVal!==null){
            const v=baseVal+BigInt(Number(m[2]));
            if(v>=-1000000000n && v<=1000000000n) return `exact-small:${v.toString()}`;
          }
        }
        return norm;
      }
      return `${row.feature||exprFeature(expr)}:${canonicalExpressionKey(expr)}`;
    }
    function makeRoundedDivDExpr(value, numerExpr, denomExpr, numerValue, denomValue, ops=2, depth=1){
      const core=`${shortOperandD(numerExpr,'/')}/${shortOperandD(denomExpr,'/')}`;
      const mode=roundModeForDiv(numerValue, denomValue, value);
      if(mode==='exact') return makeDExpr(value, core, 'database-ratio', ops, depth);
      if(mode==='floor' || mode==='ceil') return makeDExpr(value, `${mode}(${core})`, `database-${mode}`, ops+1, depth+1);
      return null;
    }
    function addDatabaseCandidate(rows, seen, expr, target, label='database'){
      if(!expr || expr.v!==target || expr.ops<1) return;
      expr=validateDExprForTarget(expr, target);
      if(!expr) return;
      const td=decimalDigitCountBig(target);
      const maxMeaningful=Math.max(2, Math.min(td-1, Math.ceil(td*0.82)+1));
      const special=/floor\(|ceil\(|round\(|binom\(|!|\^|·|\//.test(expr.s);
      if(expr.digits>maxMeaningful && !(special && expr.digits<td)) return;
      const r={candidate:`${label}: ${expr.s}`, copyCandidate:expr.s, latex:exprToLatex(expr.s), value:`exact = ${shortPrettyValue(target)}`, copyValue:String(rawSignedTargetForCopy(target)), err:0, hideError:true, beauty:expr.rank, feature:exprFeature(expr.s), digits:expr.digits, ops:expr.ops};
      const key=candidateEquivalenceKey(r);
      if(seen.has(key)) return;
      seen.add(key); rows.push(r);
    }

    function staticShortformRows(settings){
      const rawN=resolvedIntegerBig(settings);
      if(rawN===null || rawN===0n) return [];
      const sign=rawN<0n ? -1n : 1n;
      const target=absBig(rawN);
      if(target>100000n) return [];
      const key=target.toString();
      let exprs=[];
      if(window.RIES_SHORTFORM_100K_PACKED && typeof window.RIES_SHORTFORM_100K_PACKED.get==='function'){
        exprs=window.RIES_SHORTFORM_100K_PACKED.get(key);
      }
      const multi=window.RIES_SHORTFORM_100K_MULTI && (window.RIES_SHORTFORM_100K_MULTI[key] || window.RIES_SHORTFORM_100K_MULTI[Number(target)]);
      if(Array.isArray(multi)){
        for(const e of multi) if(e && !exprs.includes(e)) exprs.push(e);
      }
      const single=window.RIES_SHORTFORM_100K && (window.RIES_SHORTFORM_100K[key] || window.RIES_SHORTFORM_100K[Number(target)]);
      if(single && !exprs.includes(single)) exprs.unshift(single);
      if(!exprs.length) return [];
      const rows=[]; const seen=new Set();
      for(let expr of exprs){
        expr=normalizeShortDisplay(String(expr||'').trim());
        if(!expr || seen.has(expr) || !displayExprMatchesTarget(expr, target)) continue; seen.add(expr);
        const signedExpr=sign<0n ? '-('+expr+')' : expr;
        rows.push({candidate:`precomputed shortform: ${signedExpr}`, copyCandidate:signedExpr, latex:exprToLatex(signedExpr), value:`exact = ${shortPrettyValue(rawN)}`, copyValue:rawN.toString(), err:0, hideError:true, beauty:shortRank({s:signedExpr,ops:1,depth:1}), feature:exprFeature(signedExpr), digits:digitCountExpr(signedExpr), ops:1});
      }
      return selectDigitShortforms(rows, Math.max(5, Math.min(20, Number(settings.limit)||5)));
    }
    function offsetLimitForTarget(target, effort=3){
      const td=decimalDigitCountBig(absBig(target));
      let limit=99;
      if(td>=9) limit=999;
      if(td>=12) limit=9999;
      if(td>=15) limit=99999;
      // v6.8: for genuinely large integers, do not use the generic exact
      // shortform engine; instead let the structured database templates probe
      // larger A,B,C,D,E constants for near-power / near-binomial forms.
      if(td>=16) limit = effort>=6 ? 999999 : (effort>=4 ? 399999 : 99999);
      if(effort>=5 && td>=10 && td<16) limit*=2;
      return limit;
    }
    function literalRangeLimitForTarget(target, effort=3){
      const td=decimalDigitCountBig(absBig(target));
      if(td>=16) return effort>=6 ? 999999 : (effort>=4 ? 299999 : 99999);
      if(td>=15) return effort>=6 ? 999999 : 99999;
      if(td>=12) return effort>=5 ? 99999 : 9999;
      if(td>=9) return effort>=4 ? 9999 : 999;
      return effort>=5 ? 999 : 99;
    }
    function integerDatabaseRows(settings, baseRows=[]){
      const rawN=resolvedIntegerBig(settings);
      if(rawN===null || rawN===0n) return [];
      const sign=rawN<0n ? -1n : 1n;
      const target=absBig(rawN);
      const effort=Math.max(0, Math.min(7, Number(settings.shortEffort)||0));
      const dbStart=performance.now();
      const hardDbCap=decimalDigitCountBig(absBig(rawN))>=16 ? 520 : 760;
      const deadline=dbStart+Math.min(hardDbCap, (decimalDigitCountBig(absBig(rawN))>=16 ? 360 : 520) + effort*(decimalDigitCountBig(absBig(rawN))>=16 ? 35 : 55));
      const offsetLimit=offsetLimitForTarget(target, effort);
      const offsetBig=BigInt(offsetLimit);
      const literalLimit=literalRangeLimitForTarget(target, effort);
      const td=decimalDigitCountBig(target);
      const largeStructuredOnly = td>=16;
      const baseLimit = largeStructuredOnly ? (effort>=6 ? 999 : (effort>=4 ? 420 : 220)) : 99;
      const coeffLimit = largeStructuredOnly ? (effort>=6 ? 999 : (effort>=4 ? 399 : 149)) : 29;
      const denLimit = largeStructuredOnly ? (effort>=6 ? 999 : (effort>=4 ? 399 : 149)) : 49;
      const binomNLimit = largeStructuredOnly ? (effort>=6 ? 220 : 150) : 99;
      const rows=[]; const seen=new Set();
      function addExpr(e){ addDatabaseCandidate(rows,seen,e,target,'database'); }
      function addWideAffineOffset(core, off, label='wide affine database'){
        if(!core || absBig(off)>9999n) return;
        let e=core;
        if(off!==0n){
          const d=makeDExpr(absBig(off), absBig(off).toString(), 'literal');
          e = off>0n ? combineD(core,'+',d,target,label) : combineD(core,'-',d,target,label);
        }
        addExpr(e);
      }
      function scaledWideCore(baseExpr, coeff, prod, kind){
        if(coeff===1) return baseExpr;
        return combineD(makeDExpr(BigInt(coeff), String(coeff), 'literal'), '*', baseExpr, prod, kind);
      }
      function scanWideAffineLargeIntegerSync(){
        if(!largeStructuredOnly) return;
        const dCap=9999n;
        const lo=target>dCap ? target-dCap : 0n;
        const hi=target+dCap;
        for(let A=100; A<=1000 && performance.now()<deadline; A++){
          let p=1n;
          for(let B=1; B<=100 && performance.now()<deadline; B++){
            p*=BigInt(A);
            if(B<7) continue;
            if(p>hi) break;
            for(let C=1; C<=9; C++){
              const prod=p*BigInt(C);
              if(prod<lo) continue;
              if(prod>hi) break;
              const base=makeDExpr(p, `${A}^${B}`, 'wide-affine-power', 1, 1);
              addWideAffineOffset(scaledWideCore(base,C,prod,'wide-affine-power-scale'), target-prod, 'wide power database');
            }
          }
        }
        for(let A=100; A<=1000 && performance.now()<deadline; A++){
          let v=1n;
          const maxB=Math.min(100, Math.floor(A/2));
          for(let B=1; B<=maxB && performance.now()<deadline; B++){
            v=(v*BigInt(A-B+1))/BigInt(B);
            if(B<7) continue;
            if(v>hi) break;
            for(let C=1; C<=9; C++){
              const prod=v*BigInt(C);
              if(prod<lo) continue;
              if(prod>hi) break;
              const base=makeDExpr(v, `binom(${A},${B})`, 'wide-affine-binom', 1, 1);
              addWideAffineOffset(scaledWideCore(base,C,prod,'wide-affine-binom-scale'), target-prod, 'wide binomial database');
            }
          }
        }
      }
      const cap=target*220n+BigInt(Math.max(100000, offsetLimit*10));
      scanWideAffineLargeIntegerSync();
      const powList=[];
      for(let B=2; B<=baseLimit && performance.now()<deadline; B++){
        let v=1n;
        for(let C=1; C<=39; C++){
          v*=BigInt(B); if(v>cap*50n) break;
          const e=makeDExpr(v, `${B}^${C}`, 'db-power', 1, 1);
          e._base=B; e._exp=C; e._family='power';
          powList.push(e);
        }
      }
      const binomList=[];
      for(let B=2; B<=binomNLimit && performance.now()<deadline; B++){
        const maxC=Math.floor(B/2);
        for(let C=1; C<=maxC; C++){
          const v=binomBigCapped(BigInt(B), BigInt(C), cap*50n); if(v===null) break;
          const e=makeDExpr(v, `binom(${B},${C})`, 'db-binom', 1, 1); e._n=B; e._k=C; e._family='binom'; binomList.push(e);
        }
      }
      const factList=[];
      let f=1n;
      for(let B=1; B<=Math.min(120, baseLimit) && performance.now()<deadline; B++){
        f*=BigInt(B);
        if(B>=4){
          if(f>cap*100n && B>30) break;
          const e=makeDExpr(f, `${B}!`, 'db-factorial', 1, 1); e._n=B; e._family='factorial'; factList.push(e);
        }
      }
      function scanLinear(list, label){
        for(const x of list){
          if(performance.now()>deadline) return;
          for(let A=1; A<=coeffLimit; A++){
            if((A & 31)===0 && performance.now()>deadline) return;
            const a=makeDExpr(BigInt(A), String(A), 'literal');
            const prod=x.v*BigInt(A);
            const D=target-prod;
            if(D>=-offsetBig && D<=offsetBig){
              const d=compactLiteralD(absBig(D));
              let e=combineD(a,'*',x,prod,'db-product');
              if(D>0n) e=combineD(e,'+',d,target,'db-offset');
              else if(D<0n) e=combineD(e,'-',d,target,'db-offset');
              addExpr(e);
            }
            if(A>=2){
              const q=roundDiv(x.v, BigInt(A));
              const Dr=target-q;
              if(Dr>=-offsetBig && Dr<=offsetBig){
                const ratio=makeRoundedDivDExpr(q,x,a,x.v,BigInt(A),(x.ops||0)+1, (x.depth||0)+1);
                let e=ratio; const d=compactLiteralD(absBig(Dr));
                if(Dr>0n) e=combineD(ratio,'+',d,target,'db-offset');
                else if(Dr<0n) e=combineD(ratio,'-',d,target,'db-offset');
                addExpr(e);
              }
            }
          }
        }
      }
      scanLinear(powList,'power');
      scanLinear(binomList,'binom');
      function scanExtendedDenominators(list){
        for(const x of list){
          if(performance.now()>deadline) return;
          for(let off=-offsetLimit; off<=offsetLimit; off++){
            if((off & 255)===0 && performance.now()>deadline) return;
            const q=target-BigInt(off);
            if(q<=0n) continue;
            const a0=x.v/q;
            for(let da=-3n; da<=3n; da++){
              const A=a0+da;
              if(A<2n || A>BigInt(literalLimit)) continue;
              const mode=roundModeForDiv(x.v,A,q);
              if(mode==='round') continue;
              const a=makeDExpr(A, A.toString(), 'literal');
              let e=makeRoundedDivDExpr(q,x,a,x.v,A,(x.ops||0)+1,(x.depth||0)+1);
              if(off!==0){
                const d=compactLiteralD(BigInt(Math.abs(off)));
                e=off>0 ? combineD(e,'+',d,target,'db-offset') : combineD(e,'-',d,target,'db-offset');
              }
              addExpr(e);
            }
          }
        }
      }
      scanExtendedDenominators(powList);
      scanExtendedDenominators(binomList);
      // Extra v6.1 database families: small-height rational multiples, two-term power sums,
      // mixed binomial/power sums, and products of two compact powers. These are deterministic
      // reverse checks, not random templates: every candidate is exact and then ranked.
      function gcdNum(a,b){ a=Math.abs(Number(a)); b=Math.abs(Number(b)); while(b){ const t=a%b; a=b; b=t; } return a; }
      function literalD(n){ return makeDExpr(BigInt(n), String(n), 'literal'); }
      function addOffsetExpr(core, off){
        let e=core;
        if(off!==0){ const d=literalD(Math.abs(off)); e = off>0 ? combineD(core,'+',d,target,'db-offset') : combineD(core,'-',d,target,'db-offset'); }
        addExpr(e);
      }
      function fractionText(A,D,x){
        if(A===D) return x.s;
        if(A===1) return `${shortOperandD(x,'/')}/${D}`;
        return `${A}/${D}·${shortOperandD(x,'*')}`;
      }
      function scanFractionalScale(list){
        for(const x of list){
          if(performance.now()>deadline) return;
          const base = x._base || x._n || 1;
          for(let A=1; A<=Math.min(coeffLimit, largeStructuredOnly ? 999 : 39); A++){
            if((A & 31)===0 && performance.now()>deadline) return;
            for(let D=2; D<=denLimit; D++){
              if((D & 63)===0 && performance.now()>deadline) return;
              if(gcdNum(A,D)!==1) continue;
              if(x._base && gcdNum(D,x._base)!==1) continue;
              const num=x.v*BigInt(A);
              const qFloor=num/BigInt(D);
              const qCeil=(num+BigInt(D)-1n)/BigInt(D);
              if(num%BigInt(D)===0n){
                const E=target-qFloor;
                if(E>=-offsetBig && E<=offsetBig){
                  const core=makeDExpr(qFloor, fractionText(A,D,x), 'db-rational-scale', (x.ops||0)+2, (x.depth||0)+1);
                  addOffsetExpr(core, Number(E));
                }
              }
              for(const [q,mode] of [[qFloor,'floor'],[qCeil,'ceil']]){
                const E=target-q;
                if(E>=-offsetBig && E<=offsetBig){
                  const core=makeDExpr(q, `${mode}(${fractionText(A,D,x)})`, `db-${mode}-rational-scale`, (x.ops||0)+3, (x.depth||0)+2);
                  addOffsetExpr(core, Number(E));
                }
              }
            }
          }
        }
      }
      function bestByValue(list){
        const m=new Map();
        for(const e of list){ const k=e.v.toString(); const old=m.get(k); if(!old || cmpExpr(e,old)<0) m.set(k,e); }
        return m;
      }
      const powMap=bestByValue(powList);
      const binomMap=bestByValue(binomList);
      const factMap=bestByValue(factList);
      const powPool=[...powMap.values()].sort(cmpExpr).slice(0, 6200);
      const binomPool=[...binomMap.values()].sort(cmpExpr).slice(0, 5200);
      const factPool=[...factMap.values()].sort(cmpExpr).slice(0, 80);
      function scanSignedPair(leftList, rightMap, label){
        for(const x of leftList){
          if(performance.now()>deadline) return;
          for(let E=-offsetLimit; E<=offsetLimit; E++){
            if((E & 255)===0 && performance.now()>deadline) return;
            const wantSum=target-BigInt(E)-x.v;
            const y=rightMap.get(wantSum.toString());
            if(y){
              let core=combineD(x,'+',y,x.v+y.v,`db-${label}-sum`);
              addOffsetExpr(core,E);
            }
            const wantDiff=x.v+BigInt(E)-target;
            const z=rightMap.get(wantDiff.toString());
            if(z && wantDiff>0n){
              let core=combineD(x,'-',z,x.v-z.v,`db-${label}-diff`);
              addOffsetExpr(core,E);
            }
          }
        }
      }
      function scanProductPair(leftList, rightMap, label){
        for(const x of leftList){
          if(performance.now()>deadline) return;
          if(x.v<=1n) continue;
          for(let E=-offsetLimit; E<=offsetLimit; E++){
            if((E & 255)===0 && performance.now()>deadline) return;
            const q=target-BigInt(E);
            if(q>0n && q%x.v===0n){
              const y=rightMap.get((q/x.v).toString());
              if(y){ const core=combineD(x,'*',y,x.v*y.v,`db-${label}-product`); addOffsetExpr(core,E); }
            }
          }
        }
      }
      scanFractionalScale(powPool.slice(0,1600));
      scanFractionalScale(binomPool.slice(0,1200));
      scanFractionalScale(factPool);
      scanSignedPair(powPool, powMap, 'power-power');       // A^B ± C^D + E
      scanSignedPair(powPool.slice(0,3600), binomMap, 'power-binom');
      scanSignedPair(binomPool.slice(0,3200), powMap, 'binom-power');
      scanSignedPair(factPool, powMap, 'factorial-power');
      scanProductPair(powPool.slice(0,3600), powMap, 'power-power');
      // A^D * B! + C and rounded B! / A^D + C
      for(const fac of factList){
        if(performance.now()>deadline) break;
        for(let A=2; A<=Math.min(coeffLimit, largeStructuredOnly ? 999 : 29); A++){
          if((A & 31)===0 && performance.now()>deadline) break;
          let ap=1n;
          for(let D=1; D<=39; D++){
            if((D & 7)===0 && performance.now()>deadline) break;
            ap*=BigInt(A); if(ap>cap*100n && fac.v>target+offsetBig) break;
            const ae=makeDExpr(ap, `${A}^${D}`, 'db-power', 1, 1);
            const prod=ap*fac.v;
            if(prod<=target+offsetBig){
              const C=target-prod;
              if(C>=-offsetBig && C<=offsetBig){
                const c=compactLiteralD(absBig(C));
                let e=combineD(ae,'*',fac,prod,'db-factorial-product');
                if(C>0n) e=combineD(e,'+',c,target,'db-offset');
                else if(C<0n) e=combineD(e,'-',c,target,'db-offset');
                addExpr(e);
              }
            }
            if(ap>0n){
              const q=roundDiv(fac.v,ap);
              const C=target-q;
              if(C>=-offsetBig && C<=offsetBig){
                const ratio=makeRoundedDivDExpr(q,fac,ae,fac.v,ap,(fac.ops||0)+(ae.ops||0)+1, Math.max(fac.depth||0,ae.depth||0)+1);
                const c=compactLiteralD(absBig(C));
                let e=ratio;
                if(C>0n) e=combineD(e,'+',c,target,'db-offset');
                else if(C<0n) e=combineD(e,'-',c,target,'db-offset');
                addExpr(e);
              }
            }
          }
        }
      }
      let out=selectDigitShortforms(rows,5);
      if(sign<0n){
        out=out.map(r=>{
          const expr=String(r.candidate).replace(/^[^:]+:\s*/, '');
          const label=String(r.candidate).includes(':') ? String(r.candidate).split(':')[0] : 'database';
          return {...r, candidate:`${label}: -(${expr})`, latex:`-\\left(${r.latex||exprToLatex(expr)}\\right)`, value:`exact = -${shortPrettyValue(target)}`};
        });
      }
      settings._databaseMs=Math.max(0, Math.round(performance.now()-dbStart));
      return out;
    }

    async function integerDatabaseRowsResponsive(settings, onProgress=null){
      const rawN=resolvedIntegerBig(settings);
      if(rawN===null || rawN===0n) return [];
      const sign=rawN<0n ? -1n : 1n;
      const target=absBig(rawN);
      const effort=Math.max(0, Math.min(7, Number(settings.shortEffort)||0));
      const td=decimalDigitCountBig(target);
      const dbStart=performance.now();
      // v7 restores the strong v6.1-v6.5 integer database families, but every
      // nested template scan is sliced through checkpoint(), so this status can
      // never monopolize the UI thread.  Effort 3 is deliberately close to the
      // old allowed database time; higher efforts get a real quality increase.
      const budgetMs=Math.min(td>=16 ? (effort>=4 ? 8200 : 6200) : 4400, (td>=16 ? 2200 : 1350) + effort*(td>=16 ? 720 : 360));
      settings._databaseBudgetMs=budgetMs;
      const deadline=dbStart + budgetMs;
      const offsetLimit=offsetLimitForTarget(target, effort);
      const offsetBig=BigInt(offsetLimit);
      const literalLimit=literalRangeLimitForTarget(target, effort);
      const largeStructured=td>=16;
      const baseLimit=largeStructured ? (effort>=6 ? 900 : (effort>=4 ? 620 : 320)) : (effort>=5 ? 160 : 120);
      const coeffLimit=largeStructured ? (effort>=6 ? 1200 : (effort>=4 ? 700 : 240)) : (effort>=5 ? 59 : 39);
      const denLimit=largeStructured ? (effort>=6 ? 1200 : (effort>=4 ? 700 : 240)) : (effort>=5 ? 79 : 59);
      const binomNLimit=largeStructured ? (effort>=6 ? 620 : (effort>=4 ? 430 : 240)) : (effort>=5 ? 150 : 120);
      const binomKLimit=largeStructured ? (effort>=6 ? 58 : (effort>=4 ? 42 : 28)) : (effort>=5 ? 24 : 18);
      const cap=target*360n+BigInt(Math.max(100000, Math.min(offsetLimit,999999)*12));
      const rows=[]; const seen=new Set();
      let lastYield=performance.now();
      let loops=0;
      async function checkpoint(label='database'){
        loops++;
        const now=performance.now();
        if(now>deadline || activeShortformRun?.stopped) return false;
        if(now-lastYield>18 || (loops&255)===0){
          lastYield=now;
          settings._databaseMs=Math.round(now-dbStart);
          settings._databasePhase=label;
          if(onProgress) onProgress({elapsed:settings._databaseMs, budgetMs, rows:rows.length, label});
          await yieldToUI();
        }
        return performance.now()<deadline && !activeShortformRun?.stopped;
      }
      function addExpr(e){ addDatabaseCandidate(rows,seen,e,target,'database'); }
      function addOffsetExpr(core, off, label='db-offset'){
        if(!core || absBig(off)>offsetBig) return;
        let e=core;
        if(off!==0n){
          const d=compactLiteralD(absBig(off));
          e = off>0n ? combineD(core,'+',d,target,label) : combineD(core,'-',d,target,label);
        }
        addExpr(e);
      }
      function addWideAffineOffset(core, off, label='wide affine database'){
        // The v10.7.1 wide A^B/binomial comparison intentionally uses a tight
        // visible correction window: for 16+ digit inputs D is at most ±9999.
        if(!core || absBig(off)>9999n) return;
        let e=core;
        if(off!==0n){
          const d=makeDExpr(absBig(off), absBig(off).toString(), 'literal');
          e = off>0n ? combineD(core,'+',d,target,label) : combineD(core,'-',d,target,label);
        }
        addExpr(e);
      }
      function scaledWideCore(baseExpr, coeff, prod, kind){
        if(coeff===1) return baseExpr;
        return combineD(makeDExpr(BigInt(coeff), String(coeff), 'literal'), '*', baseExpr, prod, kind);
      }
      async function scanWideAffineLargeInteger(){
        if(!largeStructured) return;
        const dCap=9999n;
        const lo=target>dCap ? target-dCap : 0n;
        const hi=target+dCap;
        for(let A=100; A<=1000; A++){
          if(!await checkpoint('wide powers')) return;
          let p=1n;
          for(let B=1; B<=100; B++){
            if((B&15)===0 && !await checkpoint('wide powers')) return;
            p*=BigInt(A);
            if(B<7) continue;
            if(p>hi) break;
            for(let C=1; C<=9; C++){
              const prod=p*BigInt(C);
              if(prod<lo) continue;
              if(prod>hi) break;
              const base=makeDExpr(p, `${A}^${B}`, 'wide-affine-power', 1, 1);
              addWideAffineOffset(scaledWideCore(base,C,prod,'wide-affine-power-scale'), target-prod, 'wide power database');
            }
          }
        }
        for(let A=100; A<=1000; A++){
          if(!await checkpoint('wide binomials')) return;
          let v=1n;
          const maxB=Math.min(100, Math.floor(A/2));
          for(let B=1; B<=maxB; B++){
            if((B&15)===0 && !await checkpoint('wide binomials')) return;
            v=(v*BigInt(A-B+1))/BigInt(B);
            if(B<7) continue;
            if(v>hi) break;
            for(let C=1; C<=9; C++){
              const prod=v*BigInt(C);
              if(prod<lo) continue;
              if(prod>hi) break;
              const base=makeDExpr(v, `binom(${A},${B})`, 'wide-affine-binom', 1, 1);
              addWideAffineOffset(scaledWideCore(base,C,prod,'wide-affine-binom-scale'), target-prod, 'wide binomial database');
            }
          }
        }
      }
      await scanWideAffineLargeInteger();
      const powList=[];
      for(let B=2; B<=baseLimit; B++){
        if(!await checkpoint('powers')) break;
        let v=1n;
        for(let C=1; C<=64; C++){
          if((C&7)===0 && !await checkpoint('powers')) break;
          v*=BigInt(B);
          if(v>cap*80n) break;
          const e=makeDExpr(v, C===1 ? String(B) : `${B}^${C}`, 'db-power', C===1?0:1, C===1?0:1);
          e._base=B; e._exp=C; e._family='power';
          powList.push(e);
        }
      }
      const binomList=[];
      for(let N=2; N<=binomNLimit; N++){
        if(!await checkpoint('binomial')) break;
        const maxK=Math.min(binomKLimit, Math.floor(N/2));
        for(let K=1; K<=maxK; K++){
          if((K&7)===0 && !await checkpoint('binomial')) break;
          const v=binomBigCapped(BigInt(N), BigInt(K), cap*80n); if(v===null) break;
          const e=makeDExpr(v, `binom(${N},${K})`, 'db-binom', 1, 1);
          e._n=N; e._k=K; e._family='binom';
          binomList.push(e);
        }
      }
      const factList=[]; let f=1n;
      for(let N=1; N<=Math.min(180, baseLimit); N++){
        if(!await checkpoint('factorial')) break;
        f*=BigInt(N);
        if(N>=4){
          if(f>cap*120n && N>38) break;
          const e=makeDExpr(f, `${N}!`, 'db-factorial', 1, 1); e._n=N; e._family='factorial'; factList.push(e);
        }
      }
      async function scanPowerBinomSubstringDatabase(){
        // v10: for every 9+ digit integer target, ask whether the decimal input
        // appears as a contiguous digit substring of compact r*A^B or
        // r*binom(A,B) values.  The substring database is intentionally reported
        // as at most one result: many families such as A and 10A share long digit
        // runs, so the best representative is chosen by smallest A/N first, then
        // expression digit-count, then digit-sum.  The scan is fully budgeted and
        // checkpointed so it cannot monopolize high-effort runs.
        const targetStr=target.toString();
        if(targetStr.length<9 || sign<0n) return;
        const maxDigits=100;
        const minWholeDigits=Math.max(16, targetStr.length);
        const ten100=10n**100n;
        const rList=[];
        for(let r=1;r<=20;r++) rList.push(r);
        const seenSub=new Set();
        let bestSub=null;
        function shortDigits(s){
          s=String(s);
          return s.length>72 ? `${s.slice(0,34)}…${s.slice(-34)} (${s.length} digits)` : s;
        }
        function digitSumText(s){
          return [...String(s)].reduce((a,ch)=>a+(/[0-9]/.test(ch)?Number(ch):0),0);
        }
        function betterSubstring(a,b){
          if(!b) return true;
          const cmp=(a.A-b.A) || (a.exprDigits-b.exprDigits) || (a.exprDigitSum-b.exprDigitSum) || (a.r-b.r) || (a.B-b.B) || (a.pos-b.pos);
          return cmp<0;
        }
        function considerSubstring(expr, exprLatex, whole, family, meta){
          const text=whole.toString();
          if(text.length<minWholeDigits || text.length>maxDigits) return;
          const pos=text.indexOf(targetStr);
          if(pos<0) return;
          const key=`${family}|${meta.A}|${meta.B}|${meta.r}|${pos}|${text.length}`;
          if(seenSub.has(key)) return;
          seenSub.add(key);
          const exprDigits=digitCountExpr(expr);
          const exprDigitSum=digitSumText(expr);
          const digits=exprDigits+targetStr.length;
          const beauty=shortRank({s:expr,digits:exprDigits,ops:2,depth:1}) + 1200000 + pos*5;
          const exact=text===targetStr;
          const row={
            candidate:`database substring${exact?' exact':''}: ${targetStr} in ${expr}`,
            copyCandidate:expr,
            latex:`${exprLatex || exprToLatex(expr)}\\;\\text{ contains }\\;${targetStr}`,
            value: exact ? `exact decimal string = ${shortDigits(text)}` : `target digits occur at positions ${pos+1}-${pos+targetStr.length} of ${shortDigits(text)}`,
            copyValue:text,
            err:0,
            hideError:true,
            noCandidateCopy:false,
            beauty,
            feature:'substring',
            digits,
            ops:2,
            substringMatch:true
          };
          const candidate={row, A:Number(meta.A)||0, B:Number(meta.B)||0, r:Number(meta.r)||1, pos, exprDigits, exprDigitSum};
          if(betterSubstring(candidate,bestSub)) bestSub=candidate;
        }
        const substringBudgetEnd=Math.min(deadline, performance.now()+Math.max(130, Math.min(700, 220+effort*70)));
        const baseMax=effort>=6 ? 2400 : (effort>=4 ? 1350 : 620);
        const expMax=effort>=6 ? 230 : (effort>=4 ? 170 : 115);
        for(let A=2; A<=baseMax; A++){
          if(performance.now()>substringBudgetEnd || !await checkpoint('substring powers')) break;
          let v=1n;
          for(let B=1; B<=expMax; B++){
            if((B&15)===0 && (performance.now()>substringBudgetEnd || !await checkpoint('substring powers'))) break;
            v*=BigInt(A);
            if(v>=ten100) break;
            if(B<5) continue;
            const vDigits=v.toString().length;
            if(vDigits>maxDigits) break;
            if(vDigits+2<targetStr.length) continue;
            for(const r of rList){
              const whole=v*BigInt(r);
              if(whole>=ten100) break;
              const expr=r===1 ? `${A}^${B}` : `${r}·${A}^${B}`;
              considerSubstring(expr, exprToLatex(expr), whole, 'power', {A, B, r});
            }
          }
        }
        const nMax=effort>=6 ? 760 : (effort>=4 ? 520 : 300);
        const kMax=effort>=6 ? 96 : (effort>=4 ? 70 : 48);
        const cap=ten100-1n;
        for(let N=6; N<=nMax; N++){
          if(performance.now()>substringBudgetEnd || !await checkpoint('substring binomial')) break;
          const maxK=Math.min(kMax, Math.floor(N/2));
          for(let K=5; K<=maxK; K++){
            if((K&7)===0 && (performance.now()>substringBudgetEnd || !await checkpoint('substring binomial'))) break;
            const bv=binomBigCapped(BigInt(N), BigInt(K), cap);
            if(bv===null) break;
            const ds=bv.toString().length;
            if(ds>maxDigits) break;
            if(ds+2<targetStr.length) continue;
            for(const r of rList){
              const whole=bv*BigInt(r);
              if(whole>=ten100) break;
              const expr=r===1 ? `binom(${N},${K})` : `${r}·binom(${N},${K})`;
              considerSubstring(expr, exprToLatex(expr), whole, 'binom', {A:N, B:K, r});
            }
          }
        }
        if(bestSub && !activeShortformRun?.stopped){
          const row=bestSub.row;
          const eq='substring-best|'+targetStr;
          if(!seen.has(eq)){ seen.add(eq); rows.push(row); }
        }
      }
      await scanPowerBinomSubstringDatabase();
      async function scanLinear(list, sliceLimit){
        const slice=list.slice(0, sliceLimit);
        for(const x of slice){
          if(!await checkpoint('linear templates')) return;
          for(let A=1; A<=coeffLimit; A++){
            if((A&15)===0 && !await checkpoint('linear templates')) return;
            const a=makeDExpr(BigInt(A), String(A), 'literal');
            const prod=x.v*BigInt(A);
            const D=target-prod;
            if(D>=-offsetBig && D<=offsetBig){
              const core=A===1 ? x : combineD(a,'*',x,prod,'db-product');
              addOffsetExpr(core,D,'db-offset');
            }
            if(A>=2){
              const q=roundDiv(x.v, BigInt(A));
              const Dr=target-q;
              if(Dr>=-offsetBig && Dr<=offsetBig){
                const ratio=makeRoundedDivDExpr(q,x,a,x.v,BigInt(A),(x.ops||0)+1,(x.depth||0)+1);
                addOffsetExpr(ratio,Dr,'db-offset');
              }
            }
          }
        }
      }
      await scanLinear(powList, largeStructured ? 6200 : 5200);
      await scanLinear(binomList, largeStructured ? 5600 : 4600);
      await scanLinear(factList, 160);
      async function scanExtendedDenominators(list, sliceLimit){
        const slice=list.slice(0,sliceLimit);
        const offStep=Math.max(1, Math.ceil((offsetLimit*2+1)/Math.max(900, 900+effort*360)));
        for(const x of slice){
          if(!await checkpoint('extended denominators')) return;
          for(let off=-offsetLimit; off<=offsetLimit; off+=offStep){
            if((off&255)===0 && !await checkpoint('extended denominators')) return;
            const q=target-BigInt(off);
            if(q<=0n) continue;
            const a0=x.v/q;
            for(let da=-4n; da<=4n; da++){
              const A=a0+da;
              if(A<2n || A>BigInt(literalLimit)) continue;
              const mode=roundModeForDiv(x.v,A,q);
              if(mode==='invalid') continue;
              const a=compactLiteralD(A, {denominator:true});
              const core=makeRoundedDivDExpr(q,x,a,x.v,A,(x.ops||0)+(a.ops||0)+1,Math.max(x.depth||0,a.depth||0)+1);
              addOffsetExpr(core,BigInt(off),'db-offset');
            }
          }
        }
      }
      await scanExtendedDenominators(powList, largeStructured ? 3600 : 3000);
      await scanExtendedDenominators(binomList, largeStructured ? 3000 : 2400);
      function bestByValue(list){
        const m=new Map();
        for(const e of list){ const k=e.v.toString(); const old=m.get(k); if(!old || cmpExpr(e,old)<0) m.set(k,e); }
        return m;
      }
      function arrByValue(m){ return [...m.values()].sort((a,b)=>a.v<b.v?-1:a.v>b.v?1:0); }
      function lowerBoundArr(arr, value){ let lo=0, hi=arr.length; while(lo<hi){ const mid=(lo+hi)>>1; if(arr[mid].v<value) lo=mid+1; else hi=mid; } return lo; }
      async function forRange(arr, lo, hi, maxCount, cb, label){
        if(hi<lo) return;
        let idx=lowerBoundArr(arr, lo), count=0;
        while(idx<arr.length && arr[idx].v<=hi && count<maxCount){
          if((count&15)===0 && !await checkpoint(label)) return;
          cb(arr[idx]); idx++; count++;
        }
      }
      const powMap=bestByValue(powList), binomMap=bestByValue(binomList), factMap=bestByValue(factList);
      const powArr=arrByValue(powMap), binomArr=arrByValue(binomMap), factArr=arrByValue(factMap);
      const powPool=[...powMap.values()].sort(cmpExpr).slice(0, largeStructured ? 7200 : 6200);
      const binomPool=[...binomMap.values()].sort(cmpExpr).slice(0, largeStructured ? 6200 : 5200);
      const factPool=[...factMap.values()].sort(cmpExpr).slice(0, 120);
      const pairTake=largeStructured ? (effort>=6 ? 42 : 30) : 26;
      async function scanSignedPair(leftList, rightArr, label){
        for(const x of leftList){
          if(!await checkpoint(`pair ${label}`)) return;
          // x + y + E = target
          await forRange(rightArr, target-offsetBig-x.v, target+offsetBig-x.v, pairTake, y=>{
            const core=combineD(x,'+',y,x.v+y.v,`db-${label}-sum`);
            addOffsetExpr(core,target-core.v,'db-offset');
          }, `pair ${label}`);
          // x - y + E = target
          await forRange(rightArr, x.v-target-offsetBig, x.v-target+offsetBig, pairTake, y=>{
            if(x.v>y.v){ const core=combineD(x,'-',y,x.v-y.v,`db-${label}-diff`); addOffsetExpr(core,target-core.v,'db-offset'); }
          }, `pair ${label}`);
        }
      }
      async function scanProductPair(leftList, rightArr, label){
        for(const x of leftList){
          if(!await checkpoint(`product ${label}`)) return;
          if(x.v<=1n) continue;
          const lo=(target>offsetBig ? target-offsetBig : 1n);
          const hi=target+offsetBig;
          const yLo=(lo + x.v - 1n)/x.v;
          const yHi=hi/x.v;
          await forRange(rightArr, yLo, yHi, pairTake, y=>{
            const prod=x.v*y.v;
            const core=combineD(x,'*',y,prod,`db-${label}-product`);
            addOffsetExpr(core,target-prod,'db-offset');
          }, `product ${label}`);
        }
      }
      async function scanFractionalScale(list, sliceLimit){
        const slice=list.slice(0, sliceLimit);
        const AMax=Math.min(coeffLimit, largeStructured ? (effort>=6?760:(effort>=4?520:260)) : 59);
        const DMax=Math.min(denLimit, largeStructured ? (effort>=6?760:(effort>=4?520:260)) : 79);
        for(const x of slice){
          if(!await checkpoint('fractional templates')) return;
          for(let A=1; A<=AMax; A++){
            if((A&15)===0 && !await checkpoint('fractional templates')) return;
            for(let D=2; D<=DMax; D++){
              if((D&31)===0 && !await checkpoint('fractional templates')) return;
              if(gcdBig(BigInt(A),BigInt(D))!==1n) continue;
              if(x._base && gcdBig(BigInt(D),BigInt(x._base))!==1n) continue;
              const num=x.v*BigInt(A);
              const den=BigInt(D);
              const qFloor=num/den;
              const qCeil=(num+den-1n)/den;
              const fracText=A===1 ? `${shortOperandD(x,'/')}/${D}` : `${A}/${D}·${shortOperandD(x,'*')}`;
              if(num%den===0n){
                const E=target-qFloor;
                if(E>=-offsetBig && E<=offsetBig){ const core=makeDExpr(qFloor, fracText, 'db-rational-scale', (x.ops||0)+2, (x.depth||0)+1); addOffsetExpr(core,E,'db-offset'); }
              }
              for(const [q,mode] of [[qFloor,'floor'],[qCeil,'ceil']]){
                const E=target-q;
                if(E>=-offsetBig && E<=offsetBig){ const core=makeDExpr(q, `${mode}(${fracText})`, `db-${mode}-rational-scale`, (x.ops||0)+3, (x.depth||0)+2); addOffsetExpr(core,E,'db-offset'); }
              }
            }
          }
        }
      }
      await scanFractionalScale(powPool, largeStructured ? 2600 : 1800);
      await scanFractionalScale(binomPool, largeStructured ? 2200 : 1400);
      await scanFractionalScale(factPool, 120);
      // Restored mixed structured families from older strong versions.
      await scanSignedPair(powPool, powArr, 'power-power');
      await scanSignedPair(powPool.slice(0, largeStructured?5200:3600), binomArr, 'power-binom');
      await scanSignedPair(binomPool.slice(0, largeStructured?4600:3200), powArr, 'binom-power');
      await scanSignedPair(binomPool.slice(0, largeStructured?3600:2600), binomArr, 'binom-binom');
      await scanSignedPair(factPool, powArr, 'factorial-power');
      await scanSignedPair(factPool, binomArr, 'factorial-binom');
      await scanProductPair(powPool.slice(0, largeStructured?5200:3600), powArr, 'power-power');
      await scanProductPair(powPool.slice(0, largeStructured?3600:2400), binomArr, 'power-binom');
      await scanProductPair(binomPool.slice(0, largeStructured?3200:2200), powArr, 'binom-power');
      // A^D * B! + C and floor/ceil(B! / A^D) + C.
      for(const fac of factPool){
        if(!await checkpoint('factorial power templates')) break;
        for(let A=2; A<=Math.min(coeffLimit, largeStructured?180:49); A++){
          if((A&15)===0 && !await checkpoint('factorial power templates')) break;
          let ap=1n;
          for(let D=1; D<=56; D++){
            if((D&7)===0 && !await checkpoint('factorial power templates')) break;
            ap*=BigInt(A); if(ap>cap*120n && fac.v>target+offsetBig) break;
            const ae=makeDExpr(ap, D===1 ? String(A) : `${A}^${D}`, 'db-power', D===1?0:1, D===1?0:1);
            const prod=ap*fac.v;
            if(prod<=target+offsetBig){ addOffsetExpr(combineD(ae,'*',fac,prod,'db-factorial-product'), target-prod, 'db-offset'); }
            if(ap>0n){
              const q=roundDiv(fac.v,ap);
              const C=target-q;
              if(C>=-offsetBig && C<=offsetBig){
                const ratio=makeRoundedDivDExpr(q,fac,ae,fac.v,ap,(fac.ops||0)+(ae.ops||0)+1,Math.max(fac.depth||0,ae.depth||0)+1);
                addOffsetExpr(ratio,C,'db-offset');
              }
            }
          }
        }
      }
      let out=selectDigitShortforms(rows, Math.max(5, Math.min(20, Number(settings.limit)||5)));
      if(sign<0n) out=applyIntegerSign(out, sign, target);
      settings._databaseMs=Math.max(0, Math.round(performance.now()-dbStart));
      settings._databasePhase='done';
      if(onProgress) onProgress({elapsed:settings._databaseMs, budgetMs, rows:out.length, label:'done'});
      await yieldToUI();
      return out;
    }
    function integerRowTargetValue(row){
      const raw=String(row?.copyValue ?? row?.value ?? '').trim();
      const m=raw.match(/-?\d+/);
      return m ? BigInt(m[0]) : null;
    }
    function integerRowFormulaIsValid(row){
      const cand=String(row?.candidate||'');
      // Substring rows are not formulas equal to the target; they report where
      // the target digits occur inside a larger exact integer, so do not apply
      // the formula=target validator to them.
      if(/database substring/.test(cand)) return true;
      const target=integerRowTargetValue(row);
      if(target===null) return true;
      const expr=parseLabelFreeCandidate(cand);
      return displayExprMatchesTarget(expr, target);
    }
    function selectDigitShortforms(rows, limit=5){
      const sorted=[...rows]
        .filter(r=>!/round\(/.test(String(r.candidate||'')+String(r.latex||'')))
        .filter(integerRowFormulaIsValid)
        .sort((a,b)=>(a.digits-b.digits)||(a.beauty-b.beauty)||(a.ops-b.ops)||a.candidate.length-b.candidate.length);
      const picked=[]; const usedKeys=new Set(); const usedFeatures=new Map();
      for(const r of sorted){
        if(picked.length>=limit) break;
        const key=candidateEquivalenceKey(r);
        if(usedKeys.has(key)) continue;
        const f=r.feature||exprFeature(parseLabelFreeCandidate(r.candidate));
        const cap = picked.length<3 ? 1 : 2;
        if((usedFeatures.get(f)||0)>=cap) continue;
        picked.push(r); usedKeys.add(key); usedFeatures.set(f,(usedFeatures.get(f)||0)+1);
      }
      for(const r of sorted){
        if(picked.length>=limit) break;
        const key=candidateEquivalenceKey(r);
        if(usedKeys.has(key)) continue;
        picked.push(r); usedKeys.add(key);
      }
      return picked.slice(0,limit);
    }
    function searchBudgetSequence(effort, target){
      const td=decimalDigitCountBig(target);
      const ceiling=[6,7,8,10,12,14,16,19][Math.max(0,Math.min(7,effort))];
      const maxBudget=Math.max(2, Math.min(td, ceiling));
      const seq=[];
      for(let d=2; d<=maxBudget; d++) seq.push(d);
      return seq;
    }
    function earlyShortformStop(selected, target, limit, budget, effort, elapsedMs){
      if(!selected.length) return false;
      const td=decimalDigitCountBig(target);
      const best=selected[0];
      if(best.digits<=2 && best.ops<=4) return true;
      if(selected.length>=Math.min(3,limit) && best.digits<=3 && elapsedMs>400) return true;
      if(selected.length>=limit && best.digits<=Math.min(4, Math.max(2,td-4)) && elapsedMs>650) return true;
      if(effort>=4 && selected.length>=limit && best.digits<=4 && budget>=4) return true;
      return false;
    }
    function applyIntegerSign(rows, sign, target){
      if(sign>=0n) return rows;
      return rows.map(r=>{
        const expr=String(r.candidate).replace(/^[^:]+:\s*/, '');
        const label=String(r.candidate).includes(':') ? String(r.candidate).split(':')[0] : 'shortform';
        return {...r, candidate:`${label}: -(${expr})`, latex:`-\\left(${r.latex||exprToLatex(expr)}\\right)`, value:`exact = -${shortPrettyValue(target)}`};
      });
    }
    function decimalDecompositionFallback(sign, target, limit=3){
      const td=decimalDigitCountBig(target);
      if(td<=1) return [];
      const rows=[];
      const maxK=Math.min(td-1, Math.max(1, Math.ceil(td/2)));
      for(let k=1;k<=maxK;k++){
        const base=10n**BigInt(k);
        const hi=target/base, lo=target%base;
        if(hi===0n) continue;
        const hiE=makeDExpr(hi, hi.toString(), 'literal');
        const powE=makeDExpr(base, `10^${k}`, 'decimal-base', 1, 1);
        let expr=combineD(hiE,'*',powE,hi*base,'decimal-split');
        if(lo>0n) expr=combineD(expr,'+',compactLiteralD(lo),target,'decimal-split');
        expr=validateDExprForTarget(expr, target);
        if(!expr) continue;
        expr.rank=shortRank(expr)+2500000;
        rows.push({candidate:`fallback exact: ${expr.s}`, latex:exprToLatex(expr.s), value:`exact = ${shortPrettyValue(target)}`, copyValue:target.toString(), err:0, hideError:true, beauty:expr.rank, feature:'fallback', digits:expr.digits, ops:expr.ops||2});
      }
      rows.sort((a,b)=>(a.digits-b.digits)||(a.beauty-b.beauty)||a.candidate.length-b.candidate.length);
      const out=rows.slice(0,limit);
      return sign<0n ? applyIntegerSign(out, sign, target) : out;
    }

    function lowerBoundBigArray(arr, value){
      let lo=0, hi=arr.length;
      while(lo<hi){ const mid=(lo+hi)>>1; if(arr[mid].v < value) lo=mid+1; else hi=mid; }
      return lo;
    }
    function sqrtFloorBig(n){
      n=BigInt(n); if(n<=0n) return 0n;
      let x=1n << BigInt(Math.ceil(n.toString(2).length/2));
      while(true){ const y=(x+n/x)>>1n; if(y>=x) return x*x>n ? x-1n : x; x=y; }
    }
    function powerRatioShortformFallback(sign, target, limit=3, ms=520){
      const deadline=performance.now()+ms;
      const td=decimalDigitCountBig(target);
      const maxDenDigits=Math.max(3, Math.min(6, Math.ceil(td*0.55)));
      const maxDen=10n**BigInt(maxDenDigits)-1n;
      const maxOff=10n**BigInt(Math.max(1, Math.min(5, Math.ceil(td*0.32))))-1n;
      const maxBase=td<=10 ? 6500 : 3600;
      const maxExp=td<=10 ? 90 : 70;
      const cap=target*maxDen*3n;
      const rows=[]; const seen=new Map();
      function addExpr(expr){
        if(!expr || expr.v!==target) return;
        expr=validateDExprForTarget(expr, target);
        if(!expr) return;
        expr.rank=shortRank(expr)-700000;
        const row={candidate:`fallback power-ratio: ${expr.s}`, latex:exprToLatex(expr.s), value:`exact = ${shortPrettyValue(target)}`, copyValue:target.toString(), err:0, hideError:true, beauty:expr.rank, feature:exprFeature(expr.s), digits:expr.digits, ops:expr.ops||3};
        const key=candidateEquivalenceKey(row);
        if(seen.has(key)){
          const idx=seen.get(key);
          const cmp=(row.digits-rows[idx].digits)||(row.beauty-rows[idx].beauty)||(row.candidate.length-rows[idx].candidate.length);
          if(cmp<0) rows[idx]=row;
          return;
        }
        seen.set(key, rows.length); rows.push(row);
      }
      function compactDenominatorExpr(den){
        let best=compactLiteralD(den, {denominator:true});
        const dn=Number(den);
        if(Number.isFinite(dn) && dn>1){
          for(let d=2; d<=8; d++){
            const approx=Math.max(2, Math.round(Math.pow(dn, 1/d)));
            for(let c=Math.max(2, approx-2); c<=approx+2; c++){
              const v=powBigCapped(BigInt(c), BigInt(d), den);
              if(v===den){
                const e=makeDExpr(den, `${c}^${d}`, 'fallback-den-power', 1, 1);
                if(cmpExpr(e,best)<0) best=e;
              }
            }
          }
        }
        return best;
      }
      function addPowerDiv(a,b,pow,den,denExpr,off){
        if(den<2n || den>maxDen) return;
        denExpr=denExpr || compactDenominatorExpr(den);
        const q=roundDiv(pow,den);
        if(target-q!==off) return;
        if(absBig(off)>maxOff) return;
        const powE=makeDExpr(pow, `${a}^${b}`, 'fallback-power', 1, 1);
        let e=makeRoundedDivDExpr(q,powE,denExpr,pow,den,(powE.ops||0)+(denExpr.ops||0)+1,Math.max(powE.depth||0,denExpr.depth||0)+1);
        if(!e) return;
        if(off!==0n){
          const d=compactLiteralD(absBig(off));
          e = off>0n ? combineD(e,'+',d,target,'fallback-offset') : combineD(e,'-',d,target,'fallback-offset');
        }
        addExpr(e);
      }
      const denPowers=[]; const denSeen=new Set();
      for(let c=2;c<=420 && performance.now()<deadline;c++){
        let v=BigInt(c);
        for(let d=1;d<=8;d++){
          if(v>maxDen) break;
          const key=v.toString();
          if(!denSeen.has(key)){
            denSeen.add(key);
            denPowers.push({v, expr:makeDExpr(v, d===1 ? String(c) : `${c}^${d}`, 'fallback-den-power', d===1?0:1, d===1?0:1)});
          }
          v*=BigInt(c);
        }
      }
      denPowers.sort((a,b)=>a.v<b.v?-1:a.v>b.v?1:0);
      for(let a=2; a<=maxBase && performance.now()<deadline; a++){
        let pow=BigInt(a)*BigInt(a);
        for(let b=2; b<=maxExp && performance.now()<deadline; b++){
          if(pow>cap) break;
          if(pow*2n>=target){
            const q0=roundDiv(pow,target);
            for(let delta=-9; delta<=9; delta++){
              if(performance.now()>deadline || isUserInputPending()) break;
              const den=q0+BigInt(delta);
              if(den>=2n && den<=maxDen){
                const q=roundDiv(pow,den); const off=target-q;
                if(absBig(off)<=maxOff) addPowerDiv(a,b,pow,den,compactDenominatorExpr(den),off);
              }
            }
            const pos=lowerBoundBigArray(denPowers, q0);
            for(let j=Math.max(0,pos-10); j<Math.min(denPowers.length,pos+11); j++){
              if(performance.now()>deadline || isUserInputPending()) break;
              const den=denPowers[j]; const q=roundDiv(pow,den.v); const off=target-q;
              if(absBig(off)<=maxOff) addPowerDiv(a,b,pow,den.v,den.expr,off);
            }
          }
          pow*=BigInt(a);
        }
      }
      rows.sort((a,b)=>(a.digits-b.digits)||(a.beauty-b.beauty)||a.candidate.length-b.candidate.length);
      const out=rows.slice(0,limit);
      return sign<0n ? applyIntegerSign(out, sign, target) : out;
    }

    function diverseShortformFallback(sign, target, limit=5, ms=900){
      const deadline=performance.now()+ms;
      const td=decimalDigitCountBig(target);
      const maxOff=10n**BigInt(Math.max(1, Math.min(5, Math.ceil(td*0.28))))-1n;
      const cap=target*250n+1000000n;
      const rows=[]; const seen=new Set();
      function addRow(expr,label='fallback diverse'){
        if(!expr || expr.v!==target) return;
        expr=validateDExprForTarget(expr, target);
        if(!expr) return;
        expr.rank=shortRank(expr)-520000;
        const row={candidate:`${label}: ${expr.s}`, latex:exprToLatex(expr.s), value:`exact = ${shortPrettyValue(target)}`, copyValue:target.toString(), err:0, hideError:true, beauty:expr.rank, feature:exprFeature(expr.s), digits:expr.digits, ops:expr.ops||3};
        const key=candidateEquivalenceKey(row); if(seen.has(key)) return; seen.add(key); rows.push(row);
      }
      function addOffset(core, off, label){
        if(absBig(off)>maxOff) return;
        let e=core;
        if(off!==0n){ const d=compactLiteralD(absBig(off)); e=off>0n ? combineD(core,'+',d,target,'diverse-offset') : combineD(core,'-',d,target,'diverse-offset'); }
        addRow(e,label);
      }
      function makeStructuredList(){
        const list=[]; const seenVals=new Map();
        function push(e){
          if(!e || e.v<=0n || e.v>cap*20n) return;
          const k=e.v.toString(); const old=seenVals.get(k);
          if(!old || cmpExpr(e,old)<0) seenVals.set(k,e);
        }
        for(let a=2; a<=140 && performance.now()<deadline; a++){
          let v=BigInt(a);
          for(let b=1; b<=56; b++){
            if(v>cap*20n) break;
            const e=makeDExpr(v, b===1 ? String(a) : `${a}^${b}`, b===1?'literal':'diverse-power', b===1?0:1, b===1?0:1);
            e._base=a; e._exp=b; push(e); v*=BigInt(a);
          }
        }
        let f=1n;
        for(let a=1; a<=48 && performance.now()<deadline; a++){
          f*=BigInt(a); if(f>cap*20n && a>18) break;
          if(a>=4) push(makeDExpr(f, `${a}!`, 'diverse-factorial', 1, 1));
        }
        for(let n=5; n<=180 && performance.now()<deadline; n++){
          for(let k=2; k<=Math.min(22, Math.floor(n/2)); k++){
            const v=binomBigCapped(BigInt(n),BigInt(k),cap*20n); if(v===null) break;
            push(makeDExpr(v, `binom(${n},${k})`, 'diverse-binom', 1, 1));
          }
        }
        return [...seenVals.values()].sort((x,y)=>x.v<y.v?-1:x.v>y.v?1:0);
      }
      const powers=makeStructuredList();
      function nearByValue(v, radius=10){ const p=lowerBoundBigArray(powers,v); return powers.slice(Math.max(0,p-radius), Math.min(powers.length,p+radius+1)); }
      // A^B ± C^D + E
      for(const x of powers){
        if(performance.now()>deadline) break;
        for(const y of nearByValue(target-x.v,8)) addOffset(combineD(x,'+',y,x.v+y.v,'diverse-power-sum'), target-(x.v+y.v), 'fallback structured-sum');
        if(x.v>target){ for(const y of nearByValue(x.v-target,8)) addOffset(combineD(x,'-',y,x.v-y.v,'diverse-power-diff'), target-(x.v-y.v), 'fallback structured-difference'); }
        else { for(const y of nearByValue(target+x.v,8)) addOffset(combineD(y,'-',x,y.v-x.v,'diverse-power-diff'), target-(y.v-x.v), 'fallback structured-difference'); }
      }
      // A^B * C^D + E, where each side is compact.
      for(const x of powers){
        if(performance.now()>deadline) break;
        if(x.v<=1n || x.v>target+maxOff) continue;
        const q=target/x.v;
        for(const y of nearByValue(q,6)){
          const prod=x.v*y.v;
          addOffset(combineD(x,'*',y,prod,'diverse-power-product'), target-prod, 'fallback structured-product');
        }
      }
      // floor/ceil((A/D)*B^C)+E with two-digit A,D and gcd(A,D)=1.
      function gcdSmall(a,b){ a=Math.abs(a); b=Math.abs(b); while(b){ const t=a%b; a=b; b=t; } return a; }
      for(const p of powers){
        if(performance.now()>deadline) break;
        if(p.v<10n) continue;
        for(let D=2; D<=99; D++){
          if((D&15)===0 && (performance.now()>deadline || isUserInputPending())) break;
          const approxA=Number((target*BigInt(D)*1000000n)/(p.v||1n))/1000000;
          for(let A=Math.max(1,Math.floor(approxA)-1); A<=Math.min(99,Math.ceil(approxA)+1); A++){
            if(A<1 || gcdSmall(A,D)!==1) continue;
            const num=p.v*BigInt(A);
            const df=BigInt(D);
            const qF=floorDiv(num,df), qC=ceilDiv(num,df), qR=roundDiv(num,df);
            const aE=makeDExpr(BigInt(A), String(A), 'literal');
            const dE=makeDExpr(BigInt(D), String(D), 'literal');
            const fracText=`${A}/${D}·${shortOperandD(p,'*')}`;
            for(const [q,mode] of [[qF,'floor'],[qC,'ceil']]){
              const core=makeDExpr(q, `${mode}(${fracText})`, `diverse-${mode}-frac-power`, (p.ops||0)+3, (p.depth||0)+2);
              addOffset(core, target-q, 'fallback fractional-scale');
            }
          }
        }
      }
      // v7.2 extra natural fallback families.  These deliberately use the
      // actual expression as the label payload, avoiding misleading A^B+C^D
      // labels when one side is a binomial or factorial.
      // 1) binom(n,k) ± compact power/factorial offset + E.
      const binoms=powers.filter(e=>/binom\(/.test(e.s)).sort(cmpExpr).slice(0,900);
      const nonBinoms=powers.filter(e=>!/binom\(/.test(e.s)).sort(cmpExpr).slice(0,1600);
      for(const b of binoms){
        if(performance.now()>deadline) break;
        for(const y of nearByValue(target-b.v,5)) addOffset(combineD(b,'+',y,b.v+y.v,'diverse-binom-plus'), target-(b.v+y.v), 'fallback binomial-mix');
        if(b.v>target){ for(const y of nearByValue(b.v-target,5)) addOffset(combineD(b,'-',y,b.v-y.v,'diverse-binom-minus'), target-(b.v-y.v), 'fallback binomial-mix'); }
      }
      // 2) A * binom(n,k) ± E with small A, useful for combinatorial multiples.
      for(const b of binoms){
        if(performance.now()>deadline) break;
        if(b.v===0n) continue;
        const approx=Number(target/b.v);
        if(!Number.isFinite(approx)) continue;
        for(let A=Math.max(2,Math.floor(approx)-2); A<=Math.min(999,Math.ceil(approx)+2); A++){
          if(performance.now()>deadline || isUserInputPending()) break;
          const aE=compactLiteralD(BigInt(A));
          const prod=b.v*BigInt(A);
          addOffset(combineD(aE,'*',b,prod,'diverse-binom-multiple'), target-prod, 'fallback binomial-multiple');
        }
      }
      // 3) near square / near triangular core: m^2 ± compact expression.
      const roots=[sqrtFloorBig(target), sqrtFloorBig(target+maxOff), sqrtFloorBig(target>maxOff?target-maxOff:0n)];
      for(const r0 of roots){
        for(let delta=-3; delta<=3; delta++){
          if(performance.now()>deadline) break;
          const r=r0+BigInt(delta); if(r<2n) continue;
          const sq=r*r;
          addOffset(makeDExpr(sq, `${r.toString()}^2`, 'diverse-square', 1, 1), target-sq, 'fallback near-square');
          const tri=r*(r+1n)/2n;
          addOffset(makeDExpr(tri, `binom(${(r+1n).toString()},2)`, 'diverse-triangular', 1, 1), target-tri, 'fallback near-triangular');
        }
      }
      // 4) quotient with structured numerator plus small outside offset: floor((S±R)/d)+E.
      const smallDenoms=[];
      for(let d=2; d<=180; d++) smallDenoms.push(compactLiteralD(BigInt(d), {denominator:true}));
      for(const dE of smallDenoms){
        if(performance.now()>deadline) break;
        const d=dE.v;
        const center=target*d;
        for(const sExpr of nearByValue(center,7)){
          if(performance.now()>deadline || isUserInputPending()) break;
          const q=floorDiv(sExpr.v,d);
          addOffset(makeDExpr(q, `floor(${shortOperandD(sExpr,'/')}/${shortOperandD(dE,'/')})`, 'diverse-floor-quotient', (sExpr.ops||0)+(dE.ops||0)+2, Math.max(sExpr.depth||0,dE.depth||0)+1), target-q, 'fallback structured-quotient');
          const cq=ceilDiv(sExpr.v,d);
          addOffset(makeDExpr(cq, `ceil(${shortOperandD(sExpr,'/')}/${shortOperandD(dE,'/')})`, 'diverse-ceil-quotient', (sExpr.ops||0)+(dE.ops||0)+2, Math.max(sExpr.depth||0,dE.depth||0)+1), target-cq, 'fallback structured-quotient');
        }
      }

      rows.sort((a,b)=>(a.digits-b.digits)||(a.beauty-b.beauty)||(a.ops-b.ops)||a.candidate.length-b.candidate.length);
      const out=selectDigitShortforms(rows,limit);
      return sign<0n ? applyIntegerSign(out,sign,target) : out;
    }

    async function integerShortformRowsAsync(settings, onUpdate){
      const rawN=resolvedIntegerBig(settings);
      if(rawN===null || rawN===0n) return [];
      const run=activeShortformRun;
      const startTime=performance.now();
      const effort=Math.max(0, Math.min(7, Number(settings.shortEffort)||0));
      const sign=rawN<0n ? -1n : 1n;
      const target=absBig(rawN);
      const targetDigitsForBudget=decimalDigitCountBig(target);
      let shortBudgetMs=[900,1700,3200,5200,7000,10500,16500,26000][effort];
      if(targetDigitsForBudget>=9 && targetDigitsForBudget<16){
        // v10: medium integers were the most common source of high-effort
        // stalls.  Keep all exact families enabled, but make each Continue level
        // a predictable finite pass and spend the saved time on structured DB
        // probes rather than giant ratio/fraction attempts.
        shortBudgetMs=Math.min(shortBudgetMs, [620,850,1200,1750,2600,3800,5600,8000][effort]);
      }
      if(targetDigitsForBudget<=8 && effort>=4){
        shortBudgetMs=Math.min(shortBudgetMs, [800,1100,1600,2300,3200,4400,6200,8600][effort]);
      }
      if(targetDigitsForBudget>=16){
        shortBudgetMs=Math.min(shortBudgetMs, [900,1400,2200,3400,5200,7600,11500,17000][effort]);
      }
      const hardDeadline=startTime + Math.ceil(shortBudgetMs*1.5);
      const deadline=startTime + shortBudgetMs;
      settings._shortformBudgetMs=shortBudgetMs;
      settings._shortformHardBudgetMs=Math.ceil(shortBudgetMs*1.5);
      const budgets=searchBudgetSequence(effort,target);
      const rows=[]; const seen=new Set();
      let maxDbSize=0, lastBudget=0, stoppedEarly=false;
      const limit=Math.max(1, Math.min(20, settings.limit||5));
      if(effort>=3 && !run?.stopped && performance.now()<deadline){
        const td=decimalDigitCountBig(target);
        const wideCfg=digitSearchConfig(effort,target);
        wideCfg.maxDigits=Math.max(wideCfg.maxDigits, Math.min(td, Math.ceil(td*0.75)));
        settings._shortformPhase='wide structured backup';
        settings._shortformMs=Math.round(performance.now()-startTime);
        if(onUpdate) onUpdate(applyIntegerSign(selectDigitShortforms(rows, limit), sign, target), {budget:wideCfg.maxDigits, maxDbSize, effort, elapsed:settings._shortformMs, phase:'wide structured backup'});
        await yieldToUI();
        structuredBackupSearch(rows,seen,target,wideCfg,timeSliceDeadline(Math.max(45, 70+effort*12), deadline));
        let selected=applyIntegerSign(selectDigitShortforms(rows, limit), sign, target);
        settings._shortformMs=Math.round(performance.now()-startTime);
        settings._shortformMaxDigits=wideCfg.maxDigits;
        settings._shortformEffort=effort;
        settings._shortformDbSize=maxDbSize;
        if(onUpdate) onUpdate(selected, {budget:wideCfg.maxDigits, maxDbSize, effort, elapsed:settings._shortformMs, phase:'wide'});
        await idle();
      }
      for(const budget of budgets){
        if(run?.stopped || shortAbort(deadline)) break;
        lastBudget=budget;
        const cfg=digitSearchConfig(effort,target);
        cfg.maxDigits=budget;
        tuneIntegerConfigForDigitBudget(cfg, target);
        // Spend small, predictable slices per budget so that Stop can return quickly and early gems are displayed.
        const remaining=Math.max(0, deadline-performance.now());
        if(remaining<=0) break;
        const budgetIndex=budgets.indexOf(budget)+1;
        const budgetsLeft=Math.max(1, budgets.length-budgetIndex+1);
        const perBudget=Math.max(120, remaining/budgetsLeft);
        const lowBudget = budget<=3;
        const dbSlice=Math.max(35, Math.min(lowBudget?120:220, remaining*0.30, perBudget*(lowBudget?0.18:0.36), [70,85,105,125,145,170,195,220][effort] + budget*7));
        settings._shortformPhase=`building DB for digit budget ${budget}`;
        settings._shortformMaxDigits=budget;
        settings._shortformMs=Math.round(performance.now()-startTime);
        if(onUpdate) onUpdate(applyIntegerSign(selectDigitShortforms(rows, limit), sign, target), {budget, maxDbSize, effort, elapsed:settings._shortformMs, phase:settings._shortformPhase});
        await yieldToUI();
        const db=buildDigitSearchDB(target, settings, cfg, timeSliceDeadline(dbSlice, deadline));
        maxDbSize=Math.max(maxDbSize, db.byValue.size);
        const phaseStart=performance.now();
        const phaseBudget=Math.max(45, Math.min(lowBudget?150:260, deadline-phaseStart, perBudget*(lowBudget?0.26:0.44), [80,95,115,140,165,195,225,260][effort] + budget*8));
        settings._shortformPhase='direct/reverse exact pass';
        if(onUpdate) onUpdate(applyIntegerSign(selectDigitShortforms(rows, limit), sign, target), {budget, maxDbSize, effort, elapsed:Math.round(performance.now()-startTime), phase:settings._shortformPhase});
        await yieldToUI();
        directAndReverseSearch(rows,seen,target,db,cfg,timeSliceDeadline(Math.min(lowBudget?45:85, phaseBudget*0.20), deadline));
        let selected=applyIntegerSign(selectDigitShortforms(rows, limit), sign, target);
        settings._shortformMs=Math.round(performance.now()-startTime);
        settings._shortformMaxDigits=budget;
        settings._shortformEffort=effort;
        settings._shortformDbSize=maxDbSize;
        settings._shortformStopped=!!run?.stopped;
        if(onUpdate) onUpdate(selected, {budget, maxDbSize, effort, elapsed:settings._shortformMs});
        if(run?.stopped) break;
        await yieldToUI();
        settings._shortformPhase='ratio exact pass';
        if(onUpdate) onUpdate(applyIntegerSign(selectDigitShortforms(rows, limit), sign, target), {budget, maxDbSize, effort, elapsed:Math.round(performance.now()-startTime), phase:settings._shortformPhase});
        ratioSearch(rows,seen,target,db,cfg,timeSliceDeadline(Math.min(lowBudget?55:110, phaseBudget*0.28), deadline));
        selected=applyIntegerSign(selectDigitShortforms(rows, limit), sign, target);
        settings._shortformMs=Math.round(performance.now()-startTime);
        if(onUpdate) onUpdate(selected, {budget, maxDbSize, effort, elapsed:settings._shortformMs});
        if(run?.stopped) break;
        if(earlyShortformStop(selected,target,limit,budget,effort,settings._shortformMs)){ stoppedEarly=true; break; }
        await yieldToUI();
        settings._shortformPhase='rational powers';
        if(onUpdate) onUpdate(applyIntegerSign(selectDigitShortforms(rows, limit), sign, target), {budget, maxDbSize, effort, elapsed:Math.round(performance.now()-startTime), phase:settings._shortformPhase});
        rationalPowerSearch(rows,seen,target,db,cfg,timeSliceDeadline(Math.min(lowBudget?45:90, phaseBudget*0.18), deadline));
        if(run?.stopped || shortAbort(deadline)) break;
        await yieldToUI();
        settings._shortformPhase='final reverse pass';
        if(onUpdate) onUpdate(applyIntegerSign(selectDigitShortforms(rows, limit), sign, target), {budget, maxDbSize, effort, elapsed:Math.round(performance.now()-startTime), phase:settings._shortformPhase});
        await yieldToUI();
        if(shortAbort(deadline) || performance.now()>hardDeadline) break;
        directAndReverseSearch(rows,seen,target,db,cfg,timeSliceDeadline(Math.min(lowBudget?28:55, phaseBudget*0.10), deadline));
        selected=applyIntegerSign(selectDigitShortforms(rows, limit), sign, target);
        settings._shortformMs=Math.round(performance.now()-startTime);
        if(onUpdate) onUpdate(selected, {budget, maxDbSize, effort, elapsed:settings._shortformMs});
        if(run?.stopped) break;
        if(earlyShortformStop(selected,target,limit,budget,effort,settings._shortformMs)){ stoppedEarly=true; break; }
        await idle();
      }
      let selectedRaw=selectDigitShortforms(rows, limit);
      const td=decimalDigitCountBig(target);
      const needBackup = selectedRaw.length===0 || (selectedRaw[0] && selectedRaw[0].digits>Math.max(2, Math.ceil(td*0.72)));
      if(needBackup && !run?.stopped && performance.now()<deadline){
        const cfg=digitSearchConfig(effort,target);
        cfg.maxDigits=Math.max(cfg.maxDigits, Math.min(td-1, Math.ceil(td*0.72)));
        await yieldToUI();
        structuredBackupSearch(rows,seen,target,cfg,Math.min(deadline, performance.now()+Math.max(45, 70+effort*14)));
        selectedRaw=selectDigitShortforms(rows, limit);
      }
      if(target<=100000000n && effort>=4 && !run?.stopped && performance.now()<deadline){
        await yieldToUI();
        const bestBefore=selectedRaw[0]?.digits ?? 999;
        const capMs = bestBefore<=3 ? Math.max(260, 300+effort*35) : Math.max(360, 440+effort*70);
        await smallIntegerExhaustiveSearchAsync(rows,seen,target,effort,Math.min(deadline, performance.now()+capMs), info=>{
          selectedRaw=selectDigitShortforms(rows, limit);
          settings._shortformMs=Math.round(performance.now()-startTime);
          settings._shortformPhase=info.label;
          settings._shortformDbSize=Math.max(maxDbSize, rows.length);
          if(onUpdate) onUpdate(applyIntegerSign(selectedRaw, sign, target), {budget:lastBudget, maxDbSize:settings._shortformDbSize, effort, elapsed:settings._shortformMs, phase:info.label, exhaustive:true, frac:info.frac});
        });
        selectedRaw=selectDigitShortforms(rows, limit);
      }
      let selected=applyIntegerSign(selectedRaw, sign, target);
      const qualityCut=Math.max(2, Math.ceil(td*0.75));
      if((!selected.length || (selected[0] && selected[0].digits>qualityCut)) && !run?.stopped && performance.now()<hardDeadline){
        const fbRemain=Math.max(0, hardDeadline-performance.now());
        const fbLimit=Math.min(5, limit);
        const noShortYet=!selected.length;
        const diverseMs=noShortYet ? 220 : Math.max(60, Math.min(360, fbRemain*0.38));
        const ratioMs=noShortYet ? 260 : Math.max(60, Math.min(360, fbRemain-diverseMs));
        const naturalFb=[...diverseShortformFallback(sign, target, fbLimit, diverseMs), ...powerRatioShortformFallback(sign, target, fbLimit, ratioMs)];
        const fb=naturalFb.length ? naturalFb : [];
        if(fb.length){
          const merged=[...selected, ...fb];
          const seenFinal=new Set();
          selected=merged.sort((a,b)=>(a.digits-b.digits)||((a.beauty||0)-(b.beauty||0))||a.candidate.length-b.candidate.length)
            .filter(r=>{ const k=candidateEquivalenceKey(r); if(seenFinal.has(k)) return false; seenFinal.add(k); return true; })
            .slice(0,limit);
        }
      }
      if(!selected.length){
        // v6.8: do not surface the worst A*10^B+C exact fallback.  Empty is
        // better than a mechanically restated decimal split.
        selected=[];
      }
      settings._shortformMs=Math.round(performance.now()-startTime);
      settings._shortformMaxDigits=lastBudget;
      settings._shortformEffort=effort;
      settings._shortformDbSize=maxDbSize;
      settings._shortformStopped=!!run?.stopped;
      settings._shortformEarly=stoppedEarly;
      return selected;
    }

    // v6.3 high-precision expression evaluator.  It is intentionally read-only and
    // sandbox-free: expressions are parsed into tokens and evaluated with
    // decimal.js when available, with BigInt side-carrying for exact integer
    // arithmetic such as 125!*7 or 3^257*6^2.
    const HP_PI = '3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196442881097566593344612847564823378678316527120190914564856692346034861045432664821339360726024914127372458700660631558817488152092096282925409171536436789259036001133053054882046652138414695194151160943305727036575959195309218611738193261179310511854807446237996274956735188575272489122793818301194912983367336244065664308602139494639522473719070217986094370277053921717629317675238467481846766940513200056812714526356082778577134275778960917363717872146844090122495343014654958537105079227968925892354201995611212902196086403441815981362977477130996051870721134999999837297804995105973173281609631859502445945534690830264252230825334468503526193118817101000313783875288658753320838142061717766914730359825349042875546873115956286388235378759375195778185778053217122680661300192787661119590921642019893809525720106548586327886593615338182796823030195203530185296899577362259941389124972177528347913151557485724';
    const HP_E = '2.71828182845904523536028747135266249775724709369995957496696762772407663035354759457138217852516642742746639193200305992181741359662904357290033429526059563073813232862794349076323382988075319525101901157383418793070215408914993488416750924476146066808226480016847741185374234544243710753907774499206955170276183860626133138458300075204493382656029760673711320070932870912744374704723069697720931014169283681902551510865746377211125238978442505695369677078544996996794686445490598793163688923009879312773617821542499922957635148220826989519366803318252886939849646510582093923982948879332036250944311730123819706841614039701983767932068328237646480429531180232878250981945581530175671736133206981125099618188159304169035159888851934580727386673858942287922849989208680582574927961048419844436346324496848756023362482704197862320900216099023530436994184914631409343173814364054625315209618369088870701676839642437814059271456354906130310720851038375051011574770417189861068739696552126715468895703503540212340784981933432106817012100562788023519303322474501585390473041995777709350366041699732972508868769664036';
    const HP_GAMMA_QUARTER = '3.62560990822190831193068515586767200299516768288006546743337799956991924353872912161836013672338430036147175139242071996589152409402255997742645889036145060641374489685419499920192677303799463089221241231832370799208439736990709390562092923234287027419144860395713683503686548799596836847647585148909040416634076303397180668059577342379085590807145783129763563688255879288111906351681585084947488150278867310731052487982516636612879316418441744382764575480091991477680192281509261199432299783783536345955434194743707480852281647492382207765758264184768229225624189134785920659156756563875507328166896282474154785932393787382451296744434518203051071750663878373996462064982483241536680954974218844410454251761639071919577279017756154543055019888415374653873552143860099081231395707515335427226551048545306108674886732775173541649799032139912715540614055414755623822673386217458027153660103915304533002159885720335369221232686386770780674096284705275930838450601160100678669518246421402801055529863927553021792388749586364245258749340568761019828200356362794363065724514138808203635564907486816223637742526510883';
    const HP_GAMMA_THIRD = '2.67893853470774763365569294097467764412868937795730110095042832759041761016774381954098288904118878941915904920007226333571908456950447225997771336770846976816728982305000321834255032224715694181755544995272878439477944130576582840161231914159646652603372758402058063551394324103201583941538270085524052103233879895506936389263868391670728169154230962733188118647749652229105564440907800963416463532740195630152839860021250692996870571006864544754497067121216435474129286262442530553451675106636988879570294949999982290487540576987349454035944864661812760112550108100713577485091828788892826908111303965083006183990301838902848679386952398156500343120973567253436209211040973608149288669032999939199763507245847443710428535667893467156890589821447841213999549940936167692556916329004659357582947195283689259634270652842602971481151466731234090052039992929338082203933022760958233103710997601572632652329111786322508664193551371299921064947621578824174236659573837886013296585503165598854530660096972909944455435470791516329812437272925770865223783180242927892577348653291794704487681544117500118829420957743205';
    const HP_SQRT_PI = '1.77245385090551602729816748334114518279754945612238712821380778985291128459103218137495065673854466541622682362428257066623615286572442260252509370960278706846203769865310512284992517302895082622893209537926796280017463901535147972051670019018523401858544697449491264031392177552590621640541933250090639840761373347747515343366798978936585183640879545116516173876005906739343179133280985484624818490205465485219561325156164746751504273876105610799612710721006037204448367236529661370809432349883166842421384570960912042042778577806869476657000521830568512541339663694465418151071669388332194292935706226886522442054214994804992075648639887483850593064021821402928581123306497894520362114907896228738940324597819851313487126651250629326004465638210967502681249693059542046156076195221739152507020779275809905433290066222306761446966124818874306997883520506146444385418530797357425717918563595974995995226384924220388910396640644729397284134504300214056423343303926175613417633632001703765416347632066927654181283576249032690450848532013419243598973087119379948293873011126256165881888478597787596376136321863425';

    function shouldTryHighPrecision(raw){
      const s=String(raw||'').trim(); if(!s) return false;
      if(integerInputBig(s)!==null) return false;
      if(parseDecimalComplex(s)) return false;
      return /[a-zA-ZπΓ!^()*/]|\d\s*[πa-zA-Z]/.test(s);
    }
    function hpAvailable(){ return typeof Decimal !== 'undefined'; }
    function hpD(x){ return new Decimal(x); }
    function hpZero(){ return new Decimal(0); }
    function hpOne(){ return new Decimal(1); }
    function hpIsZero(x){ return x.isZero && x.isZero(); }
    function hpC(re, im, bi=null){ return {re, im, bi}; }
    function hpReal(x, bi=null){ return hpC(x, hpZero(), bi); }
    function hpIsReal(z){ return z && z.im && z.im.isZero(); }
    function hpPromote(v){ return v && v.re ? v : hpReal(hpD(v)); }
    function hpAdd(a,b){ a=hpPromote(a); b=hpPromote(b); const bi=(a.bi!==null&&b.bi!==null&&hpIsReal(a)&&hpIsReal(b)) ? a.bi+b.bi : null; return hpC(a.re.plus(b.re), a.im.plus(b.im), bi); }
    function hpSub(a,b){ a=hpPromote(a); b=hpPromote(b); const bi=(a.bi!==null&&b.bi!==null&&hpIsReal(a)&&hpIsReal(b)) ? a.bi-b.bi : null; return hpC(a.re.minus(b.re), a.im.minus(b.im), bi); }
    function hpNeg(a){ a=hpPromote(a); return hpC(a.re.neg(), a.im.neg(), a.bi!==null ? -a.bi : null); }
    function hpMul(a,b){ a=hpPromote(a); b=hpPromote(b); const re=a.re.times(b.re).minus(a.im.times(b.im)); const im=a.re.times(b.im).plus(a.im.times(b.re)); const bi=(a.bi!==null&&b.bi!==null&&hpIsReal(a)&&hpIsReal(b)) ? a.bi*b.bi : null; return hpC(re,im,bi); }
    function hpDiv(a,b){ a=hpPromote(a); b=hpPromote(b); const den=b.re.times(b.re).plus(b.im.times(b.im)); if(den.isZero()) throw Error('division by zero'); return hpC(a.re.times(b.re).plus(a.im.times(b.im)).div(den), a.im.times(b.re).minus(a.re.times(b.im)).div(den), null); }
    function hpAbs(a){ a=hpPromote(a); if(hpIsReal(a)) return hpReal(a.re.abs(), a.bi!==null ? (a.bi<0n?-a.bi:a.bi) : null); return hpReal(a.re.times(a.re).plus(a.im.times(a.im)).sqrt()); }
    function hpSqrt(a){
      a=hpPromote(a);
      if(hpIsReal(a) && a.re.gte(0)) return hpReal(a.re.sqrt(), null);
      if(hpIsReal(a)) return hpC(hpZero(), a.re.neg().sqrt(), null);
      const r=a.re.times(a.re).plus(a.im.times(a.im)).sqrt();
      const re=r.plus(a.re).div(2).sqrt();
      const imSign=a.im.lt(0) ? -1 : 1;
      const im=r.minus(a.re).div(2).sqrt().times(imSign);
      return hpC(re, im, null);
    }
    function hpExp(a){ a=hpPromote(a); if(hpIsReal(a)) return hpReal(a.re.exp()); const er=a.re.exp(); return hpC(er.times(Decimal.cos(a.im)), er.times(Decimal.sin(a.im))); }
    function hpLog(a){ a=hpPromote(a); if(hpIsReal(a) && a.re.gt(0)) return hpReal(a.re.ln()); const r=a.re.times(a.re).plus(a.im.times(a.im)).sqrt().ln(); const theta=Decimal.atan2 ? Decimal.atan2(a.im, a.re) : hpD(Math.atan2(Number(a.im),Number(a.re))); return hpC(r, theta); }
    function hpPow(a,b){
      a=hpPromote(a); b=hpPromote(b);
      if(hpIsReal(b) && b.bi!==null && b.bi>=0n && b.bi<=10000n){
        let n=b.bi, r=hpReal(hpOne(),1n), x=a;
        while(n>0n){ if(n&1n) r=hpMul(r,x); x=hpMul(x,x); n>>=1n; }
        return r;
      }
      if(hpIsReal(a) && a.re.gte(0) && hpIsReal(b)) return hpReal(a.re.pow(b.re));
      return hpExp(hpMul(b,hpLog(a)));
    }
    function hpSin(a){ a=hpPromote(a); if(hpIsReal(a)) return hpReal(Decimal.sin(a.re)); return hpC(Decimal.sin(a.re).times(Decimal.cosh(a.im)), Decimal.cos(a.re).times(Decimal.sinh(a.im))); }
    function hpCos(a){ a=hpPromote(a); if(hpIsReal(a)) return hpReal(Decimal.cos(a.re)); return hpC(Decimal.cos(a.re).times(Decimal.cosh(a.im)), Decimal.sin(a.re).times(Decimal.sinh(a.im)).neg()); }
    function hpTan(a){ return hpDiv(hpSin(a), hpCos(a)); }
    function hpFactorialBig(n){ n=BigInt(n); if(n<0n) throw Error('factorial of negative integer'); if(n>5000n) throw Error('factorial too large'); let r=1n; for(let k=2n;k<=n;k++) r*=k; return r; }
    function hpFactorial(a){ a=hpPromote(a); if(!hpIsReal(a) || a.bi===null || a.bi<0n) throw Error('factorial requires a nonnegative integer'); const bi=hpFactorialBig(a.bi); return hpReal(hpD(bi.toString()), bi); }
    function hpRequireInt(a,name){ a=hpPromote(a); if(!hpIsReal(a) || a.bi===null) throw Error(name+' requires integer arguments'); return a.bi; }
    function hpBinom(a,b){ let n=hpRequireInt(a,'binom'), k=hpRequireInt(b,'binom'); if(n<0n||k<0n||k>n) throw Error('bad binom'); if(k>n-k) k=n-k; if(k>5000n) throw Error('binom loop too large'); let r=1n; for(let i=1n;i<=k;i++) r=(r*(n-k+i))/i; return hpReal(hpD(r.toString()), r); }
    function hpPerm(a,b){ let n=hpRequireInt(a,'A'), k=hpRequireInt(b,'A'); if(n<0n||k<0n||k>n||k>5000n) throw Error('bad permutation'); let r=1n; for(let i=0n;i<k;i++) r*=n-i; return hpReal(hpD(r.toString()), r); }
    function hpGcdFn(a,b){ let x=hpRequireInt(a,'gcd'), y=hpRequireInt(b,'gcd'); x=x<0n?-x:x; y=y<0n?-y:y; while(y){ const t=x%y; x=y; y=t; } return hpReal(hpD(x.toString()), x); }
    function hpLcmFn(a,b){ const x=hpRequireInt(a,'lcm'), y=hpRequireInt(b,'lcm'); if(x===0n || y===0n) return hpReal(hpZero(),0n); const g=hpGcdFn(hpReal(hpD(x.toString()),x),hpReal(hpD(y.toString()),y)).bi; const r=(x/g)*y; const z=r<0n?-r:r; return hpReal(hpD(z.toString()), z); }
    function hpFibFn(a){ const n=hpRequireInt(a,'fib'); if(n<0n||n>10000n) throw Error('bad fib'); let x=0n,y=1n; for(let i=0n;i<n;i++){ const t=x+y; x=y; y=t; } return hpReal(hpD(x.toString()), x); }
    function hpCatalan(a){ const n=hpRequireInt(a,'catalan'); if(n<0n||n>3000n) throw Error('bad catalan'); const b=hpBinom(hpReal(hpD((2n*n).toString()),2n*n), hpReal(hpD(n.toString()),n)).bi; const r=b/(n+1n); return hpReal(hpD(r.toString()), r); }
    function hpGamma(a){
      a=hpPromote(a);
      if(!hpIsReal(a)) throw Error('gamma currently supports real arguments');
      if(a.bi!==null && a.bi>=1n && a.bi<=5000n){ const bi=hpFactorialBig(a.bi-1n); return hpReal(hpD(bi.toString()), bi); }
      const x=a.re;
      if(x.eq(hpD('0.5'))) return hpReal(hpD(HP_SQRT_PI));
      if(x.eq(hpD('0.25'))) return hpReal(hpD(HP_GAMMA_QUARTER));
      if(x.eq(hpD('0.3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333')) || x.eq(hpOne().div(3))) return hpReal(hpD(HP_GAMMA_THIRD));
      throw Error('gamma high-precision path supports positive integers, 1/2, 1/3 and 1/4 in this browser build');
    }
    function hpTokenize(raw){
      let s=String(raw||'').replace(/[−–—]/g,'-').replace(/[×·]/g,'*').replace(/[÷]/g,'/').replace(/π/g,'pi').replace(/Γ/g,'gamma').replace(/\s+/g,'');
      const toks=[]; let i=0;
      while(i<s.length){
        const ch=s[i];
        if(/[0-9.]/.test(ch)){
          const m=s.slice(i).match(/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?/); if(!m) throw Error('bad number');
          toks.push({type:'num', value:m[0]}); i+=m[0].length; continue;
        }
        if(/[A-Za-z_]/.test(ch)){
          const m=s.slice(i).match(/^[A-Za-z_][A-Za-z_0-9]*/); toks.push({type:'id', value:m[0].toLowerCase()}); i+=m[0].length; continue;
        }
        if('+-*/^()!,'.includes(ch)){ toks.push({type: ch==='('||ch===')'||ch===',' ? ch : 'op', value:ch}); i++; continue; }
        throw Error('unsupported character '+ch);
      }
      const out=[];
      function leftValue(t){ return t && (t.type==='num' || t.type===')' || (t.type==='id' && !hpFunctionNames.has(t.value)) || (t.type==='op' && t.value==='!')); }
      function rightValue(t){ return t && (t.type==='num' || t.type==='(' || t.type==='id'); }
      for(let j=0;j<toks.length;j++){
        const prev=out[out.length-1], cur=toks[j];
        if(leftValue(prev) && rightValue(cur)) out.push({type:'op', value:'*', implicit:true});
        out.push(cur);
      }
      return out;
    }
    const hpFunctionNames=new Set(['sqrt','sin','cos','tan','log','ln','exp','abs','gamma','factorial','fact','binom','choose','ncr','c','perm','npr','a','gcd','lcm','fib','fibonacci','catalan']);
    function hpToRpn(tokens){
      const out=[], ops=[]; const prec={'+':1,'-':1,'*':2,'/':2,'^':4,'u+':5,'u-':5,'!':6}; const rightAssoc={'^':true,'u+':true,'u-':true}; let prev=null;
      for(const tok of tokens){
        if(tok.type==='num'){ out.push(tok); prev=tok; continue; }
        if(tok.type==='id'){
          if(hpFunctionNames.has(tok.value)) ops.push({type:'fn', value:tok.value}); else out.push(tok);
          prev=tok; continue;
        }
        if(tok.type==='('){ ops.push(tok); prev=tok; continue; }
        if(tok.type===','){ while(ops.length && ops[ops.length-1].type!=='(') out.push(ops.pop()); prev=tok; continue; }
        if(tok.type===')'){
          while(ops.length && ops[ops.length-1].type!=='(') out.push(ops.pop());
          if(!ops.length) throw Error('mismatched parentheses'); ops.pop();
          if(ops.length && ops[ops.length-1].type==='fn') out.push(ops.pop());
          prev=tok; continue;
        }
        if(tok.type==='op'){
          let op=tok.value;
          if((op==='+' || op==='-') && (!prev || (prev.type==='op' && prev.value!=='!') || prev.type==='(' || prev.type===',')) op='u'+op;
          while(ops.length){
            const top=ops[ops.length-1]; if(top.type==='(' || top.type==='fn') { if(top.type==='fn' && op==='!') out.push(ops.pop()); else break; }
            else if(top.type==='op' && ((rightAssoc[op] ? prec[op] < prec[top.value] : prec[op] <= prec[top.value]))) out.push(ops.pop()); else break;
          }
          ops.push({type:'op', value:op}); prev={type:'op', value:op}; continue;
        }
      }
      while(ops.length){ const op=ops.pop(); if(op.type==='(') throw Error('mismatched parentheses'); out.push(op); }
      return out;
    }
    function hpEvalRpn(rpn){
      const stack=[];
      function pop(){ if(!stack.length) throw Error('missing operand'); return stack.pop(); }
      for(const tok of rpn){
        if(tok.type==='num'){
          const bi=/^[+\-]?\d+$/.test(tok.value) ? BigInt(tok.value) : null;
          stack.push(hpReal(hpD(tok.value), bi)); continue;
        }
        if(tok.type==='id'){
          if(tok.value==='pi') stack.push(hpReal(hpD(HP_PI)));
          else if(tok.value==='e') stack.push(hpReal(hpD(HP_E)));
          else if(tok.value==='phi') stack.push(hpReal(hpOne().plus(hpD(5).sqrt()).div(2)));
          else if(tok.value==='i' || tok.value==='j') stack.push(hpC(hpZero(),hpOne(),null));
          else throw Error('unknown identifier '+tok.value);
          continue;
        }
        if(tok.type==='op'){
          if(tok.value==='u+') { stack.push(pop()); continue; }
          if(tok.value==='u-') { stack.push(hpNeg(pop())); continue; }
          if(tok.value==='!') { stack.push(hpFactorial(pop())); continue; }
          const b=pop(), a=pop();
          if(tok.value==='+') stack.push(hpAdd(a,b));
          else if(tok.value==='-') stack.push(hpSub(a,b));
          else if(tok.value==='*') stack.push(hpMul(a,b));
          else if(tok.value==='/') stack.push(hpDiv(a,b));
          else if(tok.value==='^') stack.push(hpPow(a,b));
          else throw Error('unknown operator '+tok.value);
          continue;
        }
        if(tok.type==='fn'){
          if(['binom','choose','ncr','c'].includes(tok.value)){ const b=pop(), a=pop(); stack.push(hpBinom(a,b)); }
          else if(['perm','npr','a'].includes(tok.value)){ const b=pop(), a=pop(); stack.push(hpPerm(a,b)); }
          else if(tok.value==='gcd'){ const b=pop(), a=pop(); stack.push(hpGcdFn(a,b)); }
          else if(tok.value==='lcm'){ const b=pop(), a=pop(); stack.push(hpLcmFn(a,b)); }
          else {
            const a=pop();
            if(tok.value==='sqrt') stack.push(hpSqrt(a));
            else if(tok.value==='sin') stack.push(hpSin(a));
            else if(tok.value==='cos') stack.push(hpCos(a));
            else if(tok.value==='tan') stack.push(hpTan(a));
            else if(tok.value==='log' || tok.value==='ln') stack.push(hpLog(a));
            else if(tok.value==='exp') stack.push(hpExp(a));
            else if(tok.value==='abs') stack.push(hpAbs(a));
            else if(tok.value==='gamma') stack.push(hpGamma(a));
            else if(tok.value==='factorial' || tok.value==='fact') stack.push(hpFactorial(a));
            else if(tok.value==='fib' || tok.value==='fibonacci') stack.push(hpFibFn(a));
            else if(tok.value==='catalan') stack.push(hpCatalan(a));
            else throw Error('unknown function '+tok.value);
          }
        }
      }
      if(stack.length!==1) throw Error('bad expression');
      return stack[0];
    }
    function hpEvalExpression(raw, digits=1100, timeLimitMs=5000){
      if(!shouldTryHighPrecision(raw)) return null;
      if(!hpAvailable()) return {error:'decimal.js is not loaded; high-precision expression evaluation is unavailable.'};
      const start=performance.now();
      const oldPrec=Decimal.precision;
      Decimal.set({precision: Math.max(80, Math.min(1200, digits+25)), toExpNeg: -1000000, toExpPos: 1000000});
      try{
        const rpn=hpToRpn(hpTokenize(raw));
        if(performance.now()-start>timeLimitMs) throw Error('time limit exceeded while parsing');
        const z=hpEvalRpn(rpn);
        if(performance.now()-start>timeLimitMs) throw Error('time limit exceeded while evaluating');
        return {z, ms:Math.round(performance.now()-start)};
      }catch(e){ return {error:e && e.message ? e.message : String(e)}; }
      finally{ Decimal.set({precision: oldPrec || 20}); }
    }
    function hpFormatDecimal(x, sig=100){
      x=hpD(x); if(!x.isFinite()) return x.toString();
      if(x.isZero()) return '0';
      return x.toSignificantDigits(sig).toFixed();
    }
    function hpTrimZeros(s){ return String(s).replace(/(\.\d*?)0+$/,'$1').replace(/\.$/,''); }
    function hpExpandableHtml(label, full, first=100){
      full=String(full);
      const preview=full.length>first ? full.slice(0,first)+'…' : full;
      if(full.length<=first) return `<span>${escapeHtml(preview)}</span>`;
      return `<div><span>${escapeHtml(preview)}</span><details><summary>show 1000 digits</summary><code>${escapeHtml(full)}</code></details></div>`;
    }
    function hpFormatResultHtml(z){
      if(z.bi!==null && hpIsReal(z)){
        const s=z.bi.toString();
        if(s.length<=1000) return `<code>${escapeHtml(s)}</code>`;
        return hpExpandableHtml('integer', s.slice(0,1000), 100) + `<div class="muted">integer has ${s.length} digits; first 1000 shown</div>`;
      }
      if(hpIsReal(z)){
        const full=hpTrimZeros(hpFormatDecimal(z.re,1000));
        return hpExpandableHtml('decimal', full, 100);
      }
      const re=hpTrimZeros(hpFormatDecimal(z.re,1000));
      const imAbs=hpTrimZeros(hpFormatDecimal(z.im.abs(),1000));
      const sign=z.im.lt(0)?' − ':' + ';
      const full=`${re}${sign}${imAbs}i`;
      const preview=`${re.slice(0,80)}${re.length>80?'…':''}${sign}${imAbs.slice(0,80)}${imAbs.length>80?'…':''}i`;
      return `<div><span>${escapeHtml(preview)}</span><details><summary>show 1000 digits</summary><code>${escapeHtml(full)}</code></details></div>`;
    }
    function hpPlainPreview(z, sig=100){
      if(!z) return '';
      if(z.bi!==null && hpIsReal(z)) return z.bi.toString();
      if(hpIsReal(z)) return hpTrimZeros(hpFormatDecimal(z.re,sig));
      const re=hpTrimZeros(hpFormatDecimal(z.re,Math.max(40,Math.floor(sig/2))));
      const im=hpTrimZeros(hpFormatDecimal(z.im.abs(),Math.max(40,Math.floor(sig/2))));
      return `${re}${z.im.lt(0)?' − ':' + '}${im}i`;
    }
    function highPrecisionEval(settings){
      const ev=hpEvalExpression(settings.raw, 1100, 5000);
      if(!ev) return null;
      settings._hpEval=ev;
      if(!ev.error && hpIsReal(ev.z)){
        const approx=Number(ev.z.bi!==null ? ev.z.bi.toString() : ev.z.re.toString());
        const hpDec = ev.z.bi!==null ? ev.z.bi.toString() : hpTrimZeros(hpFormatDecimal(ev.z.re,120));
        if(Number.isFinite(approx)) { settings.target=approx; settings.normalizedRaw=hpDec; settings.parsedComplex=parseDecimalComplex(hpDec); settings.complexTarget=false; }
      }
      if(!ev.error && ev.z.bi!==null && hpIsReal(ev.z)) settings._hpIntegerString=ev.z.bi.toString();
      return ev;
    }
    function renderHighPrecision(ev, settings){
      if(!hpPanel || !hpContent) return;
      if(!ev){ hpPanel.hidden=true; hpContent.innerHTML=''; return; }
      hpPanel.hidden=false;
      if(ev.error){
        hpPanel.classList.add('hp-error');
        hpContent.innerHTML=`<div class="hp-error-line">Not evaluated: ${escapeHtml(ev.error)}</div>`;
        return;
      }
      hpPanel.classList.remove('hp-error');
      const z=ev.z;
      const type = !hpIsReal(z) ? 'complex high-precision value' : (z.bi!==null ? 'exact integer value' : 'high-precision decimal value');
      const meta = z.bi!==null && hpIsReal(z) ? `${z.bi.toString().length} digit(s), ${ev.ms} ms` : `first 100 digits shown, expandable to 1000 · ${ev.ms} ms`;
      const html=hpFormatResultHtml(z);
      const plain=hpPlainPreview(z,1000);
      hpContent.innerHTML=`<div class="hp-meta"><span>${escapeHtml(type)}</span><span>${escapeHtml(meta)}</span></div><div class="hp-value">${html}<div class="copy-row">${copyButtonHtml(plain,'high precision value')}</div></div>`;
    }

    const BASE_DIGITS='0123456789abcdefghijklmnopqrstuvwxyz';
    function baseDigitChar(n){ return BASE_DIGITS[Number(n)] || '?'; }
    function bigIntToBaseString(n, base){
      base=BigInt(base); if(n===0n) return '0'; const neg=n<0n; if(neg) n=-n;
      let out=''; while(n>0n){ out=baseDigitChar(n%base)+out; n/=base; }
      return neg?'-'+out:out;
    }
    function rationalToBaseString(q, base, fracLimit=96){
      if(!q) return '';
      let num=q.num, den=q.den; const neg=num<0n; if(neg) num=-num;
      const b=BigInt(base); const intPart=num/den; let rem=num%den;
      let out=(neg?'-':'')+bigIntToBaseString(intPart,b);
      if(rem===0n) return out;
      out+='.'; const seen=new Map(); let frac='';
      for(let i=0;i<fracLimit && rem!==0n;i++){
        if(seen.has(rem)){ const j=seen.get(rem); frac=frac.slice(0,j)+'('+frac.slice(j)+')'; rem=0n; break; }
        seen.set(rem,i); rem*=b; const d=rem/den; rem%=den; frac+=baseDigitChar(d);
      }
      out+=frac; if(rem!==0n) out+='…'; return out;
    }
    function decimalToBaseString(x, base, fracLimit=160){
      if(!hpAvailable()) return '';
      const old={precision:Decimal.precision, rounding:Decimal.rounding, toExpNeg:Decimal.toExpNeg, toExpPos:Decimal.toExpPos};
      try{
        Decimal.set({precision:Math.max(80, fracLimit+80), toExpNeg:-1000000, toExpPos:1000000});
        x=hpD(x); if(!x.isFinite()) return x.toString();
        const neg=x.lt(0); if(neg) x=x.neg();
        const b=hpD(base); let int=x.floor(); let frac=x.minus(int);
        let intStr;
        try{ intStr=bigIntToBaseString(BigInt(int.toFixed(0)), BigInt(base)); }
        catch(e){ intStr=int.toString(); }
        let out=(neg?'-':'')+intStr;
        if(frac.isZero()) return out;
        out+='.'; let fs='';
        for(let i=0;i<fracLimit && !frac.isZero();i++){
          frac=frac.times(b); const d=frac.floor(); fs+=baseDigitChar(BigInt(d.toFixed(0))); frac=frac.minus(d);
        }
        return out+fs+(frac.isZero()?'':'…');
      }finally{ Decimal.set(old); }
    }
    function statsForRepresentation(rep, base){
      const counts=Array(base).fill(0); const alphabet=BASE_DIGITS.slice(0,base);
      for(const ch of String(rep).toLowerCase()){
        const idx=alphabet.indexOf(ch); if(idx>=0) counts[idx]++;
      }
      return counts;
    }
    function countsBarHtml(counts, base){
      const max=Math.max(1,...counts);
      return `<div class="digit-stats">`+counts.map((c,i)=>`<div class="digit-count"><span>${escapeHtml(baseDigitChar(BigInt(i)))}</span><b style="--w:${Math.max(3,Math.round(c/max*100))}%"></b><em>${c}</em></div>`).join('')+`</div>`;
    }
    function continuedFractionFromRational(q, limit=80){
      if(!q) return [];
      let n=q.num, d=q.den; const out=[];
      for(let i=0;i<limit && d!==0n;i++){
        const a=floorDiv(n,d); out.push(a.toString()); const r=n-a*d; if(r===0n) break; n=d; d=r;
      }
      return out;
    }
    function continuedFractionFromDecimal(x, limit=80){
      if(!hpAvailable()) return [];
      x=hpD(x); const out=[];
      for(let i=0;i<limit;i++){
        const a=x.floor(); out.push(a.toFixed(0)); const r=x.minus(a); if(r.abs().lt('1e-980')) break; x=hpOne().div(r);
      }
      return out;
    }
    function compactCfHtml(cf){
      if(!cf || !cf.length) return '<span class="muted">not available</span>';
      const preview=cf.slice(0,28).join(', ')+(cf.length>28?', …':'');
      const full=cf.join(', ');
      const payload='['+full+']';
      return `<code>[${escapeHtml(preview)}]</code>${copyButtonHtml(payload,'continued fraction')}`+(cf.length>28?`<details><summary>show ${cf.length} terms</summary><code>[${escapeHtml(full)}]</code>${copyButtonHtml(payload,'continued fraction full')}</details>`:'');
    }
    function numberToolsShouldAppear(settings){
      const raw=String(settings?.raw || '').trim();
      if(!raw) return false;
      if(integerInputBig(raw)!==null) return true;
      const direct=parseDecimalComplex(raw);
      if(direct) return false;
      return !!(settings?._hpEval && !settings._hpEval.error && hpIsReal(settings._hpEval.z));
    }
    function currentNumberDescriptor(settings){
      if(!numberToolsShouldAppear(settings)) return null;
      const intRaw=integerInputBig(settings && settings.raw);
      if(intRaw!==null) return {kind:'integer', bi:intRaw, label:'integer input'};
      if(settings._hpEval && !settings._hpEval.error && hpIsReal(settings._hpEval.z)){
        const z=settings._hpEval.z;
        if(z.bi!==null) return {kind:'integer', bi:z.bi, label:'computed integer'};
        return {kind:'decimal', dec:z.re, label:'computed decimal'};
      }
      return null;
    }
    function renderNumberTools(settings){
      if(!numberToolsContent) return;
      const desc=currentNumberDescriptor(settings);
      if(!desc){ numberToolsContent.innerHTML='<p class="muted">This panel is shown only for exact integer input or real computable expressions; plain decimal inputs are intentionally hidden.</p>'; return; }
      const bases=[2,3,5,10,16];
      const reps={};
      for(const b of bases){
        if(desc.kind==='integer') reps[b]=bigIntToBaseString(desc.bi,b);
        else if(desc.kind==='rational') reps[b]=rationalToBaseString(desc.q,b,120);
        else reps[b]=decimalToBaseString(desc.dec,b,160);
      }
      let cf=[];
      if(desc.kind==='rational') cf=continuedFractionFromRational(desc.q,100);
      else if(desc.kind==='integer') cf=[desc.bi.toString()];
      else cf=continuedFractionFromDecimal(desc.dec,100);
      const baseRows=bases.map(b=>`<tr><td>${b}</td><td><code>${escapeHtml(reps[b])}</code>${copyButtonHtml(reps[b], 'base '+b+' representation')}</td></tr>`).join('');
      const statCards=bases.map(b=>`<div class="stat-card"><h4>base ${b}</h4>${countsBarHtml(statsForRepresentation(reps[b],b),b)}</div>`).join('');
      numberToolsContent.innerHTML=`
        <div class="tool-block"><h3>High-precision continued fraction</h3><p class="muted">Computed after opening this panel; terms are capped for responsiveness.</p>${compactCfHtml(cf)}</div>
        <div class="tool-block"><h3>Common base expansions</h3><div class="base-table-wrap"><table class="data mini"><thead><tr><th>base</th><th>representation</th></tr></thead><tbody>${baseRows}</tbody></table></div></div>
        <div class="tool-block"><h3>Digit statistics</h3><div class="stats-grid">${statCards}</div></div>`;
    }
    function prepareNumberTools(settings){
      if(!numberTools || !numberToolsContent) return;
      if(!numberToolsShouldAppear(settings)){
        numberTools.hidden=true; numberTools.open=false; numberToolsContent.innerHTML=''; window.__lastRIESSettings=null; return;
      }
      numberTools.hidden=false; numberTools.open=false;
      numberToolsContent.innerHTML='<p class="muted">Open this panel to compute continued fractions, base expansions, and digit statistics for integer input or a computed expression value.</p>';
      window.__lastRIESSettings=settings;
    }
    function copyButtonHtml(text, label='copy'){
      return `<button class="copy-btn" type="button" data-copy="${escapeHtml(text)}" aria-label="Copy ${escapeHtml(label)}">copy</button>`;
    }
    function stripHtmlText(html){
      return String(html || '').replace(/<[^>]*>/g,' ').replace(/\s+/g,' ').trim();
    }
    let tesseractAnimationToken=0;
    let activeTesseractRaf=0;
    function tesseractVertices(){
      const v=[];
      for(const x of [-1,1]) for(const y of [-1,1]) for(const z of [-1,1]) for(const w of [-1,1]) v.push([x,y,z,w]);
      return v;
    }
    function tesseractEdges(verts){
      const edges=[];
      for(let i=0;i<verts.length;i++) for(let j=i+1;j<verts.length;j++){
        let diff=0; for(let k=0;k<4;k++) if(verts[i][k]!==verts[j][k]) diff++;
        if(diff===1) edges.push([i,j]);
      }
      return edges;
    }
    const TESS_VERTS=tesseractVertices();
    const TESS_EDGES=tesseractEdges(TESS_VERTS);
    function rotatePlane4(p,i,j,a){
      const c=Math.cos(a), s=Math.sin(a), xi=p[i], xj=p[j];
      p[i]=c*xi-s*xj; p[j]=s*xi+c*xj;
    }
    function so4Direction(t){
      // Six SO(4) coordinate planes.  The vector is renormalized so the
      // instantaneous angular speed stays essentially constant while the
      // direction drifts slowly and smoothly.
      const raw=[
        Math.sin(.17*t+0.4)+.62*Math.cos(.071*t+1.7),
        Math.cos(.13*t+1.2)+.38*Math.sin(.053*t+4.1),
        Math.sin(.11*t+2.4)+.48*Math.cos(.061*t+0.6),
        Math.cos(.097*t+3.2)+.42*Math.sin(.047*t+2.9),
        Math.sin(.083*t+5.1)+.36*Math.cos(.059*t+5.4),
        Math.cos(.073*t+0.8)+.44*Math.sin(.041*t+1.1)
      ];
      const norm=Math.hypot(...raw)||1;
      return raw.map(x=>x/norm);
    }
    function projectTesseractPoint(p){
      // 4D -> 3D -> 2D perspective, then blend depth into both axes to avoid
      // flattened-looking projections.
      const dw=3.15-p[3]*0.34;
      const x3=p[0]/dw, y3=p[1]/dw, z3=p[2]/dw;
      const dz=2.55-z3*0.48;
      return [x3/dz + .20*z3/dz, y3/dz - .12*p[3]/dw];
    }
    function startTesseractAnimation(stage){
      if(!stage) return;
      const token=++tesseractAnimationToken;
      stage.dataset.tesseractToken=String(token);
      if(activeTesseractRaf) cancelAnimationFrame(activeTesseractRaf);
      stage.innerHTML='<canvas class="tesseract-canvas" width="144" height="144" aria-hidden="true"></canvas>';
      const canvas=stage.querySelector('canvas');
      const ctx=canvas.getContext('2d');
      const dpr=Math.min(2, window.devicePixelRatio||1);
      const cssSize=112;
      canvas.width=Math.round(cssSize*dpr);
      canvas.height=Math.round(cssSize*dpr);
      canvas.style.width=cssSize+'px'; canvas.style.height=cssSize+'px';
      ctx.setTransform(dpr,0,0,dpr,0,0);
      let last=performance.now();
      const planes=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]];
      const angles=[0,0,0,0,0,0];
      const omega=.74;
      function frame(now){
        if(stage.dataset.tesseractToken!==String(token) || !document.body.contains(stage)) return;
        const dt=Math.min(.050, Math.max(.001,(now-last)/1000)); last=now;
        const dir=so4Direction(now/1000);
        for(let i=0;i<6;i++) angles[i]+=dir[i]*omega*dt;
        const pts=TESS_VERTS.map(v=>{
          const p=v.slice();
          for(let i=0;i<6;i++) rotatePlane4(p,planes[i][0],planes[i][1],angles[i]);
          return projectTesseractPoint(p);
        });
        let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
        for(const [x,y] of pts){ minX=Math.min(minX,x); maxX=Math.max(maxX,x); minY=Math.min(minY,y); maxY=Math.max(maxY,y); }
        const span=Math.max(maxX-minX, maxY-minY, .001);
        const scale=72/span;
        const cx=(minX+maxX)/2, cy=(minY+maxY)/2;
        const mapped=pts.map(([x,y])=>[cssSize/2+(x-cx)*scale, cssSize/2+(y-cy)*scale]);
        ctx.clearRect(0,0,cssSize,cssSize);
        ctx.lineWidth=1.55;
        ctx.lineCap='round';
        ctx.shadowBlur=9;
        for(let i=0;i<TESS_EDGES.length;i++){
          const [a,b]=TESS_EDGES[i];
          const hue=(now*.028+i*11)%360;
          ctx.strokeStyle=`hsla(${hue}, 78%, 56%, .74)`;
          ctx.shadowColor=`hsla(${(hue+28)%360}, 92%, 62%, .22)`;
          ctx.beginPath(); ctx.moveTo(mapped[a][0],mapped[a][1]); ctx.lineTo(mapped[b][0],mapped[b][1]); ctx.stroke();
        }
        activeTesseractRaf=requestAnimationFrame(frame);
      }
      activeTesseractRaf=requestAnimationFrame(frame);
    }
    function setSearchStatus(text, progress=0.08, phase='search'){
      if(!statusEl) return;
      let pct=Math.max(2, Math.min(100, progress*100));
      const prev=Number(statusEl.dataset.progress || 0);
      if(statusEl.classList.contains('searching')) pct=Math.max(prev, pct);
      statusEl.dataset.progress=String(pct);
      statusEl.className='notice status-line searching rich-search';
      statusEl.style.setProperty('--progress', pct.toFixed(2)+'%');
      if(!statusEl.querySelector('.search-status-grid')){
        statusEl.innerHTML=`<div class="search-status-grid"><div class="search-status-main"><strong></strong><span></span><div class="search-progress-wrap"><div class="search-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(pct)}"><i></i></div><b class="search-progress-label"></b></div></div><div class="tesseract-widget"><div class="geometry-stage" aria-hidden="true"></div></div></div>`;
        startTesseractAnimation(statusEl.querySelector('.geometry-stage'));
      }
      const strong=statusEl.querySelector('.search-status-main strong');
      const span=statusEl.querySelector('.search-status-main span');
      const bar=statusEl.querySelector('.search-progress');
      const label=statusEl.querySelector('.search-progress-label');
      if(strong) strong.textContent=phase;
      if(span) span.textContent=text;
      if(bar) bar.setAttribute('aria-valuenow', String(Math.round(pct)));
      if(label) label.textContent=Math.round(pct)+'%';
    }
    document.addEventListener('click', ev=>{
      const btn=ev.target && ev.target.closest ? ev.target.closest('[data-copy]') : null;
      if(!btn) return;
      const text=btn.getAttribute('data-copy') || '';
      const done=()=>{ const old=btn.textContent; btn.textContent='copied'; btn.classList.add('copied'); setTimeout(()=>{ btn.textContent=old || 'copy'; btn.classList.remove('copied'); }, 900); };
      const fallbackCopy=()=>{ const ta=document.createElement('textarea'); ta.value=text; ta.setAttribute('readonly',''); ta.style.position='fixed'; ta.style.opacity='0'; document.body.appendChild(ta); ta.select(); try{ document.execCommand('copy'); done(); }catch(e){} document.body.removeChild(ta); };
      if(navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done).catch(fallbackCopy);
      else fallbackCopy();
    });

    function updatePreview(settings){ return; }
    function syncParameterModuleVisibility(){
      document.querySelectorAll('[data-module-toggle]').forEach(cb=>{
        const id=cb.getAttribute('data-module-toggle');
        const block=document.querySelector(`[data-module-block="${id}"]`);
        const body=document.querySelector(`[data-module-body="${id}"]`);
        const on=!!cb.checked;
        if(block) block.classList.toggle('is-disabled', !on);
        if(body) body.hidden=!on;
      });
    }
    function syncLevelDefaultBudgets(force=false){
      const level=Number(document.getElementById('level')?.value || DEFAULT_RIES_LEVEL);
      const v=String(riesLevelDefaultModuleBudgetMs(level));
      for(const id of ['riesBudgetMs','logBudgetMs','mobiusBudgetMs','lfuncBudgetMs']){
        const el=document.getElementById(id);
        if(!el) continue;
        if(force || el.dataset.userEdited!=='1') el.value=v;
      }
    }
    function mathCopyFromCandidate(candidate){
      const c=String(candidate||'').trim();
      if(!c) return '';
      if(/factorization|handoff|status|remaining composite/i.test(c)) return '';
      const m=c.match(/^[^:]{1,40}:\s*(.+)$/);
      const expr=(m?m[1]:c).trim();
      if(!/[0-9πφeExx\^+\-*\/√=()!·]|binom|log|sin|cos|tan|floor|ceil|round/.test(expr)) return '';
      return expr;
    }
    function copyHtmlMaybe(text,label){
      if(text===undefined || text===null) return '';
      const t=String(text);
      return t ? copyButtonHtml(t,label) : '';
    }
    function renderRows(rows, options={}){
      rows = Array.isArray(rows) ? rows : [];
      lastRenderedRows = rows.slice();
      if(options.final){
        currentResultAllRows = (options.allRows || rows).slice();
        currentResultDiscoveryRows = (options.discoveryRows || (!currentResultSorted ? rows : currentResultDiscoveryRows) || rows).slice();
        currentResultSettings = options.settings || currentResultSettings;
        currentResultSorted = !!options.sorted;
        updateResultTools(true);
      }else if(options.clearTools){
        clearResultTools();
      }
      const head=document.querySelector('.data thead tr');
      const hasRichMath=rows.some(r=>r && (r.latex || r.qLatex));
      const showError=rows.length===0 || rows.some(r=>!(r && r.hideError));
      const cols=2+(showError?1:0);
      if(head){
        const errHead=showError ? '<th>error</th>' : '';
        head.innerHTML = `<th>candidate</th><th>value / root</th>${errHead}`;
      }
      resultBody.innerHTML = rows.map(r=>{
        const candidateText=String(r.candidate || '');
        const candidateCopy = r.noCandidateCopy ? '' : (r.copyCandidate ?? mathCopyFromCandidate(candidateText));
        const displayLatex = r.latex ? sanitizeLatexForDisplay(r.latex) : '';
        const displayQLatex = r.qLatex ? sanitizeLatexForDisplay(r.qLatex) : '';
        const latexCopy = r.copyLatex !== undefined ? sanitizeLatexForDisplay(r.copyLatex) : displayLatex;
        const valuePlain = r.copyValue !== undefined ? String(r.copyValue || '') : (r.valueHtml ? stripHtmlText(r.valueHtml) : String(r.value ?? ''));
        const mainValue = r.valueHtml ? r.valueHtml : escapeHtml(String(r.value ?? ''));
        const meta=[];
        if(displayLatex) meta.push(`<div class="result-meta-line"><span class="result-meta-label">formula</span><span class="latex-render">\\(${escapeHtml(displayLatex)}\\)</span>${copyHtmlMaybe(latexCopy,'formula')}</div>`);
        if(r.modForm) meta.push(`<div class="result-meta-line"><span class="result-meta-label">form</span><code>${escapeHtml(r.modForm.code || `${r.modForm.level}.${r.modForm.weight}.${r.modForm.index}`)}</code> <span class="muted">Level ${escapeHtml(r.modForm.level)} · weight ${escapeHtml(r.modForm.weight)} · #${escapeHtml(r.modForm.index)}</span></div>`);
        if(displayQLatex) meta.push(`<div class="result-meta-line"><span class="result-meta-label">q-expansion</span><span class="latex-render q-expansion-render">\\(${escapeHtml(displayQLatex)}\\)</span>${copyHtmlMaybe(displayQLatex,'q-expansion')}</div>`);
        const metaBlock = meta.length ? `<div class="result-meta-block">${meta.join('')}</div>` : '';
        const valueCell = `<div class="result-value-stack"><div class="result-value-main">${mainValue}${copyHtmlMaybe(valuePlain,'value')}</div>${metaBlock}</div>`;
        const errText = r.errText || fmtErr(r.err);
        const errCell = showError ? `<td>${r.hideError ? '<span class="muted">—</span>' : escapeHtml(errText)}</td>` : '';
        return `<tr><td><code>${escapeHtml(candidateText)}</code>${copyHtmlMaybe(candidateCopy,'candidate')}</td><td>${valueCell}</td>${errCell}</tr>`;
      }).join('') || `<tr><td colspan="${cols}">No RIES/table results under the current settings. High-precision values, if any, are shown above.</td></tr>`;
      if(hasRichMath){
        const typeset=()=>{ if(window.MathJax && MathJax.typesetPromise){ MathJax.typesetPromise([resultBody]).catch(()=>{}); return true; } return false; };
        if(!typeset()) setTimeout(typeset, 80);
      }
    }

    const solveRunCache = new Map();
    const integerGlobalCache = new Map();
    function integerGlobalCacheKey(settings){
      const n=resolvedIntegerBig(settings);
      if(n===null) return null;
      return JSON.stringify({n:n.toString(), limit:Math.max(1,Math.min(20,Number(settings?.moduleLimits?.integer || settings.limit)||5)), modules:settings.modules||{}, integerOptions:settings.integerOptions||{}, moduleLimits:settings.moduleLimits||{}, stageBudgets:settings.stageBudgets||{}, external:!!settings.allowExternalFactorization});
    }
    function getIntegerGlobalCache(settings){
      const key=integerGlobalCacheKey(settings) || 'none';
      let c=integerGlobalCache.get(key);
      if(!c){ c={factor:new Map(), staticRows:null, db:new Map(), short:new Map()}; integerGlobalCache.set(key,c); }
      return c;
    }
    function solveCacheKey(settings){
      const pc=settings.parsedComplex ? canonicalComplexString(settings.parsedComplex) : (settings.normalizedRaw || settings.raw || '');
      const checked=[...document.querySelectorAll('[data-sym]:checked')].map(x=>x.dataset.sym).sort().join('');
      const digits=document.getElementById('digits')?.value || '';
      return JSON.stringify({target:pc, restrict:settings.restrict, only:settings.only, never:settings.never, checked, digits, doEq:settings.doEq, doAlg:settings.doAlg, doLog:settings.doLog, modules:settings.modules||{}, constDbTransforms:settings.constDbTransforms||{}, constDbPasses:settings.constDbPasses||{}, hardDbOptions:settings.hardDbOptions||{}, hypDataOptions:settings.hypDataOptions||{}, intsumDbOptions:settings.intsumDbOptions||{}, logOptions:settings.logOptions||{}, mobiusOptions:settings.mobiusOptions||{}, linearComboOptions:settings.linearComboOptions||{}, lfuncOptions:settings.lfuncOptions||{}, integerOptions:settings.integerOptions||{}, moduleLimits:settings.moduleLimits||{}, stageBudgets:settings.stageBudgets||{}, limit:settings.limit, maxAbs:settings.maxAbs, maxRelError:settings.maxRelError, external:settings.allowExternalFactorization});
    }
    function getSolveRunCache(settings){
      const key=solveCacheKey(settings);
      let c=solveRunCache.get(key);
      if(!c){ c={key, full:new Map(), integer:{factor:new Map(), staticRows:null, db:new Map(), short:new Map()}, decimal:{main:new Map()}}; solveRunCache.set(key,c); }
      return c;
    }
    function rowDisplayCompare(a,b){
      const da=Number.isFinite(a?.digits) ? a.digits : digitCountExpr(a?.candidate||'');
      const db=Number.isFinite(b?.digits) ? b.digits : digitCountExpr(b?.candidate||'');
      if(da!==db) return da-db;
      const ba=Number.isFinite(a?.beauty) ? a.beauty : shortRank({s:String(a?.candidate||''),ops:a?.ops||2,depth:1});
      const bb=Number.isFinite(b?.beauty) ? b.beauty : shortRank({s:String(b?.candidate||''),ops:b?.ops||2,depth:1});
      if(ba!==bb) return ba-bb;
      const ea=Number(a?.err||0), eb=Number(b?.err||0);
      if(Number.isFinite(ea) && Number.isFinite(eb) && ea!==eb) return Math.abs(ea)-Math.abs(eb);
      return String(a?.candidate||'').length-String(b?.candidate||'').length;
    }
    function mergeUniqueRows(...groups){
      const order=[], map=new Map();
      for(const group of groups){
        for(const r of (group||[])){
          if(!r) continue;
          const k=candidateEquivalenceKey(r)+'|'+(r.modForm?.code||'')+'|'+(r.qLatex||'')+'|'+(r.value||r.copyValue||'');
          if(!map.has(k)){ map.set(k,r); order.push(k); continue; }
          if(rowDisplayCompare(r,map.get(k))<0) map.set(k,r);
        }
      }
      return order.map(k=>map.get(k)).filter(Boolean);
    }
    function normalizeResultTextKey(s){
      return String(s||'')
        .replace(/[−–—]/g,'-')
        .replace(/\s+/g,'')
        .replace(/\\left|\\right/g,'')
        .replace(/[()]/g,'')
        .toLowerCase();
    }
    function resultRowCategory(r){
      const c=String(r?.candidate||'');
      if(r?.lfuncCategory) return `lfunc-${r.lfuncCategory}`;
      if(r?.specialConstant || /^constant match:/.test(c)) return 'constant';
      if(/^RIES equation:/.test(c)) return 'ries';
      if(/^Möbius relation:|^Mobius relation:/i.test(c) || r?.mobiusCategory) return 'mobius';
      if(r?.hardDbCategory || /^hard constant database:/i.test(c)) return 'harddb';
      if(r?.hypDataCategory || /^hypergeometric database:/i.test(c)) return 'hypdata';
      if(r?.intsumDbCategory || /^integral\/sum database:/i.test(c)) return 'intsumdb';
      if(r?.constantDbCategory || /^constant database:/i.test(c)) return 'constantdb';
      if(r?.lowPrecisionLinearCombo || /^low-precision linear combo:/i.test(c)) return 'linearcombo';
      if(/algebraic/.test(c)) return 'algebraic';
      if(/^log match|^log-combination relation|^log\|c\| linear relation|^log\|c\|/.test(c)) return 'log';
      if(/factorization/.test(c)) return 'factorization';
      if(r?.hideError || /exact/.test(c)) return 'exact';
      return 'other';
    }
    function resultRowRelativeError(r, settings){
      if(!r) return Infinity;
      if(r.hideError || r.err===0 || r.err==='0') return 0;
      const e=Number(r.err);
      if(!Number.isFinite(e)) return Infinity;
      if(r.lfuncCategory) return Math.abs(e); // L-function matcher stores relative residuals.
      const target=Number(settings?.target);
      const denom=Number.isFinite(target) ? Math.max(1,Math.abs(target)) : 1;
      return Math.abs(e)/denom;
    }
    function resultVerifiedDigits(r, settings){
      const rel=resultRowRelativeError(r,settings);
      if(rel===0) return 99;
      if(!Number.isFinite(rel) || rel<=0) return -99;
      return -Math.log10(rel);
    }
    function log10MagnitudeAny(x){
      if(x===undefined || x===null) return 0;
      if(typeof x==='bigint'){
        const a=absBig(x);
        const str=a.toString();
        if(str==='0') return 0;
        // v10.7.1 bugfix: the previous large-integer approximation was also
        // applied to one- and two-digit heights, producing negative log10 values
        // (e.g. height 9 -> -14.05).  That made many algebraic polynomials look
        // artificially shorter than L/log formulas in confidence sorting.
        if(str.length<=16) return Math.log10(Math.max(1, Number(str)));
        const head=Number(str.slice(0,16));
        return str.length-16 + Math.log10(Math.max(1,head));
      }
      const n=Number(x);
      if(!Number.isFinite(n) || n===0) return 0;
      return Math.log10(Math.max(1, Math.abs(n)));
    }
    function resultBodyText(r){
      const c=String(r?.candidate||'');
      const m=c.match(/^[^:]+:\s*(.*)$/);
      return (m?m[1]:c).replace(/copy/gi,'').trim();
    }
    function stripFormulaDecorations(s){
      return String(s||'')
        .replace(/\\operatorname\{([^}]+)\}/g,'$1')
        .replace(/\\(?:left|right|cdot|times|,|!|;|:)/g,'')
        .replace(/\\(?:frac)\{([^{}]+)\}\{([^{}]+)\}/g,'$1/$2')
        .replace(/\\(?:sqrt)\{([^{}]+)\}/g,'sqrt($1)')
        .replace(/\\(?:pi)/g,'π')
        .replace(/\\(?:Gamma)/g,'Γ')
        .replace(/\\/g,'')
        .replace(/\bcopy\b/gi,'')
        .replace(/\s+/g,' ')
        .trim();
    }
    function resultFormulaText(r){
      const cat=resultRowCategory(r);
      const body=resultBodyText(r);
      const latex=stripFormulaDecorations(r?.copyLatex || r?.latex || '');
      if(cat==='lfunc-rational' || cat==='lfunc-quadratic' || cat==='lfunc-log'){
        const m=String(r?.candidate||'').match(/:\s*(x\s*=\s*[^\n]+)$/i);
        return stripFormulaDecorations(m ? m[1] : body);
      }
      if(cat==='log'){
        const m=body.match(/(x\s*≈\s*.*)$/i);
        return stripFormulaDecorations(m ? m[1] : (latex || body));
      }
      if(cat==='mobius') return stripFormulaDecorations(latex || body);
      if(cat==='constant') return stripFormulaDecorations(latex || body);
      if(cat==='algebraic') return stripFormulaDecorations(body);
      if(cat==='ries') return stripFormulaDecorations(body);
      return stripFormulaDecorations(latex || body);
    }
    function formulaVisibleLength(text){
      const s=stripFormulaDecorations(text)
        .replace(/\s+/g,'')
        .replace(/\boperatorname\b/g,'')
        .replace(/\*/g,'·')
        .replace(/\^\((-?\d+(?:\/\d+)?)\)/g,'^$1');
      let len=0;
      for(const ch of s){
        if('πΓ√≈=·+-/^()'.includes(ch)) len += 1;
        else if(/[A-Za-z]/.test(ch)) len += .85;
        else if(/[0-9]/.test(ch)) len += 1;
        else len += .55;
      }
      return len;
    }
    function resultFormulaLengthScore(r){
      const formula=resultFormulaText(r);
      const cat=resultRowCategory(r);
      let len=formulaVisibleLength(formula);
      if(cat==='algebraic'){
        len += Number(r.degree||0)*2.5 + log10MagnitudeAny(r.height!==undefined?r.height:1)*8;
      }
      if(cat==='log'){
        const terms=Number.isFinite(Number(r.terms)) ? Number(r.terms) : ((formula.match(/\*|·|\^|π|Γ|log|exp|e/g)||[]).length || 1);
        const h=Number.isFinite(Number(r.height)) ? Number(r.height) : 0;
        len += Math.max(0, terms-1)*4 + Math.log10(1+Math.max(0,h))*9;
      }
      if(cat==='mobius'){
        const terms=Number.isFinite(Number(r.terms)) ? Number(r.terms) : ((formula.match(/\+|−|-|π|Γ|log|exp|sin|cos|ζ|√/g)||[]).length || 1);
        const h=Number.isFinite(Number(r.height)) ? Number(r.height) : 0;
        len += Math.max(0, terms-2)*2.5 + Math.log10(1+Math.max(0,h))*6;
      }
      if(cat==='harddb'){
        const terms=Number.isFinite(Number(r.terms)) ? Number(r.terms) : 2;
        len += Math.max(0, terms-2)*1.1;
      }
      if(cat==='hypdata'){
        const terms=Number.isFinite(Number(r.terms)) ? Number(r.terms) : 2;
        len += 16 + Math.max(0, terms-2)*1.25;
      }
      if(cat==='intsumdb'){
        const terms=Number.isFinite(Number(r.terms)) ? Number(r.terms) : 2;
        len += 18 + Math.max(0, terms-2)*1.25;
      }
      if(cat==='lfunc-rational'){
        const m=formula.match(/\b(\d+)\b/g)||[];
        len += Math.max(0,m.length-1)*1.5 + resultIntegerTokenScore(r)*.08;
      }
      if(cat==='lfunc-quadratic') len += 22;
      if(cat==='lfunc-log') len += 35;
      if(cat==='ries') len += Math.max(0, (formula.match(/sqrt|√|exp|log|sin|cos|tan|\^/g)||[]).length-1)*2;
      return len;
    }
    function resultIntegerTokenScore(r){
      const text=[resultFormulaText(r), r?.value||'', r?.copyValue||''].join(' ');
      const nums=(text.match(/[-+]?\d+/g)||[]).map(x=>Math.abs(Number(x))).filter(Number.isFinite);
      if(!nums.length) return 0;
      const sum=nums.reduce((a,b)=>a+Math.min(10000,b),0);
      const max=Math.max(...nums);
      return Math.log10(1+sum)*10 + Math.log10(1+max)*5 + nums.length*1.5;
    }
    function resultPrettyCompactnessScore(r){
      const cat=resultRowCategory(r);
      const len=resultFormulaLengthScore(r);
      const intScore=resultIntegerTokenScore(r);
      if(cat==='exact') return 10 + len*.7 + intScore*.15;
      if(cat==='factorization') return 60 + len*.9 + intScore*.2;
      if(cat==='constant') return 16 + len*.75 + intScore*.12;
      if(cat==='log') return 20 + len*.9 + intScore*.10;
      if(cat==='mobius') return 21 + len*.9 + intScore*.10;
      if(cat==='harddb') return 22 + len*.88 + intScore*.10;
      if(cat==='hypdata') return 24 + len*.92 + intScore*.10;
      if(cat==='intsumdb') return 25 + len*.92 + intScore*.10;
      if(cat==='lfunc-rational') return 23 + len*.92 + intScore*.12;
      if(cat==='ries') return 24 + len*.95 + intScore*.14;
      if(cat==='algebraic') return 28 + len*1.0 + intScore*.18;
      if(cat==='lfunc-quadratic') return 60 + len*1.05 + intScore*.15;
      if(cat==='lfunc-log') return 75 + len*1.08 + intScore*.16;
      return 120 + len + intScore*.2;
    }
    function resultComplexityScore(r){
      return resultPrettyCompactnessScore(r);
    }
    function resultInputPrecisionRatio(r, settings){
      const sig=Math.max(1, Math.min(60, typedInputPrecision(settings || {})));
      const rel=resultRowRelativeError(r,settings);
      const typedTol=typedRelativeToleranceNumber(sig, 1, 1, 60);
      if(rel===0) return 0;
      if(!Number.isFinite(rel)) return Infinity;
      return Math.abs(rel)/Math.max(typedTol,1e-300);
    }
    function resultAcceptBucket(r, settings){
      const compact=resultPrettyCompactnessScore(r);
      const ratio=resultInputPrecisionRatio(r,settings);
      if(!Number.isFinite(ratio)) return 9;
      if(ratio<=1) return 0;
      if(ratio<=100) return 1;
      // Extra-short formulas are often the mathematically meaningful target even
      // when a dense relation has 2–4 more accidental residual digits.  Keep
      // them in the first confidence screen, but below fully typed-precision hits.
      if(compact<=55 && ratio<=10000) return 2;
      if(compact<=80 && ratio<=2000) return 2;
      if(ratio<=10000) return 3;
      return 6;
    }
    function resultConfidenceScore(r, settings){
      const cat=resultRowCategory(r);
      const compact=resultPrettyCompactnessScore(r);
      const bucket=resultAcceptBucket(r,settings);
      // v9.6 sorting-only change: precision is a gate/bucket, not the main
      // ordering signal.  Inside a precision bucket, expression length,
      // coefficient height and visible simplicity dominate.  This intentionally
      // promotes results such as x≈π^(-1)Γ(1/4), x=L(f,1), x≈exp(π), and compact
      // low-height algebraic equations over much longer LLL artefacts whose
      // residual is only modestly smaller.
      let score=bucket*100000 + compact*100;
      if(cat==='log' && compact<55) score-=900;
      if(cat==='mobius' && compact<65) score-=860;
      if(cat==='harddb' && compact<70) score-=850;
      if(cat==='hypdata' && compact<78) score-=835;
      if(cat==='intsumdb' && compact<82) score-=832;
      if(cat==='constantdb' && compact<70) score-=840;
      if(cat==='linearcombo' && compact<78) score-=830;
      if(cat==='lfunc-rational' && compact<55) score-=820;
      if(cat==='constant' && compact<55) score-=780;
      if(cat==='ries' && compact<65) score-=620;
      if(cat==='algebraic' && compact<95) score-=520;
      if(cat==='lfunc-quadratic') score+=1200;
      if(cat==='lfunc-log') score+=1600;
      return score;
    }
    function modulePriority(cat){
      if(cat==='exact') return 0;
      if(cat==='ries') return 1;
      if(cat==='linearcombo') return 2;
      if(cat==='log') return 3;
      if(cat==='mobius') return 4;
      if(cat==='harddb') return 5;
      if(cat==='hypdata') return 6;
      if(cat==='intsumdb') return 7;
      if(cat==='constantdb') return 8;
      if(cat==='lfunc-rational') return 9;
      if(cat==='constant') return 10;
      if(cat==='algebraic') return 11;
      if(cat==='lfunc-quadratic') return 12;
      if(cat==='lfunc-log') return 13;
      if(cat==='factorization') return 14;
      return 10;
    }
    function resultLengthFirstScore(r){
      // v10.7.1: the confidence view is a presentation ranking, not a pure
      // numerical-residual ranking.  Use only the visible formula/equation text
      // plus small structural penalties here; never let a long algebraic row win
      // just because its residual has a few more accidental digits.
      const cat=resultRowCategory(r);
      let len=resultFormulaLengthScore(r);
      if(cat==='lfunc-rational') len -= 8; // x = L(f,1) should be visibly short.
      if(cat==='log') len += Math.max(0, Number(r?.terms||0)-1)*1.5;
      if(cat==='mobius') len += Math.max(0, Number(r?.terms||0)-2)*1.2;
      if(cat==='harddb') len += Math.max(0, Number(r?.terms||0)-2)*1.0;
      if(cat==='hypdata') len += 10 + Math.max(0, Number(r?.terms||0)-2)*1.1;
      if(cat==='intsumdb') len += 11 + Math.max(0, Number(r?.terms||0)-2)*1.1;
      if(cat==='constantdb') len += Math.max(0, Number(r?.terms||0)-2)*1.1;
      if(cat==='linearcombo') len += Math.max(0, Number(r?.terms||0)-1)*1.0 + Math.max(0, Number(r?.denominator||1)-1)*0.08;
      return len;
    }
    function resultNumericHeight(r){
      try{
        if(typeof r?.height==='bigint') return Number(r.height>1000000000000n ? 1000000000000n : r.height);
        const h=Number(r?.height);
        return Number.isFinite(h) ? Math.abs(h) : 1e9;
      }catch(e){ return 1e9; }
    }
    function resultExceptionalSimple(r, settings){
      const cat=resultRowCategory(r);
      const ratio=resultInputPrecisionRatio(r,settings);
      if(!(ratio<=10 || resultRowRelativeError(r,settings)===0)) return false;
      const terms=Number.isFinite(Number(r?.terms)) ? Number(r.terms) : ((resultFormulaText(r).match(/[+−-]/g)||[]).length+1);
      const height=resultNumericHeight(r);
      const len=resultFormulaLengthScore(r);
      if(cat==='mobius') return terms<=2 && height<=3 && len<=72;
      if(cat==='linearcombo') return terms<=2 && height<=12 && len<=78;
      if(cat==='log') return terms<=2 && height<=20 && len<=76;
      if(cat==='harddb') return terms<=2 && height<=36 && len<=82;
      if(cat==='constantdb') return terms<=2 && height<=24 && len<=82;
      if(cat==='hypdata') return terms<=2 && height<=28 && len<=110;
      if(cat==='intsumdb') return terms<=2 && height<=28 && len<=120;
      if(cat==='ries') return len<=58;
      if(cat==='algebraic') return Number(r?.degree||99)<=2 && height<=100 && len<=95;
      if(cat==='lfunc-rational') return len<=70;
      if(cat==='constant') return len<=55;
      return len<=48;
    }
    function rowConfidenceCompare(settings){
      return (a,b)=>{
        // v11.6: confidence ordering remains module-wise round-robin, but each
        // module queue is mostly precision-first.  Exceptionally simple, low-height
        // rows may pass a slightly more accurate row only when they are still within
        // about one order of magnitude of the typed-input tolerance.
        const ba=resultAcceptBucket(a,settings), bb=resultAcceptBucket(b,settings);
        if(ba!==bb) return ba-bb;
        const sa=resultExceptionalSimple(a,settings), sb=resultExceptionalSimple(b,settings);
        if(sa!==sb) return sa ? -1 : 1;
        const va=resultVerifiedDigits(a,settings), vb=resultVerifiedDigits(b,settings);
        if(Math.abs(va-vb)>1.0) return vb-va;
        const ca=resultPrettyCompactnessScore(a), cb=resultPrettyCompactnessScore(b);
        if(Math.abs(ca-cb)>1e-9) return ca-cb;
        if(Math.abs(va-vb)>1e-9) return vb-va;
        const ea=resultRowRelativeError(a,settings), eb=resultRowRelativeError(b,settings);
        if(ea!==eb) return ea-eb;
        const ma=modulePriority(resultRowCategory(a)), mb=modulePriority(resultRowCategory(b));
        if(ma!==mb) return ma-mb;
        return String(a.candidate||'').localeCompare(String(b.candidate||''));
      };
    }
    function resultRoundRobinModuleKey(r){
      const cat=resultRowCategory(r);
      // Keep the visible RIES submodules independent in the confidence merge.
      // In particular, exact-input algebraic and ordinary algebraic
      // rows both map to algebraic, so their own candidates are length-sorted and
      // interleaved like every other module instead of being emitted as a block.
      if(r?.lfuncCategory) return `lfunc-${r.lfuncCategory}`;
      return cat || 'other';
    }
    function confidenceRoundRobinHeadCompare(settings){
      const cmp=rowConfidenceCompare(settings);
      return (a,b)=>{
        const c=cmp(a,b);
        if(c!==0) return c;
        const ma=modulePriority(resultRowCategory(a)), mb=modulePriority(resultRowCategory(b));
        if(ma!==mb) return ma-mb;
        return String(resultRoundRobinModuleKey(a)).localeCompare(String(resultRoundRobinModuleKey(b)));
      };
    }
    function confidenceSortedRows(rows, settings){
      const source=(rows||[]).filter(Boolean);
      const cmp=rowConfidenceCompare(settings);
      const groups=new Map();
      const moduleFirstIndex=new Map();
      source.forEach((r,idx)=>{
        const k=resultRoundRobinModuleKey(r);
        if(!groups.has(k)){ groups.set(k,[]); moduleFirstIndex.set(k,idx); }
        groups.get(k).push(r);
      });
      // Sort every submodule independently by visible length first.  Algebraic
      // exact and non-exact rows share the same algebraic queue, so they cannot
      // produce a residual-ordered block ahead of shorter L/log rows.
      for(const g of groups.values()) g.sort(cmp);
      const headCmp=confidenceRoundRobinHeadCompare(settings);
      const modules=[...groups.keys()].sort((ka,kb)=>{
        const a=groups.get(ka)[0], b=groups.get(kb)[0];
        const c=headCmp(a,b);
        if(c!==0) return c;
        return (moduleFirstIndex.get(ka)||0)-(moduleFirstIndex.get(kb)||0);
      });
      const out=[];
      let depth=0;
      while(out.length<source.length){
        const batch=[];
        for(const k of modules){
          const r=groups.get(k)[depth];
          if(r) batch.push(r);
        }
        if(!batch.length) break;
        // Re-sort each complete layer by visible length.  This is the important
        // final pass: after all asynchronous modules have finished, we collect
        // the first item from every module, order that layer, then do the same
        // for second items, third items, and so on.
        batch.sort(headCmp);
        out.push(...batch);
        depth++;
      }
      return out;
    }
    function renderFinalDefault(allRows, discoveryRows, settings){
      const all=(allRows||[]).slice();
      const discovery=(discoveryRows&&discoveryRows.length ? discoveryRows : all).slice();
      const display=confidenceSortedRows(all, settings);
      renderRows(display,{final:true, allRows:all, discoveryRows:discovery, settings, sorted:true});
    }
    function resultEquivalentKey(r, settings){
      if(!r) return '';
      const cat=resultRowCategory(r);
      // Within one target, rational/log/quadratic L-function derivations that use
      // the same modular form and same L-value are mathematically competing
      // explanations of the same number. Keep the clearest, best-verified one.
      if(r.lfuncCategory){
        const lkey=r.lfuncEntryKey || `${r.modForm?.code||''}|${r.lfuncLabel || ((String(r.value||'').match(/L\(f,\d\)/)||[''])[0])}`;
        return `lfunc:${lkey}:${canonicalTargetString(settings?.raw||'', settings?.target)}`;
      }
      if(cat==='algebraic'){
        const m=String(r.candidate||'').match(/algebraic:\s*(.+)$/i);
        return 'alg:'+normalizeResultTextKey(m?m[1]:r.candidate);
      }
      if(cat==='log') return 'log:'+normalizeResultTextKey(r.candidate)+'|'+normalizeResultTextKey(r.value||r.copyValue||'');
      if(cat==='mobius') return 'mobius:'+normalizeResultTextKey(r.candidate);
      if(r?.hardDbCategory) return 'harddb:'+String(r.constantDbId||'')+':'+normalizeResultTextKey(r.candidate);
      if(r?.hypDataCategory) return 'hypdata:'+String(r.constantDbId||'')+':'+normalizeResultTextKey(r.candidate);
      if(cat==='constantdb') return 'constantdb:'+normalizeResultTextKey(r.candidate);
      if(cat==='linearcombo') return 'linearcombo:'+normalizeResultTextKey(r.candidate);
      return candidateEquivalenceKey(r)+'|'+normalizeResultTextKey(r.value||r.copyValue||'')+'|'+(r.modForm?.code||'');
    }
    function dedupeEquivalentRows(rows, settings){
      const map=new Map(), order=[];
      for(const r of (rows||[])){
        if(!r) continue;
        const k=resultEquivalentKey(r,settings);
        if(!map.has(k)){ map.set(k,r); order.push(k); continue; }
        const old=map.get(k);
        if(resultConfidenceScore(r,settings) < resultConfidenceScore(old,settings)) map.set(k,r);
      }
      return order.map(k=>map.get(k)).filter(Boolean);
    }
    function updateResultTools(final=false){
      if(!resultTools) return;
      if(!final || !currentResultAllRows.length){ resultTools.hidden=true; return; }
      resultTools.hidden=false;
      const n=currentResultAllRows.length;
      const shown=currentResultSorted ? 'confidence order' : 'discovery/group order';
      if(resultToolsMeta) resultToolsMeta.textContent=`${n} current result${n===1?'':'s'} shown · ${shown}; equivalent L/rational/quadratic/log duplicates are collapsed.`;
      if(sortConfidenceBtn) sortConfidenceBtn.hidden=!!currentResultSorted;
      if(sortDiscoveryBtn) sortDiscoveryBtn.hidden=!currentResultSorted;
    }
    function clearResultTools(){
      currentResultAllRows=[]; currentResultDiscoveryRows=[]; currentResultSettings=null; currentResultSorted=false;
      updateResultTools(false);
    }
    async function solve(){
      if(activeSolveRun) activeSolveRun.stopped=true;
      if(activeShortformRun) activeShortformRun.stopped=true;
      const settings=readSettings(); updatePreview(settings);
      const run={stopped:false, epoch:inputStateEpoch, raw:String(settings.raw||'').trim(), kind:'solve'};
      activeSolveRun=run;
      const isContinuation=!!pendingContinueSolve && String(settings.raw||'').trim()===String(lastSolvedRaw||'').trim();
      pendingContinueSolve=false;
      const seedRows=isContinuation ? lastRenderedRows.slice() : [];
      if(!isContinuation) clearResultTools();
      const runCache=getSolveRunCache(settings);
      lastSolvedRaw=settings.raw;
      continueBtn.disabled=true;
      if(statusEl) statusEl.dataset.progress='0';
      setSearchStatus('Preparing search space…', .06, 'initializing');
      stopBtn.disabled=false;
      await nextPaint();
      abortIfStaleOrStopped(run);
      const t0=performance.now();
      let rows = seedRows.slice();
      if(seedRows.length) renderRows(seedRows);
      const hpEv = highPrecisionEval(settings);
      renderHighPrecision(hpEv, settings);
      prepareNumberTools(settings);
      if(!Number.isFinite(settings.target) && !settings.complexTarget && !settings._hpIntegerString){
        if(hpEv){ renderRows([]); statusEl.className='notice status-line good'; statusEl.textContent= hpEv.error ? 'High-precision expression could not be evaluated; no RIES target was available.' : 'High-precision value shown separately. RIES search skipped because the value is complex or outside finite double range.'; resetContinueState(); if(activeSolveRun===run) activeSolveRun=null; stopBtn.disabled=true; return; }
        statusEl.textContent='Please enter a valid real number, decimal complex number, or supported computable expression.'; statusEl.className='notice status-line bad'; resetContinueState(); if(activeSolveRun===run) activeSolveRun=null; stopBtn.disabled=true; return;
      }
      // v7.1: if a computable expression evaluates to an exact integer, use that
      // integer as the target and run the same factorization/database/shortform
      // pipeline as for literal integer input.
      const integerValue = resolvedIntegerBig(settings);
      const isInteger = integerValue !== null;
      const fullCacheKey = isInteger ? `integer-effort:${settings.shortEffort}` : `real-level:${settings.level}`;
      if(runCache.full.has(fullCacheKey)){
        rows=runCache.full.get(fullCacheKey).slice();
        const cachedRows=dedupeEquivalentRows(rows,settings);
        renderFinalDefault(cachedRows, cachedRows, settings);
        statusEl.className='notice status-line good';
        statusEl.textContent=`Returned ${cachedRows.length} cached result(s); no repeated computation for this input/effort.`;
        if(isInteger){
          const curEffort=Math.max(0, Math.min(7, Number(document.getElementById('shortEffort')?.value || 0)));
          if(curEffort>=7) setContinueState('shortform', 'Max effort reached', true); else setContinueState('shortform', `Continue at effort ${curEffort+1}`, false);
        }else{
          const curLevel=Math.max(1, Math.min(9, Number(document.getElementById('level')?.value || 4)));
          if(curLevel>=9) setContinueState('ries', 'Max RIES level reached', true); else setContinueState('ries', `Continue at RIES level ${curLevel+1}`, false);
        }
        if(activeSolveRun===run) activeSolveRun=null;
        stopBtn.disabled=true;
        return;
      }
      let constants=[];
      if(isInteger){
        if(settings.modules?.integer===false){
          renderFinalDefault([], [], settings);
          runCache.full.set(fullCacheKey, []);
          statusEl.className='notice status-line';
          statusEl.textContent='Integer input detected, but the Integer search module is disabled in Parameters.';
          resetContinueState();
          if(activeSolveRun===run) activeSolveRun=null;
          stopBtn.disabled=true;
          return;
        }
        runBtn.disabled=true;
        run.kind='integer';
        activeShortformRun=run;
        stopBtn.disabled=false;
        const icache=getIntegerGlobalCache(settings);
        const curEffort=Math.max(0, Math.min(7, Number(document.getElementById('shortEffort')?.value || 0)));
        try{
          setSearchStatus('Factoring integer first…', .10, 'integer phase');
          await nextPaint();
          abortIfStaleOrStopped(run);
          let factor=null;
          for(let e=curEffort;e>=0;e--){ if(icache.factor.has(e)){ factor=icache.factor.get(e); break; } }
          if(settings.integerOptions?.factor!==false && (!factor || !(factor||[]).some(r=>/^integer factorization$/.test(r.candidate||'')))){
            factor = await factorRowsAsync(settingsForModule(settings,'integer'), info=>{
              const elapsed=info?.elapsed ?? 0;
              const lim=info?.limitMs || 1;
              const phasePct=.10 + Math.min(.10, (elapsed/lim)*.10);
              const found=(info?.factors || []).length;
              setSearchStatus(`Factoring integer first… ${elapsed} ms / ${Math.round(lim)} ms${found?`, ${found} factor(s) found`:''}`, phasePct, 'integer phase');
            });
            icache.factor.set(curEffort, factor);
          }
          if(settings.integerOptions?.factor===false) factor=[];
          rows=mergeUniqueRows(seedRows, rows, factor);
          renderRows(rows);
          abortIfStaleOrStopped(run);
          await nextPaint();
          abortIfStaleOrStopped(run);
          const rawAbsForDb=absBig(resolvedIntegerBig(settings)||0n);
          let shortformReady=isShortformDbReady();
          if(settings.integerOptions?.db!==false && rawAbsForDb<=100000n){
            setSearchStatus('Loading precomputed shortform database…', .20, 'integer database');
            await nextPaint();
            abortIfStaleOrStopped(run);
            shortformReady = await ensureShortformDbLoaded();
            abortIfStaleOrStopped(run);
            if(!shortformReady) console.warn('RIES precomputed shortform database is unavailable; structured and exact searches will still run.');
          }else{
            // v8.5: the 100k precomputed table is useful for small targets, but
            // parsing the multi-megabyte file before every 11+ digit integer
            // made Continue look frozen.  Large targets use the deterministic
            // structured database immediately; compactLiteralD will still use
            // the 100k table if it was already loaded by a prior small query.
            setSearchStatus('Skipping 100k precomputed table for this large integer; using structured database directly…', .22, 'integer database');
            await nextPaint();
            abortIfStaleOrStopped(run);
          }
          setSearchStatus(settings.integerOptions?.db===false ? 'Skipping integer database by Parameters…' : 'Checking precomputed and structured integer database…', .24, 'integer database');
          await nextPaint();
          abortIfStaleOrStopped(run);
          if(settings.integerOptions?.db!==false && !icache.staticRows) icache.staticRows=staticShortformRows(settingsForModule(settings,'integer'));
          if(settings.integerOptions?.db!==false && !icache.db.has(curEffort)){
            await yieldAndCheck(run);
            const dbRows=await integerDatabaseRowsResponsive(settingsForModule(settings,'integer'), info=>{
              const elapsed=Number(info?.elapsed||0);
              const budget=Number(info?.budgetMs || settings._databaseBudgetMs || 1);
              const pct=.24 + Math.min(.18, Math.max(0, elapsed/budget)*.18);
              const found=Number(info?.rows||0);
              const label=info?.label || settings._databasePhase || 'database';
              setSearchStatus(`Structured integer database · ${label} · ${Math.round(elapsed)} ms / ${Math.round(budget)} ms${found?`, ${found} candidate(s)`:''}`, pct, 'integer database');
            });
            abortIfStaleOrStopped(run);
            icache.db.set(curEffort, dbRows);
          }
          const dbGroups=[]; if(settings.integerOptions?.db!==false) for(const [e,rs] of icache.db.entries()) if(e<=curEffort) dbGroups.push(rs);
          const priorShortGroups=[]; if(settings.integerOptions?.shortform!==false) for(const [e,rs] of icache.short.entries()) if(e<curEffort) priorShortGroups.push(rs);
          rows=mergeUniqueRows(seedRows, factor, settings.integerOptions?.db!==false ? icache.staticRows : [], ...dbGroups, ...priorShortGroups);
          await yieldAndCheck(run);
          const rawAbs=absBig(resolvedIntegerBig(settings)||0n);
          const integerDisplayLimit=Math.max(5, Math.min(20, Number(settings?.moduleLimits?.integer || settings.limit)||5));
          const dbShortRows=selectDigitShortforms(rows.filter(r=>/(precomputed shortform|database):/.test(r.candidate||'')),integerDisplayLimit);
          const td=decimalDigitCountBig(rawAbs);
          const dbBestDigits=dbShortRows[0]?.digits ?? 999;
          if(rawAbs<=100000n && dbShortRows.length>=5 && dbBestDigits<=Math.max(2,Math.ceil(td*.55))){
            const factorOnly=rows.filter(r=>/^integer factorization/.test(r.candidate||''));
            rows=mergeUniqueRows(factorOnly, dbShortRows);
            renderFinalDefault(rows, rows, settings);
            runCache.full.set(fullCacheKey, rows.slice());
            const dt=Math.round(performance.now()-t0);
            statusEl.className='notice status-line good';
            statusEl.textContent=`Returned ${rows.length} result(s) in ${dt} ms. Precomputed/database search already found five compact, non-equivalent forms; deeper shortform search skipped.`;
            if(curEffort>=7) setContinueState('shortform', 'Max effort reached', true);
            else setContinueState('shortform', `Continue at effort ${curEffort+1}`, false);
            return;
          }
          renderRows(rows);
          await idle();
          abortIfStaleOrStopped(run);
          if(settings.integerOptions?.shortform!==false && !icache.short.has(curEffort)){
            await nextPaint();
            abortIfStaleOrStopped(run);
            const baseRows=rows.slice();
            const shortRows = await integerShortformRowsAsync(settingsForModule(settings,'integer'), (partial, info={})=>{
              const priorShort=[]; for(const [e,rs] of icache.short.entries()) if(e<curEffort) priorShort.push(rs);
              const merged=mergeUniqueRows(baseRows, ...priorShort, partial);
              renderRows(merged);
              const dt= Math.round(performance.now()-t0);
              const note = partial.length ? `best uses ${partial[0].digits} digit(s)` : 'no shortform yet';
              const tdNow=Math.max(1, decimalDigitCountBig(absBig(resolvedIntegerBig(settings)||0n)));
              const budgetProgress=Number(settings._shortformMaxDigits || 1)/tdNow;
              const extraProgress=info?.exhaustive ? Math.min(.14, Number(info.frac||0)*.14) : 0;
              const progressBase=Math.min(.96, .32 + budgetProgress*.48 + extraProgress);
              const phase=info?.phase || settings._shortformPhase || 'exact search';
              setSearchStatus(`Exact integer shortforms · ${phase} · effort ${settings.shortEffort}, digit budget ${settings._shortformMaxDigits ?? '?'}, ${settings._shortformDbSize ?? 0} cached forms, ${note}, ${dt} ms.`, progressBase, 'shortform search');
            });
            abortIfStaleOrStopped(run);
            icache.short.set(curEffort, shortRows);
          }
          const shortGroups=[]; if(settings.integerOptions?.shortform!==false) for(const [e,rs] of icache.short.entries()) if(e<=curEffort) shortGroups.push(rs);
          rows=mergeUniqueRows(seedRows, rows, ...shortGroups);
          renderFinalDefault(rows, rows, settings);
          runCache.full.set(fullCacheKey, rows.slice());
          const dt=Math.round(performance.now()-t0);
          const reason = run.stopped || settings._shortformStopped ? 'stopped by user' : (settings._shortformEarly ? 'ended early after finding a very small expression' : 'search phase complete');
          statusEl.className='notice status-line good';
          statusEl.textContent=`Returned ${rows.length} integer result(s) in ${dt} ms. Cached previous effort results were reused; newly searched effort ${curEffort}. Integer algebraic search is skipped.`;
          if(curEffort>=7) setContinueState('shortform', 'Max effort reached', true);
          else setContinueState('shortform', `Continue at effort ${curEffort+1}`, false);
        }catch(e){
          const msg=String(e && e.message);
          if(msg!=='RIES_STOPPED' && msg!=='RIES_STALE_INPUT') throw e;
          if(msg==='RIES_STOPPED'){
            rows=mergeUniqueRows(seedRows, rows);
            if(rows.length) renderFinalDefault(rows, rows, settings);
            runCache.full.set(fullCacheKey, rows.slice());
            statusEl.className='notice status-line';
            statusEl.textContent='Stopped. Current integer results are kept on screen and cached for this input.';
          }
        }finally{
          if(activeShortformRun===run) activeShortformRun=null;
          if(activeSolveRun===run) activeSolveRun=null;
          stopBtn.disabled=true;
          runBtn.disabled=false;
        }
        return;
      }
      runBtn.disabled=true;
      run.kind='decimal';
      stopBtn.disabled=false;
      try{
        await nextPaint();
        abortIfStaleOrStopped(run);
        setSearchStatus('Building RIES equation candidates…', .18, 'formula search');
        await nextPaint();
        abortIfStaleOrStopped(run);
        const runHighPrecisionAlg=shouldRunHighPrecisionAlgebraic(settings);
        const runLowPrecisionAlg=shouldRunLowPrecisionAlgebraic(settings);
        if(settings.doEq && !runHighPrecisionAlg && Number.isFinite(settings.target) && !settings.complexTarget){ constants=generateConstants(settingsForModule(settings,'riesEq')); rows=rows.concat(equationSearch(constants, settingsForModule(settings,'riesEq'))); }
        if(shouldRunLowPrecisionLinearComboRows(settings)){
          setSearchStatus('Checking sparse low-precision linear combinations…', .32, 'linear-combination search');
          await nextPaint();
          abortIfStaleOrStopped(run);
          const lcRows=lowPrecisionLinearComboRows(settingsForModule(settings,'linearCombo'), info=>{
            abortIfStaleOrStopped(run);
            const frac=Number(info?.frac||0);
            setSearchStatus(`Checking sparse low-precision linear combinations · ${info?.phase||'scan'} ${(frac*100).toFixed(0)}% · ${info?.rows?.length || 0} candidate(s)`, .32 + Math.min(.08, frac*.08), 'linear-combination search');
            if(info?.rows?.length) renderRows(mergeUniqueRows(rows, info.rows));
          });
          abortIfStaleOrStopped(run);
          if(lcRows.length){ rows=mergeUniqueRows(rows,lcRows); renderRows(rows); await nextPaint(); abortIfStaleOrStopped(run); }
        }
        if(hardDbPotentiallyRunnable(settings)){
          setSearchStatus('Loading filtered 80k hard-constant database package…', .40, 'loading package');
          await nextPaint();
          abortIfStaleOrStopped(run);
          const hardDbLoaded=await ensureHardDbLoaded({settings, stage:hardDbMaxStage(settings), label:'filtered hard-constant database', phase:'loading package', baseProgress:.40, spanProgress:.045});
          abortIfStaleOrStopped(run);
          if(hardDbLoaded && hardDbShouldRun(settings)){
            setSearchStatus('Checking filtered 80k hard-constant database…', .445, 'hard-constant database search');
            await nextPaint();
            abortIfStaleOrStopped(run);
            const hdbRows=await hardDbRowsAsync(settings, info=>{
              abortIfStaleOrStopped(run);
              const done=Number(info?.done||0), total=Math.max(1,Number(info?.total||1));
              const frac=Math.min(1, done/total);
              setSearchStatus(`Checking filtered 80k hard-constant database · ${info?.phase||'scan'} ${(frac*100).toFixed(0)}% · ${info?.rows?.length || 0} candidate(s)`, .445 + Math.min(.045, frac*.045), 'hard-constant database search');
            });
            abortIfStaleOrStopped(run);
            if(hdbRows.length){ rows=mergeUniqueRows(rows,hdbRows); renderRows(rows); await nextPaint(); abortIfStaleOrStopped(run); }
          }else{
            console.warn('Filtered hard-constant database package is unavailable; continuing with other modules.');
          }
        }
        if(hypDataPotentiallyRunnable(settings)){
          const hypStage=hypDataMaxStage(settings);
          const hypLevel=RIES_HYPDATA_ASSET_LEVELS[Math.max(0,hypStage-1)]?.level || 4;
          setSearchStatus(`Loading hypergeometric pFq database chunks through level ${hypLevel}…`, .49, 'loading package');
          await nextPaint();
          abortIfStaleOrStopped(run);
          const hypLoaded=await ensureHypDataLoaded({settings, stage:hypStage, label:'hypergeometric pFq database', phase:'loading package', baseProgress:.49, spanProgress:.05});
          abortIfStaleOrStopped(run);
          if(hypLoaded && isHypDataReady(hypStage)){
            setSearchStatus(`Checking hypergeometric pFq database · cumulative level ${hypLevel}…`, .54, 'hypergeometric database search');
            await nextPaint();
            abortIfStaleOrStopped(run);
            const hypRows=await hypDataRowsAsync(settings, info=>{
              abortIfStaleOrStopped(run);
              const done=Number(info?.done||0), total=Math.max(1,Number(info?.total||1));
              const frac=Math.min(1, done/total);
              setSearchStatus(`Checking hypergeometric pFq database · ${info?.phase||'scan'} ${(frac*100).toFixed(0)}% · ${info?.rows?.length || 0} candidate(s)`, .54 + Math.min(.05, frac*.05), 'hypergeometric database search');
            });
            abortIfStaleOrStopped(run);
            if(hypRows.length){ rows=mergeUniqueRows(rows,hypRows); renderRows(rows); await nextPaint(); abortIfStaleOrStopped(run); }
          }else{
            console.warn('Hypergeometric pFq database package is unavailable; continuing with other modules.');
          }
        }
        if(intsumDbPotentiallyRunnable(settings)){
          const intStage=intsumDbMaxStage(settings);
          const intLevel=RIES_INTSUMDB_ASSET_LEVELS[Math.max(0,intStage-1)]?.level || 4;
          setSearchStatus(`Loading integral/sum database chunks through level ${intLevel}…`, .585, 'loading package');
          await nextPaint();
          abortIfStaleOrStopped(run);
          const intLoaded=await ensureIntsumDbLoaded({settings, stage:intStage, label:'integral/sum database', phase:'loading package', baseProgress:.585, spanProgress:.045});
          abortIfStaleOrStopped(run);
          if(intLoaded && isIntsumDbReady(intStage)){
            setSearchStatus(`Checking integral/sum database · cumulative level ${intLevel}…`, .63, 'integral/sum database search');
            await nextPaint();
            abortIfStaleOrStopped(run);
            const intRows=await intsumDbRowsAsync(settings, info=>{
              abortIfStaleOrStopped(run);
              const done=Number(info?.done||0), total=Math.max(1,Number(info?.total||1));
              const frac=Math.min(1, done/total);
              setSearchStatus(`Checking integral/sum database · ${info?.phase||'scan'} ${(frac*100).toFixed(0)}% · ${info?.rows?.length || 0} candidate(s)`, .63 + Math.min(.045, frac*.045), 'integral/sum database search');
            });
            abortIfStaleOrStopped(run);
            if(intRows.length){ rows=mergeUniqueRows(rows,intRows); renderRows(rows); await nextPaint(); abortIfStaleOrStopped(run); }
          }else{
            console.warn('Integral/sum database package is unavailable; continuing with other modules.');
          }
        }
        if(lfuncShouldRun(settings)){
          setSearchStatus('Checking L-function database against decimal input…', .675, 'L-function search');
          await nextPaint();
          abortIfStaleOrStopped(run);
          const curLevelForLfunc=Math.max(1, Math.min(9, Number(document.getElementById('level')?.value || 4)));
          const lfuncEffort=Math.max(0, Math.min(7, curLevelForLfunc-DEFAULT_RIES_LEVEL));
          const lfRows=await lfuncRowsAsync(settingsForModule(settings,'lfunc'), lfuncEffort, info=>{
            const done=Number(info?.done||0), total=Math.max(1,Number(info?.total||1));
            const frac=Math.min(1, done/total);
            setSearchStatus(`Checking L-function database · effort ${lfuncEffort}, ${info?.phase||'scan'} ${(frac*100).toFixed(0)}%`, .45 + frac*.07, 'L-function search');
          });
          abortIfStaleOrStopped(run);
          if(lfRows.length){ rows=mergeUniqueRows(rows,lfRows); renderRows(rows); await nextPaint(); abortIfStaleOrStopped(run); }
          const scRows=specialDecimalConstantRows(settingsForModule(settings,'lfunc'), lfuncEffort);
          if(scRows.length){ rows=mergeUniqueRows(rows,scRows); renderRows(rows); await nextPaint(); abortIfStaleOrStopped(run); }
        }
        let algMaxHeightForFilter=1000000000000n;
        if(runHighPrecisionAlg){
          setSearchStatus('Running high-precision algebraic relation search…', .56, 'algebraic search');
          await nextPaint();
          abortIfStaleOrStopped(run);
          let maxH; try{ maxH=BigInt(document.getElementById('algHeight').value.trim() || '1000000000000'); }catch(e){ maxH=1000000000000n; }
          algMaxHeightForFilter=maxH;
          const deg=Math.max(8, Math.min(14, Number(document.getElementById('algDegree').value)||10));
          const autoPrec=typedInputPrecision(settings);
          const prec=Math.max(1, Math.min(120, autoPrec));
          const slack=Math.max(2, Math.min(6, autoPrec<=12 ? 2 : 3));
          rows=rows.concat(exactInputAlgebraicRows(settingsForModule(settings,'algebraic'), maxH, settings.moduleLimits?.algebraic || settings.limit));
          rows=rows.concat(relationCandidates(settingsForModule(settings,'algebraic'), deg, prec, maxH, Math.max(settings.moduleLimits?.algebraic || settings.limit,12), slack));
        }
        setSearchStatus(runHighPrecisionAlg ? 'Checking logarithmic combinations and final ranking…' : 'Skipping early algebraic reconstruction; checking RIES/log/database results first…', .82, 'final pass');
        await nextPaint();
        abortIfStaleOrStopped(run);
        if(settings.doLog && !runHighPrecisionAlg && Number.isFinite(settings.target) && !settings.complexTarget) rows=rows.concat(logRelationRows(settings.target, settingsForModule(settings,'log')));
        if(shouldRunMobiusRows(settings)){
          setSearchStatus('Checking Möbius fractional-linear constant relations…', .86, 'Möbius search');
          await nextPaint();
          abortIfStaleOrStopped(run);
          rows=rows.concat(mobiusRelationRows(settingsForModule(settings,'mobius')));
        }
        if(shouldRunConstantDbRows(settings)){
          setSearchStatus('Checking low-precision constant database matches…', .88, 'constant database');
          await nextPaint();
          abortIfStaleOrStopped(run);
          const cdbRows=await constantDbRowsAsync(settingsForModule(settings,'constantDb'), info=>{
            abortIfStaleOrStopped(run);
            const frac=Number(info?.frac||0);
            const phase=info?.phase || 'scan';
            setSearchStatus(`Checking low-precision constant database matches · ${phase} ${(frac*100).toFixed(0)}% · ${info?.rows?.length || 0} candidate(s)`, .88 + Math.min(.09, frac*.09), 'constant database');
            if(info?.rows?.length) renderRows(mergeUniqueRows(rows, info.rows));
          });
          abortIfStaleOrStopped(run);
          rows=rows.concat(cdbRows);
        }
        if(runLowPrecisionAlg){
          setSearchStatus('Running low-precision algebraic relation search last…', .975, 'algebraic search');
          await nextPaint();
          abortIfStaleOrStopped(run);
          let maxH; try{ maxH=BigInt(document.getElementById('algHeight').value.trim() || '1000000000000'); }catch(e){ maxH=1000000000000n; }
          maxH = maxH > 100000000n ? 100000000n : maxH;
          algMaxHeightForFilter=maxH;
          const deg=Math.max(6, Math.min(10, Number(document.getElementById('algDegree').value)||8));
          const autoPrec=Math.max(1, Math.min(17, typedInputPrecision(settings)));
          const slack=Math.max(2, Math.min(5, autoPrec<=10 ? 2 : 3));
          const lpRows=await relationCandidatesAsync(settingsForModule(settings,'algebraic'), deg, autoPrec, maxH, Math.max(settings.moduleLimits?.algebraic || settings.limit,10), slack, info=>{
            abortIfStaleOrStopped(run);
            const frac=Number(info?.frac||0);
            const phase=info?.phase || 'algebraic batch';
            setSearchStatus(`Running low-precision algebraic relation search last · ${phase} ${(frac*100).toFixed(0)}% · ${info?.rows?.length || 0} candidate(s)`, .975 + Math.min(.02, frac*.02), 'algebraic search');
            if(info?.rows?.length) renderRows(mergeUniqueRows(rows, info.rows));
          }, ()=>abortIfStaleOrStopped(run));
          abortIfStaleOrStopped(run);
          rows=rows.concat(lpRows);
        }
        const byErr=(a,b)=>(Number.isFinite(a.err)?a.err:1e9)-(Number.isFinite(b.err)?b.err:1e9);
        const algRanker=(a,b)=>(a.score??1e99)-(b.score??1e99) || (a.degree||99)-(b.degree||99) || Number((a.height||0n)-(b.height||0n)) || byErr(a,b);
        const logRowRE=/^(?:log match|log-combination relation|log\|c\| linear relation):/;
        let allRows=dedupeEquivalentRows(rows, settings);
        if(Number.isFinite(Number(settings.maxRelError))){
          const maxRel=Number(settings.maxRelError);
          allRows=allRows.filter(r=>resultRowRelativeError(r,settings)<=maxRel);
        }
        function groupIndex(r){
          const c=String(r?.candidate||'');
          if(runHighPrecisionAlg){
            if(/algebraic/.test(c)) return 0;
            if(/^constant match:/.test(c) || r?.specialConstant) return 1;
            if(r?.lfuncCategory==='rational') return 2;
            if(r?.lfuncCategory==='quadratic') return 3;
            if(r?.lfuncCategory==='log') return 4;
            if(/^RIES equation:/.test(c)) return 5;
            if(/^low-precision linear combo:/i.test(c) || r?.lowPrecisionLinearCombo) return 6;
            if(logRowRE.test(c)) return 7;
            if(/^Möbius relation:|^Mobius relation:/i.test(c) || r?.mobiusCategory) return 8;
            if(/^hard constant database:/i.test(c) || r?.hardDbCategory) return 9;
            if(/^hypergeometric database:/i.test(c) || r?.hypDataCategory) return 10;
            if(/^constant database:/i.test(c) || r?.constantDbCategory) return 11;
            return 11;
          }
          if(runLowPrecisionAlg){
            if(/^RIES equation:/.test(c)) return 0;
            if(/^low-precision linear combo:/i.test(c) || r?.lowPrecisionLinearCombo) return 1;
            if(/^constant match:/.test(c) || r?.specialConstant) return 2;
            if(logRowRE.test(c)) return 3;
            if(/^Möbius relation:|^Mobius relation:/i.test(c) || r?.mobiusCategory) return 4;
            if(/^hard constant database:/i.test(c) || r?.hardDbCategory) return 5;
            if(/^hypergeometric database:/i.test(c) || r?.hypDataCategory) return 6;
            if(/^constant database:/i.test(c) || r?.constantDbCategory) return 7;
            if(/algebraic/.test(c)) return 6;
            if(r?.lfuncCategory==='rational') return 5;
            if(r?.lfuncCategory==='log') return 7;
            if(r?.lfuncCategory==='quadratic') return 8;
            return 8;
          }
          if(/^RIES equation:/.test(c)) return 0;
          if(/^low-precision linear combo:/i.test(c) || r?.lowPrecisionLinearCombo) return 1;
          if(/^constant match:/.test(c) || r?.specialConstant) return 2;
          if(logRowRE.test(c)) return 3;
          if(/^Möbius relation:|^Mobius relation:/i.test(c) || r?.mobiusCategory) return 4;
          if(/^hard constant database:/i.test(c) || r?.hardDbCategory) return 5;
          if(/^hypergeometric database:/i.test(c) || r?.hypDataCategory) return 6;
          if(/^constant database:/i.test(c) || r?.constantDbCategory) return 7;
          if(/algebraic/.test(c)) return 6;
          if(r?.lfuncCategory==='rational') return 5;
          if(r?.lfuncCategory==='log') return 7;
          if(r?.lfuncCategory==='quadratic') return 8;
          return 8;
        }
        function withinGroupCompare(a,b){
          const ca=String(a?.candidate||''), cb=String(b?.candidate||'');
          if(/algebraic/.test(ca) || /algebraic/.test(cb)) return algRanker(a,b);
          if(a?.lfuncCategory || b?.lfuncCategory) return (a.score??1e99)-(b.score??1e99) || byErr(a,b);
          return byErr(a,b) || resultComplexityScore(a)-resultComplexityScore(b);
        }
        const displayRows=[...allRows].sort((a,b)=>groupIndex(a)-groupIndex(b) || withinGroupCompare(a,b));
        abortIfStaleOrStopped(run);
        renderFinalDefault(allRows, displayRows, settings);
        runCache.full.set(fullCacheKey, allRows.slice());
        const dt=Math.round(performance.now()-t0);
        statusEl.className='notice status-line good';
        statusEl.textContent=`Returned ${allRows.length} result(s) in ${dt} ms. Confidence order is shown by default; use “Original order” to inspect discovery order.`;
        const curLevel=Math.max(1, Math.min(9, Number(document.getElementById('level')?.value || 4)));
        if(curLevel>=9) setContinueState('ries', 'Max RIES level reached', true);
        else setContinueState('ries', `Continue at RIES level ${curLevel+1}`, false);
      }catch(e){
        const msg=String(e && e.message);
        if(msg!=='RIES_STOPPED' && msg!=='RIES_STALE_INPUT') throw e;
        if(msg==='RIES_STOPPED'){
          const kept=mergeUniqueRows(seedRows, rows);
          if(kept.length) renderFinalDefault(kept, kept, settings);
          statusEl.className='notice status-line';
          statusEl.textContent='Stopped. Current decimal results are kept on screen.';
        }
      }finally{
        if(activeSolveRun===run) activeSolveRun=null;
        stopBtn.disabled=true;
        runBtn.disabled=false;
      }
    }
    (function initRIESPage(){
      const required={resultBody,hpPanel,hpContent,numberTools,numberToolsContent,statusEl,paramToggle,parametersPanel,stopBtn,continueBtn,runBtn,targetInput};
      const missing=Object.entries(required).filter(([,el])=>!el).map(([id])=>id);
      const logBasisMissing=!document.getElementById('defaultLogBasis') || !document.getElementById('extraLogBasis');
      if(missing.length || logBasisMissing){
        const msg='RIES UI failed to initialize because required element(s) are missing: '+missing.concat(logBasisMissing?['log basis panels']:[]).join(', ')+'.';
        console.error(msg);
        if(document.body){
          const box=document.createElement('div');
          box.className='notice bad';
          box.style.margin='16px';
          box.textContent=msg+' Please reload this v11.2 build; the page is protected from a blank-screen crash.';
          document.body.prepend(box);
        }
        return;
      }
      paramToggle.addEventListener('click', ()=>{ const open=parametersPanel.hidden; parametersPanel.hidden=!open; paramToggle.setAttribute('aria-expanded', String(open)); paramToggle.textContent = open ? 'Hide parameters' : 'Parameters'; });
      runBtn.addEventListener('click', solve);
      if(sortConfidenceBtn) sortConfidenceBtn.addEventListener('click', ()=>{
        if(!currentResultAllRows.length) return;
        const sorted=confidenceSortedRows(currentResultAllRows, currentResultSettings || readSettings());
        renderRows(sorted,{final:true, allRows:currentResultAllRows, discoveryRows:currentResultDiscoveryRows.length?currentResultDiscoveryRows:currentResultAllRows, settings:currentResultSettings || readSettings(), sorted:true});
      });
      if(sortDiscoveryBtn) sortDiscoveryBtn.addEventListener('click', ()=>{
        const rows=currentResultDiscoveryRows.length ? currentResultDiscoveryRows : currentResultAllRows;
        renderRows(rows,{final:true, allRows:currentResultAllRows, discoveryRows:rows, settings:currentResultSettings || readSettings(), sorted:false});
      });
      stopBtn.addEventListener('click', ()=>{ stopActiveSolve(); });
      continueBtn.addEventListener('click', ()=>{
        const mode=continueBtn.dataset.mode || '';
        if(mode==='shortform'){
          const sel=document.getElementById('shortEffort');
          const next=Math.min(7, (Number(sel.value)||0)+1);
          sel.value=String(next);
          continueBtn.disabled=true;
          pendingContinueSolve=true;
          setSearchStatus(`Continuing deterministic shortform search at effort ${next}…`, .12, 'continuing');
          solve();
        }else if(mode==='ries'){
          const sel=document.getElementById('level');
          const next=Math.min(9, (Number(sel.value)||4)+1);
          sel.value=String(next);
          continueBtn.disabled=true;
          pendingContinueSolve=true;
          setSearchStatus(`Continuing RIES equation search at level ${next}…`, .12, 'continuing');
          solve();
        }
      });
      targetInput.addEventListener('input', ()=>{
        const current=targetInput.value.trim();
        if(current!==lastInputSnapshot){
          lastInputSnapshot=current;
          document.getElementById('shortEffort').value=DEFAULT_SHORT_EFFORT;
          document.getElementById('level').value=DEFAULT_RIES_LEVEL;
          resetSearchFrameworkForInputChange();
          updatePreview(readSettings());
        }
      });
      targetInput.addEventListener('keydown', ev=>{ if(ev.key==='Enter'){ ev.preventDefault(); solve(); } });
      if(numberTools){ numberTools.addEventListener('toggle', ()=>{ if(numberTools.open && window.__lastRIESSettings){ numberToolsContent.innerHTML='<p class="muted">Computing number expansions…</p>'; setTimeout(()=>renderNumberTools(window.__lastRIESSettings),0); } }); }
      fillLogBasis();
      document.querySelectorAll('[data-auto-depth-budget]').forEach(el=>{
        el.addEventListener('input', ()=>{ el.dataset.userEdited='1'; });
      });
      document.getElementById('level')?.addEventListener('change', ()=>syncLevelDefaultBudgets(false));
      syncParameterModuleVisibility();
      syncLevelDefaultBudgets(false);
      parametersPanel.addEventListener('change', ev=>{
        if(ev.target && ev.target.matches('[data-module-toggle]')) syncParameterModuleVisibility();
        try{ updatePreview(readSettings()); }catch(e){}
      });
      lastInputSnapshot=targetInput.value.trim();
      updatePreview(readSettings());

    if(typeof window !== 'undefined'){
      window.readSettings = readSettings;
      window.lfuncRowsAsync = lfuncRowsAsync;
      window.solve = solve;
      window.ensureShortformDbLoaded = ensureShortformDbLoaded;
      window.renderRows = renderRows;
      window.confidenceSortedRows = confidenceSortedRows;
      window.dedupeEquivalentRows = dedupeEquivalentRows;
      window.resultConfidenceScore = resultConfidenceScore;
      window.resultLengthFirstScore = resultLengthFirstScore;
      window.lfuncFormulaLatex = lfuncFormulaLatex;
      window.__RIES_LFUNC_TEST__ = { lfuncEffortConfig, LFUNC_MONOMIALS, lfuncLogConstants };
      window.__RIES_MOBIUS_TEST__ = { mobiusConstants, mobiusRelationRows, mobiusRowsForVariant, mobiusSparseRowsForVariant, shouldRunMobiusRows, mobiusEffort };
      window.__RIES_EQUATION_TEST__ = { generateConstants, generateLHS, equationSearch, exprToLatex, sanitizeLatexForDisplay, latexNormalizeSigns, latexMulScalar, latexPow };
      window.__RIES_LINEAR_COMBO_TEST__ = { lowPrecisionLinearComboRows, lowPrecisionLinearComboBasisConstants, shouldRunLowPrecisionLinearComboRows, lowPrecisionLinearComboRelTol, lowPrecisionLinearComboPairTasks };
      window.__RIES_HARDDB_TEST__ = { hardDbRowsAsync, hardDbShouldRun, hardDbPotentiallyRunnable, hardDbLevelEnabled, hardDbMaxStage, hardDbLoadedChunks, ensureHardDbLoaded, isHardDbReady, hardDbRelTol, hardDbRationalsHeight10, hardDbRationalsHeight, hardDbFormulaLatex, hardDbMakeTargetSpecs, sanitizeLatexForDisplay, resultRowCategory, confidenceSortedRows };
      window.__RIES_HYPDATA_TEST__ = { hypDataRowsAsync, hypDataSearch, hypDataPotentiallyRunnable, ensureHypDataLoaded, isHypDataReady, hypDataLoadedChunks, hypDataMaxStage, hypDataRelTol, hypDataMkLatex, hypDataMkText, hypDataMulLatex, RIES_HYPDATA_ASSET_LEVELS, sanitizeLatexForDisplay, resultRowCategory, confidenceSortedRows };
      window.__RIES_INTSUMDB_TEST__ = { intsumDbRowsAsync, intsumDbSearch, intsumDbPotentiallyRunnable, ensureIntsumDbLoaded, isIntsumDbReady, intsumDbLoadedChunks, intsumDbMaxStage, intsumDbRelTol, intsumDbMulLatex, intsumDbMulText, RIES_INTSUMDB_ASSET_LEVELS, sanitizeLatexForDisplay, resultRowCategory, confidenceSortedRows };
      window.__RIES_CONSTDB_TEST__ = { constantDbRecords, shouldRunConstantDbRows, constantDbRows, constantDbRowsAsync, constDbFindQuadraticRatio, constDbFindPolynomialRatio, constDbFindLinearRelation, constDbPslqLinearRelation, constDbTryRelation_b_1_c_c2, constDbTryRelation_b_1_c_c2_c3, constDbTryRelation_b_1_c_invc, constDbTryRelation_b_1_c_c2_c3_invc, constDbFindAlgebraicRatioLLL, constDbTransformRows, constDbExtraSubsetRows, constDbLogLinearRows, constDbPriorityTransformedPolynomialRows, constDbPriorityRelationRecords, constDbIsPriorityNoiseConstant, constDbRelationUsesTargetNontrivially, constantDbBudgetMs, constDbMaxRelativeError, typedDecimalScaleDigits, typedInputPrecisionForDouble, riesLevelModuleBudgetMs };
      window.__RIES_LOG_TEST__ = { logConstants, logContinueEffort, logContinuationRemovalOrder, logContinuationBasisRows, logRelationRows, logProductString, logProductLatex, logLinearConstantLatex, linearCombinationLatex, directSparseLogRows, resetSearchFrameworkForInputChange, solveRunCache, integerGlobalCache, lfuncProgressCache, typedInputPrecision, typedInputPrecisionDigits, matchToleranceDigits, typedRelativeToleranceNumber, linearRelations };
      window.__RIES_INTEGER_TEST__ = { exactIntegerValueFromDisplay, displayExprMatchesTarget, integerRowFormulaIsValid, integerDatabaseRowsResponsive, integerShortformRowsAsync, staticShortformRows, selectDigitShortforms, exprToLatex, simplifyIntegerExpressionDisplay, simplifyDExprIfBetter, makeDExpr };
      window.__RIES_PRECISION_TEST__ = { typedInputPrecision, typedInputPrecisionDigits, typedInputPrecisionForDouble, matchToleranceDigits, typedRelativeToleranceNumber, linearRelations, logRelationRows, lfuncRowsAsync, specialDecimalConstantRows, parseDecimalComplex, rationalToNumber, numberToolsShouldAppear, currentNumberDescriptor, decimalToBaseString, stageBudgetValueToMs, riesLevelDefaultModuleBudgetMs };
    }
      resultBody.innerHTML = '<tr><td colspan="3">Enter a target and press Solve.</td></tr>';
    })();
