
    const resultBody = document.getElementById('resultBody');
    const statusEl = document.getElementById('status');
    const previewEl = document.getElementById('commandPreview');
    const paramToggle = document.getElementById('paramToggle');
    const parametersPanel = document.getElementById('parametersPanel');
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
      const expr = raw.replaceAll('π','pi').replaceAll('^','**');
      if(/^[0-9a-zA-Z_+\-*/().,\s*]+$/.test(expr)){
        try{
          const v = Function('"use strict"; const pi=Math.PI, e=Math.E, phi=(1+Math.sqrt(5))/2, sqrt=Math.sqrt, sin=Math.sin, cos=Math.cos, tan=Math.tan, log=Math.log, ln=Math.log, exp=Math.exp, abs=Math.abs, pow=Math.pow; return ('+expr+');')();
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
    function decimalPrecision(raw){
      let s=String(raw).trim(); if(!/^[+\-]?\d*(\.\d*)?([eE][+\-]?\d+)?$/.test(s) || !/[0-9]/.test(s)) return 12;
      let [mant, expstr] = s.toLowerCase().split('e'); const exp = expstr ? Number(expstr) : 0;
      const frac = mant.includes('.') ? mant.length - mant.indexOf('.') - 1 : 0;
      return Math.max(0, Math.min(17, frac - exp));
    }
    function readSettings(){
      const only = new Set(document.getElementById('onlySyms').value.trim().split(''));
      const never = new Set(document.getElementById('neverSyms').value.trim().split(''));
      const checked = new Set([...document.querySelectorAll('[data-sym]:checked')].map(x=>x.dataset.sym));
      const digits = new Set(document.getElementById('digits').value.replace(/[^0-9]/g,'').split(''));
      const restrict = document.getElementById('restrictMode').value;
      const tolRaw = document.getElementById('tolerance').value.trim();
      const maxAbs = parseTarget(document.getElementById('maxAbs').value) || 1e9;
      function allowed(sym){
        if(/[0-9]/.test(sym) && !digits.has(sym)) return false;
        if(only.size && !only.has(sym)) return false;
        if(never.has(sym)) return false;
        if(/[pefnrsqlESTC+\-*/^vL]/.test(sym) && !checked.has(sym) && !/[0-9]/.test(sym)) return false;
        if(restrict === 'rational' && 'pefqslESTC^vL'.includes(sym)) return false;
        if(restrict === 'constructible' && 'pelESTC^vL'.includes(sym)) return false;
        return true;
      }
      const raw = document.getElementById('target').value.trim();
      const target = parseTarget(raw);
      const normalizedRaw = canonicalTargetString(raw, target);
      return { raw, normalizedRaw, target, level: Number(document.getElementById('level').value), shortEffort: Number(document.getElementById('shortEffort')?.value || 0), limit: Math.max(1, Math.min(50, Number(document.getElementById('limit').value)||5)), restrict, allowed, tol: tolRaw ? Math.abs(parseTarget(tolRaw)) : Infinity, maxAbs, only: [...only].join(''), never: [...never].join(''), doEq:document.getElementById('doEq').checked, doExpr:false, doAlg:document.getElementById('doAlg').checked, doLog:document.getElementById('doLog').checked };
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
          rows.push({candidate, value:`x = ${fmtValue(root)}`, err, c:f.c+rhs.c});
        }
      }
      return rows.sort((a,b)=>a.err-b.err||a.c-b.c||a.candidate.length-b.candidate.length).slice(0,settings.limit);
    }
    function absBig(x){ return x < 0n ? -x : x; }
    function gcdBig(a,b){ a=absBig(a); b=absBig(b); while(b){ const t=a%b; a=b; b=t; } return a; }
    function roundDiv(n,d){ if(d<0n){ n=-n; d=-d; } return n>=0n ? (2n*n + d)/(2n*d) : -((2n*(-n)+d)/(2n*d)); }
    function decimalScaledPowers(raw, deg, prec){
      let s=String(raw).trim(); if(!/^[+\-]?\d*(\.\d*)?([eE][+\-]?\d+)?$/.test(s) || !/[0-9]/.test(s)) return null;
      let sign=1n; if(s[0]==='-'){ sign=-1n; s=s.slice(1); } else if(s[0]==='+') s=s.slice(1);
      let [mant, expstr] = s.toLowerCase().split('e'); let exp = expstr ? Number(expstr) : 0; let frac = 0;
      if(mant.includes('.')){ frac = mant.length - mant.indexOf('.') - 1; mant = mant.replace('.',''); }
      mant = mant.replace(/^0+(?=\d)/,'') || '0'; let p = BigInt(mant) * sign; let den = 10n ** BigInt(frac);
      if(exp > 0) p *= 10n ** BigInt(exp); if(exp < 0) den *= 10n ** BigInt(-exp);
      const scale = 10n ** BigInt(Math.max(0,prec)); const scaled=[]; let pn=1n, dn=1n;
      for(let i=0;i<=deg;i++){ scaled.push(roundDiv(scale*pn,dn)); pn*=p; dn*=den; }
      return {scaled, scale, source:'decimal'};
    }
    function doubleScaledPowers(value, deg, prec){
      if(!Number.isFinite(value)) return null; const p = Math.min(Math.max(0,prec), 15); const scale = 10n ** BigInt(p); const scaled=[];
      for(let i=0;i<=deg;i++){ const v = Math.pow(value,i); if(!Number.isFinite(v) || Math.abs(v)>1e120) return null; scaled.push(BigInt(Math.round(Number(scale)*v))); }
      return {scaled, scale, source:'double'};
    }
    function dotNum(a,b){ let s=0; for(let i=0;i<a.length;i++) s += Number(a[i]) * Number(b[i]); return s; }
    function lllReduce(rows, delta=0.75){
      rows = rows.map(r=>r.slice()); const n=rows.length; let B=[], mu=[], bstar=[];
      function recompute(){ B=[]; mu=Array.from({length:n},()=>Array(n).fill(0)); bstar=[]; for(let i=0;i<n;i++){ let v=rows[i].map(Number); for(let j=0;j<i;j++){ const m=B[j] ? dotNum(rows[i], bstar[j]) / B[j] : 0; mu[i][j]=Number.isFinite(m)?m:0; for(let k=0;k<v.length;k++) v[k] -= mu[i][j]*bstar[j][k]; } bstar[i]=v; B[i]=v.reduce((s,x)=>s+x*x,0); if(!Number.isFinite(B[i])) B[i]=Number.MAX_VALUE/16; } }
      recompute(); let k=1, iter=0;
      while(k<n && iter++<12000){
        for(let j=k-1;j>=0;j--){ const q=Math.round(mu[k][j]); if(q!==0 && Number.isFinite(q)){ const Q=BigInt(q); for(let c=0;c<rows[k].length;c++) rows[k][c] -= Q*rows[j][c]; recompute(); } }
        if(B[k] >= (delta - mu[k][k-1]*mu[k][k-1]) * B[k-1]) k++;
        else { const tmp=rows[k]; rows[k]=rows[k-1]; rows[k-1]=tmp; recompute(); k=Math.max(k-1,1); }
      }
      return rows;
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
    function relationCandidates(raw, maxDegree, prec, maxHeight, limit, slack){
      const rows=[]; const seen=new Set(); const val=parseTarget(raw);
      for(let deg=1; deg<=maxDegree; deg++){
        const data = decimalScaledPowers(raw, deg, prec) || doubleScaledPowers(val, deg, prec);
        if(!data) continue;
        const basis=[]; for(let i=0;i<=deg;i++){ const row=Array(deg+2).fill(0n); row[i]=1n; row[deg+1]=data.scaled[i]; basis.push(row); }
        const red=lllReduce(basis);
        const maxResidual = 10n ** BigInt(Math.max(0, Math.ceil(prec/2) + slack));
        for(const r of red){
          const coeff=normalizeCoeffs(r.slice(0,deg+1)); const pd=polyDegree(coeff); if(pd===0) continue;
          const h=coeffHeight(coeff); if(h===0n || h>maxHeight) continue;
          let residual=0n; for(let i=0;i<coeff.length;i++) residual += coeff[i]*(data.scaled[i] || 0n); residual=absBig(residual);
          if(residual > maxResidual) continue;
          const key=coeff.join(','); if(seen.has(key)) continue; seen.add(key);
          const root=refinePolyRoot(coeff, val); const err=Math.abs(root-val); const fallback=Math.abs(polyEvalNum(coeff,val));
          rows.push({candidate:`algebraic: ${polyString(coeff)}`, value:Number.isFinite(root)?`x = ${fmtValue(root)}`:`P(x) = ${fmtValue(fallback)}`, err:Number.isFinite(err)?err:fallback, degree:pd, height:h, residual});
        }
      }
      return rows.sort((a,b)=>a.degree-b.degree || Number(a.height-b.height) || a.err-b.err).slice(0,limit);
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
      {id:'loglog5', label:'log(log 5)', value:Math.log(Math.log(5)), default:false, kind:'log', product:'log(5)'},
      {id:'log7', label:'log(7)', value:Math.log(7), default:false, kind:'log', product:'7'},
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
    function linearRelations(values, labels, prec, maxHeight, limit, slack){
      const scaleNum = Math.pow(10, Math.min(17, Math.max(4, prec))); const scale = BigInt(Math.round(scaleNum));
      const basis=[]; for(let i=0;i<values.length;i++){ const row=Array(values.length+1).fill(0n); row[i]=1n; row[values.length]=BigInt(Math.round(values[i]*scaleNum)); basis.push(row); }
      const red=lllReduce(basis); const rows=[]; const seen=new Set(); const maxResidual=10n ** BigInt(Math.max(0, slack));
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
    function logProductString(rel, consts){
      const den=rel.coeff[0]; const parts=[];
      for(let i=1;i<rel.coeff.length;i++){
        const factor = logFactor(consts[i-1], -rel.coeff[i], den);
        if(factor) parts.push(factor);
      }
      return parts.join(' * ') || '1';
    }
    function logRelationRows(target, settings){
      if(!Number.isFinite(target) || target===0) return [];
      let maxH; try{ maxH=BigInt(document.getElementById('logHeight').value.trim() || '400'); }catch(e){ maxH=400n; }
      const precRaw=document.getElementById('logPrecision').value.trim();
      const autoPrec=decimalPrecision(settings.normalizedRaw);
      const prec=Math.max(4, Math.min(17, precRaw==='' ? autoPrec : Number(precRaw)||autoPrec));
      const slack=Math.max(0, Math.min(12, Number(document.getElementById('logSlack').value)||2));
      const consts=selectedLogConstants(); if(!consts.length) return [];
      const y=Math.log(Math.abs(target)); const values=[y, ...consts.map(c=>c.value)]; const labels=['log|x|', ...consts.map(c=>c.label)];
      const rels=linearRelations(values, labels, prec, maxH, settings.limit, slack);
      const left = target < 0 ? '−x' : 'x';
      return rels.map(rel=>{
        const product=logProductString(rel, consts);
        const productValue=Math.exp(rel.rhs);
        return {candidate:`log match: ${left} ≈ ${product}`, value:`${left} = ${fmtValue(Math.abs(target))}; product = ${fmtValue(productValue)}`, err:rel.err};
      });
    }

    function integerInputBig(raw){
      const s=String(raw||'').trim();
      if(!/^[+-]?\d+$/.test(s)) return null;
      try { return BigInt(s); } catch(e){ return null; }
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
    function factorBigIntWithin(n, ms=10000){
      const deadline=performance.now()+ms;
      const factors=[];
      let sign='';
      if(n<0n){ sign='-1'; n=-n; }
      function rec(x){
        if(performance.now()>deadline) return false;
        if(x===1n) return true;
        for(const p of [2n,3n,5n,7n,11n,13n,17n,19n,23n,29n,31n,37n,41n,43n,47n]){
          if(x===p){ factors.push(p); return true; }
          if(x%p===0n){ factors.push(p); return rec(x/p); }
        }
        if(isProbablePrime(x)){ factors.push(x); return true; }
        const d=pollardRho(x, deadline);
        if(!d) return false;
        return rec(d) && rec(x/d);
      }
      const complete=rec(n);
      factors.sort((a,b)=>a<b?-1:a>b?1:0);
      return {complete, sign, factors, ms:Math.round(10000-(deadline-performance.now()))};
    }
    function factorRows(settings){
      const n=integerInputBig(settings.raw);
      if(n===null) return [];
      if(n===0n) return [{candidate:'integer factorization', value:'0 has no finite prime factorization.', err:0}];
      if(absBig(n)===1n) return [{candidate:'integer factorization', value:`${n.toString()} is a unit.`, err:0}];
      const out=factorBigIntWithin(n, 10000);
      if(!out.factors.length) return [{candidate:'integer factorization', value:`No nontrivial factor found within 10 s for ${n.toString()}.`, err:0}];
      const counts=new Map();
      for(const f of out.factors) counts.set(f.toString(), (counts.get(f.toString())||0)+1);
      const parts=[]; if(out.sign) parts.push(out.sign);
      for(const [p,e] of counts) parts.push(e===1?p:`${p}^${e}`);
      return [{candidate: out.complete ? 'integer factorization' : 'integer factorization (partial, 10 s cutoff)', value:`${n.toString()} = ${parts.join(' × ')}${out.complete?'': ' × …'} (${out.ms} ms)`, err:0}];
    }
    function decimalDigitCountBig(n){ return absBig(n).toString().length; }
    function shortPrettyValue(v){ const s=v.toString(); return s.length > 34 ? s.slice(0,18)+'…'+s.slice(-12) : s; }
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
        .replace(/\(([^()]+)\)/g, (m,x)=>/^[\w!^]+$/.test(x)?x:m);
    }
    function shortRank(e){
      const s=e.s || '';
      const digits=e.digits ?? digitCountExpr(s);
      const ops=e.ops||0;
      const depth=e.depth||0;
      const nums=(s.match(/\d+/g)||[]).map(x=>Number(x)).filter(Number.isFinite);
      let r=digits*1000000 + ops*9000 + depth*900 + s.length*18;
      r += nums.reduce((a,b)=>a+Math.min(160,b),0)*1.6;
      if(/binom\(/.test(s)) r -= 23000;
      if(/round\(/.test(s)) r -= 19000;
      if(/floor\(|ceil\(/.test(s)) r -= 12000;
      if(/!/.test(s)) r -= 16000;
      if(/\^/.test(s)) r -= 14000;
      if(/\+1$/.test(s)||/-1$/.test(s)) r -= 3800;
      if(/[+\-][2-9]$/.test(s)) r -= 1600;
      if(/\d{4,}/.test(s)) r += 6000;
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
      if(op==='/' || op==='-') return /[+\-]/.test(s) ? `(${s})` : s;
      return s;
    }
    function combineD(a,op,b,v,kind='compound'){
      if((op==='+' || op==='*') && a.v!==undefined && b.v!==undefined && a.v<b.v){ const t=a; a=b; b=t; }
      let s;
      if(op==='+') s=`${shortOperandD(a,'+')}+${shortOperandD(b,'+')}`;
      else if(op==='-') s=`${shortOperandD(a,'-')}-${shortOperandD(b,'-')}`;
      else if(op==='*') s=`${shortOperandD(a,'*')}·${shortOperandD(b,'*')}`;
      else if(op==='/') s=`${shortOperandD(a,'/')}/${shortOperandD(b,'/')}`;
      else s=`${op}(${a.s},${b.s})`;
      return makeDExpr(v, s, kind, (a.ops||0)+(b.ops||0)+1, Math.max(a.depth||0,b.depth||0)+1);
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
      effort=Math.max(0, Math.min(6, Number(effort)||0));
      const td=decimalDigitCountBig(target);
      const ceiling=[6,7,8,9,10,11,12][effort];
      return {
        effort,
        timeMs: Math.min(64000, 1000*Math.pow(2, effort)),
        maxDigits: Math.max(2, Math.min(td, ceiling)),
        literalCap: [420,900,1800,4000,9000,18000,42000][effort],
        argCap: [140,220,340,520,820,1300,2100][effort],
        maxFactN: [36,48,64,84,110,145,190][effort],
        maxPowBase: [65,95,140,210,340,580,900][effort],
        maxPowExp: [90,150,230,360,560,840,1300][effort],
        maxBinomN: [140,210,320,480,720,1050,1500][effort],
        maxBinomK: [24,34,46,62,82,108,140][effort],
        pairProbe: [2600,5200,9200,16000,28000,46000,74000][effort],
        dbSoftLimit: [16000,32000,62000,110000,190000,300000,480000][effort],
        denomProbe: [900,1700,3200,6000,10500,17000,26000][effort],
        residualProbe: [70,130,220,360,600,920,1400][effort],
        reverseProbe: [500,1000,1900,3500,6200,10000,15000][effort],
        boundDigits: Math.min(120, td + [16,20,26,32,42,56,70][effort])
      };
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
      function add(e){ if(performance.now()>deadline || !e || e.v>bound) return false; return addBest(byValue,e,cfg); }
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
          if(performance.now()>deadline) break;
          if(a.v>=2n && a.v<=BigInt(cfg.maxFactN)){
            const v=powFactorialBig(a.v, bound);
            if(v!==null){ const e=makeDExpr(v, `${shortOperandD(a,'^')}!`, 'factorial', (a.ops||0)+1, (a.depth||0)+1); if(add(e)) changed=true; addArg(e); }
          }
        }
        const nowArgs=[...argMap.values()].sort(cmpExpr).slice(0, Math.min(argMap.size, cfg.reverseProbe));
        const baseSources=nowArgs.filter(e=>e.v>=2n && e.v<=BigInt(cfg.maxPowBase));
        const expSources=nowArgs.filter(e=>e.v>=2n && e.v<=BigInt(cfg.maxPowExp));
        for(const a of baseSources){
          if(performance.now()>deadline) break;
          for(const b of expSources){
            if(performance.now()>deadline) break;
            if(a.digits+b.digits>cfg.maxDigits) continue;
            if(b.v>1500n || (a.v>100n && b.v>260n) || (a.v>30n && b.v>700n)) continue;
            const v=powBigCapped(a.v,b.v,bound);
            if(v!==null){ const e=makeDExpr(v, `${shortOperandD(a,'^')}^${shortOperandD(b,'^')}`, 'power', (a.ops||0)+(b.ops||0)+1, Math.max(a.depth||0,b.depth||0)+1); if(add(e)) changed=true; addArg(e); }
          }
        }
        const nSources=nowArgs.filter(e=>e.v>=2n && e.v<=BigInt(cfg.maxBinomN));
        const kSources=nowArgs.filter(e=>e.v>=0n && e.v<=BigInt(cfg.maxBinomK));
        for(const n of nSources){
          if(performance.now()>deadline) break;
          for(const k of kSources){
            if(performance.now()>deadline) break;
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
      expr.s=normalizeShortDisplay(expr.s); expr.digits=digitCountExpr(expr.s); expr.rank=shortRank(expr);
      const targetDigits=decimalDigitCountBig(target);
      if(expr.digits>cfg.maxDigits) return;
      const special=/round\(|floor\(|ceil\(|binom\(|!|\^|·|\//.test(expr.s);
      if(!(expr.digits<targetDigits || (targetDigits<=3 && expr.digits<=targetDigits && special))) return;
      const key=expr.s;
      if(seen.has(key)) return;
      seen.add(key);
      rows.push({candidate:`${label}: ${expr.s}`, value:`exact = ${shortPrettyValue(target)}`, err:0, beauty:expr.rank, feature:exprFeature(expr.s), digits:expr.digits, ops:expr.ops});
    }
    function directAndReverseSearch(rows,seen,target,db,cfg,deadline){
      const direct=bestValueExpr(db,target);
      if(direct) addDigitCandidate(rows,seen,direct,target,cfg,'direct exact');
      const residuals=residualPool(db,cfg,target);
      for(const r of residuals){
        if(performance.now()>deadline) return;
        if(target>=r.v){ const a=bestValueExpr(db,target-r.v); if(a) addDigitCandidate(rows,seen,combineD(a,'+',r,target,'sum'),target,cfg,'offset exact'); }
        const a2=bestValueExpr(db,target+r.v); if(a2) addDigitCandidate(rows,seen,combineD(a2,'-',r,target,'offset exact'),target,cfg,'offset exact');
      }
      const divisors=db.atoms.filter(e=>e.v>1n && e.digits<=cfg.maxDigits-1).slice(0,cfg.reverseProbe);
      for(const d of divisors){
        if(performance.now()>deadline) return;
        if(target%d.v===0n){ const q=bestValueExpr(db,target/d.v); if(q) addDigitCandidate(rows,seen,combineD(q,'*',d,target,'product'),target,cfg,'product exact'); }
        // reverse with small offsets: target = q*d +/- r
        for(const r of residuals.slice(0, Math.min(90,cfg.residualProbe))){
          if(performance.now()>deadline) return;
          if(target>=r.v && (target-r.v)%d.v===0n){ const q=bestValueExpr(db,(target-r.v)/d.v); if(q) addDigitCandidate(rows,seen,combineD(combineD(q,'*',d,target-r.v,'product'),'+',r,target,'compound'),target,cfg,'reverse exact'); }
          if((target+r.v)%d.v===0n){ const q=bestValueExpr(db,(target+r.v)/d.v); if(q) addDigitCandidate(rows,seen,combineD(combineD(q,'*',d,target+r.v,'product'),'-',r,target,'compound'),target,cfg,'reverse exact'); }
        }
      }
    }
    function ratioSearch(rows,seen,target,db,cfg,deadline){
      const denoms=db.atoms.filter(e=>e.v>1n && e.digits<=cfg.maxDigits-1).slice(0,cfg.denomProbe);
      const residuals=residualPool(db,cfg,target).slice(0,Math.min(160,cfg.residualProbe));
      function wrapRatio(numer, denom, n, mode){
        let core=`${shortOperandD(numer,'/')}/${shortOperandD(denom,'/')}`;
        let e;
        if(mode==='exact') e=makeDExpr(n, core, 'ratio', numer.ops+denom.ops+1, Math.max(numer.depth,denom.depth)+1);
        else e=makeDExpr(n, `${mode}(${core})`, `${mode}-ratio`, numer.ops+denom.ops+2, Math.max(numer.depth,denom.depth)+2);
        return e;
      }
      for(const d of denoms){
        if(performance.now()>deadline) return;
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
          if(performance.now()>deadline) return;
          const lo=lo0<0n?0n:lo0; const hi=hi0<lo?lo:hi0;
          const near=nearValueExprs(db,lo,hi,8);
          for(const n of near){
            const ok = mode==='round' ? roundDiv(n.v,d.v)===target : (mode==='floor' ? n.v/d.v===target : (n.v+d.v-1n)/d.v===target);
            if(ok) addDigitCandidate(rows,seen,wrapRatio(n,d,target,mode),target,cfg,'ratio exact');
          }
        }
        // one small residual on numerator: round((A +/- r)/d)
        for(const r of residuals){
          if(performance.now()>deadline) return;
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
      const bp=powBigCapped(base,p,null); const nq=powBigCapped(n,q,null);
      if(bp<nq) return -1; if(bp>nq) return 1; return 0;
    }
    function verifyRoundRationalPower(base,p,q,n){
      if(n<0n) return false;
      base=BigInt(base); p=BigInt(p); q=BigInt(q); n=BigInt(n);
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
      const bases=db.argSources.filter(e=>e.v>=2n && e.v<=90n).sort(cmpExpr).slice(0,90);
      const qSources=db.argSources.filter(e=>e.v>=2n && e.v<=160n).sort(cmpExpr).slice(0,180);
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
        if(performance.now()>deadline) return;
        const logb=Math.log(Number(b.v)); if(!Number.isFinite(logb)||logb<=0) continue;
        for(const r of residuals){
          if(performance.now()>deadline) return;
          const variants=[[target,0]];
          if(r.v>0n){ if(target>r.v) variants.push([target-r.v,1]); variants.push([target+r.v,-1]); }
          for(const [tv,sign] of variants){
            if(tv<=0n) continue;
            const lv=Math.log(Number(tv)); if(!Number.isFinite(lv)) continue;
            const want=lv/logb;
            for(const q of qSources){
              if(performance.now()>deadline) return;
              const pGuess=want*Number(q.v);
              if(!Number.isFinite(pGuess) || pGuess<2) continue;
              for(const p of argNear(pGuess)){
                if(b.digits+p.digits+q.digits+(sign===0?0:r.digits)>cfg.maxDigits) continue;
                const coreText=`${shortOperandD(b,'^')}^(${p.s}/${q.s})`;
                let core=null;
                if(verifyRoundRationalPower(b.v,p.v,q.v,tv)) core=makeDExpr(tv,`round(${coreText})`,'rounded-power',(b.ops||0)+(p.ops||0)+(q.ops||0)+3,Math.max(b.depth||0,p.depth||0,q.depth||0)+2);
                else if(verifyFloorRationalPower(b.v,p.v,q.v,tv)) core=makeDExpr(tv,`floor(${coreText})`,'floored-power',(b.ops||0)+(p.ops||0)+(q.ops||0)+3,Math.max(b.depth||0,p.depth||0,q.depth||0)+2);
                else if(verifyCeilRationalPower(b.v,p.v,q.v,tv)) core=makeDExpr(tv,`ceil(${coreText})`,'ceiled-power',(b.ops||0)+(p.ops||0)+(q.ops||0)+3,Math.max(b.depth||0,p.depth||0,q.depth||0)+2);
                if(core) addDigitCandidate(rows,seen,buildWithResidual(core,r,sign),target,cfg,'power exact');
              }
            }
          }
        }
      }
    }
    function selectDigitShortforms(rows, limit=5){
      const sorted=rows.sort((a,b)=>(a.digits-b.digits)||(a.beauty-b.beauty)||(a.ops-b.ops)||a.candidate.length-b.candidate.length);
      const picked=[]; const used=new Set();
      for(const r of sorted){ if(picked.length>=limit) break; if(!used.has(r.feature)){ picked.push(r); used.add(r.feature); } }
      for(const r of sorted){ if(picked.length>=limit) break; if(!picked.includes(r)) picked.push(r); }
      return picked.slice(0,limit);
    }
    function integerShortformRows(settings){
      const rawN=integerInputBig(settings.raw);
      if(rawN===null || rawN===0n) return [];
      const startTime=performance.now();
      const effort=Math.max(0, Math.min(6, Number(settings.shortEffort)||0));
      const deadline=startTime + Math.min(64000, 1000*Math.pow(2, effort));
      const sign=rawN<0n ? -1n : 1n;
      const target=absBig(rawN);
      const cfg=digitSearchConfig(effort,target);
      const db=buildDigitSearchDB(target, settings, cfg, Math.min(deadline, startTime+cfg.timeMs*0.52));
      const rows=[]; const seen=new Set();
      const t1=performance.now();
      const rem=Math.max(100, deadline-t1);
      directAndReverseSearch(rows,seen,target,db,cfg,Math.min(deadline,t1+rem*0.35));
      ratioSearch(rows,seen,target,db,cfg,Math.min(deadline,t1+rem*0.78));
      rationalPowerSearch(rows,seen,target,db,cfg,Math.min(deadline,t1+rem*0.92));
      directAndReverseSearch(rows,seen,target,db,cfg,deadline);
      let selected=selectDigitShortforms(rows, Math.max(1, Math.min(5, settings.limit||5)));
      if(sign<0n){
        selected=selected.map(r=>({...r, candidate:r.candidate.replace(/: /, ': -(')+')', value:r.value.replace('exact = ', 'exact = -')}));
      }
      settings._shortformMs=Math.round(performance.now()-startTime);
      settings._shortformMaxDigits=cfg.maxDigits;
      settings._shortformEffort=effort;
      settings._shortformDbSize=db.byValue.size;
      return selected;
    }
    function updatePreview(settings){ if(!previewEl) return; const parts=['ries']; if(settings.only) parts.push(`-S${settings.only}`); if(settings.never) parts.push(`-N${settings.never}`); if(settings.restrict==='rational') parts.push('-r'); if(settings.restrict==='integer') parts.push('-i'); parts.push(`-l${settings.level}`); parts.push(String(settings.raw)); previewEl.textContent = 'Approximate CLI analogue: ' + parts.join(' '); }
    function renderRows(rows){ resultBody.innerHTML = rows.map(r=>`<tr><td><code>${r.candidate}</code></td><td>${r.value}</td><td>${fmtErr(r.err)}</td></tr>`).join('') || '<tr><td colspan="3">No results under the current settings.</td></tr>'; }
    function solve(){
      const settings=readSettings(); updatePreview(settings);
      if(!Number.isFinite(settings.target)){ statusEl.textContent='Please enter a valid target number.'; statusEl.className='notice status-line bad'; return; }
      statusEl.textContent='Solving…'; statusEl.className='notice status-line';
      setTimeout(()=>{
        const t0=performance.now();
        const integerValue = integerInputBig(settings.raw);
        const isInteger = integerValue !== null;
        let rows = [];
        let constants=[];
        if(isInteger){
          rows = integerShortformRows(settings);
          renderRows(rows);
          const dtShort=Math.round(performance.now()-t0);
          statusEl.className='notice status-line';
          statusEl.textContent=rows.length ? `Returned ${rows.length} digit-min shortform candidate(s) in ${dtShort} ms at effort ${settings._shortformEffort ?? settings.shortEffort} (max ${settings._shortformMaxDigits ?? '?'} written digits, ${settings._shortformDbSize ?? '?'} cached exact forms). Integer algebraic search is skipped; factorization is continuing…` : `No meaningful digit-saving shortform found in ${dtShort} ms at effort ${settings.shortEffort} after building ${settings._shortformDbSize ?? '?'} cached exact forms. Integer algebraic search is skipped; factorization is continuing…`;
          setTimeout(()=>{
            rows=rows.concat(factorRows(settings));
            renderRows(rows);
            const dt=Math.round(performance.now()-t0);
            statusEl.className='notice status-line good';
            statusEl.textContent=`Returned ${rows.length} integer result(s) in ${dt} ms. Shortform uses digit-minimizing exact search; algebraic-number candidates were skipped for this integer input.`;
          }, 20);
          return;
        }
        setTimeout(()=>{
          if(settings.doEq){ constants=generateConstants(settings); rows=rows.concat(equationSearch(constants, settings)); }
          if(settings.doAlg){ let maxH; try{ maxH=BigInt(document.getElementById('algHeight').value.trim() || '1000000'); }catch(e){ maxH=1000000n; } const deg=Math.max(1, Math.min(10, Number(document.getElementById('algDegree').value)||6)); const precRaw=document.getElementById('algPrecision').value.trim(); const prec=precRaw==='' ? decimalPrecision(settings.normalizedRaw) : Math.max(0, Math.min(17, Number(precRaw)||0)); const slack=Math.max(0, Math.min(30, Number(document.getElementById('algResidualPower').value)||0)); rows=rows.concat(relationCandidates(settings.normalizedRaw, deg, prec, maxH, settings.limit, slack)); }
          if(settings.doLog) rows=rows.concat(logRelationRows(settings.target, settings));
          renderRows(rows); const dt=Math.round(performance.now()-t0); statusEl.className='notice status-line good'; statusEl.textContent=`Returned ${rows.length} result(s) in ${dt} ms.`;
        }, 0);
      },20);
    }
    paramToggle.addEventListener('click', ()=>{ const open=parametersPanel.hidden; parametersPanel.hidden=!open; paramToggle.setAttribute('aria-expanded', String(open)); paramToggle.textContent = open ? 'Hide parameters' : 'Parameters'; });
    document.getElementById('runBtn').addEventListener('click', solve);
    document.getElementById('target').addEventListener('keydown', ev=>{ if(ev.key==='Enter'){ ev.preventDefault(); solve(); } });
    document.getElementById('exampleBtn').addEventListener('click', ()=>{ document.getElementById('target').value='2.5063'; solve(); });
    document.getElementById('sqrt2Btn').addEventListener('click', ()=>{ document.getElementById('target').value='1.4142135623730950488'; solve(); });
    document.getElementById('plasticBtn').addEventListener('click', ()=>{ document.getElementById('target').value='1.32471795724474602596'; solve(); });
    document.getElementById('logExampleBtn').addEventListener('click', ()=>{ document.getElementById('target').value='2*pi^3/5'; solve(); });
    fillLogBasis();
    updatePreview(readSettings());
    resultBody.innerHTML = '<tr><td colspan="3">Enter a target and press Solve.</td></tr>';
  