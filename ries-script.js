
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
      return { raw, normalizedRaw, target, level: Number(document.getElementById('level').value), limit: Math.max(1, Math.min(50, Number(document.getElementById('limit').value)||5)), restrict, allowed, tol: tolRaw ? Math.abs(parseTarget(tolRaw)) : Infinity, maxAbs, only: [...only].join(''), never: [...never].join(''), doEq:document.getElementById('doEq').checked, doExpr:false, doAlg:document.getElementById('doAlg').checked, doLog:document.getElementById('doLog').checked };
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
    function bitLengthBig(n){ n=absBig(n); return n===0n ? 0 : n.toString(2).length; }
    function decimalDigitCountBig(n){ return absBig(n).toString().length; }
    function shortPrettyValue(v){ const s=v.toString(); return s.length > 30 ? s.slice(0,16)+'…'+s.slice(-10) : s; }
    function powBigLimited(a,e,limit){
      a=BigInt(a); e=BigInt(e); let r=1n;
      while(e>0n){
        if(e&1n){ r*=a; if(limit && r>limit) return null; }
        e >>= 1n;
        if(e){ a*=a; if(limit && a>limit && e>1n) a=limit+1n; }
      }
      return r;
    }
    function binomBigLimited(n,k,limit){
      n=BigInt(n); k=BigInt(k);
      if(k<0n || n<0n || k>n) return null;
      if(k>n-k) k=n-k;
      let r=1n;
      for(let i=1n;i<=k;i++){
        r = (r * (n-k+i)) / i;
        if(limit && r>limit) return null;
      }
      return r;
    }
    function exprFeature(s){
      if(s.includes('round(')) return 'round';
      if(s.includes('binom(')) return 'binom';
      if(s.includes('!')) return 'factorial';
      if(s.includes('^')) return 'power';
      if(s.includes('·')) return 'product';
      if(s.includes('+') || s.includes('-')) return 'offset';
      return 'simple';
    }
    function numberTokenPenalty(s){
      const tokens=String(s).match(/\d+/g) || [];
      let cost=0;
      for(const t of tokens){
        const n=Number(t);
        if(Number.isFinite(n)){
          if(n<=12) cost += t.length*3.0;
          else if(n<=60) cost += t.length*5.2;
          else if(n<=160) cost += t.length*8.2;
          else cost += t.length*18.0;
        } else cost += t.length*22.0;
      }
      return cost;
    }
    function expressionBeauty(s, ops=0, depth=0, kind='compound'){
      let score = numberTokenPenalty(s) + s.length*0.34 + ops*7.0 + depth*3.0;
      if(kind==='literal' && /^\d+$/.test(s) && BigInt(s)>20n) score += decimalDigitCountBig(BigInt(s))*36 + 120;
      if(s.includes('binom(')) score += 11;
      if(s.includes('round(')) score += 10;
      if(s.includes('floor(') || s.includes('ceil(')) score += 13;
      if(s.includes('!')) score += 5;
      if(s.includes('^')) score += 4;
      if(/\+1$/.test(s) || /-1$/.test(s)) score -= 7;
      if(/\+[23]$/.test(s) || /-[23]$/.test(s)) score -= 4;
      if(/binom\(\d+!,\d+!\)/.test(s)) score -= 10;
      if(/round\([^)]*!/.test(s)) score -= 5;
      if(kind==='binary-sum') score += 22;
      return score;
    }
    function makeExpr(v,s,ops=0,depth=0,kind='atom',extra=0){
      const e={v:BigInt(v), s, ops, depth, kind};
      e.manualCost=expressionBeauty(s,ops,depth,kind)+extra;
      return e;
    }
    function shortExprCost(expr){ return Number.isFinite(expr.manualCost) ? expr.manualCost : expressionBeauty(expr.s, expr.ops||0, expr.depth||0, expr.kind||'compound'); }
    function shortResultScore(row){ return row.beauty; }
    function shortOperand(e, op){
      if(!e || !e.s) return '';
      if(op==='-' && /[+\-]/.test(e.s)) return `(${e.s})`;
      if(op==='/' && /[+\-·/]/.test(e.s)) return `(${e.s})`;
      if(op==='*' && /[+\-]/.test(e.s)) return `(${e.s})`;
      return e.s;
    }
    function shortCombineExpr(a, op, b, v){
      if(op==='+' && b.v!==undefined && a.v!==undefined && b.v>a.v){ const t=a; a=b; b=t; }
      if(op==='*' && b.v!==undefined && a.v!==undefined && b.v<a.v){ const t=a; a=b; b=t; }
      const as=shortOperand(a,op), bs=shortOperand(b,op);
      let s;
      if(op==='+') s=`${as}+${b.s}`;
      else if(op==='-') s=`${as}-${shortOperand(b,'-')}`;
      else if(op==='*') s=`${as}·${bs}`;
      else if(op==='/') s=`${as}/${bs}`;
      else s=`${op}(${a.s},${b.s})`;
      return makeExpr(v, s, (a.ops||0)+(b.ops||0)+1, Math.max(a.depth||0,b.depth||0)+1, 'compound');
    }
    function verifyRoundedPowerCandidate(base,p,q,n){
      if(n<=0n || q<=0n || p<0n) return false;
      if(q>40n || p>420n) return false;
      const midPow=powBigLimited(BigInt(base), p, null);
      if(midPow===null) return false;
      const mid=(2n**q)*midPow;
      const left=(2n*n-1n)**q;
      const right=(2n*n+1n)**q;
      return left <= mid && mid < right;
    }
    function compareRationalPowerToInteger(base,p,q,n){
      if(n<=0n || q<=0n || p<0n || q>40n || p>420n) return 0;
      const left=powBigLimited(BigInt(base), p, null);
      const right=BigInt(n)**q;
      if(left<right) return -1;
      if(left>right) return 1;
      return 0;
    }
    function exactRoundRationalPower(base,p,q){
      if(q<=0n || p<0n || q>40n || p>420n) return null;
      const exp=Number(p)/Number(q);
      if(!Number.isFinite(exp) || exp<0 || exp>90) return null;
      const approx=Math.pow(Number(base), exp);
      if(!Number.isFinite(approx) || approx<0.5 || approx>1e24) return null;
      const center=BigInt(Math.round(approx));
      const log2=Math.log2(Math.max(1, approx));
      const spacing=Math.pow(2, Math.max(0, Math.floor(log2)-48));
      const radius=Math.min(4096, Math.max(64, Math.ceil(spacing*4)));
      for(let off=0; off<=radius; off++){
        for(const sign of off===0 ? [0] : [-1,1]){
          const n=center + BigInt(sign*off);
          if(verifyRoundedPowerCandidate(BigInt(base),p,q,n)) return n;
        }
      }
      return null;
    }
    function addBest(map, expr, bound=null){
      if(!expr || expr.v<0n) return;
      if(bound && expr.v>bound) return;
      const key=expr.v.toString();
      const old=map.get(key);
      const ec=shortExprCost(expr), oc=old ? shortExprCost(old) : Infinity;
      if(!old || ec<oc || (Math.abs(ec-oc)<1e-9 && expr.s.length<old.s.length)) map.set(key,expr);
    }
    function buildShortAtoms(target, deadline){
      const absT=absBig(target);
      const ratioBound = absT * 100000000n + 1000000000n;
      const nearBound = absT * 256n + 1000000n;
      const bound = ratioBound > nearBound ? ratioBound : nearBound;
      const best=new Map();
      function add(v,s,ops=0,depth=0,kind='atom',extra=0){
        if(performance.now()>deadline) return;
        addBest(best, makeExpr(v,s,ops,depth,kind,extra), bound);
      }
      for(let i=0;i<=250;i++) add(BigInt(i), String(i), 0, 0, 'literal');
      let fact=1n;
      const factorialExprs=[];
      for(let i=1;i<=80;i++){
        fact *= BigInt(i);
        if(i>=2){ const e=makeExpr(fact, `${i}!`, 1, 1, 'factorial'); factorialExprs.push({n:BigInt(i), v:fact, e}); addBest(best,e,bound); }
        if(i>24 && fact>bound) break;
      }
      // Early high-yield atoms are generated before the larger power/binomial sweeps,
      // so very large inputs still have time to find the surprising forms.
      const earlyNSources=[];
      for(const f of factorialExprs){ if(f.n>=3n && f.n<=6n && f.v<=220n) earlyNSources.push({v:f.v, s:f.e.s, ops:f.e.ops, depth:f.e.depth}); }
      for(let n=5;n<=80;n++) earlyNSources.push({v:BigInt(n), s:String(n), ops:0, depth:0});
      const earlyKSources=[];
      for(const f of factorialExprs){ if(f.n>=3n && f.n<=5n && f.v<=60n) earlyKSources.push({v:f.v, s:f.e.s, ops:f.e.ops, depth:f.e.depth}); }
      for(let k=2;k<=24;k++) earlyKSources.push({v:BigInt(k), s:String(k), ops:0, depth:0});
      for(const ns of earlyNSources){
        for(const ks of earlyKSources){
          if(ks.v<=1n || ks.v>=ns.v || ks.v>ns.v/2n || ks.v>32n) continue;
          const v=binomBigLimited(ns.v,ks.v,bound);
          if(v!==null) add(v, `binom(${ns.s},${ks.s})`, (ns.ops||0)+(ks.ops||0)+1, Math.max(ns.depth||0,ks.depth||0)+1, 'binomial', -6);
        }
      }
      if(Number.isFinite(Number(absT)) && Number(absT)>0){
        const earlyP=[];
        for(let p=2;p<=42;p++) earlyP.push({v:BigInt(p), s:String(p), ops:0, depth:0});
        for(const f of factorialExprs){ if(f.n>=3n && f.n<=5n) earlyP.push({v:f.v, s:f.e.s, ops:f.e.ops, depth:f.e.depth}); }
        for(const f of factorialExprs){ if(f.n===5n){ for(let c=2;c<=3;c++) earlyP.push({v:f.v*BigInt(c), s:`${c}·${f.e.s}`, ops:f.e.ops+1, depth:f.e.depth+1}); } }
        for(let base=2;base<=16;base++){
          for(const ps of earlyP){
            for(let q=2;q<=18;q++){
              const exp=Number(ps.v)/q;
              if(exp<2 || exp>65) continue;
              const rough=Math.pow(base, exp), tnum=Number(absT);
              if(!Number.isFinite(rough) || !Number.isFinite(tnum) || Math.abs(Math.log(rough)-Math.log(tnum))>0.8) continue;
              const val=exactRoundRationalPower(BigInt(base), ps.v, BigInt(q));
              if(val!==null && val<=bound){
                const core=`${base}^(${ps.s}/${q})`;
                add(val, `round(${core})`, (ps.ops||0)+3, (ps.depth||0)+2, 'rounded-power', -14);
                const cmp=compareRationalPowerToInteger(BigInt(base), ps.v, BigInt(q), val);
                const floorVal = cmp<0 ? val-1n : val;
                const ceilVal = cmp>0 ? val+1n : val;
                if(floorVal>0n) add(floorVal, `floor(${core})`, (ps.ops||0)+3, (ps.depth||0)+2, 'floored-power', -8);
                if(ceilVal>0n) add(ceilVal, `ceil(${core})`, (ps.ops||0)+3, (ps.depth||0)+2, 'ceiled-power', -8);
              }
            }
          }
        }
      }
      for(let a=2;a<=50;a++){
        let v=1n;
        for(let b=1;b<=220;b++){
          v *= BigInt(a);
          if(b>=2) add(v, `${a}^${b}`, 1, 1, 'power');
          if(v>bound) break;
        }
      }
      for(let a=2;a<=12;a++){
        for(let b=2;b<=24;b++){
          if(performance.now()>deadline) break;
          const base=BigInt(a)**BigInt(b);
          if(base>bound) continue;
          for(let c=2;c<=9;c++) add(base*BigInt(c), `${c}·${a}^${b}`, 2, 2, 'scaled-power', 16);
        }
      }
      const nSources=[];
      for(const f of factorialExprs){ if(f.n>=3n && f.n<=6n && f.v<=220n) nSources.push({v:f.v, s:f.e.s, ops:f.e.ops, depth:f.e.depth}); }
      for(let n=5;n<=170;n++) nSources.push({v:BigInt(n), s:String(n), ops:0, depth:0});
      const kSources=[];
      for(const f of factorialExprs){ if(f.n>=3n && f.n<=5n && f.v<=60n) kSources.push({v:f.v, s:f.e.s, ops:f.e.ops, depth:f.e.depth}); }
      for(let k=2;k<=60;k++) kSources.push({v:BigInt(k), s:String(k), ops:0, depth:0});
      const seenBinom=new Set();
      for(const ns of nSources){
        if(performance.now()>deadline) break;
        for(const ks of kSources){
          if(ks.v<=1n || ks.v>=ns.v || ks.v>ns.v/2n || ks.v>32n) continue;
          const key=ns.s+','+ks.s; if(seenBinom.has(key)) continue; seenBinom.add(key);
          const v=binomBigLimited(ns.v,ks.v,bound);
          if(v!==null) add(v, `binom(${ns.s},${ks.s})`, (ns.ops||0)+(ks.ops||0)+1, Math.max(ns.depth||0,ks.depth||0)+1, 'binomial', -2);
        }
      }
      if(Number.isFinite(Number(absT)) && Number(absT)>0){
        const pSources=[];
        for(let p=2;p<=80;p++) pSources.push({v:BigInt(p), s:String(p), ops:0, depth:0});
        for(const f of factorialExprs){ if(f.n>=3n && f.n<=5n) pSources.push({v:f.v, s:f.e.s, ops:f.e.ops, depth:f.e.depth}); }
        for(const f of factorialExprs){ if(f.n===5n){ for(let c=2;c<=3;c++) pSources.push({v:f.v*BigInt(c), s:`${c}·${f.e.s}`, ops:f.e.ops+1, depth:f.e.depth+1}); } }
        for(let base=2;base<=16;base++){
          for(const ps of pSources){
            for(let q=2;q<=24;q++){
              if(performance.now()>deadline) break;
              const exp=Number(ps.v)/q;
              if(exp<2 || exp>70) continue;
              const rough=Math.pow(base, exp);
              if(!Number.isFinite(rough) || rough<1 || rough>Number.MAX_VALUE) continue;
              const tnum=Number(absT);
              if(!Number.isFinite(tnum) || Math.abs(Math.log(rough)-Math.log(tnum))>0.8) continue;
              const val=exactRoundRationalPower(BigInt(base), ps.v, BigInt(q));
              if(val!==null && val<=bound){
                const core=`${base}^(${ps.s}/${q})`;
                add(val, `round(${core})`, (ps.ops||0)+3, (ps.depth||0)+2, 'rounded-power', -10);
                const cmp=compareRationalPowerToInteger(BigInt(base), ps.v, BigInt(q), val);
                const floorVal = cmp<0 ? val-1n : val;
                const ceilVal = cmp>0 ? val+1n : val;
                if(floorVal>0n) add(floorVal, `floor(${core})`, (ps.ops||0)+3, (ps.depth||0)+2, 'floored-power', -5);
                if(ceilVal>0n) add(ceilVal, `ceil(${core})`, (ps.ops||0)+3, (ps.depth||0)+2, 'ceiled-power', -5);
              }
            }
          }
        }
      }
      const atoms=[...best.values()].sort((a,b)=>shortExprCost(a)-shortExprCost(b) || a.s.length-b.s.length || (a.v<b.v?-1:a.v>b.v?1:0));
      const byValue=new Map(atoms.map(e=>[e.v.toString(), e]));
      const small=atoms.filter(e=>e.v<=1000000n).slice(0,1800);
      return {atoms, byValue, small, bound};
    }
    function bestResidualExpr(r, atomsData, deadline){
      r=absBig(r);
      if(r===0n) return makeExpr(0n,'0',0,0,'literal');
      const direct=atomsData.byValue.get(r.toString());
      let best=direct || null;
      function consider(e){ if(!e) return; if(!best || shortExprCost(e)<shortExprCost(best) || (Math.abs(shortExprCost(e)-shortExprCost(best))<1e-9 && e.s.length<best.s.length)) best=e; }
      if(r<=1000000000000n){
        for(const a of atomsData.small){
          if(performance.now()>deadline) break;
          if(a.v>r) continue;
          const b=atomsData.byValue.get((r-a.v).toString());
          if(b && b.v!==0n) consider(shortCombineExpr(a,'+',b,r));
        }
      }
      const parts=[]; let x=r, k=0;
      while(x>0n && parts.length<=7){ if(x&1n) parts.push(k); x >>= 1n; k++; }
      if(x===0n && parts.length>0 && parts.length<=5){
        const terms=parts.reverse().map(k=> k===0 ? '1' : (k===1 ? '2' : `2^${k}`));
        const ssum=terms.join('+');
        consider(makeExpr(r,ssum,parts.length-1+parts.filter(k=>k>=2).length,2,'binary-sum',24));
      }
      return best;
    }
    function nearestAtoms(target, atoms, count, deadline){
      const rows=[];
      for(const a of atoms){ if(performance.now()>deadline) break; rows.push(a); }
      rows.sort((a,b)=>{ const da=absBig(a.v-target), db=absBig(b.v-target); if(da!==db) return da<db?-1:1; return shortExprCost(a)-shortExprCost(b); });
      return rows.slice(0,count);
    }
    function normalizeAdditiveDisplay(s){
      s=String(s).replace(/\+0$/,'').replace(/-0$/,'');
      if(/[()\/]/.test(s) || !s.includes('+')) return s;
      const terms=s.split('+');
      if(terms.some(t=>!t || /[()\/-]/.test(t))) return s;
      function rank(t){ let m=t.match(/^(\d+)\^(\d+)$/); if(m) return 1000000 + Number(m[2])*100 + Number(m[1]); m=t.match(/^(\d+)!$/); if(m) return 900000 + Number(m[1]); m=t.match(/^binom/); if(m) return 850000; m=t.match(/^\d+$/); if(m) return Number(m[0]); return 500000 - t.length; }
      return terms.slice().sort((a,b)=>rank(b)-rank(a) || a.localeCompare(b)).join('+');
    }
    function makeShortRow(expr, target, label='exact shortform'){
      const display=normalizeAdditiveDisplay(expr.s);
      return {candidate:`${label}: ${display}`, value:`exact = ${shortPrettyValue(expr.v)}`, err:0, errBig:0n, beauty:shortExprCost(expr), feature:exprFeature(display)};
    }
    function addShortRow(rows, seen, expr, target, label){
      if(!expr || expr.v===undefined || !expr.s) return;
      if(expr.v!==target) return;
      const display=normalizeAdditiveDisplay(expr.s);
      const key=display+'|'+expr.v.toString();
      if(seen.has(key)) return;
      seen.add(key);
      rows.push(makeShortRow({...expr,s:display}, target, label));
    }
    function bestDenominatorExpr(d, atomsData){
      d=absBig(d);
      const direct=atomsData.byValue.get(d.toString());
      if(direct) return direct;
      // Keep ratio denominators visually clean; use a short binary form only when tiny.
      const parts=[]; let x=d, k=0;
      while(x>0n && parts.length<=4){ if(x&1n) parts.push(k); x >>= 1n; k++; }
      if(x===0n && parts.length>1 && parts.length<=3){
        const terms=parts.reverse().map(k=> k===0 ? '1' : (k===1 ? '2' : `2^${k}`));
        return makeExpr(d, terms.join('+'), parts.length-1+parts.filter(k=>k>=2).length, 2, 'binary-sum', 26);
      }
      return null;
    }
    function adjustmentResiduals(target, atomsData){
      const base=[];
      for(let i=1;i<=24;i++) base.push(makeExpr(BigInt(i),String(i),0,0,'literal'));
      for(const a of atomsData.small){ if(a.v>0n && a.v<=100000n) base.push(a); }
      const seen=new Set(), out=[];
      for(const e of base.sort((a,b)=>shortExprCost(a)-shortExprCost(b))){
        const k=e.v.toString(); if(seen.has(k)) continue; seen.add(k); out.push(e); if(out.length>=42) break;
      }
      return out;
    }
    function combineAdjustment(baseExpr, finalTarget, residualExpr, mode){
      if(mode==='plus') return shortCombineExpr(baseExpr,'+',residualExpr,finalTarget);
      if(mode==='minus') return shortCombineExpr(baseExpr,'-',residualExpr,finalTarget);
      return baseExpr;
    }
    function shortDivisionCandidatesForVariant(rows, seen, variant, finalTarget, residualExpr, mode, atomsData, deadline, maxNums=1600){
      if(variant<=0n) return;
      const nums=atomsData.atoms.filter(e=>e.v>variant/4n && e.v>1n && e.s !== variant.toString()).slice(0,maxNums);
      for(const a of nums){
        if(performance.now()>deadline) break;
        const d0=roundDiv(a.v, variant);
        for(const dd of [d0-2n,d0-1n,d0,d0+1n,d0+2n]){
          if(dd<=1n || dd>100000000n) continue;
          const b=bestDenominatorExpr(dd, atomsData);
          if(!b || b.s.length>24) continue;
          const floorV=a.v/dd;
          const ceilV=(a.v+dd-1n)/dd;
          const roundV=roundDiv(a.v,dd);
          const den=shortOperand(b,'/');
          const bases=[];
          if(roundV===variant) bases.push(makeExpr(roundV, `round(${a.s}/${den})`, (a.ops||0)+(b.ops||0)+2, Math.max(a.depth||0,b.depth||0)+1, 'rounded-ratio', -8));
          if(floorV===variant) bases.push(makeExpr(floorV, `floor(${a.s}/${den})`, (a.ops||0)+(b.ops||0)+2, Math.max(a.depth||0,b.depth||0)+1, 'floored-ratio', -2));
          if(ceilV===variant) bases.push(makeExpr(ceilV, `ceil(${a.s}/${den})`, (a.ops||0)+(b.ops||0)+2, Math.max(a.depth||0,b.depth||0)+1, 'ceiled-ratio', -2));
          for(const base of bases){
            const expr=combineAdjustment(base, finalTarget, residualExpr, mode);
            addShortRow(rows, seen, expr, finalTarget, 'exact shortform');
          }
        }
      }
    }
    function shortDivisionCandidates(rows, seen, target, atomsData, deadline){
      const zero=makeExpr(0n,'0',0,0,'literal');
      const residuals=adjustmentResiduals(target, atomsData);
      shortDivisionCandidatesForVariant(rows, seen, target, target, zero, 'none', atomsData, deadline, 600);
      for(const r of residuals.slice(0,24)){
        if(performance.now()>deadline) break;
        if(target>r.v) shortDivisionCandidatesForVariant(rows, seen, target-r.v, target, r, 'plus', atomsData, deadline, 950);
        shortDivisionCandidatesForVariant(rows, seen, target+r.v, target, r, 'minus', atomsData, deadline, 650);
      }
      shortDivisionCandidatesForVariant(rows, seen, target, target, zero, 'none', atomsData, deadline, 1800);
      for(const r of residuals.slice(24)){
        if(performance.now()>deadline) break;
        if(target>r.v) shortDivisionCandidatesForVariant(rows, seen, target-r.v, target, r, 'plus', atomsData, deadline, 350);
        shortDivisionCandidatesForVariant(rows, seen, target+r.v, target, r, 'minus', atomsData, deadline, 250);
      }
    }
    function shortProductCandidates(rows, seen, target, atomsData, deadline){
      const probe=atomsData.atoms.filter(e=>e.v>1n && e.v<=target && e.s.length<=18).slice(0,2600);
      for(const a of probe){
        if(performance.now()>deadline) break;
        if(target%a.v===0n){
          const q=target/a.v;
          const b=bestResidualExpr(q, atomsData, deadline);
          if(b && b.v>1n && b.s!==target.toString()) addShortRow(rows, seen, shortCombineExpr(a,'*',b,target), target, 'exact shortform');
        }
      }
    }
    function shortAtomResidualCandidates(rows, seen, target, atomsData, deadline){
      for(const a of nearestAtoms(target, atomsData.atoms, 3600, deadline)){
        if(performance.now()>deadline) break;
        const diff=target-a.v;
        if(diff===0n){ if(!(a.kind==='literal' && target>20n)) addShortRow(rows, seen, a, target, 'exact shortform'); continue; }
        const r=bestResidualExpr(diff<0n ? -diff : diff, atomsData, deadline);
        if(!r || r.s.length>34) continue;
        const expr=diff>0n ? shortCombineExpr(a,'+',r,target) : shortCombineExpr(a,'-',r,target);
        addShortRow(rows, seen, expr, target, 'exact shortform');
      }
    }
    function shortTwoAtomCandidates(rows, seen, target, atomsData, deadline){
      const probe=atomsData.atoms.slice(0,3600);
      for(const a of probe){
        if(performance.now()>deadline) break;
        if(a.v<=target){
          const b=atomsData.byValue.get((target-a.v).toString());
          if(b && b.v!==0n) addShortRow(rows, seen, shortCombineExpr(a,'+',b,target), target, 'exact shortform');
        }
        const b=atomsData.byValue.get((a.v-target).toString());
        if(b && b.v!==0n) addShortRow(rows, seen, shortCombineExpr(a,'-',b,target), target, 'exact shortform');
      }
    }
    function emergencyExactCandidates(rows, seen, target){
      // Last-resort exact decompositions: less surprising than the main search, but never approximate.
      const decimal=target.toString();
      for(const k of [3,4,6,8,9,12]){
        const scale=10n**BigInt(k);
        if(target<=scale) continue;
        const q=target/scale, r=target%scale;
        const base=makeExpr(q*scale, `${q.toString()}·10^${k}`, 2, 2, 'decimal-split', 120);
        const expr = r===0n ? base : shortCombineExpr(base, '+', makeExpr(r,r.toString(),0,0,'literal',80), target);
        addShortRow(rows, seen, expr, target, 'exact fallback');
      }
      const tnum=Number(target);
      if(Number.isFinite(tnum) && tnum>1){
        for(let base=2;base<=20;base++){
          const e=Math.max(2, Math.floor(Math.log(tnum)/Math.log(base)));
          for(const ee of [e-1,e,e+1]){
            if(ee<2 || ee>220) continue;
            const pv=BigInt(base)**BigInt(ee);
            if(pv===target) addShortRow(rows, seen, makeExpr(pv,`${base}^${ee}`,1,1,'power'), target, 'exact fallback');
            else if(pv<target){
              const r=target-pv;
              if(r.toString().length <= Math.max(4, decimal.length-3)) addShortRow(rows, seen, shortCombineExpr(makeExpr(pv,`${base}^${ee}`,1,1,'power'), '+', makeExpr(r,r.toString(),0,0,'literal',90), target), target, 'exact fallback');
            } else {
              const r=pv-target;
              if(r.toString().length <= Math.max(4, decimal.length-3)) addShortRow(rows, seen, shortCombineExpr(makeExpr(pv,`${base}^${ee}`,1,1,'power'), '-', makeExpr(r,r.toString(),0,0,'literal',90), target), target, 'exact fallback');
            }
          }
        }
      }
      const bits=[]; let x=target, pos=0;
      while(x>0n && bits.length<=14){ if(x&1n) bits.push(pos); x >>= 1n; pos++; }
      if(x===0n && bits.length>1 && bits.length<=14){
        const terms=bits.reverse().map(k=>k===0?'1':(k===1?'2':`2^${k}`));
        addShortRow(rows, seen, makeExpr(target, terms.join('+'), terms.length-1+terms.filter(t=>t.includes('^')).length, 3, 'binary-sum', 90), target, 'exact fallback');
      }
    }
    function literalFallbackRow(rows, seen, target){
      addShortRow(rows, seen, makeExpr(target,target.toString(),0,0,'literal'), target, 'literal fallback');
    }
    function selectDiverseShortforms(rows, limit=5){
      const sorted=rows.sort((a,b)=>shortResultScore(a)-shortResultScore(b) || a.candidate.length-b.candidate.length);
      const nonLiteral=sorted.filter(r=>!r.candidate.startsWith('literal fallback'));
      const pool=nonLiteral.length>=limit ? nonLiteral : sorted;
      const picked=[]; const used=new Map();
      for(const r of pool){
        const f=r.feature || 'simple';
        const usedCount=used.get(f)||0;
        if(usedCount===0 || picked.length<2){ picked.push(r); used.set(f,usedCount+1); }
        if(picked.length>=limit) break;
      }
      for(const r of pool){ if(picked.length>=limit) break; if(!picked.includes(r)) picked.push(r); }
      return picked.slice(0,limit);
    }
    function integerShortformRows(settings){
      const rawN=integerInputBig(settings.raw);
      if(rawN===null || rawN===0n) return [];
      const startTime=performance.now();
      const deadline=startTime+1150;
      const sign = rawN<0n ? -1n : 1n;
      const target=absBig(rawN);
      const atomsData=buildShortAtoms(target, startTime+720);
      const rows=[]; const seen=new Set();
      shortAtomResidualCandidates(rows, seen, target, atomsData, deadline);
      shortTwoAtomCandidates(rows, seen, target, atomsData, deadline);
      shortDivisionCandidates(rows, seen, target, atomsData, deadline);
      shortProductCandidates(rows, seen, target, atomsData, deadline);
      if(rows.length<5) emergencyExactCandidates(rows, seen, target);
      literalFallbackRow(rows, seen, target);
      let selected=selectDiverseShortforms(rows, 5);
      if(sign<0n){
        selected=selected.map(r=>{
          const expr=r.candidate.replace(/^([^:]+):\s*/, '$1: -(')+')';
          const val=r.value.replace('exact = ', 'exact = -');
          return {...r, candidate:expr, value:val};
        });
      }
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
          statusEl.textContent=`Returned ${rows.length} exact shortform candidate(s) in ${dtShort} ms. Integer algebraic search is skipped; factorization is continuing…`;
          setTimeout(()=>{
            rows=rows.concat(factorRows(settings));
            renderRows(rows);
            const dt=Math.round(performance.now()-t0);
            statusEl.className='notice status-line good';
            statusEl.textContent=`Returned ${rows.length} exact integer result(s) in ${dt} ms. Algebraic-number candidates were skipped for this integer input.`;
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
  