import random, math
from collections import deque, defaultdict

# =========================
# Game
# =========================
class Minesweeper:
    __slots__=("height","width","mines_count","mines","board","mines_found","first_move_made","_nb")
    def __init__(self,height=8,width=8,mines=8):
        self.height,self.width,self.mines_count=height,width,mines
        self.mines=set(); self.mines_found=set()
        self.board=[[False]*width for _ in range(height)]
        self.first_move_made=False; self._nb={}
    def _neighbors(self,cell):
        if cell in self._nb: return self._nb[cell]
        r0,c0=cell; out=[]
        for r in (r0-1,r0,r0+1):
            if 0<=r<self.height:
                for c in (c0-1,c0,c0+1):
                    if 0<=c<self.width and (r,c)!=cell: out.append((r,c))
        self._nb[cell]=out; return out
    def make_safe_first_move(self,cell):
        """保证点击的格子安全，保证周围安全。"""
        if self.first_move_made: return
        self.first_move_made=True
        safe={cell}; r0,c0=cell
        for r in range(r0-1,r0+2):
            for c in range(c0-1,c0+2):
                if 0<=r<self.height and 0<=c<self.width: safe.add((r,c))
        cap=self.height*self.width-len(safe)
        if self.mines_count>cap: raise ValueError(f"Too many mines: mines={self.mines_count}, cap={cap}")
        while len(self.mines)<self.mines_count:
            r,c=random.randrange(self.height),random.randrange(self.width)
            if (r,c) in safe or self.board[r][c]: continue
            self.mines.add((r,c)); self.board[r][c]=True
    def make_safe_first_move2(self, cell):
        """只保证点击的格子安全，不保证周围安全。"""
        if self.first_move_made: return
        self.first_move_made = True
        r0, c0 = cell
        cap = self.height * self.width - 1
        if self.mines_count > cap: raise ValueError(f"Too many mines: mines={self.mines_count}, cap={cap}")
        while len(self.mines) < self.mines_count:
            r, c = random.randrange(self.height), random.randrange(self.width)
            if (r, c) != cell and (r, c) not in self.mines:
                self.mines.add((r, c)); self.board[r][c] = True

    def is_mine(self,cell): r,c=cell; return self.board[r][c]
    def nearby_mines(self,cell): return sum(1 for nb in self._neighbors(cell) if self.is_mine(nb))
    def reveal_chain(self,cell):
        if self.is_mine(cell): return {cell:-1}
        res,q,vis={},deque([cell]),{cell}
        while q:
            cur=q.popleft()
            if self.is_mine(cur): continue
            n=self.nearby_mines(cur); res[cur]=n
            if n==0:
                for nb in self._neighbors(cur):
                    if nb in vis: continue
                    vis.add(nb)
                    if not self.is_mine(nb): q.append(nb)
        return res
    def won(self): return self.mines_found==self.mines

# =========================
# Logic sentence
# =========================
class Sentence:
    __slots__=("cells","count")
    def __init__(self,cells,count): self.cells,setcount=set(cells),int(count); self.count=setcount
    def __eq__(self,o): return self.cells==o.cells and self.count==o.count
    def known_mines(self): return set(self.cells) if self.count==len(self.cells) and self.count>0 else set()
    def known_safes(self): return set(self.cells) if self.count==0 else set()
    def mark_mine(self,cell):
        if cell in self.cells: self.cells.remove(cell); self.count-=1
    def mark_safe(self,cell):
        if cell in self.cells: self.cells.remove(cell)

# =========================
# AI
# =========================
class MinesweeperAI:
    __slots__=("height","width","total_mines","moves_made","mines","safes","knowledge",
               "MAX_EXACT_COMP","MC_BASE","MC_PER_VAR","MC_CAP","_all","_nb",
               "P_TOL","MAX_ENDGAME","RISK_W")
    def __init__(self,height=8,width=8,mines=15):
        self.height,self.width,self.total_mines=height,width,mines
        self.moves_made=set(); self.mines=set(); self.safes=set(); self.knowledge=[]
        self.MAX_EXACT_COMP=20; self.MC_BASE=2500; self.MC_PER_VAR=120; self.MC_CAP=14000
        self._all={(r,c) for r in range(height) for c in range(width)}; self._nb={}
        self.P_TOL=0.05          # within minP + tol, prefer more info
        self.MAX_ENDGAME=15      # <= this, do decision-tree for max win prob
        self.RISK_W=10.0         # penalty weight inside tolerance band

    # ---- marks ----
    def mark_mine(self,cell):
        if cell in self.mines: return
        self.mines.add(cell)
        for s in self.knowledge: s.mark_mine(cell)
    def mark_safe(self,cell):
        if cell in self.safes: return
        self.safes.add(cell)
        for s in self.knowledge: s.mark_safe(cell)

    # ---- helpers ----
    def _neighbors(self,cell):
        if cell in self._nb: return self._nb[cell]
        r0,c0=cell; out=[]
        for r in (r0-1,r0,r0+1):
            if 0<=r<self.height:
                for c in (c0-1,c0,c0+1):
                    if 0<=c<self.width and (r,c)!=cell: out.append((r,c))
        self._nb[cell]=out; return out

    # ---- knowledge ----
    def add_knowledge_batch(self,revealed_dict):
        moves,mines,safes,know=self.moves_made,self.mines,self.safes,self.knowledge
        for cell,count in revealed_dict.items():
            if count<0 or cell in moves: continue
            moves.add(cell); self.mark_safe(cell)
            unknown=set(); adj=0
            for n in self._neighbors(cell):
                if n in mines: adj+=1
                elif n not in safes and n not in moves: unknown.add(n)
            newc=count-adj
            if unknown and 0<=newc<=len(unknown):
                sen=Sentence(unknown,newc)
                if sen not in know: know.append(sen)
        self.update_knowledge(); self.infer_new_sentences()

    def update_knowledge(self):
        know=self.knowledge; changed=True
        while changed:
            changed=False; mines_new=set(); safes_new=set()
            for s in know:
                if not s.cells: continue
                if s.count==0: safes_new|=s.cells
                elif s.count==len(s.cells): mines_new|=s.cells
            for c in mines_new:
                if c not in self.mines: self.mark_mine(c); changed=True
            for c in safes_new:
                if c not in self.safes: self.mark_safe(c); changed=True
            know[:]=[s for s in know if s.cells and 0<=s.count<=len(s.cells)]

    def infer_new_sentences(self):
        know=self.knowledge; changed=True
        while changed:
            changed=False; know.sort(key=lambda s: len(s.cells)); new=[]
            n=len(know)
            for i in range(n):
                a=know[i]; ac=a.cells
                if not ac: continue
                for j in range(i+1,n):
                    b=know[j]
                    if len(ac)>len(b.cells): continue
                    if ac.issubset(b.cells):
                        diff=b.cells-ac; dc=b.count-a.count
                        if diff and 0<=dc<=len(diff):
                            ns=Sentence(diff,dc)
                            if ns not in know and ns not in new: new.append(ns); changed=True
            if new: know.extend(new); self.update_knowledge()

    # ---- moves ----
    def make_safe_move(self):
        avail=self.safes-self.moves_made
        return min(avail) if avail else None

    def make_random_move(self):
        mv=self.make_safe_move()
        if mv is not None: return mv

        unknown=list(self._all-self.moves_made-self.safes-self.mines)
        if not unknown: return None

        # endgame: maximize win probability (decision tree over posterior assignments)
        if len(unknown)<=self.MAX_ENDGAME:
            mv=self._endgame_best_move(unknown)
            if mv is not None: return mv

        probs,_=self._mine_probabilities()
        if not probs: return None

        zeros=[c for c,p in probs.items() if p==0.0 and c not in self.moves_made and c not in self.mines]
        if zeros: return min(zeros)

        cand=[(probs[c],c) for c in probs if c not in self.moves_made and c not in self.mines]
        if not cand: return None
        cand.sort(); minp=cand[0][0]
        band=[c for p,c in cand if p<=minp+self.P_TOL]

        if len(band)==1: return band[0]
        best=None; bestScore=-1e100
        for c in band:
            p=probs[c]; gain=self._info_gain_heuristic(c)
            score=gain - self.RISK_W*((p-minp)/max(1e-9,self.P_TOL))
            if score>bestScore or (abs(score-bestScore)<1e-12 and p<probs.get(best,1)):
                bestScore=score; best=c
        return best

    def _info_gain_heuristic(self,cell):
        unk=0
        for nb in self._neighbors(cell):
            if nb not in self.moves_made and nb not in self.safes and nb not in self.mines: unk+=1
        deg=0
        for s in self.knowledge:
            if cell in s.cells: deg+=1
        return 3*deg+unk

    # =========================
    # Endgame decision tree (<= MAX_ENDGAME unknown)
    # =========================
    def _endgame_best_move(self,unknown_cells):
        n=len(unknown_cells); idx={c:i for i,c in enumerate(unknown_cells)}
        rem=self.total_mines-len(self.mines)
        if rem<0 or rem>n: return None

        cons=[]
        for s in self.knowledge:
            inter=s.cells & set(unknown_cells)
            if not inter: continue
            m=0
            for c in inter: m|=1<<idx[c]
            cons.append((m,s.count))

        belief=[]
        for mask in range(1<<n):
            if mask.bit_count()!=rem: continue
            ok=True
            for cm,t in cons:
                if (mask&cm).bit_count()!=t: ok=False; break
            if ok: belief.append(mask)
        if not belief: return None
        belief=tuple(sorted(belief))
        full=(1<<n)-1

        # precompute neighbor masks + known-mine adj counts
        neigh_mask=[0]*n; neigh_known=[0]*n
        known_mines=self.mines
        for i,c in enumerate(unknown_cells):
            km=0; mm=0
            for nb in self._neighbors(c):
                if nb in known_mines: km+=1
                j=idx.get(nb)
                if j is not None: mm |= 1<<j
            neigh_mask[i]=mm; neigh_known[i]=km

        def reveal_from(start, mine_mask, already):
            if (mine_mask>>start)&1: return None
            q=[start]; vis=0; pairs=[]
            while q:
                i=q.pop()
                if (vis>>i)&1 or (already>>i)&1: continue
                vis |= 1<<i
                num = neigh_known[i] + (mine_mask & neigh_mask[i]).bit_count()
                pairs.append((i,num))
                if num==0:
                    nb = neigh_mask[i] & ~vis & ~already & ~mine_mask
                    while nb:
                        lsb = nb & -nb
                        q.append(lsb.bit_length()-1)
                        nb -= lsb
            vis &= ~already
            if not vis: return 0, ()
            pairs.sort()
            return vis, tuple(pairs)

        memo={}
        def all_won(bel, revealed):
            nr = ~revealed & full
            for m in bel:
                if ((~m)&full) & nr: return False
            return True

        def solve(bel, revealed):
            key=(revealed, bel)
            if key in memo: return memo[key]
            if all_won(bel, revealed): memo[key]=1.0; return 1.0

            # available actions: unrevealed, not certain-mine
            andm=full
            for m in bel: andm &= m
            acts = (~revealed & ~andm) & full
            if acts==0:  # no safe-to-click candidates (should mean already won)
                memo[key]=1.0 if all_won(bel,revealed) else 0.0
                return memo[key]

            best=0.0
            baseN=len(bel)
            a=acts
            while a:
                lsb=a & -a; i=lsb.bit_length()-1; a-=lsb
                groups=defaultdict(list)
                minecnt=0
                for m in bel:
                    if (m>>i)&1: minecnt+=1; continue
                    r = reveal_from(i, m, revealed)
                    if r is None: minecnt+=1; continue
                    v,pairs=r
                    groups[(v,pairs)].append(m)
                vprob=0.0
                denom=baseN
                if minecnt:
                    # lose branch contributes 0
                    pass
                for (v,pairs), gb in groups.items():
                    v2=revealed | v
                    vprob += (len(gb)/denom) * solve(tuple(gb), v2)
                if vprob>best: best=vprob
                if best>=1.0-1e-15: break
            memo[key]=best; return best

        # pick move with max win probability
        revealed0=0
        bestp=-1.0; besti=None
        acts=full
        # don't click certain mines even at root
        andm=full
        for m in belief: andm &= m
        acts &= ~andm
        a=acts
        while a:
            lsb=a & -a; i=lsb.bit_length()-1; a-=lsb
            groups=defaultdict(list); minecnt=0
            for m in belief:
                if (m>>i)&1: minecnt+=1; continue
                r=reveal_from(i,m,revealed0)
                if r is None: minecnt+=1; continue
                v,pairs=r
                groups[(v,pairs)].append(m)
            vprob=0.0; denom=len(belief)
            for (v,pairs), gb in groups.items():
                vprob += (len(gb)/denom) * solve(tuple(gb), v)
            if vprob>bestp: bestp=vprob; besti=i
        return unknown_cells[besti] if besti is not None else None

    # =========================
    # Probabilities (frontier components + exact/MC)
    # =========================
    def _mine_probabilities(self):
        unknown=list(self._all-self.moves_made-self.safes-self.mines)
        if not unknown: return {}, "no-unknown"
        rem_mines=max(0,self.total_mines-len(self.mines))

        frontier=set()
        for s in self.knowledge: frontier |= s.cells
        frontier &= set(unknown)
        uninformed=[c for c in unknown if c not in frontier]

        if not frontier:
            p=rem_mines/len(unknown)
            return {c:p for c in unknown}, f"uniform({len(unknown)})"

        comps=self._frontier_components_unionfind(frontier)
        infos=[]
        for comp in comps:
            n=len(comp)
            if n<=self.MAX_EXACT_COMP: infos.append(self._enum_component_exact(comp))
            else:
                samples=min(self.MC_CAP,self.MC_BASE+self.MC_PER_VAR*n)
                infos.append(self._enum_component_mc(comp,samples))

        probs_front,e_front=self._combine_components(infos,rem_mines,len(uninformed))
        probs=dict(probs_front)
        if uninformed:
            e_out=max(0.0,rem_mines-e_front)
            p_out=min(1.0,max(0.0,e_out/len(uninformed)))
            for c in uninformed: probs[c]=p_out
        for c,p in probs.items():
            if p==0.0: self.mark_safe(c)
            elif p==1.0: self.mark_mine(c)
        return probs, f"frontierE={e_front:.2f}"

    def _frontier_components_unionfind(self,frontier):
        parent={c:c for c in frontier}; rank={c:0 for c in frontier}
        def find(x):
            while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
            return x
        def union(a,b):
            ra,rb=find(a),find(b)
            if ra==rb: return
            if rank[ra]<rank[rb]: ra,rb=rb,ra
            parent[rb]=ra
            if rank[ra]==rank[rb]: rank[ra]+=1
        for s in self.knowledge:
            cells=[c for c in s.cells if c in frontier]
            if len(cells)>1:
                base=cells[0]
                for x in cells[1:]: union(base,x)
        comps=defaultdict(set)
        for c in frontier: comps[find(c)].add(c)
        return list(comps.values())

    def _constraints_for_comp(self,comp_set,idx):
        cons=[]
        for s in self.knowledge:
            inter=s.cells & comp_set
            if inter:
                cons.append(([idx[c] for c in inter], s.count))
        return cons

    def _enum_component_exact(self,comp_cells):
        cells=list(comp_cells); n=len(cells); idx={c:i for i,c in enumerate(cells)}
        cons=self._constraints_for_comp(comp_cells,idx)
        v2c=[[] for _ in range(n)]
        for ci,(vs,t) in enumerate(cons):
            for v in vs: v2c[v].append(ci)
        need=[t for _,t in cons]; rem=[len(vs) for vs,_ in cons]; assign=[-1]*n
        order=sorted(range(n), key=lambda v: len(v2c[v]), reverse=True)
        dist=defaultdict(int); cmd={c:defaultdict(int) for c in cells}
        def bt(k,mu):
            if k==n:
                for x in need:
                    if x: return
                dist[mu]+=1
                for i,c in enumerate(cells):
                    if assign[i]==1: cmd[c][mu]+=1
                return
            v=order[k]
            ok=True
            for ci in v2c[v]:
                r=rem[ci]-1; nd=need[ci]
                if nd<0 or nd>r: ok=False; break
            if ok:
                assign[v]=0
                for ci in v2c[v]: rem[ci]-=1
                bt(k+1,mu)
                for ci in v2c[v]: rem[ci]+=1
                assign[v]=-1
            ok=True
            for ci in v2c[v]:
                r=rem[ci]-1; nd=need[ci]-1
                if nd<0 or nd>r: ok=False; break
            if ok:
                assign[v]=1
                for ci in v2c[v]: rem[ci]-=1; need[ci]-=1
                bt(k+1,mu+1)
                for ci in v2c[v]: rem[ci]+=1; need[ci]+=1
                assign[v]=-1
        bt(0,0)
        return {"cells":cells,"dist":dist,"cell_mine_dist":cmd,"exact":True}

    def _enum_component_mc(self,comp_cells,samples):
        cells=list(comp_cells); n=len(cells); idx={c:i for i,c in enumerate(cells)}
        cons=self._constraints_for_comp(comp_cells,idx)
        v2c=[[] for _ in range(n)]
        for ci,(vs,t) in enumerate(cons):
            for v in vs: v2c[v].append(ci)
        order=sorted(range(n), key=lambda v: len(v2c[v]), reverse=True)
        dist=defaultdict(int); cmd={c:defaultdict(int) for c in cells}; rnd=random.random
        def one():
            need=[t for _,t in cons]; rem=[len(vs) for vs,_ in cons]; assign=[0]*n; mu=0
            for v in order:
                vc=v2c[v]
                ok0=True
                for ci in vc:
                    r=rem[ci]-1; nd=need[ci]
                    if nd<0 or nd>r: ok0=False; break
                ok1=True
                for ci in vc:
                    r=rem[ci]-1; nd=need[ci]-1
                    if nd<0 or nd>r: ok1=False; break
                if not ok0 and not ok1: return None
                if ok0 and ok1:
                    s0=s1=0
                    for ci in vc:
                        r=rem[ci]-1
                        s0 += (r-need[ci]); s1 += (r-(need[ci]-1))
                    den=s0+s1; p1=0.5 if den<=0 else max(0.15,min(0.85,s1/den))
                    val=1 if rnd()<p1 else 0
                else: val=1 if ok1 else 0
                assign[v]=val; mu+=val
                if val:
                    for ci in vc: rem[ci]-=1; need[ci]-=1
                else:
                    for ci in vc: rem[ci]-=1
            for x in need:
                if x: return None
            return assign,mu
        got=0; trials=0; cap=samples*30
        while got<samples and trials<cap:
            trials+=1; res=one()
            if res is None: continue
            assign,k=res; got+=1; dist[k]+=1
            for i,c in enumerate(cells):
                if assign[i]: cmd[c][k]+=1
        if got==0: dist[0]=1
        return {"cells":cells,"dist":dist,"cell_mine_dist":cmd,"exact":False,"got":got}

    def _combine_components(self,infos,rem_mines,outside_n):
        m=len(infos); dists=[info["dist"] for info in infos]
        pre=[defaultdict(int) for _ in range(m+1)]; pre[0][0]=1
        for i in range(m):
            pi,po,di=pre[i],pre[i+1],dists[i]
            for t,w in pi.items():
                for k,cnt in di.items(): po[t+k]+=w*cnt
        suf=[defaultdict(int) for _ in range(m+1)]; suf[m][0]=1
        for i in range(m-1,-1,-1):
            si,sn,di=suf[i],suf[i+1],dists[i]
            for t,w in sn.items():
                for k,cnt in di.items(): si[t+k]+=w*cnt
        comb=[math.comb(outside_n,k) for k in range(outside_n+1)]
        totalW=0; e_front_num=0
        for kf,ways in pre[m].items():
            out_need=rem_mines-kf
            if 0<=out_need<=outside_n:
                w=ways*comb[out_need]
                totalW+=w; e_front_num+=w*kf
        if totalW==0:
            front=sum(len(info["cells"]) for info in infos)
            p=rem_mines/max(1,front+outside_n)
            probs={}
            for info in infos:
                for c in info["cells"]: probs[c]=max(0.0,min(1.0,p))
            return probs, min(rem_mines,front)*p
        e_front=e_front_num/totalW; probs={}
        for j,info in enumerate(infos):
            ways_rest=defaultdict(int)
            for a,wa in pre[j].items():
                for b,wb in suf[j+1].items(): ways_rest[a+b]+=wa*wb
            distj,cmd=info["dist"],info["cell_mine_dist"]
            for cell in info["cells"]:
                num=0; cm=cmd[cell]
                for kj,cntk in distj.items():
                    cellmine=cm.get(kj,0)
                    if not cellmine: continue
                    for mr,wr in ways_rest.items():
                        out_need=rem_mines-(kj+mr)
                        if 0<=out_need<=outside_n: num += cellmine*wr*comb[out_need]
                probs[cell]=num/totalW
        return probs,e_front

    # ---- compatibility ----
    def calculate_safe_cells_if_safe(self,cell):
        if cell in self.safes or cell in self.mines: return 0
        return self._info_gain_heuristic(cell)
    def solve_frontier_lp(self,frontier_list,total_rem_mines):
        probs,_=self._mine_probabilities()
        frontier=[c for c in frontier_list if c in probs]
        if not frontier: return 0,0,{}
        return 0,min(len(frontier),total_rem_mines),{c:probs[c] for c in frontier}

# =========================
# Example runner (optional)
# =========================
def _play_one(h=16,w=30,mines=99,seed=None, firstmv=1):
    if seed is not None: random.seed(seed)
    game=Minesweeper(h,w,mines); ai=MinesweeperAI(h,w,mines)
    first=(h//2,w//2);
    # --- Modified section ---
    if firstmv == 1:
        game.make_safe_first_move(first)
    elif firstmv == 2:
        game.make_safe_first_move2(first)
    else:
        raise ValueError("firstmv must be 1 or 2")
    # -----------------------
    rev=game.reveal_chain(first)
    if rev.get(first)==-1: return False
    ai.add_knowledge_batch(rev)
    while True:
        if len(ai.moves_made)==h*w-mines: return True
        mv=ai.make_random_move()
        if mv is None or game.is_mine(mv): return False
        ai.add_knowledge_batch(game.reveal_chain(mv))

# =========================
# Browser API (Pyodide)
# =========================
_GAME=None; _AI=None; _REVEALED=set(); _REVEALED_N={}
_LOST=_WON=_FIRST=False; _H=_W=_M=0; _FIRST_MV_MODE=1 # Default mode
def _to_list_cell(cell): return [int(cell[0]),int(cell[1])]

def ms_new_game(h,w,mines,seed=None, firstmv=1):
    global _GAME,_AI,_REVEALED,_REVEALED_N,_LOST,_WON,_FIRST,_H,_W,_M, _FIRST_MV_MODE
    import random as _random
    _H,_W,_M=int(h),int(w),int(mines)
    _FIRST_MV_MODE = int(firstmv) # Store the mode
    if seed is not None: _random.seed(int(seed))
    _GAME=Minesweeper(_H,_W,_M); _AI=MinesweeperAI(_H,_W,_M)
    _REVEALED=set(); _REVEALED_N={}
    _LOST=_WON=_FIRST=False
    return ms_get_state()

def ms_get_state():
    return {"h":_H,"w":_W,"mines":_M,
            "revealed":[[r,c,int(_REVEALED_N.get((r,c),0))] for (r,c) in _REVEALED],
            "ai_mines":[_to_list_cell(c) for c in _AI.mines],
            "lost":bool(_LOST),"won":bool(_WON),"revealed_count":len(_REVEALED)}

def ms_step():
    global _GAME,_AI,_REVEALED,_REVEALED_N,_LOST,_WON,_FIRST,_H,_W,_M
    if _GAME is None:
        return {"move":None,"newly":[],"ai_mines":[],"lost":False,"won":False,"stuck":True,"revealed_count":0}
    if _LOST or _WON:
        return {"move":None,"newly":[],"ai_mines":[_to_list_cell(c) for c in _AI.mines],
                "lost":bool(_LOST),"won":bool(_WON),"stuck":False,"revealed_count":len(_REVEALED)}
    mv=_AI.make_random_move()
    if mv is None:
        return {"move":None,"newly":[],"ai_mines":[_to_list_cell(c) for c in _AI.mines],
                "lost":False,"won":False,"stuck":True,"revealed_count":len(_REVEALED)}
    if not _FIRST:
        # --- Modified section ---
        if _FIRST_MV_MODE == 1:
            _GAME.make_safe_first_move(mv)
        elif _FIRST_MV_MODE == 2:
            _GAME.make_safe_first_move2(mv)
        else:
            raise ValueError("Invalid _FIRST_MV_MODE, must be 1 or 2")
        # -----------------------
        _FIRST=True
    results=_GAME.reveal_chain(mv); newly=[]
    if any(v==-1 for v in results.values()):
        _LOST=True; _REVEALED.add(mv); _REVEALED_N[mv]=-1
        return {"move":_to_list_cell(mv),"newly":[[mv[0],mv[1],-1]],
                "ai_mines":[_to_list_cell(c) for c in _AI.mines],
                "lost":True,"won":False,"stuck":False,"revealed_count":len(_REVEALED)}
    _AI.add_knowledge_batch(results)
    for (r,c),n in results.items():
        if (r,c) not in _REVEALED:
            _REVEALED.add((r,c)); _REVEALED_N[(r,c)]=int(n)
            newly.append([r,c,int(n)])
    if len(_REVEALED)==_H*_W-_M: _WON=True
    return {"move":_to_list_cell(mv),"newly":newly,
            "ai_mines":[_to_list_cell(c) for c in _AI.mines],
            "lost":bool(_LOST),"won":bool(_WON),"stuck":False,"revealed_count":len(_REVEALED)}