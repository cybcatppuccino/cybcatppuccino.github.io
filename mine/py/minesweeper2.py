import random, math
from collections import deque, defaultdict

# =========================
# Game Logic
# =========================

class Minesweeper:
    __slots__ = ("height", "width", "mines_count", "mines", "board", "mines_found", "first_move_made", "_nb")
    
    def __init__(self, height=8, width=8, mines=8):
        self.height, self.width, self.mines_count = height, width, mines
        self.mines = set()
        self.mines_found = set()
        self.board = [[False] * width for _ in range(height)]
        self.first_move_made = False
        self._nb = {}

    def _neighbors(self, cell):
        if cell in self._nb:
            return self._nb[cell]
        r0, c0 = cell
        out = []
        for r in (r0 - 1, r0, r0 + 1):
            if 0 <= r < self.height:
                for c in (c0 - 1, c0, c0 + 1):
                    if 0 <= c < self.width and (r, c) != cell:
                        out.append((r, c))
        self._nb[cell] = out
        return out

    def _xorshift32(self, seed):
        """Xorshift32 PRNG implementation matching JS version"""
        x = seed if seed != 0 else 0x12345678
        
        def next_int():
            nonlocal x
            x = x & 0xFFFFFFFF  # Ensure 32-bit unsigned
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= (x >> 17) & 0xFFFFFFFF
            x ^= (x << 5) & 0xFFFFFFFF
            return x & 0xFFFFFFFF
        
        return next_int

    def _rand_int(self, rng_func, n):
        """Generate random integer in [0, n) using the PRNG"""
        u = rng_func()
        return u % n

    def _gen_mines_layout(self, seed, first_click, mode):
        """Deterministic mine placement using xorshift32 PRNG"""
        sr, sc = first_click
        safe = {(sr, sc)}
        
        # Add neighboring cells to safe zone based on mode
        if mode == 1:
            for rr, cc in self._neighbors((sr, sc)):
                safe.add((rr, cc))
        # mode == 2: only the clicked cell is safe (already added)

        # Collect all possible mine positions
        cells = []
        for r in range(self.height):
            for c in range(self.width):
                if (r, c) not in safe:
                    cells.append((r, c))
        
        if self.mines_count > len(cells):
            raise ValueError(f"Too many mines: mines={self.mines_count}, available={len(cells)}")

        # Fisher-Yates partial shuffle using deterministic PRNG
        rng_func = self._xorshift32(seed)
        for i in range(self.mines_count):
            j = i + self._rand_int(rng_func, len(cells) - i)
            cells[i], cells[j] = cells[j], cells[i]

        # Create mine layout
        layout = [[False] * self.width for _ in range(self.height)]
        for i in range(self.mines_count):
            r, c = cells[i]
            layout[r][c] = True
        
        return layout

    def make_safe_first_move(self, cell, seed=None):
        """保证点击的格子安全，保证周围安全。使用确定性随机数生成。"""
        if self.first_move_made:
            return
        self.first_move_made = True
        
        # Generate deterministic mine layout
        layout = self._gen_mines_layout(seed or 0, cell, 1)
        
        # Apply the layout
        self.mines = set()
        for r in range(self.height):
            for c in range(self.width):
                if layout[r][c]:
                    self.mines.add((r, c))
                    self.board[r][c] = True

    def make_safe_first_move2(self, cell, seed=None):
        """只保证点击的格子安全，不保证周围安全。使用确定性随机数生成。"""
        if self.first_move_made:
            return
        self.first_move_made = True
        
        # Generate deterministic mine layout
        layout = self._gen_mines_layout(seed or 0, cell, 2)
        
        # Apply the layout
        self.mines = set()
        for r in range(self.height):
            for c in range(self.width):
                if layout[r][c]:
                    self.mines.add((r, c))
                    self.board[r][c] = True

    def is_mine(self, cell):
        r, c = cell
        return self.board[r][c]

    def nearby_mines(self, cell):
        return sum(1 for nb in self._neighbors(cell) if self.is_mine(nb))

    def reveal_chain(self, cell):
        if self.is_mine(cell):
            return {cell: -1}
        res, q, vis = {}, deque([cell]), {cell}
        while q:
            cur = q.popleft()
            if self.is_mine(cur):
                continue
            n = self.nearby_mines(cur)
            res[cur] = n
            if n == 0:
                for nb in self._neighbors(cur):
                    if nb in vis:
                        continue
                    vis.add(nb)
                    if not self.is_mine(nb):
                        q.append(nb)
        return res

    def won(self):
        return self.mines_found == self.mines

# =========================
# Logic sentence
# =========================
class Sentence:
    __slots__=("cells","count")
    def __init__(self,cells,count):
        self.cells = set(cells)
        self.count = int(count)
    def __eq__(self,o): return self.cells==o.cells and self.count==o.count
    def known_mines(self): return set(self.cells) if self.count==len(self.cells) and self.count>0 else set()
    def known_safes(self): return set(self.cells) if self.count==0 else set()
    def mark_mine(self,cell):
        if cell in self.cells: self.cells.remove(cell); self.count-=1
    def mark_safe(self,cell):
        if cell in self.cells: self.cells.remove(cell)

# =========================
# AI Optimized (Hybrid: Gaussian + Exact/MC)
# =========================
class MinesweeperAI:
    __slots__=("height","width","total_mines","moves_made","mines","safes","knowledge",
               "_all","_nb","P_TOL","MAX_ENDGAME","RISK_W","LOW_P_FIRST",
               "MAX_EXACT_VAR", "MC_SAMPLES")

    def __init__(self,height=8,width=8,mines=15):
        self.height,self.width,self.total_mines=height,width,mines
        self.moves_made=set(); self.mines=set(); self.safes=set(); self.knowledge=[]
        self._all={(r,c) for r in range(height) for c in range(width)}; self._nb={}
        
        # Parameters
        self.P_TOL=0.05
        self.MAX_ENDGAME=15
        self.RISK_W=10.0
        self.LOW_P_FIRST=0.05
        
        # Optimization Parameters
        self.MAX_EXACT_VAR = 40    # Threshold to switch from Exact to MC inside a component
        self.MC_SAMPLES = 40000    # Samples for Local MC

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
        mv = self.make_safe_move()
        if mv is not None: return mv

        unknown = list(self._all - self.moves_made - self.safes - self.mines)
        if not unknown: return None

        probs, _ = self._mine_probabilities(mark=False)
        if not probs: return None

        cand = [(probs[c], c) for c in probs if c not in self.moves_made and c not in self.mines]
        if not cand: return None
        cand.sort()
        minp = cand[0][0]

        if minp <= self.LOW_P_FIRST:
            low = [(p, c) for (p, c) in cand if p <= self.LOW_P_FIRST]
            pbest = min(p for p, _ in low)
            pool = [c for p, c in low if abs(p - pbest) <= 1e-4]
            return max(pool, key=lambda c: (self._info_gain_heuristic(c), -c[0], -c[1]))

        if len(unknown) <= self.MAX_ENDGAME:
            mv = self._endgame_best_move(unknown)
            if mv is not None: return mv

        band = [c for p, c in cand if p <= minp + self.P_TOL]
        if len(band) == 1: return band[0]

        best = None; bestScore = -1e100
        for c in band:
            p = probs[c]
            gain = self._info_gain_heuristic(c)
            score = gain - self.RISK_W * ((p - minp) / max(1e-4, self.P_TOL))
            if score > bestScore or (abs(score - bestScore) < 1e-4 and p < probs.get(best, 1)):
                bestScore = score; best = c
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
    # Endgame decision tree
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
            andm=full
            for m in bel: andm &= m
            acts = (~revealed & ~andm) & full
            if acts==0:
                memo[key]=1.0 if all_won(bel,revealed) else 0.0
                return memo[key]

            best=0.0; baseN=len(bel); a=acts
            while a:
                lsb=a & -a; i=lsb.bit_length()-1; a-=lsb
                groups=defaultdict(list); minecnt=0
                for m in bel:
                    if (m>>i)&1: minecnt+=1; continue
                    r = reveal_from(i, m, revealed)
                    if r is None: minecnt+=1; continue
                    v,pairs=r
                    groups[(v,pairs)].append(m)
                vprob=0.0; denom=baseN
                if minecnt: pass
                for (v,pairs), gb in groups.items():
                    vprob += (len(gb)/denom) * solve(tuple(gb), revealed | v)
                if vprob>best: best=vprob
                if best>=1.0-1e-4: break
            memo[key]=best; return best

        bestp=-1.0; besti=None; revealed0=0
        acts=full
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
    # CORE OPTIMIZATION: HYBRID SOLVER
    # =========================
    def _mine_probabilities(self, *, mark=False):
        """
        Calculates mine probabilities using Adaptive Strategy:
        1. Small Component -> Direct Exact Backtracking (Fastest)
        2. Medium/Large Component -> Gaussian Reduce -> Exact Backtracking
        3. Huge Component -> Gaussian Reduce -> Local Monte Carlo
        """
        unknown = list(self._all - self.moves_made - self.safes - self.mines)
        if not unknown: return {}, "no-unknown"

        frontier = set()
        for s in self.knowledge:
            if s.cells: frontier |= s.cells
        frontier &= set(unknown)
        outside = [c for c in unknown if c not in frontier]

        # 1. Decompose
        components = self._find_components(frontier)
        comp_results = []
        
        GAUSSIAN_THRESHOLD = 25 # 只有变量数大于此值才开销消元，否则直接算更快

        # 2. Solve components
        for comp_cells in components:
            n_comp = len(comp_cells)
            
            # Strategy Selection
            if n_comp <= GAUSSIAN_THRESHOLD:
                # Fast Path: Skip Gaussian matrix construction overhead
                reduced_cells = list(comp_cells)
                reduced_cons = self._get_component_constraints(comp_cells)
                knowns = {}
                fixed_mines = 0
            else:
                # Heavy Path: Use Gaussian to reduce variables
                reduced_cells, reduced_cons, knowns, fixed_mines = \
                    self._simplify_component_gaussian(comp_cells)

            # Apply knowns immediately (if marking)
            if mark and knowns:
                for c, val in knowns.items():
                    if val == 0: self.mark_safe(c)
                    elif val == 1: self.mark_mine(c)
            
            # Solve remaining variables
            n_rem = len(reduced_cells)
            if n_rem == 0:
                res = {fixed_mines: (1, {})}
            elif n_rem <= self.MAX_EXACT_VAR:
                res = self._solve_component_exact(reduced_cells, reduced_cons, fixed_mines)
            else:
                res = self._solve_component_mc(reduced_cells, reduced_cons, fixed_mines, self.MC_SAMPLES)
            
            # Re-inject knowns into results
            final_res = {}
            for m_count, (w, c_counts) in res.items():
                for c, val in knowns.items():
                    if val == 1: c_counts[c] = w
                final_res[m_count] = (w, c_counts)
            
            comp_results.append(final_res)

        # 3. Convolution
        total_rem_mines = self.total_mines - len(self.mines)
        if total_rem_mines < 0: total_rem_mines = 0

        probs, meta = self._combine_components_convolution(comp_results, len(outside), total_rem_mines)

        # Fallback: If convolution failed (inconsistent state), return uniform probs
        # This prevents the "All 0" bug.
        if not probs:
            if len(unknown) > 0:
                uniform_p = total_rem_mines / len(unknown)
                # Clamp
                uniform_p = max(0.0, min(1.0, uniform_p))
                probs = {c: uniform_p for c in unknown}
                probs["outside"] = uniform_p
                meta = "fallback-uniform"
            else:
                return {}, "empty"

        # 4. Finalize
        p_out = probs.get("outside", 0.0)
        final_probs = {}
        for c in frontier:
            final_probs[c] = probs.get(c, 0.0)
        for c in outside:
            final_probs[c] = p_out

        # Auto-mark
        VERY_LOW = 0.002
        if mark:
            for c, pc in final_probs.items():
                if pc <= VERY_LOW: self.mark_safe(c)
                elif pc >= 1.0 - VERY_LOW: self.mark_mine(c)

        return final_probs, f"hybrid({len(components)}) {meta}"

    def _find_components(self, frontier):
        if not frontier: return []
        cell_to_idx = defaultdict(list)
        frontier_list = list(frontier)
        active_sentences = []
        for s in self.knowledge:
            if not s.cells: continue
            rel = s.cells.intersection(frontier)
            if rel:
                idx = len(active_sentences)
                active_sentences.append(rel)
                for c in rel: cell_to_idx[c].append(idx)
        
        visited = set()
        components = []
        for start in frontier_list:
            if start in visited: continue
            comp = set()
            q = [start]; visited.add(start)
            while q:
                curr = q.pop()
                comp.add(curr)
                for s_idx in cell_to_idx[curr]:
                    for neighbor in active_sentences[s_idx]:
                        if neighbor not in visited:
                            visited.add(neighbor); q.append(neighbor)
            components.append(comp)
        return components

    def _simplify_component_gaussian(self, cells):
        """
        Performs Gaussian Elimination on the component constraints.
        Returns: (remaining_cells, reduced_constraints, known_dict, fixed_mines_count)
        """
        cells_list = list(cells)
        n = len(cells_list)
        c_to_idx = {c: i for i, c in enumerate(cells_list)}
        
        # Build Matrix: Rows are equations. Cols are variables. Last col is value.
        # We only take constraints fully inside the component.
        matrix = []
        vars_set = set(cells)
        for s in self.knowledge:
            if s.cells and s.cells.issubset(vars_set):
                row = [0] * (n + 1)
                for c in s.cells:
                    row[c_to_idx[c]] = 1
                row[n] = s.count
                matrix.append(row)
        
        if not matrix:
            return cells_list, [], {}, 0

        # Gaussian Elimination (Forward)
        rows = len(matrix)
        pivot_row = 0
        for col in range(n):
            if pivot_row >= rows: break
            # Find pivot
            pivot = -1
            for r in range(pivot_row, rows):
                if matrix[r][col] != 0:
                    pivot = r; break
            if pivot == -1: continue
            
            # Swap
            matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
            
            # Eliminate
            curr = matrix[pivot_row]
            # Assume coefficients are 1 for Minesweeper usually, but let's be generic
            # For pure Python speed, we avoid division if possible, but subtraction is fine.
            # Since we only have 0/1 vars, standard subtraction works.
            for r in range(rows):
                if r != pivot_row and matrix[r][col] != 0:
                    factor = matrix[r][col] # Should be 1
                    target = matrix[r]
                    # row[r] = row[r] - factor * row[pivot]
                    for k in range(col, n + 1):
                        target[k] -= factor * curr[k]
            pivot_row += 1

        # Extract Knowns
        knowns = {} # cell -> 0 or 1
        fixed_mines = 0
        
        # Analyze rows for singletons or obvious contradictions
        # Also clean up the matrix rows to be valid constraints
        new_cons_data = []
        
        # Reverse pass isn't strictly needed for "identification", 
        # but inspecting rows where sum(abs(coefs)) == abs(val) helps.
        # Simple heuristic: If a row has 1 var with coeff C and val V => var = V/C.
        
        for row in matrix:
            # Check for single variable
            non_zeros = []
            val = row[n]
            for i in range(n):
                if row[i] != 0: non_zeros.append((i, row[i]))
            
            if not non_zeros: continue # 0 = 0 or 0 = non-zero (contradiction)
            
            if len(non_zeros) == 1:
                idx, coef = non_zeros[0]
                # In minesweeper logic, var must be 0 or 1.
                # If coef*x = val, then x = val/coef.
                if coef != 0 and val % coef == 0:
                    res = val // coef
                    if res == 0 or res == 1:
                        c = cells_list[idx]
                        if c not in knowns:
                            knowns[c] = res
                            if res == 1: fixed_mines += 1
            else:
                # Add to new constraints if valid
                # We only want positive constraints for the backtracker?
                # Actually, the backtracker expects subset sums. 
                # The Gaussian matrix might produce x1 - x2 = 0.
                # Transforming general linear eq to subset sum is hard.
                # STRATEGY: We ONLY use Gaussian to find knowns. 
                # We DO NOT pass the reduced matrix to the backtracker.
                # We pass the ORIGINAL constraints, filtered by knowns.
                pass

        # Apply knowns to simplify original constraints
        remaining_cells = [c for c in cells_list if c not in knowns]
        rem_set = set(remaining_cells)
        rem_idx = {c: i for i, c in enumerate(remaining_cells)}
        
        reduced_cons = []
        for s in self.knowledge:
            if not s.cells: continue
            # Only consider constraints relevant to this component
            if not s.cells.intersection(vars_set): continue
            
            # Filter knowns
            current_count = s.count
            current_vars = []
            relevant = False
            valid = True
            
            for c in s.cells:
                if c in knowns:
                    current_count -= knowns[c]
                elif c in rem_set:
                    current_vars.append(rem_idx[c])
                    relevant = True
            
            if valid and relevant and 0 <= current_count <= len(current_vars):
                # Avoid duplicates
                cons_entry = (tuple(sorted(current_vars)), current_count)
                reduced_cons.append(cons_entry)
        
        # Dedup constraints
        reduced_cons = sorted(list(set(reduced_cons)))
        
        return remaining_cells, reduced_cons, knowns, fixed_mines

    def _solve_component_exact(self, cells, cons, base_mines):
        n = len(cells)
        # v2c map
        v2c = [[] for _ in range(n)]
        for ci, (vs, t) in enumerate(cons):
            for v in vs: v2c[v].append(ci)
        
        need = [t for _, t in cons]
        remv = [len(vs) for vs, _ in cons]
        assign = [0]*n
        
        order = sorted(range(n), key=lambda v: len(v2c[v]), reverse=True)
        
        total_ways = defaultdict(int)
        cell_ways = defaultdict(lambda: [0]*n)

        def bt(k, mF):
            if k == n:
                for x in need: 
                    if x!=0: return
                total_ways[mF + base_mines] += 1
                cw = cell_ways[mF + base_mines]
                for i in range(n):
                    if assign[i]: cw[i] += 1
                return

            v = order[k]
            # Try 0
            ok=True
            for ci in v2c[v]:
                if need[ci] > remv[ci]-1: ok=False; break
            if ok:
                assign[v]=0
                for ci in v2c[v]: remv[ci]-=1
                bt(k+1, mF)
                for ci in v2c[v]: remv[ci]+=1
            
            # Try 1
            ok=True
            for ci in v2c[v]:
                if need[ci]-1 < 0: ok=False; break
            if ok:
                assign[v]=1
                for ci in v2c[v]: remv[ci]-=1; need[ci]-=1
                bt(k+1, mF+1)
                for ci in v2c[v]: remv[ci]+=1; need[ci]+=1
        
        bt(0,0)
        
        output = {}
        for m, w in total_ways.items():
            cw = {}
            for i in range(n):
                if cell_ways[m][i]>0: cw[cells[i]] = cell_ways[m][i]
            output[m] = (w, cw)
        return output

    def _solve_component_mc(self, cells, cons, base_mines, samples):
        n = len(cells)
        v2c = [[] for _ in range(n)]
        for ci, (vs, t) in enumerate(cons):
            for v in vs: v2c[v].append(ci)
        
        order = sorted(range(n), key=lambda v: len(v2c[v]), reverse=True)
        rnd = random.random
        
        total_ways = defaultdict(float)
        cell_ways = defaultdict(lambda: [0.0]*n)
        
        got = 0; trials = 0; cap = samples * 5
        
        while got < samples and trials < cap:
            trials += 1
            need = [t for _, t in cons]
            remv = [len(vs) for vs, _ in cons]
            assign = [0]*n
            mF = 0; q = 1.0; ok_all = True
            
            for v in order:
                vc = v2c[v]
                ok0 = True
                for ci in vc:
                    if need[ci] > remv[ci]-1: ok0=False; break
                ok1 = True
                for ci in vc:
                    if need[ci]-1 < 0: ok1=False; break
                
                if not ok0 and not ok1: ok_all=False; break
                
                if ok0 and ok1:
                    q *= 0.5
                    val = 1 if rnd() < 0.5 else 0
                else:
                    val = 1 if ok1 else 0
                
                assign[v] = val; mF += val
                if val:
                    for ci in vc: remv[ci]-=1; need[ci]-=1
                else:
                    for ci in vc: remv[ci]-=1
            
            if not ok_all or any(x!=0 for x in need): continue
            
            iw = 1.0 / q
            total_ways[mF + base_mines] += iw
            cw = cell_ways[mF + base_mines]
            for i in range(n):
                if assign[i]: cw[i] += iw
            got += 1
            
        output = {}
        for m, w in total_ways.items():
            if w <= 0: continue
            cw = {}
            for i in range(n):
                if cell_ways[m][i]>0: cw[cells[i]] = cell_ways[m][i]
            output[m] = (w, cw)
        return output
    def _combine_components_convolution(self, comp_results, n_outside, total_mines):
        dists = []
        for res in comp_results:
            dists.append({k: w for k, (w, _) in res.items()})
        
        # Convolve components
        comps_dist = {0: 1}
        for d in dists:
            new_dist = defaultdict(int)
            for k1, w1 in comps_dist.items():
                for k2, w2 in d.items():
                    if k1 + k2 <= total_mines:
                        new_dist[k1 + k2] += w1 * w2
            comps_dist = new_dist
            
        # Add outside
        comb = math.comb
        Z = 0.0
        
        for k_front, w_front in comps_dist.items():
            rem = total_mines - k_front
            if 0 <= rem <= n_outside:
                w_out = comb(n_outside, rem)
                Z += w_front * w_out
        
        # Fix: If Z is 0 (inconsistent), return empty immediately
        if Z <= 1e-4: return {}, "inconsistent"

        final_probs = {}
        
        # Calc frontier probs
        for i, res in enumerate(comp_results):
            # Convolve Rest
            rest_dist = {0: 1}
            for j, d in enumerate(dists):
                if i == j: continue
                new_dist = defaultdict(int)
                for k1, w1 in rest_dist.items():
                    for k2, w2 in d.items():
                        if k1 + k2 <= total_mines:
                            new_dist[k1 + k2] += w1 * w2
                rest_dist = new_dist
            
            for k_local, (w_local, cell_counts) in res.items():
                if w_local == 0: continue
                ways_rest_out = 0.0
                target = total_mines - k_local
                for k_rest, w_rest in rest_dist.items():
                    k_out = target - k_rest
                    if 0 <= k_out <= n_outside:
                        ways_rest_out += w_rest * comb(n_outside, k_out)
                
                if ways_rest_out == 0: continue
                factor = ways_rest_out / Z
                for cell, wc in cell_counts.items():
                    final_probs[cell] = final_probs.get(cell, 0.0) + wc * factor
        
        # Calc outside prob
        avg_out = 0.0
        for k_front, w_front in comps_dist.items():
            k_out = total_mines - k_front
            if 0 <= k_out <= n_outside:
                w_out = comb(n_outside, k_out)
                prob_split = (w_front * w_out) / Z
                avg_out += k_out * prob_split
        
        final_probs["outside"] = avg_out / n_outside if n_outside > 0 else 0.0
        return final_probs, "exact-conv"

    def _get_component_constraints(self, cells):
        """
        Fast extraction of constraints for a component without Gaussian overhead.
        """
        cells_list = list(cells)
        c_to_idx = {c: i for i, c in enumerate(cells_list)}
        vars_set = set(cells)
        
        cons = []
        for s in self.knowledge:
            if s.cells and s.cells.issubset(vars_set):
                # Convert cell coordinates to indices
                idxs = tuple(sorted(c_to_idx[c] for c in s.cells))
                cons.append((idxs, s.count))
        # Dedup
        return sorted(list(set(cons)))

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
# Browser API (Pyodide)
# =========================
_GAME=None; _AI=None; _REVEALED=set(); _REVEALED_N={}
_SEED = None
_LOST=_WON=_FIRST=False; _H=_W=_M=0; _FIRST_MV_MODE=1
def _to_list_cell(cell): return [int(cell[0]),int(cell[1])]

def ms_new_game(h,w,mines,seed=None, firstmv=1):
    global _GAME,_AI,_REVEALED,_REVEALED_N,_LOST,_WON,_FIRST,_H,_W,_M, _FIRST_MV_MODE, _SEED
    _REVEALED=set(); _REVEALED_N={}; _LOST=_WON=_FIRST=False
    print(f"[PY DEBUG] Received seed: {seed}")  # <-- 新增
    if seed is not None:
        try: seed = (int(seed) & 0xFFFFFFFF); 
        except: seed=None
    if seed is None:
        import time; seed = int(time.time() * 492357816) % 2147483647;
    print(f"[PY DEBUG] Seed used: {seed}")  # <-- 新增
    _H,_W,_M=int(h),int(w),int(mines); _FIRST_MV_MODE = int(firstmv)
    _GAME=Minesweeper(_H,_W,_M); _AI=MinesweeperAI(_H,_W,_M)
    _SEED = seed; state = ms_get_state(); state["seed"] = seed
    return state

def ms_get_state():
    global _GAME, _AI, _SEED
    return {
        "h": _H, "w": _W, "mines": _M,
        "first": bool(_FIRST), "firstmv": int(_FIRST_MV_MODE),
        "mines_pos": [_to_list_cell(c) for c in (_GAME.mines if _GAME else set())],
        "revealed": [[r, c, int(_REVEALED_N.get((r, c), 0))] for (r, c) in _REVEALED],
        "ai_mines": [_to_list_cell(c) for c in (_AI.mines if _AI else set())],
        "ai_moves": [_to_list_cell(c) for c in (_AI.moves_made if _AI else set())],
        "ai_safes": [_to_list_cell(c) for c in (_AI.safes if _AI else set())],
        "lost": bool(_LOST), "won": bool(_WON), "seed": _SEED, "revealed_count": len(_REVEALED),
    }

def ms_set_state(st):
    global _GAME, _AI, _REVEALED, _REVEALED_N, _LOST, _WON, _FIRST, _H, _W, _M, _FIRST_MV_MODE, _SEED
    if st is None: raise ValueError("st is None")
    if hasattr(st, "to_py"): st = st.to_py()
    elif not isinstance(st, dict): st = dict(st)
    _H=int(st["h"]); _W=int(st["w"]); _M=int(st["mines"])
    _LOST=bool(st.get("lost",False)); _WON=bool(st.get("won",False))
    _FIRST=bool(st.get("first",False)); _FIRST_MV_MODE=int(st.get("firstmv",1))
    _SEED=st.get("seed",None)
    _GAME = Minesweeper(_H, _W, _M); _GAME.first_move_made = bool(_FIRST)
    mines_pos = st.get("mines_pos")
    if mines_pos is None: raise ValueError("missing mines_pos")
    _GAME.mines = set(); _GAME.board = [[False] * _W for _ in range(_H)]
    for rr, cc in mines_pos:
        r=int(rr); c=int(cc); _GAME.mines.add((r, c)); _GAME.board[r][c] = True
    _REVEALED = set(); _REVEALED_N = {}
    for rr, cc, nn in (st.get("revealed") or []):
        r=int(rr); c=int(cc); n=int(nn); _REVEALED.add((r, c)); _REVEALED_N[(r, c)] = n
    _AI = MinesweeperAI(_H, _W, _M)
    for rr, cc in (st.get("ai_mines") or []): _AI.mark_mine((int(rr), int(cc)))
    revealed_nums = {}
    for (r, c), n in _REVEALED_N.items():
        if n >= 0: revealed_nums[(r, c)] = n
    if revealed_nums: _AI.add_knowledge_batch(revealed_nums)
    return ms_get_state()

def ms_make_safe_move():
    global _GAME, _AI, _REVEALED, _REVEALED_N, _LOST, _WON, _FIRST, _H, _W, _M, _SEED
    if _GAME is None: return {"move":None,"newly":[],"ai_mines":[],"lost":False,"won":False,"stuck":False,"revealed_count":0}
    mv = _AI.make_safe_move()
    if mv is None: return {"move":None,"newly":[],"ai_mines":[c for c in _AI.mines],"lost":False,"won":False,"stuck":False,"revealed_count":len(_REVEALED)}
    if not _FIRST:
        if _FIRST_MV_MODE == 1: _GAME.make_safe_first_move(mv,seed=_SEED)
        elif _FIRST_MV_MODE == 2: _GAME.make_safe_first_move2(mv,seed=_SEED)
        _FIRST=True
    results=_GAME.reveal_chain(mv); newly=[]
    if any(v==-1 for v in results.values()): _LOST=True; _REVEALED.add(mv); _REVEALED_N[mv]=-1; newly.append([mv[0],mv[1],-1])
    else:
        _AI.add_knowledge_batch(results)
        for (r,c),n in results.items():
            if (r,c) not in _REVEALED: _REVEALED.add((r,c)); _REVEALED_N[(r,c)]=int(n); newly.append([r,c,int(n)])
    if len(_REVEALED)==_H*_W-_M: _WON=True
    return {"move":[mv[0],mv[1]],"newly":newly,"ai_mines":[c for c in _AI.mines],"lost":bool(_LOST),"won":bool(_WON),"stuck":False,"revealed_count":len(_REVEALED)}

def ms_get_analysis():
    global _AI
    if _AI is None: return {"probs": {}, "next_move": None}
    pd_tuple_keys, _ = _AI._mine_probabilities(mark=False)
    pd_str_keys = {f"({r},{c})": prob for (r, c), prob in pd_tuple_keys.items()}
    nm = _AI.make_random_move()
    return {"probs": pd_str_keys, "next_move": list(nm) if nm else None}

def ms_step():
    global _GAME,_AI,_REVEALED,_REVEALED_N,_LOST,_WON,_FIRST,_H,_W,_M,_SEED
    if _GAME is None: return {"move":None,"newly":[],"ai_mines":[],"lost":False,"won":False,"stuck":True,"revealed_count":0}
    mv=_AI.make_random_move()
    if mv is None: return {"move":None,"newly":[],"ai_mines":[c for c in _AI.mines],"lost":False,"won":False,"stuck":True,"revealed_count":len(_REVEALED)}
    if not _FIRST:
        if _FIRST_MV_MODE == 1: _GAME.make_safe_first_move(mv,seed=_SEED)
        elif _FIRST_MV_MODE == 2: _GAME.make_safe_first_move2(mv,seed=_SEED)
        _FIRST=True
    results=_GAME.reveal_chain(mv); newly=[]
    if any(v==-1 for v in results.values()): 
        _LOST=True; _REVEALED.add(mv); _REVEALED_N[mv]=-1
        return {"move":[mv[0],mv[1]],"newly":[[mv[0],mv[1],-1]],"ai_mines":[c for c in _AI.mines],"lost":True,"won":False,"stuck":False,"revealed_count":len(_REVEALED)}
    _AI.add_knowledge_batch(results)
    for (r,c),n in results.items():
        if (r,c) not in _REVEALED: 
            _REVEALED.add((r,c)); _REVEALED_N[(r,c)]=int(n); newly.append([r,c,int(n)])
    if len(_REVEALED)==_H*_W-_M: _WON=True
    return {"move":[mv[0],mv[1]],"newly":newly,"ai_mines":[c for c in _AI.mines],"lost":bool(_LOST),"won":bool(_WON),"stuck":False,"revealed_count":len(_REVEALED)}

def ms_step_at(r, c):
    global _GAME, _AI, _REVEALED, _REVEALED_N, _LOST, _WON, _FIRST, _H, _W, _M
    if _GAME is None: return {"move": None, "newly": [], "ai_mines": [], "lost": False, "won": False, "stuck": True, "revealed_count": 0}
    mv = (int(r), int(c))
    if not _FIRST:
        if _FIRST_MV_MODE == 1: _GAME.make_safe_first_move(mv,seed=_SEED)
        elif _FIRST_MV_MODE == 2: _GAME.make_safe_first_move2(mv,seed=_SEED)
        _FIRST = True
    results = _GAME.reveal_chain(mv); newly = []
    if any(v == -1 for v in results.values()):
        _LOST = True; _REVEALED.add(mv); _REVEALED_N[mv] = -1; newly.append([mv[0], mv[1], -1])
    else:
        _AI.add_knowledge_batch(results)
        for (rr, cc), n in results.items():
            if (rr, cc) not in _REVEALED:
                _REVEALED.add((rr, cc)); _REVEALED_N[(rr, cc)] = int(n); newly.append([rr, cc, int(n)])
    if len(_REVEALED) == _H * _W - _M: _WON = True
    return {"move": [mv[0], mv[1]], "newly": newly, "ai_mines": [list(cell) for cell in _AI.mines], "lost": bool(_LOST), "won": bool(_WON), "stuck": False, "revealed_count": len(_REVEALED)}

def ms_load_board(data):
    global _GAME,_AI,_REVEALED,_REVEALED_N,_LOST,_WON,_FIRST,_H,_W,_M,_SEED
    
    # 处理可能的序列化字符串输入
    if isinstance(data, str):
        import json
        data = json.loads(data)
    
    _REVEALED=set(); _REVEALED_N={}; _LOST=_WON=_FIRST=False
    _H,_W,_M=int(data["height"]),int(data["width"]),int(data["mines"])
    _SEED=data.get("seed")
    if _SEED is not None: random.seed(_SEED)
    _GAME=Minesweeper(_H,_W,_M); _AI=MinesweeperAI(_H,_W,_M)
    _GAME.first_move_made=bool(data.get("first_move_made", True))
    
    # 处理 field
    field=data["field"]
    if hasattr(field, 'to_py'):
        field = [str(row) for row in field.to_py()]
    else:
        field = [str(row) for row in field]
    
    assert len(field)==_H and all(len(row)==_W for row in field)
    
    # 处理可见信息
    nums={}
    for r,row in enumerate(field):
        for c,ch in enumerate(row):
            cell=(r,c)
            if ch=='F': 
                _AI.mark_mine(cell)
            elif ch.isdigit(): 
                _REVEALED.add(cell); _REVEALED_N[cell]=int(ch); nums[cell]=int(ch)
    
    # 加载真实雷区布局（如果提供）
    if "mines_layout" in data:
        mines_layout = data["mines_layout"]
        # 安全处理嵌套数组
        try:
            if hasattr(mines_layout, 'to_py'):
                mines_layout = mines_layout.to_py()
            
            layout_list = list(mines_layout) if hasattr(mines_layout, '__iter__') else mines_layout
            
            for r, row_data in enumerate(layout_list):
                # 处理每一行
                if hasattr(row_data, 'to_py'):
                    row = list(row_data.to_py())
                else:
                    row = list(row_data) if hasattr(row_data, '__iter__') else row_data
                
                for c, has_mine_value in enumerate(row):
                    has_mine = int(has_mine_value) if hasattr(has_mine_value, '__int__') else int(str(has_mine_value))
                    if has_mine == 1:
                        cell = (r, c)
                        _GAME.mines.add(cell)
                        _GAME.board[r][c] = True
        except Exception as e:
            print(f"Warning: Failed to load mines_layout: {e}")
    
    if nums: _AI.add_knowledge_batch(nums)
    return ms_get_state()

def ms_board_info():
    global _GAME,_AI,_REVEALED,_REVEALED_N,_H,_W,_M,_SEED
    if _GAME is None or _H<=0 or _W<=0: return {"error":"No active game"}
    
    # 构建可见字段信息
    field=[['H']*_W for _ in range(_H)]
    for r,c in _AI.mines:
        if 0<=r<_H and 0<=c<_W: field[r][c]='F'
    for (r,c),n in _REVEALED_N.items():
        if 0<=r<_H and 0<=c<_W: field[r][c]=str(n)
    
    # 构建真实雷区布局
    mines_layout = []
    for r in range(_H):
        row = []
        for c in range(_W):
            has_mine = 1 if _GAME.board[r][c] else 0
            row.append(has_mine)
        mines_layout.append(row)
    
    return {
        "height": _H,
        "width": _W, 
        "mines": _M,
        "seed": _SEED,
        "first_move_made": _GAME.first_move_made,
        "field": [''.join(row) for row in field],
        "mines_layout": mines_layout
    }

# =========================
# Example runner (optional)
# =========================
def _play_one(h=16,w=30,mines=99,seed=None, firstmv=1):
    if seed is not None: random.seed(seed)
    game=Minesweeper(h,w,mines); ai=MinesweeperAI(h,w,mines)
    first=(h//2,w//2);
    # --- Modified section ---
    if firstmv == 1:
        game.make_safe_first_move(first,seed=seed)
    elif firstmv == 2:
        game.make_safe_first_move2(first,seed=seed)
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


"""
{
  "height": int,
  "width": int, 
  "mines": int,
  "seed": int or null,
  "first_move_made": bool,
  "field": ["HHF...", "..."],     // 可见信息：H/F/0-8
  "mines_layout": [[0,1,0,...], [...]]  // 真实雷区：0=无雷, 1=有雷
}

"""