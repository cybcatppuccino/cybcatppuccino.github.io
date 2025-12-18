import random, math
from collections import deque, defaultdict

# -------------------------
# Game
# -------------------------

class Minesweeper:
    def __init__(self, height=8, width=8, mines=8):
        self.height, self.width = height, width
        self.mines = set()
        self.mines_count = mines
        self.board = [[False] * width for _ in range(height)]
        self.mines_found = set()
        self.first_move_made = False

    def make_safe_first_move(self, cell):
        if self.first_move_made: return
        self.first_move_made = True

        safe = {cell}
        r0, c0 = cell
        for r in range(r0-1, r0+2):
            for c in range(c0-1, c0+2):
                if 0 <= r < self.height and 0 <= c < self.width:
                    safe.add((r, c))

        cap = self.height * self.width - len(safe)
        if self.mines_count > cap:
            raise ValueError(f"Too many mines for safe-first-move zone: mines={self.mines_count}, cap={cap}")

        while len(self.mines) < self.mines_count:
            r, c = random.randrange(self.height), random.randrange(self.width)
            if (r, c) in safe or self.board[r][c]: 
                continue
            self.mines.add((r, c))
            self.board[r][c] = True

    def is_mine(self, cell): 
        r, c = cell
        return self.board[r][c]

    def nearby_mines(self, cell):
        r0, c0 = cell
        cnt = 0
        for r in range(r0-1, r0+2):
            for c in range(c0-1, c0+2):
                if (r, c) == cell: 
                    continue
                if 0 <= r < self.height and 0 <= c < self.width and self.board[r][c]:
                    cnt += 1
        return cnt

    def reveal_chain(self, cell):
        """BFS: 点到0展开周围安全格。返回 {cell: number}；如果点到雷 => {cell:-1}
           [FIX] 展开过程中不会把雷加入队列/结果
        """
        if self.is_mine(cell): 
            return {cell: -1}

        def neigh(x):
            r0, c0 = x
            for r in range(r0-1, r0+2):
                for c in range(c0-1, c0+2):
                    if (r, c) != x and 0 <= r < self.height and 0 <= c < self.width:
                        yield (r, c)

        res, q, vis = {}, deque([cell]), {cell}
        while q:
            cur = q.popleft()
            if self.is_mine(cur):   # 防御：理论上不会发生
                continue
            n = self.nearby_mines(cur)
            res[cur] = n
            if n == 0:
                for nb in neigh(cur):
                    if nb in vis: 
                        continue
                    vis.add(nb)
                    if not self.is_mine(nb):   # [FIX] 不扩展雷
                        q.append(nb)
        return res

    def won(self): 
        return self.mines_found == self.mines


# -------------------------
# Logic sentence
# -------------------------

class Sentence:
    def __init__(self, cells, count):
        self.cells, self.count = set(cells), count

    def __eq__(self, other): 
        return self.cells == other.cells and self.count == other.count

    def __str__(self): 
        return f"{self.cells} = {self.count}"

    def known_mines(self): 
        return set(self.cells) if self.count == len(self.cells) and self.count > 0 else set()

    def known_safes(self): 
        return set(self.cells) if self.count == 0 else set()

    def mark_mine(self, cell):
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        if cell in self.cells:
            self.cells.remove(cell)


# -------------------------
# AI
# -------------------------

class MinesweeperAI:
    def __init__(self, height=8, width=8, mines=15):
        self.height, self.width, self.total_mines = height, width, mines
        self.moves_made, self.mines, self.safes = set(), set(), set()
        self.knowledge = []

        # 针对 16x30/99：更偏向 MC（分量很大），小分量精确
        self.MAX_EXACT_COMP = 18
        self.MC_BASE = 2500
        self.MC_PER_VAR = 120
        self.MC_CAP = 14000

    # ---- marks ----
    def mark_mine(self, cell):
        if cell in self.mines: 
            return
        self.mines.add(cell)
        for s in self.knowledge: 
            s.mark_mine(cell)

    def mark_safe(self, cell):
        if cell in self.safes: 
            return
        self.safes.add(cell)
        for s in self.knowledge: 
            s.mark_safe(cell)

    # ---- helpers ----
    def _neighbors(self, cell):
        r0, c0 = cell
        for r in range(r0-1, r0+2):
            for c in range(c0-1, c0+2):
                if (r, c) != cell and 0 <= r < self.height and 0 <= c < self.width:
                    yield (r, c)

    def _all_cells(self):
        return {(r, c) for r in range(self.height) for c in range(self.width)}

    # ---- knowledge ----
    def add_knowledge_batch(self, revealed_dict):
        for cell, count in revealed_dict.items():
            if count < 0:
                continue
            if cell in self.moves_made:
                continue
            self.moves_made.add(cell)
            self.mark_safe(cell)

            neigh = set(self._neighbors(cell))
            unknown, adj_mines = set(), 0
            for n in neigh:
                if n in self.mines: 
                    adj_mines += 1
                elif n not in self.safes and n not in self.moves_made:
                    unknown.add(n)

            newc = count - adj_mines
            if unknown and 0 <= newc <= len(unknown):
                sen = Sentence(unknown, newc)
                if sen not in self.knowledge:
                    self.knowledge.append(sen)

        self.update_knowledge()
        self.infer_new_sentences()

    def update_knowledge(self):
        changed = True
        while changed:
            changed = False

            mines, safes = set(), set()
            for s in self.knowledge:
                mines |= s.known_mines()
                safes |= s.known_safes()

            for m in mines:
                if m not in self.mines:
                    self.mark_mine(m); changed = True
            for sf in safes:
                if sf not in self.safes:
                    self.mark_safe(sf); changed = True

            # clean
            nk = []
            for s in self.knowledge:
                if not s.cells: 
                    continue
                if 0 <= s.count <= len(s.cells):
                    nk.append(s)
            self.knowledge = nk

    def infer_new_sentences(self):
        changed = True
        while changed:
            changed = False
            self.knowledge.sort(key=lambda s: len(s.cells))
            new = []
            for i, a in enumerate(self.knowledge):
                for b in self.knowledge[i+1:]:
                    if a.cells.issubset(b.cells):
                        diff = b.cells - a.cells
                        dc = b.count - a.count
                        if diff and 0 <= dc <= len(diff):
                            ns = Sentence(diff, dc)
                            if ns not in self.knowledge and ns not in new:
                                new.append(ns)
                                changed = True
            if new:
                self.knowledge.extend(new)
                self.update_knowledge()

    # ---- moves ----
    def make_safe_move(self):
        avail = sorted(self.safes - self.moves_made)
        return avail[0] if avail else None

    # [接口保留] 原名 make_random_move，但现在是“最优选择”
    def make_random_move(self):
        # 1) 先走必然安全
        mv = self.make_safe_move()
        if mv is not None:
            print(f"[AI] Safe move: {mv}")
            return mv

        # 2) 概率评估后选择
        probs, meta = self._mine_probabilities()
        if not probs:
            return None

        # 先拿 P=0
        zeros = [c for c, p in probs.items() if p == 0.0 and c not in self.moves_made and c not in self.mines]
        if zeros:
            mv = sorted(zeros)[0]
            print(f"[AI] Prob=0 move: {mv}")
            return mv

        # 最小概率，tie-break 信息增益
        candidates = [(p, c) for c, p in probs.items() if c not in self.moves_made and c not in self.mines]
        if not candidates:
            return None
        candidates.sort()
        best_p = candidates[0][0]
        best = [c for p, c in candidates if abs(p - best_p) < 1e-12]

        if len(best) == 1:
            mv = best[0]
            print(f"[AI] Choose minP: {mv}, P={best_p:.4f} | {meta}")
            return mv

        # 快速信息增益启发（不 deepcopy）
        scored = []
        for c in best:
            gain = self._info_gain_heuristic(c)
            scored.append((gain, c))
        scored.sort(reverse=True)
        mv = scored[0][1]
        print(f"[AI] Choose tie(minP={best_p:.4f}) by gain: {mv} gain={scored[0][0]} | {meta}")
        return mv

    def _info_gain_heuristic(self, cell):
        # 越“贴近数字约束/边界”越可能带来新信息；同时 prefer 周围未开多（潜在展开）
        unk = 0
        for nb in self._neighbors(cell):
            if nb not in self.moves_made and nb not in self.safes and nb not in self.mines:
                unk += 1
        deg = 0
        for s in self.knowledge:
            if cell in s.cells:
                deg += 1
        return 3 * deg + unk

    # ---- probability core ----
    def _mine_probabilities(self):
        all_cells = self._all_cells()
        unknown = list(all_cells - self.moves_made - self.safes - self.mines)
        if not unknown:
            return {}, "no-unknown"
        rem_mines = max(0, self.total_mines - len(self.mines))

        # frontier
        frontier = set()
        for s in self.knowledge:
            frontier |= s.cells
        frontier &= set(unknown)
        uninformed = [c for c in unknown if c not in frontier]

        if not frontier:
            p = rem_mines / len(unknown)
            print(f"[AI] No constraints. Uniform P={p:.4f} over {len(unknown)} cells")
            return {c: p for c in unknown}, f"uniform({len(unknown)})"

        comps = self._frontier_components(frontier)
        print(f"[AI] unknown={len(unknown)} frontier={len(frontier)} uninformed={len(uninformed)} comps={len(comps)} remM={rem_mines}")

        infos = []
        for comp in comps:
            sents = self._sentences_restricted(comp)
            n = len(comp)
            if n <= self.MAX_EXACT_COMP:
                info = self._enum_component_exact(comp, sents)
                infos.append(info)
                print(f"[AI]  comp size={n}: exact, distK={len(info['dist'])}")
            else:
                samples = min(self.MC_CAP, self.MC_BASE + self.MC_PER_VAR * n)
                info = self._enum_component_mc(comp, sents, samples)
                infos.append(info)
                print(f"[AI]  comp size={n}: MC samples={info['got']}/{samples}, distK={len(info['dist'])}")

        probs_frontier, e_front = self._combine_components(infos, rem_mines, len(uninformed))
        probs = dict(probs_frontier)

        if uninformed:
            e_out = max(0.0, rem_mines - e_front)
            p_out = min(1.0, max(0.0, e_out / len(uninformed)))
            for c in uninformed:
                probs[c] = p_out

        # 若出现确定 0/1，可直接标注（exact 组合时才真正可靠；MC 可能抖动，但 0/1 极少出现）
        for c, p in probs.items():
            if p == 0.0: self.mark_safe(c)
            elif p == 1.0: self.mark_mine(c)

        meta = f"frontierE={e_front:.2f}"
        return probs, meta

    def _frontier_components(self, frontier):
        # cell graph: connect if appear in same sentence
        adj = {c: set() for c in frontier}
        for s in self.knowledge:
            cells = list(s.cells & frontier)
            for i in range(len(cells)):
                a = cells[i]
                for j in range(i+1, len(cells)):
                    b = cells[j]
                    adj[a].add(b); adj[b].add(a)

        seen, comps = set(), []
        for c in frontier:
            if c in seen: 
                continue
            stack, comp = [c], set([c])
            seen.add(c)
            while stack:
                x = stack.pop()
                for y in adj.get(x, ()):
                    if y not in seen:
                        seen.add(y)
                        comp.add(y)
                        stack.append(y)
            comps.append(comp)
        return comps

    def _sentences_restricted(self, comp):
        out = []
        for s in self.knowledge:
            inter = s.cells & comp
            if inter:
                out.append(Sentence(inter, s.count))
        return out

    # ---- exact enum (small comp) ----
    def _enum_component_exact(self, comp_cells, sents):
        cells = list(comp_cells)
        idx = {c:i for i,c in enumerate(cells)}
        cons = []
        for s in sents:
            vs = [idx[c] for c in s.cells if c in idx]
            if vs: cons.append((vs, s.count))

        n = len(cells)
        # var->constraints for faster updates
        v2c = [[] for _ in range(n)]
        for ci, (vs, t) in enumerate(cons):
            for v in vs: v2c[v].append(ci)

        need = [t for _, t in cons]          # remaining mines needed in constraint
        rem  = [len(vs) for vs, _ in cons]   # remaining vars in constraint
        assign = [-1]*n

        # order: most constrained first
        order = sorted(range(n), key=lambda v: len(v2c[v]), reverse=True)

        dist = defaultdict(int)
        cell_mine_dist = {c: defaultdict(int) for c in cells}

        def apply(v, val):
            assign[v] = val
            for ci in v2c[v]:
                rem[ci] -= 1
                need[ci] -= val

        def undo(v, val):
            assign[v] = -1
            for ci in v2c[v]:
                rem[ci] += 1
                need[ci] += val

        def ok_constraints(v, val):
            for ci in v2c[v]:
                r = rem[ci] - 1
                nd = need[ci] - val
                if nd < 0 or nd > r:
                    return False
            return True

        def bt(k, mines_used):
            if k == n:
                for ci in range(len(cons)):
                    if need[ci] != 0: 
                        return
                dist[mines_used] += 1
                for i, c in enumerate(cells):
                    if assign[i] == 1:
                        cell_mine_dist[c][mines_used] += 1
                return

            v = order[k]

            # try 0 then 1
            if ok_constraints(v, 0):
                apply(v, 0); bt(k+1, mines_used); undo(v, 0)
            if ok_constraints(v, 1):
                apply(v, 1); bt(k+1, mines_used+1); undo(v, 1)

        bt(0, 0)
        return {"cells": cells, "dist": dist, "cell_mine_dist": cell_mine_dist, "exact": True}

    # ---- MC enum (large comp) ----
    def _enum_component_mc(self, comp_cells, sents, samples):
        cells = list(comp_cells)
        idx = {c:i for i,c in enumerate(cells)}
        cons = []
        for s in sents:
            vs = [idx[c] for c in s.cells if c in idx]
            if vs: cons.append((vs, s.count))

        n = len(cells)
        v2c = [[] for _ in range(n)]
        for ci, (vs, t) in enumerate(cons):
            for v in vs: v2c[v].append(ci)

        # order: most constrained first
        order = sorted(range(n), key=lambda v: len(v2c[v]), reverse=True)

        dist = defaultdict(int)
        cell_mine_dist = {c: defaultdict(int) for c in cells}

        def one_sample():
            need = [t for _, t in cons]
            rem  = [len(vs) for vs, _ in cons]
            assign = [-1]*n
            mines_used = 0

            for v in order:
                # check feasibility for val=0/1 using only affected constraints
                ok0 = True
                for ci in v2c[v]:
                    r = rem[ci] - 1
                    nd = need[ci] - 0
                    if nd < 0 or nd > r: ok0 = False; break

                ok1 = True
                for ci in v2c[v]:
                    r = rem[ci] - 1
                    nd = need[ci] - 1
                    if nd < 0 or nd > r: ok1 = False; break

                if not ok0 and not ok1:
                    return None

                # 随机但带一点启发：若某值更“紧”，更倾向选能保持余量的那一边
                if ok0 and ok1:
                    # 简单余量评分：sum( (rem-need) ) 越大越松
                    slack0 = 0
                    slack1 = 0
                    for ci in v2c[v]:
                        r = rem[ci] - 1
                        slack0 += (r - (need[ci] - 0))
                        slack1 += (r - (need[ci] - 1))
                    p1 = 0.5
                    if slack0 + slack1 > 0:
                        p1 = max(0.15, min(0.85, slack1 / (slack0 + slack1)))
                    val = 1 if random.random() < p1 else 0
                else:
                    val = 1 if ok1 else 0

                assign[v] = val
                mines_used += val
                for ci in v2c[v]:
                    rem[ci] -= 1
                    need[ci] -= val

            # verify all satisfied
            for ci in range(len(cons)):
                if need[ci] != 0:
                    return None
            return assign, mines_used

        got = 0
        trials = 0
        cap_trials = samples * 40
        while got < samples and trials < cap_trials:
            trials += 1
            res = one_sample()
            if res is None:
                continue
            assign, k = res
            got += 1
            dist[k] += 1
            for i, c in enumerate(cells):
                if assign[i] == 1:
                    cell_mine_dist[c][k] += 1

        if got == 0:
            # fallback: give something non-empty
            dist[0] = 1

        return {"cells": cells, "dist": dist, "cell_mine_dist": cell_mine_dist, "exact": False, "got": got}

    # ---- global combine with total mines + outside comb ----
    def _combine_components(self, infos, rem_mines, outside_n):
        m = len(infos)
        dists = [info["dist"] for info in infos]

        # prefix DP
        pre = [defaultdict(int) for _ in range(m+1)]
        pre[0][0] = 1
        for i in range(m):
            for t, w in pre[i].items():
                for k, cnt in dists[i].items():
                    pre[i+1][t+k] += w * cnt

        # suffix DP
        suf = [defaultdict(int) for _ in range(m+1)]
        suf[m][0] = 1
        for i in range(m-1, -1, -1):
            for t, w in suf[i+1].items():
                for k, cnt in dists[i].items():
                    suf[i][t+k] += w * cnt

        totalW = 0
        e_front_num = 0
        for kf, ways in pre[m].items():
            out_need = rem_mines - kf
            if 0 <= out_need <= outside_n:
                w = ways * math.comb(outside_n, out_need)
                totalW += w
                e_front_num += w * kf

        if totalW == 0:
            # inconsistent: fallback uniform on frontier only
            front = sum(len(info["cells"]) for info in infos)
            p = rem_mines / max(1, front + outside_n)
            probs = {}
            for info in infos:
                for c in info["cells"]:
                    probs[c] = max(0.0, min(1.0, p))
            return probs, min(rem_mines, front) * p

        e_front = e_front_num / totalW
        probs = {}

        # compute each comp cell marginal with DP rest * comb(outside)
        for j, info in enumerate(infos):
            ways_rest = defaultdict(int)
            for a, wa in pre[j].items():
                for b, wb in suf[j+1].items():
                    ways_rest[a+b] += wa * wb

            distj = info["dist"]
            cmd = info["cell_mine_dist"]
            for cell in info["cells"]:
                num = 0
                for kj, cntk in distj.items():
                    cellmine = cmd[cell].get(kj, 0)
                    if not cellmine:
                        continue
                    for mr, wr in ways_rest.items():
                        kf = kj + mr
                        out_need = rem_mines - kf
                        if 0 <= out_need <= outside_n:
                            num += cellmine * wr * math.comb(outside_n, out_need)
                probs[cell] = num / totalW

        return probs, e_front

    # ---- compatibility methods (kept) ----
    def calculate_safe_cells_if_safe(self, cell):
        """[接口保留] 原先深拷贝推理很慢；这里改为快速启发式：返回一个“潜在安全增益”估计值"""
        if cell in self.safes: return 0
        if cell in self.mines: return 0
        # 简单：周围 unknown 多、且参与约束多 => 若安全可能带来更多推理
        return self._info_gain_heuristic(cell)

    def solve_frontier_lp(self, frontier_list, total_rem_mines):
        """[接口保留] 不再使用 LP；返回近似 min/max 以及 probs（后验估计）
           min/max 用当前分量 dist 的可行 k 范围近似
        """
        probs, _ = self._mine_probabilities()
        frontier = [c for c in frontier_list if c in probs]
        if not frontier:
            return 0, 0, {}

        # 用边界的概率和近似 min/max（保守：0..len(frontier)）
        pavg = sum(probs[c] for c in frontier) / len(frontier)
        min_m = 0
        max_m = min(len(frontier), total_rem_mines)
        # 你若依赖 min/max，可更紧：round(pavg*len(frontier))±something，这里保持保守
        return min_m, max_m, {c: probs[c] for c in frontier}


# -------------------------
# Example runner (optional)
# -------------------------

def _play_one(h=16, w=30, mines=99, seed=None, verbose=True):
    if seed is not None:
        random.seed(seed)
    game = Minesweeper(h, w, mines)
    ai = MinesweeperAI(h, w, mines)

    first = (h//2, w//2)
    game.make_safe_first_move(first)
    rev = game.reveal_chain(first)
    if rev.get(first) == -1:
        return False
    ai.add_knowledge_batch(rev)

    while True:
        if len(ai.moves_made) == h*w - mines:
            return True
        mv = ai.make_random_move()
        if mv is None:
            return False
        if game.is_mine(mv):
            return False
        ai.add_knowledge_batch(game.reveal_chain(mv))


# =========================
# Browser API (Pyodide)
# =========================

# 可选：如果你想彻底静音 AI 里的 print：
# 方案A：直接把 MinesweeperAI.make_random_move 里的 print 删掉/注释掉（最简单）
# 方案B：这里全局屏蔽 print（会影响所有print）
# import builtins
# builtins.print = lambda *args, **kwargs: None

_GAME = None
_AI = None
_REVEALED = set()
_LOST = False
_WON = False
_FIRST = False
_H = _W = _M = 0

def _to_list_cell(cell):
    return [int(cell[0]), int(cell[1])]

def ms_new_game(h, w, mines, seed=None):
    """
    返回 full state:
    {
      h,w,mines,
      revealed: [[r,c,n],...],
      ai_mines: [[r,c],...],
      lost, won,
      revealed_count
    }
    """
    global _GAME, _AI, _REVEALED, _LOST, _WON, _FIRST, _H, _W, _M
    import random as _random

    _H, _W, _M = int(h), int(w), int(mines)
    if seed is not None:
        _random.seed(int(seed))

    _GAME = Minesweeper(_H, _W, _M)
    _AI = MinesweeperAI(_H, _W, _M)
    _REVEALED = set()
    _LOST = False
    _WON = False
    _FIRST = False
    return ms_get_state()

def ms_get_state():
    global _GAME, _AI, _REVEALED, _LOST, _WON, _H, _W, _M
    # 这里 full state 暂时不存每个格子的数字，只返回已翻开集合（数字在 reveal_chain 里给出）
    # 如果你想刷新后仍显示数字，可以把“翻开时的n”存下来；这里为了简单略过。
    revealed_list = []
    # 由于我们没持久化每格n，这里只把已翻开格重新计算 nearby_mines（安全）；
    # 如果输了，踩雷那格会显示 X（在 step delta 里处理），full state 不管它也行。
    for (r, c) in _REVEALED:
        if _GAME.is_mine((r, c)):
            n = -1
        else:
            n = _GAME.nearby_mines((r, c))
        revealed_list.append([r, c, n])

    return {
        "h": _H, "w": _W, "mines": _M,
        "revealed": revealed_list,
        "ai_mines": [_to_list_cell(c) for c in _AI.mines],
        "lost": bool(_LOST),
        "won": bool(_WON),
        "revealed_count": len(_REVEALED),
    }

def ms_step():
    """
    返回 delta:
    {
      move: [r,c] or null,
      newly: [[r,c,n], ...]  # n=-1 表示踩雷
      ai_mines: [[r,c], ...]
      lost, won, stuck
      revealed_count
    }
    """
    global _GAME, _AI, _REVEALED, _LOST, _WON, _FIRST, _H, _W, _M

    if _GAME is None:
        return {"move": None, "newly": [], "ai_mines": [], "lost": False, "won": False, "stuck": True, "revealed_count": 0}

    if _LOST or _WON:
        return {"move": None, "newly": [], "ai_mines": [_to_list_cell(c) for c in _AI.mines],
                "lost": bool(_LOST), "won": bool(_WON), "stuck": False, "revealed_count": len(_REVEALED)}

    mv = _AI.make_safe_move()
    if mv is None:
        mv = _AI.make_random_move()
    if mv is None:
        return {"move": None, "newly": [], "ai_mines": [_to_list_cell(c) for c in _AI.mines],
                "lost": False, "won": False, "stuck": True, "revealed_count": len(_REVEALED)}

    if not _FIRST:
        _GAME.make_safe_first_move(mv)
        _FIRST = True

    results = _GAME.reveal_chain(mv)
    newly = []

    # 踩雷
    if any(v == -1 for v in results.values()):
        _LOST = True
        _REVEALED.add(mv)
        newly.append([mv[0], mv[1], -1])
        return {
            "move": _to_list_cell(mv),
            "newly": newly,
            "ai_mines": [_to_list_cell(c) for c in _AI.mines],
            "lost": True, "won": False, "stuck": False,
            "revealed_count": len(_REVEALED),
        }

    # 安全：批量更新
    _AI.add_knowledge_batch(results)
    for (r, c), n in results.items():
        if (r, c) not in _REVEALED:
            _REVEALED.add((r, c))
            newly.append([r, c, int(n)])

    if len(_REVEALED) == _H * _W - _M:
        _WON = True

    return {
        "move": _to_list_cell(mv),
        "newly": newly,
        "ai_mines": [_to_list_cell(c) for c in _AI.mines],
        "lost": bool(_LOST), "won": bool(_WON), "stuck": False,
        "revealed_count": len(_REVEALED),
    }

