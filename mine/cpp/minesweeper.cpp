#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>
#include <set>
#include <map>
#include <queue>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <ctime>
#include <tuple>
#include <sstream>
#include <unordered_set>
#include <iterator>

using namespace emscripten;

// ==========================================
// Utils & Config
// ==========================================
typedef std::pair<int, int> Cell;

double nCr(int n, int r) {
    if (r < 0 || r > n) return 0.0;
    if (r == 0 || r == n) return 1.0;
    if (r > n / 2) r = n - r;
    double res = 1.0;
    for (int i = 1; i <= r; ++i) res = res * (n - i + 1) / i;
    return res;
}

// ==========================================
// Game Logic
// ==========================================
class Minesweeper {
public:
    int H, W, M;
    std::set<Cell> mines;
    std::vector<std::vector<bool>> board;
    std::set<Cell> revealed;
    std::map<Cell, int> revealed_nums;
    bool first_move_made = false;
    bool lost = false;
    bool won = false;

    Minesweeper(int h, int w, int m) : H(h), W(w), M(m) {
        board.resize(H, std::vector<bool>(W, false));
    }

    bool is_valid(int r, int c) const {
        return r >= 0 && r < H && c >= 0 && c < W;
    }

    // Xorshift32 PRNG
    // 确保与 Python 版本的逻辑完全对齐
    class Xorshift32 {
    private:
        uint32_t x;
    public:
        // Python: x = seed if seed != 0 else 0x12345678
        // 注意：这里接收 uint32_t，调用方需处理 signed int 到 unsigned 的转换
        Xorshift32(uint32_t seed) : x(seed != 0 ? seed : 0x12345678) {}
        
        uint32_t next() {
            // C++ 的 uint32_t 自动处理了 Python 中的 & 0xFFFFFFFF
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            return x;
        }
        
        // Python: u = rng_func(); return u % n
        uint32_t next_int(uint32_t n) {
            return next() % n;
        }
    };

    // 修改参数 seed 为 uint32_t，防止有符号整数溢出导致的问题
    // 如果外部传入的是 int，请在调用前 static_cast<uint32_t>(seed)
    void place_mines(Cell safe_cell, int mode, uint32_t seed) {
        if (first_move_made) return;
        first_move_made = true;
        
        // 2. 创建安全区 (Safe Zone)
        // 使用 set 保证查找效率，逻辑与 Python 的 set 相同
        std::set<Cell> safe_zone;
        safe_zone.insert(safe_cell);
        
        if (mode == 1) { // 3x3 Safe
            // 遍历周围 8 格
            for (int r = safe_cell.first - 1; r <= safe_cell.first + 1; ++r) {
                for (int c = safe_cell.second - 1; c <= safe_cell.second + 1; ++c) {
                    if (is_valid(r, c)) {
                        safe_zone.insert({r, c});
                    }
                }
            }
        }
        // mode == 2 (1x1 Safe) 已经在上面 insert(safe_cell) 处理了

        // 计算最大雷数
        int cap = H * W - static_cast<int>(safe_zone.size());
        if (M > cap) M = cap; 

        // 3. 收集所有候选格子
        // 关键：遍历顺序必须严格是 行优先 (Row-Major)，即 (0,0), (0,1), ..., (1,0)...
        // 这与 Python 的双重循环 range(H), range(W) 一致
        std::vector<Cell> candidates;
        candidates.reserve(H * W); // 预分配内存优化
        
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                Cell cell = {r, c};
                // 只有不在安全区的才加入候选
                if (safe_zone.find(cell) == safe_zone.end()) {
                    candidates.push_back(cell);
                }
            }
        }

        if (M > static_cast<int>(candidates.size())) {
            throw std::runtime_error("Too many mines for given safe zone");
        }

        // 4. Fisher-Yates 洗牌算法
        // 使用确定性随机数
        Xorshift32 rng(seed);
        for (int i = 0; i < M; ++i) {
            // Python: j = i + rng.next_int(len(cells) - i)
            // C++: candidates.size() 对应 len(cells)
            uint32_t remaining = static_cast<uint32_t>(candidates.size() - i);
            int j = i + rng.next_int(remaining);
            std::swap(candidates[i], candidates[j]);
        }

        // 5. 放置地雷
        for (auto& row : board) std::fill(row.begin(), row.end(), false);
        mines.clear();
        for (int i = 0; i < M; ++i) {
            Cell mine_cell = candidates[i];
            mines.insert(mine_cell);
            board[mine_cell.first][mine_cell.second] = true;
        }
    }

    std::vector<Cell> get_neighbors(Cell c) {
        std::vector<Cell> nb;
        for (int r = c.first - 1; r <= c.first + 1; ++r) {
            if (r < 0 || r >= H) continue;
            for (int col = c.second - 1; col <= c.second + 1; ++col) {
                if (col < 0 || col >= W) continue;
                if (r == c.first && col == c.second) continue;
                nb.push_back({r, col});
            }
        }
        return nb;
    }

    int count_nearby_mines(Cell c) {
        int cnt = 0;
        for (auto& n : get_neighbors(c)) {
            if (board[n.first][n.second]) cnt++;
        }
        return cnt;
    }

    std::map<Cell, int> reveal(Cell start) {
        std::map<Cell, int> newly;
        if (!is_valid(start.first, start.second)) return newly;
        
        if (board[start.first][start.second]) {
            newly[start] = -1;
            return newly;
        }

        std::queue<Cell> q;
        q.push(start);
        std::set<Cell> visited;
        visited.insert(start);

        while (!q.empty()) {
            Cell curr = q.front(); q.pop();
            int n = count_nearby_mines(curr);
            newly[curr] = n;
            
            if (n == 0) {
                for (auto& nb : get_neighbors(curr)) {
                    if (visited.find(nb) == visited.end() && !board[nb.first][nb.second]) { // 修复了 visited.count 逻辑
                        visited.insert(nb);
                        q.push(nb);
                    }
                }
            }
        }
        return newly;
    }
};

// ==========================================
// AI Logic (Optimized & Safer Hybrid Solver)
// Interface-compatible: keep class name, public fields, and methods used by JS unchanged.
// Key changes vs old C++:
// 1) Gaussian is ONLY used to derive known 0/1 variables (safe), never to generate new linear constraints.
// 2) Component solve thresholds aligned with Python: GAUSSIAN_THRESHOLD=25, MAX_EXACT_VAR=40.
// 3) MC uses importance sampling (q, iw=1/q) and degree-based variable order (less bias, more symmetric).
// 4) Convolution uses log-comb to avoid nCr overflow/underflow; Z checked for finite.
// 5) Knowledge update is made stable: mark_mine/mark_safe also update sentences immediately.
// 6) Simplify/infer loop is deterministic and does not rely on tight arbitrary caps for correctness.
// ==========================================

// ==========================================
// AI Logic (Hybrid Solver) - "No-freeze" patched
// Interface-compatible: class name, public fields, and public methods unchanged.
// Key fixes:
//   1) infer_new_sentences(): add global seen set + real-growth changed + hard budgets => no hang.
//   2) normalize_knowledge(): faster deterministic dedup without stringstream signature.
// ==========================================

typedef std::pair<int,int> Cell;

// ---- numeric helpers ----
static inline long double logC(int n, int k) {
    if (k < 0 || k > n) return -INFINITY;
    if (k == 0 || k == n) return 0.0L;
    return lgammal((long double)n + 1.0L) - lgammal((long double)k + 1.0L) - lgammal((long double)(n - k) + 1.0L);
}
static inline long double expSafe(long double x) {
    if (x == -INFINITY) return 0.0L;
    if (x > 11300.0L) return INFINITY;
    if (x < -11300.0L) return 0.0L;
    return expl(x);
}

// ==========================================
// Constraint (subset-sum form only)
// ==========================================
struct Constraint {
    std::vector<int> vars; // indices in component variable list
    int val = 0;
    bool operator<(const Constraint& o) const { return vars < o.vars || (vars == o.vars && val < o.val); }
    bool operator==(const Constraint& o) const { return vars == o.vars && val == o.val; }
};

// ==========================================
// MinesweeperAI
// ==========================================
class MinesweeperAI {
public:
    int H, W, TotalMines;
    std::set<Cell> mines_found;
    std::set<Cell> safes_found;
    std::set<Cell> moves_made;

    struct Sentence {
        std::set<Cell> cells;
        int count = 0;
    };
    std::vector<Sentence> knowledge;

    // Strategy params (keep names used by interface code)
    const double LOW_P_FIRST = 0.05;
    const double RISK_W = 10.0;
    const double P_TOL = 0.05;

    // Hybrid params
    static constexpr bool DBG_PROB = false; // 想静默就改成 false

    const int GAUSSIAN_THRESHOLD = 25;
    const int MAX_EXACT_VAR = 70;
    const int MC_SAMPLES = 100000;
    const int MAX_EXACT_NODES = 3000000;

    static void dbg_component_header(const char* tag, int n_comp, int n_known, int n_rem, int n_cons) {
    if constexpr (!DBG_PROB) return;
    std::cerr << "[AI][prob][" << tag << "] comp=" << n_comp
              << " known=" << n_known
              << " rem=" << n_rem
              << " cons=" << n_cons
              << "\n";
    }


    MinesweeperAI(int h, int w, int m) : H(h), W(w), TotalMines(m) {}

    // ---- marking updates knowledge immediately ----
    void mark_mine(Cell c) {
        if (mines_found.count(c)) return;
        mines_found.insert(c);
        for (auto& s : knowledge) {
            if (s.cells.erase(c)) s.count -= 1;
        }
    }
    void mark_safe(Cell c) {
        if (safes_found.count(c)) return;
        safes_found.insert(c);
        for (auto& s : knowledge) {
            s.cells.erase(c);
        }
    }

    // Add revealed cells with numbers (-1 ignored)
    void add_knowledge(std::map<Cell,int> revealed) {
        for (auto const& kv : revealed) {
            Cell c = kv.first;
            int val = kv.second;
            if (val < 0) continue;
            if (moves_made.count(c)) continue;

            moves_made.insert(c);
            mark_safe(c);

            std::set<Cell> unknown;
            int adj_mines = 0;
            for (int r = c.first - 1; r <= c.first + 1; ++r) {
                if (r < 0 || r >= H) continue;
                for (int col = c.second - 1; col <= c.second + 1; ++col) {
                    if (col < 0 || col >= W) continue;
                    if (r == c.first && col == c.second) continue;
                    Cell n = {r, col};
                    if (mines_found.count(n)) adj_mines++;
                    else if (!safes_found.count(n) && !moves_made.count(n)) unknown.insert(n);
                }
            }
            int newc = val - adj_mines;
            if (!unknown.empty() && 0 <= newc && newc <= (int)unknown.size()) {
                knowledge.push_back({unknown, newc});
            }
        }
        normalize_knowledge();
        infer_new_sentences(); // patched: no hang
    }

    // Safe move
    Cell make_safe_move() {
        for (auto& c : safes_found) if (!moves_made.count(c)) return c;
        return {-1, -1};
    }

    // Heuristic
    int info_gain_heuristic(Cell cell) {
        int unk = 0;
        for (int r = cell.first - 1; r <= cell.first + 1; ++r) {
            if (r < 0 || r >= H) continue;
            for (int c = cell.second - 1; c <= cell.second + 1; ++c) {
                if (c < 0 || c >= W) continue;
                if (r == cell.first && c == cell.second) continue;
                Cell nb = {r,c};
                if (!moves_made.count(nb) && !safes_found.count(nb) && !mines_found.count(nb)) unk++;
            }
        }
        int deg = 0;
        for (auto& s : knowledge) if (s.cells.count(cell)) deg++;
        return 3*deg + unk;
    }

    // Random move policy
    Cell make_random_move() {
        Cell safe = make_safe_move();
        if (safe.first != -1) return safe;

        std::vector<Cell> unknown;
        unknown.reserve((size_t)H*W);
        for (int r=0; r<H; ++r) for (int c=0; c<W; ++c) {
            Cell cell={r,c};
            if (!moves_made.count(cell) && !mines_found.count(cell) && !safes_found.count(cell)) unknown.push_back(cell);
        }
        if (unknown.empty()) return {-1,-1};

        bool valid=false;
        auto probs = compute_probabilities(valid);
        if (probs.empty()) return unknown[0];

        std::vector<std::pair<double,Cell>> cand;
        cand.reserve(probs.size());
        for (auto const& kv : probs) {
            if (!moves_made.count(kv.first) && !mines_found.count(kv.first)) cand.push_back({kv.second, kv.first});
        }
        if (cand.empty()) return {-1,-1};
        std::sort(cand.begin(), cand.end());
        double min_p = cand[0].first;

        if (min_p <= LOW_P_FIRST) {
            std::vector<Cell> low_pool;
            for (auto& p : cand) if (p.first <= LOW_P_FIRST) low_pool.push_back(p.second);

            double best_p = 1.0;
            for (auto& c : low_pool) best_p = std::min(best_p, probs[c]);
            std::vector<Cell> best_pool;
            for (auto& c : low_pool) if (std::abs(probs[c] - best_p) <= 1e-9) best_pool.push_back(c);

            Cell best_c = best_pool[0];
            std::tuple<int,int,int> best_score = {-1,0,0};
            for (auto& c : best_pool) {
                int ig = info_gain_heuristic(c);
                std::tuple<int,int,int> score = {ig, -c.first, -c.second};
                if (score > best_score) { best_score = score; best_c = c; }
            }
            return best_c;
        }

        Cell eg = new_solve_endgame(unknown);
        if (eg.first != -1) return eg;

        std::vector<Cell> band;
        for (auto& p : cand) if (p.first <= min_p + P_TOL) band.push_back(p.second);
        if (band.size() == 1) return band[0];

        Cell best_c = band[0];
        double best_score = -1e100;
        for (auto& c : band) {
            double p = probs[c];
            double gain = (double)info_gain_heuristic(c);
            double risk_term = (p - min_p) / std::max(1e-9, P_TOL);
            double score = gain - RISK_W * risk_term;
            if (score > best_score || (std::abs(score - best_score) < 1e-12 && p < probs[best_c])) {
                best_score = score; best_c = c;
            }
        }
        return best_c;
    }

    // ==========================================
    // PROBABILITY ENGINE
    // ==========================================
    typedef std::map<int, std::pair<long double, std::vector<long double>>> ComponentResult;

    std::map<Cell,double> compute_probabilities(bool& valid_global) {
        normalize_knowledge();

        std::vector<Cell> unknown;
        unknown.reserve((size_t)H*W);
        for (int r=0; r<H; ++r) for (int c=0; c<W; ++c) {
            Cell cell={r,c};
            if (!moves_made.count(cell) && !mines_found.count(cell) && !safes_found.count(cell)) unknown.push_back(cell);
        }
        if (unknown.empty()) { valid_global = true; return {}; }

        std::set<Cell> frontier_set;
        for (auto& s : knowledge) for (auto& c : s.cells) frontier_set.insert(c);

        std::vector<Cell> frontier, outside;
        frontier.reserve(unknown.size());
        outside.reserve(unknown.size());
        for (auto& c : unknown) (frontier_set.count(c) ? frontier : outside).push_back(c);

        auto components = find_components(frontier);
        std::vector<ComponentResult> comp_results;
        comp_results.reserve(components.size());
        for (auto& comp : components) comp_results.push_back(solve_component_safe(comp));

        int mines_left = TotalMines - (int)mines_found.size();
        if (mines_left < 0) mines_left = 0;

        // Convolve mine-count distributions
        std::map<int, long double> global_dist;
        global_dist[0] = 1.0L;
        for (auto& res : comp_results) {
            std::map<int, long double> next;
            for (auto const& [k1, w1] : global_dist) {
                for (auto const& [k2, pair_w] : res) {
                    if (k1 + k2 <= mines_left) next[k1 + k2] += w1 * pair_w.first;
                }
            }
            global_dist.swap(next);
        }

        int n_out = (int)outside.size();

        // logZ = log sum_{k_front} w_front * C(n_out, mines_left-k_front)
        long double logZ = -INFINITY;
        for (auto const& [k_front, w_front] : global_dist) {
            int k_out = mines_left - k_front;
            if (k_out < 0 || k_out > n_out) continue;
            if (w_front <= 0) continue;
            long double term = logl(w_front) + logC(n_out, k_out);
            if (logZ == -INFINITY) logZ = term;
            else {
                logZ = (logZ > term) ? (logZ + log1pl(expl(term - logZ)))
                                     : (term + log1pl(expl(logZ - term)));
            }
        }

        std::map<Cell,double> final_probs;
        if (!std::isfinite((double)logZ) || logZ < -100000.0L) {
            valid_global = false;
            double p = (double)mines_left / (double)unknown.size();
            p = std::max(0.0, std::min(1.0, p));
            for (auto& c : unknown) final_probs[c] = p;
            return final_probs;
        }
        valid_global = true;

        // prefix/suffix for rest dist
        std::vector<std::map<int,long double>> pref(comp_results.size()+1), suff(comp_results.size()+1);
        pref[0][0]=1.0L;
        for (size_t i=0;i<comp_results.size();++i){
            auto& A=pref[i]; auto& R=comp_results[i];
            std::map<int,long double> B;
            for (auto const& [k1,w1]:A)
                for (auto const& [k2,pw]:R)
                    if (k1+k2<=mines_left) B[k1+k2]+=w1*pw.first;
            pref[i+1].swap(B);
        }
        suff[comp_results.size()][0]=1.0L;
        for (int i=(int)comp_results.size()-1;i>=0;--i){
            auto& A=suff[i+1]; auto& R=comp_results[i];
            std::map<int,long double> B;
            for (auto const& [k1,w1]:A)
                for (auto const& [k2,pw]:R)
                    if (k1+k2<=mines_left) B[k1+k2]+=w1*pw.first;
            suff[i].swap(B);
        }

        // Frontier cell probs
        for (size_t i = 0; i < comp_results.size(); ++i) {
            std::map<int,long double> rest_dist;
            for (auto const& [k1,w1] : pref[i]) {
                for (auto const& [k2,w2] : suff[i+1]) {
                    if (k1 + k2 <= mines_left) rest_dist[k1+k2] += w1*w2;
                }
            }

            auto& comp_res = comp_results[i];
            auto& comp_cells = components[i];

            for (auto const& [k_local, val] : comp_res) {
                long double w_local = val.first;
                auto const& cell_counts = val.second;
                if (w_local <= 0) continue;

                int target = mines_left - k_local;
                if (target < 0) continue;

                long double logWays = -INFINITY;
                for (auto const& [k_rest, w_rest] : rest_dist) {
                    int k_out = target - k_rest;
                    if (k_out < 0 || k_out > n_out) continue;
                    if (w_rest <= 0) continue;
                    long double term = logl(w_rest) + logC(n_out, k_out);
                    if (logWays == -INFINITY) logWays = term;
                    else {
                        logWays = (logWays > term) ? (logWays + log1pl(expl(term - logWays)))
                                                   : (term + log1pl(expl(logWays - term)));
                    }
                }
                if (logWays == -INFINITY) continue;

                long double factor = expSafe(logWays - logZ);
                if (!(factor > 0)) continue;

                for (size_t c_idx = 0; c_idx < comp_cells.size(); ++c_idx) {
                    long double add = cell_counts[c_idx] * factor; // already in probability space after /Z
                    final_probs[comp_cells[c_idx]] += (double)add;
                }
            }
        }

        // Outside probability = E[k_out]/n_out
        long double avg_out = 0.0L;
        for (auto const& [k_front, w_front] : global_dist) {
            int k_out = mines_left - k_front;
            if (k_out < 0 || k_out > n_out) continue;
            if (w_front <= 0) continue;
            long double prob_split = expSafe((logl(w_front) + logC(n_out, k_out)) - logZ);
            avg_out += (long double)k_out * prob_split;
        }
        double p_out = (n_out > 0) ? (double)(avg_out / (long double)n_out) : 0.0;
        p_out = std::max(0.0, std::min(1.0, p_out));
        for (auto& c : outside) final_probs[c] = p_out;

        // Clamp and ensure all unknowns present
        for (auto& c : unknown) {
            if (!final_probs.count(c)) final_probs[c] = p_out;
            final_probs[c] = std::max(0.0, std::min(1.0, final_probs[c]));
        }
        return final_probs;
    }

private:
    // --------------------------
    // NO-FREEZE budgets
    // --------------------------
    const int MAX_INFER_PASSES = 256;            // hard cap on subset-inference passes
    const int MAX_KNOWLEDGE_SIZE = 12000;        // cap sentence count to avoid O(n^2) blowups
    const int MAX_NEW_SENTENCES_PER_PASS = 4000; // cap per pass to avoid giant allocations

    

    // Global seen set: prevents re-adding same inferred sentence forever.
    std::unordered_set<std::uint64_t> seen_sentences;

    // deterministic hash for Sentence (count + cells)
    static inline std::uint64_t hash_sentence(const Sentence& s) {
        // 64-bit mix; collision is possible but extremely unlikely; acceptable as "termination guard".
        std::uint64_t h = 1469598103934665603ULL; // FNV offset basis
        auto mix = [&](std::uint64_t x){
            h ^= x;
            h *= 1099511628211ULL; // FNV prime
        };
        mix((std::uint64_t)(std::uint32_t)s.count);
        mix((std::uint64_t)(std::uint32_t)s.cells.size());
        for (auto const& c : s.cells) {
            // pack r,c into 64
            std::uint64_t x = ((std::uint64_t)(std::uint32_t)c.first << 32) ^ (std::uint64_t)(std::uint32_t)c.second;
            mix(x);
        }
        return h;
    }

    // ==========================================
    // Knowledge maintenance (stable + faster)
    // ==========================================
    void normalize_knowledge() {
        // validate counts
        for (auto& s : knowledge) {
            if (s.count < 0) {s.count = -999999; if constexpr (DBG_PROB) std::cerr << "[AI] invalid sentence pruned\n";};
            if (s.count > (int)s.cells.size()) {s.count = -999999; if constexpr (DBG_PROB) std::cerr << "[AI] invalid sentence pruned\n";};
        }

        // remove empty/invalid
        std::vector<Sentence> cleaned;
        cleaned.reserve(knowledge.size());
        for (auto& s : knowledge) {
            if (!s.cells.empty() && s.count >= 0 && s.count <= (int)s.cells.size())
                cleaned.push_back(s);
        }

        // sort + dedup (no stringstream)
        auto cmp = [](const Sentence& a, const Sentence& b){
            if (a.cells.size() != b.cells.size()) return a.cells.size() < b.cells.size();
            if (a.count != b.count) return a.count < b.count;
            // lexicographic compare of sets
            auto ia = a.cells.begin(), ib = b.cells.begin();
            while (ia != a.cells.end() && ib != b.cells.end()) {
                if (*ia != *ib) return *ia < *ib;
                ++ia; ++ib;
            }
            return false;
        };
        std::sort(cleaned.begin(), cleaned.end(), cmp);
        cleaned.erase(std::unique(cleaned.begin(), cleaned.end(), [](const Sentence& a, const Sentence& b){
            return a.count==b.count && a.cells==b.cells;
        }), cleaned.end());
        knowledge.swap(cleaned);

        // Propagate direct mines/safes until fixed point
        bool changed=true;
        while(changed){
            changed=false;
            std::vector<Cell> to_mine, to_safe;
            to_mine.reserve(64);
            to_safe.reserve(64);

            for (auto& s : knowledge) {
                if (s.cells.empty()) continue;
                if (s.count == 0) {
                    for (auto& c: s.cells) if (!safes_found.count(c)) to_safe.push_back(c);
                } else if (s.count == (int)s.cells.size()) {
                    for (auto& c: s.cells) if (!mines_found.count(c)) to_mine.push_back(c);
                }
            }

            for (auto& c: to_safe) {
                if (!safes_found.count(c)) { mark_safe(c); changed=true; }
            }
            for (auto& c: to_mine) {
                if (!mines_found.count(c)) { mark_mine(c); changed=true; }
            }

            // prune empty/invalid
            std::vector<Sentence> nk;
            nk.reserve(knowledge.size());
            for (auto& s: knowledge) {
                if (!s.cells.empty() && 0<=s.count && s.count <= (int)s.cells.size()) nk.push_back(s);
            }
            knowledge.swap(nk);
        }
    }

    // *** patched: termination-guaranteed subset inference ***
    void infer_new_sentences() {
        // seed seen with current knowledge (do not clear; no need)
        for (auto const& s : knowledge) {
            seen_sentences.insert(hash_sentence(s));
        }

        int passes = 0;
        while (passes++ < MAX_INFER_PASSES) {
            normalize_knowledge();
            if ((int)knowledge.size() >= MAX_KNOWLEDGE_SIZE) return;

            // sort by size for subset inference
            std::sort(knowledge.begin(), knowledge.end(),
                      [](const Sentence& a, const Sentence& b){ return a.cells.size() < b.cells.size(); });

            const size_t before = knowledge.size();
            std::vector<Sentence> newS;
            newS.reserve(256);

            for (size_t i=0;i<knowledge.size();++i){
                const auto& A = knowledge[i];
                if (A.cells.empty()) continue;

                for (size_t j=i+1;j<knowledge.size();++j){
                    const auto& B = knowledge[j];
                    if (A.cells.size() > B.cells.size()) continue;

                    // subset?
                    if (!std::includes(B.cells.begin(), B.cells.end(),
                                       A.cells.begin(), A.cells.end())) continue;

                    Sentence ns;
                    ns.count = B.count - A.count;
                    std::set_difference(B.cells.begin(), B.cells.end(),
                                        A.cells.begin(), A.cells.end(),
                                        std::inserter(ns.cells, ns.cells.begin()));

                    if (ns.cells.empty()) continue;
                    if (ns.count < 0 || ns.count > (int)ns.cells.size()) continue;

                    // seen guard (prevents endless re-adding)
                    std::uint64_t hs = hash_sentence(ns);
                    if (seen_sentences.find(hs) != seen_sentences.end()) continue;
                    seen_sentences.insert(hs);

                    newS.push_back(std::move(ns));
                    if ((int)newS.size() >= MAX_NEW_SENTENCES_PER_PASS) break;
                    if ((int)(before + newS.size()) >= MAX_KNOWLEDGE_SIZE) break;
                }
                if ((int)newS.size() >= MAX_NEW_SENTENCES_PER_PASS) break;
                if ((int)(before + newS.size()) >= MAX_KNOWLEDGE_SIZE) break;
            }

            if (newS.empty()) return;

            knowledge.insert(knowledge.end(), newS.begin(), newS.end());
            normalize_knowledge();

            // IMPORTANT: only continue if knowledge actually grew (prevents "added then all pruned" infinite spin)
            if (knowledge.size() <= before) return;
        }
        // exceeded pass budget => stop to avoid freezing
    }

    // ==========================================
    // Components
    // ==========================================
    std::vector<std::vector<Cell>> find_components(const std::vector<Cell>& frontier) {
        if (frontier.empty()) return {};
        std::set<Cell> frontier_set(frontier.begin(), frontier.end());

        std::unordered_map<long long, std::vector<int>> cell_to_sent;
        std::vector<std::vector<Cell>> active;
        active.reserve(knowledge.size());

        auto hcell = [&](const Cell& c)->long long { return ((long long)c.first<<32) ^ (unsigned)c.second; };

        for (auto& s : knowledge) {
            if (s.cells.empty()) continue;
            std::vector<Cell> rel;
            for (auto& c : s.cells) if (frontier_set.count(c)) rel.push_back(c);
            if (rel.empty()) continue;
            int idx = (int)active.size();
            active.push_back(std::move(rel));
            for (auto& c : active.back()) cell_to_sent[hcell(c)].push_back(idx);
        }

        std::unordered_set<long long> visited;
        std::vector<std::vector<Cell>> comps;
        comps.reserve(frontier.size());

        for (auto& start : frontier) {
            long long hs = hcell(start);
            if (visited.count(hs)) continue;
            std::vector<Cell> comp;
            std::vector<Cell> stack; stack.push_back(start);
            visited.insert(hs);

            while(!stack.empty()){
                Cell cur = stack.back(); stack.pop_back();
                comp.push_back(cur);
                auto it = cell_to_sent.find(hcell(cur));
                if (it == cell_to_sent.end()) continue;
                for (int si : it->second) {
                    for (auto& nb : active[si]) {
                        long long hn = hcell(nb);
                        if (!visited.count(hn)) { visited.insert(hn); stack.push_back(nb); }
                    }
                }
            }
            std::sort(comp.begin(), comp.end());
            comps.push_back(std::move(comp));
        }
        return comps;
    }

    std::vector<Constraint> get_constraints_inside(const std::vector<Cell>& cells) {
        std::set<Cell> scope(cells.begin(), cells.end());
        std::map<Cell,int> idx;
        for (int i=0;i<(int)cells.size();++i) idx[cells[i]]=i;

        std::vector<Constraint> cons;
        for (auto& s : knowledge) {
            if (s.cells.empty()) continue;
            bool inside=true;
            for (auto& c : s.cells) { if (!scope.count(c)) { inside=false; break; } }
            if (!inside) continue;
            Constraint ct;
            ct.val = s.count;
            ct.vars.reserve(s.cells.size());
            for (auto& c : s.cells) ct.vars.push_back(idx[c]);
            std::sort(ct.vars.begin(), ct.vars.end());
            cons.push_back(std::move(ct));
        }
        std::sort(cons.begin(), cons.end());
        cons.erase(std::unique(cons.begin(), cons.end()), cons.end());
        return cons;
    }

    // GF(2) elimination for parity constraints only.
// Returns (knowns, fixed_mines). If contradiction in GF(2): fixed_mines = -1.
//
// NOTE: This does NOT preserve full Minesweeper constraints; it only uses mod2 parity.
// So knowns will usually be empty; contradiction detection is only mod2-level.
std::pair<std::map<Cell,int>, int> gaussian_find_knowns_only(const std::vector<Cell>& cells) {
    std::map<Cell,int> knowns;
    int fixed_mines = 0;
    const int n = (int)cells.size();
    if (n == 0) return {knowns, 0};

    // Build scope + index map
    std::set<Cell> scope(cells.begin(), cells.end());
    std::map<Cell,int> c2i;
    for (int i=0;i<n;++i) c2i[cells[i]] = i;

    // We store each row as bitset of variables + rhs bit (parity of count)
    // Use blocks of 64 bits for speed
    const int B = (n + 63) / 64;

    struct Row {
        std::vector<uint64_t> bits; // size B
        uint64_t rhs;              // 0/1
    };

    std::vector<Row> rows;
    rows.reserve(knowledge.size());

    for (auto& s : knowledge) {
        if (s.cells.empty()) continue;

        bool inside = true;
        for (auto& c : s.cells) { if (!scope.count(c)) { inside=false; break; } }
        if (!inside) continue;

        Row r;
        r.bits.assign(B, 0ULL);
        for (auto& c : s.cells) {
            int idx = c2i[c];
            r.bits[idx >> 6] ^= (1ULL << (idx & 63)); // XOR set (same as set for unique cells)
        }
        r.rhs = (uint64_t)(s.count & 1); // parity only
        // skip empty row early unless it is contradictory
        bool any = false;
        for (uint64_t w : r.bits) { if (w) { any=true; break; } }
        if (!any) {
            if (r.rhs) return {std::map<Cell,int>{}, -1}; // 0 = 1 mod2
            continue; // 0 = 0 mod2
        }
        rows.push_back(std::move(r));
    }

    if (rows.empty()) return {knowns, 0};

    // Gaussian elimination in GF(2)
    int m = (int)rows.size();
    std::vector<int> where(n, -1); // which row is pivot for column i
    int row = 0;

    auto testbit = [&](const Row& r, int col)->bool{
        return (r.bits[col >> 6] >> (col & 63)) & 1ULL;
    };

    for (int col=0; col<n && row<m; ++col) {
        int sel = -1;
        for (int i=row; i<m; ++i) {
            if (testbit(rows[i], col)) { sel = i; break; }
        }
        if (sel == -1) continue;
        std::swap(rows[row], rows[sel]);
        where[col] = row;

        // eliminate in all other rows
        for (int i=0; i<m; ++i) {
            if (i == row) continue;
            if (!testbit(rows[i], col)) continue;
            // row_i ^= row_pivot
            for (int b=0;b<B;++b) rows[i].bits[b] ^= rows[row].bits[b];
            rows[i].rhs ^= rows[row].rhs;
        }
        ++row;
    }

    // Check contradictions: 0 = 1 mod2
    for (int i=0;i<m;++i) {
        bool any=false;
        for (uint64_t w : rows[i].bits) { if (w) { any=true; break; } }
        if (!any && (rows[i].rhs & 1ULL)) {
            return {std::map<Cell,int>{}, -1};
        }
    }

    // Try to extract "known variables" (rare in GF(2)):
    // If a row has exactly one '1' variable: x = rhs (mod2), but x is 0/1 so it fixes x exactly.
    for (int i=0;i<m;++i) {
        int one = -1;
        int cnt1 = 0;
        for (int col=0; col<n; ++col) {
            if (testbit(rows[i], col)) {
                one = col;
                if (++cnt1 > 1) break;
            }
        }
        if (cnt1 == 1) {
            int val = (int)(rows[i].rhs & 1ULL); // x = rhs
            Cell c = cells[one];
            if (!knowns.count(c)) {
                knowns[c] = val;
                if (val == 1) ++fixed_mines;
            }
        }
    }

    return {knowns, fixed_mines};
}

    std::pair<std::vector<Cell>, std::vector<Constraint>> apply_knowns_filter_constraints(
    const std::vector<Cell>& comp_cells,
    const std::map<Cell,int>& knowns)
{
    // remaining cells
    std::vector<Cell> rem_cells;
    rem_cells.reserve(comp_cells.size());
    for (auto& c : comp_cells) if (!knowns.count(c)) rem_cells.push_back(c);

    // index map (NO operator[] usage for lookup later!)
    std::map<Cell,int> rem_idx;
    for (int i=0;i<(int)rem_cells.size();++i) rem_idx.emplace(rem_cells[i], i);

    // scope for "inside component" constraints
    std::set<Cell> scope(comp_cells.begin(), comp_cells.end());

    std::vector<Constraint> reduced;
    reduced.reserve(knowledge.size());

    for (auto& s : knowledge) {
        if (s.cells.empty()) continue;

        bool inside = true;
        for (auto& c : s.cells) { if (!scope.count(c)) { inside=false; break; } }
        if (!inside) continue;

        int cnt = s.count;
        Constraint ct;
        ct.vars.reserve(s.cells.size());

        for (auto& c : s.cells) {
            auto itk = knowns.find(c);
            if (itk != knowns.end()) {
                cnt -= itk->second;
            } else {
                auto itv = rem_idx.find(c);
                if (itv == rem_idx.end()) {
                    // This should never happen. If it happens, the old code would silently map to 0 -> catastrophic.
                    if constexpr (DBG_PROB) {
                        std::cerr << "[AI][prob][apply_knowns] ERROR: cell not in rem_idx but also not in knowns.\n";
                        std::cerr << "  cell=(" << c.first << "," << c.second << ")\n";
                        std::cerr << "  comp_cells=" << comp_cells.size() << " rem_cells=" << rem_cells.size()
                                  << " knowns=" << knowns.size() << "\n";
                        std::cerr << "  sentence.count=" << s.count << " sentence.size=" << s.cells.size() << "\n";
                    }
                    // Fail hard in debug; in release, just return empty to trigger uniform fallback.
#ifndef NDEBUG
                    assert(false && "apply_knowns_filter_constraints: inconsistent rem_idx mapping");
#endif
                    return {{}, {}};
                }
                ct.vars.push_back(itv->second);
            }
        }

        if (ct.vars.empty()) continue;
        if (cnt < 0 || cnt > (int)ct.vars.size()) continue;

        std::sort(ct.vars.begin(), ct.vars.end());
        ct.vars.erase(std::unique(ct.vars.begin(), ct.vars.end()), ct.vars.end());
        ct.val = cnt;
        reduced.push_back(std::move(ct));
    }

    std::sort(reduced.begin(), reduced.end());
    reduced.erase(std::unique(reduced.begin(), reduced.end()), reduced.end());

    return {rem_cells, reduced};
}


    ComponentResult solve_component_safe(const std::vector<Cell>& comp_cells) {
    const int n_comp = (int)comp_cells.size();

    std::map<Cell,int> knowns;
    int fixed_mines = 0;

    std::vector<Cell> rem_cells = comp_cells;
    std::vector<Constraint> rem_cons;

    // --- choose path ---
    if (n_comp > GAUSSIAN_THRESHOLD) {
        auto pr0 = gaussian_find_knowns_only(comp_cells);
        knowns = std::move(pr0.first);
        fixed_mines = pr0.second;

        if (fixed_mines < 0) {
            // Gaussian detected contradiction inside this component.
            dbg_component_header("comp-contradiction", n_comp, 0, 0, 0);
            return ComponentResult(); // empty => global logZ will likely fail -> uniform fallback
        }

        auto pr = apply_knowns_filter_constraints(comp_cells, knowns);
        rem_cells = std::move(pr.first);
        rem_cons  = std::move(pr.second);

        if (rem_cells.empty() && (!knowns.empty() || fixed_mines > 0)) {
            // OK: everything resolved by knowns
        } else if (rem_cells.empty() && rem_cons.empty() && knowns.empty() && fixed_mines==0) {
            // This can happen if apply_knowns failed and returned empty.
            // Treat as "unable to solve component reliably".
            dbg_component_header("comp-applyknowns-failed", n_comp, (int)knowns.size(), 0, 0);
            return ComponentResult();
        }
    } else {
        rem_cons = get_constraints_inside(comp_cells);
    }

    const int n_rem = (int)rem_cells.size();
    dbg_component_header("comp", n_comp, (int)knowns.size(), n_rem, (int)rem_cons.size());

    // --- solve remaining ---
    ComponentResult base_res;
    if (n_rem == 0) {
        base_res[fixed_mines] = {1.0L, std::vector<long double>()};
    } else if (n_rem <= MAX_EXACT_VAR) {
        base_res = solve_exact_budgeted(n_rem, rem_cons, fixed_mines, MAX_EXACT_NODES);
        if (base_res.empty()) {
            if constexpr (DBG_PROB) std::cerr << "[AI][prob][comp] exact aborted -> MC\n";
            base_res = solve_mc_is(n_rem, rem_cons, fixed_mines, MC_SAMPLES);
        }
    } else {
        base_res = solve_mc_is(n_rem, rem_cons, fixed_mines, MC_SAMPLES);
    }

    if (base_res.empty()) {
        if constexpr (DBG_PROB) std::cerr << "[AI][prob][comp] solve produced empty result\n";
        return ComponentResult();
    }

    // --- expand back to full component vector ---
    ComponentResult out;
    std::map<Cell,int> comp_idx;
    for (int i=0;i<n_comp;++i) comp_idx[comp_cells[i]] = i;

    for (auto const& [mcount, ww] : base_res) {
        long double w = ww.first;
        std::vector<long double> full(n_comp, 0.0L);

        // inject known mines as "count = w"
        for (auto const& kv : knowns) {
            if (kv.second == 1) full[comp_idx[kv.first]] = w;
        }
        // inject remaining var mine-counts
        if (!ww.second.empty()) {
            // ww.second[i] means: weighted mine-count for rem_cells[i]
            for (int i=0;i<n_rem;++i) {
                full[comp_idx[rem_cells[i]]] = ww.second[i];
            }
        }

        out[mcount] = {w, std::move(full)};
    }

    // Optional sanity checks (debug)
    if constexpr (DBG_PROB) {
        for (auto const& [mcount, ww] : out) {
            long double w = ww.first;
            auto const& v = ww.second;
            for (auto x : v) {
                if (x < -1e-12L || x > w + 1e-12L) {
                    std::cerr << "[AI][prob][comp] SANITY FAIL: x=" << (double)x
                              << " w=" << (double)w << " mcount=" << mcount << "\n";
                    break;
                }
            }
        }
    }

    return out;
}


    ComponentResult solve_exact_budgeted(int n, const std::vector<Constraint>& cons, int base_mines, int node_budget) {
        std::vector<std::vector<int>> v2c(n);
        for (int i=0;i<(int)cons.size();++i) for (int v : cons[i].vars) v2c[v].push_back(i);

        std::vector<int> need(cons.size()), rem(cons.size());
        for (int i=0;i<(int)cons.size();++i) { need[i]=cons[i].val; rem[i]=(int)cons[i].vars.size(); }

        std::vector<int> assign(n,0);
        std::vector<int> order(n);
        std::vector<std::pair<int,int>> deg(n);
        for (int i=0;i<n;++i) deg[i] = {-(int)v2c[i].size(), i};
        std::sort(deg.begin(), deg.end());
        for (int i=0;i<n;++i) order[i]=deg[i].second;

        ComponentResult res;
        int nodes = 0;
        bool aborted = false;

        auto rec = [&](auto&& self, int k, int mF) -> void {
            if (aborted) return;
            nodes++;
            if (nodes > node_budget) { aborted = true; return; }

            if (k == n) {
                for (int x : need) if (x != 0) return;
                int total = mF + base_mines;
                auto& slot = res[total];
                slot.first += 1.0L;
                if (slot.second.empty()) slot.second.assign(n, 0.0L);
                for (int i=0;i<n;++i) if (assign[i]) slot.second[i] += 1.0L;
                return;
            }
            int v = order[k];

            // try 0
            bool ok=true;
            for (int ci : v2c[v]) if (need[ci] > rem[ci]-1) { ok=false; break; }
            if (ok) {
                assign[v]=0;
                for (int ci : v2c[v]) rem[ci]--;
                self(self, k+1, mF);
                for (int ci : v2c[v]) rem[ci]++;
            }

            // try 1
            ok=true;
            for (int ci : v2c[v]) if (need[ci]-1 < 0) { ok=false; break; }
            if (ok) {
                assign[v]=1;
                for (int ci : v2c[v]) { rem[ci]--; need[ci]--; }
                self(self, k+1, mF+1);
                for (int ci : v2c[v]) { rem[ci]++; need[ci]++; }
            }
        };

        rec(rec, 0, 0);
        if (aborted) return ComponentResult();
        return res;
    }

    ComponentResult solve_mc_is(int n, const std::vector<Constraint>& cons, int base_mines, int samples) {
        ComponentResult res;
        if (n==0) { res[base_mines] = {1.0L, {}}; return res; }

        std::vector<std::vector<int>> v2c(n);
        for (int ci=0;ci<(int)cons.size();++ci) for (int v : cons[ci].vars) v2c[v].push_back(ci);

        std::vector<int> order(n);
        {
            std::vector<std::pair<int,int>> deg(n);
            for (int v=0;v<n;++v) deg[v] = {-(int)v2c[v].size(), v};
            std::sort(deg.begin(), deg.end());
            for (int i=0;i<n;++i) order[i]=deg[i].second;
        }

        std::mt19937 rng(1337);
        std::uniform_real_distribution<double> U(0.0, 1.0);

        int got=0, trials=0;
        int cap = std::max(samples*5, 2000);

        while (got < samples && trials < cap) {
            trials++;
            std::vector<int> need(cons.size()), rem(cons.size());
            for (int i=0;i<(int)cons.size();++i) { need[i]=cons[i].val; rem[i]=(int)cons[i].vars.size(); }

            std::vector<int> assign(n,0);
            int mF = 0;
            long double q = 1.0L;
            bool ok_all = true;

            for (int idx=0; idx<n; ++idx) {
                int v = order[idx];
                bool ok0=true, ok1=true;
                for (int ci : v2c[v]) { if (need[ci] > rem[ci]-1) { ok0=false; break; } }
                for (int ci : v2c[v]) { if (need[ci]-1 < 0) { ok1=false; break; } }
                if (!ok0 && !ok1) { ok_all=false; break; }

                int val;
                if (ok0 && ok1) {
                    q *= 0.5L;
                    val = (U(rng) < 0.5) ? 1 : 0;
                } else {
                    val = ok1 ? 1 : 0;
                }
                assign[v] = val; mF += val;

                for (int ci : v2c[v]) {
                    rem[ci]--;
                    if (val) need[ci]--;
                }
            }
            if (!ok_all) continue;
            bool valid=true;
            for (int x : need) if (x!=0) { valid=false; break; }
            if (!valid) continue;

            long double iw = (q > 0) ? (1.0L / q) : 0.0L;
            int total = mF + base_mines;
            auto& slot = res[total];
            slot.first += iw;
            if (slot.second.empty()) slot.second.assign(n, 0.0L);
            for (int i=0;i<n;++i) if (assign[i]) slot.second[i] += iw;
            got++;
        }

        if (res.empty()) {
            res[base_mines] = {1.0L, std::vector<long double>(n, 0.0L)};
        }
        return res;
    }

    // ==========================================
    // ENDGAME SOLVER
    // ==========================================

    static constexpr int EG_TH_A = 15;
    static constexpr int EG_TH_B = 30;
    static constexpr int EG_WAYS_LIMIT = 1500;
    static constexpr uint64_t EG_WORK_BUDGET = 3000000; // 目标~0.2s，按环境调大/调小
    
    uint32_t eg_full_mask;
    std::vector<uint32_t> eg_belief_masks;     // size = B (<= EG_WAYS_LIMIT)
    std::vector<uint32_t> eg_neighbor_masks;   // size = N
    std::vector<uint8_t>  eg_neighbor_known;   // size = N

    std::unordered_map<uint64_t, double> endgame_memo; // memo
    std::unordered_map<uint64_t, int> endgame_memo_lines; // state -> best winning lines

    std::vector<uint8_t> eg_nbCnt; // size = B * eg_N, 每个belief下每个cell的相邻雷数(不含known)
    uint64_t eg_work = 0;
    int eg_N;
    bool eg_abort = false;

    struct CellHash {
    size_t operator()(const Cell& p) const noexcept {
        // pack (r,c) into 64-bit, then hash
        uint64_t x = (uint64_t)(uint32_t)p.first << 32 | (uint32_t)p.second;
        // splitmix64 finalizer (short, good enough)
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        x ^= (x >> 31);
        return (size_t)x;
    }
};

static inline int popcnt64(uint64_t x){ return __builtin_popcountll(x); }
static inline int ctz64(uint64_t x){ return __builtin_ctzll(x); }

static inline uint64_t mix64(uint64_t x){
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// 位集哈希：把所有word混进来（足够稳定做memo key）
static inline uint64_t hash_bitset(const std::vector<uint64_t>& bs){
    uint64_t h = 1469598103934665603ull;
    for (uint64_t w : bs) { h ^= mix64(w); h *= 1099511628211ull; }
    return h;
}

Cell solve_endgame(const std::vector<Cell>& unknown, int rem_mines_override) {
    eg_N = (int)unknown.size();
    int rem_mines = rem_mines_override;
    fprintf(stdout, "[AI][eg] enter solve_endgame N=%d rem=%d\n", eg_N, rem_mines);
    fflush(stdout);

    if (eg_N <= 0) return {-1,-1};
    if (eg_N > EG_TH_B) return {-1,-1};
    
    eg_full_mask = (eg_N==32)?0xFFFFFFFFu:((1u<<eg_N)-1u);


    if (rem_mines < 0 || rem_mines > eg_N) return {-1,-1};

    std::unordered_map<Cell,int,CellHash> u_idx;
    u_idx.reserve(eg_N*2);
    for (int i=0;i<eg_N;++i) u_idx[unknown[i]] = i;

    std::vector<std::pair<uint32_t,int>> cons;
    cons.reserve(knowledge.size());
    for (auto& s : knowledge) {
        uint32_t mask = 0;
        for (auto& c : s.cells) {
            auto it = u_idx.find(c);
            if (it != u_idx.end()) mask |= (1u << it->second);
        }
        if (mask) cons.push_back({mask, s.count});
    }

    eg_belief_masks.clear();
    eg_belief_masks.reserve(EG_WAYS_LIMIT);

    const uint32_t lim = (1u << eg_N);
    for (uint32_t m=0; m<lim; ++m) {
        if (__builtin_popcount(m) != rem_mines) continue;
        bool ok = true;
        for (auto& p : cons) {
            if ((int)__builtin_popcount(m & p.first) != p.second) { ok=false; break; }
        }
        if (ok) {
            eg_belief_masks.push_back(m);
            if ((int)eg_belief_masks.size() > EG_WAYS_LIMIT) break;
        }
    }
    if (eg_belief_masks.empty() || (int)eg_belief_masks.size() > EG_WAYS_LIMIT) return {-1,-1};

    // neighbor precompute
    eg_neighbor_masks.assign(eg_N, 0);
    eg_neighbor_known.assign(eg_N, 0);
    for (int i=0;i<eg_N;++i) {
        uint32_t mask=0; int known=0;
        int r0=unknown[i].first, c0=unknown[i].second;
        for (int r=r0-1;r<=r0+1;++r){
            if(r<0||r>=H) continue;
            for (int c=c0-1;c<=c0+1;++c){
                if(c<0||c>=W) continue;
                if(r==r0 && c==c0) continue;
                Cell nb={r,c};
                if (mines_found.count(nb)) known++;
                auto it = u_idx.find(nb);
                if (it != u_idx.end()) mask |= (1u << it->second);
            }
        }
        eg_neighbor_masks[i]=mask;
        eg_neighbor_known[i]=(uint8_t)known;
    }

    // nbCount precompute: nbCnt[bi*N + u] = popcount(belief_mines & neighbor_mask[u])
    const int B = (int)eg_belief_masks.size();
    eg_nbCnt.assign((size_t)B * (size_t)eg_N, 0);
    for (int bi=0; bi<B; ++bi) {
        uint32_t mines = eg_belief_masks[bi];
        for (int u=0; u<eg_N; ++u) {
            eg_nbCnt[(size_t)bi*eg_N + u] = (uint8_t)__builtin_popcount(mines & eg_neighbor_masks[u]);
        }
    }

    endgame_memo_lines.clear();
    eg_work = 0; eg_abort = false;

    const int W = (B + 63) / 64;
    std::vector<uint64_t> bset(W, ~0ull);
    if (B % 64) bset[W-1] = (1ull << (B % 64)) - 1ull;

    uint32_t all_mines = eg_full_mask;
    for (auto m : eg_belief_masks) all_mines &= m;

    int bestLines = -1, best_i = -1;

    // 顶层候选粗排：按safe_cnt降序（更快抬高cutoff）
    std::vector<std::pair<int,int>> cand; cand.reserve(eg_N);
    for (int i=0;i<eg_N;++i) if (!((all_mines>>i)&1u)) {
        int safe=0;
        for (int bi=0; bi<B; ++bi) safe += (((eg_belief_masks[bi]>>i)&1u)==0);
        cand.push_back({-safe, i});
    }
    std::sort(cand.begin(), cand.end());

    for (auto [negSafe, i] : cand) {
        int lines = solve_eg_rec(0u, bset, i, bestLines);
        if (eg_abort) return {-1,-1};
        if (lines > bestLines) { bestLines = lines; best_i = i; }
        if (bestLines >= B) break;
    }

    printf("[AI][eg][solver] beliefs=%d work=%llu best_prob=%f\n",
           B, (unsigned long long)eg_work, (double)bestLines / (double)B);
    fflush(stdout);

    return (best_i>=0)? unknown[best_i] : Cell{-1,-1};
}

int solve_eg_rec(uint32_t revealed, const std::vector<uint64_t>& beliefSet, int forcedMove, int cutoff) {
    if (eg_abort) return 0;
    if (eg_work > EG_WORK_BUDGET) { eg_abort = true; return 0; }

    const int B = (int)eg_belief_masks.size();
    const int W = (B + 63) / 64;

    // cnt + all/any（遍历子集）
    int cnt = 0;
    uint32_t all_m = eg_full_mask, any_m = 0;
    for (int wi=0; wi<W; ++wi) {
        uint64_t w = beliefSet[wi];
        cnt += popcnt64(w);
        while (w) {
            int b = ctz64(w), bi = wi*64 + b;
            uint32_t m = eg_belief_masks[bi];
            all_m &= m; any_m |= m;
            w &= (w - 1);
        }
    }
    if (cnt == 0) return 0;
    if (cnt <= cutoff) return cnt; // 上界：最多赢cnt条线

    uint32_t und = (~revealed) & eg_full_mask;
    uint32_t uncertain = und & (any_m ^ all_m);
    if (uncertain == 0) return 1; // JS语义：不可区分=>1条线

    uint64_t key = mix64(((uint64_t)revealed<<32) ^ hash_bitset(beliefSet));
    if (forcedMove < 0) {
        auto it = endgame_memo_lines.find(key);
        if (it != endgame_memo_lines.end()) return std::min(it->second, cnt);
    }

    auto eval_move = [&](int i, int localCut)->int{
        if ((all_m>>i)&1u) return 0;

        // safeSet
        std::vector<uint64_t> safeSet = beliefSet;
        int safe_cnt = 0;
        for (int wi=0; wi<W; ++wi) {
            uint64_t w = safeSet[wi], keep = 0;
            while (w) {
                int b = ctz64(w), bi = wi*64 + b;
                if (((eg_belief_masks[bi]>>i)&1u)==0u) keep |= (1ull<<b);
                w &= (w - 1);
            }
            safeSet[wi] = keep;
            safe_cnt += popcnt64(keep);
        }
        if (safe_cnt == 0) return 0;
        if (safe_cnt <= localCut) return safe_cnt; // move级上界剪枝

        struct Bucket { uint32_t rev; std::vector<uint64_t> bs; int sz; };
        std::unordered_map<uint64_t,int> idx;
        idx.reserve((size_t)safe_cnt*2);
        std::vector<Bucket> buckets; buckets.reserve(16);

        for (int wi=0; wi<W; ++wi) {
            uint64_t w = safeSet[wi];
            while (w) {
                eg_work++; if (eg_work > EG_WORK_BUDGET) { eg_abort = true; return 0; }

                int b = ctz64(w), bi = wi*64 + b;
                uint32_t mines = eg_belief_masks[bi];

                uint32_t curRev = revealed | (1u<<i);
                uint32_t qmask = (1u<<i), processed = 0;
                uint64_t fp = 0;

                while (qmask) {
                    int u = __builtin_ctz(qmask);
                    qmask &= (qmask - 1);
                    if ((processed>>u)&1u) continue;
                    processed |= (1u<<u);

                    // 查表：nb = known + nbCnt[bi*N+u]
                    int nb = (int)eg_neighbor_known[u] + (int)eg_nbCnt[(size_t)bi*eg_N + u];

                    uint64_t token = (uint64_t)(u & 31u) | ((uint64_t)(nb & 15u) << 6);
                    fp ^= mix64(token + 0x9e3779b97f4a7c15ull);

                    if (nb == 0) {
                        uint32_t add = eg_neighbor_masks[u] & ~curRev;
                        curRev |= add;
                        qmask |= add;
                    }
                }

                uint64_t k = mix64(((uint64_t)curRev<<32) ^ fp);
                auto it = idx.find(k);
                if (it == idx.end()) {
                    int id = (int)buckets.size();
                    idx.emplace(k, id);
                    buckets.push_back(Bucket{curRev, std::vector<uint64_t>(W,0ull), 0});
                    it = idx.find(k);
                }
                Bucket& buck = buckets[it->second];
                buck.bs[wi] |= (1ull<<b);
                buck.sz++;

                w &= (w - 1);
            }
        }

        std::sort(buckets.begin(), buckets.end(),
                  [](const Bucket& a, const Bucket& b){ return a.sz > b.sz; });

        int result = 0, remaining = safe_cnt;
        for (auto& buck : buckets) {
            remaining -= buck.sz;
            int child = solve_eg_rec(buck.rev, buck.bs, -1, std::max(localCut - result, 0));
            if (eg_abort) return 0;
            result += child;

            if (result + remaining <= localCut) return result + remaining; // 精确剪枝
            if (result >= safe_cnt) break;
        }
        return result;
    };

    int best = 0;
    if (forcedMove >= 0) return eval_move(forcedMove, cutoff);

    // 候选：只在uncertain中；粗排用safe_cnt（计算一次即可，避免重复）
    std::vector<std::pair<int,int>> cand;
    {
        uint32_t cm = uncertain;
        while (cm) {
            int i = __builtin_ctz(cm); cm &= (cm - 1);
            int safe = 0;
            for (int bi=0; bi<B; ++bi) {
                int wi = bi>>6, b = bi&63;
                if ((beliefSet[wi]>>b)&1ull) safe += (((eg_belief_masks[bi]>>i)&1u)==0);
            }
            cand.push_back({-safe, i});
        }
    }
    std::sort(cand.begin(), cand.end());

    for (auto [negSafe, i] : cand) {
        int wl = eval_move(i, best);
        if (eg_abort) return 0;
        if (wl > best) best = wl;
        if (best >= cnt) break;
    }

    endgame_memo_lines[key] = best;
    return best;
}

// 连通分支分块组件

struct DSU {
    std::vector<int> p, r;
    DSU(int n): p(n), r(n,0){ for(int i=0;i<n;i++) p[i]=i; }
    int f(int x){ while(p[x]!=x) x=p[x]=p[p[x]]; return x; }
    void u(int a,int b){
        a=f(a); b=f(b); if(a==b) return;
        if(r[a]<r[b]) std::swap(a,b);
        p[b]=a; if(r[a]==r[b]) r[a]++;
    }
};

static void split_components_by_constraints(
    const std::vector<Cell>& unknown,
    const std::vector<Sentence>& knowledge,   // 用你实际的类型替换
    std::vector<std::vector<Cell>>& comps_out
){
    const int N = (int)unknown.size();
    std::unordered_map<Cell,int,CellHash> idx; idx.reserve(N*2);
    for(int i=0;i<N;i++) idx[unknown[i]]=i;

    DSU dsu(N);

    for (auto const& s : knowledge) {
        int first = -1;
        for (auto const& c : s.cells) {
            auto it = idx.find(c);
            if (it == idx.end()) continue;
            if (first < 0) first = it->second;
            else dsu.u(first, it->second);
        }
    }

    std::unordered_map<int, std::vector<Cell>> mp;
    mp.reserve(N*2);
    for(int i=0;i<N;i++) mp[dsu.f(i)].push_back(unknown[i]);

    comps_out.clear();
    comps_out.reserve(mp.size());
    for (auto &kv : mp) comps_out.push_back(std::move(kv.second));
}

static bool component_ways_by_k_small(
    const std::vector<Cell>& comp,
    const std::vector<Sentence>& knowledge,
    std::vector<int>& waysByK,              // size = n+1
    int waysLimitPerK = 1000000             // 防炸：可按需
){
    const int n = (int)comp.size();
    waysByK.assign(n+1, 0);
    if (n==0) { waysByK[0]=1; return true; }
    if (n > EG_TH_A) return false;

    std::unordered_map<Cell,int,CellHash> idx; idx.reserve(n*2);
    for(int i=0;i<n;i++) idx[comp[i]]=i;

    struct Con { uint32_t mask; int cnt; };
    std::vector<Con> cons; cons.reserve(knowledge.size());

    for (auto const& s : knowledge) {
        uint32_t mask = 0;
        for (auto const& c : s.cells) {
            auto it = idx.find(c);
            if (it != idx.end()) mask |= (1u << it->second);
        }
        if (!mask) continue;
        int bits = __builtin_popcount(mask);
        if (s.count < 0 || s.count > bits) return false;
        cons.push_back({mask, s.count});
    }

    auto ok = [&](uint32_t m)->bool{
        for (auto const& con : cons)
            if (__builtin_popcount(m & con.mask) != con.cnt) return false;
        return true;
    };

    const uint32_t lim = (1u<<n);
    for (uint32_t m=0; m<lim; ++m) {
        if (!ok(m)) continue;
        int k = __builtin_popcount(m);
        int &w = waysByK[k];
        if (++w > waysLimitPerK) return false; // 太复杂，别把它当“小可独立组件”
    }
    return true;
}

static bool is_independent_solvable_small_component(
    const std::vector<Cell>& comp,
    const std::vector<Sentence>& knowledge,
    int& fixedMinesOut
){
    std::vector<int> waysByK;
    if (!component_ways_by_k_small(comp, knowledge, waysByK)) return false;
    int kFound = -1;
    for (int k=0;k<(int)waysByK.size();++k) if (waysByK[k] > 0) {
        if (kFound != -1) return false; // 不唯一
        kFound = k;
    }
    if (kFound < 0) return false; // 无解（矛盾）
    fixedMinesOut = kFound;
    return true;
}

int count_component_worlds_budgeted(const std::vector<Cell>& unknown, int rem_mines_override, int limit = EG_WAYS_LIMIT) const {
    const int N = (int)unknown.size();
    if (N == 0) return (rem_mines_override==0)?1:0;
    if (N > EG_TH_B) return limit + 1;

    const int rem_mines = rem_mines_override;
    if (rem_mines < 0 || rem_mines > N) return 0;

    std::unordered_map<Cell,int,CellHash> idx;
    for (int i = 0; i < N; ++i) idx[unknown[i]] = i;

    struct Con { uint32_t mask; int cnt; };
    std::vector<Con> cons;
    cons.reserve(knowledge.size());

    for (auto const& s : knowledge) {
        uint32_t mask = 0;
        for (auto const& c : s.cells) {
            auto it = idx.find(c);
            if (it != idx.end()) mask |= (1u << it->second);
        }
        if (!mask) continue;
        const int bits = __builtin_popcount(mask);
        if (s.count < 0 || s.count > bits) return 0;
        cons.push_back({mask, s.count});
    }

    auto check = [&](uint32_t m) -> bool {
        for (auto const& con : cons)
            if (__builtin_popcount(m & con.mask) != con.cnt) return false;
        return true;
    };

    if (rem_mines == 0) return check(0) ? 1 : 0;
    if (rem_mines == N) {
        uint32_t all = (N == 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);
        return check(all) ? 1 : 0;
    }

    uint32_t comb = (1u << rem_mines) - 1u;
    const uint32_t full = (N == 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);

    int ways = 0;
    while (true) {
        if (check(comb)) {
            if (++ways > limit) return limit + 1;
        }
        uint32_t x = comb & -comb;
        uint32_t y = comb + x;
        if (y == 0) break;
        uint32_t next = (((comb ^ y) >> 2) / x) | y;
        if (next & ~full) break;
        comb = next;
    }
    return ways;
}


Cell new_solve_endgame(const std::vector<Cell>& unknown) {
    const int globalRem = TotalMines - (int)mines_found.size();
    fprintf(stdout, "[AI][eg] try unknown=%d globalRem=%d knowledge=%zu\n",
            (int)unknown.size(), globalRem, knowledge.size());
    fflush(stdout);

    if (unknown.empty()) return {-1,-1};
    if (globalRem < 0) return {-1,-1};

    // 1) 分组件
    std::vector<std::vector<Cell>> comps;
    split_components_by_constraints(unknown, knowledge, comps);
    fprintf(stdout, "[AI][eg] split comps=%d\n", (int)comps.size());
    fflush(stdout);

    // 2) 剥离可独立小组件：记录下来，rest保留“非独立/大”的部分
    int fixedMines = 0;
    std::vector<Cell> rest;
    rest.reserve(unknown.size());

    struct SmallComp { std::vector<Cell> cells; int mines; };
    std::vector<SmallComp> solvables;
    solvables.reserve(comps.size());

    for (auto const& comp : comps) {
        int k = 0;
        bool solvable = ((int)comp.size() <= EG_TH_A) &&
                        is_independent_solvable_small_component(comp, knowledge, k);

        if (solvable) {
            fixedMines += k;
            solvables.push_back(SmallComp{comp, k});
            continue; // 剥离
        }
        rest.insert(rest.end(), comp.begin(), comp.end());
    }

    const int restN  = (int)rest.size();
    const int restRem = globalRem - fixedMines;

    fprintf(stdout, "[AI][eg] after peel: solvables=%d fixedMines=%d restN=%d restRem=%d\n",
            (int)solvables.size(), fixedMines, restN, restRem);
    fflush(stdout);

    if (restRem < 0 || restRem > restN) {
        fprintf(stdout, "[AI][eg] inconsistent: restRem out of range\n");
        fflush(stdout);
        return {-1,-1};
    }

    // 3) 如果剩余为空：优先解决一个小分支（你要求的行为）
    if (restN == 0) {
        if (solvables.empty()) {
            // 理论上不该发生：rest空但没有solvable组件
            fprintf(stdout, "[AI][eg] restN==0 but no solvables?!\n");
            fflush(stdout);
            return {-1,-1};
        }

        // 选择一个小分支：建议选 size 最大的（更“值得”先处理）
        int best = 0;
        for (int i=1;i<(int)solvables.size();++i) {
            if (solvables[i].cells.size() > solvables[best].cells.size())
                best = i;
        }

        fprintf(stdout, "[AI][eg] restN==0 -> solve one small comp: idx=%d size=%d mines=%d\n",
                best, (int)solvables[best].cells.size(), solvables[best].mines);
        fflush(stdout);

        // 单独对该组件调用 endgame（关键）
        return solve_endgame(solvables[best].cells, solvables[best].mines);
    }

    // 4) 否则：对剩余大分支按旧规则进入 endgame
    if (restN <= EG_TH_A) {
        fprintf(stdout, "[AI][eg][A*] restN=%d restRem=%d (fixedM=%d)\n", restN, restRem, fixedMines);
        fflush(stdout);
        return solve_endgame(rest, restRem);
    }

    if (restN <= EG_TH_B) {
        int ways = count_component_worlds_budgeted(rest, restRem, EG_WAYS_LIMIT);
        fprintf(stdout, "[AI][eg][B*] restN=%d restRem=%d fixedM=%d ways=%d\n",
                restN, restRem, fixedMines, ways);
        fflush(stdout);

        if (ways >= 1 && ways <= EG_WAYS_LIMIT) return solve_endgame(rest, restRem);
    }

    return {-1,-1};
}


// End of class

};

// ==========================================
// Globals & Interface
// ==========================================
Minesweeper* _GAME = nullptr;
MinesweeperAI* _AI = nullptr;

int _SEED_USED = 0;
int _FIRST_MV_MODE = 1;
bool _FIRST = false; 

val ms_get_state() {
    val res = val::object();
    if (!_GAME) return res;
    res.set("h", _GAME->H);
    res.set("w", _GAME->W);
    res.set("mines", _GAME->M);
    res.set("first", _GAME->first_move_made);
    res.set("firstmv", _FIRST_MV_MODE);
    res.set("lost", _GAME->lost);
    res.set("won", _GAME->won);
    res.set("seed", _SEED_USED);
    res.set("revealed_count", (int)_GAME->revealed.size());
    val mines_pos = val::array();
    for(auto& c : _GAME->mines) {
        val pt = val::array(); pt.call<void>("push", c.first); pt.call<void>("push", c.second);
        mines_pos.call<void>("push", pt);
    }
    res.set("mines_pos", mines_pos);
    val rev = val::array();
    for(auto& c : _GAME->revealed) {
        val item = val::array(); item.call<void>("push", c.first); item.call<void>("push", c.second); item.call<void>("push", _GAME->revealed_nums[c]);
        rev.call<void>("push", item);
    }
    res.set("revealed", rev);
    val ai_mines = val::array();
    if (_AI) for(auto& c : _AI->mines_found) {
        val pt = val::array(); pt.call<void>("push", c.first); pt.call<void>("push", c.second);
        ai_mines.call<void>("push", pt);
    }
    res.set("ai_mines", ai_mines);
    return res;
}


val ms_new_game(int h, int w, int m, val seed_val, int firstmv) {
    // 清理旧游戏
    if (_GAME) delete _GAME;
    if (_AI) delete _AI;
    
    // 重置第一次移动标记
    _FIRST = false;
    
    int seed;
    if (seed_val.isNull() || seed_val.isUndefined()) {
        // 生成随机种子
        seed = static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count() 
            % 2147483647);
        std::cout << "Generated random seed: " << seed << std::endl; // 调试用
    } else {
        try {
            seed = seed_val.as<int>();
            std::cout << "Using fixed seed: " << seed << std::endl; // 调试用
        } catch (...) {
            seed = static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count() 
                % 2147483647);
            std::cout << "Fallback to random seed: " << seed << std::endl; // 调试用
        }
    }
    
    _SEED_USED = seed;
    _FIRST_MV_MODE = firstmv;
    _GAME = new Minesweeper(h, w, m);
    _AI = new MinesweeperAI(h, w, m);
    
    val state = ms_get_state();
    state.set("seed", _SEED_USED);
    return state;
}



val ms_step_impl(Cell mv, bool from_ai_random) {
    val res = val::object();
    if (!_GAME) return res;
    if (mv.first == -1) {
        res.set("stuck", true); res.set("newly", val::array());
        return res;
    }

    if (!_GAME->first_move_made) _GAME->place_mines(mv, _FIRST_MV_MODE, _SEED_USED);

    auto newly_map = _GAME->reveal(mv);
    val newly_arr = val::array();
    bool hit_mine = false;
    for (auto const& [c, n] : newly_map) {
        val item = val::array(); item.call<void>("push", c.first); item.call<void>("push", c.second); item.call<void>("push", n);
        newly_arr.call<void>("push", item);
        if (n == -1) { _GAME->lost = true; hit_mine = true; } 
        else { _GAME->revealed.insert(c); _GAME->revealed_nums[c] = n; }
    }
    if (!hit_mine && _GAME->revealed.size() == (size_t)(_GAME->H * _GAME->W - _GAME->M)) _GAME->won = true;
    if (_AI && !hit_mine) _AI->add_knowledge(newly_map);

    res.set("move", val::array()); res["move"].call<void>("push", mv.first); res["move"].call<void>("push", mv.second);
    res.set("newly", newly_arr);
    val ai_mines = val::array();
    if (_AI) for(auto& c : _AI->mines_found) {
        val pt = val::array(); pt.call<void>("push", c.first); pt.call<void>("push", c.second);
        ai_mines.call<void>("push", pt);
    }
    res.set("ai_mines", ai_mines);
    res.set("lost", _GAME->lost);
    res.set("won", _GAME->won);
    res.set("revealed_count", (int)_GAME->revealed.size());
    res.set("stuck", false);
    return res;
}

val ms_make_safe_move() {
    if (!_AI) return val::object();
    Cell mv = _AI->make_safe_move();
    if (mv.first == -1) {
        val res = val::object();
        res.set("stuck", false); // Not technically stuck, just no safe move
        res.set("move", val::null());
        res.set("newly", val::array());
        res.set("ai_mines", val::array());
        if(_AI) for(auto& c : _AI->mines_found) {
             val pt = val::array(); pt.call<void>("push", c.first); pt.call<void>("push", c.second);
             res["ai_mines"].call<void>("push", pt);
        }
        res.set("lost", _GAME->lost); res.set("won", _GAME->won);
        res.set("revealed_count", (int)_GAME->revealed.size());
        return res;
    }
    return ms_step_impl(mv, false);
}

val ms_step() {
    if (!_AI) return val::object();
    Cell mv = _AI->make_random_move();
    return ms_step_impl(mv, true);
}

val ms_step_at(int r, int c) {
    return ms_step_impl({r, c}, false);
}

val ms_get_analysis() {
    val res = val::object();
    if (!_AI) return res;
    bool valid = false;
    std::map<Cell, double> probs = _AI->compute_probabilities(valid);
    val p_obj = val::object();
    for (auto const& [c, p] : probs) {
        std::string key = "(" + std::to_string(c.first) + "," + std::to_string(c.second) + ")";
        p_obj.set(key, p);
    }
    res.set("probs", p_obj);
    Cell mv = _AI->make_random_move();
    if (mv.first != -1) {
        val mv_arr = val::array(); mv_arr.call<void>("push", mv.first); mv_arr.call<void>("push", mv.second);
        res.set("next_move", mv_arr);
    } else {
        res.set("next_move", val::null());
    }
    return res;
}

val ms_version() { 
    return val("cpp-2025-12-23-EGDBG");
}



// ==========================================
// NEW FUNCTIONS
// ==========================================
val ms_load_board(val data) {
    if (_GAME) delete _GAME;
    if (_AI) delete _AI;

    int h = data["height"].as<int>();
    int w = data["width"].as<int>();
    int m = data["mines"].as<int>();

    _GAME = new Minesweeper(h, w, m);
    _AI = new MinesweeperAI(h, w, m);
    
    // 设置首步状态
    if (data.hasOwnProperty("first_move_made")) {
        _GAME->first_move_made = data["first_move_made"].as<bool>();
    } else {
        _GAME->first_move_made = true;
    }

    // 加载可见字段信息
    val field = data["field"];
    for (int r = 0; r < h; ++r) {
        std::string row = field[r].as<std::string>();
        for (int c = 0; c < w; ++c) {
            Cell cell = {r, c};
            char ch = row[c];
            if (ch == 'F') {
                _AI->mark_mine(cell);  // 告诉AI这里有雷（基于可见信息）
            } else if (isdigit(ch)) {
                int n = ch - '0';
                _GAME->revealed.insert(cell);
                _GAME->revealed_nums[cell] = n;
            }
        }
    }

    // 加载真实雷区布局（如果提供）
    if (data.hasOwnProperty("mines_layout")) {
        val mines_layout = data["mines_layout"];
        for (int r = 0; r < h; ++r) {
            val row = mines_layout[r];
            for (int c = 0; c < w; ++c) {
                int has_mine = row[c].as<int>();
                if (has_mine == 1) {
                    Cell cell = {r, c};
                    _GAME->mines.insert(cell);
                    _GAME->board[r][c] = true;
                }
            }
        }
    }

    // 将已揭示的数字添加到AI知识库
    std::map<Cell, int> nums;
    for (auto& kv : _GAME->revealed_nums) {
        nums[kv.first] = kv.second;
    }
    if (!nums.empty()) {
        _AI->add_knowledge(nums);
    }

    return ms_get_state();
}

val ms_board_info() {
    val res = val::object();
    if (!_GAME || _GAME->H <= 0 || _GAME->W <= 0) {
        res.set("error", "No active game");
        return res;
    }

    int h = _GAME->H, w = _GAME->W;
    
    // 构建可见字段信息
    std::vector<std::string> field(h, std::string(w, 'H'));
    for (auto& c : _AI->mines_found) {
        if (c.first >= 0 && c.first < h && c.second >= 0 && c.second < w) {
            field[c.first][c.second] = 'F';
        }
    }
    for (auto& kv : _GAME->revealed_nums) {
        Cell c = kv.first;
        if (c.first >= 0 && c.first < h && c.second >= 0 && c.second < w) {
            field[c.first][c.second] = '0' + kv.second;
        }
    }

    // 构建真实雷区布局
    val mines_layout = val::array();
    for (int r = 0; r < h; ++r) {
        val row = val::array();
        for (int c = 0; c < w; ++c) {
            int has_mine = _GAME->board[r][c] ? 1 : 0;
            row.call<void>("push", has_mine);
        }
        mines_layout.call<void>("push", row);
    }

    // 组装返回数据
    val field_arr = val::array();
    for (auto& row_str : field) {
        field_arr.call<void>("push", row_str);
    }
    
    res.set("height", h);
    res.set("width", w);
    res.set("mines", _GAME->M);
    res.set("seed", _SEED_USED);
    res.set("first_move_made", _GAME->first_move_made);
    res.set("field", field_arr);
    res.set("mines_layout", mines_layout);
    
    return res;
}

val ms_set_state(val st) {
    if (_GAME) delete _GAME;
    if (_AI) delete _AI;

    int h = st["h"].as<int>(), w = st["w"].as<int>(), m = st["mines"].as<int>();
    _GAME = new Minesweeper(h, w, m);
    _AI = new MinesweeperAI(h, w, m);

    _GAME->first_move_made = st["first"].as<bool>();
    _FIRST_MV_MODE = st["firstmv"].as<int>();
    _SEED_USED = st["seed"].as<int>();

    // 重建地雷信息（但不告诉 AI）
    val mines_pos = st["mines_pos"];
    for (unsigned i = 0; i < mines_pos["length"].as<unsigned>(); ++i) {
        val pt = mines_pos[i];
        Cell c = {pt[0].as<int>(), pt[1].as<int>()};
        _GAME->mines.insert(c);
        _GAME->board[c.first][c.second] = true;
        // 不要在这里告诉 AI
    }

    // 重建已揭示信息
    val revealed = st["revealed"];
    std::map<Cell, int> nums;
    for (unsigned i = 0; i < revealed["length"].as<unsigned>(); ++i) {
        val item = revealed[i];
        Cell c = {item[0].as<int>(), item[1].as<int>()};
        int n = item[2].as<int>();
        _GAME->revealed.insert(c);
        _GAME->revealed_nums[c] = n;
        if (n >= 0) nums[c] = n;
    }

    // 只告诉 AI 用户标记的地雷
    val ai_mines = st["ai_mines"];
    for (unsigned i = 0; i < ai_mines["length"].as<unsigned>(); ++i) {
        val pt = ai_mines[i];
        Cell c = {pt[0].as<int>(), pt[1].as<int>()};
        _AI->mark_mine(c);
    }

    if (!nums.empty()) _AI->add_knowledge(nums);
    return ms_get_state();
}


EMSCRIPTEN_BINDINGS(my_module) {
    emscripten::function("ms_version", &ms_version);
    emscripten::function("ms_new_game", &ms_new_game);
    emscripten::function("ms_get_state", &ms_get_state);
    emscripten::function("ms_step", &ms_step);
    emscripten::function("ms_step_at", &ms_step_at);
    emscripten::function("ms_make_safe_move", &ms_make_safe_move);
    emscripten::function("ms_get_analysis", &ms_get_analysis);

    // === 新增函数 ===
    emscripten::function("ms_load_board", &ms_load_board);
    emscripten::function("ms_board_info", &ms_board_info);
    emscripten::function("ms_set_state", &ms_set_state);
}
