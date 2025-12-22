#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>
#include <set>
#include <map>
#include <queue>
#include <random>
#include <algorithm>
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

    void place_mines(Cell safe_cell, int mode, int seed) {
        if (first_move_made) return;
        first_move_made = true;
        std::mt19937 gen(seed);
        
        std::set<Cell> safe_zone;
        safe_zone.insert(safe_cell);
        if (mode == 1) { // 3x3 Safe
            for (int r = safe_cell.first - 1; r <= safe_cell.first + 1; ++r)
                for (int c = safe_cell.second - 1; c <= safe_cell.second + 1; ++c)
                    if (is_valid(r, c)) safe_zone.insert({r, c});
        }
        // mode == 2 is 1x1 Safe (already inserted safe_cell)

        int cap = H * W - safe_zone.size();
        if (M > cap) M = cap; 

        int attempts = 0;
        while (mines.size() < (size_t)M) {
            attempts++;
            if (attempts > 1000000) break; // Safety break

            int r = gen() % H;
            int c = gen() % W;
            Cell cand = {r, c};
            if (safe_zone.count(cand) || mines.count(cand)) continue;
            mines.insert(cand);
            board[r][c] = true;
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
                    if (!visited.count(nb) && !board[nb.first][nb.second]) {
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
// AI Logic
// ==========================================
struct Constraint {
    std::vector<int> vars;
    int val;
    bool operator<(const Constraint& o) const { return vars < o.vars || (vars == o.vars && val < o.val); }
    bool operator==(const Constraint& o) const { return vars == o.vars && val == o.val; }
};

class MinesweeperAI {
public:
    int H, W, TotalMines;
    std::set<Cell> mines_found;
    std::set<Cell> safes_found;
    std::set<Cell> moves_made;
    
    struct Sentence {
        std::set<Cell> cells;
        int count;
    };
    std::vector<Sentence> knowledge;

    // Config from Python
    const int MAX_ENDGAME_SIZE = 15; 
    const double LOW_P_FIRST = 0.05;
    const double RISK_W = 10.0;
    const double P_TOL = 0.05;

    MinesweeperAI(int h, int w, int m) : H(h), W(w), TotalMines(m) {}

    void mark_mine(Cell c) {
        if (mines_found.count(c)) return;
        mines_found.insert(c);
    }

    void mark_safe(Cell c) {
        if (safes_found.count(c)) return;
        safes_found.insert(c);
    }

    void add_knowledge(std::map<Cell, int> revealed) {
        for (auto const& [c, val] : revealed) {
            if (val == -1) continue;
            moves_made.insert(c);
            mark_safe(c);
            
            std::set<Cell> unknown;
            int mines_nearby = 0;
            
            for (int r = c.first - 1; r <= c.first + 1; ++r) {
                if(r<0||r>=H) continue;
                for (int col = c.second - 1; col <= c.second + 1; ++col) {
                    if(col<0||col>=W) continue;
                    if(r==c.first && col==c.second) continue;
                    Cell n = {r, col};
                    if (mines_found.count(n)) mines_nearby++;
                    else if (!safes_found.count(n) && !moves_made.count(n)) unknown.insert(n);
                }
            }
            
            int new_count = val - mines_nearby;
            if (!unknown.empty()) {
                knowledge.push_back({unknown, new_count});
            }
        }
        simplify();
    }

    std::string get_signature(const std::set<Cell>& cells, int count) {
        std::stringstream ss;
        ss << count << "|";
        for (auto& c : cells) ss << c.first << "," << c.second << ";";
        return ss.str();
    }

    // Robust Simplify (Reconstruction Method)
    void simplify() {
        int passes = 0;
        bool changed = true;
        std::unordered_set<std::string> history;
        for(const auto& s : knowledge) history.insert(get_signature(s.cells, s.count));

        while (changed) {
            passes++;
            if (passes > 5) break; 
            changed = false;

            std::vector<Sentence> clean_knowledge;
            
            for (const auto& s : knowledge) {
                std::set<Cell> next_cells;
                int next_count = s.count;
                for (const auto& c : s.cells) {
                    if (mines_found.count(c)) next_count--;
                    else if (!safes_found.count(c)) next_cells.insert(c);
                }

                if (next_cells.empty()) continue;
                if (next_count == 0) {
                    for (const auto& c : next_cells) {
                        if (!safes_found.count(c)) { mark_safe(c); changed = true; }
                    }
                    continue; 
                } else if (next_count == (int)next_cells.size()) {
                    for (const auto& c : next_cells) {
                        if (!mines_found.count(c)) { mark_mine(c); changed = true; }
                    }
                    continue; 
                }
                clean_knowledge.push_back({next_cells, next_count});
            }

            if (changed) { knowledge = clean_knowledge; continue; }

            std::vector<Sentence> unique_knowledge;
            std::unordered_set<std::string> current_pass_sigs;
            std::sort(clean_knowledge.begin(), clean_knowledge.end(), [](const Sentence& a, const Sentence& b){
                return a.cells.size() < b.cells.size();
            });

            for (const auto& s : clean_knowledge) {
                std::string sig = get_signature(s.cells, s.count);
                if (current_pass_sigs.find(sig) == current_pass_sigs.end()) {
                    current_pass_sigs.insert(sig);
                    unique_knowledge.push_back(s);
                }
            }
            knowledge = unique_knowledge;

            if (knowledge.size() > 100) break; 

            std::vector<Sentence> new_inferred;
            for (size_t i = 0; i < knowledge.size(); ++i) {
                const auto& A = knowledge[i];
                for (size_t j = i + 1; j < knowledge.size(); ++j) {
                    const auto& B = knowledge[j];
                    if (A.cells.size() == B.cells.size()) continue;
                    if (B.cells.size() > A.cells.size() + 6) continue;

                    if (std::includes(B.cells.begin(), B.cells.end(), A.cells.begin(), A.cells.end())) {
                        int d_count = B.count - A.count;
                        int d_size = B.cells.size() - A.cells.size();
                        if (d_count >= 0 && d_count <= d_size && d_size > 0) {
                            std::set<Cell> diff;
                            std::set_difference(B.cells.begin(), B.cells.end(), A.cells.begin(), A.cells.end(), std::inserter(diff, diff.begin()));
                            std::string sig = get_signature(diff, d_count);
                            if (history.find(sig) == history.end()) {
                                history.insert(sig);
                                new_inferred.push_back({diff, d_count});
                                changed = true;
                            }
                        }
                    }
                    if (new_inferred.size() > 10) break;
                }
                if (new_inferred.size() > 10) break;
            }
            if (!new_inferred.empty()) knowledge.insert(knowledge.end(), new_inferred.begin(), new_inferred.end());
        }
    }

    Cell make_safe_move() {
        for(auto& c : safes_found) {
            if (!moves_made.count(c)) return c;
        }
        return {-1, -1};
    }

    int info_gain_heuristic(Cell cell) {
        int unk = 0;
        for (int r = cell.first - 1; r <= cell.first + 1; ++r) {
            if(r<0||r>=H) continue;
            for (int c = cell.second - 1; c <= cell.second + 1; ++c) {
                if(c<0||c>=W) continue;
                if(r==cell.first && c==cell.second) continue;
                Cell nb = {r, c};
                if (!moves_made.count(nb) && !safes_found.count(nb) && !mines_found.count(nb)) unk++;
            }
        }
        int deg = 0;
        for (auto& s : knowledge) {
            if (s.cells.count(cell)) deg++;
        }
        return 3 * deg + unk;
    }

    Cell make_random_move() {
        Cell safe = make_safe_move();
        if (safe.first != -1) return safe;

        std::vector<Cell> unknown;
        for (int r=0; r<H; ++r) for(int c=0; c<W; ++c) {
            Cell cell = {r,c};
            if (!moves_made.count(cell) && !mines_found.count(cell) && !safes_found.count(cell)) {
                unknown.push_back(cell);
            }
        }
        if (unknown.empty()) return {-1, -1};

        bool valid = false;
        std::map<Cell, double> probs = compute_probabilities(valid);
        if (probs.empty()) return unknown[0]; // Should not happen if unknown not empty

        std::vector<std::pair<double, Cell>> cand;
        for(auto& kv : probs) {
            if (!moves_made.count(kv.first) && !mines_found.count(kv.first)) {
                cand.push_back({kv.second, kv.first});
            }
        }
        if (cand.empty()) return {-1, -1};
        std::sort(cand.begin(), cand.end());

        double min_p = cand[0].first;

        // Strategy 1: Low Probability Priority (<= 5%)
        if (min_p <= LOW_P_FIRST) {
            std::vector<Cell> low_pool;
            for(auto& p : cand) if (p.first <= LOW_P_FIRST) low_pool.push_back(p.second);
            
            // Find best in low pool
            double best_p_low = 1.0;
            for(auto& c : low_pool) if (probs[c] < best_p_low) best_p_low = probs[c];
            
            std::vector<Cell> best_pool;
            for(auto& c : low_pool) if (std::abs(probs[c] - best_p_low) <= 1e-9) best_pool.push_back(c);

            Cell best_c = best_pool[0];
            std::tuple<int, int, int> best_score = {-1, 0, 0};

            for(auto& c : best_pool) {
                int ig = info_gain_heuristic(c);
                std::tuple<int, int, int> score = {ig, -c.first, -c.second};
                if (score > best_score) { best_score = score; best_c = c; }
            }
            return best_c;
        }

        // Strategy 2: Endgame Solver
        if (unknown.size() <= (size_t)MAX_ENDGAME_SIZE) {
            Cell eg_move = solve_endgame(unknown);
            if (eg_move.first != -1) return eg_move;
        }

        // Strategy 3: Standard Info Gain with Risk Weight
        std::vector<Cell> band;
        for(auto& p : cand) if (p.first <= min_p + P_TOL) band.push_back(p.second);
        
        if (band.size() == 1) return band[0];

        Cell best_c = band[0];
        double best_score = -1e100;

        for(auto& c : band) {
            double p = probs[c];
            double gain = (double)info_gain_heuristic(c);
            double risk_term = (p - min_p) / std::max(1e-9, P_TOL);
            double score = gain - RISK_W * risk_term;
            if (score > best_score || (std::abs(score - best_score) < 1e-12 && p < probs[best_c])) {
                best_score = score;
                best_c = c;
            }
        }
        return best_c;
    }

    // ==========================================
    // PROBABILITY ENGINE
    // ==========================================
    typedef std::map<int, std::pair<double, std::vector<double>>> ComponentResult;

    std::map<Cell, double> compute_probabilities(bool& valid_global) {
        std::vector<Cell> unknown;
        std::set<Cell> frontier_set;
        for (int r=0; r<H; ++r) for(int c=0; c<W; ++c) {
            Cell cell = {r,c};
            if (!moves_made.count(cell) && !mines_found.count(cell) && !safes_found.count(cell)) {
                unknown.push_back(cell);
            }
        }
        if (unknown.empty()) { valid_global = true; return {}; }

        for (auto& s : knowledge) for (auto& c : s.cells) frontier_set.insert(c);
        
        std::vector<Cell> frontier;
        std::vector<Cell> outside;
        for (auto& c : unknown) {
            if (frontier_set.count(c)) frontier.push_back(c);
            else outside.push_back(c);
        }

        auto components = find_components(frontier);
        std::vector<ComponentResult> comp_results;
        for (auto& comp : components) comp_results.push_back(solve_component(comp));

        int mines_left = TotalMines - mines_found.size();
        if (mines_left < 0) mines_left = 0;

        std::map<int, double> global_dist; global_dist[0] = 1.0;
        for (auto& res : comp_results) {
            std::map<int, double> next_dist;
            for (auto const& [k1, w1] : global_dist) {
                for (auto const& [k2, pair_w] : res) {
                    if (k1 + k2 <= mines_left) next_dist[k1 + k2] += w1 * pair_w.first;
                }
            }
            global_dist = next_dist;
        }

        double Z = 0.0;
        int n_out = outside.size();
        for (auto const& [k_front, w_front] : global_dist) {
            int k_out = mines_left - k_front;
            if (k_out >= 0 && k_out <= n_out) Z += w_front * nCr(n_out, k_out);
        }

        std::map<Cell, double> final_probs;
        if (Z < 1e-9) { 
            valid_global = false; 
            double p = (unknown.size() > 0) ? (double)mines_left / (double)unknown.size() : 0.0;
            for(auto& c : unknown) final_probs[c] = p;
            return final_probs;
        }

        valid_global = true;
        for (size_t i = 0; i < comp_results.size(); ++i) {
            std::map<int, double> rest_dist; rest_dist[0] = 1.0;
            for (size_t j = 0; j < comp_results.size(); ++j) {
                if (i == j) continue;
                std::map<int, double> next;
                for (auto const& [k1, w1] : rest_dist) {
                    for (auto const& [k2, pair_w] : comp_results[j]) {
                        if (k1 + k2 <= mines_left) next[k1 + k2] += w1 * pair_w.first;
                    }
                }
                rest_dist = next;
            }

            auto& comp_res = comp_results[i];
            auto& comp_cells = components[i];
            
            for (auto const& [k_local, val] : comp_res) {
                double w_local = val.first;
                const std::vector<double>& cell_counts = val.second;
                if (w_local <= 1e-12) continue;

                double ways_rest_out = 0.0;
                int target = mines_left - k_local;
                for (auto const& [k_rest, w_rest] : rest_dist) {
                    int k_out = target - k_rest;
                    if (k_out >= 0 && k_out <= n_out) ways_rest_out += w_rest * nCr(n_out, k_out);
                }

                if (ways_rest_out <= 0) continue;
                double factor = ways_rest_out / Z;
                for (size_t c_idx = 0; c_idx < comp_cells.size(); ++c_idx) {
                    final_probs[comp_cells[c_idx]] += cell_counts[c_idx] * factor;
                }
            }
        }

        double avg_out = 0.0;
        for (auto const& [k_front, w_front] : global_dist) {
            int k_out = mines_left - k_front;
            if (k_out >= 0 && k_out <= n_out) {
                double prob_split = (w_front * nCr(n_out, k_out)) / Z;
                avg_out += k_out * prob_split;
            }
        }
        double p_out = (n_out > 0) ? avg_out / n_out : 0.0;
        for (auto& c : outside) final_probs[c] = p_out;
        return final_probs;
    }

private:
    std::vector<std::vector<Cell>> find_components(const std::vector<Cell>& frontier) {
        std::map<Cell, int> c_to_idx;
        for(size_t i=0; i<frontier.size(); ++i) c_to_idx[frontier[i]] = i;
        std::vector<std::vector<int>> adj(frontier.size());
        for (auto& s : knowledge) {
            std::vector<int> idxs;
            for (auto& c : s.cells) if (c_to_idx.count(c)) idxs.push_back(c_to_idx[c]);
            for (size_t i=0; i<idxs.size(); ++i) for (size_t j=i+1; j<idxs.size(); ++j) {
                adj[idxs[i]].push_back(idxs[j]); adj[idxs[j]].push_back(idxs[i]);
            }
        }
        std::vector<std::vector<Cell>> comps;
        std::vector<bool> visited(frontier.size(), false);
        for (size_t i=0; i<frontier.size(); ++i) {
            if (visited[i]) continue;
            std::vector<Cell> comp;
            std::queue<int> q; q.push(i); visited[i]=true;
            while(!q.empty()){
                int u = q.front(); q.pop();
                comp.push_back(frontier[u]);
                for (int v : adj[u]) if (!visited[v]) { visited[v]=true; q.push(v); }
            }
            comps.push_back(comp);
        }
        return comps;
    }

    ComponentResult solve_component(const std::vector<Cell>& cells) {
        int n = cells.size();
        if (n <= 12) return solve_exact(n, get_constraints(cells), 0);
        
        auto [reduced_cells, reduced_cons, knowns, fixed] = gaussian_reduce(cells);
        int n_rem = reduced_cells.size();
        ComponentResult res;
        if (n_rem == 0) res[fixed] = {1.0, std::vector<double>()};
        else if (n_rem <= 20) res = solve_exact(n_rem, reduced_cons, fixed);
        else res = solve_mc(n_rem, reduced_cons, fixed, 40000);

        std::map<Cell, int> cell_to_idx; for(int i=0; i<n; ++i) cell_to_idx[cells[i]] = i;
        std::map<Cell, int> reduced_map; for(int i=0; i<n_rem; ++i) reduced_map[reduced_cells[i]] = i;

        ComponentResult final_res;
        for (auto& [m, val] : res) {
            std::vector<double> full_counts(n, 0.0);
            for (auto const& [c, k_val] : knowns) if (k_val == 1) full_counts[cell_to_idx[c]] = val.first;
            for (int i=0; i<n_rem; ++i) full_counts[cell_to_idx[reduced_cells[i]]] = val.second[i];
            final_res[m] = {val.first, full_counts};
        }
        return final_res;
    }

    std::vector<Constraint> get_constraints(const std::vector<Cell>& cells) {
        std::map<Cell, int> lookup; for(size_t i=0; i<cells.size(); ++i) lookup[cells[i]] = i;
        std::set<Cell> scope(cells.begin(), cells.end());
        std::vector<Constraint> cons;
        for (auto& s : knowledge) {
            bool subset = true; for (auto& c : s.cells) if (!scope.count(c)) { subset=false; break; }
            if (subset) {
                std::vector<int> vars; for(auto& c : s.cells) vars.push_back(lookup[c]);
                cons.push_back({vars, s.count});
            }
        }
        std::sort(cons.begin(), cons.end());
        cons.erase(unique(cons.begin(), cons.end()), cons.end());
        return cons;
    }

    std::tuple<std::vector<Cell>, std::vector<Constraint>, std::map<Cell, int>, int> 
    gaussian_reduce(const std::vector<Cell>& cells) {
        int n = cells.size();
        std::map<Cell, int> c_map; for(int i=0; i<n; ++i) c_map[cells[i]] = i;
        std::set<Cell> scope(cells.begin(), cells.end());
        std::vector<std::vector<int>> mat;
        for (auto& s : knowledge) {
            bool ok = true; for(auto& c : s.cells) if(!scope.count(c)) { ok=false; break; }
            if(ok && !s.cells.empty()) {
                std::vector<int> row(n + 1, 0); for(auto& c : s.cells) row[c_map[c]] = 1;
                row[n] = s.count; mat.push_back(row);
            }
        }
        int rows = mat.size(), piv = 0;
        for (int col = 0; col < n && piv < rows; ++col) {
            int sel = -1; for (int r = piv; r < rows; ++r) if (mat[r][col]) { sel = r; break; }
            if (sel == -1) continue;
            std::swap(mat[piv], mat[sel]);
            for (int r = 0; r < rows; ++r) if (r != piv && mat[r][col]) for (int k = col; k <= n; ++k) mat[r][k] -= mat[piv][k];
            piv++;
        }
        std::map<Cell, int> knowns; int fixed_mines = 0;
        for (auto& row : mat) {
            int nz = 0, nzi = -1;
            for (int i=0; i<n; ++i) if (row[i]) { nz++; nzi = i; }
            if (nz == 1) {
                int res = row[n] / row[nzi];
                if (res == 0 || res == 1) { knowns[cells[nzi]] = res; if (res) fixed_mines++; }
            }
        }
        std::vector<Cell> reduced_cells; for(auto& c : cells) if(!knowns.count(c)) reduced_cells.push_back(c);
        std::map<Cell, int> r_map; for(size_t i=0; i<reduced_cells.size(); ++i) r_map[reduced_cells[i]] = i;
        std::vector<Constraint> reduced_cons;
        for (auto& s : knowledge) {
            bool relevant = false; for(auto& c : s.cells) if(scope.count(c)) { relevant=true; break; }
            if (!relevant) continue;
            int count = s.count; std::vector<int> vars; bool valid = true;
            for(auto& c : s.cells) {
                if (knowns.count(c)) count -= knowns[c];
                else if (r_map.count(c)) vars.push_back(r_map[c]);
                else { valid=false; break; }
            }
            if (valid && count >= 0 && !vars.empty()) {
                std::sort(vars.begin(), vars.end()); reduced_cons.push_back({vars, count});
            }
        }
        std::sort(reduced_cons.begin(), reduced_cons.end());
        reduced_cons.erase(unique(reduced_cons.begin(), reduced_cons.end()), reduced_cons.end());
        return {reduced_cells, reduced_cons, knowns, fixed_mines};
    }

    ComponentResult solve_exact(int n, const std::vector<Constraint>& cons, int base_mines) {
        std::vector<std::vector<int>> v2c(n);
        for(size_t i=0; i<cons.size(); ++i) for(int v : cons[i].vars) v2c[v].push_back(i);
        std::vector<int> need(cons.size()), rem(cons.size());
        for(size_t i=0; i<cons.size(); ++i) { need[i]=cons[i].val; rem[i]=cons[i].vars.size(); }
        std::vector<int> assign(n);
        std::vector<std::pair<int,int>> deg(n); for(int i=0; i<n; ++i) deg[i] = {-(int)v2c[i].size(), i};
        std::sort(deg.begin(), deg.end()); std::vector<int> order(n); for(int i=0; i<n; ++i) order[i] = deg[i].second;
        ComponentResult res; 
        auto rec = [&](auto&& self, int k, int mF) -> void {
            if (k == n) {
                for (int x : need) if (x != 0) return;
                int total = mF + base_mines;
                res[total].first += 1.0;
                if (res[total].second.empty()) res[total].second.resize(n, 0.0);
                for(int i=0; i<n; ++i) if(assign[i]) res[total].second[i] += 1.0;
                return;
            }
            int v = order[k];
            // Try 0
            bool ok = true; for(int ci : v2c[v]) if(need[ci] > rem[ci]-1) { ok=false; break; }
            if (ok) {
                assign[v]=0; for(int ci : v2c[v]) rem[ci]--;
                self(self, k+1, mF);
                for(int ci : v2c[v]) rem[ci]++;
            }
            // Try 1
            ok = true; for(int ci : v2c[v]) if(need[ci]-1 < 0) { ok=false; break; }
            if (ok) {
                assign[v]=1; for(int ci : v2c[v]) { rem[ci]--; need[ci]--; }
                self(self, k+1, mF+1);
                for(int ci : v2c[v]) { rem[ci]++; need[ci]++; }
            }
        };
        rec(rec, 0, 0); return res;
    }

    ComponentResult solve_mc(int n, const std::vector<Constraint>& cons, int base_mines, int samples) {
        if (n == 0) { ComponentResult r; r[base_mines] = {1.0, {}}; return r; }
        std::vector<std::vector<int>> v2c(n); for(size_t i=0; i<cons.size(); ++i) for(int v : cons[i].vars) if(v>=0&&v<n) v2c[v].push_back(i);
        ComponentResult res;
        std::mt19937 rng(1337); std::uniform_real_distribution<double> dist(0.0, 1.0);
        int trials = 0, got = 0, max_trials = std::min(samples * 10, 80000); 
        while (got < samples && trials < max_trials) {
            trials++;
            std::vector<int> need(cons.size()), rem(cons.size());
            for(size_t i=0; i<cons.size(); ++i) { need[i]=cons[i].val; rem[i]=cons[i].vars.size(); }
            std::vector<int> assign(n, 0); bool possible = true;
            for (int v = 0; v < n; v++) {
                bool z = false, o = false;
                for(int ci : v2c[v]) { if(need[ci] >= rem[ci]) o = true; if(need[ci] <= 0) z = true; }
                if (z && o) { possible = false; break; }
                int val = o ? 1 : (z ? 0 : (dist(rng) < 0.25)); 
                assign[v] = val;
                for(int ci : v2c[v]) { rem[ci]--; if(val) need[ci]--; }
            }
            if (possible) {
                bool valid = true; for(size_t i=0; i<cons.size(); ++i) if(need[i] != 0) { valid = false; break; }
                if (valid) {
                    got++; int mF = 0; for(int val : assign) if(val) mF++;
                    int total = mF + base_mines;
                    res[total].first += 1.0;
                    if (res[total].second.empty()) res[total].second.resize(n, 0.0);
                    for(int i=0; i<n; ++i) if(assign[i]) res[total].second[i] += 1.0;
                }
            }
        }
        if (res.empty()) res[base_mines] = {1.0, std::vector<double>(n, 0.0)};
        return res;
    }

    // ==========================================
    // ENDGAME SOLVER
    // ==========================================
    std::map<std::pair<int, std::vector<int>>, double> endgame_memo;
    std::vector<int> eg_belief; int eg_N, eg_full_mask;
    std::vector<int> eg_neighbor_masks, eg_neighbor_known;
    
    Cell solve_endgame(const std::vector<Cell>& unknown) {
        eg_N = unknown.size(); eg_full_mask = (1 << eg_N) - 1;
        int rem_mines = TotalMines - mines_found.size();
        if (rem_mines < 0 || rem_mines > eg_N) return {-1, -1};

        std::map<Cell, int> u_idx; for(int i=0; i<eg_N; ++i) u_idx[unknown[i]] = i;
        std::vector<std::pair<int, int>> cons;
        for(auto& s : knowledge) {
            int mask = 0; for(auto& c : s.cells) if (u_idx.count(c)) mask |= (1 << u_idx[c]);
            if (mask) cons.push_back({mask, s.count});
        }

        eg_belief.clear();
        for (int m = 0; m <= eg_full_mask; ++m) {
            if (__builtin_popcount(m) != rem_mines) continue;
            bool ok = true;
            for (auto& p : cons) if (__builtin_popcount(m & p.first) != p.second) { ok = false; break; }
            if (ok) eg_belief.push_back(m);
        }
        if (eg_belief.empty()) return {-1, -1};

        eg_neighbor_masks.assign(eg_N, 0); eg_neighbor_known.assign(eg_N, 0);
        for (int i = 0; i < eg_N; ++i) {
            int mask = 0, known = 0;
            for (int r = unknown[i].first - 1; r <= unknown[i].first + 1; ++r) {
                if(r<0||r>=H) continue;
                for (int c = unknown[i].second - 1; c <= unknown[i].second + 1; ++c) {
                    if(c<0||c>=W) continue;
                    if(r==unknown[i].first && c==unknown[i].second) continue;
                    Cell nb = {r, c};
                    if (mines_found.count(nb)) known++;
                    if (u_idx.count(nb)) mask |= (1 << u_idx[nb]);
                }
            }
            eg_neighbor_masks[i] = mask; eg_neighbor_known[i] = known;
        }

        endgame_memo.clear();
        double best_p = -1.0; int best_i = -1;
        std::vector<int> all_indices(eg_belief.size()); for(size_t i=0; i<eg_belief.size(); ++i) all_indices[i] = i;
        int all_mines_mask = eg_full_mask; for(int m : eg_belief) all_mines_mask &= m;
        
        for (int i = 0; i < eg_N; ++i) {
            if ((all_mines_mask >> i) & 1) continue; 
            std::map<std::pair<int, std::vector<std::pair<int,int>>>, std::vector<int>> outcomes;
            for (int b_idx : all_indices) {
                int mines = eg_belief[b_idx];
                if ((mines >> i) & 1) continue;
                int current_rev = (1 << i);
                std::queue<int> q; q.push(i); int processed = 0;
                std::vector<std::pair<int, int>> obs; 
                while(!q.empty()){
                    int u = q.front(); q.pop();
                    if ((processed >> u) & 1) continue;
                    processed |= (1 << u);
                    int nb_mines = eg_neighbor_known[u] + __builtin_popcount(mines & eg_neighbor_masks[u]);
                    obs.push_back({u, nb_mines});
                    if (nb_mines == 0) {
                        int nbs = eg_neighbor_masks[u];
                        for(int k=0; k<eg_N; ++k) if ((nbs >> k) & 1) if (!((current_rev >> k) & 1)) { current_rev |= (1 << k); q.push(k); }
                    }
                }
                std::sort(obs.begin(), obs.end());
                outcomes[{current_rev, obs}].push_back(b_idx);
            }
            double p_win = 0.0;
            for (auto& kv : outcomes) p_win += ((double)kv.second.size() / eg_belief.size()) * solve_eg_rec(kv.first.first, kv.second);
            if (p_win > best_p) { best_p = p_win; best_i = i; }
        }
        if (best_i != -1) return unknown[best_i];
        return {-1, -1};
    }

    double solve_eg_rec(int revealed, const std::vector<int>& belief_indices) {
        if (belief_indices.empty()) return 0.0;
        bool all_done = true;
        for (int idx : belief_indices) if ((~revealed & ~eg_belief[idx]) & eg_full_mask) { all_done = false; break; }
        if (all_done) return 1.0;
        std::pair<int, std::vector<int>> key = {revealed, belief_indices};
        if (endgame_memo.count(key)) return endgame_memo[key];
        int all_mines = eg_full_mask; for (int idx : belief_indices) all_mines &= eg_belief[idx];
        int possible_moves = (~revealed) & eg_full_mask & (~all_mines);
        if (possible_moves == 0) return endgame_memo[key] = (all_done ? 1.0 : 0.0);
        double max_p = 0.0;
        for (int i = 0; i < eg_N; ++i) {
            if (!((possible_moves >> i) & 1)) continue;
            std::map<std::pair<int, std::vector<std::pair<int,int>>>, std::vector<int>> outcomes;
            for (int b_idx : belief_indices) {
                int mines = eg_belief[b_idx];
                if ((mines >> i) & 1) continue;
                int current_rev = revealed | (1 << i);
                std::queue<int> q; q.push(i); int processed = 0; 
                std::vector<std::pair<int, int>> obs; 
                while(!q.empty()){
                    int u = q.front(); q.pop();
                    if ((processed >> u) & 1) continue;
                    processed |= (1 << u);
                    int nb_mines = eg_neighbor_known[u] + __builtin_popcount(mines & eg_neighbor_masks[u]);
                    obs.push_back({u, nb_mines});
                    if (nb_mines == 0) {
                        int nbs = eg_neighbor_masks[u];
                        for(int k=0; k<eg_N; ++k) if ((nbs >> k) & 1) if (!((current_rev >> k) & 1)) { current_rev |= (1 << k); q.push(k); }
                    }
                }
                std::sort(obs.begin(), obs.end());
                outcomes[{current_rev, obs}].push_back(b_idx);
            }
            double p_move = 0.0;
            for (auto& kv : outcomes) p_move += ((double)kv.second.size() / belief_indices.size()) * solve_eg_rec(kv.first.first, kv.second);
            if (p_move > max_p) max_p = p_move;
        }
        return endgame_memo[key] = max_p;
    }
};

// ==========================================
// Globals & Interface
// ==========================================
Minesweeper* _GAME = nullptr;
MinesweeperAI* _AI = nullptr;
int _SEED_USED = 0;
int _FIRST_MV_MODE = 1;

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
    if (_GAME) delete _GAME;
    if (_AI) delete _AI;
    int seed;
    if (seed_val.isNull() || seed_val.isUndefined()) seed = (int)std::time(NULL);
    else seed = seed_val.as<int>();
    _SEED_USED = seed;
    _FIRST_MV_MODE = firstmv;
    _GAME = new Minesweeper(h, w, m);
    _AI = new MinesweeperAI(h, w, m);
    return ms_get_state();
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
    return val("cpp-2025-12-22-SIMPLE-WORKING");
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

    val mines_pos = st["mines_pos"];
    for (unsigned i = 0; i < mines_pos["length"].as<unsigned>(); ++i) {
        val pt = mines_pos[i];
        Cell c = {pt[0].as<int>(), pt[1].as<int>()};
        _GAME->mines.insert(c);
        _GAME->board[c.first][c.second] = true;
        _AI->mark_mine(c);
    }

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

