
#include <stdint.h>
#ifndef SOLVER_ONLY
typedef struct __attribute__((packed)) {
    uint8_t first0, first1, first2, child_count;
    int8_t score_black;
    uint8_t move;
} Node;
extern const uint8_t db_nodes[];
#define NODE_COUNT 553050
static int8_t input_moves[64];
static int8_t output_options[36 * 3];
static inline const Node *node_at(uint32_t idx) { return (const Node *)(db_nodes + idx * 6u); }
static inline uint32_t first_child(const Node *n) {
    return (uint32_t)n->first0 | ((uint32_t)n->first1 << 8) | ((uint32_t)n->first2 << 16);
}
static inline int8_t decode_move(uint8_t m) { return (int8_t)m; }
__attribute__((export_name("input_ptr"))) int input_ptr(void) { return (int)(uintptr_t)input_moves; }
__attribute__((export_name("output_ptr"))) int output_ptr(void) { return (int)(uintptr_t)output_options; }
__attribute__((export_name("node_count"))) int node_count(void) { return NODE_COUNT; }
__attribute__((export_name("lookup_node"))) int lookup_node(int moves_ptr, int len) {
    if (len < 0 || len > 64) return -1;
    const int8_t *moves = (const int8_t *)(uintptr_t)moves_ptr;
    uint32_t idx = 0;
    for (int i = 0; i < len; i++) {
        const Node *n = node_at(idx);
        uint32_t start = first_child(n);
        int found = 0;
        for (uint32_t j = 0; j < n->child_count; j++) {
            uint32_t child_idx = start + j;
            if (child_idx >= NODE_COUNT) return -1;
            const Node *child = node_at(child_idx);
            if (decode_move(child->move) == moves[i]) {
                idx = child_idx; found = 1; break;
            }
        }
        if (!found) return -1;
    }
    return (int)idx;
}
__attribute__((export_name("get_node_score_black"))) int get_node_score_black(int moves_ptr, int len) {
    int idx = lookup_node(moves_ptr, len);
    if (idx < 0) return -128;
    return node_at((uint32_t)idx)->score_black;
}
__attribute__((export_name("get_black_best_move"))) int get_black_best_move(int moves_ptr, int len) {
    if ((len & 1) != 0) return -128;
    int idx = lookup_node(moves_ptr, len);
    if (idx < 0) return -128;
    const Node *n = node_at((uint32_t)idx);
    if (n->child_count == 0) return -128;
    return decode_move(node_at(first_child(n))->move);
}
__attribute__((export_name("get_white_options"))) int get_white_options(int moves_ptr, int len) {
    if ((len & 1) == 0) return -1;
    int idx = lookup_node(moves_ptr, len);
    if (idx < 0) return -1;
    const Node *n = node_at((uint32_t)idx);
    uint32_t start = first_child(n);
    for (uint32_t i = 0; i < n->child_count; i++) {
        const Node *child = node_at(start + i);
        int8_t reply = -128;
        if (child->child_count > 0) reply = decode_move(node_at(first_child(child))->move);
        output_options[i * 3] = decode_move(child->move);
        output_options[i * 3 + 1] = reply;
        output_options[i * 3 + 2] = child->score_black;
    }
    return n->child_count;
}

#endif /* SOLVER_ONLY */
#ifdef SOLVER_ONLY
static int8_t output_options[36 * 3];
__attribute__((export_name("output_ptr"))) int output_ptr(void) { return (int)(uintptr_t)output_options; }
#endif

/* Exact 6x6 endgame solver used when the opening tree reaches its depth boundary. */
#define FULL_MASK ((1ULL << 36) - 1ULL)
#define TT_BITS 18
#define TT_SIZE (1u << TT_BITS)
#define TT_MASK (TT_SIZE - 1u)
#define SEARCH_INF 1000
#define SEARCH_NODE_LIMIT 50000000u

typedef struct {
    uint64_t key;
    int16_t score;
    uint8_t depth;
    uint8_t flag;
    int8_t best_move;
    uint8_t pad[3];
} TTEntry;
static TTEntry transposition[TT_SIZE];
static int8_t search_board[36];
static uint32_t search_node_count = 0;
static int search_aborted = 0;
static int last_search_exact = 0;
static int last_search_score = 0;
static int8_t last_search_move = -1;

static const int8_t search_dx[8] = {-1,0,1,-1,1,-1,0,1};
static const int8_t search_dy[8] = {-1,-1,-1,0,0,1,1,1};

static inline int inside(int x, int y) { return x >= 0 && x < 6 && y >= 0 && y < 6; }
static inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }

static uint64_t flips_for(uint64_t own, uint64_t opp, int move) {
    int x0 = move % 6, y0 = move / 6;
    uint64_t flips = 0;
    for (int d = 0; d < 8; d++) {
        int x = x0 + search_dx[d], y = y0 + search_dy[d];
        uint64_t line = 0;
        while (inside(x, y)) {
            uint64_t bit = 1ULL << (y * 6 + x);
            if (opp & bit) { line |= bit; x += search_dx[d]; y += search_dy[d]; continue; }
            if ((own & bit) && line) flips |= line;
            break;
        }
    }
    return flips;
}

static uint64_t legal_moves(uint64_t black, uint64_t white, int side) {
    uint64_t own = side == 0 ? black : white;
    uint64_t opp = side == 0 ? white : black;
    uint64_t empty = FULL_MASK & ~(black | white);
    uint64_t out = 0;
    while (empty) {
        int sq = __builtin_ctzll(empty);
        empty &= empty - 1;
        if (flips_for(own, opp, sq)) out |= 1ULL << sq;
    }
    return out;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}
static uint64_t position_hash(uint64_t black, uint64_t white, int side) {
    uint64_t h = mix64(black + 0x9e3779b97f4a7c15ULL);
    h ^= mix64(white + 0x3c6ef372fe94f82aULL);
    h ^= side ? 0xd1b54a32d192ed03ULL : 0;
    return h ? h : 1ULL;
}

static int move_priority(int move, int tt_move) {
    if (move == tt_move) return 100000;
    if (move == 0 || move == 5 || move == 30 || move == 35) return 10000;
    int x = move % 6, y = move / 6;
    int score = (x == 0 || x == 5 || y == 0 || y == 5) ? 300 : 0;
    if ((x == 1 && y == 1) || (x == 4 && y == 1) || (x == 1 && y == 4) || (x == 4 && y == 4)) score -= 500;
    if ((x == 1 || x == 4) && (y == 0 || y == 5)) score -= 250;
    if ((x == 0 || x == 5) && (y == 1 || y == 4)) score -= 250;
    return score;
}

static int terminal_black_score(uint64_t black, uint64_t white) {
    int diff = popcount64(black) - popcount64(white);
    int empty = 36 - popcount64(black | white);
    if (diff > 0) return diff + empty;
    if (diff < 0) return diff - empty;
    return 0;
}
static int terminal_side_score(uint64_t black, uint64_t white, int side) {
    int black_score = terminal_black_score(black, white);
    return side == 0 ? black_score : -black_score;
}

static int negamax_exact(uint64_t black, uint64_t white, int side, int alpha, int beta, int empties, int8_t *best_out) {
    if (++search_node_count > SEARCH_NODE_LIMIT) {
        search_aborted = 1;
        if (best_out) *best_out = -1;
        return 0;
    }
    uint64_t moves = legal_moves(black, white, side);
    if (!moves) {
        uint64_t other = legal_moves(black, white, 1 - side);
        if (!other) {
            if (best_out) *best_out = -1;
            return terminal_side_score(black, white, side);
        }
        int8_t ignored = -1;
        int score = -negamax_exact(black, white, 1 - side, -beta, -alpha, empties, &ignored);
        if (best_out) *best_out = -1;
        return score;
    }
    if (empties <= 0) {
        if (best_out) *best_out = -1;
        return terminal_side_score(black, white, side);
    }

    int alpha_original = alpha;
    int beta_original = beta;
    uint64_t key = position_hash(black, white, side);
    TTEntry *entry = &transposition[(uint32_t)key & TT_MASK];
    int tt_move = -1;
    if (entry->key == key) {
        tt_move = entry->best_move;
        if (entry->depth >= empties) {
            if (entry->flag == 1) { if (best_out) *best_out = entry->best_move; return entry->score; }
            if (entry->flag == 2 && entry->score > alpha) alpha = entry->score;
            else if (entry->flag == 3 && entry->score < beta) beta = entry->score;
            if (alpha >= beta) { if (best_out) *best_out = entry->best_move; return entry->score; }
        }
    }

    int ordered[36], n = 0;
    uint64_t scan = moves;
    while (scan) { int sq = __builtin_ctzll(scan); scan &= scan - 1; ordered[n++] = sq; }
    for (int i = 1; i < n; i++) {
        int v = ordered[i], pv = move_priority(v, tt_move), j = i - 1;
        while (j >= 0 && move_priority(ordered[j], tt_move) < pv) { ordered[j + 1] = ordered[j]; j--; }
        ordered[j + 1] = v;
    }

    int best = -SEARCH_INF;
    int best_move = ordered[0];
    uint64_t own = side == 0 ? black : white;
    uint64_t opp = side == 0 ? white : black;
    for (int i = 0; i < n; i++) {
        int move = ordered[i];
        uint64_t bit = 1ULL << move;
        uint64_t flips = flips_for(own, opp, move);
        uint64_t next_black, next_white;
        if (side == 0) {
            next_black = black | bit | flips;
            next_white = white & ~flips;
        } else {
            next_black = black & ~flips;
            next_white = white | bit | flips;
        }
        int8_t ignored = -1;
        int score = -negamax_exact(next_black, next_white, 1 - side, -beta, -alpha, empties - 1, &ignored);
        if (score > best) { best = score; best_move = move; }
        if (best > alpha) alpha = best;
        if (alpha >= beta || search_aborted) break;
    }
    if (best_out) *best_out = (int8_t)best_move;
    if (!search_aborted) {
        entry->key = key;
        entry->score = (int16_t)best;
        entry->depth = (uint8_t)empties;
        entry->best_move = (int8_t)best_move;
        entry->flag = best <= alpha_original ? 3 : (best >= beta_original ? 2 : 1);
    }
    return best;
}

static void read_search_board(uint64_t *black, uint64_t *white) {
    *black = 0; *white = 0;
    for (int i = 0; i < 36; i++) {
        if (search_board[i] > 0) *black |= 1ULL << i;
        else if (search_board[i] < 0) *white |= 1ULL << i;
    }
}

__attribute__((export_name("search_board_ptr")))
int search_board_ptr(void) { return (int)(uintptr_t)search_board; }
__attribute__((export_name("solve_is_exact")))
int solve_is_exact(void) { return last_search_exact; }
__attribute__((export_name("solve_nodes")))
int solve_nodes(void) { return (int)search_node_count; }
__attribute__((export_name("solve_score_black")))
int solve_score_black(void) { return last_search_score; }
__attribute__((export_name("solve_black_position")))
int solve_black_position(void) {
    uint64_t black, white; read_search_board(&black, &white);
    int empties = 36 - popcount64(black | white);
    search_node_count = 0; search_aborted = 0;
    int8_t best = -1;
    int score = negamax_exact(black, white, 0, -SEARCH_INF, SEARCH_INF, empties, &best);
    last_search_move = best; last_search_score = score; last_search_exact = !search_aborted;
    return (int)best;
}
__attribute__((export_name("solve_white_options")))
int solve_white_options(void) {
    uint64_t black, white; read_search_board(&black, &white);
    int empties = 36 - popcount64(black | white);
    uint64_t moves = legal_moves(black, white, 1);
    search_node_count = 0; search_aborted = 0;
    int count = 0;
    if (!moves) {
        uint64_t other = legal_moves(black, white, 0);
        if (!other) { last_search_score = terminal_black_score(black, white); last_search_exact = 1; return 0; }
        int8_t reply = -1;
        int score = negamax_exact(black, white, 0, -SEARCH_INF, SEARCH_INF, empties, &reply);
        output_options[0] = -1; output_options[1] = reply; output_options[2] = (int8_t)score;
        last_search_score = score; last_search_exact = !search_aborted;
        return 1;
    }
    uint64_t scan = moves;
    while (scan && count < 36) {
        int move = __builtin_ctzll(scan); scan &= scan - 1;
        uint64_t flips = flips_for(white, black, move);
        uint64_t next_black = black & ~flips;
        uint64_t next_white = white | (1ULL << move) | flips;
        int8_t reply = -1;
        int score = negamax_exact(next_black, next_white, 0, -SEARCH_INF, SEARCH_INF, empties - 1, &reply);
        output_options[count * 3] = (int8_t)move;
        output_options[count * 3 + 1] = reply;
        output_options[count * 3 + 2] = (int8_t)score;
        count++;
        if (search_aborted) break;
    }
    last_search_score = count ? output_options[2] : 0;
    last_search_move = count ? output_options[0] : -1;
    last_search_exact = !search_aborted && count > 0;
    return count;
}
