#include <stdint.h>
#include <stddef.h>

#define BLACK 1
#define WHITE 2
#define MAX_TREE_BLOCKS 8192
#define MAX_REPLIES 36
#define NO_DB_SCORE 127

typedef struct {
    int8_t cell[36];
    int8_t side;
} Board;

static const uint8_t *tree_bytes = 0;
static const int8_t *node_bytes = 0;
static uint32_t tree_length = 0;
static uint32_t tree_bits = 0;
static uint32_t nodes = 0;
static uint32_t rank512[MAX_TREE_BLOCKS + 1];
static int8_t reply_buffer[MAX_REPLIES * 2];
static uint32_t heap_top = 0;

extern unsigned char __heap_base;

static const int8_t DX[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
static const int8_t DY[8] = {0, 1, 1, 1, 0, -1, -1, -1};

__attribute__((visibility("default")))
uint32_t alloc(uint32_t size) {
    if (heap_top == 0) {
        heap_top = ((uint32_t)(uintptr_t)&__heap_base + 7u) & ~7u;
    }
    uint32_t result = heap_top;
    uint32_t end = result + size;
    uint32_t current_pages = __builtin_wasm_memory_size(0);
    uint32_t needed_pages = (end + 65535u) >> 16;
    if (needed_pages > current_pages) {
        uint32_t grow = needed_pages - current_pages;
        if (__builtin_wasm_memory_grow(0, grow) == (size_t)-1) {
            return 0;
        }
    }
    heap_top = end;
    return result;
}

static uint32_t popcount64(uint64_t x) {
    return (uint32_t)__builtin_popcountll(x);
}

static uint64_t load64(const uint8_t *p) {
    return ((uint64_t)p[0]) |
           ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
}

static uint32_t rank1(uint32_t pos) {
    uint32_t block = pos >> 9;
    uint32_t offset = pos & 511u;
    uint32_t result = rank512[block];
    uint32_t words = offset >> 6;
    const uint8_t *p = tree_bytes + block * 64u;
    for (uint32_t i = 0; i < words; i++) {
        result += popcount64(load64(p + i * 8u));
    }
    uint32_t rem = offset & 63u;
    if (rem) {
        uint64_t word = load64(p + words * 8u);
        result += popcount64(word & ((((uint64_t)1) << rem) - 1u));
    }
    return result;
}

static uint32_t rank0(uint32_t pos) {
    return pos - rank1(pos);
}

/* Returns the bit position immediately after the k-th zero (1-based). */
static uint32_t select0(uint32_t k) {
    uint32_t lo = 0;
    uint32_t hi = tree_bits;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (rank0(mid) < k) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static int testbit(uint32_t pos) {
    return (tree_bytes[pos >> 3] >> (pos & 7u)) & 1u;
}

static uint32_t child_count(uint32_t node_id) {
    uint32_t p = select0(node_id);
    uint32_t count = 0;
    while (p + count < tree_bits && testbit(p + count)) count++;
    return count;
}

static uint32_t child_id(uint32_t node_id, uint32_t child_index) {
    uint32_t p = select0(node_id) + 1u + child_index;
    return rank1(p);
}

static int node_move(uint32_t node_id) {
    if (node_id == 0 || node_id > nodes) return -2;
    return (uint8_t)node_bytes[(node_id - 1u) * 2u];
}

static int node_value(uint32_t node_id) {
    if (node_id == 0 || node_id > nodes) return 0;
    return (int8_t)node_bytes[(node_id - 1u) * 2u + 1u];
}

__attribute__((visibility("default")))
int init_db(uint32_t tree_ptr, uint32_t tree_len,
            uint32_t data_ptr, uint32_t data_len) {
    if (tree_len == 0 || data_len < 2 || (data_len & 1u)) return 0;
    if ((tree_len & 63u) != 0) return 0;
    tree_bytes = (const uint8_t *)(uintptr_t)tree_ptr;
    node_bytes = (const int8_t *)(uintptr_t)data_ptr;
    tree_length = tree_len;
    tree_bits = tree_len * 8u;
    nodes = data_len / 2u;
    uint32_t blocks = tree_bits / 512u;
    if (blocks > MAX_TREE_BLOCKS) return 0;

    uint32_t total = 0;
    for (uint32_t i = 0; i < blocks; i++) {
        rank512[i] = total;
        const uint8_t *p = tree_bytes + i * 64u;
        for (uint32_t j = 0; j < 8; j++) {
            total += popcount64(load64(p + j * 8u));
        }
    }
    rank512[blocks] = total;
    if (total != nodes) return 0; /* Includes the dummy root edge. */
    return 1;
}

static void init_board(Board *b) {
    for (int i = 0; i < 36; i++) b->cell[i] = 0;
    b->cell[14] = WHITE;
    b->cell[15] = BLACK;
    b->cell[20] = BLACK;
    b->cell[21] = WHITE;
    b->side = BLACK;
}

static int can_place(const Board *b, int side, int idx) {
    if (idx < 0 || idx >= 36 || b->cell[idx] != 0) return 0;
    int other = side == BLACK ? WHITE : BLACK;
    int x = idx % 6;
    int y = idx / 6;
    for (int d = 0; d < 8; d++) {
        int cx = x + DX[d];
        int cy = y + DY[d];
        if (cx < 0 || cx >= 6 || cy < 0 || cy >= 6) continue;
        int p = cy * 6 + cx;
        if (b->cell[p] != other) continue;
        cx += DX[d];
        cy += DY[d];
        while (cx >= 0 && cx < 6 && cy >= 0 && cy < 6) {
            p = cy * 6 + cx;
            if (b->cell[p] == 0) break;
            if (b->cell[p] == side) return 1;
            cx += DX[d];
            cy += DY[d];
        }
    }
    return 0;
}

/* Legal placements only. Returns zero when a pass is required. */
static int legal_places(const Board *b, int side, int out[36]) {
    int n = 0;
    for (int idx = 0; idx < 36; idx++) {
        if (can_place(b, side, idx)) out[n++] = idx;
    }
    return n;
}

/* Legal moves including a forced pass as the single move -1. */
static int legal_moves(const Board *b, int side, int out[36]) {
    int n = legal_places(b, side, out);
    if (n == 0) {
        out[0] = -1;
        return 1;
    }
    return n;
}

static int apply_move(Board *b, int move) {
    int side = b->side;
    if (move == -1) {
        if (legal_places(b, side, (int[36]){0}) != 0) return 0;
        b->side = (int8_t)(side == BLACK ? WHITE : BLACK);
        return 1;
    }
    if (!can_place(b, side, move)) return 0;

    int other = side == BLACK ? WHITE : BLACK;
    int x = move % 6;
    int y = move / 6;
    int flips[36];
    int flip_count = 0;

    for (int d = 0; d < 8; d++) {
        int cx = x + DX[d];
        int cy = y + DY[d];
        int line[6];
        int line_count = 0;
        while (cx >= 0 && cx < 6 && cy >= 0 && cy < 6) {
            int p = cy * 6 + cx;
            if (b->cell[p] == other) {
                line[line_count++] = p;
            } else if (b->cell[p] == side) {
                if (line_count > 0) {
                    for (int j = 0; j < line_count; j++) flips[flip_count++] = line[j];
                }
                break;
            } else {
                break;
            }
            cx += DX[d];
            cy += DY[d];
        }
    }

    if (flip_count == 0) return 0;
    b->cell[move] = (int8_t)side;
    for (int i = 0; i < flip_count; i++) b->cell[flips[i]] = (int8_t)side;
    b->side = (int8_t)(side == BLACK ? WHITE : BLACK);
    return 1;
}

/* The four opening moves are symmetry-equivalent. Map each to canonical B3. */
static int choose_transform(int first_move) {
    switch (first_move) {
        case 8:  return 1; /* C2 -> B3: transpose */
        case 13: return 0; /* B3 -> B3: identity */
        case 22: return 2; /* E4 -> B3: 180-degree rotation */
        case 27: return 3; /* D5 -> B3: anti-diagonal reflection */
        default: return -1;
    }
}

static int transform_move(int move, int transform) {
    if (move < 0) return -1;
    int x = move % 6;
    int y = move / 6;
    int nx = x;
    int ny = y;
    switch (transform) {
        case 1: nx = y;     ny = x;     break;
        case 2: nx = 5 - x; ny = 5 - y; break;
        case 3: nx = 5 - y; ny = 5 - x; break;
        default: break;
    }
    return ny * 6 + nx;
}

static int db_move_index(int move) {
    return move < 0 ? 36 : move;
}

/*
 * Replay the history. Black moves are checked against the stored optimal move.
 * White moves select one child by their row-major legal-move index.
 * On an unrecognized Black branch, board replay continues but db_ok becomes 0.
 */
static int walk_history(const int8_t *moves, int len, Board *board,
                        int *transform_out, int *db_ok_out,
                        uint32_t *cursor_out) {
    if (len < 0 || len > 64) return 0;
    int transform = len == 0 ? 0 : choose_transform((int)moves[0]);
    int db_ok = transform >= 0;
    if (transform < 0) transform = 0;

    init_board(board);
    uint32_t cursor = 1;

    for (int i = 0; i < len; i++) {
        int raw = (int)moves[i];
        if (raw < -1 || raw >= 36) return 0;
        int move = transform_move(raw, transform);

        if (board->side == BLACK) {
            if (db_ok && node_move(cursor) != db_move_index(move)) db_ok = 0;
        } else {
            if (db_ok) {
                int legal[36];
                int count = legal_moves(board, WHITE, legal);
                int child_index = -1;
                for (int j = 0; j < count; j++) {
                    if (legal[j] == move) {
                        child_index = j;
                        break;
                    }
                }
                if (child_index < 0 || child_count(cursor) != (uint32_t)count) {
                    db_ok = 0;
                } else {
                    cursor = child_id(cursor, (uint32_t)child_index);
                    if (cursor == 0 || cursor > nodes) db_ok = 0;
                }
            }
        }

        if (!apply_move(board, move)) return 0;
    }

    *transform_out = transform;
    *db_ok_out = db_ok;
    *cursor_out = cursor;
    return 1;
}


/* Exact 6x6 negamax fallback. The opening DB covers the common prefix; this
 * transposition-table search is used only when neither opening tree applies. */
typedef struct {
    uint64_t me;
    uint64_t opp;
} FastBoard;

typedef struct {
    uint64_t key;
    int8_t value;
    int8_t best_move;
    uint8_t depth;
    uint8_t flag; /* 0 exact, 1 lower bound, 2 upper bound */
} TTEntry;

typedef struct {
    int8_t move;
    int32_t order;
    FastBoard next;
} OrderedMove;

#define TT_BITS 17u
#define TT_SIZE (1u << TT_BITS)
#define TT_MASK (TT_SIZE - 1u)
#define BOARD_MASK ((((uint64_t)1) << 36) - 1u)

static TTEntry transposition[TT_SIZE];
static int tt_enabled = 1;
static int award_unplayed = 0;

static const uint64_t COL0_MASK =
    (((uint64_t)1) << 0) | (((uint64_t)1) << 6) |
    (((uint64_t)1) << 12) | (((uint64_t)1) << 18) |
    (((uint64_t)1) << 24) | (((uint64_t)1) << 30);
static const uint64_t COL5_MASK =
    (((uint64_t)1) << 5) | (((uint64_t)1) << 11) |
    (((uint64_t)1) << 17) | (((uint64_t)1) << 23) |
    (((uint64_t)1) << 29) | (((uint64_t)1) << 35);
static const uint64_t ROW0_MASK = (((uint64_t)1) << 6) - 1u;
static const uint64_t ROW5_MASK = ROW0_MASK << 30;

static uint64_t shift_dir(uint64_t bits, int dir) {
    switch (dir) {
        case 0: return (bits & ~COL0_MASK) >> 1; /* west */
        case 1: return (bits & ~(COL0_MASK | ROW5_MASK)) << 5; /* southwest */
        case 2: return (bits & ~ROW5_MASK) << 6; /* south */
        case 3: return (bits & ~(COL5_MASK | ROW5_MASK)) << 7; /* southeast */
        case 4: return (bits & ~COL5_MASK) << 1; /* east */
        case 5: return (bits & ~(COL5_MASK | ROW0_MASK)) >> 5; /* northeast */
        case 6: return (bits & ~ROW0_MASK) >> 6; /* north */
        default: return (bits & ~(COL0_MASK | ROW0_MASK)) >> 7; /* northwest */
    }
}

static uint64_t fast_legal_moves(FastBoard b) {
    uint64_t empty = BOARD_MASK & ~(b.me | b.opp);
    uint64_t moves = 0;
    for (int d = 0; d < 8; d++) {
        uint64_t x = shift_dir(b.me, d) & b.opp;
        for (int k = 0; k < 4; k++) x |= shift_dir(x, d) & b.opp;
        moves |= shift_dir(x, d) & empty;
    }
    return moves;
}

static uint64_t fast_flips(FastBoard b, int move) {
    uint64_t pos = ((uint64_t)1) << move;
    uint64_t flips = 0;
    for (int d = 0; d < 8; d++) {
        uint64_t x = shift_dir(pos, d) & b.opp;
        for (int k = 0; k < 4; k++) x |= shift_dir(x, d) & b.opp;
        if (shift_dir(x, d) & b.me) flips |= x;
    }
    return flips;
}

static FastBoard fast_apply(FastBoard b, int move) {
    FastBoard next;
    if (move < 0) {
        next.me = b.opp;
        next.opp = b.me;
    } else {
        uint64_t pos = ((uint64_t)1) << move;
        uint64_t flips = fast_flips(b, move);
        next.me = b.opp ^ flips;
        next.opp = b.me | flips | pos;
    }
    return next;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static uint64_t board_key(FastBoard b, int pass) {
    uint64_t a = mix64(b.me + 0x9e3779b97f4a7c15ULL);
    uint64_t c = mix64(b.opp + 0xd1b54a32d192ed03ULL);
    uint64_t key = a ^ ((c << 23) | (c >> 41));
    if (pass) key ^= 0xa0761d6478bd642fULL;
    return key ? key : 1u;
}

static int positional_order(int move) {
    int x = move % 6;
    int y = move / 6;
    if ((x == 0 || x == 5) && (y == 0 || y == 5)) return 100;
    if (x == 0 || x == 5 || y == 0 || y == 5) return 20;
    if ((x == 1 || x == 4) && (y == 1 || y == 4)) return -20;
    return 0;
}

static int terminal_value(FastBoard b) {
    int me = (int)__builtin_popcountll(b.me);
    int opp = (int)__builtin_popcountll(b.opp);
    int diff = me - opp;
    if (award_unplayed && diff != 0) {
        int empty = 36 - me - opp;
        diff += diff > 0 ? empty : -empty;
    }
    return diff;
}

static int search_debug_checks = 0;
static int search_debug_error = 0;

static int debug_check_fast_board(FastBoard b, uint64_t fast_moves) {
    Board cell_board;
    for (int i = 0; i < 36; i++) {
        uint64_t bit = ((uint64_t)1) << i;
        cell_board.cell[i] = (b.me & bit) ? BLACK : ((b.opp & bit) ? WHITE : 0);
    }
    cell_board.side = BLACK;
    int legal[36];
    int count = legal_places(&cell_board, BLACK, legal);
    uint64_t slow_moves = 0;
    for (int i = 0; i < count; i++) slow_moves |= ((uint64_t)1) << legal[i];
    if (slow_moves != fast_moves) return 0;

    for (int i = 0; i < count; i++) {
        Board next_cell = cell_board;
        if (!apply_move(&next_cell, legal[i])) return 0;
        FastBoard next_fast = fast_apply(b, legal[i]);
        uint64_t next_black = 0;
        uint64_t next_white = 0;
        for (int j = 0; j < 36; j++) {
            uint64_t bit = ((uint64_t)1) << j;
            if (next_cell.cell[j] == BLACK) next_black |= bit;
            else if (next_cell.cell[j] == WHITE) next_white |= bit;
        }
        if (next_fast.me != next_white || next_fast.opp != next_black) return 0;
    }
    return 1;
}

static int negamax(FastBoard b, int pass, int depth, int alpha, int beta) {
    uint64_t moves = fast_legal_moves(b);
    if (search_debug_checks && !debug_check_fast_board(b, moves)) search_debug_error = 1;
    if (moves == 0) {
        if (pass) {
            return terminal_value(b);
        }
        FastBoard swapped;
        swapped.me = b.opp;
        swapped.opp = b.me;
        return -negamax(swapped, 1, depth, -beta, -alpha);
    }
    if (depth <= 0) {
        return terminal_value(b);
    }

    uint64_t key = board_key(b, pass);
    TTEntry *entry = &transposition[(uint32_t)(key ^ (key >> 32)) & TT_MASK];
    int original_alpha = alpha;
    int original_beta = beta;
    int tt_move = -1;

    if (tt_enabled && entry->depth != 0 && entry->key == key) {
        tt_move = entry->best_move;
        if (entry->depth >= depth) {
            if (entry->flag == 0) return entry->value;
            if (entry->flag == 1 && entry->value >= beta) return entry->value;
            if (entry->flag == 2 && entry->value <= alpha) return entry->value;
        }
    }

    OrderedMove ordered[36];
    int count = 0;
    uint64_t bits = moves;
    while (bits) {
        int move = __builtin_ctzll(bits);
        bits &= bits - 1u;
        FastBoard next = fast_apply(b, move);
        int mobility = (int)__builtin_popcountll(fast_legal_moves(next));
        int order = positional_order(move) * 100 - mobility;
        if (move == tt_move) order += 100000;
        ordered[count].move = (int8_t)move;
        ordered[count].order = order;
        ordered[count].next = next;
        count++;
    }

    /* Stable insertion sort: corners/edges, TT move, then low opponent mobility. */
    for (int i = 1; i < count; i++) {
        OrderedMove current = ordered[i];
        int j = i;
        while (j > 0 && ordered[j - 1].order < current.order) {
            ordered[j] = ordered[j - 1];
            j--;
        }
        ordered[j] = current;
    }

    int best = -100;
    int best_move = -1;
    for (int i = 0; i < count; i++) {
        int value = -negamax(ordered[i].next, 0, depth - 1, -beta, -alpha);
        if (value > best) {
            best = value;
            best_move = ordered[i].move;
        }
        if (value > alpha) alpha = value;
        if (alpha >= beta) break;
    }

    if (tt_enabled && (entry->depth == 0 || entry->key != key || depth >= entry->depth)) {
        entry->key = key;
        entry->value = (int8_t)best;
        entry->best_move = (int8_t)best_move;
        entry->depth = (uint8_t)depth;
        if (best <= original_alpha) entry->flag = 2;
        else if (best >= original_beta) entry->flag = 1;
        else entry->flag = 0;
    }
    return best;
}

static FastBoard to_fast_board(const Board *b) {
    uint64_t black = 0;
    uint64_t white = 0;
    for (int i = 0; i < 36; i++) {
        if (b->cell[i] == BLACK) black |= ((uint64_t)1) << i;
        else if (b->cell[i] == WHITE) white |= ((uint64_t)1) << i;
    }
    FastBoard result;
    if (b->side == BLACK) {
        result.me = black;
        result.opp = white;
    } else {
        result.me = white;
        result.opp = black;
    }
    return result;
}

static int pack_result(int move, int value, int transform) {
    int original_move = move < 0 ? -1 : transform_move(move, transform);
    uint32_t packed = ((uint32_t)(uint8_t)value << 8) |
                      (uint32_t)(original_move + 2);
    return (int32_t)packed;
}

__attribute__((visibility("default")))
int32_t debug_apply_match(uint32_t moves_ptr, uint32_t len, int move) {
    Board board;
    int transform = 0;
    int db_ok = 0;
    uint32_t cursor = 1;
    if (!walk_history((const int8_t *)(uintptr_t)moves_ptr, (int)len,
                      &board, &transform, &db_ok, &cursor)) return 0;
    Board next_cell = board;
    if (!apply_move(&next_cell, move)) return 0;
    FastBoard fast = to_fast_board(&board);
    FastBoard next_fast = fast_apply(fast, move);
    FastBoard from_cell = to_fast_board(&next_cell);
    return next_fast.me == from_cell.me && next_fast.opp == from_cell.opp;
}

__attribute__((visibility("default")))
void set_award_unplayed(int enabled) {
    award_unplayed = enabled;
}

__attribute__((visibility("default")))
void set_search_debug(int enabled) {
    search_debug_checks = enabled;
    search_debug_error = 0;
}

__attribute__((visibility("default")))
int32_t get_search_debug_error(void) {
    return search_debug_error;
}

__attribute__((visibility("default")))
void set_tt_enabled(int enabled) {
    tt_enabled = enabled;
}

__attribute__((visibility("default")))
void clear_transposition(void) {
    for (uint32_t i = 0; i < TT_SIZE; i++) transposition[i].depth = 0;
}

__attribute__((visibility("default")))
int32_t debug_movegen_match(uint32_t moves_ptr, uint32_t len) {
    Board board;
    int transform = 0;
    int db_ok = 0;
    uint32_t cursor = 1;
    if (!walk_history((const int8_t *)(uintptr_t)moves_ptr, (int)len,
                      &board, &transform, &db_ok, &cursor)) return 0;
    int legal[36];
    int count = legal_places(&board, board.side, legal);
    uint64_t cell_mask = 0;
    for (int i = 0; i < count; i++) cell_mask |= ((uint64_t)1) << legal[i];
    FastBoard fast = to_fast_board(&board);
    return cell_mask == fast_legal_moves(fast);
}

__attribute__((visibility("default")))
uint64_t debug_fast_legal_mask(uint32_t moves_ptr, uint32_t len) {
    Board board;
    int transform = 0;
    int db_ok = 0;
    uint32_t cursor = 1;
    if (!walk_history((const int8_t *)(uintptr_t)moves_ptr, (int)len,
                      &board, &transform, &db_ok, &cursor)) return 0;
    FastBoard fast = to_fast_board(&board);
    return fast_legal_moves(fast);
}

__attribute__((visibility("default")))
int32_t search_black(uint32_t moves_ptr, uint32_t len) {
    Board board;
    int transform = 0;
    int db_ok = 0;
    uint32_t cursor = 1;
    if (!walk_history((const int8_t *)(uintptr_t)moves_ptr, (int)len,
                      &board, &transform, &db_ok, &cursor)) {
        return 0;
    }
    if (board.side != BLACK) return 0;

    FastBoard fast = to_fast_board(&board);
    uint64_t legal = fast_legal_moves(fast);
    int depth = 36 - (int)__builtin_popcountll(fast.me | fast.opp);

    if (legal == 0) {
        FastBoard swapped;
        swapped.me = fast.opp;
        swapped.opp = fast.me;
        int value = -negamax(swapped, 1, depth, -100, 100);
        return pack_result(-1, value, transform);
    }

    uint64_t key = board_key(fast, 0);
    int value = negamax(fast, 0, depth, -100, 100);
    TTEntry *entry = &transposition[(uint32_t)(key ^ (key >> 32)) & TT_MASK];
    int best_move = (tt_enabled && entry->key == key) ? entry->best_move : -1;
    if (best_move < 0 || best_move >= 36) {
        /* Defensive fallback: search legal moves with full windows. */
        int best = -100;
        uint64_t bits = legal;
        while (bits) {
            int move = __builtin_ctzll(bits);
            bits &= bits - 1u;
            int candidate = -negamax(fast_apply(fast, move), 0,
                                     depth - 1, -100, 100);
            if (candidate > best) {
                best = candidate;
                best_move = move;
            }
        }
        value = best;
    }
    return pack_result(best_move, value, transform);
}

/* Packed return: zero = database miss; low byte = move + 2; next byte = signed score. */
__attribute__((visibility("default")))
int32_t black_lookup(uint32_t moves_ptr, uint32_t len) {
    if (len & 1u) return 0;
    Board board;
    int transform = 0;
    int db_ok = 0;
    uint32_t cursor = 1;
    if (!walk_history((const int8_t *)(uintptr_t)moves_ptr, (int)len,
                      &board, &transform, &db_ok, &cursor) || !db_ok) {
        return 0;
    }

    int move = node_move(cursor);
    if (move < 0 || move > 36) return 0;
    int original_move = move == 36 ? -1 : transform_move(move, transform);
    int value = node_value(cursor);
    uint32_t packed = ((uint32_t)(uint8_t)value << 8) |
                      (uint32_t)(original_move + 2);
    return (int32_t)packed;
}

/*
 * For a position after Black's move, return every legal White reply.
 * reply_buffer contains [white_move, black_value] pairs.
 * A value of 127 means the Black database does not cover that reply.
 */
__attribute__((visibility("default")))
int white_replies(uint32_t moves_ptr, uint32_t len) {
    if ((len & 1u) == 0 || len == 0) return -1;

    Board board;
    int transform = 0;
    int db_ok = 0;
    uint32_t cursor = 1;
    if (!walk_history((const int8_t *)(uintptr_t)moves_ptr, (int)len,
                      &board, &transform, &db_ok, &cursor)) {
        return -1;
    }
    if (board.side != WHITE) return -1;

    int legal[36];
    int count = legal_moves(&board, WHITE, legal);
    if (db_ok && child_count(cursor) != (uint32_t)count) db_ok = 0;

    for (int i = 0; i < count; i++) {
        int move = legal[i];
        int original_move = transform_move(move, transform);
        int value = NO_DB_SCORE;
        if (db_ok) {
            uint32_t child = child_id(cursor, (uint32_t)i);
            if (child == 0 || child > nodes) {
                db_ok = 0;
            } else {
                value = node_value(child);
            }
        }
        reply_buffer[i * 2] = (int8_t)original_move;
        reply_buffer[i * 2 + 1] = (int8_t)value;
    }
    return count;
}

__attribute__((visibility("default")))
uint32_t get_replies_ptr(void) {
    return (uint32_t)(uintptr_t)reply_buffer;
}

__attribute__((visibility("default")))
uint32_t get_node_count(void) {
    return nodes;
}
