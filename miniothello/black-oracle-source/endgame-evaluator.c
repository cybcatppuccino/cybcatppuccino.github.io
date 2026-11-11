/*
 * Mini Othello 6x6 FULL-DEPTH EXACT endgame WASM.
 *
 * From 18 occupied cells onwards, this engine solves each legal candidate
 * all the way to terminal (including forced passes).  No search-budget limit,
 * no heuristic leaf evaluation, no iterative depth limit and no partial values.
 *
 * Alpha-beta negamax with principal variation search and a transposition table.
 * TT keys compare both FULL 36-bit board bitmasks, so table collisions cannot
 * create false results.  Bound entries are handled by standard fail-soft rules.
 * Move ordering only affects speed, NEVER the returned score.
 *
 * Score values are final disc differences in the side-to-move perspective,
 * including empty squares awarded to the winner at the end of the game.
 * A completed result is always exact.  Intended for a Web Worker.
 */
#include <stdint.h>

#define FULL 0xfffffffffULL
#define COL0 0x41041041ULL
#define COL5 (COL0 << 5)
#define NOT_COL0 (FULL ^ COL0)
#define NOT_COL5 (FULL ^ COL5)
#define TTBITS 20
#define TTSIZE (1u<<TTBITS)
#define TTMASK (TTSIZE-1)
#define INF 100
#define UNKNOWN 127

/* 24 MiB TT in WASM linear memory, complete keys prevent hash collisions. */
typedef struct {
  uint64_t own, opp;
  int16_t score;
  int8_t best;
  uint8_t flag; /* 0 empty, 1 exact, 2 lower, 3 upper */
  uint8_t empties;
  uint8_t padding[3];
} Entry;
static Entry tt[TTSIZE];
static int8_t input[36];
static int8_t root_moves[36];
static int16_t root_scores[36];
static uint8_t root_exact[36];
static int32_t root_count,root_done,root_total_empty;
static uint64_t root_own,root_opp,node_count;

__attribute__((export_name("board_ptr"))) int board_ptr(void) { return (int)(uintptr_t)input; }
__attribute__((export_name("moves_ptr"))) int moves_ptr(void) { return (int)(uintptr_t)root_moves; }
__attribute__((export_name("scores_ptr"))) int scores_ptr(void) { return (int)(uintptr_t)root_scores; }
__attribute__((export_name("exact_ptr"))) int exact_ptr(void) { return (int)(uintptr_t)root_exact; }
__attribute__((export_name("result_count"))) int result_count(void) { return root_count; }
__attribute__((export_name("result_done"))) int result_done(void) { return root_done; }
__attribute__((export_name("result_depth"))) int result_depth(void) { return root_total_empty; }
__attribute__((export_name("result_exact"))) int result_exact(void) { return root_count == root_done; }
/* JS Number can represent this exactly for realistic searches (< 2^53 nodes). */
__attribute__((export_name("result_nodes"))) double result_nodes(void) { return (double)node_count; }

static inline int pop(uint64_t x) { return __builtin_popcountll(x); }
static inline uint64_t e(uint64_t x) { return (x<<1)&NOT_COL0; }
static inline uint64_t w(uint64_t x) { return (x>>1)&NOT_COL5; }
static inline uint64_t s(uint64_t x) { return (x<<6)&FULL; }
static inline uint64_t n(uint64_t x) { return x>>6; }
static inline uint64_t se(uint64_t x) { return (x<<7)&NOT_COL0; }
static inline uint64_t sw(uint64_t x) { return (x<<5)&NOT_COL5; }
static inline uint64_t ne(uint64_t x) { return (x>>5)&NOT_COL0; }
static inline uint64_t nw(uint64_t x) { return (x>>7)&NOT_COL5; }

#define GROW(F) do { uint64_t t=F(own)&opp; \
 t|=F(t)&opp; t|=F(t)&opp; t|=F(t)&opp; t|=F(t)&opp; \
 result|=F(t)&empty; } while(0)
static inline uint64_t legal(uint64_t own,uint64_t opp) {
  uint64_t empty=FULL&~(own|opp),result=0;
  GROW(e);GROW(w);GROW(s);GROW(n);GROW(se);GROW(sw);GROW(ne);GROW(nw);
  return result&empty;
}
#undef GROW

#define FLIP(F) do { uint64_t chain=F(bit)&opp; if(chain) { \
 uint64_t x=F(chain); while(x&opp) {chain|=x;x=F(x);} \
 if(x&own) out|=chain; } } while(0)
static inline uint64_t flips(uint64_t own,uint64_t opp,int pos) {
  uint64_t bit=1ULL<<pos,out=0;
  FLIP(e);FLIP(w);FLIP(s);FLIP(n);FLIP(se);FLIP(sw);FLIP(ne);FLIP(nw);
  return out;
}
#undef FLIP

static inline int final_score(uint64_t own,uint64_t opp,int empties) {
  int difference=pop(own)-pop(opp);
  return difference>0 ? difference+empties : difference<0 ? difference-empties : 0;
}

static inline uint32_t hash_index(uint64_t own,uint64_t opp) {
  uint64_t x=(own*0x9e3779b97f4a7c15ULL)^(opp*0xbf58476d1ce4e5b9ULL);
  x^=x>>29;x*=0x94d049bb133111ebULL;x^=x>>31;
  return (uint32_t)x&TTMASK;
}
static const int8_t weights[36]={
   100,-18, 16, 16,-18,100,
   -18,-36, -4, -4,-36,-18,
    16, -4,  2,  2, -4, 16,
    16, -4,  2,  2, -4, 16,
   -18,-36, -4, -4,-36,-18,
   100,-18, 16, 16,-18,100
};
static inline int move_priority(int move,int ttbest) {
  return move==ttbest?1000:weights[move];
}

static int solve(uint64_t own,uint64_t opp,int empties,int alpha,int beta) {
  node_count++;
  const int full_window = alpha <= -36 && beta >= 36;
  Entry *entry=&tt[hash_index(own,opp)];
  const int matched=entry->flag && entry->own==own && entry->opp==opp;
  int cached_move=-1;
  if(matched) {
    cached_move=entry->best;
    if(entry->flag==1) return entry->score;
    if(!full_window && entry->flag==2 && entry->score>alpha)alpha=entry->score;
    if(!full_window && entry->flag==3 && entry->score<beta)beta=entry->score;
    if(alpha>=beta)return entry->score;
  }

  const int search_alpha=alpha,search_beta=beta;
  uint64_t moves=legal(own,opp);
  if(!moves) {
    if(!legal(opp,own))return final_score(own,opp,empties);
    /* A pass does not fill a square, and can happen only once consecutively. */
    const int result=-solve(opp,own,empties,-beta,-alpha);
    if(!matched || entry->flag!=1) {
      /* No known best move in a pass position. */
      if(!entry->flag || !matched || entry->empties<=empties) {
        entry->own=own;entry->opp=opp;entry->empties=empties;
        entry->score=result;entry->best=-1;
        entry->flag=result<=search_alpha?3:result>=search_beta?2:1;
      }
    }
    return result;
  }

  int ordering[36],count=0;
  while(moves) {
    int m=__builtin_ctzll(moves);moves&=moves-1;
    int p=move_priority(m,cached_move),j=count++;
    while(j && move_priority(ordering[j-1],cached_move)<p) {
      ordering[j]=ordering[j-1];j--;
    }
    ordering[j]=m;
  }
  int best=-INF,bestmove=ordering[0];
  for(int i=0;i<count;i++) {
    int move=ordering[i];uint64_t flip=flips(own,opp,move);
    uint64_t new_own=own|flip|(1ULL<<move),new_opp=opp&~flip;
    int value;
    if(i==0) {
      value=-solve(new_opp,new_own,empties-1,-beta,-alpha);
    } else {
      /* PVS: cut quickly on inferior moves, then re-search any improving one. */
      value=-solve(new_opp,new_own,empties-1,-alpha-1,-alpha);
      if(value>alpha && value<beta)
        value=-solve(new_opp,new_own,empties-1,-beta,-alpha);
    }
    if(value>best){best=value;bestmove=move;}
    if(best>alpha)alpha=best;
    if(alpha>=beta)break;
  }
  /* A fully searched state is exact; cutoffs are only proven bounds. */
  if(!matched || entry->flag!=1 || (best>search_alpha && best<search_beta)) {
    if(matched || !entry->flag || entry->empties<=empties) {
      entry->own=own;entry->opp=opp;entry->empties=empties;
      entry->score=best;entry->best=bestmove;
      entry->flag=best<=search_alpha?3:best>=search_beta?2:1;
    }
  }
  return best;
}

/* Must be called once per position, after writing 36 input cell values. */
__attribute__((export_name("begin"))) int begin(int side) {
  uint64_t black=0,white=0;
  for(int i=0;i<36;i++) {
    if(input[i]==1)black|=1ULL<<i;
    else if(input[i]==-1)white|=1ULL<<i;
  }
  root_own=side==1?black:white;
  root_opp=side==1?white:black;
  root_total_empty=36-pop(black|white);
  root_count=0;root_done=0;node_count=0;
  uint64_t moves=legal(root_own,root_opp);
  if(!moves) {
    if(legal(root_opp,root_own))root_moves[root_count++]=-1;
  } else {
    while(moves){int m=__builtin_ctzll(moves);moves&=moves-1;root_moves[root_count++]=m;}
  }
  for(int i=0;i<root_count;i++) {root_exact[i]=0;root_scores[i]=UNKNOWN;}
  return root_count;
}

/* Solve exactly ONE candidate.  It returns only after ALL possible terminal
 * continuations affecting this score have been resolved through alpha-beta.
 * Root candidates have full [-100,+100] windows (no root cutoff).
 * Returns the index of the newly solved candidate or -1 once finished.
 */
__attribute__((export_name("solve_next"))) int solve_next(void) {
  if(root_done>=root_count)return -1;
  const int i=root_done,move=root_moves[i];
  int score;
  if(move<0) score=-solve(root_opp,root_own,root_total_empty,-INF,INF);
  else {
    uint64_t bit=1ULL<<move,flip=flips(root_own,root_opp,move);
    score=-solve(root_opp&~flip,root_own|bit|flip,root_total_empty-1,-INF,INF);
  }
  root_scores[i]=(int16_t)score;
  root_exact[i]=1;
  root_done++;
  return i;
}

/* Clang wasm32 -nostdlib may lower some operations to these built-ins. */
void *memset(void *dest,int c,unsigned long n) {
  unsigned char *p=(unsigned char*)dest;
  for(unsigned long i=0;i<n;i++)p[i]=(unsigned char)c;
  return dest;
}
void *memcpy(void *dest,const void *src,unsigned long n) {
  unsigned char *d=(unsigned char*)dest;
  const unsigned char *s=(const unsigned char*)src;
  for(unsigned long i=0;i<n;i++)d[i]=s[i];
  return dest;
}
