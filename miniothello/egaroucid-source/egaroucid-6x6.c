/*
 * MiniOthello: independent 6x6 Egaroucid-inspired search core.
 * C -> wasm32, zero libc, no network/NN weights, 36-bit bitboards.
 * Algorithmic ideas: NegaScout/PVS, null-window searches, depth-aware TT,
 * enhanced transposition cutoff, region parity, safe stable-edge bounds,
 * last 1..4 specialized search, iterative-deepening per-candidate values.
 * The supplied 8x8 Egaroucid evaluate data is not used (board-size-specific).
 * All reported completed full-depth terminal values obey winner-takes-empties.
 * No time/node cap and no probabilistic cutoff in this engine.
 */
#include <stdint.h>
#include <stddef.h>

#define FULL 0xfffffffffULL
#define COL0 0x41041041ULL
#define COL5 (COL0<<5)
#define NC0 (FULL^COL0)
#define NC5 (FULL^COL5)
#define INF 200000
#define TT_BITS 20
#define TT_SIZE (1u<<TT_BITS)
#define TT_MASK (TT_SIZE-1)

/* Imported hook lets a synchronous, uninterrupted WASM search post progress
 * from its Web Worker every ~500ms. JS never provides search deadlines. */
__attribute__((import_module("env"),import_name("notify")))
extern void notify(void);

typedef struct {
  uint64_t own,opp;
  int32_t value;       /* hundredths of one disc, regardless of exactness */
  int8_t depth,best,flag,solved; /* flag: 0 empty,1 exact,2 lower,3 upper */
} TTEntry;
/* 2^20 entries x 24 bytes = 24 MiB of wasm linear memory. */
_Static_assert(sizeof(TTEntry) == 24, "TT layout must be 24 bytes");
static TTEntry table[TT_SIZE];
static int8_t input[36], root_moves[36];
static int32_t result_scores[36], temporary_scores[36];
static uint8_t root_exact_flags[36];
static int root_count, completed_depth, searching_depth, empty_count, exact_done, engine_mode;
static uint64_t root_own,root_opp,node_count;
static uint64_t neighbors[36];

__attribute__((export_name("board_ptr"))) int board_ptr(void) {return (int)(uintptr_t)input;}
__attribute__((export_name("moves_ptr"))) int moves_ptr(void) {return (int)(uintptr_t)root_moves;}
__attribute__((export_name("scores_ptr"))) int scores_ptr(void) {return (int)(uintptr_t)result_scores;}
__attribute__((export_name("exact_ptr"))) int exact_ptr(void) {return (int)(uintptr_t)root_exact_flags;}
__attribute__((export_name("result_count"))) int result_count(void) {return root_count;}
__attribute__((export_name("result_depth"))) int result_depth(void) {return completed_depth;}
__attribute__((export_name("searching_depth"))) int next_depth(void) {return searching_depth;}
__attribute__((export_name("result_exact"))) int result_exact(void) {return exact_done;}
__attribute__((export_name("result_empties"))) int result_empties(void) {return empty_count;}
__attribute__((export_name("result_nodes"))) double result_nodes(void) {return (double)node_count;}

static inline int pop(uint64_t x) {return __builtin_popcountll(x);}
static inline uint64_t east(uint64_t x) {return (x<<1)&NC0;}
static inline uint64_t west(uint64_t x) {return (x>>1)&NC5;}
static inline uint64_t south(uint64_t x) {return (x<<6)&FULL;}
static inline uint64_t north(uint64_t x) {return x>>6;}
static inline uint64_t se(uint64_t x) {return (x<<7)&NC0;}
static inline uint64_t sw(uint64_t x) {return (x<<5)&NC5;}
static inline uint64_t ne(uint64_t x) {return (x>>5)&NC0;}
static inline uint64_t nw(uint64_t x) {return (x>>7)&NC5;}
#define PROP(F) do {uint64_t a=F(own)&opp; a|=F(a)&opp; a|=F(a)&opp; a|=F(a)&opp; a|=F(a)&opp; possible|=F(a)&empty;} while(0)
static inline uint64_t legal(uint64_t own,uint64_t opp) {
  uint64_t empty=FULL&~(own|opp),possible=0;
  PROP(east);PROP(west);PROP(south);PROP(north);
  PROP(se);PROP(sw);PROP(ne);PROP(nw);
  return possible&empty;
}
#undef PROP
#define CAPTURE(F) do {uint64_t a=F(bit)&opp;if(a){uint64_t p=F(a);while(p&opp){a|=p;p=F(p);}if(p&own)out|=a;}}while(0)
static inline uint64_t flips(uint64_t own,uint64_t opp,int m) {
  uint64_t bit=1ULL<<m,out=0;
  CAPTURE(east);CAPTURE(west);CAPTURE(south);CAPTURE(north);
  CAPTURE(se);CAPTURE(sw);CAPTURE(ne);CAPTURE(nw);
  return out;
}
#undef CAPTURE
static inline int final_score(uint64_t own,uint64_t opp,int empties) {
  int diff=pop(own)-pop(opp);
  return 100*(diff>0?diff+empties:diff<0?diff-empties:0);
}
static inline uint32_t idx(uint64_t a,uint64_t b) {
  uint64_t x=(a*0x9e3779b97f4a7c15ULL)^(b*0xbf58476d1ce4e5b9ULL);
  x^=x>>29;x*=0x94d049bb133111ebULL;x^=x>>31;
  return (uint32_t)x&TT_MASK;
}
static inline TTEntry *probe(uint64_t a,uint64_t b) {
  TTEntry *e=&table[idx(a,b)];
  return e->flag && e->own==a && e->opp==b?e:0;
}

static const int8_t weights[36]={
  120,-28,18,18,-28,120,
  -28,-54,-4,-4,-54,-28,
  18,-4,4,4,-4,18,
  18,-4,4,4,-4,18,
  -28,-54,-4,-4,-54,-28,
  120,-28,18,18,-28,120
};
static inline uint64_t adjacent8(uint64_t x) {
  return east(x)|west(x)|south(x)|north(x)|se(x)|sw(x)|ne(x)|nw(x);
}

/* Guaranteed stable only: corners + their contiguous same-color edge chains.
 * The test is deliberately conservative; never mistake an unstable edge for
 * stable when using it to prove exact endgame alpha/beta bounds. */
static uint64_t stable_edges(uint64_t own) {
  uint64_t mask=0;
  const int corners[4]={0,5,30,35};
  const int steps[4][2]={{1,6},{-1,6},{1,-6},{-1,-6}};
  for(int c=0;c<4;c++) {
    int p=corners[c];
    if(!(own&(1ULL<<p)))continue;
    mask|=1ULL<<p;
    for(int axis=0;axis<2;axis++) {
      int step=steps[c][axis];
      for(int k=1;k<6;k++) {
        int m=p+k*step;
        if(m<0 || m>=36 || !(own&(1ULL<<m)))break;
        mask|=1ULL<<m;
      }
    }
  }
  return mask;
}

static int estimate(uint64_t own,uint64_t opp,int empties) {
  int pieces=36-empties;
  int difference=pop(own)-pop(opp);
  int mobility=pop(legal(own,opp))-pop(legal(opp,own));
  int corners=0, positional=0;
  uint64_t a=own,b=opp;
  while(a){int sq=__builtin_ctzll(a);a&=a-1;positional+=weights[sq];}
  while(b){int sq=__builtin_ctzll(b);b&=b-1;positional-=weights[sq];}
  const uint64_t cms=(1ULL<<0)|(1ULL<<5)|(1ULL<<30)|(1ULL<<35);
  corners=pop(own&cms)-pop(opp&cms);
  int frontier=pop(own&adjacent8(FULL&~(own|opp)))-pop(opp&adjacent8(FULL&~(own|opp)));
  int stability=pop(stable_edges(own))-pop(stable_edges(opp));
  /* Evaluation is intentionally 6x6-specific, inexpensive and untrained.
   * Units converted to hundredths of a disc; sign is side-to-move-relative. */
  int raw=difference*(8+(pieces>23?pieces-23:0)*3)
    +mobility*(pieces<22?62:36)+corners*140+positional*2
    +stability*48-frontier*12;
  if(empties<12)raw+=((empties&1)?1:-1)*22;
  raw=raw*100/140;
  if(raw>3500)raw=3500;
  if(raw< -3500)raw=-3500;
  return raw;
}

/* Connected orthogonal empty regions; odd regions matter most in late game. */
static uint64_t odd_regions(uint64_t empties_mask) {
  uint64_t rest=empties_mask,odd=0;
  while(rest) {
    uint64_t component=rest&-rest,old=0;
    while(component!=old) {
      old=component;
      uint64_t adjacent=0,walk=component;
      while(walk) {int s=__builtin_ctzll(walk);walk&=walk-1;adjacent|=neighbors[s];}
      component|=adjacent&empties_mask;
    }
    if(pop(component)&1)odd|=component;
    rest&=~component;
  }
  return odd;
}

static inline void tick(void) {
  node_count++;
  if((node_count&8191ULL)==0)notify();
}

/* This fast 1..4-empties specialized path skips hash/mobility evaluation,
 * sorting and generic search-node state. Still full-width and exact. */
static int last_four(uint64_t own,uint64_t opp,int empties,int alpha,int beta) {
  tick();
  uint64_t moves=legal(own,opp);
  if(!moves) {
    if(!legal(opp,own))return final_score(own,opp,empties);
    return -last_four(opp,own,empties,-beta,-alpha);
  }
  int best=-INF;
  while(moves) {
    int m=__builtin_ctzll(moves);moves&=moves-1;
    uint64_t f=flips(own,opp,m),newOwn=own|f|(1ULL<<m);
    uint64_t newOpp=opp&~f;
    int value;
    if(empties==1)value=-final_score(newOpp,newOwn,0);
    else value=-last_four(newOpp,newOwn,empties-1,-beta,-alpha);
    if(value>best)best=value;
    if(value>alpha)alpha=value;
    if(alpha>=beta)break;
  }
  return best;
}

static int move_order(int m,int ttbest,int empties,uint64_t odd,
                      uint64_t new_own,uint64_t new_opp) {
  if(m==ttbest)return 100000;
  int p=weights[m]*10;
  if(empties<=14 && (odd&(1ULL<<m)))p+=290;
  /* Fastest-first order at exact endgame: fewest opponent moves first. */
  if(empties<=17)p-=pop(legal(new_opp,new_own))*45;
  return p;
}

/* Depth is number of remaining DISC PLACEMENTS, pass does not spend depth.
 * TT entries for approximate depths are never confused with exact deeper
 * results. Flags use the searched window, not a guessed exact value. */
static int search(uint64_t own,uint64_t opp,int empties,int depth,int alpha,int beta) {
  tick();
  uint64_t moves=legal(own,opp);
  if(!moves) {
    if(!legal(opp,own))return final_score(own,opp,empties);
    return -search(opp,own,empties,depth,-beta,-alpha);
  }
  if(depth<=0)return estimate(own,opp,empties);
  if(empties<=4 && depth>=empties) return last_four(own,opp,empties,alpha,beta);

  int exact_search=depth>=empties;
  if(exact_search && engine_mode) {
    int minScore=(2*pop(stable_edges(own))-36)*100;
    int maxScore=(36-2*pop(stable_edges(opp)))*100;
    if(minScore>=beta)return minScore; /* proven lower bound */
    if(maxScore<=alpha)return maxScore; /* proven upper bound */
  }

  TTEntry *entry=probe(own,opp);
  int ttbest=-1;
  if(entry) {
    ttbest=entry->best;
    if(entry->depth==depth || (entry->solved && exact_search)) {
      if(entry->flag==1)return entry->value;
      if(entry->flag==2 && entry->value>alpha)alpha=entry->value;
      if(entry->flag==3 && entry->value<beta)beta=entry->value;
      if(alpha>=beta)return entry->value;
    }
  }
  int alpha0=alpha,beta0=beta;
  uint64_t odd=engine_mode && empties<=14?odd_regions(FULL&~(own|opp)):0;
  struct Candidate {int m,priority;uint64_t newOwn,newOpp;} order[36];
  int count=0;
  while(moves) {
    int m=__builtin_ctzll(moves);moves&=moves-1;
    uint64_t flip=flips(own,opp,m);
    uint64_t newOwn=own|flip|(1ULL<<m),newOpp=opp&~flip;
    int pri=move_order(m,ttbest,empties,odd,newOwn,newOpp);
    int j=count++;
    while(j && order[j-1].priority<pri) {order[j]=order[j-1];j--;}
    order[j].m=m;order[j].priority=pri;
    order[j].newOwn=newOwn;order[j].newOpp=newOpp;
  }
  int best=-INF,bestmove=order[0].m;
  for(int j=0;j<count;j++) {
    uint64_t childOwn=order[j].newOpp,childOpp=order[j].newOwn;
    /* Enhanced Transposition Cutoff: only a verified child UPPER bound can
     * prove our LOWER bound. Never compare different heuristic depths. */
    if(engine_mode && j && depth>=4) {
      TTEntry *child=probe(childOwn,childOpp);
      if(child && (child->depth==depth-1 || (child->solved && exact_search))
         && (child->flag==1 || child->flag==3) && -child->value>=beta) {
        best=-child->value;bestmove=order[j].m;
        break;
      }
    }
    int value;
    if(j==0) value=-search(childOwn,childOpp,empties-1,depth-1,-beta,-alpha);
    else {
      value=-search(childOwn,childOpp,empties-1,depth-1,-alpha-1,-alpha);
      if(value>alpha && value<beta)
        value=-search(childOwn,childOpp,empties-1,depth-1,-beta,-alpha);
    }
    if(value>best){best=value;bestmove=order[j].m;}
    if(value>alpha)alpha=value;
    if(alpha>=beta)break;
  }
  TTEntry *put=&table[idx(own,opp)];
  int flag=best<=alpha0?3:best>=beta0?2:1;
  if(!put->flag || (put->own==own&&put->opp==opp) || depth>=put->depth || put->solved==0) {
    put->own=own;put->opp=opp;put->value=best;
    put->best=(int8_t)bestmove;put->depth=(int8_t)depth;
    put->flag=(int8_t)flag;put->solved=(int8_t)(exact_search && flag==1);
  }
  return best;
}

__attribute__((export_name("begin"))) int begin(int side,int mode) {
  uint64_t black=0,white=0;
  for(int i=0;i<36;i++) {
    if(input[i]==1)black|=1ULL<<i;
    else if(input[i]==-1)white|=1ULL<<i;
    int x=i%6,y=i/6;uint64_t n=0;
    if(x>0)n|=1ULL<<(i-1);
    if(x<5)n|=1ULL<<(i+1);
    if(y>0)n|=1ULL<<(i-6);
    if(y<5)n|=1ULL<<(i+6);
    neighbors[i]=n;
  }
  root_own=side==1?black:white;
  root_opp=side==1?white:black;
  empty_count=36-pop(black|white);
  root_count=0;completed_depth=0;searching_depth=0;exact_done=0;node_count=0;
  engine_mode=mode;
  uint64_t moves=legal(root_own,root_opp);
  if(!moves) {if(legal(root_opp,root_own))root_moves[root_count++]=-1;}
  else while(moves) {int m=__builtin_ctzll(moves);moves&=moves-1;root_moves[root_count++]=m;}
  for(int i=0;i<root_count;i++){result_scores[i]=0;root_exact_flags[i]=0;}
  return root_count;
}

/* Completes an ENTIRE iteration for ALL root candidates before publishing.
 * No partial-depth values can be mistaken for completed iterative results. */
__attribute__((export_name("search_depth"))) int search_depth(int depth) {
  if(depth<1 || exact_done)return completed_depth;
  if(depth>empty_count)depth=empty_count;
  searching_depth=depth;
  int indices[36];
  for(int i=0;i<root_count;i++)indices[i]=i;
  /* Previous iteration values order root candidates without changing which
   * candidates are searched or the independent, complete-root score window. */
  for(int i=1;i<root_count;i++) {
    int k=indices[i],j=i;
    while(j && result_scores[indices[j-1]]<result_scores[k]) {
      indices[j]=indices[j-1];j--;
    }
    indices[j]=k;
  }
  for(int k=0;k<root_count;k++) {
    int i=indices[k],move=root_moves[i];
    int value;
    if(move<0) value=-search(root_opp,root_own,empty_count,depth,-INF,INF);
    else {
      uint64_t flip=flips(root_own,root_opp,move);
      value=-search(root_opp&~flip,root_own|flip|(1ULL<<move),empty_count-1,depth-1,-INF,INF);
    }
    temporary_scores[i]=value;
    /* Publish heartbeats while a potentially long root candidate runs. */
    notify();
  }
  for(int i=0;i<root_count;i++) {
    result_scores[i]=temporary_scores[i];
    root_exact_flags[i]=(uint8_t)(depth>=empty_count);
  }
  completed_depth=depth;
  searching_depth=0;
  exact_done=depth>=empty_count;
  notify();
  return completed_depth;
}

/* Runtime support when linking freestanding wasm32. */
void *memset(void *p,int c,size_t n) {unsigned char *b=(unsigned char*)p;for(size_t i=0;i<n;i++)b[i]=(unsigned char)c;return p;}
void *memcpy(void *p,const void *v,size_t n) {unsigned char *a=(unsigned char*)p;const unsigned char *b=(const unsigned char*)v;for(size_t i=0;i<n;i++)a[i]=b[i];return p;}
