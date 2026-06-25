# v4 AI audit

## Requirements implemented

1. Continuous analysis can be paused and resumed without clearing the last stable result.
2. Stable results survive Analysis-off and page reload through structured local storage.
3. Re-entering a searched position displays the cached result immediately and resumes from the next depth.
4. Following a displayed PV seeds the resulting child position.
5. All three recommended root moves are legal-action buttons.
6. Local, Player-vs-AI, and AI-vs-AI modes can be switched during a game.
7. Play AI uses finite searches at levels 1–10 and is independent from continuous Analysis.

## Correctness review

### Stale Worker results

Mode changes and board changes increment request tokens. Main-thread clients reject any result carrying an old token. A finite search may finish internally after cancellation because Worker message delivery is cooperative, but its result cannot alter the board.

### Cached move safety

A cached UCI move is never applied directly. The UI resolves it through the current legal move generator. If it is no longer legal, no move is made.

### Search-context identity

- persistent analysis keys include board, side, halfmove clock, and recent history;
- repetition hashes intentionally omit clocks;
- TT keys include the halfmove clock to avoid fifty-move-horizon contamination;
- threefold remains checked before TT lookup.

### Depth preservation

The cache refuses to replace a deeper stable result with a shallower non-terminal update. Resume seeds are validated by replaying legal UCI moves. Invalid tails are truncated.

### Endgame safety

Low-material nodes retain the v3 conservative policy: null move and aggressive forward pruning are disabled in likely zugzwang endings; advanced pawns and promotions receive reduced LMR and bounded extensions. No complete tablebase is claimed.

## Efficiency changes

- typed-array evaluation cache reduces repeated full evaluation;
- capture history improves tactical ordering without allocating objects;
- persistent TT and root/PV ordering reduce repeated work;
- child-PV cache avoids restarting chosen engine lines at depth one;
- finite play and continuous analysis use separate Workers;
- position caches use bounded LRU-like eviction rather than unbounded growth.

A same-runtime initial-position benchmark measured a warm-run median near 134k NPS for v4 versus 112k for v3. This is environment-specific and should not be interpreted as Elo. Cached branch continuation is the more important practical improvement.

## PGN regression

`data/book-benchmark.json` was regenerated with Orion JS 4.0 at fixed depth 8 without book injection. The sample is retained as a warning system, not as an optimization target that overrides general chess logic. A recorded oracle continuation can differ from the engine's shallow preferred move; deeper searches may narrow or reverse the difference.

## External MCTS/PPO archive

The supplied archive was inspected but not integrated. Direct use would introduce incompatible rules, Python/Ray/PyTorch dependencies, opaque model assumptions, and unresolved licensing. No measurable engine improvement justified those costs. See `MCTS-REVIEW.md`.
