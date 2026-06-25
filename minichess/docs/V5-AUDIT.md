# v5 AI correctness and endgame audit

## 1. Reported false mate

### Position

After:

```text
1. b4 cxb4 2. Rxb4
```

the Gardnerfish-padded FEN is:

```text
8/8/1rnbqk2/1p1ppp2/1R6/2PPPP2/2NBQK2/8 b - - 0 2
```

The reported `2...Bc5 #1` cannot be correct. `Bc5` is not check because White's pawn on e3 blocks the diagonal `c5-d4-e3-f2`; White also has many legal replies.

### Root cause

At a non-PV node, v4 could prune every generated move before performing a real child search. `bestScore` therefore retained its internal sentinel `-INF`. Negamax negated the sentinel on return, creating `+INF`, which entered the mate-score band and was rendered as `#1`.

This was a search-control bug, not a FEN, SAN, bishop-ray, or legal-move-generation error.

### Corrections

1. Late-move/futility and SEE pruning now require `legalIndex > 0`; the first legal move is always searched.
2. An excluded singular-search node with no candidate returns a failed bound, not mate/stalemate.
3. Root mate scores are replayed through the legal move generator and rejected unless the encoded terminal checkmate is reached.
4. Cached mate PVs undergo the same validation before reuse.
5. The persistent cache namespace moved to v5, so old false mate records cannot reappear.
6. Mate cache scores are not used as aspiration anchors.
7. Deadline unwinding restores a complete root snapshot before line validation.

### Regression coverage

`tests/v5-engine-tests.mjs` verifies:

- `Bc5` is legal;
- `Bc5` is not check;
- White has at least eight legal replies;
- a search produces no mate label in any MultiPV line;
- the engine emits no rejected INF/mate sentinel in the corrected path.

## 2. Repetition audit

The former per-node scan of all historical root hashes was replaced with a counted map. The current search stack still tracks exact per-ply hashes.

The root keeps formal threefold semantics. Below the root, a second occurrence is scored as a stable cycle draw so the engine does not repeatedly expand reversible loops. Repetition is resolved before a TT cutoff, preventing a context-insensitive transposition value from overriding a line-specific cycle.

## 3. Low-material analysis

A bounded exact DTM proof layer was added for positions containing at most six pieces. It uses iterative distance limits and an AND/OR definition of forced mate. Defender branches must all remain mating; one escape refutes the current proof. Proven lines are legally replayed in the test suite.

This design improves short sparse mating positions without shipping a large precomputed database. It is intentionally conservative:

- an expired budget yields no proof;
- a cycle is an escape;
- fifty-move and insufficient-material outcomes are non-mates;
- the ordinary Alpha-Beta result remains available when no proof is found.

## 4. Large-score handling

- Dynamic aspiration widths reduce repeated research when evaluations are already far from equality.
- The existing mate-distance convention prefers quick wins and longest defence.
- Check-evasion extension is bounded.
- Sparse and promotion-sensitive positions retain reduced pruning.
- Only verified mate values can terminate continuous analysis as solved.

## 5. Cache and result format

- Engine identifier: `Orion JS 5.0`.
- Persistent storage key: `gardner-analysis-cache-v5`.
- v3/v4 results are ignored.
- Mate verification and DTM provenance are serialized.
- Heuristic W/D/L fields were removed from the engine result and UI.

## 6. Validation performed

The v5 suite includes all prior core, Worker, cache, play-mode, and AI-vs-AI tests plus the reported-line regression and exact DTM replay. The original PGN/PDF archive is checked against `data/SOURCE_INTEGRITY.sha256` before release.

## 7. Known limits

The new proof cache is not a complete Gardner endgame tablebase. It cannot certify all six-piece positions, and proof time grows rapidly with branching factor. A future C++/WASM engine could add a retrograde WDL/DTM database with canonical position indexing, symmetry reduction, legality filtering, and compressed lookup. Until then the browser build labels only results it actually proves within its bounded budget.
