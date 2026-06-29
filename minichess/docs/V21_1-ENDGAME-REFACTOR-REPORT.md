# v21.1 Endgame Refactor Report

## Scope

v21.1 is a targeted endgame-result refactor over v21.  It changes only the files needed to make the public analysis result contract stricter, prevent fake proven-mate labels, and make depth/PV publication more conservative.

## Main contract changes

1. `js/engine/result-contract.js` is the new single source of truth for UI-visible result semantics.
2. Ordinary search may publish only numeric evaluations.
3. PV-only mate candidates are downgraded to finite evaluations and marked as candidates, not proof.
4. Tablebase bridge wins are published as proven upper bounds (`db_bridge_mate_bound`), not exact mate distances.
5. Exact `mate in N · proven` requires proof provenance (`mateProof` or `endgameProof`) and cannot be inferred from a PV replay alone.
6. Depth publication is stricter: a non-solved line at depth `N` must carry a PV of at least `N` plies to be considered complete.

## Function audit

### `result-contract.js`

- `contractKindForLine()`: classifies each public line into exact DB, exact mate proof, mate bound, bridge draw, ordinary search, mate candidate, internal proof seed, or live progress.
- `normalizeLineContract()`: strips unsupported mate/proof flags and converts unproven mate-looking scores to finite evaluations.
- `normalizeResultContract()`: applies the contract to every line before ranking/caching/display.
- `isPublishableLine()`: prevents internal WDL/proof seeds from replacing visible evaluations.

### `result-quality.js`

- `pvTargetForDepth()` now requires PV length to grow with displayed depth.
- `lineHasCompletePv()` no longer treats shallow ordinary lines as automatically complete.
- `classifyResultShallow()` recognizes bridge mate bounds through the explicit contract, not through `mateVerified`.
- `withResultQuality()` normalizes the result contract before and after legacy quality ranking.

### `worker.js`

- Bridge mate publication now sets `mateVerified: false`, `mateProof: false`, `mateUpperBound: true`, and `resultContract: db_bridge_mate_bound`.
- Bridge draw publication receives `resultContract: db_bridge_draw`.
- Visible-line stability checks now use `isPublishableLine()`.

### `minifish.js`

- Minifish is bumped to `Minifish JS 21.1`.
- `pvEndsInMate()` remains a local PV sanity check, but its result is no longer a proof flag.
- PV-only mate lines become `mate_candidate` finite evaluations.
- Iterative deepening no longer stops because a single PV reaches mate.

### `engine.js`

- Orion is bumped to `Orion JS 21.1`.
- Root PV completion target is expanded and tied to displayed depth.
- Public lines carry `resultContract`/`resultKindV2` so UI/cache do not infer proof status from ambiguous legacy fields.

### `analysis-panel.js`

- The UI no longer renders `mateVerified` alone as `proven`.
- Proven mate labels are rendered only through `canDisplayMateIn()`.
- Tablebase bridge wins are shown as mate upper bounds.
- Ordinary/mate-candidate lines are shown as evaluations.

### `analysis-cache.js`

- Cache namespace/schema were bumped to v21.1/33.
- Result contract fields are persisted with verified cached results.

### `play-worker.js`

- Playing AI worker uses the same publishability gate for visible search results.

## Validation run

The following checks were run after patching:

- `node --check` on all modified JavaScript files.
- `tests/core-tests.mjs`
- `tests/engine-tests.mjs`
- `tests/engine-regression-tests.mjs`
- `tests/analysis-cache-tests.mjs`
- `tests/v20_2-score-integrity-tests.mjs`
- `tests/v20_3-mate-scheduler-tests.mjs`
- `tests/v20_4-cache-and-endgame-tests.mjs`
- `tests/v20_5-side-and-mate-integrity-tests.mjs`
- `tests/v21_1-result-contract-tests.mjs`

## Remaining limitation

v21.1 fixes the result semantics and the UI proof gate.  It does not magically make every winning position produce an immediate exact shortest mate.  If the AND/OR proof layer has not completed, the line is intentionally displayed as a finite evaluation or as a proven upper bound only when a certificate-backed bridge proof exists.
