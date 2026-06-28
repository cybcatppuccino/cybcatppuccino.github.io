# Changelog

## v19.4

- Updated visible release labels, saved-game namespace, and engine identity to v19.4 / `Orion JS 19.4`.
- Removed the persistent `AnalysisCache` module, worker position-result caches, PV-child cache seeding, and tablebase full-analysis result cache. New analysis and AI roots clear prior TT, static-evaluation, heuristic, structural-profile, and non-mate proof state before searching.
- Added a deliberately narrow in-memory `MateProofStore`: only replay-validated local mates and explicit database mates for the exact root are reusable. Ordinary evaluations, partial PVs, draws, and non-mate tablebase results always start a new full analysis.
- Strengthened conservative forcing-sequence handling by ordering quiet checks ahead of ordinary killers/countermoves and adding a bounded one-legal-reply extension only in compact checking endgames.
- Reorganized the static-evaluation table into two-way set-associative buckets while preserving the prior 524,288-entry memory budget and exact-score lookup semantics.
- Added an exact-root recommendation layer so the legal current AI best-move arrow survives delayed UI paints and transient short-PV updates.
- Made `Standard · Syzygy` the default board/piece presentation, added independently selectable board and piece themes, and converted Analysis / Book / Edit / New controls to colored icon-only buttons.

## v19.3

- Updated visible release labels, browser cache-buster, game-state namespace, and Orion engine/cache namespace to v19.3 / `Orion JS 19.3`.
- Added an in-session exact-tablebase promotion retry: when an initial five-piece GTB request races block availability, root/child WDL warming is followed by bounded exact re-probes in the same worker session; no manual Start/restart is needed to surface the eventual Exact TB result.
- Made direct Exact TB outrank local verified-mate and live centipawn results for result selection, caching, and display.
- Deferred internal synchronous WDL sentinel scores (`+220.00` / `-220.00`) while a root Exact TB promotion is pending, then retained database PV/DTM information against later streamed snapshots so the panel cannot flash between database mate and centipawn sentinel output.

## v19.2

- Updated visible release labels, browser cache-buster, game-state namespace, and Orion engine/cache namespace to v19.2 / `Orion JS 19.2`.
- Fixed the actual MultiPV output path: narrow-window secondary/tertiary candidates now rebuild their legal continuations from the repetition-aware TT before live or completed results are published.
- Rebuilt TT path context while extending a PV, so the continuation lookup uses the correct repetition history instead of stale salts from another root branch.
- Prevented a short live PVS line from overwriting a longer already-calculated prefix for the same root move; only matching prefixes may retain an older tail.
- Applied the same matching-prefix safeguard in streamed Worker-result merging, preventing a later chunk from reintroducing a stale or short secondary/tertiary PV.
- Finished-depth validation now checks the top three visible candidates, not only the first PV, before calling a MultiPV result complete.

## v19.1

- Updated visible release labels, browser cache-buster, game-state namespace, and Orion engine/cache namespace to v19.1 / `Orion JS 19.1`.
- Changed the analysis PV rows from one-line ellipsis previews to wrapped, complete SAN text. All three visible candidate moves now show every principal-variation ply already calculated by the engine.
- On compact mobile layouts, retained the top three candidates instead of hiding the third row; the existing analysis list scroll remains available for long PVs.

## v19

- Updated visible release labels, browser cache-buster, game-state namespace, and Orion engine/cache namespace to v19 / `Orion JS 19`.
- Removed estimated win-rate labels from live principal variations. Candidate evaluation scores remain synchronized only by the existing fixed 500 ms UI paint cadence.
- Reworked the Moves panel into conventional paired move-number notation (`1. White move Black move`), with vertical mobile scrolling and compact Copy line / Copy tree controls.
- Added PGN movetext export for the current navigated line and complete recursive-annotation-variation tree export, including a Gardner/FEN setup header for exact board reconstruction.
- Added an editor Move piece tool, selected by default, which marks a source piece on the first click and moves it (overwriting the destination) on the second click; Erase square remains alongside it.
- Moved New game beside Analysis, Book, and Edit. A one-step in-memory New game snapshot lets Undo restore the complete previous local game tree and current position.

## v18.4

- Updated visible release labels, storage namespace, and browser script cache-buster to v18.4 / `Orion JS 18.4`.
- The existing 500 ms analysis paint cadence now shows the newest cumulative node total and an adaptive estimate to the next requested depth (`current/target`), without persisting transient progress snapshots.
- Live principal-variation rows now show the current estimated winning chance for the side to move; the top three may intentionally fluctuate while a depth is still being searched.
- Hardened direct GTB move selection: a root tablebase result is no longer paired with an arbitrary child after an incomplete child probe. At least one verified child must preserve the root WDL.
- Enabled synchronous loaded-WDL probes in checked ≤5-piece search and quiescence nodes, while retaining the excluded-move safeguard used by singular search.
- Retained the low-progress hard-draw policy, but grants confirmed low-progress roots a conservative 25% audit budget and selectively protects sacrificial attacks, quiet king entries, and compact zugzwang candidates from ordinary reduction.
- Made transposition-table path history salts order-independent while preserving repetition-sensitive state, recovering safe transposition reuse for equivalent reversible-history multisets.

## v18.3

- Kept intentional boot-time AI cache clearing, documented the behavior, and retained separate current-game restoration.
- Removed the 50-move automatic draw to align game rules with the fixed GTB tablebase convention.
- Unified AI play with shared result-quality logic and trust direct GTB WDL results, including draws and DTM-bound displays, as terminal.
- Added full reversible repetition context to analysis-cache keys and incremental repetition/path salts to transposition-table locks.
- Idle-scheduled dirty localStorage persistence; added lazy worker creation/disposal by mode; bounded tablebase WDL memory and queued direct/background tablebase requests.
- Removed runtime practical-seed/practical-tablebase paths and use the stable complete 111-table manifest.

## v18.2

- Updated app/engine/cache/tablebase labels to v18.2 / `Orion JS 18.2`.
- Changed streamed analysis rendering to a fixed trailing 500 ms cadence; tablebase, mate, and solved updates also wait for the next scheduled paint.
- Stabilized mate/tablebase-bound display so later live centipawn scores do not replace a stronger bound for the same root move.
- Skipped redundant async DTM annotation when all visible lines already have mate/tablebase-bound/exact tablebase information.
- Retired v18.1/v18 persistent analysis cache buckets for a clean v18.2 stability baseline, while allowing current-game state restore from v18.1.
- Applied shared result-quality selection on the UI stream path so stronger cached/bound displays are not overwritten by weaker live updates before painting.
