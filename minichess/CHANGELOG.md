# Changelog

## v19.7

- Updated release labels, saved-game/cache namespaces, browser cache-buster and engine identity to v19.7 / `Orion JS 19.7`.
- Added a bounded **exact-tablebase bridge proof** pass for six-piece analysis roots whose completed principal variation reaches a supplied <=5-piece exact WDL+DTM tablebase node.
- Bridge proof uses an AND/OR certificate: the winning side selects a bounded legal policy candidate; the resisting side is enumerated exhaustively; every proven leaf is either immediate terminal checkmate/stalemate or a resident exact WDL+DTM tablebase node. A missing, WDL-only, losing, or drawing leaf cannot become a mate certificate.
- A successful winning certificate publishes `≤#N` / `≤-#N` as a finite DTM **upper bound**, never as an exact mate distance. The ordinary iterative engine remains active and may later replace it with a shorter bound or an independently verified exact mate.
- Added dual-controller bridge-draw proof. A completed PV that reaches an exact WDL=0 tablebase node is now a bridge trigger (not a result by itself); Analysis may publish exact `0.00` only when both White and Black can force an exact tablebase draw against every opposing reply. A one-sided route to a draw is not enough.
- The bridge prover treats <=5-piece territory as a strict exact-tablebase boundary and uses resident synchronous probes only. This prevents an unproven sub-tablebase branch from expanding into speculative search.
- Bridge proof intentionally ignores nonresident tablebase values for move ordering after pre-warming the required exact family, preventing block-loading state from changing proof validity or the bounded candidate set.
- Preserved v19.5/v19.6 result-ownership and display stability: completed score/PV snapshots remain atomic; the UI keeps its fixed 500 ms paint cadence; ordinary raw WDL sentinels cannot overwrite a certified bridge result.
