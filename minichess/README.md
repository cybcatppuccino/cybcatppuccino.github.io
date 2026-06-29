# Gardner MiniChess v19.8 patch notes

This directory is a **v19.7 → v19.8 source patch**. It does not include the Gardner tablebase files; keep the tablebases already installed for v19.7.

## What changed

v19.7 correctly refused to call a speculative tablebase tail a root mate, but could leave a non-final bridge path as a confusing `TB bridge · verifying` state. v19.8 replaces that state with a bounded numerical audit:

1. Ordinary alpha-beta keeps exact tablebase WDL values internally for pruning.
2. Once a completed stable PV exposes a non-draw tablebase signal, the Worker preloads the reachable exact blocks and starts a separate bounded audit for that root move.
3. In the audit, exact WDL leaves score as finite `+10.00` / `-10.00`; non-tablebase replies keep their ordinary alpha-beta evaluation.
4. The displayed score is therefore the strongest defence found by the audit, rather than a raw `±220.00` sentinel or a textual placeholder.
5. The independent AND/OR bridge prover continues in parallel. Only its full all-replies certificate may promote the result to `mate ≤ N`; only an exact root tablebase or verified dual-controller bridge draw may publish `0.00` as a solved draw.

## Interpretation

A finite mixed-WDL score is intentionally **not** a mathematical mate claim or a permanent cache entry. It says that, at the bounded audit depth and under its exact tablebase leaves, the selected root move retains at least the shown finite evaluation against the searched defensive replies. It is a more useful and stable ordinary evaluation while the engine continues looking for a full bridge proof.

For example, a branch family that all resolves to exact white-winning tablebase leaves is displayed as `+10.00`, not `+220.00`, and not as a made-up mate distance. If a non-tablebase black defence is found at `+5.30`, that finite value becomes the relevant display score for the audited root move.

## Files

Use `PATCH_FILE_LIST.txt` to overlay only the changed/new files on a v19.7 installation.

## Tests

Run from the project root after tablebase files are available:

```bash
node tests/v19_8-tablebase-mixed-bound-tests.mjs
node tests/v19_8-worker-tablebase-mixed-bound-tests.mjs
```
