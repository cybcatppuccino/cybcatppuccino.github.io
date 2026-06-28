# Gardner MiniChess Lab v19.3

This is the v19.3 differential source package. Overlay these files on the v19.2 project root; only modified or new files are included in the release archive.

## What changed in v19.3

- Updated visible release labels, browser cache-buster, game-state namespace, and Orion engine/cache namespace to v19.3 / `Orion JS 19.3`.
- Added a bounded automatic exact-tablebase handoff for ≤5-piece roots. If the first full GTB request races metadata/block warming, the same worker warms the root/children and performs bounded exact re-probes; an Exact TB result promotes itself without requiring a Stop/Start local-engine cycle.
- Direct Exact TB now outranks local verified-mate and ordinary live evaluation results. Its optimal database PV and DTM display stay authoritative once received.
- While that handoff is pending, the UI suppresses the engine-internal synchronous-WDL sentinel (`+220.00` / `-220.00`) instead of flashing between it and database mate output. If bounded retries genuinely cannot read exact data, the ordinary engine result remains available as a fallback.

## Verification

Run `node tests/v19_3-exact-tablebase-handoff-tests.mjs` after applying the patch. The test does not require the large GTB binary corpus.
