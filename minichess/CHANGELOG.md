# Changelog

## v17.3

- Updated all app/version labels and cache keys to v17.3 / `Orion JS 17.3`.
- Fixed the `previous.result` crash in the persistent analysis cache when the first streamed result for a position was PV-incomplete.
- Hardened analysis/play cache selection: score depth, PV depth, and PV completeness are now compared before replacing a cached artifact; incomplete live streams are displayed but not persisted as resume artifacts.
- Added a regression test for `nrbkq/ppppp/5/PPPPP/QKRBN w - - 0 1` to ensure the analysis worker no longer crashes on the reported position.
- Optimized exact tablebase loading on the web:
  - `analyze()` now starts with WDL-only probing and loads DTM blocks only for the relevant candidate pool.
  - full exact blocks reuse already-loaded WDL blocks instead of downloading/decompressing WDL twice.
  - metadata, WDL block, and full block requests are de-duplicated while in flight.
  - WDL neighborhood warming is now fire-and-forget, so it no longer delays the first analysis/search chunk.
- Cleaned up duplicated `probeWdl()` tablebase code paths and kept <=5-piece WDL probing available to the searcher whenever the needed WDL block has already been loaded.
- Included the full <=5-piece manifest file in the patch again so deployments missing the database directory still get the 111-table manifest metadata.
