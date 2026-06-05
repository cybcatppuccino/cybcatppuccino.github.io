# RIES v11.4 changelog

- Added the uploaded 420000-row hard-constant database matcher as the third decimal search stage, after RIES/equation and sparse low-precision linear-combination search, before the L-function stage.
- Rebuilt the hard-constant database as one direct script asset: `assets/ries-harddb-v11_4-direct.js` (about 9.72MB). It is loaded by a normal `<script>` tag and does not use `fetch()`, gzip, or `DecompressionStream` at query time.
- The direct asset stores a Float64 value table plus compact dictionary-coded formula metadata. Search scans the numeric table directly and only decodes metadata for the final hits.
- The new module tests `abs(x/A)` against reduced rationals of height <= 10 and tests `log|x|/log|A|`, `log|x|/A`, and `x/log|A|` against {-2,-1,-1/2,1/2,1,2}.
- Returns at most five hard-database candidates and merges them into the existing global ranking.
- Result rows include the relation, the reconstructed formula family for A, database row id/category, and a 20-significant-digit display from the direct 64-bit numeric table.
- Speed update: the final low-precision algebraic relation pass now tests only the current typed input precision, instead of sweeping several adjusted precisions.
