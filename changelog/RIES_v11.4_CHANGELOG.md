# RIES v11.4 changelog

- Added the uploaded 420000-row hard-constant database matcher as the third decimal search stage, after RIES/equation and sparse linear-combination search, before the L-function stage.
- Replaced the 378MB JSONL source with compact assets: `assets/ries-harddb-v11_4-qlog.bin` (1.68MB) and `assets/ries-harddb-v11_4-meta.tsv.gz` (1.16MB).
- The new module tests `abs(x/A)` against reduced rationals of height <= 10 and tests `log|x|/log|A|`, `log|x|/A`, and `x/log|A|` against {-2,-1,-1/2,1/2,1,2}.
- Returns at most five hard-database candidates and merges them into the existing global ranking.
- Result rows include the relation, the reconstructed formula family for A, database row id/category, and the compressed q-log value display.
