# RIES v11.1.4 changelog

- Changed `constantDbBudgetMs()` so level-4 and level-5 constant database searches use a 99-second budget.
- Kept the v11.1.2 1.2× budget behavior for levels outside 4/5.
- Tightened the async constant database inner probe slices and yield cadence so long level-4/5 searches continue to let the progress UI and SO(4) cube repaint.
- No search modules were removed or reordered in this patch.
