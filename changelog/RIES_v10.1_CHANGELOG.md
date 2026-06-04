# RIES v10.1

- Replaced the v10 `assets/shortform100k.js` direct object with a packed, lazy lookup table generated from `shortforms_top3.jsonl` for integer targets `0 <= n <= 100000`.
- Preserves the v10 integer search pipeline while allowing the `<=100000` precomputed shortform phase to use the new table first.
- Stores up to three non-literal shortforms per integer, after conservative neutral-operation simplification such as `0 + x -> x`, `x + 0 -> x`, `x * 1 -> x`, `x / 1 -> x`, `x^1 -> x`, and `0!`/`1! -> 1`.
- Uses a math-template/dictionary encoding in the JavaScript source rather than a compressed archive; the generated database file is under 2 MB.
- Keeps backward compatibility with the old `window.RIES_SHORTFORM_100K` / `window.RIES_SHORTFORM_100K_MULTI` variables if they are present.
