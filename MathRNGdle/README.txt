Math RNGdle Static v11
======================

Open index.html directly in a modern browser. No server, Python environment,
or network connection is required for normal play.

v11 fairness / dominance pass
-----------------------------
- Reworked scoring around mathematical evidence rather than raw badge count.
  When a stronger property strictly implies a weaker one, or two badges encode
  the same certificate, the weaker/duplicate contribution is suppressed.
- Added systematic dominance handling for palindrome / mirror / repeated-block
  structures, same-base digit-pattern implications, pseudoprime hierarchies,
  perfect-power hierarchies, figurate inclusions, duplicated reversal formulas,
  repeated combinatorial certificates, and other proven implication chains.
- Removed several vacuous boundary cases where a property was automatically true
  with too few independent objects (for example a single mirrored pair, a
  two-point arithmetic progression, or degenerate very-small moduli).
- Palindromes no longer gain a second large reward from
  n * reverse(n) = n^2; repdigits and repeated blocks similarly do not harvest
  multiple scores from automatic consequences of the same visible structure.
- Cross-base resonance remains supported, but base-level evidence that is merely
  an automatic consequence of a stronger representation is prevented from being
  counted again where the v11 dominance rules apply.
- The dominance system intentionally keeps properties that are merely correlated
  but are not mathematically equivalent or strictly implied by one another.

Witness / display improvements
------------------------------
- Corrected the decimal half-complement witness so the displayed pairings match
  the actual left-half / right-half condition.
- Existing colored witness markup is preserved, with color used to distinguish
  mathematical objects and operands rather than simply tinting the whole card.
- OEIS cards show a richer local sequence context: nearby members around the
  current number (up to 11 in the local game-domain view), the current member's
  position in that local sequence data, plus the available Formula, Comment,
  Example and Keywords metadata.

Scoring / rarity
----------------
- Scores and rarity tables in data/config.js are the v11 snapshot from the final
  sandbox work state. The scoring distribution naturally shifts downward at the
  top when duplicated mathematical evidence is removed.
- Jackpot and Ultra Jackpot continue to use the v11 percentile/rank data stored
  in data/config.js.
- Every random roll still uses Web Crypto with rejection sampling, so each
  integer 0..999,999 is sampled uniformly without modulo bias.

Data layout
-----------
- 50 lazy-loaded number chunks
- OEIS enrichment shards under data/oeis/
- Main configuration: data/config.js
- Badge catalogue: data/catalog.js
- Result scoring / dominance logic: app.js
- Witness rendering: witness.js
