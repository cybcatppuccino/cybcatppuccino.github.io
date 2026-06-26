# v7 implementation audit

> Historical note: the exhaustive tablebase resource plan below is superseded by the practical v7.1 design in `V7.1-TABLEBASE-AUDIT.md`.

## Requested changes

### Verified mate persistence

The v5/v6 engine validated mates before display, but a solved result could still be replaced by a later ordinary result or restarted after reopening the same position. v7 treats replay-verified mate lines as durable solved artifacts.

Changes:

- persistent cache schema and storage namespace advanced to v7;
- `solved` derives only from the first line's `mateVerified` flag;
- non-solved updates cannot overwrite a saved solved result;
- cached mates are replayed from the exact current root before restoration;
- Worker-side result selection prioritizes a solved entry over merely deeper entries;
- solved resume emits the cached result and enters `complete` without beginning iterative deepening;
- following a PV creates child cache entries with rebased mate score, DTM and remaining PV, followed by child-root replay validation.

This removes repeated proof work without trusting stale or malformed mate data.

### Lazy PGN archive and collapsed interface

- Gardner rules no longer use the HTML `open` attribute.
- Game Tree no longer uses the HTML `open` attribute.
- No PGN/library load is issued during application startup.
- Enabling Book calls `ensureLibraryLoaded()` before displaying book arrows.
- Opening Game Tree calls the same idempotent loader before matching/rendering archive nodes.
- Failed loads reset partial state and may be retried.
- Closed desktop Game Tree becomes a narrow side rail; tablet uses a compact row.

The current game/AI tree can render before the archive is loaded; book matches remain empty until demand loading finishes.

## Offline tablebase generator review

The new tool is intentionally separate from the web runtime. It implements a Gardner-specific, Syzygy-inspired WDL+DTM pipeline rather than pretending orthodox 8×8 Syzygy files can be reused.

Core design:

- exact 25-square combinatorial indexing per canonical material signature;
- both side-to-move states represented;
- Numba-compiled legality, move generation, make-move, rank/unrank and predecessor kernels;
- dependency closure for captures and promotions;
- disk-backed WDL, degree, DTM and maximum-child-DTM arrays;
- distance-bucket retrograde propagation;
- atomic JSON checkpoints and one write-ahead transaction;
- committed bucket lengths recorded so restart can truncate uncommitted tails;
- completed tables written as independently gzip-compressed WDL/DTM blocks with SHA-256 hashes;
- lazy Python probing with an LRU block cache.

Semantics:

- WDL is from the side-to-move perspective;
- DTM is exact plies to checkmate under shortest-win/longest-loss play;
- stalemate, insufficient progress through cycles, and resolved non-wins become draws;
- the generated metric ignores the 50-move rule and advertises that fact in the manifest.

## Resource estimates

The built-in exact index estimator reports:

| Scope | Tables | Raw slots | Packed WDL+DTM before gzip |
|---|---:|---:|---:|
| ≤3 pieces | 6 | 139,200 | 0.30 MiB |
| ≤4 pieces | 36 | 16,837,200 | 36.13 MiB |
| ≤5 pieces | 146 | 1,079,437,200 | 2.26 GiB |
| ≤6 pieces | 511 | 55,643,947,200 | 116.60 GiB |

With default per-table checkpoint cleanup, the largest six-piece class needs an estimated 2.38 GiB of checkpoint disk at one time. The complete compressed output is data-dependent; the documentation advises reserving 130–150 GiB total. The full set is expected to take weeks on a typical desktop, so selected material families are the practical first target.

## Validation performed

JavaScript:

- all source files passed Node syntax checking;
- all existing core/search/cache/Worker/play/layout regression suites passed;
- new solved-cache persistence test passed;
- new PV mate rebase and child replay test passed;
- solved Worker resume emitted one cached result and did not search again;
- static lazy-load/default-collapse guards passed.

Python:

- smoke generation/probe test passed;
- interruption/recovery test passed by replaying a durable WAL and resuming to a valid table;
- compressed block checksum verification passed on generated fixtures;
- a manual KQvK generation/probe completed;
- a 607,200-slot KQRvK-family benchmark completed, including JIT warm-up, dependencies and compression.

## Known boundary

The generator has not been run here for every one of the 511 through-six-piece material classes. Correctness is covered by the algorithmic tests and small/medium fixtures, while the user's local full build will be the first complete production-scale run. v8 should integrate only verified output whose manifest and block checksums pass.
