# RIES v9.2

- Reset all per-input search state when the target input changes, including integer caches, solve caches, pending Continue state, and L-function progress caches.
- Added stale-input protection so an older async integer/decimal run cannot write results after the user has typed a new target.
- Made Stop use a single active-run flag so integer Continue and decimal phases can be interrupted at the next UI yield while keeping current rows.
- Improved log|c| matching so sparse low-height products stay ahead of noisy LLL relations and first Continue prioritizes Gamma/log examples.
