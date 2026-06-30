## v12.0.2

- Split hypergeometric and integral/sum assets into search-index and display-metadata packages, so search loads only the required numeric/index data and lazy-loads formula metadata only after hits.
- Prestarted HardDB/hypdata/intsum index loads in parallel, skipped unnecessary hypdata real-index decoding for complex targets, switched heavy progress loops to time-sliced yielding, and cached L-function Decimal values.
- Cleaned integral/sum candidate text and LaTeX for redundant generated zero/unit factors and powers.

# RIES changelog

## v12.0.1

- Restored the active hypergeometric pFq database to cumulative level loading: level4 loads `2F1/3F2`, level5 adds `4F3/5F4`, and level6 adds all remaining pFq families.
- Renamed active lazy database assets and chunk globals to versionless names.
- Removed active cache-buster query strings from runtime script/asset loading.
- Corrected lazy loading progress display so the shown maximum equals the real byte upper bound, and so progress maps exactly to each loading slice.
- Corrected L-function scan progress so it remains inside its configured progress range.
- Kept the v12 consolidated tests under versionless filenames and added coverage for pFq layer boundaries and progress helper behavior.
