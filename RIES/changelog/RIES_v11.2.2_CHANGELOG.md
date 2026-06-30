# RIES v11.2.2 changelog

Focused constant-database update on top of v11.2.1.

## Constant database only

- Added a constant-database coefficient-bound policy for LLL/PSLQ relation probes: target linear-combination coefficients are bounded by `|a_i| <= 100`.
- Added bounded PSLQ probes before constant-database LLL relation checks.
- Reworked the constant-database floating LLL reducer so it keeps exact integer row operations but does Gram-Schmidt on cached Float64 rows, avoiding repeated BigInt-to-Number dot products.
- For wall-clocked constant-database scans, degree-3 ratio probes now use bounded PSLQ/LLL first instead of the previous exhaustive coefficient recursion. The exhaustive recursion remains available for no-deadline/offline calls.
- Replaced the slow H^4 explicit relation scans in the async/sync constant-database deep pass with bounded PSLQ/LLL probes over the same intended relation families.
- Added a conservative cubic relation family for simple `b,1,c,c^2,c^3` relations, including the regression sample `x = (log(pi))^3 + 1`.

## Budgets

- level 4: 15 seconds
- level 5: 45 seconds
- level 6: 135 seconds

## Guardrails

- Other RIES modules continue to use their existing LLL/PSLQ paths.
- The five v11.2.1 constant-database transforms remain: `x`, `exp(x)`, `log(x)`, `1/x`, `x^2`.
- Target-independent subset relations remain filtered.
