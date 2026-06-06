# RIES v11.6.1 Changelog

Small maintenance update from v11.6.

## Fixed

- Corrected LaTeX for log-combination linear relations involving `log(log ·)` constants.  The linear-combination renderer now uses the actual constant labels, so a relation such as `log|log|x|| ≈ log(log 2) + log(log 3)` no longer renders as `log|log|x|| ≈ log 2 + log 3`.
- Kept the log-product renderer separate from the linear log-combination renderer, because the former intentionally exponentiates logarithmic constants while the latter must display the constants themselves.
- Increased high-precision Decimal context while computing common base expansions, including base 10, so the decimal expansion is not silently truncated to the default Decimal precision.
- The continued-fraction/base-expansion/digit-statistics panel is now hidden for plain decimal input and shown only for exact integer input or a real computable expression value.

## Parameters

- Stage time-limit fields now display real active defaults rather than `0` placeholders.
- Entering `0` in a stage time-limit field now means no internal time deadline for that module/stage.
- Hard constant database time budgets are exposed separately for depth 4 and depth 5.
- Depth-linked module budgets for RIES/log/Möbius/L-function update with the depth selector until the user edits them manually.
